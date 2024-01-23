#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_conv_transpose_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_t* w = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(w));
	const ccv_nnc_tensor_t* bias = input_size > 2 ? inputs[2] : 0;
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	const int groups = cmd.info.convolution.groups;
	assert(cmd.info.convolution.count % groups == 0);
	const int group_size = cmd.info.convolution.count / groups;
	// Make sure the weights output dimension matches the network convolution kernels
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(b, bstride);
	assert(!bias || bias->info.dim[0] == cmd.info.convolution.count);
	const int batch_size = (a_nd == CCV_NNC_MAX_DIM + 2) ? a->info.dim[0] : 1;
	const int dilation[CCV_NNC_MAX_DIM] = {
		ccv_max(cmd.info.convolution.dilation[0], 1),
		ccv_max(cmd.info.convolution.dilation[1], 1)
	};
	if (a->info.format == CCV_TENSOR_FORMAT_NHWC)
	{
		// Make sure the weights dimension matches the network dimension
		assert(w->info.dim[1] == cmd.info.size.dim[0]);
		assert(w->info.dim[2] == cmd.info.size.dim[1]);
		assert(w->info.dim[CCV_NNC_MAX_DIM + 1] * groups == cmd.info.convolution.count);
		const int wdim[CCV_NNC_MAX_DIM] = {
			(w->info.dim[1] - 1) * dilation[0] + 1,
			(w->info.dim[2] - 1) * dilation[1] + 1
		};
		assert(w->info.dim[0] == adim[CCV_NNC_MAX_DIM]);
		assert(b->info.format == CCV_TENSOR_FORMAT_NHWC);
		const int channel_size = w->info.dim[CCV_NNC_MAX_DIM + 1];
		const int input_channel_size = w->info.dim[0];
		const int hwc = w->info.dim[1] * w->info.dim[2] * channel_size;
		assert(bdim[CCV_NNC_MAX_DIM] == cmd.info.convolution.count);
		parallel_for(idx, cmd.info.convolution.count * batch_size) {
			int c;
			const int bidx = idx / cmd.info.convolution.count;
			const int k = idx % cmd.info.convolution.count;
			float* ap = a->data.f32 + bidx * astride[0];
			float* bp = b->data.f32 + bidx * bstride[0] + k;
			// kernel weight for one dim.
			float* wp = w->data.f32 + (k % group_size);
			float biasval = bias ? bias->data.f32[k] : 0;
			// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int d[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			for (i[0] = 0; i[0] < bdim[0]; i[0]++)
			{
				for (i[1] = 0; i[1] < bdim[1]; i[1]++)
					bp[i[1] * bstride[CCV_NNC_MAX_DIM]] = biasval;
				bp += bstride[CCV_NNC_MAX_DIM - 1];
			}
			bp = b->data.f32 + bidx * bstride[0] + k;
			for (i[0] = 0; i[0] < adim[0]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, bdim, n, m);
				m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
				const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
				d[0] = n0 * dilation[0] - n[0];
				n[0] = n0;
				m[0] = m[0] - n[0];
				float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
				for (i[1] = 0; i[1] < adim[1]; i[1]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, bdim, n, m);
					m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
					const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
					d[1] = n1 * dilation[1] - n[1];
					n[1] = n1;
					m[1] = m[1] - n[1];
					float* wpz = wpu + n[1] * channel_size;
					const float* const apz = ap + i[1] * astride[CCV_NNC_MAX_DIM];
					float* bpz = bp + d[0] * bstride[CCV_NNC_MAX_DIM - 1] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * bstride[CCV_NNC_MAX_DIM];
					for (j[0] = 0; j[0] < m[0]; j[0]++)
					{
						for (j[1] = 0; j[1] < m[1]; j[1]++)
						{
							float p = bpz[j[1] * dilation[1] * bstride[CCV_NNC_MAX_DIM]];
							for (c = 0; c < input_channel_size; c++)
								 p += wpz[j[1] * channel_size + c * hwc] * apz[c];
							bpz[j[1] * dilation[1] * bstride[CCV_NNC_MAX_DIM]] = p;
						}
						wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
						bpz += bstride[CCV_NNC_MAX_DIM - 1] * dilation[0];
					}
				}
				ap += astride[CCV_NNC_MAX_DIM - 1];
				bp += bstride[CCV_NNC_MAX_DIM - 1] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
			}
		} parallel_endfor
	} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW) {
		// Make sure the weights dimension matches the network dimension
		assert(w->info.dim[1] * groups == cmd.info.convolution.count);
		assert(w->info.dim[2] == cmd.info.size.dim[0]);
		assert(w->info.dim[3] == cmd.info.size.dim[1]);
		const int wdim[CCV_NNC_MAX_DIM] = {
			(w->info.dim[2] - 1) * dilation[0] + 1,
			(w->info.dim[3] - 1) * dilation[1] + 1
		};
		assert(w->info.dim[0] == adim[0]);
		assert(b->info.format == CCV_TENSOR_FORMAT_NCHW);
		const int channel_size = w->info.dim[1];
		const int input_channel_size = w->info.dim[0];
		const int hw = w->info.dim[2] * w->info.dim[3];
		const int chw = channel_size * hw;
		assert(bdim[0] == cmd.info.convolution.count);
		parallel_for(idx, cmd.info.convolution.count * batch_size) {
			int c;
			const int bidx = idx / cmd.info.convolution.count;
			const int k = idx % cmd.info.convolution.count;
			float* ap = a->data.f32 + bidx * astride[0];
			float* bp = b->data.f32 + bidx * bstride[0] + k * bstride[1];
			// kernel weight for one dim.
			float* wp = w->data.f32 + (k % group_size) * hw;
			float biasval = bias ? bias->data.f32[k] : 0;
			// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int d[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			for (i[0] = 0; i[0] < bdim[1]; i[0]++)
			{
				for (i[1] = 0; i[1] < bdim[2]; i[1]++)
					bp[i[1]] = biasval;
				bp += bstride[CCV_NNC_MAX_DIM];
			}
			bp = b->data.f32 + bidx * bstride[0] + k * bstride[1];
			for (i[0] = 0; i[0] < adim[1]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, bdim + 1, n, m);
				m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
				const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
				d[0] = n0 * dilation[0] - n[0];
				n[0] = n0;
				m[0] = m[0] - n[0];
				float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM + 1];
				for (i[1] = 0; i[1] < adim[2]; i[1]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, bdim + 1, n, m);
					m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
					const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
					d[1] = n1 * dilation[1] - n[1];
					n[1] = n1;
					m[1] = m[1] - n[1];
					float* wpz = wpu + n[1];
					const float* apz = ap + i[1];
					float* bpz = bp + d[0] * bstride[CCV_NNC_MAX_DIM] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * bstride[CCV_NNC_MAX_DIM + 1];
					for (j[0] = 0; j[0] < m[0]; j[0]++)
					{
						for (j[1] = 0; j[1] < m[1]; j[1]++)
						{
							float p = bpz[j[1] * dilation[1] * bstride[CCV_NNC_MAX_DIM + 1]];
							for (c = 0; c < input_channel_size; c++)
								 p += wpz[j[1] + c * chw] * apz[c * astride[1]];
							bpz[j[1] * dilation[1] * bstride[CCV_NNC_MAX_DIM + 1]] = p;
						}
						wpz += w->info.dim[CCV_NNC_MAX_DIM + 1];
						apz += astride[CCV_NNC_MAX_DIM] * dilation[0];
					}
				}
				ap += astride[CCV_NNC_MAX_DIM];
				bp += bstride[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
			}
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_transpose_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_transpose_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_transpose_back;
}

