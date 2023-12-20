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

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
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
	assert(w->info.dim[0] == cmd.info.convolution.count);
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
		const int wdim[CCV_NNC_MAX_DIM] = {
			(w->info.dim[1] - 1) * dilation[0] + 1,
			(w->info.dim[2] - 1) * dilation[1] + 1
		};
		assert(w->info.dim[CCV_NNC_MAX_DIM + 1] * groups == adim[CCV_NNC_MAX_DIM]);
		assert(b->info.format == CCV_TENSOR_FORMAT_NHWC);
		const int channel_size = w->info.dim[CCV_NNC_MAX_DIM + 1];
		assert(bdim[CCV_NNC_MAX_DIM] == cmd.info.convolution.count);
		parallel_for(idx, cmd.info.convolution.count * batch_size) {
			int c;
			const int bidx = idx / cmd.info.convolution.count;
			const int k = idx % cmd.info.convolution.count;
			const int gidx = k / group_size;
			float* ap = a->data.f32 + bidx * astride[0];
			float* bp = b->data.f32 + bidx * bstride[0] + k;
			// kernel weight for one dim.
			float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * channel_size;
			float biasval = bias ? bias->data.f32[k] : 0;
			// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int d[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			for (i[0] = 0; i[0] < bdim[0]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, adim, n, m);
				m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
				const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
				d[0] = n0 * dilation[0] - n[0];
				n[0] = n0;
				m[0] = m[0] - n[0];
				float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
				for (i[1] = 0; i[1] < bdim[1]; i[1]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, adim, n, m);
					m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
					const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
					d[1] = n1 * dilation[1] - n[1];
					n[1] = n1;
					m[1] = m[1] - n[1];
					float p = biasval;
					float* wpz = wpu + n[1] * channel_size;
					float* apz = ap + d[0] * astride[CCV_NNC_MAX_DIM - 1] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * astride[CCV_NNC_MAX_DIM] + gidx * channel_size;
					for (j[0] = 0; j[0] < m[0]; j[0]++)
					{
						for (j[1] = 0; j[1] < m[1]; j[1]++)
							for (c = 0; c < channel_size; c++)
								p += wpz[j[1] * channel_size + c] * apz[j[1] * dilation[1] * astride[CCV_NNC_MAX_DIM] + c];
						wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
						apz += astride[CCV_NNC_MAX_DIM - 1] * dilation[0];
					}
					bp[i[1] * bstride[CCV_NNC_MAX_DIM]] = p;
				}
				bp += bstride[CCV_NNC_MAX_DIM - 1];
				ap += astride[CCV_NNC_MAX_DIM - 1] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
			}
		} parallel_endfor
	} else if (a->info.format == CCV_TENSOR_FORMAT_NCHW) {
		// Make sure the weights dimension matches the network dimension
		assert(w->info.dim[2] == cmd.info.size.dim[0]);
		assert(w->info.dim[3] == cmd.info.size.dim[1]);
		const int wdim[CCV_NNC_MAX_DIM] = {
			(w->info.dim[2] - 1) * dilation[0] + 1,
			(w->info.dim[3] - 1) * dilation[1] + 1
		};
		assert(w->info.dim[1] * groups == adim[0]);
		assert(b->info.format == CCV_TENSOR_FORMAT_NCHW);
		const int channel_size = w->info.dim[1];
		const int hw = w->info.dim[2] * w->info.dim[3];
		assert(bdim[0] == cmd.info.convolution.count);
		parallel_for(idx, cmd.info.convolution.count * batch_size) {
			int c;
			const int bidx = idx / cmd.info.convolution.count;
			const int k = idx % cmd.info.convolution.count;
			const int gidx = k / group_size;
			float* ap = a->data.f32 + bidx * astride[0];
			float* bp = b->data.f32 + bidx * bstride[0] + k * bstride[1];
			// kernel weight for one dim.
			float* wp = w->data.f32 + k * hw * channel_size;
			float biasval = bias ? bias->data.f32[k] : 0;
			// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int d[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			for (i[0] = 0; i[0] < bdim[1]; i[0]++)
			{
				SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, adim + 1, n, m);
				m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
				const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
				d[0] = n0 * dilation[0] - n[0];
				n[0] = n0;
				m[0] = m[0] - n[0];
				float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM + 1];
				for (i[1] = 0; i[1] < bdim[2]; i[1]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, adim + 1, n, m);
					m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
					const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
					d[1] = n1 * dilation[1] - n[1];
					n[1] = n1;
					m[1] = m[1] - n[1];
					float p = biasval;
					float* wpz = wpu + n[1];
					float* apz = ap + d[0] * astride[CCV_NNC_MAX_DIM] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * astride[CCV_NNC_MAX_DIM + 1] + gidx * channel_size * astride[1];
					for (j[0] = 0; j[0] < m[0]; j[0]++)
					{
						for (j[1] = 0; j[1] < m[1]; j[1]++)
							for (c = 0; c < channel_size; c++)
								p += wpz[j[1] + c * hw] * apz[j[1] * dilation[1] * astride[CCV_NNC_MAX_DIM + 1] + c * astride[1]];
						wpz += w->info.dim[CCV_NNC_MAX_DIM + 1];
						apz += astride[CCV_NNC_MAX_DIM] * dilation[0];
					}
					bp[i[1]] = p;
				}
				bp += bstride[CCV_NNC_MAX_DIM];
				ap += astride[CCV_NNC_MAX_DIM] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
			}
		} parallel_endfor
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0]; // gradients
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_t* w = output_size > 1 ? outputs[1] : 0;
	assert(CCV_IS_TENSOR_CONTIGUOUS(w));
	ccv_nnc_tensor_t* bias = output_size > 2 ? outputs[2] : 0;
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0]; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		if (w)
			memset(w->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		if (bias)
			memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == CCV_NNC_MAX_DIM + 1 || g_nd == CCV_NNC_MAX_DIM + 2);
	const int* gdim = (g_nd == CCV_NNC_MAX_DIM + 1) ? g->info.dim : g->info.dim + 1;
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(g, gstride);
	const int groups = cmd.info.convolution.groups;
	if (w)
		assert(w->info.dim[CCV_NNC_MAX_DIM + 1] * groups == adim[CCV_NNC_MAX_DIM]);
	assert(cmd.info.convolution.count % groups == 0);
	const int group_size = cmd.info.convolution.count / groups;
	const int channel_size = w ? w->info.dim[CCV_NNC_MAX_DIM + 1] : inputs[2]->info.dim[CCV_NNC_MAX_DIM + 1];
	const int batch_size = (a_nd == CCV_NNC_MAX_DIM + 2) ? a->info.dim[0] : 1;
	const int dilation[CCV_NNC_MAX_DIM] = {
		ccv_max(cmd.info.convolution.dilation[0], 1),
		ccv_max(cmd.info.convolution.dilation[1], 1)
	};
	const int wdim[CCV_NNC_MAX_DIM] = {
		(w->info.dim[1] - 1) * dilation[0] + 1,
		(w->info.dim[2] - 1) * dilation[1] + 1
	};
	if (w)
	{
		parallel_for(k, cmd.info.convolution.count) {
			int c;
			const int gidx = k / group_size;
			// kernel weight for one dim.
			float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * w->info.dim[3];
			float biasval = 0;
			int i[CCV_NNC_MAX_DIM];
			int n[CCV_NNC_MAX_DIM];
			int d[CCV_NNC_MAX_DIM];
			int m[CCV_NNC_MAX_DIM];
			int j[CCV_NNC_MAX_DIM];
			int bidx;
			for (bidx = 0; bidx < batch_size; bidx++)
			{
				const float* ap = a->data.f32 + bidx * astride[0];
				const float* gp = g->data.f32 + bidx * gstride[0] + k;
				for (i[0] = 0; i[0] < gdim[0]; i[0]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, adim, n, m);
					m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
					const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
					d[0] = n0 * dilation[0] - n[0];
					n[0] = n0;
					m[0] = m[0] - n[0];
					float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
					for (i[1] = 0; i[1] < gdim[1]; i[1]++)
					{
						SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, adim, n, m);
						m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
						const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
						d[1] = n1 * dilation[1] - n[1];
						n[1] = n1;
						m[1] = m[1] - n[1];
						const float v = gp[i[1] * gstride[CCV_NNC_MAX_DIM]];
						if (v == 0) // shortcut if v is zero
							continue;
						biasval += v;
						float* wpz = wpu + n[1] * channel_size;
						const float* apz = ap + d[0] * astride[CCV_NNC_MAX_DIM - 1] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * astride[CCV_NNC_MAX_DIM] + gidx * channel_size;
						for (j[0] = 0; j[0] < m[0]; j[0]++)
						{
							for (j[1] = 0; j[1] < m[1]; j[1]++)
								for (c = 0; c < channel_size; c++)
									wpz[j[1] * channel_size + c] += v * apz[j[1] * dilation[1] * astride[CCV_NNC_MAX_DIM] + c];
							wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
							apz += astride[CCV_NNC_MAX_DIM - 1] * dilation[0];
						}
					}
					gp += gstride[CCV_NNC_MAX_DIM - 1];
					ap += astride[CCV_NNC_MAX_DIM - 1] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
				}
			}
			if (bias)
				bias->data.f32[k] = biasval;
		} parallel_endfor
	}
	// If h is available, therefore, we need to propagate the gradients back
	if (h)
	{
		assert(h);
		const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
		assert(h_nd == CCV_NNC_MAX_DIM + 1 || h_nd == CCV_NNC_MAX_DIM + 2);
		const int* hdim = (h_nd == CCV_NNC_MAX_DIM + 1) ? h->info.dim : h->info.dim + 1;
		int hstride[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_stride(h, hstride);
		// reset it to 0.
		ccv_nnc_tensor_zero(h);
		w = inputs[2];
		assert(CCV_IS_TENSOR_CONTIGUOUS(w));
		int bidx;
		for (bidx = 0; bidx < batch_size; bidx++)
		{
			int k;
			for (k = 0; k < cmd.info.convolution.count; k++)
			{
				int c;
				const int gidx = k / group_size;
				float* hp = h->data.f32 + bidx * hstride[0];
				const float* gp = g->data.f32 + bidx * gstride[0] + k;
				// kernel weight for one dim.
				float* wp = w->data.f32 + k * w->info.dim[1] * w->info.dim[2] * channel_size;
				// This block will be cause in each for-loop, therefore, you can use it to generate some temporary variables.
				int i[CCV_NNC_MAX_DIM];
				int n[CCV_NNC_MAX_DIM];
				int d[CCV_NNC_MAX_DIM];
				int m[CCV_NNC_MAX_DIM];
				int j[CCV_NNC_MAX_DIM];
				for (i[0] = 0; i[0] < gdim[0]; i[0]++)
				{
					SET_BORDER_OFFSET_SIZE_FOR(0, i, hint, wdim, hdim, n, m);
					m[0] = (m[0] + n[0] - 1) / dilation[0] + 1;
					const int n0 = (n[0] + dilation[0] - 1) / dilation[0];
					d[0] = n0 * dilation[0] - n[0];
					n[0] = n0;
					m[0] = m[0] - n[0];
					const float* wpu = wp + n[0] * w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
					for (i[1] = 0; i[1] < gdim[1]; i[1]++)
					{
						SET_BORDER_OFFSET_SIZE_FOR(1, i, hint, wdim, hdim, n, m);
						m[1] = (m[1] + n[1] - 1) / dilation[1] + 1;
						const int n1 = (n[1] + dilation[1] - 1) / dilation[1];
						d[1] = n1 * dilation[1] - n[1];
						n[1] = n1;
						m[1] = m[1] - n[1];
						const float v = gp[i[1] * gstride[CCV_NNC_MAX_DIM]];
						if (v == 0) // shortcut if v is zero
							continue;
						const float* wpz = wpu + n[1] * channel_size;
						float* hpz = hp + d[0] * hstride[CCV_NNC_MAX_DIM - 1] + (ccv_max(i[1] * hint.stride.dim[1] - hint.border.begin[1], 0) + d[1]) * hstride[CCV_NNC_MAX_DIM] + gidx * channel_size;
						for (j[0] = 0; j[0] < m[0]; j[0]++)
						{
							for (j[1] = 0; j[1] < m[1]; j[1]++)
								for (c = 0; c < channel_size; c++)
									hpz[j[1] * dilation[1] * hstride[CCV_NNC_MAX_DIM] + c] += v * wpz[j[1] * channel_size + c];
							wpz += w->info.dim[CCV_NNC_MAX_DIM] * channel_size;
							hpz += hstride[CCV_NNC_MAX_DIM - 1] * dilation[0];
						}
					}
					gp += gstride[CCV_NNC_MAX_DIM - 1];
					hp += hstride[CCV_NNC_MAX_DIM - 1] * (ccv_max((i[0] + 1) * hint.stride.dim[0] - hint.border.begin[0], 0) - ccv_max(i[0] * hint.stride.dim[0] - hint.border.begin[0], 0));
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_back;
}

