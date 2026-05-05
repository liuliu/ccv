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
#include <string.h>

static int _ccv_nnc_rmsnorm_gated_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int elementwise_affine = cmd.info.rmsnorm_gated.elementwise_affine;
	assert(input_size == (elementwise_affine ? 3 : 2));
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const gate = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const scale = elementwise_affine ? (ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.datatype == CCV_32F);
	assert(gate->info.datatype == CCV_32F);
	assert(!scale || scale->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(gate->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	int sdim[CCV_NNC_MAX_DIM_ALLOC];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int axis_offset = ccv_max(CCV_NNC_MAX_DIM + 2 - a_nd, 0);
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(gate, rdim);
	assert(ccv_nnc_tensor_view_check_dim(gate, adim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	if (scale)
		ccv_nnc_tensor_view_get_dim(scale, sdim);
	memcpy(rdim, adim, sizeof(rdim));
	int i;
	for (i = 0; i < cmd.info.rmsnorm_gated.count; i++)
	{
		const int axis = cmd.info.rmsnorm_gated.axis[i];
		assert(axis >= 0 && axis < a_nd);
		rdim[axis + axis_offset] = 1;
	}
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int gate_stride[CCV_NNC_MAX_DIM_ALLOC];
	int scale_stride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(gate, gate_stride);
	if (scale)
		ccv_nnc_tensor_view_get_stride(scale, scale_stride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int rstride[CCV_NNC_MAX_DIM_ALLOC];
	rstride[3] = 1;
	rstride[2] = rdim[3];
	rstride[1] = rdim[2] * rstride[2];
	rstride[0] = rdim[1] * rstride[1];
	int n = 1;
	int rcount = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM + 2; i++)
	{
		n *= adim[i];
		rcount *= rdim[i];
	}
	for (i = 0; i < CCV_NNC_MAX_DIM + 2; i++)
		n /= rdim[i];
	const float inv_n = 1. / n;
	float* const varp = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * rcount, CCV_TENSOR_CPU_MEMORY);
	memset(varp, 0, sizeof(float) * rcount);
	const float* const ap = a->data.f32;
	const float* const gatep = gate->data.f32;
	const float* const scalep = scale ? scale->data.f32 : 0;
	float* const bp = b->data.f32;
	int x;
	int idx[CCV_NNC_MAX_DIM + 2];
	for (idx[0] = 0; idx[0] < adim[0]; idx[0]++)
	{
		const float* const ap0 = ap + idx[0] * astride[0];
		float* const varp0 = rdim[0] == 1 ? varp : varp + idx[0] * rstride[0];
		for (idx[1] = 0; idx[1] < adim[1]; idx[1]++)
		{
			const float* const ap1 = ap0 + idx[1] * astride[1];
			float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + idx[1] * rstride[1];
			for (idx[2] = 0; idx[2] < adim[2]; idx[2]++)
			{
				const float* const ap2 = ap1 + idx[2] * astride[2];
				float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + idx[2] * rstride[2];
				if (rdim[3] == 1)
					for (x = 0; x < adim[3]; x++)
					{
						const float v = ap2[x * astride[3]];
						varp2[0] += v * v;
					}
				else
					for (x = 0; x < adim[3]; x++)
					{
						const float v = ap2[x * astride[3]];
						varp2[x * rstride[3]] += v * v;
					}
			}
		}
	}
	const float epsilon = cmd.info.rmsnorm_gated.epsilon;
	for (i = 0; i < rcount; i++)
		varp[i] = 1. / sqrtf(varp[i] * inv_n + epsilon);
	for (idx[0] = 0; idx[0] < adim[0]; idx[0]++)
	{
		const float* const ap0 = ap + idx[0] * astride[0];
		const float* const gatep0 = gatep + idx[0] * gate_stride[0];
		float* const bp0 = bp + idx[0] * bstride[0];
		const float* const varp0 = rdim[0] == 1 ? varp : varp + idx[0] * rstride[0];
		const float* const scalep0 = scale ? (sdim[0] == 1 ? scalep : scalep + idx[0] * scale_stride[0]) : 0;
		for (idx[1] = 0; idx[1] < adim[1]; idx[1]++)
		{
			const float* const ap1 = ap0 + idx[1] * astride[1];
			const float* const gatep1 = gatep0 + idx[1] * gate_stride[1];
			float* const bp1 = bp0 + idx[1] * bstride[1];
			const float* const varp1 = rdim[1] == 1 ? varp0 : varp0 + idx[1] * rstride[1];
			const float* const scalep1 = scale ? (sdim[1] == 1 ? scalep0 : scalep0 + idx[1] * scale_stride[1]) : 0;
			for (idx[2] = 0; idx[2] < adim[2]; idx[2]++)
			{
				const float* const ap2 = ap1 + idx[2] * astride[2];
				const float* const gatep2 = gatep1 + idx[2] * gate_stride[2];
				float* const bp2 = bp1 + idx[2] * bstride[2];
				const float* const varp2 = rdim[2] == 1 ? varp1 : varp1 + idx[2] * rstride[2];
				const float* const scalep2 = scale ? (sdim[2] == 1 ? scalep1 : scalep1 + idx[2] * scale_stride[2]) : 0;
				for (x = 0; x < adim[3]; x++)
				{
					const float z = gatep2[x * gate_stride[3]];
					const float ez = expf(-fabsf(z));
					const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
					const float inv_std = rdim[3] == 1 ? varp2[0] : varp2[x * rstride[3]];
					const float scale_v = scale ? scalep2[sdim[3] == 1 ? 0 : x * scale_stride[3]] : 1;
					bp2[x * bstride[3]] = ap2[x * astride[3]] * inv_std * scale_v * z * sigmoid;
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_gated_forw;
}
