#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

static int _ccv_nnc_signed_sqrt_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int backward = cmd.cmd == CCV_NNC_SIGNED_SQRT_BACKWARD;
	const int used_inputs = backward ? 2 : 1;
	const float minimum = cmd.info.signed_sqrt.minimum_magnitude;
	if (input_size < used_inputs || output_size != 1 || !outputs[0] || !(minimum > 0) || !isfinite(minimum))
		return CCV_NNC_EXEC_INVALID;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	int i;
	for (i = 0; i < used_inputs; i++)
		if (!inputs[i] || inputs[i]->info.datatype != CCV_32F || b->info.datatype != CCV_32F || memcmp(inputs[i]->info.dim, b->info.dim, sizeof(b->info.dim)) != 0)
			return CCV_NNC_EXEC_INVALID;
	const size_t count = ccv_nnc_tensor_count(b->info);
	if (count == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[backward ? 1 : 0];
	const ccv_nnc_tensor_view_t* const g = backward ? (const ccv_nnc_tensor_view_t*)inputs[0] : 0;
	const int contiguous = CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(b) && (!g || CCV_IS_TENSOR_CONTIGUOUS(g));
	int dim[CCV_NNC_MAX_DIM_ALLOC], astride[CCV_NNC_MAX_DIM_ALLOC], bstride[CCV_NNC_MAX_DIM_ALLOC], gstride[CCV_NNC_MAX_DIM_ALLOC];
	int a_nd = 0;
	if (!contiguous)
	{
		ccv_nnc_tensor_view_get_dim(a, dim);
		a_nd = ccv_nnc_tensor_nd(dim);
		ccv_nnc_tensor_view_get_stride(a, astride);
		ccv_nnc_tensor_view_get_stride(b, bstride);
		if (g)
			ccv_nnc_tensor_view_get_stride(g, gstride);
	}
	size_t j;
	for (j = 0; j < count; j++)
	{
		size_t ai = j, bi = j, gi = j;
		if (!contiguous)
		{
			ai = bi = gi = 0;
			size_t remaining = j;
			for (i = a_nd - 1; i >= 0; i--)
			{
				const size_t index = remaining % dim[i];
				remaining /= dim[i];
				ai += index * astride[i];
				bi += index * bstride[i];
				if (g)
					gi += index * gstride[i];
			}
		}
		const float x = a->data.f32[ai];
		const float ax = fabsf(x);
		if (backward)
			b->data.f32[bi] = ax >= minimum ? g->data.f32[gi] * 0.5f / sqrtf(ax) : 0;
		else
			b->data.f32[bi] = copysignf(sqrtf(fmaxf(ax, minimum)), x);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGNED_SQRT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_signed_sqrt_exec;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGNED_SQRT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_signed_sqrt_exec;
}
