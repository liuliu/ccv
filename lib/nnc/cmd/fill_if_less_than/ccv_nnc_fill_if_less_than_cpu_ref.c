#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static void _ccv_nnc_fill_if_less_than_cpu_ref_f32(const float fill, const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const selector, const ccv_nnc_tensor_view_t* const threshold, ccv_nnc_tensor_view_t* const b)
{
	assert(CCV_NNC_MAX_DIM == 2);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(selector->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(threshold->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int selector_dim[CCV_NNC_MAX_DIM_ALLOC];
	int threshold_dim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(selector, selector_dim);
	ccv_nnc_tensor_view_get_dim(threshold, threshold_dim);
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(ccv_nnc_tensor_view_check_broadcast_dim(selector, adim));
	assert(ccv_nnc_tensor_view_check_broadcast_dim(threshold, adim));
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int selector_stride[CCV_NNC_MAX_DIM_ALLOC];
	int threshold_stride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(selector, selector_stride);
	ccv_nnc_tensor_view_get_stride(threshold, threshold_stride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	const float* const ap = a->data.f32;
	const float* const selector_p = selector->data.f32;
	const float* const threshold_p = threshold->data.f32;
	float* const bp = b->data.f32;
	int i[CCV_NNC_MAX_DIM + 2];
	int x;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		const float* const ap0 = ap + i[0] * astride[0];
		const float* const selector_p0 = selector_dim[0] == 1 ? selector_p : selector_p + i[0] * selector_stride[0];
		const float* const threshold_p0 = threshold_dim[0] == 1 ? threshold_p : threshold_p + i[0] * threshold_stride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			const float* const ap1 = ap0 + i[1] * astride[1];
			const float* const selector_p1 = selector_dim[1] == 1 ? selector_p0 : selector_p0 + i[1] * selector_stride[1];
			const float* const threshold_p1 = threshold_dim[1] == 1 ? threshold_p0 : threshold_p0 + i[1] * threshold_stride[1];
			float* const bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				const float* const ap2 = ap1 + i[2] * astride[2];
				const float* const selector_p2 = selector_dim[2] == 1 ? selector_p1 : selector_p1 + i[2] * selector_stride[2];
				const float* const threshold_p2 = threshold_dim[2] == 1 ? threshold_p1 : threshold_p1 + i[2] * threshold_stride[2];
				float* const bp2 = bp1 + i[2] * bstride[2];
				for (x = 0; x < adim[3]; x++)
				{
					const float selector_value = selector_p2[(selector_dim[3] == 1 ? 0 : x) * selector_stride[3]];
					const float threshold_value = threshold_p2[(threshold_dim[3] == 1 ? 0 : x) * threshold_stride[3]];
					bp2[x * bstride[3]] = selector_value < threshold_value ? fill : ap2[x * astride[3]];
				}
			}
		}
	}
}

static int _ccv_nnc_fill_if_less_than_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	assert(inputs[0] && inputs[1] && inputs[2] && outputs[0]);
	assert(inputs[0]->info.datatype == CCV_32F && inputs[1]->info.datatype == CCV_32F && inputs[2]->info.datatype == CCV_32F && outputs[0]->info.datatype == CCV_32F);
	_ccv_nnc_fill_if_less_than_cpu_ref_f32(cmd.info.fill_if_less_than.value, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[1], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_fill_if_less_than_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	assert(output_size >= 1);
	assert(inputs[0] && inputs[2] && inputs[3] && outputs[0]);
	assert(inputs[0]->info.datatype == CCV_32F && inputs[2]->info.datatype == CCV_32F && inputs[3]->info.datatype == CCV_32F && outputs[0]->info.datatype == CCV_32F);
	_ccv_nnc_fill_if_less_than_cpu_ref_f32(0, (ccv_nnc_tensor_view_t*)inputs[0], (ccv_nnc_tensor_view_t*)inputs[2], (ccv_nnc_tensor_view_t*)inputs[3], (ccv_nnc_tensor_view_t*)outputs[0]);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FILL_IF_LESS_THAN_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_fill_if_less_than_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FILL_IF_LESS_THAN_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_fill_if_less_than_back;
}
