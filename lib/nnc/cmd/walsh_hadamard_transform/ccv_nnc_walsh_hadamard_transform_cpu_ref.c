#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static inline int _ccv_nnc_is_power_of_two(const int x)
{
	return x > 0 && (x & (x - 1)) == 0;
}

static inline void _ccv_nnc_walsh_hadamard_transform_row(float* const row, const int dim)
{
	int stride;
	for (stride = 1; stride < dim; stride <<= 1)
	{
		int base;
		for (base = 0; base < dim; base += stride * 2)
		{
			int i;
			for (i = 0; i < stride; i++)
			{
				const float a = row[base + i];
				const float b = row[base + i + stride];
				row[base + i] = a + b;
				row[base + i + stride] = a - b;
			}
		}
	}
}

static int _ccv_nnc_walsh_hadamard_transform_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	const int dim = adim[CCV_NNC_MAX_DIM + 1];
	assert(_ccv_nnc_is_power_of_two(dim));
	const float scale = cmd.info.walsh_hadamard_transform.scale;
	int i, j;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		const int count = ccv_nnc_tensor_count(a->info);
		assert(count % dim == 0);
		const int row_count = count / dim;
		const float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		float* const row = (float*)ccmalloc(sizeof(float) * dim);
		for (i = 0; i < row_count; i++)
		{
			memcpy(row, ap + i * dim, sizeof(float) * dim);
			_ccv_nnc_walsh_hadamard_transform_row(row, dim);
			for (j = 0; j < dim; j++)
				bp[i * dim + j] = row[j] * scale;
		}
		ccfree(row);
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int idx[CCV_NNC_MAX_DIM + 2];
	const float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	float* const row = (float*)ccmalloc(sizeof(float) * dim);
	for (idx[0] = 0; idx[0] < adim[0]; idx[0]++)
	{
		const float* const ap0 = ap + idx[0] * astride[0];
		float* const bp0 = bp + idx[0] * bstride[0];
		for (idx[1] = 0; idx[1] < adim[1]; idx[1]++)
		{
			const float* const ap1 = ap0 + idx[1] * astride[1];
			float* const bp1 = bp0 + idx[1] * bstride[1];
			for (idx[2] = 0; idx[2] < adim[2]; idx[2]++)
			{
				const float* const ap2 = ap1 + idx[2] * astride[2];
				float* const bp2 = bp1 + idx[2] * bstride[2];
				for (j = 0; j < dim; j++)
					row[j] = ap2[j * astride[CCV_NNC_MAX_DIM + 1]];
				_ccv_nnc_walsh_hadamard_transform_row(row, dim);
				for (j = 0; j < dim; j++)
					bp2[j * bstride[CCV_NNC_MAX_DIM + 1]] = row[j] * scale;
			}
		}
	}
	ccfree(row);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_walsh_hadamard_transform_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size == 1);
	return _ccv_nnc_walsh_hadamard_transform_forw(cmd, hint, flags, inputs, 1, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_walsh_hadamard_transform_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_walsh_hadamard_transform_back;
}
