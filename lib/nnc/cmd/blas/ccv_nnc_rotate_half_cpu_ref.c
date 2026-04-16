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

static int _ccv_nnc_rotate_half_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	const int half = adim[CCV_NNC_MAX_DIM + 1] / 2;
	assert(half > 0);
	assert(adim[CCV_NNC_MAX_DIM + 1] == half * 2);
	int x;
	if (!CCV_IS_TENSOR_VIEW(a) && !CCV_IS_TENSOR_VIEW(b))
	{
		const int count = ccv_nnc_tensor_count(a->info);
		assert(count % (half * 2) == 0);
		const int row_count = count / (half * 2);
		float* const ap = a->data.f32;
		float* const bp = b->data.f32;
		int i;
		if (ap == bp)
		{
			for (i = 0; i < row_count; i++)
			{
				float* const row = bp + i * half * 2;
				for (x = 0; x < half; x++)
				{
					float t;
					CCV_SWAP(row[x], row[x + half], t);
				}
			}
		} else {
			for (i = 0; i < row_count; i++)
			{
				const float* const ap0 = ap + i * half * 2;
				float* const bp0 = bp + i * half * 2;
				memcpy(bp0, ap0 + half, sizeof(float) * half);
				memcpy(bp0 + half, ap0, sizeof(float) * half);
			}
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		float* const ap0 = ap + i[0] * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			float* const ap1 = ap0 + i[1] * astride[1];
			float* const bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				float* const ap2 = ap1 + i[2] * astride[2];
				float* const bp2 = bp1 + i[2] * bstride[2];
				if (ap2 == bp2 && astride[CCV_NNC_MAX_DIM + 1] == bstride[CCV_NNC_MAX_DIM + 1])
				{
					for (x = 0; x < half; x++)
					{
						float t;
						CCV_SWAP(bp2[x * bstride[CCV_NNC_MAX_DIM + 1]], bp2[(x + half) * bstride[CCV_NNC_MAX_DIM + 1]], t);
					}
				} else {
					for (x = 0; x < half; x++)
					{
						bp2[x * bstride[CCV_NNC_MAX_DIM + 1]] = ap2[(x + half) * astride[CCV_NNC_MAX_DIM + 1]];
						bp2[(x + half) * bstride[CCV_NNC_MAX_DIM + 1]] = ap2[x * astride[CCV_NNC_MAX_DIM + 1]];
					}
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rotate_half_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size == 1);
	return _ccv_nnc_rotate_half_forw(cmd, hint, flags, inputs, 1, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_back;
}
