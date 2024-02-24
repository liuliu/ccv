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

static int _ccv_nnc_pad_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int x;
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int nd = ccv_nnc_tensor_nd(a->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	assert(offset >= 0);
	for (x = 0; x < nd; x++) // We don't support negative pad.
		{ assert(cmd.info.size.dim[x] >= 0 && cmd.info.pad.end[x] >= 0); }
	int begin[CCV_NNC_MAX_DIM_ALLOC];
	for (x = 0; x < nd; x++)
		begin[x + offset] = cmd.info.size.dim[x];
	for (x = 0; x < offset; x++)
		begin[x] = 0;
	// Non-optimal case, need to do skip if needed.
	if (cmd.info.pad.type == CCV_NNC_PAD_ZERO)
	{
		for (i[0] = 0; i[0] < bdim[0]; i[0]++)
		{
			float* const ap0 = (i[0] >= begin[0] && i[0] < adim[0] + begin[0]) ? ap + (i[0] - begin[0]) * astride[0] : 0;
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < bdim[1]; i[1]++)
			{
				float* const ap1 = (ap0 && i[1] >= begin[1] && i[1] < adim[1] + begin[1]) ? ap0 + (i[1] - begin[1]) * astride[1] : 0;
				float* bp1 = bp0 + i[1] * bstride[1];
				for (i[2] = 0; i[2] < bdim[2]; i[2]++)
				{
					float* const ap2 = (ap1 && i[2] >= begin[2] && i[2] < adim[2] + begin[2]) ? ap1 + (i[2] - begin[2]) * astride[2] : 0;
					for (x = 0; x < bdim[3]; x++)
						bp1[x] = (ap2 && x >= begin[3] && x < adim[3] + begin[3]) ? ap2[x - begin[3]] : 0;
					bp1 += bstride[2];
				}
			}
		}
	} else {
		assert(cmd.info.pad.type == CCV_NNC_PAD_REPLICATE);
		for (i[0] = 0; i[0] < bdim[0]; i[0]++)
		{
			float* const ap0 = ap + ccv_min(adim[0] - 1, ccv_max(0, i[0] - begin[0])) * astride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			for (i[1] = 0; i[1] < bdim[1]; i[1]++)
			{
				float* const ap1 = ap0 + ccv_min(adim[1] - 1, ccv_max(0, i[1] - begin[1])) * astride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				for (i[2] = 0; i[2] < bdim[2]; i[2]++)
				{
					float* const ap2 = ap1 + ccv_min(adim[2] - 1, ccv_max(0, i[2] - begin[2])) * astride[2];
					for (x = 0; x < bdim[3]; x++)
						bp1[x] = ap2[ccv_min(adim[3] - 1, ccv_max(0, x - begin[3]))];
					bp1 += bstride[2];
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_pad_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	int i[CCV_NNC_MAX_DIM + 2];
	int x;
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int nd = ccv_nnc_tensor_nd(a->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	assert(offset >= 0);
	for (x = 0; x < nd; x++) // We don't support negative pad.
		{ assert(cmd.info.size.dim[x] >= 0 && cmd.info.pad.end[x] >= 0); }
	int begin[CCV_NNC_MAX_DIM_ALLOC];
	for (x = 0; x < nd; x++)
		begin[x + offset] = cmd.info.size.dim[x];
	for (x = 0; x < offset; x++)
		begin[x] = 0;
	// Non-optimal case, need to do skip if needed.
	for (i[0] = 0; i[0] < bdim[0]; i[0]++)
	{
		float* const ap0 = ap + (i[0] + begin[0]) * astride[0];
		float* const bp0 = bp + i[0] * bstride[0];
		for (i[1] = 0; i[1] < bdim[1]; i[1]++)
		{
			float* const ap1 = ap0 + (i[1] + begin[1]) * astride[1];
			float* bp1 = bp0 + i[1] * bstride[1];
			for (i[2] = 0; i[2] < bdim[2]; i[2]++)
			{
				float* const ap2 = ap1 + (i[2] + begin[2]) * astride[2];
				for (x = 0; x < bdim[3]; x++)
					bp1[x] = ap2[x + begin[3]];
				bp1 += bstride[2];
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_back;
}
