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

static int _ccv_nnc_unique_consecutive_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(a->info.datatype == b->info.datatype);
	const int count = ccv_nnc_tensor_count(a->info);
	assert(a_nd == 1); // Can only handle 1d tensor for this.
	const int bincount = b->info.dim[0];
	assert(bincount > 0);
	assert(bincount == indices->info.dim[0]);
	int i;
	if (a->info.datatype == CCV_32F)
	{
		b->data.f32[0] = a->data.f32[0];
		float last_val = b->data.f32[0];
		int j = 1;
		int len = 1;
		for (i = 1; i < count; i++)
			if (a->data.f32[i] != last_val)
			{
				if (j >= bincount)
					break;
				indices->data.i32[j - 1] = len;
				b->data.f32[j] = a->data.f32[i];
				last_val = b->data.f32[j];
				++j;
				len = 1;
			} else
				++len;
		if (j <= count)
			indices->data.i32[j - 1] = len;
		for (i = j; i < bincount; i++)
			b->data.f32[i] = -1, indices->data.i32[i] = 0;
	} else {
		assert(a->info.datatype == CCV_32S);
		b->data.i32[0] = a->data.i32[0];
		int last_val = b->data.i32[0];
		int j = 1;
		int len = 1;
		for (i = 1; i < count; i++)
			if (a->data.i32[i] != last_val)
			{
				if (j >= bincount)
					break;
				indices->data.i32[j - 1] = len;
				b->data.i32[j] = a->data.i32[i];
				last_val = b->data.i32[j];
				++j;
				len = 1;
			} else
				++len;
		if (j <= count)
			indices->data.i32[j - 1] = len;
		for (i = j; i < bincount; i++)
			b->data.i32[i] = -1, indices->data.i32[i] = 0;
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_unique_consecutive_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_unique_consecutive_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_unique_consecutive_back;
}
