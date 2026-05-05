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

static int _ccv_nnc_swish_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const b = inputs[1];
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	assert(a->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(c->info.datatype == CCV_32F);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
		assert(a->info.dim[i] == c->info.dim[i]);
	}
	const int count = ccv_nnc_tensor_count(a->info);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	const float* const ap = a->data.f32;
	const float* const bp = b->data.f32;
	float* const cp = c->data.f32;
	for (i = 0; i < count; i++)
	{
		const float x = bp[i];
		const float sigmoid = 1.f / (1.f + expf(-beta * x));
		cp[i] = scale * ap[i] * x * sigmoid;
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_forw;
}
