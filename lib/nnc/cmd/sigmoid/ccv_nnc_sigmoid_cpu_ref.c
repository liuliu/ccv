#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_sigmoid_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const int count = ccv_nnc_tensor_count(a->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	for (i = 0; i < count; i++)
		bp[i] = 1. / (1. + exp(-ap[i]));
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sigmoid_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* g = inputs[0];
	const ccv_nnc_tensor_t* b = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(b));
	ccv_nnc_tensor_t* h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	const int count = ccv_nnc_tensor_count(b->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
		{ assert(h->info.dim[i] == b->info.dim[i]); }
	if (g)
	{
		assert(!CCV_IS_TENSOR_VIEW(g));
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
			{ assert(g->info.dim[i] == h->info.dim[i]); }
		float* const gp = g->data.f32;
		float* const bp = b->data.f32;
		float* const hp = h->data.f32;
		for (i = 0; i < count; i++)
			hp[i] = gp[i] * bp[i] * (1 - bp[i]);
	} else {
		float* const bp = b->data.f32;
		float* const hp = h->data.f32;
		for (i = 0; i < count; i++)
			hp[i] = bp[i] * (1 - bp[i]);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_back;
}
