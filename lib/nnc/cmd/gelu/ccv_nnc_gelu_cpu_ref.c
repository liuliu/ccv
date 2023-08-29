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

static int _ccv_nnc_gelu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* b = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const int count = ccv_nnc_tensor_count(a->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	float* ap = a->data.f32;
	float* bp = b->data.f32;
	if (cmd.info.gelu.tanh)
		for (i = 0; i < count; i++)
		{
			const float x = ap[i];
			bp[i] = 0.5 * x * (1 + tanh(0.797884560802865355 * (x + 0.044715 * x * x * x)));
		}
	else
		for (i = 0; i < count; i++)
		{
			const float x = ap[i];
			bp[i] = x * 0.5 * (1. + erf(x * 0.70710678118654752440));
		}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gelu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_t* g = inputs[0]; // gradient
	assert(CCV_IS_TENSOR_CONTIGUOUS(g));
	const ccv_nnc_tensor_t* a = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* h = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(h));
	const int count = ccv_nnc_tensor_count(g->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == g->info.dim[i]);
		assert(g->info.dim[i] == h->info.dim[i]);
	}
	float* ap = a->data.f32;
	float* gp = g->data.f32;
	float* hp = h->data.f32;
	if (cmd.info.gelu.tanh)
	{
		for (i = 0; i < count; i++)
		{
			const float x = ap[i];
			const float x_sq = x * x;
			const float x_cube = x_sq * x;
			const float inner = 0.797884560802865355 * (x + 0.044715 * x_cube);
			const float tanh_inner = tanh(inner);
			const float left = 0.5 * x;
			const float right = 1 + tanh_inner;
			const float left_derivative = 0.5 * right;
			const float tanh_derivative = 1 - tanh_inner * tanh_inner;
			const float inner_derivative = 0.797884560802865355 * (1 + 3 * 0.044715 * x_sq);
			const float right_derivative = left * tanh_derivative * inner_derivative;
			hp[i] = gp[i] * (left_derivative + right_derivative);
		}
	} else {
		for (i = 0; i < count; i++)
		{
			const float x = ap[i];
			const float cdf = 0.5 * (1. + erf(x * 0.70710678118654752440));
			const float pdf = exp(-0.5 * x * x) * 0.797884560802865355;
			hp[i] = gp[i] * (cdf + x * pdf);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GELU_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gelu_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GELU_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gelu_back;
}
