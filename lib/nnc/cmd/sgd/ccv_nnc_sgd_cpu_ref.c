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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_sgd_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 2);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(a->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	assert(b->info.dim[CCV_NNC_MAX_DIM + 2] == 0);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(g, adim));
	assert(ccv_nnc_tensor_view_check_dim(m, adim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(ccv_nnc_tensor_view_check_dim(n, adim));
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int ginc[CCV_NNC_MAX_DIM + 2];
	int ainc[CCV_NNC_MAX_DIM + 2];
	int minc[CCV_NNC_MAX_DIM + 2];
	int binc[CCV_NNC_MAX_DIM + 2];
	int ninc[CCV_NNC_MAX_DIM + 2];
	ccv_nnc_tensor_view_get_inc(g, ginc);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(m, minc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(n, ninc);
	const float rate = cmd.info.minimize.rate;
	const float decay = cmd.info.minimize.decay;
	const float momentum = cmd.info.minimize.momentum;
	const float dampening = cmd.info.minimize.dampening;
	const float inv_dampening = 1 - dampening;
	int i[CCV_NNC_MAX_DIM + 1];
	int x;
	float* gp = g->data.f32;
	float* ap = a->data.f32;
	float* mp = m->data.f32;
	float* bp = b->data.f32;
	float* np = n->data.f32;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				for (x = 0; x < adim[3]; x++)
				{
					np[x] = momentum * mp[x] + inv_dampening * (gp[x] + decay * ap[x]);
					bp[x] = ap[x] - rate * np[x];
				}
				gp += ginc[3];
				ap += ainc[3];
				mp += minc[3];
				bp += binc[3];
				np += ninc[3];
			}
			gp += (ginc[2] - adim[2]) * ginc[3];
			ap += (ainc[2] - adim[2]) * ainc[3];
			mp += (minc[2] - adim[2]) * minc[3];
			bp += (binc[2] - adim[2]) * binc[3];
			np += (ninc[2] - adim[2]) * ninc[3];
		}
		gp += (ginc[1] - adim[1]) * ginc[2] * ginc[3];
		ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
		mp += (minc[1] - adim[1]) * minc[2] * minc[3];
		bp += (binc[1] - adim[1]) * binc[2] * binc[3];
		np += (ninc[1] - adim[1]) * ninc[2] * ninc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sgd_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SGD_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sgd_back;
}
