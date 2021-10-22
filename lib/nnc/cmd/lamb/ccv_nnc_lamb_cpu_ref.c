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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_lamb_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 4);
	assert(output_size == 3);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const u = (ccv_nnc_tensor_view_t*)outputs[2];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(g, adim));
	assert(ccv_nnc_tensor_view_check_dim(m, adim));
	assert(ccv_nnc_tensor_view_check_dim(v, adim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(ccv_nnc_tensor_view_check_dim(n, adim));
	assert(ccv_nnc_tensor_view_check_dim(u, adim));
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int ginc[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int minc[CCV_NNC_MAX_DIM_ALLOC];
	int vinc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int ninc[CCV_NNC_MAX_DIM_ALLOC];
	int uinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_inc(g, ginc);
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(m, minc);
	ccv_nnc_tensor_view_get_inc(v, vinc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(n, ninc);
	ccv_nnc_tensor_view_get_inc(u, uinc);
	const int step = cmd.info.lamb.step;
	const float rate = cmd.info.lamb.rate;
	const float beta1 = cmd.info.lamb.beta1;
	const float beta2 = cmd.info.lamb.beta2;
	const float decay = cmd.info.lamb.decay;
	const float epsilon = cmd.info.lamb.epsilon;
	assert(step >= 1);
	const float inv_bias_correction1 = 1. / (1 - powf(beta1, step));
	const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
	int i[CCV_NNC_MAX_DIM + 1];
	int x;
	float* gp = g->data.f32;
	float* ap = a->data.f32;
	float* mp = m->data.f32;
	float* vp = v->data.f32;
	float* bp = b->data.f32;
	float* np = n->data.f32;
	float* up = u->data.f32;
	float* const update = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * adim[0] * adim[1] * adim[2] * adim[3], CCV_TENSOR_CPU_MEMORY);
	float* updatep = update;
	double update_norm = 0;
	double w_norm = 0;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				for (x = 0; x < adim[3]; x++)
				{
					const float grad = gp[x];
					const float w = ap[x];
					const float mom = np[x] = beta1 * mp[x] + (1 - beta1) * grad;
					const float vel = up[x] = beta2 * vp[x] + (1 - beta2) * grad * grad;
					const float update = updatep[x] = (mom * inv_bias_correction1) / (sqrtf(vel * inv_bias_correction2) + epsilon) + w * decay;
					w_norm += w * w;
					update_norm += update * update;
				}
				gp += ginc[3];
				ap += ainc[3];
				mp += minc[3];
				vp += vinc[3];
				np += ninc[3];
				up += uinc[3];
				updatep += adim[3];
			}
			gp += (ginc[2] - adim[2]) * ginc[3];
			ap += (ainc[2] - adim[2]) * ainc[3];
			mp += (minc[2] - adim[2]) * minc[3];
			vp += (vinc[2] - adim[2]) * vinc[3];
			np += (ninc[2] - adim[2]) * ninc[3];
			up += (uinc[2] - adim[2]) * uinc[3];
		}
		gp += (ginc[1] - adim[1]) * ginc[2] * ginc[3];
		ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
		mp += (minc[1] - adim[1]) * minc[2] * minc[3];
		vp += (vinc[1] - adim[1]) * vinc[2] * vinc[3];
		np += (ninc[1] - adim[1]) * ninc[2] * ninc[3];
		up += (uinc[1] - adim[1]) * uinc[2] * uinc[3];
	}
	w_norm = sqrt(w_norm);
	update_norm = sqrt(update_norm);
	const float trust_ratio = w_norm > 0 && update_norm > 0 ? w_norm / update_norm : 1.;
	const float rate_trust_ratio = rate * trust_ratio;
	ap = a->data.f32;
	bp = b->data.f32;
	updatep = update;
	for (i[0] = 0; i[0] < adim[0]; i[0]++)
	{
		for (i[1] = 0; i[1] < adim[1]; i[1]++)
		{
			for (i[2] = 0; i[2] < adim[2]; i[2]++)
			{
				for (x = 0; x < adim[3]; x++)
					bp[x] = ap[x] - rate_trust_ratio * updatep[x];
				ap += ainc[3];
				bp += binc[3];
				updatep += adim[3];
			}
			ap += (ainc[2] - adim[2]) * ainc[3];
			bp += (binc[2] - adim[2]) * binc[3];
		}
		ap += (ainc[1] - adim[1]) * ainc[2] * ainc[3];
		bp += (binc[1] - adim[1]) * binc[2] * binc[3];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_lamb_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAMB_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lamb_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAMB_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lamb_back;
}
