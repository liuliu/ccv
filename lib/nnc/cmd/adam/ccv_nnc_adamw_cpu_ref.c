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

static int _ccv_nnc_adamw_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	assert(output_size >= 3);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const vm = input_size >= 5 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const u = (ccv_nnc_tensor_view_t*)outputs[2];
	ccv_nnc_tensor_view_t* const um = output_size >= 4 ? (ccv_nnc_tensor_view_t*)outputs[3] : 0;
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	assert(ccv_nnc_tensor_view_check_dim(g, adim));
	assert(ccv_nnc_tensor_view_check_dim(m, adim));
	assert(ccv_nnc_tensor_view_check_dim(v, adim));
	if (vm)
		{ assert(ccv_nnc_tensor_view_check_dim(vm, adim)); }
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	assert(ccv_nnc_tensor_view_check_dim(n, adim));
	assert(ccv_nnc_tensor_view_check_dim(u, adim));
	if (um)
		{ assert(ccv_nnc_tensor_view_check_dim(um, adim)); }
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int mstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int vmstride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int nstride[CCV_NNC_MAX_DIM_ALLOC];
	int ustride[CCV_NNC_MAX_DIM_ALLOC];
	int umstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(g, gstride);
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(m, mstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	if (vm)
		ccv_nnc_tensor_view_get_stride(vm, vmstride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(n, nstride);
	ccv_nnc_tensor_view_get_stride(u, ustride);
	if (um)
		ccv_nnc_tensor_view_get_stride(um, umstride);
	const int step = cmd.info.adam.step;
	const float rate = cmd.info.adam.rate;
	const float scale = cmd.info.adam.scale;
	const float beta1 = cmd.info.adam.beta1;
	const float beta2 = cmd.info.adam.beta2;
	const float decay = cmd.info.adam.decay;
	const float epsilon = cmd.info.adam.epsilon;
	assert(step >= 1);
	const float rate_inv_bias_correction1 = rate / (1 - powf(beta1, step));
	const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
	const float rate_decay = rate * decay;
	int i[CCV_NNC_MAX_DIM + 1];
	int x;
	float* const gp = g->data.f32;
	float* const ap = a->data.f32;
	float* const mp = m->data.f32;
	float* const vp = v->data.f32;
	float* const bp = b->data.f32;
	float* const np = n->data.f32;
	float* const up = u->data.f32;
	if (cmd.info.adam.amsgrad && vm && um)
	{
		float* const vmp = vm->data.f32;
		float* const ump = um->data.f32;
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const gp0 = gp + i[0] * gstride[0];
			float* const ap0 = ap + i[0] * astride[0];
			float* const mp0 = mp + i[0] * mstride[0];
			float* const vp0 = vp + i[0] * vstride[0];
			float* const vmp0 = vmp + i[0] * vmstride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const np0 = np + i[0] * nstride[0];
			float* const up0 = up + i[0] * ustride[0];
			float* const ump0 = ump + i[0] * umstride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* gp1 = gp0 + i[1] * gstride[1];
				float* ap1 = ap0 + i[1] * astride[1];
				float* mp1 = mp0 + i[1] * mstride[1];
				float* vp1 = vp0 + i[1] * vstride[1];
				float* vmp1 = vmp0 + i[1] * vmstride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* np1 = np0 + i[1] * nstride[1];
				float* up1 = up0 + i[1] * ustride[1];
				float* ump1 = ump0 + i[1] * umstride[1];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					for (x = 0; x < adim[3]; x++)
					{
						const float grad = scale * gp1[x];
						const float mom = np1[x] = beta1 * mp1[x] + (1 - beta1) * grad;
						const float vel = up1[x] = beta2 * vp1[x] + (1 - beta2) * grad * grad;
						const float vel_hat = vel * inv_bias_correction2;
						const float vel_max_hat = ump1[x] = ccv_max(vmp1[x], vel_hat);
						bp1[x] = ap1[x] - rate_decay * ap1[x] - (mom * rate_inv_bias_correction1) / (sqrtf(vel_max_hat) + epsilon);
					}
					gp1 += gstride[2];
					ap1 += astride[2];
					mp1 += mstride[2];
					vp1 += vstride[2];
					vmp1 += vmstride[2];
					bp1 += bstride[2];
					np1 += nstride[2];
					up1 += ustride[2];
					ump1 += umstride[2];
				}
			}
		}
	} else {
		for (i[0] = 0; i[0] < adim[0]; i[0]++)
		{
			float* const gp0 = gp + i[0] * gstride[0];
			float* const ap0 = ap + i[0] * astride[0];
			float* const mp0 = mp + i[0] * mstride[0];
			float* const vp0 = vp + i[0] * vstride[0];
			float* const bp0 = bp + i[0] * bstride[0];
			float* const np0 = np + i[0] * nstride[0];
			float* const up0 = up + i[0] * ustride[0];
			for (i[1] = 0; i[1] < adim[1]; i[1]++)
			{
				float* gp1 = gp0 + i[1] * gstride[1];
				float* ap1 = ap0 + i[1] * astride[1];
				float* mp1 = mp0 + i[1] * mstride[1];
				float* vp1 = vp0 + i[1] * vstride[1];
				float* bp1 = bp0 + i[1] * bstride[1];
				float* np1 = np0 + i[1] * nstride[1];
				float* up1 = up0 + i[1] * ustride[1];
				for (i[2] = 0; i[2] < adim[2]; i[2]++)
				{
					for (x = 0; x < adim[3]; x++)
					{
						const float grad = scale * gp1[x];
						const float mom = np1[x] = beta1 * mp1[x] + (1 - beta1) * grad;
						const float vel = up1[x] = beta2 * vp1[x] + (1 - beta2) * grad * grad;
						bp1[x] = ap1[x] - rate_decay * ap1[x] - (mom * rate_inv_bias_correction1) / (sqrtf(vel * inv_bias_correction2) + epsilon);
					}
					gp1 += gstride[2];
					ap1 += astride[2];
					mp1 += mstride[2];
					vp1 += vstride[2];
					bp1 += bstride[2];
					np1 += nstride[2];
					up1 += ustride[2];
				}
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_adamw_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAMW_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adamw_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAMW_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adamw_back;
}
