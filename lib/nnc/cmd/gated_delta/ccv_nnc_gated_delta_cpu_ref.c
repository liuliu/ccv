#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>
#include <string.h>

static int _ccv_nnc_gated_delta_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6);
	assert(output_size == 2);
	const ccv_nnc_tensor_t* const q = inputs[0];
	const ccv_nnc_tensor_t* const k = inputs[1];
	const ccv_nnc_tensor_t* const v = inputs[2];
	const ccv_nnc_tensor_t* const log_decay = inputs[3];
	const ccv_nnc_tensor_t* const beta = inputs[4];
	const ccv_nnc_tensor_t* const state_in = inputs[5];
	ccv_nnc_tensor_t* const y = outputs[0];
	ccv_nnc_tensor_t* const state_out = outputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(log_decay));
	assert(CCV_IS_TENSOR_CONTIGUOUS(beta));
	assert(CCV_IS_TENSOR_CONTIGUOUS(state_in));
	assert(CCV_IS_TENSOR_CONTIGUOUS(y));
	assert(CCV_IS_TENSOR_CONTIGUOUS(state_out));
	assert(q->info.datatype == CCV_32F);
	assert(k->info.datatype == CCV_32F);
	assert(v->info.datatype == CCV_32F);
	assert(log_decay->info.datatype == CCV_32F);
	assert(beta->info.datatype == CCV_32F);
	assert(state_in->info.datatype == CCV_32F);
	assert(y->info.datatype == CCV_32F);
	assert(state_out->info.datatype == CCV_32F);
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	const int log_decay_nd = ccv_nnc_tensor_nd(log_decay->info.dim);
	const int beta_nd = ccv_nnc_tensor_nd(beta->info.dim);
	const int state_in_nd = ccv_nnc_tensor_nd(state_in->info.dim);
	const int y_nd = ccv_nnc_tensor_nd(y->info.dim);
	const int state_out_nd = ccv_nnc_tensor_nd(state_out->info.dim);
	assert(q_nd == 4);
	assert(k_nd == 4);
	assert(v_nd == 4);
	assert(log_decay_nd == 3);
	assert(beta_nd == 3);
	assert(state_in_nd == 4);
	assert(y_nd == 4);
	assert(state_out_nd == 4);
	const int B = q->info.dim[0];
	const int T = q->info.dim[1];
	const int Hk = q->info.dim[2];
	const int Dk = q->info.dim[3];
	const int Hv = v->info.dim[2];
	const int Dv = v->info.dim[3];
	assert(B > 0);
	assert(T > 0);
	assert(Hk > 0);
	assert(Dk > 0);
	assert(Hv > 0);
	assert(Dv > 0);
	assert(k->info.dim[0] == B);
	assert(k->info.dim[1] == T);
	assert(k->info.dim[2] == Hk);
	assert(k->info.dim[3] == Dk);
	assert(v->info.dim[0] == B);
	assert(v->info.dim[1] == T);
	assert(log_decay->info.dim[0] == B);
	assert(log_decay->info.dim[1] == T);
	assert(log_decay->info.dim[2] == Hv);
	assert(beta->info.dim[0] == B);
	assert(beta->info.dim[1] == T);
	assert(beta->info.dim[2] == Hv);
	assert(state_in->info.dim[0] == B);
	assert(state_in->info.dim[1] == Hv);
	assert(state_in->info.dim[2] == Dv);
	assert(state_in->info.dim[3] == Dk);
	assert(y->info.dim[0] == B);
	assert(y->info.dim[1] == T);
	assert(y->info.dim[2] == Hv);
	assert(y->info.dim[3] == Dv);
	assert(state_out->info.dim[0] == B);
	assert(state_out->info.dim[1] == Hv);
	assert(state_out->info.dim[2] == Dv);
	assert(state_out->info.dim[3] == Dk);
	assert(Hv % Hk == 0);
	const int log_decay_input = cmd.info.gated_delta.log_decay;
	const int hv_per_hk = Hv / Hk;
	const float* const qp = q->data.f32;
	const float* const kp = k->data.f32;
	const float* const vp = v->data.f32;
	const float* const gp = log_decay->data.f32;
	const float* const betap = beta->data.f32;
	const float* const state_inp = state_in->data.f32;
	float* const yp = y->data.f32;
	float* const state_outp = state_out->data.f32;
	int b, hv, dv, t, dk;
	for (b = 0; b < B; b++)
		for (hv = 0; hv < Hv; hv++)
		{
			const int hk = hv / hv_per_hk;
			for (dv = 0; dv < Dv; dv++)
			{
				const size_t state_offset = (((size_t)b * Hv + hv) * Dv + dv) * Dk;
				if (state_inp != state_outp)
					memcpy(state_outp + state_offset, state_inp + state_offset, sizeof(float) * Dk);
				float* const state_row = state_outp + state_offset;
				for (t = 0; t < T; t++)
				{
					const size_t qk_offset = (((size_t)b * T + t) * Hk + hk) * Dk;
					const float* const q_row = qp + qk_offset;
					const float* const k_row = kp + qk_offset;
					const size_t gate_offset = ((size_t)b * T + t) * Hv + hv;
					const float decay = log_decay_input ? expf(gp[gate_offset]) : gp[gate_offset];
					float memory = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						const float decayed = state_row[dk] * decay;
						state_row[dk] = decayed;
						memory += decayed * k_row[dk];
					}
					const size_t v_offset = (((size_t)b * T + t) * Hv + hv) * Dv + dv;
					const float delta = (vp[v_offset] - memory) * betap[gate_offset];
					float out = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						const float next = state_row[dk] + delta * k_row[dk];
						state_row[dk] = next;
						out += next * q_row[dk];
					}
					yp[v_offset] = out;
				}
			}
		}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gated_delta_forw;
}
