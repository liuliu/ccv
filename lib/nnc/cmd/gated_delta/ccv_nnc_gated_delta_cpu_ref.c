#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>
#include <string.h>

static inline float _ccv_nnc_gated_delta_sigmoid(const float x)
{
	return 1 / (1 + expf(-x));
}

// Elements between consecutive tokens (dimension 1).
static size_t _ccv_nnc_gated_delta_row_stride(const ccv_nnc_tensor_t* const tensor)
{
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	if (CCV_IS_TENSOR_VIEW(tensor))
		return ((const ccv_nnc_tensor_view_t*)tensor)->stride[1];
	size_t stride = 1;
	int i;
	for (i = 2; i < nd; i++)
		stride *= tensor->info.dim[i];
	return stride;
}

// Everything below a token row is packed, and batches follow each other row by row.
static int _ccv_nnc_gated_delta_rows_are_packed(const ccv_nnc_tensor_t* const tensor)
{
	if (!CCV_IS_TENSOR_VIEW(tensor) || CCV_IS_TENSOR_CONTIGUOUS(tensor))
		return 1;
	const ccv_nnc_tensor_view_t* const view = (const ccv_nnc_tensor_view_t*)tensor;
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	int i;
	int stride = 1;
	for (i = nd - 1; i >= 2; i--)
	{
		if (view->stride[i] != stride)
			return 0;
		stride *= tensor->info.dim[i];
	}
	return view->stride[1] >= stride && (tensor->info.dim[0] == 1 || view->stride[0] == view->stride[1] * tensor->info.dim[1]);
}

static int _ccv_nnc_gated_delta_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int sigmoid_gate = cmd.info.gated_delta.sigmoid_gate;
	assert(input_size == (sigmoid_gate ? 8 : 6));
	assert(output_size == 2);
	const ccv_nnc_tensor_t* const q = inputs[0];
	const ccv_nnc_tensor_t* const k = inputs[1];
	const ccv_nnc_tensor_t* const v = inputs[2];
	const ccv_nnc_tensor_t* const log_decay = inputs[3];
	const ccv_nnc_tensor_t* const beta = inputs[4];
	const ccv_nnc_tensor_t* const state_in = inputs[5];
	ccv_nnc_tensor_t* const y = outputs[0];
	ccv_nnc_tensor_t* const state_out = outputs[1];
	// q, k, v, the decay and beta may stride their token rows; everything below a row is packed.
	assert(_ccv_nnc_gated_delta_rows_are_packed(q));
	assert(_ccv_nnc_gated_delta_rows_are_packed(k));
	assert(_ccv_nnc_gated_delta_rows_are_packed(v));
	assert(_ccv_nnc_gated_delta_rows_are_packed(log_decay));
	assert(_ccv_nnc_gated_delta_rows_are_packed(beta));
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
	assert(log_decay_nd == 3 || log_decay_nd == 4);
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
	// A rank-4 decay carries one gate per key channel (KDA-style); rank-3 shares one gate per head.
	const int channel_decay = log_decay_nd == 4;
	assert(!channel_decay || log_decay->info.dim[3] == Dk);
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
	const int state_checkpoint_count = cmd.info.gated_delta.state_checkpoint_count;
	assert(state_checkpoint_count >= 0);
	assert(state_checkpoint_count < T);
	const int state_history_count = state_checkpoint_count + 1;
	assert(state_out->info.dim[1] == Hv * state_history_count);
	assert(state_out->info.dim[2] == Dv);
	assert(state_out->info.dim[3] == Dk);
	assert(Hv % Hk == 0);
	const int log_decay_input = cmd.info.gated_delta.log_decay;
	const int qk_l2norm = cmd.info.gated_delta.qk_l2norm;
	// FLA's KDA L2 norm, x / sqrt(sum(x^2) + 1e-6).
	const float l2norm_epsilon = 1e-6;
	const float q_scale = cmd.info.gated_delta.q_scale;
	const float gate_lower_bound = cmd.info.gated_delta.gate_lower_bound;
	const int sigmoid_beta = cmd.info.gated_delta.sigmoid_beta;
	assert(!sigmoid_gate || (log_decay_input && channel_decay));
	const float* const gate_bias = sigmoid_gate ? inputs[6]->data.f32 : 0;
	const float* const gate_scale = sigmoid_gate ? inputs[7]->data.f32 : 0;
	if (sigmoid_gate)
	{
		assert(inputs[6]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(inputs[6]) && ccv_nnc_tensor_count(inputs[6]->info) == Hv * Dk);
		assert(inputs[7]->info.datatype == CCV_32F && CCV_IS_TENSOR_CONTIGUOUS(inputs[7]) && ccv_nnc_tensor_count(inputs[7]->info) == Hv);
	}
	const size_t q_row_stride = _ccv_nnc_gated_delta_row_stride(q);
	const size_t k_row_stride = _ccv_nnc_gated_delta_row_stride(k);
	const size_t v_row_stride = _ccv_nnc_gated_delta_row_stride(v);
	const size_t gate_row_stride = _ccv_nnc_gated_delta_row_stride(log_decay);
	const size_t beta_row_stride = _ccv_nnc_gated_delta_row_stride(beta);
	const int hv_per_hk = Hv / Hk;
	float* const q_row = (float*)ccmalloc(sizeof(float) * Dk * 2);
	float* const k_row = q_row + Dk;
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
				const size_t state_in_offset = (((size_t)b * Hv + hv) * Dv + dv) * Dk;
				const size_t final_state_offset = (((size_t)b * state_history_count * Hv + hv) * Dv + dv) * Dk;
				if (state_inp != state_outp)
					memcpy(state_outp + final_state_offset, state_inp + state_in_offset, sizeof(float) * Dk);
				float* const state_row = state_outp + final_state_offset;
				for (t = 0; t < T; t++)
				{
					const size_t row = (size_t)b * T + t;
					const float* const q_in = qp + row * q_row_stride + (size_t)hk * Dk;
					const float* const k_in = kp + row * k_row_stride + (size_t)hk * Dk;
					float q_sum = 0, k_sum = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						q_row[dk] = q_in[dk];
						k_row[dk] = k_in[dk];
						q_sum += q_row[dk] * q_row[dk];
						k_sum += k_row[dk] * k_row[dk];
					}
					if (qk_l2norm)
					{
						const float q_norm = q_scale / sqrtf(q_sum + l2norm_epsilon);
						const float k_norm = 1 / sqrtf(k_sum + l2norm_epsilon);
						for (dk = 0; dk < Dk; dk++)
						{
							q_row[dk] *= q_norm;
							k_row[dk] *= k_norm;
						}
					}
					const float* const decay_row = gp + row * gate_row_stride + (size_t)hv * (channel_decay ? Dk : 1);
					const float head_decay = channel_decay ? 0 : (log_decay_input ? expf(decay_row[0]) : decay_row[0]);
					float memory = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						float decay = head_decay;
						// Only a per key channel decay has Dk values per head.
						if (channel_decay)
						{
							const float gate = sigmoid_gate ? gate_lower_bound * _ccv_nnc_gated_delta_sigmoid(gate_scale[hv] * (decay_row[dk] + gate_bias[(size_t)hv * Dk + dk])) : decay_row[dk];
							decay = log_decay_input ? expf(gate) : gate;
						}
						const float decayed = state_row[dk] * decay;
						state_row[dk] = decayed;
						memory += decayed * k_row[dk];
					}
					const size_t v_offset = (((size_t)b * T + t) * Hv + hv) * Dv + dv;
					const float v_in = vp[row * v_row_stride + (size_t)hv * Dv + dv];
					const float beta_in = betap[row * beta_row_stride + hv];
					const float delta = (v_in - memory) * (sigmoid_beta ? _ccv_nnc_gated_delta_sigmoid(beta_in) : beta_in);
					float out = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						const float next = state_row[dk] + delta * k_row[dk];
						state_row[dk] = next;
						out += next * q_row[dk];
					}
					yp[v_offset] = out;
					const int history_index = T - 1 - t;
					if (history_index > 0 && history_index <= state_checkpoint_count)
					{
						const size_t checkpoint_state_offset = (((size_t)b * state_history_count * Hv + history_index * Hv + hv) * Dv + dv) * Dk;
						memcpy(state_outp + checkpoint_state_offset, state_row, sizeof(float) * Dk);
					}
				}
			}
		}
	ccfree(q_row);
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
