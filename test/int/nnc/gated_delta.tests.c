#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <string.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _gated_delta_int_value(const int i, const int multiplier, const int modulus, const float scale)
{
	return (float)(((i * multiplier) % modulus) - modulus / 2) * scale;
}

static void _gated_delta_reference(const float* const q, const float* const k, const float* const v, const float* const decay, const float* const beta, const float* const state_in, const int log_decay_input, const int B, const int T, const int Hk, const int Dk, const int Hv, const int Dv, float* const y, float* const state_out)
{
	const int hv_per_hk = Hv / Hk;
	int b, hv, dv, t, dk;
	for (b = 0; b < B; b++)
		for (hv = 0; hv < Hv; hv++)
		{
			const int hk = hv / hv_per_hk;
			for (dv = 0; dv < Dv; dv++)
			{
				const size_t state_offset = (((size_t)b * Hv + hv) * Dv + dv) * Dk;
				memcpy(state_out + state_offset, state_in + state_offset, sizeof(float) * Dk);
				for (t = 0; t < T; t++)
				{
					const size_t qk_offset = (((size_t)b * T + t) * Hk + hk) * Dk;
					const size_t gate_offset = ((size_t)b * T + t) * Hv + hv;
					const float decay_value = log_decay_input ? expf(decay[gate_offset]) : decay[gate_offset];
					float memory = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						state_out[state_offset + dk] *= decay_value;
						memory += state_out[state_offset + dk] * k[qk_offset + dk];
					}
					const size_t v_offset = (((size_t)b * T + t) * Hv + hv) * Dv + dv;
					const float delta = (v[v_offset] - memory) * beta[gate_offset];
					float out = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						const float next = state_out[state_offset + dk] + delta * k[qk_offset + dk];
						state_out[state_offset + dk] = next;
						out += next * q[qk_offset + dk];
					}
					y[v_offset] = out;
				}
			}
		}
}

TEST_CASE("compare gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 2;
	const int T = 5;
	const int Hk = 2;
	const int Dk = 65;
	const int Hv = 4;
	const int Dv = 5;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hdecay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.025f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 29, 0.02f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 37, 0.015f);
	for (i = 0; i < B * T * Hv; i++)
	{
		hlog_decay->data.f32[i] = -0.015f * (float)((i % 7) + 1);
		hdecay->data.f32[i] = expf(hlog_decay->data.f32[i]);
		hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hdecay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, decay, beta, state_in), TENSOR_LIST(y_decay, state_decay), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_decay, state_decay), TENSOR_LIST(hy_decay, hstate_decay), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hstate_in), TENSOR_LIST(state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_inplace), TENSOR_LIST(y_inplace, state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_inplace, state_inplace), TENSOR_LIST(hy_inplace, hstate_inplace), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "GPU gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "GPU gated delta state should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_decay->data.f32, B * T * Hv * Dv, 1e-4, "GPU gated delta output with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_decay->data.f32, B * Hv * Dv * Dk, 1e-4, "GPU gated delta state with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_inplace->data.f32, B * T * Hv * Dv, 1e-4, "in-place GPU gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_inplace->data.f32, B * Hv * Dv * Dk, 1e-4, "in-place GPU gated delta state should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hdecay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(y_decay);
	ccv_nnc_tensor_free(state_decay);
	ccv_nnc_tensor_free(y_inplace);
	ccv_nnc_tensor_free(state_inplace);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_out);
	ccv_nnc_tensor_free(hy_decay);
	ccv_nnc_tensor_free(hstate_decay);
	ccv_nnc_tensor_free(hy_inplace);
	ccv_nnc_tensor_free(hstate_inplace);
}

TEST_CASE("compare gated delta state checkpoints with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 1;
	const int T = 3;
	const int Hk = 2;
	const int Dk = 65;
	const int Hv = 4;
	const int Dv = 5;
	const int state_checkpoint_count = 2;
	const int state_history_count = state_checkpoint_count + 1;
	const int state_len = B * Hv * Dv * Dk;
	const int state_history_len = state_len * state_history_count;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state_history = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.025f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 29, 0.02f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 37, 0.015f);
	for (i = 0; i < B * T * Hv; i++)
	{
		hlog_decay->data.f32[i] = -0.015f * (float)((i % 7) + 1);
		hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
	}
	for (i = 0; i < state_len; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state_history), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_history = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_history = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_history), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_history), TENSOR_LIST(hy, hstate_history), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "GPU checkpointed gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state_history->data.f32, hstate_history->data.f32, state_history_len, 1e-4, "GPU checkpointed gated delta state history should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state_history);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_history);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_history);
}

TEST_CASE("compare bfloat gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 1;
	const int T = 4;
	const int Hk = 2;
	const int Dk = 64;
	const int Hv = 4;
	const int Dv = 4;
	const int qk_len = B * T * Hk * Dk;
	const int v_len = B * T * Hv * Dv;
	const int gate_len = B * T * Hv;
	const int state_len = B * Hv * Dv * Dk;
	ccv_nnc_tensor_t* const hq32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < qk_len; i++)
	{
		hq32->data.f32[i] = _gated_delta_int_value(i, 13, 67, 0.018f);
		hk32->data.f32[i] = _gated_delta_int_value(i, 17, 71, 0.016f);
	}
	for (i = 0; i < v_len; i++)
		hv32->data.f32[i] = _gated_delta_int_value(i, 19, 73, 0.014f);
	for (i = 0; i < gate_len; i++)
	{
		hlog_decay->data.f32[i] = -0.01f * (float)((i % 6) + 1);
		hbeta->data.f32[i] = 0.14f + 0.04f * (float)(i % 5);
	}
	for (i = 0; i < state_len; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 79, 0.005f);
	ccv_float_to_bfloat(hq32->data.f32, (uint16_t*)hq->data.f16, qk_len);
	ccv_float_to_bfloat(hk32->data.f32, (uint16_t*)hk->data.f16, qk_len);
	ccv_float_to_bfloat(hv32->data.f32, (uint16_t*)hv->data.f16, v_len);
	ccv_bfloat_to_float((uint16_t*)hq->data.f16, hq_rounded->data.f32, qk_len);
	ccv_bfloat_to_float((uint16_t*)hk->data.f16, hk_rounded->data.f32, qk_len);
	ccv_bfloat_to_float((uint16_t*)hv->data.f16, hv_rounded->data.f32, v_len);
	_gated_delta_reference(hq_rounded->data.f32, hk_rounded->data.f32, hv_rounded->data.f32, hlog_decay->data.f32, hbeta->data.f32, hstate_in->data.f32, 1, B, T, Hk, Dk, Hv, Dv, expected_y->data.f32, expected_state->data.f32);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_bfloat_to_float((uint16_t*)hy->data.f16, hy32->data.f32, v_len);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, 2e-2, "GPU bfloat gated delta output should match scalar reference after output rounding");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_len, 5e-4, "GPU bfloat gated delta state should match scalar reference implementation");
	ccv_nnc_tensor_free(hq32);
	ccv_nnc_tensor_free(hk32);
	ccv_nnc_tensor_free(hv32);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hq_rounded);
	ccv_nnc_tensor_free(hk_rounded);
	ccv_nnc_tensor_free(hv_rounded);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy32);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare half gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 1;
	const int T = 4;
	const int Hk = 2;
	const int Dk = 64;
	const int Hv = 4;
	const int Dv = 4;
	const int qk_len = B * T * Hk * Dk;
	const int v_len = B * T * Hv * Dv;
	const int gate_len = B * T * Hv;
	const int state_len = B * Hv * Dv * Dk;
	ccv_nnc_tensor_t* const hq32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < qk_len; i++)
	{
		hq32->data.f32[i] = _gated_delta_int_value(i, 13, 67, 0.018f);
		hk32->data.f32[i] = _gated_delta_int_value(i, 17, 71, 0.016f);
	}
	for (i = 0; i < v_len; i++)
		hv32->data.f32[i] = _gated_delta_int_value(i, 19, 73, 0.014f);
	for (i = 0; i < gate_len; i++)
	{
		hlog_decay->data.f32[i] = -0.01f * (float)((i % 6) + 1);
		hbeta->data.f32[i] = 0.14f + 0.04f * (float)(i % 5);
	}
	for (i = 0; i < state_len; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 79, 0.005f);
	ccv_float_to_half_precision(hq32->data.f32, (uint16_t*)hq->data.f16, qk_len);
	ccv_float_to_half_precision(hk32->data.f32, (uint16_t*)hk->data.f16, qk_len);
	ccv_float_to_half_precision(hv32->data.f32, (uint16_t*)hv->data.f16, v_len);
	ccv_half_precision_to_float((uint16_t*)hq->data.f16, hq_rounded->data.f32, qk_len);
	ccv_half_precision_to_float((uint16_t*)hk->data.f16, hk_rounded->data.f32, qk_len);
	ccv_half_precision_to_float((uint16_t*)hv->data.f16, hv_rounded->data.f32, v_len);
	_gated_delta_reference(hq_rounded->data.f32, hk_rounded->data.f32, hv_rounded->data.f32, hlog_decay->data.f32, hbeta->data.f32, hstate_in->data.f32, 1, B, T, Hk, Dk, Hv, Dv, expected_y->data.f32, expected_state->data.f32);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_half_precision_to_float((uint16_t*)hy->data.f16, hy32->data.f32, v_len);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, 5e-3, "GPU half gated delta output should match scalar reference after output rounding");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_len, 5e-4, "GPU half gated delta state should match scalar reference implementation");
	ccv_nnc_tensor_free(hq32);
	ccv_nnc_tensor_free(hk32);
	ccv_nnc_tensor_free(hv32);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hq_rounded);
	ccv_nnc_tensor_free(hk_rounded);
	ccv_nnc_tensor_free(hv_rounded);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy32);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare aligned gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 1;
	const int T = 4;
	const int Hk = 2;
	const int Dk = 64;
	const int Hv = 4;
	const int Dv = 4;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hdecay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 11, 43, 0.02f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 47, 0.018f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 53, 0.015f);
	for (i = 0; i < B * T * Hv; i++)
	{
		hlog_decay->data.f32[i] = -0.012f * (float)((i % 5) + 1);
		hdecay->data.f32[i] = expf(hlog_decay->data.f32[i]);
		hbeta->data.f32[i] = 0.16f + 0.03f * (float)(i % 7);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 59, 0.006f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hdecay, hbeta, hstate_in), TENSOR_LIST(q, k, v, decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "aligned GPU gated delta output with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "aligned GPU gated delta state with precomputed decay should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hdecay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare large aligned gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 1;
	const int T = 31;
	const int Hk = 16;
	const int Dk = 128;
	const int Hv = 32;
	const int Dv = 128;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hdecay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 11, 43, 0.02f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 47, 0.018f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 53, 0.015f);
	for (i = 0; i < B * T * Hv; i++)
	{
		hlog_decay->data.f32[i] = -0.012f * (float)((i % 5) + 1);
		hdecay->data.f32[i] = expf(hlog_decay->data.f32[i]);
		hbeta->data.f32[i] = 0.16f + 0.03f * (float)(i % 7);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 59, 0.006f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hdecay, hbeta, hstate_in), TENSOR_LIST(q, k, v, decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "large aligned GPU gated delta output with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "large aligned GPU gated delta state with precomputed decay should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hdecay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare gated delta key widths and state checkpoints with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Key widths below one warp, between 128 and 256, and above 256, with per head and per channel decay.
	const int B = 2;
	const int T = 6;
	const int Hk = 2;
	const int Hv = 4;
	const int Dv = 6;
	const int state_checkpoint_count = 2;
	const int key_widths[] = {24, 200, 300};
	int i, j, channel_decay;
	for (j = 0; j < sizeof(key_widths) / sizeof(key_widths[0]); j++)
		for (channel_decay = 0; channel_decay < 2; channel_decay++)
		{
			const int Dk = key_widths[j];
			const int decay_len = B * T * Hv * (channel_decay ? Dk : 1);
			const int state_history_len = B * Hv * (state_checkpoint_count + 1) * Dv * Dk;
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
			ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
			ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
			ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, channel_decay ? CPU_TENSOR_NHWC(32F, B, T, Hv, Dk) : CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
			ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
			ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
			ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
			ccv_nnc_tensor_t* const expected_state_history = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * (state_checkpoint_count + 1), Dv, Dk), 0);
			for (i = 0; i < B * T * Hk * Dk; i++)
			{
				hq->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.012f);
				hk->data.f32[i] = _gated_delta_int_value(i, 17, 29, 0.01f);
			}
			for (i = 0; i < B * T * Hv * Dv; i++)
				hv->data.f32[i] = _gated_delta_int_value(i, 19, 37, 0.015f);
			for (i = 0; i < decay_len; i++)
				hlog_decay->data.f32[i] = -0.01f * (float)((i % 9) + 1);
			for (i = 0; i < B * T * Hv; i++)
				hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
			for (i = 0; i < B * Hv * Dv * Dk; i++)
				hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
			ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state_history), 0);
			ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
			ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
			ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
			ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, channel_decay ? GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dk) : GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
			ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
			ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
			ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
			ccv_nnc_tensor_t* const state_history = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv * (state_checkpoint_count + 1), Dv, Dk), 0);
			ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
			ccv_nnc_tensor_t* const hstate_history = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * (state_checkpoint_count + 1), Dv, Dk), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
			REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_history), 0), CCV_NNC_EXEC_SUCCESS, "GPU gated delta should execute for Dk=%d", Dk);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_history), TENSOR_LIST(hy, hstate_history), 0);
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "GPU gated delta output should match CPU reference implementation for Dk=%d channel_decay=%d", Dk, channel_decay);
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state_history->data.f32, hstate_history->data.f32, state_history_len, 1e-4, "GPU gated delta state history should match CPU reference implementation for Dk=%d channel_decay=%d", Dk, channel_decay);
			ccv_nnc_tensor_free(hq);
			ccv_nnc_tensor_free(hk);
			ccv_nnc_tensor_free(hv);
			ccv_nnc_tensor_free(hlog_decay);
			ccv_nnc_tensor_free(hbeta);
			ccv_nnc_tensor_free(hstate_in);
			ccv_nnc_tensor_free(expected_y);
			ccv_nnc_tensor_free(expected_state_history);
			ccv_nnc_tensor_free(q);
			ccv_nnc_tensor_free(k);
			ccv_nnc_tensor_free(v);
			ccv_nnc_tensor_free(log_decay);
			ccv_nnc_tensor_free(beta);
			ccv_nnc_tensor_free(state_in);
			ccv_nnc_tensor_free(y);
			ccv_nnc_tensor_free(state_history);
			ccv_nnc_tensor_free(hy);
			ccv_nnc_tensor_free(hstate_history);
		}
}

TEST_CASE("compare gated delta per key channel decay with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Same shape as "compare gated delta with gpu", so the per head and per channel pipelines differ only by the decay layout.
	const int B = 2;
	const int T = 5;
	const int Hk = 2;
	const int Dk = 65;
	const int Hv = 4;
	const int Dv = 5;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const hdecay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.025f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 29, 0.02f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 37, 0.015f);
	for (i = 0; i < B * T * Hv * Dk; i++)
	{
		// Spread gates across key channels so a per head broadcast would not match.
		hlog_decay->data.f32[i] = -0.04f * (float)((i * 7) % 11 + 1);
		hdecay->data.f32[i] = expf(hlog_decay->data.f32[i]);
	}
	for (i = 0; i < B * T * Hv; i++)
		hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hdecay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, decay, beta, state_in), TENSOR_LIST(y_decay, state_decay), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_decay, state_decay), TENSOR_LIST(hy_decay, hstate_decay), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hstate_in), TENSOR_LIST(state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_inplace), TENSOR_LIST(y_inplace, state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_inplace, state_inplace), TENSOR_LIST(hy_inplace, hstate_inplace), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "GPU gated delta output with per channel log decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "GPU gated delta state with per channel log decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_decay->data.f32, B * T * Hv * Dv, 1e-4, "GPU gated delta output with per channel decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_decay->data.f32, B * Hv * Dv * Dk, 1e-4, "GPU gated delta state with per channel decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_inplace->data.f32, B * T * Hv * Dv, 1e-4, "in-place GPU gated delta output with per channel log decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_inplace->data.f32, B * Hv * Dv * Dk, 1e-4, "in-place GPU gated delta state with per channel log decay should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hdecay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(y_decay);
	ccv_nnc_tensor_free(state_decay);
	ccv_nnc_tensor_free(y_inplace);
	ccv_nnc_tensor_free(state_inplace);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_out);
	ccv_nnc_tensor_free(hy_decay);
	ccv_nnc_tensor_free(hstate_decay);
	ccv_nnc_tensor_free(hy_inplace);
	ccv_nnc_tensor_free(hstate_inplace);
}

TEST_CASE("compare bfloat gated delta per key channel decay with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 2;
	const int T = 4;
	const int Hk = 2;
	const int Dk = 128;
	const int Hv = 4;
	const int Dv = 8;
	const int qk_len = B * T * Hk * Dk;
	const int v_len = B * T * Hv * Dv;
	const int gate_len = B * T * Hv;
	const int state_len = B * Hv * Dv * Dk;
	ccv_nnc_tensor_t* const hq32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hq_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hlog_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < qk_len; i++)
	{
		hq32->data.f32[i] = _gated_delta_int_value(i, 13, 67, 0.018f);
		hk32->data.f32[i] = _gated_delta_int_value(i, 17, 71, 0.016f);
	}
	for (i = 0; i < v_len; i++)
		hv32->data.f32[i] = _gated_delta_int_value(i, 19, 73, 0.014f);
	for (i = 0; i < gate_len * Dk; i++)
		hlog_decay->data.f32[i] = -0.04f * (float)((i * 7) % 11 + 1);
	for (i = 0; i < gate_len; i++)
		hbeta->data.f32[i] = 0.14f + 0.04f * (float)(i % 5);
	for (i = 0; i < state_len; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 79, 0.005f);
	ccv_float_to_bfloat(hq32->data.f32, (uint16_t*)hq->data.f16, qk_len);
	ccv_float_to_bfloat(hk32->data.f32, (uint16_t*)hk->data.f16, qk_len);
	ccv_float_to_bfloat(hv32->data.f32, (uint16_t*)hv->data.f16, v_len);
	ccv_bfloat_to_float((uint16_t*)hq->data.f16, hq_rounded->data.f32, qk_len);
	ccv_bfloat_to_float((uint16_t*)hk->data.f16, hk_rounded->data.f32, qk_len);
	ccv_bfloat_to_float((uint16_t*)hv->data.f16, hv_rounded->data.f32, v_len);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(hq_rounded, hk_rounded, hv_rounded, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_bfloat_to_float((uint16_t*)hy->data.f16, hy32->data.f32, v_len);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, 2e-2, "GPU bfloat gated delta output with per channel log decay should match CPU reference after output rounding");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_len, 5e-4, "GPU bfloat gated delta state with per channel log decay should match CPU reference implementation");
	ccv_nnc_tensor_free(hq32);
	ccv_nnc_tensor_free(hk32);
	ccv_nnc_tensor_free(hv32);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hq_rounded);
	ccv_nnc_tensor_free(hk_rounded);
	ccv_nnc_tensor_free(hv_rounded);
	ccv_nnc_tensor_free(hlog_decay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy32);
	ccv_nnc_tensor_free(hstate_out);
}

// From 256 steps with key dims that are multiples of 32, the MPS backend runs the blocked
// kernel: threadgroups of 16, 32 or 64 value columns (by Dv) over staged timestep blocks.
static void _gated_delta_blocked_case(char* __case_name__, int* __case_result__, const int B, const int T, const int Hk, const int Hv, const int Dk, const int Dv, const int channel_decay, const int log_decay, const int state_checkpoint_count, const int half, const int in_place, const int load_m)
{
	const int decay_width = channel_decay ? Dk : 1;
	const int state_history_count = state_checkpoint_count + 1;
	const int qk_len = B * T * Hk * Dk;
	const int v_len = B * T * Hv * Dv;
	const int gate_len = B * T * Hv;
	const int state_len = B * Hv * Dv * Dk;
	const int state_out_len = state_len * state_history_count;
	const ccv_nnc_tensor_param_t decay_params = channel_decay ? CPU_TENSOR_NHWC(32F, B, T, Hv, Dk) : CPU_TENSOR_NHWC(32F, B, T, Hv);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hdecay = ccv_nnc_tensor_new(0, decay_params, 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	int i;
	// Small keys keep beta * |k|^2 well below 2, so the recurrence stays bounded over hundreds of steps.
	for (i = 0; i < qk_len; i++)
	{
		hq->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.01f);
		hk->data.f32[i] = _gated_delta_int_value(i, 17, 29, 0.004f);
	}
	for (i = 0; i < v_len; i++)
		hv->data.f32[i] = _gated_delta_int_value(i, 19, 37, 0.015f);
	for (i = 0; i < gate_len * decay_width; i++)
	{
		const float g = -0.01f * (float)((i * 7) % 12 + 1);
		hdecay->data.f32[i] = log_decay ? g : expf(g);
	}
	for (i = 0; i < gate_len; i++)
		hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
	for (i = 0; i < state_len; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	ccv_nnc_tensor_t* hq16 = 0;
	ccv_nnc_tensor_t* hk16 = 0;
	ccv_nnc_tensor_t* hv16 = 0;
	if (half)
	{
		// The reference sees the same rounded inputs as the GPU.
		hq16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hk, Dk), 0);
		hk16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hk, Dk), 0);
		hv16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, T, Hv, Dv), 0);
		ccv_float_to_half_precision(hq->data.f32, (uint16_t*)hq16->data.f16, qk_len);
		ccv_float_to_half_precision(hk->data.f32, (uint16_t*)hk16->data.f16, qk_len);
		ccv_float_to_half_precision(hv->data.f32, (uint16_t*)hv16->data.f16, v_len);
		ccv_half_precision_to_float((uint16_t*)hq16->data.f16, hq->data.f32, qk_len);
		ccv_half_precision_to_float((uint16_t*)hk16->data.f16, hk->data.f32, qk_len);
		ccv_half_precision_to_float((uint16_t*)hv16->data.f16, hv->data.f32, v_len);
	}
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(log_decay, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hdecay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_param_t gpu_decay_params = decay_params;
	gpu_decay_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, half ? GPU_TENSOR_NHWC(000, 16F, B, T, Hk, Dk) : GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, half ? GPU_TENSOR_NHWC(000, 16F, B, T, Hk, Dk) : GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, half ? GPU_TENSOR_NHWC(000, 16F, B, T, Hv, Dv) : GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const decay = ccv_nnc_tensor_new(0, gpu_decay_params, 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, half ? GPU_TENSOR_NHWC(000, 16F, B, T, Hv, Dv) : GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = in_place ? state_in : ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(half ? hq16 : hq, half ? hk16 : hk, half ? hv16 : hv, hdecay, hbeta, hstate_in), TENSOR_LIST(q, k, v, decay, beta, state_in), 0);
	if (load_m)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(log_decay, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	if (load_m)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, half ? CPU_TENSOR_NHWC(16F, B, T, Hv, Dv) : CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	if (half)
		ccv_half_precision_to_float((uint16_t*)hy->data.f16, hy32->data.f32, v_len);
	else
		memcpy(hy32->data.f32, hy->data.f32, sizeof(float) * v_len);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, half ? 5e-3 : 1e-4, "blocked GPU gated delta output should match CPU reference (T %d, Dk %d, Dv %d)", T, Dk, Dv);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_out_len, half ? 5e-4 : 1e-4, "blocked GPU gated delta state should match CPU reference (T %d, Dk %d, Dv %d)", T, Dk, Dv);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hdecay);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	if (half)
	{
		ccv_nnc_tensor_free(hq16);
		ccv_nnc_tensor_free(hk16);
		ccv_nnc_tensor_free(hv16);
	}
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	if (!in_place)
		ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy32);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare blocked gated delta per key channel decay with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// 16-column threadgroups with a partial last one (Dv 40), a partial last time block (T 300), GQA and batches.
	_gated_delta_blocked_case(__case_name__, __case_result__, 2, 300, 2, 4, 128, 40, 1, 1, 0, 0, 0, 0);
	// The same with the sequence length loaded from a buffer, in place.
	_gated_delta_blocked_case(__case_name__, __case_result__, 2, 300, 2, 4, 128, 40, 1, 1, 0, 0, 1, 1);
}

TEST_CASE("compare blocked gated delta state checkpoints with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// 64-column threadgroups (Dv 64), per head linear decay.
	_gated_delta_blocked_case(__case_name__, __case_result__, 1, 300, 2, 4, 128, 64, 0, 0, 3, 0, 0, 0);
}

TEST_CASE("compare blocked gated delta narrow keys with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// 32-column threadgroups (Dv 96), two state chunks per lane (Dk 64), three value heads per key head, in place.
	_gated_delta_blocked_case(__case_name__, __case_result__, 1, 270, 1, 3, 64, 96, 1, 0, 0, 0, 1, 0);
}

TEST_CASE("compare half blocked gated delta with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// GLM-5.3 KDA head shape (Dk 128, Dv 128) with FP16 inputs and per key channel log decay.
	_gated_delta_blocked_case(__case_name__, __case_result__, 1, 257, 2, 2, 128, 128, 1, 1, 0, 1, 0, 0);
}

TEST_CASE("compare blocked gated delta with shorter time blocks for threadgroup memory with gpu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// FP32 inputs at GLM-5.3's KDA head shape (Dk 128, Dv 128, per key channel decay) need 32,832 bytes at 16 steps.
	_gated_delta_blocked_case(__case_name__, __case_result__, 1, 261, 2, 2, 128, 128, 1, 1, 0, 0, 0, 0);
	// FP16 inputs with 256-wide keys and per key channel decay need 36,928 bytes at 16 steps.
	_gated_delta_blocked_case(__case_name__, __case_result__, 1, 261, 1, 1, 256, 128, 1, 1, 0, 1, 0, 0);
}

static float _gated_delta_test_sigmoid(const float x)
{
	return 1 / (1 + expf(-x));
}

// The fused inputs: q, k and v are strided views of one convolution output, L2-normalized q and k, a raw sigmoid gate with bias and per head scale, and sigmoid(beta). The reference runs
// the plain op on those inputs prepared on the CPU.
static void _gated_delta_fused_case(char* __case_name__, int* __case_result__, const int T, const int Hk, const int Hv, const int Dk, const int Dv, const int half, const int state_checkpoint_count)
{
	const int B = 1;
	const int width = 2 * Hk * Dk + Hv * Dv;
	const int state_history_count = state_checkpoint_count + 1;
	const float q_scale = 1 / sqrtf((float)Dk);
	const float epsilon = 1e-6;
	const float lower_bound = -5;
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, width), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const hbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, Hv * Dk), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, Hv), 0);
	ccv_nnc_tensor_t* const hstate_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i, t, h, d;
	for (i = 0; i < T * width; i++)
		hx->data.f32[i] = _gated_delta_int_value(i, 13, 31, 0.12f);
	for (i = 0; i < T * Hv * Dk; i++)
		hgate->data.f32[i] = _gated_delta_int_value(i, 7, 23, 0.2f);
	for (i = 0; i < T * Hv; i++)
		hbeta->data.f32[i] = _gated_delta_int_value(i, 5, 11, 0.3f);
	for (i = 0; i < Hv * Dk; i++)
		hbias->data.f32[i] = _gated_delta_int_value(i, 3, 17, 0.1f);
	for (i = 0; i < Hv; i++)
		hscale->data.f32[i] = 0.5f + 0.25f * (float)(i % 5);
	for (i = 0; i < Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	if (half)
	{
		// The fused GPU op reads FP16 inputs; the reference prepares from the same rounded values.
		uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * T * width);
		ccv_float_to_half_precision(hx->data.f32, scratch, T * width);
		ccv_half_precision_to_float(scratch, hx->data.f32, T * width);
		ccv_float_to_half_precision(hgate->data.f32, scratch, T * Hv * Dk);
		ccv_half_precision_to_float(scratch, hgate->data.f32, T * Hv * Dk);
		ccv_float_to_half_precision(hbeta->data.f32, scratch, T * Hv);
		ccv_half_precision_to_float(scratch, hbeta->data.f32, T * Hv);
		ccfree(scratch);
	}
	// The prepared inputs for the plain op.
	ccv_nnc_tensor_t* const pq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const pk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const pv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const pdecay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const pbeta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	for (t = 0; t < T; t++)
	{
		for (h = 0; h < Hk; h++)
		{
			float qs = 0, ks = 0;
			for (d = 0; d < Dk; d++)
			{
				const float qv = hx->data.f32[t * width + h * Dk + d];
				const float kv = hx->data.f32[t * width + Hk * Dk + h * Dk + d];
				pq->data.f32[(t * Hk + h) * Dk + d] = qv;
				pk->data.f32[(t * Hk + h) * Dk + d] = kv;
				qs += qv * qv;
				ks += kv * kv;
			}
			for (d = 0; d < Dk; d++)
			{
				pq->data.f32[(t * Hk + h) * Dk + d] *= q_scale / sqrtf(qs + epsilon);
				pk->data.f32[(t * Hk + h) * Dk + d] /= sqrtf(ks + epsilon);
			}
		}
		for (i = 0; i < Hv * Dv; i++)
		{
			pv->data.f32[t * Hv * Dv + i] = hx->data.f32[t * width + 2 * Hk * Dk + i];
		}
		for (h = 0; h < Hv; h++)
		{
			for (d = 0; d < Dk; d++)
				pdecay->data.f32[(t * Hv + h) * Dk + d] = lower_bound * _gated_delta_test_sigmoid(hscale->data.f32[h] * (hgate->data.f32[(t * Hv + h) * Dk + d] + hbias->data.f32[h * Dk + d]));
			pbeta->data.f32[t * Hv + h] = _gated_delta_test_sigmoid(hbeta->data.f32[t * Hv + h]);
		}
	}
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(pq, pk, pv, pdecay, pbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_cmd_t fused = CMD_GATED_DELTA_FORWARD(1, state_checkpoint_count);
	fused.info.gated_delta.qk_l2norm = 1;
	fused.info.gated_delta.q_scale = q_scale;
	fused.info.gated_delta.sigmoid_gate = 1;
	fused.info.gated_delta.gate_lower_bound = lower_bound;
	fused.info.gated_delta.sigmoid_beta = 1;
	// Views of the convolution output with its row stride.
	const int stride[CCV_NNC_MAX_DIM_ALLOC] = { T * width, width, Dk, 1 };
	const int v_stride[CCV_NNC_MAX_DIM_ALLOC] = { T * width, width, Dv, 1 };
	const int q_ofs[CCV_NNC_MAX_DIM_ALLOC] = { 0, 0, 0, 0 };
	const int k_ofs[CCV_NNC_MAX_DIM_ALLOC] = { 0, 0, Hk, 0 };
	const int v_ofs[CCV_NNC_MAX_DIM_ALLOC] = { 0, 0, 0, 2 * Hk * Dk };
	ccv_nnc_tensor_t hx4 = ccv_nnc_tensor(hx->data.u8, CPU_TENSOR_NHWC(32F, B, T, 1, width), 0);
	ccv_nnc_tensor_t hxk = ccv_nnc_tensor(hx->data.u8, CPU_TENSOR_NHWC(32F, B, T, width / Dk, Dk), 0);
	ccv_nnc_tensor_view_t* const hqv = ccv_nnc_tensor_view_new(&hxk, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), q_ofs, stride);
	ccv_nnc_tensor_view_t* const hkv = ccv_nnc_tensor_view_new(&hxk, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), k_ofs, stride);
	ccv_nnc_tensor_view_t* const hvv = ccv_nnc_tensor_view_new(&hx4, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), v_ofs, v_stride);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const cpu_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(fused, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)hqv, (ccv_nnc_tensor_t*)hkv, (ccv_nnc_tensor_t*)hvv, hgate, hbeta, hstate_in, hbias, hscale), TENSOR_LIST(cpu_y, cpu_state), 0), CCV_NNC_EXEC_SUCCESS, "CPU fused gated delta should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, cpu_y->data.f32, T * Hv * Dv, 1e-5, "CPU fused gated delta output should match the prepared inputs (T %d)", T);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, cpu_state->data.f32, Hv * state_history_count * Dv * Dk, 1e-5, "CPU fused gated delta state should match the prepared inputs (T %d)", T);
	// The GPU reads the same views of a GPU copy, in FP32 or FP16.
	const int dt = half ? CCV_16F : CCV_32F;
	ccv_nnc_tensor_param_t xp = GPU_TENSOR_NHWC(000, 32F, B, T, width / Dk, Dk);
	xp.datatype = dt;
	ccv_nnc_tensor_param_t gp = GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dk);
	gp.datatype = dt;
	ccv_nnc_tensor_param_t bp = GPU_TENSOR_NHWC(000, 32F, B, T, Hv);
	bp.datatype = dt;
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, xp, 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, gp, 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, bp, 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, Hv * Dk), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t hxg = ccv_nnc_tensor(hx->data.u8, CPU_TENSOR_NHWC(32F, B, T, width / Dk, Dk), 0);
	ccv_nnc_tensor_t* const hxt = ccv_nnc_tensor_new(0, half ? CPU_TENSOR_NHWC(16F, B, T, width / Dk, Dk) : CPU_TENSOR_NHWC(32F, B, T, width / Dk, Dk), 0);
	ccv_nnc_tensor_t* const hgt = ccv_nnc_tensor_new(0, half ? CPU_TENSOR_NHWC(16F, B, T, Hv, Dk) : CPU_TENSOR_NHWC(32F, B, T, Hv, Dk), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, half ? CPU_TENSOR_NHWC(16F, B, T, Hv) : CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&hxg, hgate, hbeta), TENSOR_LIST(hxt, hgt, hbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hxt, hgt, hbt, hbias, hscale, hstate_in), TENSOR_LIST(x, gate, beta, bias, scale, state_in), 0);
	ccv_nnc_tensor_param_t qp = GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk);
	qp.datatype = dt;
	ccv_nnc_tensor_param_t vp = GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv);
	vp.datatype = dt;
	ccv_nnc_tensor_param_t x4p = GPU_TENSOR_NHWC(000, 32F, B, T, 1, width);
	x4p.datatype = dt;
	ccv_nnc_tensor_t x4 = ccv_nnc_tensor(x->data.u8, x4p, 0);
	ccv_nnc_tensor_view_t* const qv = ccv_nnc_tensor_view_new(x, qp, q_ofs, stride);
	ccv_nnc_tensor_view_t* const kv = ccv_nnc_tensor_view_new(x, qp, k_ofs, stride);
	ccv_nnc_tensor_view_t* const vv = ccv_nnc_tensor_view_new(&x4, vp, v_ofs, v_stride);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, vp, 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv * state_history_count, Dv, Dk), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(fused, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)qv, (ccv_nnc_tensor_t*)kv, (ccv_nnc_tensor_t*)vv, gate, beta, state_in, bias, scale), TENSOR_LIST(y, state_out), 0), CCV_NNC_EXEC_SUCCESS, "GPU fused gated delta should execute");
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, half ? CPU_TENSOR_NHWC(16F, B, T, Hv, Dv) : CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv * state_history_count, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy), TENSOR_LIST(hy32), 0);
	// The reference reads the same FP16-rounded inputs, so FP16 adds only the output's rounding.
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, T * Hv * Dv, 1e-4, "GPU fused gated delta output should match the prepared inputs (T %d, half %d)", T, half);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, Hv * state_history_count * Dv * Dk, 1e-4, "GPU fused gated delta state should match the prepared inputs (T %d, half %d)", T, half);
	ccv_nnc_tensor_view_free(qv);
	ccv_nnc_tensor_view_free(kv);
	ccv_nnc_tensor_view_free(vv);
	ccv_nnc_tensor_view_free(hqv);
	ccv_nnc_tensor_view_free(hkv);
	ccv_nnc_tensor_view_free(hvv);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(hbeta);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(hstate_in);
	ccv_nnc_tensor_free(pq);
	ccv_nnc_tensor_free(pk);
	ccv_nnc_tensor_free(pv);
	ccv_nnc_tensor_free(pdecay);
	ccv_nnc_tensor_free(pbeta);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_state);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(hxt);
	ccv_nnc_tensor_free(hgt);
	ccv_nnc_tensor_free(hbt);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy32);
	ccv_nnc_tensor_free(hstate_out);
}

TEST_CASE("compare fused gated delta with prepared inputs")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
	// Decode and verify lengths take the per-column kernel, with state checkpoints.
	_gated_delta_fused_case(__case_name__, __case_result__, 1, 4, 4, 128, 128, 0, 0);
	_gated_delta_fused_case(__case_name__, __case_result__, 5, 4, 4, 128, 128, 1, 3);
	// Prefill lengths take the blocked kernel, with a partial last time block.
	_gated_delta_fused_case(__case_name__, __case_result__, 261, 2, 2, 128, 128, 0, 0);
	_gated_delta_fused_case(__case_name__, __case_result__, 261, 2, 2, 128, 128, 1, 2);
}

#include "case_main.h"
