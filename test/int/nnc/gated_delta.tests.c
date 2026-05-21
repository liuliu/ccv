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

TEST_CASE("compare gated delta with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "MPS gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "MPS gated delta state should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_decay->data.f32, B * T * Hv * Dv, 1e-4, "MPS gated delta output with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_decay->data.f32, B * Hv * Dv * Dk, 1e-4, "MPS gated delta state with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_inplace->data.f32, B * T * Hv * Dv, 1e-4, "in-place MPS gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_inplace->data.f32, B * Hv * Dv * Dk, 1e-4, "in-place MPS gated delta state should match CPU reference implementation");
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

TEST_CASE("compare gated delta state checkpoints with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "MPS checkpointed gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state_history->data.f32, hstate_history->data.f32, state_history_len, 1e-4, "MPS checkpointed gated delta state history should match CPU reference implementation");
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

TEST_CASE("compare bfloat gated delta with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, 2e-2, "MPS bfloat gated delta output should match scalar reference after output rounding");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_len, 5e-4, "MPS bfloat gated delta state should match scalar reference implementation");
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

TEST_CASE("compare half gated delta with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy32->data.f32, v_len, 5e-3, "MPS half gated delta output should match scalar reference after output rounding");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, state_len, 5e-4, "MPS half gated delta state should match scalar reference implementation");
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

TEST_CASE("compare aligned gated delta with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS));
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "aligned MPS gated delta output with precomputed decay should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "aligned MPS gated delta state with precomputed decay should match CPU reference implementation");
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

#include "case_main.h"
