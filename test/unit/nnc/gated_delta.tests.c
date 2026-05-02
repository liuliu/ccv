#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <string.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _gated_delta_reference(const float* const q, const float* const k, const float* const v, const float* const log_decay, const float* const beta, const float* const state_in, const int B, const int T, const int Hk, const int Dk, const int Hv, const int Dv, float* const y, float* const state_out)
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
					const float decay = expf(log_decay[gate_offset]);
					float memory = 0;
					for (dk = 0; dk < Dk; dk++)
					{
						state_out[state_offset + dk] *= decay;
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

static float _gated_delta_value(const int i, const int multiplier, const int modulus, const float scale)
{
	return (float)(((i * multiplier) % modulus) - modulus / 2) * scale;
}

TEST_CASE("gated delta forward with grouped heads")
{
	const int B = 2;
	const int T = 4;
	const int Hk = 2;
	const int Dk = 5;
	const int Hv = 4;
	const int Dv = 3;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const inplace_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const inplace_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		q->data.f32[i] = _gated_delta_value(i, 13, 31, 0.03f);
		k->data.f32[i] = _gated_delta_value(i, 17, 29, 0.025f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		v->data.f32[i] = _gated_delta_value(i, 19, 37, 0.02f);
	for (i = 0; i < B * T * Hv; i++)
	{
		log_decay->data.f32[i] = -0.01f * (float)((i % 7) + 1);
		beta->data.f32[i] = 0.15f + 0.07f * (float)(i % 5);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		state_in->data.f32[i] = _gated_delta_value(i, 23, 41, 0.01f);
	_gated_delta_reference(q->data.f32, k->data.f32, v->data.f32, log_decay->data.f32, beta->data.f32, state_in->data.f32, B, T, Hk, Dk, Hv, Dv, expected_y->data.f32, expected_state->data.f32);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	memcpy(inplace_state->data.f32, state_in->data.f32, sizeof(float) * B * Hv * Dv * Dk);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, inplace_state), TENSOR_LIST(inplace_y, inplace_state), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, y->data.f32, B * T * Hv * Dv, 1e-6, "gated delta output should match scalar reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, state_out->data.f32, B * Hv * Dv * Dk, 1e-6, "gated delta state should match scalar reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, inplace_y->data.f32, B * T * Hv * Dv, 1e-6, "in-place gated delta output should match scalar reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, inplace_state->data.f32, B * Hv * Dv * Dk, 1e-6, "in-place gated delta state should match scalar reference");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(inplace_y);
	ccv_nnc_tensor_free(inplace_state);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
}

TEST_CASE("gated delta prefill matches repeated decode")
{
	const int B = 1;
	const int T = 5;
	const int Hk = 1;
	const int Dk = 4;
	const int Hv = 3;
	const int Dv = 2;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		q->data.f32[i] = _gated_delta_value(i, 7, 23, 0.035f);
		k->data.f32[i] = _gated_delta_value(i, 11, 19, 0.03f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		v->data.f32[i] = _gated_delta_value(i, 13, 29, 0.02f);
	for (i = 0; i < B * T * Hv; i++)
	{
		log_decay->data.f32[i] = -0.02f * (float)((i % 5) + 1);
		beta->data.f32[i] = 0.2f + 0.05f * (float)(i % 4);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		state_in->data.f32[i] = _gated_delta_value(i, 17, 31, 0.01f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_tensor_t* const q1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hv), 0);
	ccv_nnc_tensor_t* const beta1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hv), 0);
	ccv_nnc_tensor_t* const y1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, Hv, Dv), 0);
	ccv_nnc_tensor_t* const decode_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* state_cur = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* state_next = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	memcpy(state_cur->data.f32, state_in->data.f32, sizeof(float) * B * Hv * Dv * Dk);
	int t;
	for (t = 0; t < T; t++)
	{
		memcpy(q1->data.f32, q->data.f32 + t * Hk * Dk, sizeof(float) * Hk * Dk);
		memcpy(k1->data.f32, k->data.f32 + t * Hk * Dk, sizeof(float) * Hk * Dk);
		memcpy(v1->data.f32, v->data.f32 + t * Hv * Dv, sizeof(float) * Hv * Dv);
		memcpy(log_decay1->data.f32, log_decay->data.f32 + t * Hv, sizeof(float) * Hv);
		memcpy(beta1->data.f32, beta->data.f32 + t * Hv, sizeof(float) * Hv);
		ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q1, k1, v1, log_decay1, beta1, state_cur), TENSOR_LIST(y1, state_next), 0);
		memcpy(decode_y->data.f32 + t * Hv * Dv, y1->data.f32, sizeof(float) * Hv * Dv);
		ccv_nnc_tensor_t* const tmp = state_cur;
		state_cur = state_next;
		state_next = tmp;
	}
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y->data.f32, decode_y->data.f32, B * T * Hv * Dv, 1e-6, "prefill output should match repeated decode output");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, state_out->data.f32, state_cur->data.f32, B * Hv * Dv * Dk, 1e-6, "prefill state should match repeated decode state");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(q1);
	ccv_nnc_tensor_free(k1);
	ccv_nnc_tensor_free(v1);
	ccv_nnc_tensor_free(log_decay1);
	ccv_nnc_tensor_free(beta1);
	ccv_nnc_tensor_free(y1);
	ccv_nnc_tensor_free(decode_y);
	ccv_nnc_tensor_free(state_cur);
	ccv_nnc_tensor_free(state_next);
}

TEST_CASE("ccv_cnnp gated delta forward")
{
	const int B = 1;
	const int T = 3;
	const int Hk = 1;
	const int Dk = 4;
	const int Hv = 2;
	const int Dv = 3;
	const ccv_cnnp_model_io_t q_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t k_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t v_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t log_decay_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t beta_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t state_input = ccv_cnnp_input();
	ccv_cnnp_model_io_t outputs = ccv_cnnp_model_apply(ccv_cnnp_gated_delta("gated_delta"), MODEL_IO_LIST(q_input, k_input, v_input, log_decay_input, beta_input, state_input));
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(q_input, k_input, v_input, log_decay_input, beta_input, state_input), MODEL_IO_LIST(outputs), 0, "gated_delta");
	ccv_nnc_tensor_param_t q_params = CPU_TENSOR_NHWC(32F, B, T, Hk, Dk);
	ccv_nnc_tensor_param_t k_params = CPU_TENSOR_NHWC(32F, B, T, Hk, Dk);
	ccv_nnc_tensor_param_t v_params = CPU_TENSOR_NHWC(32F, B, T, Hv, Dv);
	ccv_nnc_tensor_param_t log_decay_params = CPU_TENSOR_NHWC(32F, B, T, Hv);
	ccv_nnc_tensor_param_t beta_params = CPU_TENSOR_NHWC(32F, B, T, Hv);
	ccv_nnc_tensor_param_t state_params = CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(q_params, k_params, v_params, log_decay_params, beta_params, state_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, k_params, 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, v_params, 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, log_decay_params, 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, beta_params, 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, state_params, 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, v_params, 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, state_params, 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, v_params, 0);
	ccv_nnc_tensor_t* const expected_state = ccv_nnc_tensor_new(0, state_params, 0);
	int i;
	for (i = 0; i < B * T * Hk * Dk; i++)
	{
		q->data.f32[i] = _gated_delta_value(i, 5, 17, 0.04f);
		k->data.f32[i] = _gated_delta_value(i, 7, 19, 0.03f);
	}
	for (i = 0; i < B * T * Hv * Dv; i++)
		v->data.f32[i] = _gated_delta_value(i, 11, 23, 0.025f);
	for (i = 0; i < B * T * Hv; i++)
	{
		log_decay->data.f32[i] = -0.03f * (float)((i % 4) + 1);
		beta->data.f32[i] = 0.12f + 0.06f * (float)(i % 3);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		state_in->data.f32[i] = _gated_delta_value(i, 13, 29, 0.01f);
	_gated_delta_reference(q->data.f32, k->data.f32, v->data.f32, log_decay->data.f32, beta->data.f32, state_in->data.f32, B, T, Hk, Dk, Hv, Dv, expected_y->data.f32, expected_state->data.f32);
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){.requires_grad = 0}, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, y->data.f32, B * T * Hv * Dv, 1e-6, "ccv_cnnp gated delta output should match scalar reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, state_out->data.f32, B * Hv * Dv * Dk, 1e-6, "ccv_cnnp gated delta state should match scalar reference");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(log_decay);
	ccv_nnc_tensor_free(beta);
	ccv_nnc_tensor_free(state_in);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(state_out);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_state);
	ccv_cnnp_model_free(model);
}

#include "case_main.h"
