#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _gated_delta_int_value(const int i, const int multiplier, const int modulus, const float scale)
{
	return (float)(((i * multiplier) % modulus) - modulus / 2) * scale;
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
		hbeta->data.f32[i] = 0.18f + 0.04f * (float)(i % 6);
	}
	for (i = 0; i < B * Hv * Dv * Dk; i++)
		hstate_in->data.f32[i] = _gated_delta_int_value(i, 23, 41, 0.008f);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(expected_y, expected_state), 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hk, Dk), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const log_decay = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const beta = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv), 0);
	ccv_nnc_tensor_t* const state_in = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const y_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const state_inplace = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_tensor_t* const hy_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, Hv, Dv), 0);
	ccv_nnc_tensor_t* const hstate_inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, Hv, Dv, Dk), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hlog_decay, hbeta, hstate_in), TENSOR_LIST(q, k, v, log_decay, beta, state_in), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_in), TENSOR_LIST(y, state_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, state_out), TENSOR_LIST(hy, hstate_out), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hstate_in), TENSOR_LIST(state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_GATED_DELTA_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, log_decay, beta, state_inplace), TENSOR_LIST(y_inplace, state_inplace), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_inplace, state_inplace), TENSOR_LIST(hy_inplace, hstate_inplace), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * Hv * Dv, 1e-4, "MPS gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_out->data.f32, B * Hv * Dv * Dk, 1e-4, "MPS gated delta state should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy_inplace->data.f32, B * T * Hv * Dv, 1e-4, "in-place MPS gated delta output should match CPU reference implementation");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_state->data.f32, hstate_inplace->data.f32, B * Hv * Dv * Dk, 1e-4, "in-place MPS gated delta state should match CPU reference implementation");
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
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
	ccv_nnc_tensor_free(y_inplace);
	ccv_nnc_tensor_free(state_inplace);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hstate_out);
	ccv_nnc_tensor_free(hy_inplace);
	ccv_nnc_tensor_free(hstate_inplace);
}

#include "case_main.h"
