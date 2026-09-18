#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("signed sqrt has signed floors and positive gradients on both sides")
{
	const float values[] = {0, -0.0f, 1e-8f, -1e-8f, 1e-6f, -1e-6f, 1, -1, 4, -4, INFINITY, -INFINITY, NAN, -NAN};
	const float expected[] = {0.001f, -0.001f, 0.001f, -0.001f, 0.001f, -0.001f, 1, -1, 2, -2, INFINITY, -INFINITY, 0.001f, -0.001f};
	const float slopes[] = {0, 0, 0, 0, 500, 500, 0.5f, 0.5f, 0.25f, 0.25f, 0, 0, 0, 0};
	const int count = sizeof(values) / sizeof(values[0]);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, count), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, a->info, 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, a->info, 0);
	memcpy(a->data.f32, values, sizeof(values));
	int i;
	for (i = 0; i < count; i++)
		g->data.f32[i] = (i & 1) ? -2 : 2;
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), CCV_NNC_EXEC_SUCCESS, "forward should run");
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, 0), TENSOR_LIST(g), 0), CCV_NNC_EXEC_SUCCESS, "backward should replace its gradient input");
	for (i = 0; i < count; i++)
	{
		REQUIRE_EQ(!!signbit(b->data.f32[i]), !!signbit(expected[i]), "forward preserves sign, including zero and NaN");
		if (isinf(expected[i]))
		{
			REQUIRE(isinf(b->data.f32[i]), "infinities remain infinite");
		} else {
			REQUIRE_EQ_WITH_TOLERANCE(b->data.f32[i], expected[i], 1e-7, "forward should implement the clamped signed root");
		}
		REQUIRE_EQ_WITH_TOLERANCE(g->data.f32[i], slopes[i] * ((i & 1) ? -2 : 2), 1e-4, "both signs use the positive slope, including clamp boundaries");
	}
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a), 0), CCV_NNC_EXEC_SUCCESS, "forward should run in place");
	for (i = 0; i < count; i++)
		REQUIRE_EQ(a->data.f32[i], b->data.f32[i], "in-place forward should match");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
}

TEST_CASE("signed sqrt validates floors and handles empty tensors")
{
	ccv_nnc_tensor_t a = ccv_nnc_tensor(0, CPU_TENSOR_NHWC(32F, 0), 0);
	ccv_nnc_tensor_t b = ccv_nnc_tensor(0, CPU_TENSOR_NHWC(32F, 0), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(&a), TENSOR_LIST(&b), 0), CCV_NNC_EXEC_SUCCESS, "empty forward is a no-op");
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(&a, &a, 0), TENSOR_LIST(&b), 0), CCV_NNC_EXEC_SUCCESS, "empty backward is a no-op");
	const float invalid[] = {0, -1, INFINITY, NAN};
	int i;
	for (i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
	{
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_FORWARD(invalid[i]), ccv_nnc_no_hint, 0, TENSOR_LIST(&a), TENSOR_LIST(&b), 0), CCV_NNC_EXEC_INVALID, "floor must be finite and positive");
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SIGNED_SQRT_BACKWARD(invalid[i]), ccv_nnc_no_hint, 0, TENSOR_LIST(&a, &a, 0), TENSOR_LIST(&b), 0), CCV_NNC_EXEC_INVALID, "backward also validates the floor");
	}
}

TEST_CASE("signed sqrt model copies its floor and retains input for backward")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_signed_sqrt(0.25f, "signed_root"), MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const original = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 1, 0);
	ccv_cnnp_model_t* const model = ccv_cnnp_model_copy(original, 1);
	ccv_cnnp_model_free(original);
	const ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, 8);
	ccv_cnnp_model_compile(model, &params, 1, CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, params, 0);
	const float values[] = {-4, -0.25f, -0.125f, -0.0f, 0, 0.125f, 0.25f, 4};
	const float expected[] = {-2, -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 2};
	const float gradients[] = {0.5f, 2, 0, 0, 0, 0, 2, 0.5f};
	memcpy(a->data.f32, values, sizeof(values));
	int i;
	for (i = 0; i < 8; i++)
		g->data.f32[i] = 2;
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){.requires_grad = 1}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	ccv_cnnp_model_backward(model, TENSOR_LIST(g), TENSOR_LIST(h), 0, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, expected, 8, 1e-6, "copied model must keep its non-default floor");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, gradients, 8, 1e-6, "backward must distinguish clamped inputs from boundary inputs");
	ccv_cnnp_model_free(model);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
}

#include "case_main.h"
