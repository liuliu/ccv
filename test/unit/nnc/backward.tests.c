#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("convolutional network of 3x5 on 21x31 for error backward propagation")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 3), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 32), 0);
	ccv_nnc_cmd_t forw_cmd = CMD_CONVOLUTION_FORWARD(1, 32, 5, 3, 3);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(forw_cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(32, 5, 3, 3), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(32), 0);
	int i, j, k;
	for (i = 0; i < 3 * 5 * 3 * 32; i++)
		w->data.f32[i] = 2;
	for (i = 0; i < 21 * 31 * 3; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 32; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_cmd_exec(forw_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_t back_cmd = CMD_CONVOLUTION_BACKWARD(1, 32, 5, 3, 3);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(32, 5, 3, 3), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(32), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 32), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 3), 0);
	for (i = 0; i < 21 * 31 * 32; i++)
		g->data.f32[i] = 1;
	ccv_nnc_cmd_exec(back_cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, gw, gbias), 0);
	ccv_dense_matrix_t* dd = ccv_dense_matrix_new(31, 21, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 31; i++)
		for (j = 0; j < 21; j++)
			dd->data.f32[(i * 21 + j) * 3] =
			dd->data.f32[(i * 21 + j) * 3 + 1] =
			dd->data.f32[(i * 21 + j) * 3 + 2] = 32 * 2 * (5 + ccv_min(i - 2, 0) + ccv_min(28 - i, 0)) * (3 + ccv_min(j - 1, 0) + ccv_min(19 - j, 0));
	REQUIRE_MATRIX_EQ(h, dd, "propagated error doesn't match the expected value");
	ccv_matrix_free(dd);
	float* dw = (float*)ccmalloc(sizeof(float) * 5 * 3 * 3 * 32);
	for (k = 0; k < 32; k++)
		for (i = 0; i < 5; i++)
			for (j = 0; j < 3; j++)
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3] =
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3 + 1] =
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3 + 2] = (31 - abs(i - 2)) * (21 - abs(j - 1));
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw, gw->data.f32, 5 * 3 * 3 * 32, 1e-4, "weight gradient doesn't match the expected value");
	ccfree(dw);
	float* dbias = (float*)ccmalloc(sizeof(float) * 32);
	for (i = 0; i < 32; i++)
		dbias[i] = 21 * 31;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias, gbias->data.f32, 32, 1e-4, "bias gradient doesn't match the expected value");
	ccfree(dbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

TEST_CASE("full connect back propagation")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	bias->data.f32[0] = 1;
	bias->data.f32[1] = 4;
	bias->data.f32[2] = 2;
	bias->data.f32[3] = -1;
	a->data.f32[0] = 5;
	a->data.f32[1] = -3;
	a->data.f32[2] = 10;
	a->data.f32[3] = 11;
	a->data.f32[4] = -1;
	float m[] = {
		0.5, 0.2, -0.3, 2, 4,
		1, 8, 2, 8, -1,
		0, 10, -1, -2, 3,
		4, 7, 8, 10, 0
	};
	float ho[] = {
		0.5 + 4 - 4,
		0.2 + 4 * 8 + 2 * 10 - 7,
		-0.3 + 4 * 2 - 2 - 8,
		2 + 4 * 8 - 2 * 2 - 10,
		4 - 4 + 2 * 3,
	};
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(m, ONE_CPU_TENSOR(4, 5), 0);
	ccv_nnc_cmd_t forw_cmd = CMD_GEMM_FORWARD(4);
	ccv_nnc_cmd_exec(forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float bo[] = {
		0.5 * 5 - 0.2 * 3 - 0.3 * 10 + 2 * 11 - 4 + 1,
		1 * 5 - 8 * 3 + 2 * 10 + 8 * 11 + 1 + 4,
		-10 * 3 - 10 - 2 * 11 - 3 + 2,
		4 * 5 - 7 * 3 + 8 * 10 + 10 * 11 - 1
	};
	ccv_nnc_tensor_t bot = ccv_nnc_tensor(bo, ONE_CPU_TENSOR(4), 0);
	REQUIRE_TENSOR_EQ(b, &bot, "forward propagation result should match expected value");
	ccv_nnc_cmd_t back_cmd = CMD_GEMM_BACKWARD(4);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 5), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(5), 0);
	// Pass in bias as gradient
	ccv_nnc_cmd_exec(back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(bias, a, w), TENSOR_LIST(h, gw, gbias), 0);
	// Therefore, gradient bias should match bias.
	REQUIRE_TENSOR_EQ(gbias, bias, "bias gradients should match expected value");
	float go[] = {
		5, -3, 10, 11, -1,
		4 * 5, -4 * 3, 4 * 10, 4 * 11, -4,
		2 * 5, -2 * 3, 2 * 10, 2 * 11, -2,
		-5, 3, -10, -11, 1
	};
	ccv_nnc_tensor_t got = ccv_nnc_tensor(go, ONE_CPU_TENSOR(4, 5), 0);
	REQUIRE_TENSOR_EQ(gw, &got, "weight gradients should match expected value");
	ccv_nnc_tensor_t hot = ccv_nnc_tensor(ho, ONE_CPU_TENSOR(5), 0);
	REQUIRE_TENSOR_EQ(h, &hot, "back propagation error should match expected value");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

TEST_CASE("full connect back propagation with batch = 2")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 4), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	bias->data.f32[0] = 1;
	bias->data.f32[1] = 4;
	bias->data.f32[2] = 2;
	bias->data.f32[3] = -1;
	a->data.f32[0] = 5;
	a->data.f32[1] = -3;
	a->data.f32[2] = 10;
	a->data.f32[3] = 11;
	a->data.f32[4] = -1;
	a->data.f32[0 + 5] = -5;
	a->data.f32[1 + 5] = 3;
	a->data.f32[2 + 5] = -10;
	a->data.f32[3 + 5] = -11;
	a->data.f32[4 + 5] = 1;
	float m[] = {
		0.5, 0.2, -0.3, 2, 4,
		1, 8, 2, 8, -1,
		0, 10, -1, -2, 3,
		4, 7, 8, 10, 0
	};
	float ho[] = {
		-(0.5 + 4 - 4),
		-(0.2 + 4 * 8 + 2 * 10 - 7),
		-(-0.3 + 4 * 2 - 2 - 8),
		-(2 + 4 * 8 - 2 * 2 - 10),
		-(4 - 4 + 2 * 3),
		0.5 + 4 - 4,
		0.2 + 4 * 8 + 2 * 10 - 7,
		-0.3 + 4 * 2 - 2 - 8,
		2 + 4 * 8 - 2 * 2 - 10,
		4 - 4 + 2 * 3,
	};
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(m, ONE_CPU_TENSOR(4, 5), 0);
	ccv_nnc_cmd_t forw_cmd = CMD_GEMM_FORWARD(4);
	ccv_nnc_cmd_exec(forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	float bo[] = {
		0.5 * 5 - 0.2 * 3 - 0.3 * 10 + 2 * 11 - 4 + 1,
		1 * 5 - 8 * 3 + 2 * 10 + 8 * 11 + 1 + 4,
		-10 * 3 - 10 - 2 * 11 - 3 + 2,
		4 * 5 - 7 * 3 + 8 * 10 + 10 * 11 - 1,
		-(0.5 * 5 - 0.2 * 3 - 0.3 * 10 + 2 * 11 - 4) + 1,
		-(1 * 5 - 8 * 3 + 2 * 10 + 8 * 11 + 1) + 4,
		-(-10 * 3 - 10 - 2 * 11 - 3) + 2,
		-(4 * 5 - 7 * 3 + 8 * 10 + 10 * 11) - 1
	};
	ccv_nnc_tensor_t bot = ccv_nnc_tensor(bo, ONE_CPU_TENSOR(2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bot, "forward propagation result should match expected value");
	ccv_nnc_cmd_t back_cmd = CMD_GEMM_BACKWARD(4);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 5), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 5), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 4), 0);
	int i;
	for (i = 0; i < 4; i++)
	{
		g->data.f32[i] = -bias->data.f32[i];
		g->data.f32[i + 4] = bias->data.f32[i];
	}
	// Pass in bias as gradient
	ccv_nnc_cmd_exec(back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, gw, gbias), 0);
	// Therefore, gradient bias should match bias.
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	REQUIRE_TENSOR_EQ(gbias, bias, "bias gradients should match expected value");
	float go[] = {
		5, -3, 10, 11, -1,
		4 * 5, -4 * 3, 4 * 10, 4 * 11, -4,
		2 * 5, -2 * 3, 2 * 10, 2 * 11, -2,
		-5, 3, -10, -11, 1
	};
	for (i = 0; i < 5 * 4; i++)
		go[i] = -go[i] * 2; // Because the gradient is negative in the first example, and the input is negative in the second example, we basically doubled the weight gradients.
	ccv_nnc_tensor_t got = ccv_nnc_tensor(go, ONE_CPU_TENSOR(4, 5), 0);
	REQUIRE_TENSOR_EQ(gw, &got, "weight gradients should match expected value");
	ccv_nnc_tensor_t hot = ccv_nnc_tensor(ho, ONE_CPU_TENSOR(2, 5), 0);
	REQUIRE_TENSOR_EQ(h, &hot, "back propagation error should match expected value");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

#include "case_main.h"
