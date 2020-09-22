#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("dynamic graph to compute log(19)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, a)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(b), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, b)->data.f32[0], logf(19), 1e-5, "log(19) result should be equal.");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph with alias")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2));
	ccv_nnc_tensor_variable_t const a0 = ccv_nnc_tensor_variable_alias_new(graph, a, DIM_ALLOC(1), DIM_ALLOC(), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t const a1 = ccv_nnc_tensor_variable_alias_new(graph, a, ccv_nnc_no_ofs, DIM_ALLOC(), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_t* const b0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_variable_set(graph, a, b0);
	b0->data.f32[0] = 10;
	b0->data.f32[1] = 11;
	ccv_nnc_tensor_t* const b1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	b1->data.f32[0] = 20;
	b1->data.f32[1] = 21;
	REQUIRE_EQ(ccv_nnc_tensor_from_variable(graph, a0)->data.f32[0], 11, "should be b0[1]");
	REQUIRE_EQ(ccv_nnc_tensor_from_variable(graph, a1)->data.f32[0], 10, "should be b0[0]");
	REQUIRE(!CCV_IS_TENSOR_VIEW(ccv_nnc_tensor_from_variable(graph, a1)), "no need to be a tensor view");
	ccv_nnc_tensor_variable_set(graph, a, b1);
	REQUIRE_EQ(ccv_nnc_tensor_from_variable(graph, a0)->data.f32[0], 21, "should be b1[1]");
	REQUIRE_EQ(ccv_nnc_tensor_from_variable(graph, a1)->data.f32[0], 20, "should be b1[0]");
	REQUIRE(!CCV_IS_TENSOR_VIEW(ccv_nnc_tensor_from_variable(graph, a1)), "no need to be a tensor view");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(b0);
	ccv_nnc_tensor_free(b1);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + 1.2 * x, f'(x) where x = 19")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, z), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	// Do gradient computation multiple times.
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	ccv_nnc_tensor_variable_t dy = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(dy), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	ccv_nnc_tensor_variable_t dz = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(dz), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dz)->data.f32[0], 1, 1e-5, "f'(z) = 1");
	ccv_nnc_tensor_variable_free(graph, dy);
	dy = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(dx), 0, 0);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(y, x), TENSOR_VARIABLE_LIST(dy, dx), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph with dense net (extensive use of alias)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1, 4));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 0.472;
	ccv_nnc_tensor_variable_t x1 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 1));
	ccv_nnc_tensor_variable_t w1 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1, 1));
	ccv_nnc_tensor_from_variable(graph, w1)->data.f32[0] = 0.234;
	ccv_nnc_tensor_variable_t b1 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, b1)->data.f32[0] = 0.1;
	ccv_nnc_tensor_variable_t x11 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 1), DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x1, w1, b1), TENSOR_VARIABLE_LIST(x11), 0, 0);
	ccv_nnc_tensor_variable_t x2 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 2));
	ccv_nnc_tensor_variable_t w2 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1, 2));
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[0] = 0.374;
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[1] = 0.886;
	ccv_nnc_tensor_variable_t b2 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, b2)->data.f32[0] = 0.2;
	ccv_nnc_tensor_variable_t x21 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 2), DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x2, w2, b2), TENSOR_VARIABLE_LIST(x21), 0, 0);
	ccv_nnc_tensor_variable_t x3 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 3));
	ccv_nnc_tensor_variable_t w3 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1, 3));
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[0] = 0.484;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[1] = 0.912;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[2] = 0.235;
	ccv_nnc_tensor_variable_t b3 = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, b3)->data.f32[0] = 0.3;
	ccv_nnc_tensor_variable_t x31 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 3), DIM_ALLOC(1, 4), CPU_TENSOR_NHWC(32F, 1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x3, w3, b3), TENSOR_VARIABLE_LIST(x31), 0, 0);
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 4), 0);
	xt->data.f32[0] = 0.472;
	xt->data.f32[1] = xt->data.f32[0] * 0.234 + 0.1;
	xt->data.f32[2] = xt->data.f32[0] * 0.374 + xt->data.f32[1] * 0.886 + 0.2;
	xt->data.f32[3] = xt->data.f32[0] * 0.484 + xt->data.f32[1] * 0.912 + xt->data.f32[2] * 0.235 + 0.3;
	REQUIRE_MATRIX_EQ(ccv_nnc_tensor_from_variable(graph, x), xt, "1x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_variable_t dw1 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(x), 0, TENSOR_VARIABLE_LIST(w1), TENSOR_VARIABLE_LIST(dw1), 0);
	REQUIRE_EQ_WITH_TOLERANCE((0.235 * 0.886 + 0.912) * 0.472, ccv_nnc_tensor_from_variable(graph, dw1)->data.f32[0], 1e-5, "the gradient should be equal to a complicated result");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("batch norm in dynamic graph (enforce inplace)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t scale = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 10));
	ccv_nnc_tensor_variable_t bias = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 10));
	ccv_nnc_tensor_variable_t mean = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 10));
	ccv_nnc_tensor_variable_t var = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 10));
	ccv_nnc_tensor_variable_t saved_mean = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t saved_inv_std = ccv_nnc_tensor_variable_new(graph);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		ccv_nnc_tensor_from_variable(graph, x)->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		ccv_nnc_tensor_from_variable(graph, scale)->data.f32[i] = 1;
	for (i = 0; i < 10; i++)
		ccv_nnc_tensor_from_variable(graph, bias)->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		ccv_nnc_tensor_from_variable(graph, mean)->data.f32[i] = 0;
	ccv_nnc_tensor_t* mean_tensor_ptr = ccv_nnc_tensor_from_variable(graph, mean);
	for (i = 0; i < 10; i++)
		ccv_nnc_tensor_from_variable(graph, var)->data.f32[i] = 0;
	ccv_nnc_tensor_t* var_tensor_ptr = ccv_nnc_tensor_from_variable(graph, var);
	ccv_nnc_dynamic_graph_exec(graph, CMD_BATCH_NORM_FORWARD(0, 0, 0.9, 0, 1, 2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, scale, bias, mean, var), TENSOR_VARIABLE_LIST(y, mean, var, saved_mean, saved_inv_std), 0, 0);
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE(mean_tensor_ptr == ccv_nnc_tensor_from_variable(graph, mean), "enforced inplace, tensor view pointer unchanged");
	REQUIRE(var_tensor_ptr == ccv_nnc_tensor_from_variable(graph, var), "enforced inplace, tensor view pointer unchanged");
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), 0);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), 0);
	ccv_nnc_tensor_t* scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* bias_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* var_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* saved_mean_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, 10), 0);
	ccv_nnc_tensor_t* saved_inv_std_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, 10), 0);
	memcpy(x_tensor->data.f32, ccv_nnc_tensor_from_variable(graph, x)->data.f32, sizeof(float) * 2 * 2 * 2 * 10);
	for (i = 0; i < 10; i++)
		scale_tensor->data.f32[i] = 1;
	memset(bias_tensor->data.f32, 0, sizeof(float) * 10);
	memset(mean_tensor->data.f32, 0, sizeof(float) * 10);
	memset(var_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_nnc_cmd_exec(CMD_BATCH_NORM_FORWARD(0, 0, 0.9, 0, 1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, scale_tensor, bias_tensor, mean_tensor, var_tensor), TENSOR_LIST(y_tensor, mean_tensor, var_tensor, saved_mean_tensor, saved_inv_std_tensor), 0);
	REQUIRE_TENSOR_EQ(y_tensor, ccv_nnc_tensor_from_variable(graph, y), "y should be equal");
	REQUIRE_TENSOR_EQ(mean_tensor, ccv_nnc_tensor_from_variable(graph, mean), "mean should be equal");
	REQUIRE_TENSOR_EQ(var_tensor, ccv_nnc_tensor_from_variable(graph, var), "var should be equal");
	REQUIRE_TENSOR_EQ(saved_mean_tensor, ccv_nnc_tensor_from_variable(graph, saved_mean), "saved_mean should be equal");
	REQUIRE_TENSOR_EQ(saved_inv_std_tensor, ccv_nnc_tensor_from_variable(graph, saved_inv_std), "saved_inv_std should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(scale_tensor);
	ccv_nnc_tensor_free(bias_tensor);
	ccv_nnc_tensor_free(mean_tensor);
	ccv_nnc_tensor_free(var_tensor);
	ccv_nnc_tensor_free(saved_mean_tensor);
	ccv_nnc_tensor_free(saved_inv_std_tensor);
}

TEST_CASE("empty inputs / outputs for dynamic graph")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t df = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_from_variable(graph, df)->data.f32[0] = 1;
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(df, 0, x), TENSOR_VARIABLE_LIST(y, 0), 0, 0);
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y)->data.f32[0], 1. / 10, 1e-5, "div backward should equal to 1 / 10");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("long dynamic graph with unused variables freed")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	int i;
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 32;
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 0.5;
	for (i = 0; i < 10; i++)
	{
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		if (i < 7)
			ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
		else {
			if (i == 7)
				ccv_nnc_tensor_variable_free(graph, y); // No longer need y.
			ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, x), TENSOR_VARIABLE_LIST(z), 0, 0);
		}
		if (i < 9)
			ccv_nnc_tensor_variable_free(graph, x);
		x = z;
	}
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	float g = 32;
	for (i = 0; i < 10; i++)
	{
		if (i < 7)
			g = g * 0.5;
		else
			g = g * g;
	}
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, x)->data.f32[0], g, 1e-5, "x should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("repeat multiple x * y with y as a constant")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	int i;
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_constant_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 32;
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 0.5;
	for (i = 0; i < 10; i++)
	{
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
		ccv_nnc_tensor_variable_free(graph, x);
		x = z;
	}
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	float g = 32;
	for (i = 0; i < 10; i++)
		g = g * 0.5;
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, x)->data.f32[0], g, 1e-5, "x should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
}

static int _ccv_tensor_variable_freed = 0;

static void _ccv_tensor_variable_hook(ccv_nnc_dynamic_graph_t* const graph, const ccv_nnc_tensor_t* const tensor, const ccv_nnc_dynamic_graph_t* const owner, void* const context)
{
	if (!owner)
		++_ccv_tensor_variable_freed;
}

TEST_CASE("repeat multiple x * y with y as a constant, compute d(x)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	int i;
	_ccv_tensor_variable_freed = 0;
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_constant_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t ox = x;
	ccv_nnc_tensor_variable_owner_hook(graph, x, _ccv_tensor_variable_hook, 0);
	ccv_nnc_tensor_variable_owner_hook(graph, y, _ccv_tensor_variable_hook, 0);
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 32;
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 0.9;
	for (i = 0; i < 4; i++)
	{
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_owner_hook(graph, z, _ccv_tensor_variable_hook, 0);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
		if (i > 0)
			ccv_nnc_tensor_variable_free(graph, x);
		x = z;
	}
	REQUIRE_EQ(0, _ccv_tensor_variable_freed, "none of these are freed");
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_owner_hook(graph, dx, _ccv_tensor_variable_hook, 0);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(x), 0, TENSOR_VARIABLE_LIST(ox), TENSOR_VARIABLE_LIST(dx), 0);
	ccv_nnc_tensor_variable_free(graph, ox);
	// We freed all 4 x (ox, when i = 1, 2, 3).
	REQUIRE_EQ(4, _ccv_tensor_variable_freed, "all are freed except dx");
	_ccv_tensor_variable_freed = 0;
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	float g = 1;
	for (i = 0; i < 4; i++)
		g = g * 0.9;
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], g, 1e-5, "x should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
	// y, dx, and z.
	REQUIRE_EQ(3, _ccv_tensor_variable_freed, "dx freed");
}

TEST_CASE("compute f(x) = x * log(x) + x, f'(x) when x = 10 (and intermediate results all freed)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_variable_t df = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(df), 0, 0);
	// x will be accumulated on to itself.
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), &df, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(x), 0);
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, x)->data.f32[0], 10 + log(10) + 1 + 1, 1e-5, "dx should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph with binded value")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	x_tensor->data.f32[0] = 10;
	ccv_nnc_tensor_variable_set(graph, x, x_tensor);
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y)->data.f32[0], log(10), 1e-5, "dx should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
}

TEST_CASE("dynamic graph to evaluate cnnp model")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, (ccv_cnnp_param_t){}, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_constant_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, z)->data.f32[0] = 5;
	int i;
	for (i = 0; i < 100; i++)
	{
		ccv_nnc_tensor_variable_t x;
		if (i % 2 == 1)
		{
			x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2, 1));
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[1] = 10;
		} else {
			x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
		}
		ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_dynamic_graph_exec(graph, CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, z), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, y), TENSOR_VARIABLE_LIST(f), 0, 0);
		ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx), 0);
		ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(), 0, 0, 0);
		ccv_nnc_tensor_variable_free(graph, x);
		ccv_nnc_tensor_variable_free(graph, y);
		ccv_nnc_tensor_variable_free(graph, f);
		ccv_nnc_tensor_variable_free(graph, dx);
	}
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_dynamic_graph_evaluate(graph, linear, 1, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y)->data.f32[0], 5, 1e-2, "linear model should be trained to generate the same value as z");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
	ccv_cnnp_model_free(linear);
}

TEST_CASE("dynamic graph to evaluate cnnp model without any parameters")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, a)->data.f32[0] = 1.23;
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, b)->data.f32[0] = 2;
	ccv_nnc_tensor_variable_t c = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_cnnp_model_t* const mul = ccv_cnnp_mul("mul");
	ccv_nnc_dynamic_graph_evaluate(graph, mul, 1, TENSOR_VARIABLE_LIST(a, b), TENSOR_VARIABLE_LIST(c), 0, 0);
	ccv_nnc_tensor_variable_t da = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(c), 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(da), 0);
	ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(), 0, 0, 0);
	ccv_cnnp_model_free(mul);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, c)->data.f32[0], 2.46, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, da)->data.f32[0], 2, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to evaluate cnnp model and simply accumulate gradients")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, (ccv_cnnp_param_t){}, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_constant_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, z)->data.f32[0] = 5;
	int i;
	for (i = 0; i < 100; i++)
	{
		ccv_nnc_tensor_variable_t x;
		if (i % 2 == 1)
		{
			x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2, 1));
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[1] = 10;
		} else {
			x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
			ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
		}
		ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_dynamic_graph_exec(graph, CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, z), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, y), TENSOR_VARIABLE_LIST(f), 0, 0);
		ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx), 0);
		ccv_nnc_tensor_variable_free(graph, x);
		ccv_nnc_tensor_variable_free(graph, y);
		ccv_nnc_tensor_variable_free(graph, f);
		ccv_nnc_tensor_variable_free(graph, dx);
		if ((i % 2) == 1)
			ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(), 0, 0, 0);
	}
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_dynamic_graph_evaluate(graph, linear, 1, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y)->data.f32[0], 5, 1e-2, "linear model should be trained to generate the same value as z");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
	ccv_cnnp_model_free(linear);
}

TEST_CASE("dynamic graph to accumulate gradients cross cnnp models")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, (ccv_cnnp_param_t){
		.no_bias = 1,
	}, 0);
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(0.2485), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(a), 0, 0);
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(linear, &input, 1, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_set_parameter(linear, ccv_cnnp_model_parameters(linear, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0), ccv_nnc_tensor_from_variable(graph, a));
	ccv_nnc_tensor_variable_t a_grad = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t saved_aux = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	int i;
	for (i = 0; i < 1000; i++)
	{
		ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = i;
		ccv_nnc_tensor_variable_t t = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_from_variable(graph, t)->data.f32[0] = -i;
		ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, a), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(z), 0, 0);
		ccv_nnc_tensor_variable_t n = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z, t), TENSOR_VARIABLE_LIST(n), 0, 0);
		ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(n, n), TENSOR_VARIABLE_LIST(f), 0, 0);
		ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(a_grad), 0);
		ccv_nnc_tensor_variable_free(graph, n);
		ccv_nnc_tensor_variable_free(graph, z);
		ccv_nnc_tensor_variable_free(graph, y);
		ccv_nnc_tensor_variable_free(graph, x);
		ccv_nnc_tensor_variable_free(graph, t);
		ccv_nnc_tensor_variable_free(graph, f);
		if (((i + 1) % 5) == 0)
		{
			DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
			REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_GRAPH_EXEC), 0, "everything should be freed");
			float lr = 0.001;
			if (i >= 200)
				lr = 0.0001;
			else if (i >= 600)
				lr = 0.00001;
			ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_SGD_FORWARD(0, lr, 0.001, 0, 0, 0), TENSOR_VARIABLE_LIST(a_grad), TENSOR_VARIABLE_LIST(a), &saved_aux, 0, 0);
		}
	}
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 5;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, a), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(z), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, z)->data.f32[0], -5, 1e-2, "linear model should be trained to generate the same value as z");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_cnnp_model_free(linear);
}

TEST_CASE("dynamic graph to accumulate gradients cross cnnp models with aliases")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_cnnp_model_t* const linear = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_dense(1, (ccv_cnnp_param_t){
			.no_bias = 1,
		}, 0),
		ccv_cnnp_reshape(DIM_ALLOC(1), DIM_ALLOC(), DIM_ALLOC(), 0),
	), 0);
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(0.2485), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(a), 0, 0);
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(linear, &input, 1, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_set_parameter(linear, ccv_cnnp_model_parameters(linear, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0), ccv_nnc_tensor_from_variable(graph, a));
	ccv_nnc_tensor_variable_t a_grad = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t saved_aux = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	int i;
	for (i = 0; i < 1000; i++)
	{
		ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = i;
		ccv_nnc_tensor_variable_t t = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_from_variable(graph, t)->data.f32[0] = -i;
		ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, a), TENSOR_VARIABLE_LIST(y), 0, 0);
		ccv_nnc_tensor_variable_t y_alias = ccv_nnc_tensor_variable_alias_new(graph, y, DIM_ALLOC(), DIM_ALLOC(1), CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y_alias), TENSOR_VARIABLE_LIST(z), 0, 0);
		ccv_nnc_tensor_variable_t n = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z, t), TENSOR_VARIABLE_LIST(n), 0, 0);
		ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(n, n), TENSOR_VARIABLE_LIST(f), 0, 0);
		ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(a_grad), 0);
		ccv_nnc_tensor_variable_free(graph, n);
		ccv_nnc_tensor_variable_free(graph, z);
		ccv_nnc_tensor_variable_free(graph, y);
		ccv_nnc_tensor_variable_free(graph, y_alias);
		ccv_nnc_tensor_variable_free(graph, x);
		ccv_nnc_tensor_variable_free(graph, t);
		ccv_nnc_tensor_variable_free(graph, f);
		if (((i + 1) % 5) == 0)
		{
			REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_GRAPH_EXEC), 0, "everything should be freed");
			DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
			float lr = 0.001;
			if (i >= 200)
				lr = 0.0001;
			else if (i >= 600)
				lr = 0.00001;
			ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_SGD_FORWARD(0, lr, 0.001, 0, 0, 0), TENSOR_VARIABLE_LIST(a_grad), TENSOR_VARIABLE_LIST(a), &saved_aux, 0, 0);
		}
	}
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 5;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, a), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_evaluate(graph, linear, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(z), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, z)->data.f32[0], -5, 1e-2, "linear model should be trained to generate the same value as z");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_cnnp_model_free(linear);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + 1.2 * x, f'(x) and sum on x = 19, 10")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, z), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	// Do gradient computation multiple times.
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	ccv_nnc_tensor_variable_free(graph, x);
	x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, z), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 10 * logf(10) + 1.2 * 10, 1e-5, "f(x) = 1.2 * 10 + 10 * log(10)");
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f), 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx), 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2 + logf(10) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19 + 1.2 + log(19) + 19 * 1 / 19");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + y' where y = 1.2 * x and y' is an alias to y")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, z)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y_hat = ccv_nnc_tensor_variable_alias_new(graph, y, ccv_nnc_no_ofs, DIM_ALLOC(), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, z), TENSOR_VARIABLE_LIST(y), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, y_hat), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	ccv_nnc_tensor_variable_free(graph, y_hat);
	ccv_nnc_tensor_variable_free(graph, f);
	ccv_nnc_tensor_variable_free(graph, x);
	ccv_nnc_tensor_variable_free(graph, z);
	ccv_nnc_tensor_variable_free(graph, y);
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_TENSOR), 0, "should have no tensor left");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + y' where y'' = 1.2 * x and both y'' and y' are aliases to y")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f), 0, 0);
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, z)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y_hat = ccv_nnc_tensor_variable_alias_new(graph, y, ccv_nnc_no_ofs, DIM_ALLOC(), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y_double_hat = ccv_nnc_tensor_variable_alias_new(graph, y, ccv_nnc_no_ofs, DIM_ALLOC(), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, z), TENSOR_VARIABLE_LIST(y_double_hat), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, y_hat), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	ccv_nnc_tensor_variable_free(graph, y_hat);
	ccv_nnc_tensor_variable_free(graph, y_double_hat);
	ccv_nnc_tensor_variable_free(graph, f);
	ccv_nnc_tensor_variable_free(graph, x);
	ccv_nnc_tensor_variable_free(graph, z);
	ccv_nnc_tensor_variable_free(graph, y);
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_TENSOR), 0, "should have no tensor left");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * y, where y[0] = 10 * z, y[1] = 2 * z, z = [2], x = [10], should free all variables")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2));
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(z), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x), 0, 0);
	ccv_nnc_tensor_variable_t y_1 = ccv_nnc_tensor_variable_alias_new(graph, y, ccv_nnc_no_ofs, DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y_2 = ccv_nnc_tensor_variable_alias_new(graph, y, DIM_ALLOC(1), DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y_1), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y_2), 0, 0);
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(f), 0, 0);
	float gt[] = {10 * 2 * 10, 2 * 2 * 10};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ccv_nnc_tensor_from_variable(graph, f)->data.f32, gt, 2, 1e-5, "should be equal");
	ccv_nnc_tensor_variable_free(graph, z);
	ccv_nnc_tensor_variable_free(graph, y_1);
	ccv_nnc_tensor_variable_free(graph, y_2);
	ccv_nnc_tensor_variable_free(graph, x);
	ccv_nnc_tensor_variable_free(graph, y);
	ccv_nnc_tensor_variable_free(graph, f);
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_GRAPH_EXEC), 0, "everything should be freed");
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_TENSOR), 0, "everything should be freed");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * y, where y[0] = 10 * z, y[1] = 2 * z, z = [2], x = [10], freed y[0], y[1] before use")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 2));
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(z), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x), 0, 0);
	ccv_nnc_tensor_variable_t y_1 = ccv_nnc_tensor_variable_alias_new(graph, y, ccv_nnc_no_ofs, DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t y_2 = ccv_nnc_tensor_variable_alias_new(graph, y, DIM_ALLOC(1), DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y_1), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y_2), 0, 0);
	ccv_nnc_tensor_variable_free(graph, y_1);
	ccv_nnc_tensor_variable_free(graph, y_2);
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(f), 0, 0);
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_GRAPH_EXEC), 3, "should keep 3 ops");
	float gt[] = {10 * 2 * 10, 2 * 2 * 10};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ccv_nnc_tensor_from_variable(graph, f)->data.f32, gt, 2, 1e-5, "should be equal");
	ccv_nnc_tensor_variable_free(graph, z);
	ccv_nnc_tensor_variable_free(graph, x);
	ccv_nnc_tensor_variable_free(graph, y);
	ccv_nnc_tensor_variable_free(graph, f);
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_GRAPH_EXEC), 0, "everything should be freed");
	REQUIRE_EQ(ccv_nnc_dynamic_graph_bookkeeping_count(graph, CCV_NNC_SYMBOL_TENSOR), 0, "everything should be freed");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("long chain to autograd, simulate random free of unused tensors due to garbage collection")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x[10];
	ccv_nnc_tensor_variable_t f[10];
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_variable_t dy[3];
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.3;
	int i, j;
	for (i = 0; i < 10; i++)
	{
		x[i] = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
		ccv_nnc_tensor_from_variable(graph, x[i])->data.f32[0] = i + 1;
		f[i] = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x[i], y), TENSOR_VARIABLE_LIST(f[i]), 0, 0);
		if (i >= 5)
		{
			if (i - 5 < 3)
				dy[i - 5] = ccv_nnc_tensor_variable_new(graph);
			ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(f[i]), 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(dy[ccv_min(2, i - 5)]), 0);
		}
		if (i == 7)
			for (j = 0; j < 5; j++)
				ccv_nnc_tensor_variable_free(graph, f[j]);
	}
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f[9])->data.f32[0], 1.3 * 10, 1e-5, "should equal to 1.3 * 10");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy[2])->data.f32[0], 8 + 9 + 10, 1e-5, "should equal to sum of the last 3");
	for (i = 0; i < 10; i++)
		ccv_nnc_tensor_variable_free(graph, x[i]);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("parallel exec with a stream, focus on uma tensors")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_CPU);
	ccv_nnc_tensor_variable_t x[2];
	x[0] = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	x[1] = ccv_nnc_tensor_variable_new(graph, CPU_TENSOR_NHWC(32F, 1));
	ccv_nnc_tensor_from_variable(graph, x[0])->data.f32[0] = 10;
	ccv_nnc_tensor_from_variable(graph, x[1])->data.f32[0] = 9;
	ccv_nnc_tensor_variable_t y[2];
	y[0] = ccv_nnc_tensor_variable_new(graph);
	y[1] = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(0.4), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x[0], x[1]), TENSOR_VARIABLE_LIST(y[0], y[1]), 2, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_stream_context_free(stream);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y[0])->data.f32[0], 10 * 0.4, 1e-5, "should match the result");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y[1])->data.f32[0], 9 * 0.4, 1e-5, "should match the result");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("parallel exec with a stream, focus on potential memory issues")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_CPU);
	ccv_nnc_tensor_variable_t x[2];
	x[0] = ccv_nnc_tensor_variable_new(graph, CPU_NUMA_TENSOR_NHWC(000, 32F, 1));
	x[1] = ccv_nnc_tensor_variable_new(graph, CPU_NUMA_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_from_variable(graph, x[0])->data.f32[0] = 10;
	ccv_nnc_tensor_from_variable(graph, x[1])->data.f32[0] = 9;
	ccv_nnc_tensor_variable_t y[2];
	y[0] = ccv_nnc_tensor_variable_new(graph);
	y[1] = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(0.4), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x[0], x[1]), TENSOR_VARIABLE_LIST(y[0], y[1]), 2, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_stream_context_free(stream);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y[0])->data.f32[0], 10 * 0.4, 1e-5, "should match the result");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y[1])->data.f32[0], 9 * 0.4, 1e-5, "should match the result");
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
