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
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, a)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(a), TENSOR_VARIABLE_LIST(b));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, b)->data.f32[0], logf(19), 1e-5, "log(19) result should be equal.");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph to compute f(x) = x * log(x) + 1.2 * x, f'(x) where x = 19")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 19;
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, f), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 1.2;
	ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(f, z), TENSOR_VARIABLE_LIST(f));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, f)->data.f32[0], 19 * logf(19) + 1.2 * 19, 1e-5, "f(x) = 1.2 * 19 + 19 * log(19)");
	// Do gradient computation multiple times.
	ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	ccv_nnc_tensor_variable_t dy = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, 0, TENSOR_VARIABLE_LIST(y), TENSOR_VARIABLE_LIST(dy));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	ccv_nnc_tensor_variable_t dz = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, f, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(dz));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dz)->data.f32[0], 1, 1e-5, "f'(z) = 1");
	ccv_nnc_dynamic_graph_backward(graph, f, 0, TENSOR_VARIABLE_LIST(y, x), TENSOR_VARIABLE_LIST(dy, dx));
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dx)->data.f32[0], logf(19) + 1 + 1.2, 1e-5, "f'(x) = 1.2 + log(19) + 19 * 1 / 19");
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, dy)->data.f32[0], 19, 1e-5, "f'(y) = 19");
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("dynamic graph with dense net (extensive use of alias)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 4));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 0.472;
	ccv_nnc_tensor_variable_t x1 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_tensor_variable_t w1 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 1));
	ccv_nnc_tensor_from_variable(graph, w1)->data.f32[0] = 0.234;
	ccv_nnc_tensor_variable_t b1 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b1)->data.f32[0] = 0.1;
	ccv_nnc_tensor_variable_t x11 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 1), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x1, w1, b1), TENSOR_VARIABLE_LIST(x11));
	ccv_nnc_tensor_variable_t x2 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 2));
	ccv_nnc_tensor_variable_t w2 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 2));
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[0] = 0.374;
	ccv_nnc_tensor_from_variable(graph, w2)->data.f32[1] = 0.886;
	ccv_nnc_tensor_variable_t b2 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b2)->data.f32[0] = 0.2;
	ccv_nnc_tensor_variable_t x21 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 2), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x2, w2, b2), TENSOR_VARIABLE_LIST(x21));
	ccv_nnc_tensor_variable_t x3 = ccv_nnc_tensor_variable_alias_new(graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 3));
	ccv_nnc_tensor_variable_t w3 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1, 3));
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[0] = 0.484;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[1] = 0.912;
	ccv_nnc_tensor_from_variable(graph, w3)->data.f32[2] = 0.235;
	ccv_nnc_tensor_variable_t b3 = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, b3)->data.f32[0] = 0.3;
	ccv_nnc_tensor_variable_t x31 = ccv_nnc_tensor_variable_alias_new(graph, x, DIM_ALLOC(0, 3), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_GEMM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x3, w3, b3), TENSOR_VARIABLE_LIST(x31));
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* xt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1, 4), 0);
	xt->data.f32[0] = 0.472;
	xt->data.f32[1] = xt->data.f32[0] * 0.234 + 0.1;
	xt->data.f32[2] = xt->data.f32[0] * 0.374 + xt->data.f32[1] * 0.886 + 0.2;
	xt->data.f32[3] = xt->data.f32[0] * 0.484 + xt->data.f32[1] * 0.912 + xt->data.f32[2] * 0.235 + 0.3;
	REQUIRE_MATRIX_EQ(ccv_nnc_tensor_from_variable(graph, x), xt, "1x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_variable_t dw1 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, x, 0, TENSOR_VARIABLE_LIST(w1), TENSOR_VARIABLE_LIST(dw1));
	REQUIRE_EQ_WITH_TOLERANCE((0.235 * 0.886 + 0.912) * 0.472, ccv_nnc_tensor_from_variable(graph, dw1)->data.f32[0], 1e-5, "the gradient should be equal to a complicated result");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("batch norm in dynamic graph (enforce inplace)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(2, 2, 2, 10));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t scale = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(10));
	ccv_nnc_tensor_variable_t bias = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(10));
	ccv_nnc_tensor_variable_t mean = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(10));
	ccv_nnc_tensor_variable_t var = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(10));
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
	ccv_nnc_dynamic_graph_exec(graph, CMD_BATCH_NORM_FORWARD(0, 0, 0.9, 0, 1, 2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, scale, bias, mean, var), TENSOR_VARIABLE_LIST(y, mean, var, saved_mean, saved_inv_std));
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE(mean_tensor_ptr == ccv_nnc_tensor_from_variable(graph, mean), "enforced inplace, tensor view pointer unchanged");
	REQUIRE(var_tensor_ptr == ccv_nnc_tensor_from_variable(graph, var), "enforced inplace, tensor view pointer unchanged");
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 2, 2, 10), 0);
	ccv_nnc_tensor_t* scale_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(10), 0);
	ccv_nnc_tensor_t* bias_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(10), 0);
	ccv_nnc_tensor_t* mean_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(10), 0);
	ccv_nnc_tensor_t* var_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(10), 0);
	ccv_nnc_tensor_t* saved_mean_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1, 1, 1, 10), 0);
	ccv_nnc_tensor_t* saved_inv_std_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1, 1, 1, 10), 0);
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
	ccv_nnc_tensor_variable_t df = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_from_variable(graph, df)->data.f32[0] = 1;
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(df, 0, x), TENSOR_VARIABLE_LIST(y, 0));
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, y)->data.f32[0], 1. / 10, 1e-5, "div backward should equal to 1 / 10");
	ccv_nnc_dynamic_graph_free(graph);
}

TEST_CASE("long dynamic graph with unused variables freed")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	int i;
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 32;
	ccv_nnc_tensor_from_variable(graph, y)->data.f32[0] = 0.5;
	for (i = 0; i < 10; i++)
	{
		ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		if (i < 7)
			ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(z));
		else {
			if (i == 7)
				ccv_nnc_tensor_variable_free(graph, y); // No longer need y.
			ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, x), TENSOR_VARIABLE_LIST(z));
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

TEST_CASE("compute f(x) = x * log(x) + x, f'(x) when x = 10 (and intermediate results all freed)")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t f = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_from_variable(graph, x)->data.f32[0] = 10;
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(y));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x, y), TENSOR_VARIABLE_LIST(y));
	ccv_nnc_dynamic_graph_exec(graph, CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(y, x), TENSOR_VARIABLE_LIST(f));
	ccv_nnc_tensor_variable_t df = ccv_nnc_tensor_variable_new(graph, ONE_CPU_TENSOR(1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_VARIABLE_LIST(df));
	ccv_nnc_dynamic_graph_backward(graph, f, df, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(x));
	DYNAMIC_GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(ccv_nnc_tensor_from_variable(graph, x)->data.f32[0], log(10) + 1 + 1, 1e-5, "dx should equal to the computed result");
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
