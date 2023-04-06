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

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		6,
		15
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		5, 7, 9
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce mean for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		2,
		5
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce mean for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		2.5, 3.5, 4.5
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("use reduce for softmax")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 100), "x");
	ccv_nnc_tensor_symbol_t max = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "max");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_MAX_FORWARD(0), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(max), "max");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(1, -1), TENSOR_SYMBOL_LIST(x, max), TENSOR_SYMBOL_LIST(y), "neg");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 100), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWEXP_FORWARD(), TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(z), "exp");
	ccv_nnc_tensor_symbol_t sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_REDUCE_SUM_FORWARD(0), TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(sum), "sum");
	ccv_nnc_tensor_symbol_t inv_sum = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "1 / sum");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWDIV_FORWARD(), TENSOR_SYMBOL_LIST(NO_TENSOR_SYMBOL, sum), TENSOR_SYMBOL_LIST(inv_sum), "inv sum");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 100), "a");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(z, inv_sum), TENSOR_SYMBOL_LIST(a), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t da = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, a);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const da_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(a, a_tensor), KV(da, da_tensor)), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const tx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 100; i++)
		x_tensor->data.f32[i] = tx_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 100; i++)
		da_tensor->data.f32[i] = 0;
	da_tensor->data.f32[88] = 1;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const ta_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tx_tensor), TENSOR_LIST(ta_tensor), 0);
	REQUIRE_TENSOR_EQ(a_tensor, ta_tensor, "softmax should match from the graph");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da_tensor, 0, ta_tensor), TENSOR_LIST(tdx_tensor), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_TENSOR_EQ(dx_tensor, tdx_tensor, "softmax backward should match from the graph");
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(tx_tensor);
	ccv_nnc_tensor_free(ta_tensor);
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(da_tensor);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("reduce max for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		3,
		6
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce max for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		4, 5, 6
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce min for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		1,
		4
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce min for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		1, 2, 3
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 1 with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	const int axis = 1;
	ccv_cnnp_model_t* const reduce_sum = ccv_cnnp_reduce_sum(&axis, 1, 0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 3);
	ccv_cnnp_model_compile(reduce_sum, TENSOR_PARAM_LIST(a_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(reduce_sum, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		6,
		15
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_cnnp_model_free(reduce_sum);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce max for [[1, 2, 3], [4, 5, 6]] on axis 0 with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	const int axis = 0;
	ccv_cnnp_model_t* const reduce_max = ccv_cnnp_reduce_max(&axis, 1, 0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 3);
	ccv_cnnp_model_compile(reduce_max, TENSOR_PARAM_LIST(a_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(reduce_max, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		4, 5, 6
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_cnnp_model_free(reduce_max);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce min for [[1, 2, 3], [4, 5, 6]] on axis 0 with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	const int axis = 0;
	ccv_cnnp_model_t* const reduce_min = ccv_cnnp_reduce_min(&axis, 1, 0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 3);
	ccv_cnnp_model_compile(reduce_min, TENSOR_PARAM_LIST(a_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(reduce_min, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		1, 2, 3
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_cnnp_model_free(reduce_min);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce mean for [[1, 2, 3], [4, 5, 6]] on axis 1 with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	const int axis = 1;
	ccv_cnnp_model_t* const reduce_mean = ccv_cnnp_reduce_mean(&axis, 1, 0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 3);
	ccv_cnnp_model_compile(reduce_mean, TENSOR_PARAM_LIST(a_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(reduce_mean, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		2,
		5
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_cnnp_model_free(reduce_mean);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("argmax for [[1, 2, 7], [5, 6, 4]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 7;
	a->data.f32[3] = 5;
	a->data.f32[4] = 6;
	a->data.f32[5] = 4;
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	int btp[] = {
		1, 1, 0
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("argmin for [[1, 2, 7], [5, 6, 4]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 7;
	a->data.f32[3] = 5;
	a->data.f32[4] = 6;
	a->data.f32[5] = 4;
	ccv_nnc_cmd_exec(CMD_ARGMIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	int btp[] = {
		0, 0, 1
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("argmax for [[1, 2, 7], [5, 6, 4]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 7;
	a->data.f32[3] = 5;
	a->data.f32[4] = 6;
	a->data.f32[5] = 4;
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	int btp[] = {
		2,
		1
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32S, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce norm2 for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		sqrt(1 + 2 * 2 + 3 * 3),
		sqrt(4 * 4 + 5 * 5 + 6 * 6)
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce norm2 for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		sqrt(1 + 4 * 4), sqrt(2 * 2 + 5 * 5), sqrt(3 * 3 + 6 * 6)
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce norm2 for [[1, 2, 3], [4, 5, 6]] on axis 1 backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	g->data.f32[0] = 0.5;
	g->data.f32[1] = 0.5;
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	float htp[] = {
		0.5 * 1 / sqrt(1 + 2 * 2 + 3 * 3),
		0.5 * 2 / sqrt(1 + 2 * 2 + 3 * 3),
		0.5 * 3 / sqrt(1 + 2 * 2 + 3 * 3),
		0.5 * 4 / sqrt(4 * 4 + 5 * 5 + 6 * 6),
		0.5 * 5 / sqrt(4 * 4 + 5 * 5 + 6 * 6),
		0.5 * 6 / sqrt(4 * 4 + 5 * 5 + 6 * 6),
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
}

TEST_CASE("reduce norm2 for [[1, 2, 3], [4, 5, 6]] on axis 0 backward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	g->data.f32[0] = 1;
	g->data.f32[1] = 1;
	g->data.f32[2] = 1;
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	float htp[] = {
		1 / sqrt(1 + 4 * 4), 2 / sqrt(2 * 2 + 5 * 5), 3 / sqrt(3 * 3 + 6 * 6),
		4 / sqrt(1 + 4 * 4), 5 / sqrt(2 * 2 + 5 * 5), 6 / sqrt(3 * 3 + 6 * 6)
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
}

TEST_CASE("reduce norm2 for [[1, 2, 3], [4, 5, 6]] on axis 0 with model")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	const int axis = 0;
	ccv_cnnp_model_t* const reduce_norm2 = ccv_cnnp_reduce_norm2(&axis, 1, 0);
	const ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, 2, 3);
	ccv_cnnp_model_compile(reduce_norm2, TENSOR_PARAM_LIST(a_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(reduce_norm2, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 0,
		.disable_outgrad = CCV_CNNP_DISABLE_OUTGRAD_ALL
	}, TENSOR_LIST(a), TENSOR_LIST(b), 0, 0);
	float btp[] = {
		sqrt(1 + 4 * 4), sqrt(2 * 2 + 5 * 5), sqrt(3 * 3 + 6 * 6)
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_cnnp_model_free(reduce_norm2);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
