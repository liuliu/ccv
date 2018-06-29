#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <inc/ccv_convnet_internal.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("dense net with GEMM as the core")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 4), "x");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x1");
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 1), "w1");
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b1");
	ccv_nnc_tensor_symbol_t x11 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 1), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x11");
	ccv_nnc_graph_exec_symbol_t e1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x1, w1, b1), TENSOR_SYMBOL_LIST(x11), "e1");
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 2), "x2");
	ccv_nnc_tensor_symbol_t w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 2), "w2");
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b2");
	ccv_nnc_tensor_symbol_t x21 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 2), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x21");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x2, w2, b2), TENSOR_SYMBOL_LIST(x21), "e2");
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 3), "x3");
	ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 3), "w3");
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b3");
	ccv_nnc_tensor_symbol_t x31 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 3), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x31");
	ccv_nnc_graph_exec_symbol_t e3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x3, w3, b3), TENSOR_SYMBOL_LIST(x31), "e3");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, GRAPH_EXEC_SYMBOL_LIST(e1), GRAPH_EXEC_SYMBOL_LIST(e3), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* x1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x1);
	x1_tensor->data.f32[0] = 0.472;
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w1);
	w1_tensor->data.f32[0] = 0.234;
	ccv_nnc_tensor_t* w2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w2);
	w2_tensor->data.f32[0] = 0.374;
	w2_tensor->data.f32[1] = 0.886;
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w3);
	w3_tensor->data.f32[0] = 0.484;
	w3_tensor->data.f32[1] = 0.912;
	w3_tensor->data.f32[2] = 0.235;
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b1);
	b1_tensor->data.f32[0] = 0.1;
	ccv_nnc_tensor_t* b2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b2);
	b2_tensor->data.f32[0] = 0.2;
	ccv_nnc_tensor_t* b3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b3);
	b3_tensor->data.f32[0] = 0.3;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_source(graph_exec_arena)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_destination(graph_exec_arena)));
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_t* xt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1, 4), 0);
	xt->data.f32[0] = 0.472;
	xt->data.f32[1] = xt->data.f32[0] * 0.234 + 0.1;
	xt->data.f32[2] = xt->data.f32[0] * 0.374 + xt->data.f32[1] * 0.886 + 0.2;
	xt->data.f32[3] = xt->data.f32[0] * 0.484 + xt->data.f32[1] * 0.912 + xt->data.f32[2] * 0.235 + 0.3;
	REQUIRE_MATRIX_EQ(x_tensor, xt, "1x4 matrix should be exactly the same");
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("dense net with GEMM as the core and autograd")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 4), "x");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x1");
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 1), "w1");
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b1");
	ccv_nnc_tensor_symbol_t x11 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 1), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x11");
	ccv_nnc_graph_exec_symbol_t e1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x1, w1, b1), TENSOR_SYMBOL_LIST(x11), "e1");
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 2), "x2");
	ccv_nnc_tensor_symbol_t w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 2), "w2");
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b2");
	ccv_nnc_tensor_symbol_t x21 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 2), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x21");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x2, w2, b2), TENSOR_SYMBOL_LIST(x21), "e2");
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, ccv_nnc_no_ofs, DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 3), "x3");
	ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 3), "w3");
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b3");
	ccv_nnc_tensor_symbol_t x31 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x, DIM_ALLOC(0, 3), DIM_ALLOC(1, 4), ONE_CPU_TENSOR(1, 1), "x31");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1, 1), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x3, w3, b3), TENSOR_SYMBOL_LIST(x31), "e3");
	ccv_nnc_graph_exec_symbol_t e4 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x31), TENSOR_SYMBOL_LIST(y), "e4");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(w1), GRAPH_EXEC_SYMBOL_LIST(e1), GRAPH_EXEC_SYMBOL_LIST(e4));
	ccv_nnc_tensor_symbol_t dw1 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w1);
	ccv_nnc_graph_exec_symbol_t de1 = ccv_nnc_graph_exec_symbol_for_backward(symbolic_graph, dw1);
	ccv_nnc_symbolic_graph_simplify(symbolic_graph,
		SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION, CCV_NNC_SIMPLIFY_GRAPH_PRUNING),
		TENSOR_SYMBOL_LIST(dw1), GRAPH_EXEC_SYMBOL_LIST(e1), GRAPH_EXEC_SYMBOL_LIST(de1));
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_t* dy_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	dy_tensor->data.f32[0] = 1;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, TENSOR_BIND_MAP(KV(dy, dy_tensor)), 0, 0, GRAPH_EXEC_SYMBOL_LIST(e1), GRAPH_EXEC_SYMBOL_LIST(de1), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x1);
	x1_tensor->data.f32[0] = 0.472;
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w1);
	w1_tensor->data.f32[0] = 0.234;
	ccv_nnc_tensor_t* w2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w2);
	w2_tensor->data.f32[0] = 0.374;
	w2_tensor->data.f32[1] = 0.886;
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w3);
	w3_tensor->data.f32[0] = 0.484;
	w3_tensor->data.f32[1] = 0.912;
	w3_tensor->data.f32[2] = 0.235;
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b1);
	b1_tensor->data.f32[0] = 0.1;
	ccv_nnc_tensor_t* b2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b2);
	b2_tensor->data.f32[0] = 0.2;
	ccv_nnc_tensor_t* b3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b3);
	b3_tensor->data.f32[0] = 0.3;
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(ccv_nnc_graph_exec_source(graph_exec_arena)), GRAPH_EXEC_LIST(ccv_nnc_graph_exec_destination(graph_exec_arena)));
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_t* dw1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dw1);
	REQUIRE_EQ_WITH_TOLERANCE((0.235 * 0.886 + 0.912) * 0.472, dw1_tensor->data.f32[0], 1e-5, "the gradient should be equal to a complicated result");
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
