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

TEST_CASE("compile symbolic graph of one node")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "c");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "prod");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, GRAPH_EXEC_SYMBOL_LIST(prod), GRAPH_EXEC_SYMBOL_LIST(prod), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	a_tensor->data.f32[0] = 1.2;
	b_tensor->data.f32[0] = 2.3;
	ccv_nnc_graph_exec_t prod_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod);
	ccv_nnc_graph_run(graph, 0, GRAPH_EXEC_LIST(prod_exec), GRAPH_EXEC_LIST(prod_exec));
	REQUIRE(a_tensor->data.f32 == c_tensor->data.f32, "trivially in-place operation, should point to the same memory region");
	REQUIRE_EQ_WITH_TOLERANCE(c_tensor->data.f32[0], 1.2 * 2.3, 1e-6, "should be equal to 1.2 * 2.3");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compile a simple symbolic graph with autogen")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(31, 21, 2), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(31, 21, 4), "b");
	ccv_nnc_cmd_t forw_cmd = CMD_CONVOLUTION_FORWARD(4, 5, 3, 2);
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(4, 5, 3, 2), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(4), "bias");
	// See what we compile to when have unused tensors.
	ccv_nnc_tensor_symbol_t unused0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "unused0");
	ccv_nnc_tensor_symbol_t unused1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "unused1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, forw_cmd, TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b), "forw");
	ccv_nnc_tensor_symbol_t m = ccv_nnc_tensor_symbol_new(symbolic_graph, b.info, "m");
	ccv_nnc_cmd_t softmax_cmd = ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, 0, ccv_nnc_cmd_auto, 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, softmax_cmd, TENSOR_SYMBOL_LIST(b), TENSOR_SYMBOL_LIST(m), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* m_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, m);
	REQUIRE(a_tensor->data.u8 != b_tensor->data.u8, "tensor a and b shouldn't share the memory.");
	REQUIRE(b_tensor->data.u8 == m_tensor->data.u8, "tensor b and m should share the memory because softmax is an inplace op.");
	REQUIRE(ccv_nnc_tensor_from_symbol(tensor_arena, unused0) == 0, "tensor unused 0 should have not pointed memory.");
	REQUIRE(ccv_nnc_tensor_from_symbol(tensor_arena, unused1) == 0, "tensor unused 0 should have not pointed memory.");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
