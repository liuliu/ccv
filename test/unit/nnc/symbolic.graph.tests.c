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
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "prod");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, GRAPH_EXEC_SYMBOL_LIST(prod), GRAPH_EXEC_SYMBOL_LIST(prod), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_SHORT_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	a_tensor->data.f32[0] = 1.2;
	b_tensor->data.f32[0] = 2.3;
	ccv_nnc_graph_exec_t prod_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, prod);
	ccv_nnc_graph_run(graph, 0, 0, GRAPH_EXEC_LIST(prod_exec), GRAPH_EXEC_LIST(prod_exec));
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
	ccv_nnc_cmd_t forw_cmd = CMD_CONVOLUTION_FORWARD(1, 4, 5, 3, 2);
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(4, 5, 3, 2), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(4), "bias");
	// See what we compile to when have unused tensors.
	ccv_nnc_tensor_symbol_t unused0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "unused0");
	ccv_nnc_tensor_symbol_t unused1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "unused1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, forw_cmd, TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b), "forw");
	ccv_nnc_tensor_symbol_t m = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(31, 21, 4), "m");
	ccv_nnc_cmd_t softmax_cmd = CMD_SOFTMAX_FORWARD();
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, softmax_cmd, TENSOR_SYMBOL_LIST(b), TENSOR_SYMBOL_LIST(m), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
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

TEST_CASE("use symbolic graph disjoin and free")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "c");
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_tensor_symbol_t p = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "p");
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "q");
	ccv_nnc_graph_exec_symbol_t prod = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z), "prod");
	ccv_nnc_graph_exec_symbol_t log0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(p), TENSOR_SYMBOL_LIST(q), "log0");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "sum");
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, sum, prod);
	ccv_nnc_graph_exec_symbol_concat(symbolic_graph, sum, log0);
	ccv_nnc_graph_exec_symbol_disjoin(symbolic_graph, sum, prod);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	REQUIRE(ccv_nnc_symbolic_graph_source_size(symbolic_graph) == 2, "sources now should contain both sum and prod");
	REQUIRE(ccv_nnc_symbolic_graph_destination_size(symbolic_graph) == 2, "destinations now should contain both log0 and prod");
	ccv_nnc_graph_exec_symbol_free(symbolic_graph, prod);
	REQUIRE(ccv_nnc_symbolic_graph_source_size(symbolic_graph) == 1, "sources now should contain sum");
	REQUIRE(ccv_nnc_symbolic_graph_destination_size(symbolic_graph) == 1, "destinations now should contain log0");
	REQUIRE(ccv_nnc_symbolic_graph_sources(symbolic_graph)->d == sum.d, "source should be sum");
	REQUIRE(ccv_nnc_symbolic_graph_destinations(symbolic_graph)->d == log0.d, "source should be log0");
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("set tensor symbol shape after computation specified")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWLOG_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "log");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_set(symbolic_graph, a, ONE_CPU_TENSOR(1));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	GRAPH_GEN(graph, CCV_NNC_SHORT_DOT_GRAPH);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 1.25;
	ccv_nnc_graph_run(graph, 0, 0, 0, 0, 0, 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	REQUIRE_EQ_WITH_TOLERANCE(b_tensor->data.f32[0], logf(1.25), 1e-5, "should be equal to logf(1.25)");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
