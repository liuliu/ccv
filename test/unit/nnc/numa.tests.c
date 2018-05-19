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

TEST_CASE("simple graph x * 0.3 + 3 compiled with NUMA")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_NUMA_TENSOR(000, 1), "a");
	const ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_NUMA_TENSOR(002, 1), "b");
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_NUMA_TENSOR(000, 1), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_NUMA_TENSOR(001, 1), "y");
	const ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_NUMA_TENSOR(002, 1), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(x, a), TENSOR_SYMBOL_LIST(y), "prod");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(y, b), TENSOR_SYMBOL_LIST(z), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 1.21;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	a_tensor->data.f32[0] = 0.3;
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	b_tensor->data.f32[0] = 3;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE(x_tensor->data.f32 != y_tensor->data.f32, "y cannot be assigned to the same place of x due to NUMA constraint");
	REQUIRE(z_tensor->data.f32 != x_tensor->data.f32, "z cannot be assigned to the same place of x due to NUMA constraint");
	REQUIRE(z_tensor->data.f32 != y_tensor->data.f32, "z cannot be assigned to the same place of y due to NUMA constraint");
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_run(graph, 0, 0, &source, 1, &destination, 1);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 1.21 * 0.3 + 3, 1e-6, "should be equal to 1.21 * 0.3 + 3");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
