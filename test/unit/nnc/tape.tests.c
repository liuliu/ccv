#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/_ccv_nnc_graph.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static int while_5(ccv_nnc_tensor_t* const* const commons, const int common_size, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const void* const data)
{
	return commons[0]->data.i64[0] < 5;
}

TEST_CASE("new tape from a graph")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_symbolic_graph_t* while_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_graph_exec_symbol_t while_symbol = ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_symbolic_graph, "for 1..5");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(while_symbolic_graph, ONE_CPU_TENSOR(1), "z1");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_symbolic_graph, ccv_nnc_cmd(CCV_NNC_NOOP, 0, CMD_GENERIC(), 0), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t prod0 = ccv_nnc_graph_exec_symbol_new(while_symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z0, x), TENSOR_SYMBOL_LIST(z1), "prod0");
	ccv_nnc_graph_exec_symbol_concat(while_symbolic_graph, noop, prod0);
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z2");
	ccv_nnc_tensor_symbol_t z3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "z3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z1, y), TENSOR_SYMBOL_LIST(z2), "prod1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, CMD_GENERIC(), 0), TENSOR_SYMBOL_LIST(z2, z0), TENSOR_SYMBOL_LIST(z3), "sum");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_while_expr(while_symbolic_graph, while_5, 0, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_while_params(while_symbolic_graph, TENSOR_SYMBOL_MAP(KV(z1, z0)));
	ccv_nnc_symbolic_graph_set_sources(while_symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_symbolic_graph, GRAPH_EXEC_SYMBOL_LIST(prod0));
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, ccv_nnc_symbolic_graph_sources(symbolic_graph), ccv_nnc_symbolic_graph_source_size(symbolic_graph), ccv_nnc_symbolic_graph_destinations(symbolic_graph), ccv_nnc_symbolic_graph_destination_size(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_t while_exec = ccv_nnc_graph_exec_from_symbol(graph_exec_arena, while_symbol);
	ccv_nnc_graph_t* while_graph = ccv_nnc_graph_from_graph_exec(graph, while_exec);
	ccv_nnc_tensor_tape_t* tape = ccv_nnc_tensor_tape_new();
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	ccv_numeric_data_t data = tensor->data; // Preserve the data field.
	tensor->type |= CCV_TAPE_ALLOC;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (0, 0).");
	tensor->data.f32[0] = 0.32;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (0, 2).");
	tensor->data.f32[0] = 0.11;
	graph->while_count = 1;
	while_graph->while_count = 1;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (1, 1).");
	tensor->data.f32[0] = 0.58;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (1, 2).");
	tensor->data.f32[0] = 0.18;
	graph->while_count = 2;
	while_graph->while_count = 3;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (2, 3).");
	tensor->data.f32[0] = 1.29;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(), TENSOR_LIST(tensor));
	REQUIRE(tensor->data.u8 != data.u8, "Tensor should assigned a different memory region with (2, 2).");
	tensor->data.f32[0] = 0.02;
	graph->while_count = 0;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 0.11, 1e-5, "Tensor should retain 0.11 with (0, 2).");
	graph->while_count = 1;
	while_graph->while_count = 1;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 0.58, 1e-5, "Tensor should retain 0.58 with (1, 1).");
	graph->while_count = 0;
	while_graph->while_count = 0;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 0.32, 1e-5, "Tensor should retain 0.32 with (0, 0).");
	graph->while_count = 2;
	while_graph->while_count = 3;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 1.29, 1e-5, "Tensor should retain 1.29 with (2, 3).");
	graph->while_count = 2;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 0.02, 1e-5, "Tensor should retain 0.02 with (2, 2).");
	graph->while_count = 1;
	while_graph->while_count = 2;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 0.18, 1e-5, "Tensor should retain 0.02 with (1, 2).");
	// In next loop, try to access data (it will return the data from previous loop).
	graph->while_count = 2;
	while_graph->while_count = 4;
	ccv_nnc_tensor_tape_io(tape, while_graph, TENSOR_LIST(tensor), TENSOR_LIST());
	REQUIRE_EQ_WITH_TOLERANCE(tensor->data.f32[0], 1.29, 1e-5, "Tensor should retain 1.29 with (2, 4).");
	tensor->data = data;
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_tape_free(tape);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
