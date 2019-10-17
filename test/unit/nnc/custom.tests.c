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

static int _ccv_nnc_cmd_custom_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (cmd.cmd == CCV_NNC_CUSTOM_FORWARD)
	{
		ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
		const ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, x, inputs[0]);
		const ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(z));
		const ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, y, outputs[0]);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y));
		ccv_nnc_dynamic_graph_free(graph);
	} else {
		ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
		const ccv_nnc_tensor_variable_t x = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, x, inputs[1]);
		const ccv_nnc_tensor_variable_t z = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(z));
		const ccv_nnc_tensor_variable_t y = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, y, inputs[2]);
		ccv_nnc_dynamic_graph_exec(graph, CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(z), TENSOR_VARIABLE_LIST(y));
		const ccv_nnc_tensor_variable_t dy = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, dy, inputs[0]);
		const ccv_nnc_tensor_variable_t dx = ccv_nnc_tensor_variable_new(graph);
		ccv_nnc_tensor_variable_set(graph, dx, outputs[0]);
		ccv_nnc_dynamic_graph_backward(graph, y, dy, TENSOR_VARIABLE_LIST(x), TENSOR_VARIABLE_LIST(dx));
		ccv_nnc_dynamic_graph_free(graph);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static ccv_nnc_cmd_vtab_t _custom_isa = {
	.exec = _ccv_nnc_cmd_custom_exec,
};

TEST_CASE("custom forward operation with dynamic graph")
{
	const ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_CUSTOM_FORWARD, &_custom_isa, (ccv_nnc_cmd_param_t){}, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, cmd, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "custom");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS | CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0,
		TENSOR_SYMBOL_LIST(y),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 10;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_EQ_WITH_TOLERANCE(y_tensor->data.f32[0], log(log(10)), 1e-5, "computed result should be identical");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("custom backward operation with dynamic graph")
{
	const ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_CUSTOM_FORWARD, &_custom_isa, (ccv_nnc_cmd_param_t){}, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, cmd, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "custom");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS | CCV_NNC_AUTOGEN_ALL_EXECS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	const ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dy), "set");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS | CCV_NNC_AUTOGEN_ALL_EXECS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0,
		TENSOR_SYMBOL_LIST(dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 10;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_EQ_WITH_TOLERANCE(dx_tensor->data.f32[0], 1. / (log(10) * 10), 1e-5, "computed result should be identical");
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
