#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("random normal distribution")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RANDOM_NORMAL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_RANDOM_NORMAL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 100000), "x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RANDOM_NORMAL_FORWARD(1, 0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(x), "random uniform");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	int i;
	double mean = 0;
	for (i = 0; i < 100000; i++)
		mean += xt->data.f32[i];
	mean = mean / 100000.0;
	double std = 0;
	for (i = 0; i < 100000; i++)
		std += (xt->data.f32[i] - mean) * (xt->data.f32[i] - mean);
	std = sqrt(std / 100000.0);
	REQUIRE_EQ_WITH_TOLERANCE(mean, 0, 1e-2, "mean should be 0");
	REQUIRE_EQ_WITH_TOLERANCE(std, 1, 1e-2, "std should be 1");
	ccv_nnc_tensor_free(xt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("random uniform distribution")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 100000), "x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RANDOM_UNIFORM_FORWARD(-8, 4), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(x), "random uniform");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	int i;
	int h[4 + 8] = {};
	for (i = 0; i < 100000; i++)
	{
		REQUIRE(xt->data.f32[i] > -8 - 1e-5, "it must be bigger than lower bound");
		REQUIRE(xt->data.f32[i] < 4 + 1e-5, "and smaller than upper bound");
		int b = (int)roundf(xt->data.f32[i] - 0.5) + 8;
		b = ccv_max(ccv_min(b, 11), 0);
		++h[b];
	}
	const int count = (int)roundf(100000. / (4 + 8));
	for (i = 0; i < 12; i++)
		{ REQUIRE(h[i] >= count - 1000 && h[i] <= count + 1000, "uniform distribution"); }
	ccv_nnc_tensor_free(xt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("random uniform distribution in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 100000), "x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RANDOM_UNIFORM_FORWARD(-8, 4), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(x), "random uniform");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100000), 0);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16t), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16t), TENSOR_LIST(xt), 0);
	int i;
	int h[4 + 8] = {};
	for (i = 0; i < 100000; i++)
	{
		REQUIRE(xt->data.f32[i] > -8 - 1e-5, "it must be bigger than lower bound");
		REQUIRE(xt->data.f32[i] < 4 + 1e-5, "and smaller than upper bound");
		int b = (int)roundf(xt->data.f32[i] - 0.5) + 8;
		b = ccv_max(ccv_min(b, 11), 0);
		++h[b];
	}
	const int count = (int)roundf(100000. / (4 + 8));
	for (i = 0; i < 12; i++)
		{ REQUIRE(h[i] >= count - 1000 && h[i] <= count + 1000, "uniform distribution"); }
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_free(x16t);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
