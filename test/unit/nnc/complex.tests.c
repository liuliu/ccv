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

TEST_CASE("compare cmul with gemm computed result")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "y");
	const ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(z),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* gemm_x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 2), 0);
	memcpy(gemm_x->data.f32, x_tensor->data.f32, sizeof(float) * 10);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	for (i = 0; i < 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* gemm_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 2, 2), 0);
	for (i = 0; i < 5; i++)
	{
		gemm_y->data.f32[i * 4] = y_tensor->data.f32[i * 2];
		gemm_y->data.f32[i * 4 + 1] = -y_tensor->data.f32[i * 2 + 1];
		gemm_y->data.f32[i * 4 + 2] = y_tensor->data.f32[i * 2 + 1];
		gemm_y->data.f32[i * 4 + 3] = y_tensor->data.f32[i * 2];
	}
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	ccv_nnc_tensor_t* gemm_z = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 2), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gemm_x, gemm_y), TENSOR_LIST(gemm_z), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, z_tensor->data.f32, gemm_z->data.f32, 10, 1e-5, "should match as if GEMM");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(gemm_x);
	ccv_nnc_tensor_free(gemm_y);
	ccv_nnc_tensor_free(gemm_z);
}

TEST_CASE("compare cmul gradient with gemm computed result")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "x");
	const ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10), "y");
	const ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(z), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(z), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(z, dx),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	const ccv_nnc_tensor_symbol_t dz = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, z);
	ccv_nnc_tensor_t* const dz_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dz);
	for (i = 0; i < 10; i++)
		dz_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const gemm_symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t gemm_x = ccv_nnc_tensor_symbol_new(gemm_symbolic_graph, CPU_TENSOR_NHWC(32F, 5, 1, 2), "x");
	const ccv_nnc_tensor_symbol_t gemm_y = ccv_nnc_tensor_symbol_new(gemm_symbolic_graph, CPU_TENSOR_NHWC(32F, 5, 2, 2), "y");
	const ccv_nnc_tensor_symbol_t gemm_z = ccv_nnc_tensor_symbol_new(gemm_symbolic_graph, ccv_nnc_tensor_auto, "z");
	ccv_nnc_graph_exec_symbol_new(gemm_symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), TENSOR_SYMBOL_LIST(gemm_x, gemm_y), TENSOR_SYMBOL_LIST(gemm_z), "gemm");
	ccv_nnc_graph_exec_symbol_autogen(gemm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(gemm_symbolic_graph, TENSOR_SYMBOL_LIST(gemm_z), TENSOR_SYMBOL_LIST(gemm_x), SYMBOLIC_GRAPH_SOURCES(gemm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(gemm_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(gemm_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	const ccv_nnc_tensor_symbol_t dgemmx = ccv_nnc_tensor_symbol_for_backward(gemm_symbolic_graph, gemm_x);
	ccv_nnc_graph_t* gemm_graph = 0;
	ccv_nnc_tensor_arena_t* gemm_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* gemm_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(gemm_symbolic_graph, ccv_nnc_default_compile_params,
		0, 0, TENSOR_SYMBOL_LIST(gemm_z, dgemmx),
		SYMBOLIC_GRAPH_SOURCES(gemm_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(gemm_symbolic_graph),
		&gemm_graph, &gemm_tensor_arena, &gemm_graph_exec_arena);
	ccv_nnc_tensor_t* gemm_x_tensor = ccv_nnc_tensor_from_symbol(gemm_tensor_arena, gemm_x);
	memcpy(gemm_x_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 10);
	const ccv_nnc_tensor_symbol_t dgemmz = ccv_nnc_tensor_symbol_for_backward(gemm_symbolic_graph, gemm_z);
	ccv_nnc_tensor_t* const dgemmz_tensor = ccv_nnc_tensor_from_symbol(gemm_tensor_arena, dgemmz);
	memcpy(dgemmz_tensor->data.f32, dz_tensor->data.f32, sizeof(float) * 10);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	for (i = 0; i < 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* gemm_y_tensor = ccv_nnc_tensor_from_symbol(gemm_tensor_arena, gemm_y);
	for (i = 0; i < 5; i++)
	{
		gemm_y_tensor->data.f32[i * 4] = y_tensor->data.f32[i * 2];
		gemm_y_tensor->data.f32[i * 4 + 1] = -y_tensor->data.f32[i * 2 + 1];
		gemm_y_tensor->data.f32[i * 4 + 2] = y_tensor->data.f32[i * 2 + 1];
		gemm_y_tensor->data.f32[i * 4 + 3] = y_tensor->data.f32[i * 2];
	}
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(gemm_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	ccv_nnc_tensor_t* gemm_z_tensor = ccv_nnc_tensor_from_symbol(gemm_tensor_arena, gemm_z);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, z_tensor->data.f32, gemm_z_tensor->data.f32, 10, 1e-5, "should match as if GEMM");
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* dgemmx_tensor = ccv_nnc_tensor_from_symbol(gemm_tensor_arena, dgemmx);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dgemmx_tensor->data.f32, 10, 1e-5, "should match as if GEMM");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(gemm_symbolic_graph);
	ccv_nnc_tensor_arena_free(gemm_tensor_arena);
	ccv_nnc_graph_exec_arena_free(gemm_graph_exec_arena);
	ccv_nnc_graph_free(gemm_graph);
}

#include "case_main.h"
