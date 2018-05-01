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

static int _case_of(ccv_nnc_tensor_t* const *const inputs, const int input_size, const void *const case_of_data)
{
	if (inputs[0]->data.f32[0] < 0)
		return -1;
	else if (inputs[0]->data.f32[0] < 1)
		return 0;
	else if (inputs[0]->data.f32[0] < 2)
		return 1;
	else
		return 2;
}

TEST_CASE("compile symbolic graph with case..of")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(2), "y");
	ccv_nnc_graph_exec_symbol_t case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_MAP(KV(x, y)), "case of");
	ccv_nnc_symbolic_graph_t* const case_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y0 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(2), "y0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_0, 0, TENSOR_SYMBOL_MAP(KV(y0, y)));
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, _case_of, 0);
	ccv_nnc_tensor_symbol_t w01 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(1, 2), "w01");
	ccv_nnc_tensor_symbol_t b01 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(1), "b01");
	ccv_nnc_tensor_symbol_t z0 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(1), "z0");
	ccv_nnc_graph_exec_symbol_new(case_0, CMD_GEMM_FORWARD(1), TENSOR_SYMBOL_LIST(x, w01, b01), TENSOR_SYMBOL_LIST(z0), "mul01");
	ccv_nnc_tensor_symbol_t w02 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(2, 1), "w02");
	ccv_nnc_tensor_symbol_t b02 = ccv_nnc_tensor_symbol_new(case_0, ONE_CPU_TENSOR(2), "b02");
	ccv_nnc_graph_exec_symbol_new(case_0, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(z0, w02, b02), TENSOR_SYMBOL_LIST(y0), "mul02");
	ccv_nnc_graph_exec_symbol_autogen(case_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_1 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(2), "y0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_1, 1, TENSOR_SYMBOL_MAP(KV(y1, y)));
	ccv_nnc_tensor_symbol_t w11 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(16, 2), "w11");
	ccv_nnc_tensor_symbol_t b11 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(16), "b11");
	ccv_nnc_tensor_symbol_t z1 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(16), "z1");
	ccv_nnc_graph_exec_symbol_new(case_1, CMD_GEMM_FORWARD(16), TENSOR_SYMBOL_LIST(x, w11, b11), TENSOR_SYMBOL_LIST(z1), "mul11");
	ccv_nnc_tensor_symbol_t w12 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(2, 16), "w12");
	ccv_nnc_tensor_symbol_t b12 = ccv_nnc_tensor_symbol_new(case_1, ONE_CPU_TENSOR(2), "b12");
	ccv_nnc_graph_exec_symbol_new(case_1, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(z1, w12, b12), TENSOR_SYMBOL_LIST(y1), "mul12");
	ccv_nnc_graph_exec_symbol_autogen(case_1, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_t* const case_2 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(2), "y2");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, case_2, 2, TENSOR_SYMBOL_MAP(KV(y2, y)));
	ccv_nnc_tensor_symbol_t w21 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(8, 2), "w21");
	ccv_nnc_tensor_symbol_t b21 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(8), "b21");
	ccv_nnc_tensor_symbol_t z2 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(8), "z2");
	ccv_nnc_graph_exec_symbol_new(case_2, CMD_GEMM_FORWARD(8), TENSOR_SYMBOL_LIST(x, w21, b21), TENSOR_SYMBOL_LIST(z2), "mul21");
	ccv_nnc_tensor_symbol_t w22 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(2, 8), "w22");
	ccv_nnc_tensor_symbol_t b22 = ccv_nnc_tensor_symbol_new(case_2, ONE_CPU_TENSOR(2), "b22");
	ccv_nnc_graph_exec_symbol_new(case_2, CMD_GEMM_FORWARD(2), TENSOR_SYMBOL_LIST(z2, w22, b22), TENSOR_SYMBOL_LIST(y2), "mul22");
	ccv_nnc_graph_exec_symbol_autogen(case_2, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);

	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = -1;
	x_tensor->data.f32[1] = 10;
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2), 0);
	yt->data.f32[0] = -1;
	yt->data.f32[1] = 10;
	ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	ccv_nnc_tensor_t* y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_TENSOR_EQ(y_tensor, yt, "skip any computation");

	x_tensor->data.f32[0] = 0.5;
	x_tensor->data.f32[1] = 1;
	ccv_nnc_tensor_t* w01_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w01);
	w01_tensor->data.f32[0] = 0.1;
	w01_tensor->data.f32[1] = -0.28;
	ccv_nnc_tensor_t* b01_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b01);
	b01_tensor->data.f32[0] = 1;
	ccv_nnc_tensor_t* w02_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w02);
	w02_tensor->data.f32[0] = 0.56;
	w02_tensor->data.f32[1] = 0.81;
	ccv_nnc_tensor_t* b02_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b02);
	b02_tensor->data.f32[0] = 0.33;
	b02_tensor->data.f32[1] = 0.12;
	ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	float z0val = 0.5 * 0.1 - 0.28 + 1;
	yt->data.f32[0] = z0val * 0.56 + 0.33;
	yt->data.f32[1] = z0val * 0.81 + 0.12;
	REQUIRE_TENSOR_EQ(y_tensor, yt, "(x * w01 + b01) * w02 + b02");

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	x_tensor->data.f32[0] = 1.11;
	x_tensor->data.f32[1] = 2.56;
	ccv_nnc_tensor_t* w11_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w11);
	float w11t[32];
	for (i = 0; i < 32; i++)
		w11t[i] = w11_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* b11_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b11);
	float b11t[16];
	for (i = 0; i < 16; i++)
		b11t[i] = b11_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* w12_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w12);
	float w12t[32];
	for (i = 0; i < 32; i++)
		w12t[i] = w12_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* b12_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b12);
	b12_tensor->data.f32[0] = 0.32;
	b12_tensor->data.f32[1] = -0.2;
	ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	yt->data.f32[0] = 0.32;
	yt->data.f32[1] = -0.2;
	for (i = 0; i < 16; i++)
	{
		float z1val = 1.11 * w11t[i * 2] + 2.56 * w11t[i * 2 + 1] + b11t[i];
		yt->data.f32[0] += z1val * w12t[i];
		yt->data.f32[1] += z1val * w12t[i + 16];
	}
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_TENSOR_EQ(y_tensor, yt, "(x * w11 + b11) * w12 + b12");

	x_tensor->data.f32[0] = 4.5;
	x_tensor->data.f32[1] = 3.33;
	ccv_nnc_tensor_t* w21_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w21);
	for (i = 0; i < 16; i++)
		w11t[i] = w21_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* b21_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b21);
	for (i = 0; i < 8; i++)
		b11t[i] = b21_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* w22_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w22);
	for (i = 0; i < 16; i++)
		w12t[i] = w22_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* b22_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b22);
	b22_tensor->data.f32[0] = 3.32;
	b22_tensor->data.f32[1] = -2;
	ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
	yt->data.f32[0] = 3.32;
	yt->data.f32[1] = -2;
	for (i = 0; i < 8; i++)
	{
		float z2val = 4.5 * w11t[i * 2] + 3.33 * w11t[i * 2 + 1] + b11t[i];
		yt->data.f32[0] += z2val * w12t[i];
		yt->data.f32[1] += z2val * w12t[i + 8];
	}
	y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	REQUIRE_TENSOR_EQ(y_tensor, yt, "(x * w21 + b21) * w22 + b22");

	ccv_nnc_tensor_free(yt);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

#include "case_main.h"
