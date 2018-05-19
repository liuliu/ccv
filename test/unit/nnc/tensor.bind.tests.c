
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

static int while_4(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 4;
}

TEST_CASE("while z = a * x + b (x <- z) compiled a and b binded to a tensor")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(1), "x");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "y");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, ONE_CPU_TENSOR(1), "z");
	ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWPROD_FORWARD(), TENSOR_SYMBOL_LIST(a, x), TENSOR_SYMBOL_LIST(y), "prod");
	ccv_nnc_graph_exec_symbol_t sum = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(y, b), TENSOR_SYMBOL_LIST(z), "sum");
	ccv_nnc_graph_exec_symbol_t noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_concat(while_graph, sum, noop);
	ccv_nnc_graph_exec_symbol_autogen(while_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_4, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(z, x)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	a_tensor->data.f32[0] = 0.3;
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(1), 0);
	b_tensor->data.f32[0] = 1.1;
	ccv_nnc_symbolic_graph_compile(symbolic_graph,
		TENSOR_BIND_MAP(KV(a, a_tensor), KV(b, b_tensor)), // Binding the tensors.
		0, 0,
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	x_tensor->data.f32[0] = 0.88;
	ccv_nnc_tensor_t* z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	ccv_nnc_graph_exec_t source = ccv_nnc_graph_exec_source(graph_exec_arena);
	ccv_nnc_graph_exec_t destination = ccv_nnc_graph_exec_destination(graph_exec_arena);
	ccv_nnc_graph_run(graph, 0, 0, &source, 1, &destination, 1);
	int i;
	float z_val = 0.88;
	for (i = 0; i < 5; i++)
		z_val = 0.3 * z_val + 1.1;
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], z_val, 1e-6, "z should be equal to a * x + b (5)");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b_tensor);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
