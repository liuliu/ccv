
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
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "x");
	ccv_nnc_symbolic_graph_t* while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(while_graph, CPU_TENSOR_NHWC(32F, 1), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(while_graph, CPU_TENSOR_NHWC(32F, 1), "b");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(while_graph, CPU_TENSOR_NHWC(32F, 1), "y");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(while_graph, CPU_TENSOR_NHWC(32F, 1), "z");
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
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	a_tensor->data.f32[0] = 0.3;
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	b_tensor->data.f32[0] = 1.1;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
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
	ccv_nnc_graph_run(graph, 0, &source, 1, &destination, 1, 0, 0);
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

TEST_CASE("compile graph (a[1] + a[0]) * a where a is a binded tensor")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2), "a");
	const ccv_nnc_tensor_symbol_t a0 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, a, ccv_nnc_no_ofs, DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1), "a[0]");
	const ccv_nnc_tensor_symbol_t a1 = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, a, DIM_ALLOC(1), DIM_ALLOC(2), CPU_TENSOR_NHWC(32F, 1), "a[1]");
	const ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_EWSUM_FORWARD(), TENSOR_SYMBOL_LIST(a0, a1), TENSOR_SYMBOL_LIST(b), "sum");
	const ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MUL_FORWARD(1), TENSOR_SYMBOL_LIST(b, a), TENSOR_SYMBOL_LIST(c), "mul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		0, 0,
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a_tensor->data.f32[0] = 1.1;
	a_tensor->data.f32[1] = 0.4;
	ccv_nnc_tensor_bind_symbol(tensor_arena, a, a_tensor);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	float ct[] = {
		(1.1 + 0.4) * 1.1,
		(1.1 + 0.4) * 0.4,
	};
	ccv_nnc_tensor_t ct_tensor = ccv_nnc_tensor(ct, CPU_TENSOR_NHWC(32F, 2), 0);
	REQUIRE_TENSOR_EQ(c_tensor, &ct_tensor, "c should be equal to [(1.1 + 0.4) * 11, (1.1 + 0.4) * 0.4]");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

TEST_CASE("compile a graph with tensor bindings, verify tensor arena doesn't allocate these tensors")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2), "a");
	const ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1), "b");
	const ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "gemm");
	const ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "d");
	const ccv_nnc_tensor_symbol_t e = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 1), "e");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(), TENSOR_SYMBOL_LIST(c, d), TENSOR_SYMBOL_LIST(e), "mul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const e_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		TENSOR_BIND_MAP(KV(a, a_tensor), KV(d, d_tensor), KV(e, e_tensor)),
		0, 0,
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	const uint64_t size = ccv_nnc_tensor_arena_size(tensor_arena);
	const uint64_t tensor_size = ccv_nnc_tensor_data_size(CPU_TENSOR_NHWC(32F, 1, 2)) + ccv_nnc_tensor_data_size(CPU_TENSOR_NHWC(32F, 1));
	REQUIRE_EQ(size, tensor_size, "tensor arena should only allocate for two symbols");
	a_tensor->data.f32[0] = 1.1;
	a_tensor->data.f32[1] = 11003;
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	b_tensor->data.f32[0] = -1;
	b_tensor->data.f32[1] = -0.12;
	d_tensor->data.f32[0] = 0.5;
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(e_tensor->data.f32[0], -(11003 * 0.12 + 1.1) * 0.5, 1e-2, "result should be equal");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(d_tensor);
	ccv_nnc_tensor_free(e_tensor);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
}

#include "case_main.h"
