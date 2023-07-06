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

TEST_CASE("schedule GPU work on one stream")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a");
	ccv_nnc_tensor_symbol_t const w = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w");
	ccv_nnc_tensor_symbol_t const bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias");
	ccv_nnc_tensor_symbol_t const b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b), "mul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(b),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(ha);
	ccv_nnc_tensor_pin_memory(hw);
	ccv_nnc_tensor_pin_memory(hbias);
	ha->data.f32[0] = 1.4;
	ha->data.f32[1] = 0.2;
	hw->data.f32[0] = 2;
	hw->data.f32[1] = 11;
	hbias->data.f32[0] = 0;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a_tensor, w_tensor, bias_tensor), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_pin_memory(hb);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(hb), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hb->data.f32[0], 1.4 * 2 + 0.2 * 11, 1e-5, "should match simple algebra");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("schedule GPU work on multiple streams")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a");
	ccv_nnc_tensor_symbol_t const w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w1");
	ccv_nnc_tensor_symbol_t const bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias1");
	ccv_nnc_tensor_symbol_t const b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w1, bias1), TENSOR_SYMBOL_LIST(b1), "mul1");
	ccv_nnc_tensor_symbol_t const w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w2");
	ccv_nnc_tensor_symbol_t const bias2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias2");
	ccv_nnc_tensor_symbol_t const b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w2, bias2), TENSOR_SYMBOL_LIST(b2), "mul2");
	ccv_nnc_tensor_symbol_t const w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w3");
	ccv_nnc_tensor_symbol_t const bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias3");
	ccv_nnc_tensor_symbol_t const b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w3, bias3), TENSOR_SYMBOL_LIST(b3), "mul3");
	ccv_nnc_tensor_symbol_t const biasc = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "biasc");
	ccv_nnc_tensor_symbol_t const c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b1, b2, biasc), TENSOR_SYMBOL_LIST(c), "mulc");
	ccv_nnc_tensor_symbol_t const biasd = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "biasd");
	ccv_nnc_tensor_symbol_t const d = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "d");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(c, b3, biasd), TENSOR_SYMBOL_LIST(d), "muld");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(d),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(ha);
	ccv_nnc_tensor_pin_memory(hw1);
	ccv_nnc_tensor_pin_memory(hbias1);
	ha->data.f32[0] = 1.4;
	ha->data.f32[1] = 0.2;
	hw1->data.f32[0] = 2;
	hw1->data.f32[1] = 11;
	hbias1->data.f32[0] = 0;
	ccv_nnc_tensor_t* const hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw2);
	ccv_nnc_tensor_pin_memory(hbias2);
	hw2->data.f32[0] = 1;
	hw2->data.f32[1] = 2.2;
	hbias2->data.f32[0] = 1;
	ccv_nnc_tensor_t* const hw3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw3);
	ccv_nnc_tensor_pin_memory(hbias3);
	hw3->data.f32[0] = 0.5;
	hw3->data.f32[1] = 1.5;
	hbias3->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const hbiasc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hbiasc);
	hbiasc->data.f32[0] = 0.2;
	ccv_nnc_tensor_t* const hbiasd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hbiasd);
	hbiasd->data.f32[0] = 0.3;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const w1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w1);
	ccv_nnc_tensor_t* const bias1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias1);
	ccv_nnc_tensor_t* const w2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w2);
	ccv_nnc_tensor_t* const bias2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias2);
	ccv_nnc_tensor_t* const w3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w3);
	ccv_nnc_tensor_t* const bias3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias3);
	ccv_nnc_tensor_t* const biasc_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, biasc);
	ccv_nnc_tensor_t* const biasd_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, biasd);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw1, hbias1, hw2, hbias2, hw3, hbias3, hbiasc, hbiasd), TENSOR_LIST(a_tensor, w1_tensor, bias1_tensor, w2_tensor, bias2_tensor, w3_tensor, bias3_tensor, biasc_tensor, biasd_tensor), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_pin_memory(hd);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d_tensor), TENSOR_LIST(hd), 0);
	const float b1v = 1.4 * 2 + 0.2 * 11;
	const float b2v = 1.4 * 1 + 0.2 * 2.2 + 1;
	const float b3v = 1.4 * 0.5 + 0.2 * 1.5 + 0.5;
	const float cv = b1v * b2v + 0.2;
	const float dv = cv * b3v + 0.3;
	REQUIRE_EQ_WITH_TOLERANCE(hd->data.f32[0], dv, 1e-5, "should match simple algebra");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw1);
	ccv_nnc_tensor_free(hbias1);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
	ccv_nnc_tensor_free(hw3);
	ccv_nnc_tensor_free(hbias3);
	ccv_nnc_tensor_free(hbiasc);
	ccv_nnc_tensor_free(hbiasd);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

static int while_5(ccv_nnc_tensor_t* const* const inputs, const int input_size, const void* const data)
{
	return inputs[0]->data.i64[0] < 5;
}

TEST_CASE("schedule GPU work with while loop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t const a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a");
	ccv_nnc_tensor_symbol_t const w1 = ccv_nnc_tensor_symbol_new(while_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2), "w1");
	ccv_nnc_tensor_symbol_t const bias1 = ccv_nnc_tensor_symbol_new(while_graph, GPU_TENSOR_NHWC(000, 32F, 2), "bias1");
	ccv_nnc_tensor_symbol_t const b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b1");
	ccv_nnc_graph_exec_symbol_t const noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t const mul1 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w1, bias1), TENSOR_SYMBOL_LIST(b1), "mul1");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, mul1);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(b1, a)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(mul1));
	ccv_nnc_tensor_symbol_t const w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w2");
	ccv_nnc_tensor_symbol_t const bias2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias2");
	ccv_nnc_tensor_symbol_t const b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w2, bias2), TENSOR_SYMBOL_LIST(b2), "mul2");
	ccv_nnc_tensor_symbol_t const w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w3");
	ccv_nnc_tensor_symbol_t const bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias3");
	ccv_nnc_tensor_symbol_t const b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b1, w3, bias3), TENSOR_SYMBOL_LIST(b3), "mul3");
	ccv_nnc_tensor_symbol_t const biasc = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "biasc");
	ccv_nnc_tensor_symbol_t const c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b2, b3, biasc), TENSOR_SYMBOL_LIST(c), "mulc");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(c),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const hbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_pin_memory(ha);
	ccv_nnc_tensor_pin_memory(hw1);
	ccv_nnc_tensor_pin_memory(hbias1);
	ha->data.f32[0] = 1.4;
	ha->data.f32[1] = 0.2;
	hw1->data.f32[0] = 1.1;
	hw1->data.f32[1] = 2.2;
	hw1->data.f32[2] = 1;
	hw1->data.f32[3] = 2;
	hbias1->data.f32[0] = 0;
	hbias1->data.f32[1] = 0;
	ccv_nnc_tensor_t* const hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw2);
	ccv_nnc_tensor_pin_memory(hbias2);
	hw2->data.f32[0] = 0.6;
	hw2->data.f32[1] = 3;
	hbias2->data.f32[0] = 0.4;
	ccv_nnc_tensor_t* const hw3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw3);
	ccv_nnc_tensor_pin_memory(hbias3);
	hw3->data.f32[0] = 0.2;
	hw3->data.f32[1] = 0.3;
	hbias3->data.f32[0] = 1;
	ccv_nnc_tensor_t* const hbiasc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hbiasc);
	hbiasc->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const w1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w1);
	ccv_nnc_tensor_t* const bias1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias1);
	ccv_nnc_tensor_t* const w2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w2);
	ccv_nnc_tensor_t* const bias2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias2);
	ccv_nnc_tensor_t* const w3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w3);
	ccv_nnc_tensor_t* const bias3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias3);
	ccv_nnc_tensor_t* const biasc_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, biasc);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw1, hbias1, hw2, hbias2, hw3, hbias3, hbiasc), TENSOR_LIST(a_tensor, w1_tensor, bias1_tensor, w2_tensor, bias2_tensor, w3_tensor, bias3_tensor, biasc_tensor), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_graph_default_stream(graph);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_pin_memory(hc);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(hc), 0);
	float av0 = 1.4;
	float av1 = 0.2;
	int i;
	for (i = 0; i < 5; i++)
	{
		const float b0 = av0 * 1.1 + av1 * 2.2;
		const float b1 = av0 * 1 + av1 * 2;
		av0 = b0;
		av1 = b1;
	}
	const float b2v = 1.4 * 0.6 + 0.2 * 3 + 0.4;
	const float b3v = av0 * 0.2 + av1 * 0.3 + 1;
	const float cv = b2v * b3v + 0.5;
	REQUIRE_EQ_WITH_TOLERANCE(hc->data.f32[0], cv, 1e-2, "should match simple algebra");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw1);
	ccv_nnc_tensor_free(hbias1);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
	ccv_nnc_tensor_free(hw3);
	ccv_nnc_tensor_free(hbias3);
	ccv_nnc_tensor_free(hbiasc);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

static int case_of_0(ccv_nnc_tensor_t* const *const inputs, const int input_size, const void* const data)
{
	return 0;
}

TEST_CASE("schedule GPU work with case..of")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a");
	ccv_nnc_tensor_symbol_t const b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b");
	ccv_nnc_graph_exec_symbol_t const case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_MAP(KV(a, b)), "case..of");
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, case_of_0, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b0");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_0, 0, TENSOR_SYMBOL_MAP(KV(b0, b)));
	ccv_nnc_tensor_symbol_t const w = ccv_nnc_tensor_symbol_new(symbolic_graph_0, GPU_TENSOR_NHWC(000, 32F, 2, 2), "w");
	ccv_nnc_tensor_symbol_t const bias = ccv_nnc_tensor_symbol_new(symbolic_graph_0, GPU_TENSOR_NHWC(000, 32F, 2), "bias");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_0, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w, bias), TENSOR_SYMBOL_LIST(b0), "mul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(b),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_pin_memory(ha);
	ccv_nnc_tensor_pin_memory(hw);
	ccv_nnc_tensor_pin_memory(hbias);
	ha->data.f32[0] = 1.4;
	ha->data.f32[1] = 0.2;
	hw->data.f32[0] = 2;
	hw->data.f32[1] = 11;
	hw->data.f32[2] = 1;
	hw->data.f32[3] = 2;
	hbias->data.f32[0] = 0;
	hbias->data.f32[1] = 0;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a_tensor, w_tensor, bias_tensor), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_pin_memory(hb);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(hb), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hb->data.f32[0], 1.4 * 2 + 0.2 * 11, 1e-5, "should match simple algebra");
	REQUIRE_EQ_WITH_TOLERANCE(hb->data.f32[1], 1.4 + 0.2 * 2, 1e-5, "should match simple algebra");
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("schedule GPU work with both while loop and case..of")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_t* const while_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_symbolic_graph_while(symbolic_graph, CCV_NNC_GRAPH_FORWARD, while_graph, "while 1..5");
	ccv_nnc_tensor_symbol_t const a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a");
	ccv_nnc_tensor_symbol_t const w1 = ccv_nnc_tensor_symbol_new(while_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2), "w1");
	ccv_nnc_tensor_symbol_t const bias1 = ccv_nnc_tensor_symbol_new(while_graph, GPU_TENSOR_NHWC(000, 32F, 2), "bias1");
	ccv_nnc_tensor_symbol_t const b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b1");
	ccv_nnc_graph_exec_symbol_t const noop = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_NOOP(), 0, 0, 0, 0, "noop");
	ccv_nnc_graph_exec_symbol_t const mul1 = ccv_nnc_graph_exec_symbol_new(while_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w1, bias1), TENSOR_SYMBOL_LIST(b1), "mul1");
	ccv_nnc_graph_exec_symbol_concat(while_graph, noop, mul1);
	ccv_nnc_symbolic_graph_set_while_expr(while_graph, while_5, 0, TENSOR_SYMBOL_LIST(ccv_nnc_tensor_symbol_for_while_count(while_graph)), GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_carry_overs(while_graph, TENSOR_SYMBOL_MAP(KV(b1, a)));
	ccv_nnc_symbolic_graph_set_sources(while_graph, GRAPH_EXEC_SYMBOL_LIST(noop));
	ccv_nnc_symbolic_graph_set_destinations(while_graph, GRAPH_EXEC_SYMBOL_LIST(mul1));
	ccv_nnc_tensor_symbol_t const b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b2");
	ccv_nnc_graph_exec_symbol_t const case_of = ccv_nnc_symbolic_graph_case_of_new(symbolic_graph, CCV_NNC_GRAPH_FORWARD, TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_MAP(KV(a, b2)), "case..of");
	ccv_nnc_symbolic_graph_set_case_of_expr(symbolic_graph, case_of, case_of_0, 0);
	ccv_nnc_symbolic_graph_t* const symbolic_graph_0 = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const b20 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "b20");
	ccv_nnc_symbolic_graph_set_case_of(symbolic_graph, case_of, symbolic_graph_0, 0, TENSOR_SYMBOL_MAP(KV(b20, b2)));
	ccv_nnc_tensor_symbol_t const w2 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, GPU_TENSOR_NHWC(000, 32F, 2, 2), "w2");
	ccv_nnc_tensor_symbol_t const bias2 = ccv_nnc_tensor_symbol_new(symbolic_graph_0, GPU_TENSOR_NHWC(000, 32F, 2), "bias2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph_0, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a, w2, bias2), TENSOR_SYMBOL_LIST(b20), "mul2");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph_0, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t const w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w3");
	ccv_nnc_tensor_symbol_t const bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias3");
	ccv_nnc_tensor_symbol_t const b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b3");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b2, w3, bias3), TENSOR_SYMBOL_LIST(b3), "mul3");
	ccv_nnc_tensor_symbol_t const w4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w4");
	ccv_nnc_tensor_symbol_t const bias4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias4");
	ccv_nnc_tensor_symbol_t const b4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b4");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b1, w4, bias4), TENSOR_SYMBOL_LIST(b4), "mul4");
	ccv_nnc_tensor_symbol_t const biasc = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "biasc");
	ccv_nnc_tensor_symbol_t const c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(b3, b4, biasc), TENSOR_SYMBOL_LIST(c), "mulc");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(c),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const hbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_pin_memory(ha);
	ccv_nnc_tensor_pin_memory(hw1);
	ccv_nnc_tensor_pin_memory(hbias1);
	ha->data.f32[0] = 1.4;
	ha->data.f32[1] = 0.2;
	hw1->data.f32[0] = 1.1;
	hw1->data.f32[1] = 2.2;
	hw1->data.f32[2] = 1;
	hw1->data.f32[3] = 2;
	hbias1->data.f32[0] = 0;
	hbias1->data.f32[1] = 0;
	ccv_nnc_tensor_t* const hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_pin_memory(hw2);
	ccv_nnc_tensor_pin_memory(hbias2);
	hw2->data.f32[0] = 0.1;
	hw2->data.f32[1] = 0.2;
	hw2->data.f32[2] = 1.2;
	hw2->data.f32[3] = 1.1;
	hbias2->data.f32[0] = 1;
	hbias2->data.f32[1] = 0;
	ccv_nnc_tensor_t* const hw3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw3);
	ccv_nnc_tensor_pin_memory(hbias3);
	hw3->data.f32[0] = 0.6;
	hw3->data.f32[1] = 3;
	hbias3->data.f32[0] = 0.4;
	ccv_nnc_tensor_t* const hw4 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias4 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hw4);
	ccv_nnc_tensor_pin_memory(hbias4);
	hw4->data.f32[0] = 0.2;
	hw4->data.f32[1] = 0.3;
	hbias4->data.f32[0] = 1;
	ccv_nnc_tensor_t* const hbiasc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_pin_memory(hbiasc);
	hbiasc->data.f32[0] = 0.5;
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const w1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w1);
	ccv_nnc_tensor_t* const bias1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias1);
	ccv_nnc_tensor_t* const w2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w2);
	ccv_nnc_tensor_t* const bias2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias2);
	ccv_nnc_tensor_t* const w3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w3);
	ccv_nnc_tensor_t* const bias3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias3);
	ccv_nnc_tensor_t* const w4_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w4);
	ccv_nnc_tensor_t* const bias4_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias4);
	ccv_nnc_tensor_t* const biasc_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, biasc);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw1, hbias1, hw2, hbias2, hw3, hbias3, hw4, hbias4, hbiasc), TENSOR_LIST(a_tensor, w1_tensor, bias1_tensor, w2_tensor, bias2_tensor, w3_tensor, bias3_tensor, w4_tensor, bias4_tensor, biasc_tensor), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_pin_memory(hc);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(hc), 0);
	float av0 = 1.4;
	float av1 = 0.2;
	int i;
	for (i = 0; i < 5; i++)
	{
		const float b0 = av0 * 1.1 + av1 * 2.2;
		const float b1 = av0 * 1 + av1 * 2;
		av0 = b0;
		av1 = b1;
	}
	const float b2v0 = 1.4 * 0.1 + 0.2 * 0.2 + 1;
	const float b2v1 = 1.4 * 1.2 + 0.2 * 1.1;
	const float b3v = b2v0 * 0.6 + b2v1 * 3 + 0.4;
	const float b4v = av0 * 0.2 + av1 * 0.3 + 1;
	const float cv = b3v * b4v + 0.5;
	REQUIRE_EQ_WITH_TOLERANCE(hc->data.f32[0], cv, 1e-2, "should match simple algebra");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw1);
	ccv_nnc_tensor_free(hbias1);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
	ccv_nnc_tensor_free(hw3);
	ccv_nnc_tensor_free(hbias3);
	ccv_nnc_tensor_free(hw4);
	ccv_nnc_tensor_free(hbias4);
	ccv_nnc_tensor_free(hbiasc);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
}

TEST_CASE("partial schedule work, one for device 0 and one for device 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) > 1 &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const a0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a0");
	ccv_nnc_tensor_symbol_t const w0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w0");
	ccv_nnc_tensor_symbol_t const bias0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias0");
	ccv_nnc_tensor_symbol_t const b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b0");
	ccv_nnc_graph_exec_symbol_t const src0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a0, w0, bias0), TENSOR_SYMBOL_LIST(b0), "mul0");
	ccv_nnc_tensor_symbol_t const c0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "c0");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.1), TENSOR_SYMBOL_LIST(b0), TENSOR_SYMBOL_LIST(c0), "scale00");
	ccv_nnc_tensor_symbol_t const d0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "d0");
	ccv_nnc_graph_exec_symbol_t const dest0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.9), TENSOR_SYMBOL_LIST(c0), TENSOR_SYMBOL_LIST(d0), "scale01");
	ccv_nnc_tensor_symbol_t const a1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 2), "a1");
	ccv_nnc_tensor_symbol_t const w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 2), "w1");
	ccv_nnc_tensor_symbol_t const bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1), "bias1");
	ccv_nnc_tensor_symbol_t const b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "b1");
	ccv_nnc_graph_exec_symbol_t const src1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a1, w1, bias1), TENSOR_SYMBOL_LIST(b1), "mul1");
	ccv_nnc_tensor_symbol_t const c1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "c1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.2), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "scale10");
	ccv_nnc_tensor_symbol_t const d1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "d1");
	ccv_nnc_graph_exec_symbol_t const dest1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.8), TENSOR_SYMBOL_LIST(c1), TENSOR_SYMBOL_LIST(d1), "scale11");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(d0, d1),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const hbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const hd0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const hd1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ha0->data.f32[0] = 0.4;
	ha0->data.f32[1] = 1.2;
	hw0->data.f32[0] = 3;
	hw0->data.f32[1] = 2;
	hbias0->data.f32[0] = -1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha0, hw0, hbias0),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, a0), ccv_nnc_tensor_from_symbol(tensor_arena, w0), ccv_nnc_tensor_from_symbol(tensor_arena, bias0)),
		0);
	ha1->data.f32[0] = 1.3;
	ha1->data.f32[1] = 0.5;
	hw1->data.f32[0] = -3;
	hw1->data.f32[1] = 5;
	hbias1->data.f32[0] = 0.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha1, hw1, hbias1),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, a1), ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, bias1)),
		0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_graph_run_with_schedule(graph, 0, 0, 0, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d0)),
		TENSOR_LIST(hd0),
		0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		TENSOR_LIST(hd1),
		0);
	REQUIRE_EQ_WITH_TOLERANCE(hd0->data.f32[0], (0.4 * 3 + 1.2 * 2 - 1) * 1.1 * 0.9, 1e-5, "result should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hd1->data.f32[0], (-1.3 * 3 + 0.5 * 5 + 0.2) * 1.2 * 0.8, 1e-5, "result should be equal");
	hd0->data.f32[0] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hd0),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d0)),
		0);
	hd1->data.f32[0] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hd1),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		0);
	// schedule device 0
	ccv_nnc_graph_static_schedule_t* const schedule0 = ccv_nnc_graph_static_schedule_new(graph, CCV_STREAM_CONTEXT_GPU, 0,
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src0)),
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest0)));
	ccv_nnc_graph_run_with_schedule(graph, 0, schedule0, 0, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d0)),
		TENSOR_LIST(hd0),
		0);
	REQUIRE_EQ_WITH_TOLERANCE(hd0->data.f32[0], (0.4 * 3 + 1.2 * 2 - 1) * 1.1 * 0.9, 1e-5, "result should be equal");
	// schedule device 1
	ccv_nnc_graph_static_schedule_t* const schedule1 = ccv_nnc_graph_static_schedule_new(graph, CCV_STREAM_CONTEXT_GPU, 0,
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src1)),
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest1)));
	ccv_nnc_graph_run_with_schedule(graph, 0, schedule1, 0, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		TENSOR_LIST(hd1),
		0);
	REQUIRE_EQ_WITH_TOLERANCE(hd1->data.f32[0], (-1.3 * 3 + 0.5 * 5 + 0.2) * 1.2 * 0.8, 1e-5, "result should be equal");
	hd0->data.f32[0] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hd0),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d0)),
		0);
	hd1->data.f32[0] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(hd1),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		0);
	// custom schedule again with both device 0 and device 1.
	ccv_nnc_graph_static_schedule_t* const schedule01 = ccv_nnc_graph_static_schedule_new(graph, CCV_STREAM_CONTEXT_GPU, 0,
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src0), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src1)),
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest0), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest1)));
	ccv_nnc_graph_run_with_schedule(graph, 0, schedule01, 0, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d0)),
		TENSOR_LIST(hd0),
		0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		TENSOR_LIST(hd1),
		0);
	REQUIRE_EQ_WITH_TOLERANCE(hd0->data.f32[0], (0.4 * 3 + 1.2 * 2 - 1) * 1.1 * 0.9, 1e-5, "result should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hd1->data.f32[0], (-1.3 * 3 + 0.5 * 5 + 0.2) * 1.2 * 0.8, 1e-5, "result should be equal");
	ccv_nnc_graph_static_schedule_free(schedule0);
	ccv_nnc_graph_static_schedule_free(schedule1);
	ccv_nnc_graph_static_schedule_free(schedule01);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(ha0);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(hw0);
	ccv_nnc_tensor_free(hw1);
	ccv_nnc_tensor_free(hbias0);
	ccv_nnc_tensor_free(hbias1);
	ccv_nnc_tensor_free(hd0);
	ccv_nnc_tensor_free(hd1);
}
TEST_CASE("partial schedule on both device 0 and then join device 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) > 1 &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t const a0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "a0");
	ccv_nnc_tensor_symbol_t const w0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2), "w0");
	ccv_nnc_tensor_symbol_t const bias0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1), "bias0");
	ccv_nnc_tensor_symbol_t const b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 1), "b0");
	ccv_nnc_graph_exec_symbol_t const src0 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a0, w0, bias0), TENSOR_SYMBOL_LIST(b0), "mul0");
	ccv_nnc_graph_exec_symbol_t const dest0 = src0;
	ccv_nnc_tensor_symbol_t const a1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 2), "a1");
	ccv_nnc_tensor_symbol_t const w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 2), "w1");
	ccv_nnc_tensor_symbol_t const bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1), "bias1");
	ccv_nnc_tensor_symbol_t const b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "b1");
	ccv_nnc_graph_exec_symbol_t const src1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(a1, w1, bias1), TENSOR_SYMBOL_LIST(b1), "mul1");
	ccv_nnc_tensor_symbol_t const c1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "c1");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.2), TENSOR_SYMBOL_LIST(b1), TENSOR_SYMBOL_LIST(c1), "scale10");
	ccv_nnc_tensor_symbol_t const d1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(001, 32F, 1, 1), "d1");
	ccv_nnc_graph_exec_symbol_t const dest1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(0.8), TENSOR_SYMBOL_LIST(c1), TENSOR_SYMBOL_LIST(d1), "scale11");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_NOOP(), TENSOR_SYMBOL_LIST(b0, d1), TENSOR_SYMBOL_LIST(), "noop");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params,
		0, 0,
		TENSOR_SYMBOL_LIST(b0, d1),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_set_default_static_schedule(graph, CCV_STREAM_CONTEXT_GPU, 0);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const hbias0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const hbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const hb0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const hd1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ha0->data.f32[0] = 0.4;
	ha0->data.f32[1] = 1.2;
	hw0->data.f32[0] = 3;
	hw0->data.f32[1] = 2;
	hbias0->data.f32[0] = -1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha0, hw0, hbias0),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, a0), ccv_nnc_tensor_from_symbol(tensor_arena, w0), ccv_nnc_tensor_from_symbol(tensor_arena, bias0)),
		0);
	ha1->data.f32[0] = 1.3;
	ha1->data.f32[1] = 0.5;
	hw1->data.f32[0] = -3;
	hw1->data.f32[1] = 5;
	hbias1->data.f32[0] = 0.2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ha1, hw1, hbias1),
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, a1), ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, bias1)),
		0);
	ccv_nnc_stream_context_t* const stream0 = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU | CCV_COMPUTE_DEVICE_000);
	ccv_nnc_stream_signal_t* const signal0 = ccv_nnc_stream_signal_new(CCV_STREAM_CONTEXT_GPU | CCV_COMPUTE_DEVICE_000);
	ccv_nnc_stream_context_t* const stream1 = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU | CCV_COMPUTE_DEVICE_001);
	// custom schedule again with both device 0 and device 1.
	ccv_nnc_graph_static_schedule_t* const schedule01 = ccv_nnc_graph_static_schedule_new(graph, CCV_STREAM_CONTEXT_GPU, 0,
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src0), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, src1)),
		GRAPH_EXEC_LIST(ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest0), ccv_nnc_graph_exec_from_symbol(graph_exec_arena, dest1)));
	ccv_nnc_graph_run_with_schedule(graph, 0, schedule01, 0, stream0);
	ccv_nnc_stream_context_emit_signal(stream0, signal0);
	ccv_nnc_stream_context_wait_signal(stream1, signal0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, d1)),
		TENSOR_LIST(hd1),
		stream1);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
		TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, b0)),
		TENSOR_LIST(hb0),
		stream0);
	ccv_nnc_stream_context_wait(stream1);
	ccv_nnc_stream_context_wait(stream0);
	REQUIRE_EQ_WITH_TOLERANCE(hb0->data.f32[0], 0.4 * 3 + 1.2 * 2 - 1, 1e-5, "result should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hd1->data.f32[0], (-1.3 * 3 + 0.5 * 5 + 0.2) * 1.2 * 0.8, 1e-5, "result should be equal");
	ccv_nnc_graph_static_schedule_free(schedule01);
	ccv_nnc_stream_context_free(stream0);
	ccv_nnc_stream_signal_free(signal0);
	ccv_nnc_stream_context_free(stream1);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(ha0);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(hw0);
	ccv_nnc_tensor_free(hw1);
	ccv_nnc_tensor_free(hbias0);
	ccv_nnc_tensor_free(hbias1);
	ccv_nnc_tensor_free(hb0);
	ccv_nnc_tensor_free(hd1);
}

#include "case_main.h"
