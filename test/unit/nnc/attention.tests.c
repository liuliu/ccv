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

TEST_CASE("implement scaled dot product attention with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "v");
	ccv_nnc_tensor_symbol_t qk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "qk");
	ccv_nnc_tensor_symbol_t sq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "sq");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 8), TENSOR_SYMBOL_LIST(q), TENSOR_SYMBOL_LIST(sq), "scaled_q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), TENSOR_SYMBOL_LIST(sq, k), TENSOR_SYMBOL_LIST(qk), "q @ k");
	ccv_nnc_tensor_symbol_t qks = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, qk, DIM_ALLOC(), DIM_ALLOC(128, 1), CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "qks");
	ccv_nnc_tensor_symbol_t s = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "s");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(qks), TENSOR_SYMBOL_LIST(s), "softmax");
	ccv_nnc_tensor_symbol_t sa = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, s, DIM_ALLOC(), DIM_ALLOC(8 * 128 * 128, 128 * 128, 128, 1), CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "sa");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "f");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), TENSOR_SYMBOL_LIST(sa, v), TENSOR_SYMBOL_LIST(r), "final");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, q);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, k);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, v);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 96; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "v");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv), TENSOR_SYMBOL_LIST(br), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	memcpy(bq_tensor->data.f32, q_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bk_tensor->data.f32, k_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bv_tensor->data.f32, v_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, r);
	ccv_nnc_tensor_t* const br_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, br);
	REQUIRE_TENSOR_EQ(r_tensor, br_tensor, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
}

#include "case_main.h"
