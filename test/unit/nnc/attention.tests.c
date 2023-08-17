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
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t tq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t tk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t tv = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "v");
	ccv_nnc_tensor_symbol_t qk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "qk");
	ccv_nnc_tensor_symbol_t sq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "sq");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(q), TENSOR_SYMBOL_LIST(tq), "transpose q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(k), TENSOR_SYMBOL_LIST(tk), "transpose k");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(tv), "transpose v");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 8), TENSOR_SYMBOL_LIST(tq), TENSOR_SYMBOL_LIST(sq), "scaled_q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), TENSOR_SYMBOL_LIST(sq, tk), TENSOR_SYMBOL_LIST(qk), "q @ k");
	ccv_nnc_tensor_symbol_t qks = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, qk, DIM_ALLOC(), DIM_ALLOC(128, 1), CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "qks");
	ccv_nnc_tensor_symbol_t s = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "s");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(qks), TENSOR_SYMBOL_LIST(s), "softmax");
	ccv_nnc_tensor_symbol_t sa = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, s, DIM_ALLOC(), DIM_ALLOC(8 * 128 * 128, 128 * 128, 128, 1), CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "sa");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "f");
	ccv_nnc_tensor_symbol_t tr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "f");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), TENSOR_SYMBOL_LIST(sa, tv), TENSOR_SYMBOL_LIST(r), "final");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(r), TENSOR_SYMBOL_LIST(tr), "final");
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
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "r");
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
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, tr);
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

TEST_CASE("implement scaled dot product attention + unify head output with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t tq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "q");
	ccv_nnc_tensor_symbol_t tk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "k");
	ccv_nnc_tensor_symbol_t tv = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "v");
	ccv_nnc_tensor_symbol_t qk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "qk");
	ccv_nnc_tensor_symbol_t sq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "sq");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(q), TENSOR_SYMBOL_LIST(tq), "transpose q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(k), TENSOR_SYMBOL_LIST(tk), "transpose k");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(tv), "transpose v");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 8), TENSOR_SYMBOL_LIST(tq), TENSOR_SYMBOL_LIST(sq), "scaled_q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), TENSOR_SYMBOL_LIST(sq, tk), TENSOR_SYMBOL_LIST(qk), "q @ k");
	ccv_nnc_tensor_symbol_t qks = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, qk, DIM_ALLOC(), DIM_ALLOC(128, 1), CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "qks");
	ccv_nnc_tensor_symbol_t s = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "s");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(qks), TENSOR_SYMBOL_LIST(s), "softmax");
	ccv_nnc_tensor_symbol_t sa = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, s, DIM_ALLOC(), DIM_ALLOC(8 * 128 * 128, 128 * 128, 128, 1), CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "sa");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), TENSOR_SYMBOL_LIST(sa, tv), TENSOR_SYMBOL_LIST(c), "c");
	ccv_nnc_tensor_symbol_t ct = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "ct");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(ct), "ct");
	ccv_nnc_tensor_symbol_t cta = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, ct, DIM_ALLOC(), DIM_ALLOC(128 * 768, 768, 1), CPU_TENSOR_NHWC(32F, 32, 128, 768), "ct");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 768, 768), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 768), "bias");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8 * 96), "r");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), TENSOR_SYMBOL_LIST(cta, w, bias), TENSOR_SYMBOL_LIST(r), "final");
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
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bias);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 96; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 768 * 768; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 768; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t bw = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 768, 768), "w");
	ccv_nnc_tensor_symbol_t bbias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 768), "bias");
	ccv_nnc_tensor_symbol_t bc = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "c");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 768), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv, NO_TENSOR_SYMBOL, bw, bbias), TENSOR_SYMBOL_LIST(br, NO_TENSOR_SYMBOL, bc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	ccv_nnc_tensor_t* const bw_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bw);
	ccv_nnc_tensor_t* const bbias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bbias);
	memcpy(bq_tensor->data.f32, q_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bk_tensor->data.f32, k_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bv_tensor->data.f32, v_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	memcpy(bw_tensor->data.f32, w_tensor->data.f32, sizeof(float) * 768 * 768);
	memcpy(bbias_tensor->data.f32, bias_tensor->data.f32, sizeof(float) * 768);
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

TEST_CASE("run scaled dot product attention with cnnp model")
{
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv), TENSOR_SYMBOL_LIST(br), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bq_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bk_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 96; i++)
		bv_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), 0);
	memcpy(q_tensor->data.f32, bq_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(k_tensor->data.f32, bk_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(v_tensor->data.f32, bv_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const br_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, br);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_new(0, br_tensor->info, 0);
	ccv_cnnp_model_t* scaled_dot_product_attention = ccv_cnnp_scaled_dot_product_attention(1.0 / 8, 0, 0, 0, 0, 0, "scaled_dot_product_attention");
	ccv_nnc_tensor_param_t qkv[3];
	qkv[0] = q_tensor->info;
	qkv[1] = k_tensor->info;
	qkv[2] = v_tensor->info;
	ccv_cnnp_model_compile(scaled_dot_product_attention, qkv, 3, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(scaled_dot_product_attention, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(r_tensor), 0, 0);
	CNNP_MODEL_GEN(scaled_dot_product_attention, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_TENSOR_EQ(r_tensor, br_tensor, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(r_tensor);
	ccv_cnnp_model_free(scaled_dot_product_attention);
}

TEST_CASE("run scaled dot product attention + unify head output with cnnp model")
{
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t bw = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 768, 768), "w");
	ccv_nnc_tensor_symbol_t bbias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 768), "bias");
	ccv_nnc_tensor_symbol_t bc = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "c");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 768), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv, NO_TENSOR_SYMBOL, bw, bbias), TENSOR_SYMBOL_LIST(br, NO_TENSOR_SYMBOL, bc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	ccv_nnc_tensor_t* const bw_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bw);
	ccv_nnc_tensor_t* const bbias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bbias);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bq_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bk_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 96; i++)
		bv_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 768 * 768; i++)
		bw_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 768; i++)
		bbias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), 0);
	memcpy(q_tensor->data.f32, bq_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(k_tensor->data.f32, bk_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(v_tensor->data.f32, bv_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	ccv_nnc_tensor_t* const br_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, br);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_new(0, br_tensor->info, 0);
	ccv_cnnp_model_t* scaled_dot_product_attention = ccv_cnnp_scaled_dot_product_attention(1.0 / 8, 0, 0, 1, 0, 1, "scaled_dot_product_attention");
	ccv_nnc_tensor_param_t qkv[3];
	qkv[0] = q_tensor->info;
	qkv[1] = k_tensor->info;
	qkv[2] = v_tensor->info;
	ccv_cnnp_model_compile(scaled_dot_product_attention, qkv, 3, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_set_parameter(scaled_dot_product_attention, ccv_cnnp_model_parameters(scaled_dot_product_attention, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0), bw_tensor);
	ccv_cnnp_model_set_parameter(scaled_dot_product_attention, ccv_cnnp_model_parameters(scaled_dot_product_attention, CCV_CNNP_PARAMETER_SELECT_BIAS, 0), bbias_tensor);
	ccv_cnnp_model_evaluate(scaled_dot_product_attention, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(r_tensor), 0, 0);
	CNNP_MODEL_GEN(scaled_dot_product_attention, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	REQUIRE_TENSOR_EQ(r_tensor, br_tensor, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(r_tensor);
	ccv_cnnp_model_free(scaled_dot_product_attention);
}

TEST_CASE("run scaled dot product attention + attention mask with cnnp model")
{
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t battn_mask = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 1, 128, 128), "attn_mask");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv, battn_mask), TENSOR_SYMBOL_LIST(br), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	ccv_nnc_tensor_t* const battn_mask_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, battn_mask);
	int i, j;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bq_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		bk_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 96; i++)
		bv_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128 * 128; i++)
		battn_mask_tensor->data.f32[i] = 0;
	for (i = 0; i < 127; i++)
		for (j = i + 1; j < 128; j++)
			battn_mask_tensor->data.f32[i * 128 + j] = -FLT_MAX;
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), 0);
	ccv_nnc_tensor_t* const attn_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 128, 128), 0);
	memcpy(q_tensor->data.f32, bq_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(k_tensor->data.f32, bk_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(v_tensor->data.f32, bv_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	memcpy(attn_mask_tensor->data.f32, battn_mask_tensor->data.f32, sizeof(float) * 128 * 128);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const br_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, br);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_new(0, br_tensor->info, 0);
	ccv_cnnp_model_t* scaled_dot_product_attention = ccv_cnnp_scaled_dot_product_attention(1.0 / 8, 0, 1, 0, 0, 0, "scaled_dot_product_attention");
	ccv_nnc_tensor_param_t qkv[4];
	qkv[0] = q_tensor->info;
	qkv[1] = k_tensor->info;
	qkv[2] = v_tensor->info;
	qkv[3] = attn_mask_tensor->info;
	ccv_cnnp_model_compile(scaled_dot_product_attention, qkv, 4, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(scaled_dot_product_attention, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(q_tensor, k_tensor, v_tensor, attn_mask_tensor), TENSOR_LIST(r_tensor), 0, 0);
	CNNP_MODEL_GEN(scaled_dot_product_attention, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_TENSOR_EQ(r_tensor, br_tensor, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(attn_mask_tensor);
	ccv_nnc_tensor_free(r_tensor);
	ccv_cnnp_model_free(scaled_dot_product_attention);
}

TEST_CASE("implement gradient of scaled dot product attention with fine-grained symbolic graph")
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t tq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t tk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t tv = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "v");
	ccv_nnc_tensor_symbol_t qk = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "qk");
	ccv_nnc_tensor_symbol_t sq = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 64), "sq");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(q), TENSOR_SYMBOL_LIST(tq), "transpose q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(k), TENSOR_SYMBOL_LIST(tk), "transpose k");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(v), TENSOR_SYMBOL_LIST(tv), "transpose v");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SCALAR_MUL_FORWARD(1.0 / 8), TENSOR_SYMBOL_LIST(tq), TENSOR_SYMBOL_LIST(sq), "scaled_q");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), TENSOR_SYMBOL_LIST(sq, tk), TENSOR_SYMBOL_LIST(qk), "q @ k");
	ccv_nnc_tensor_symbol_t qks = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, qk, DIM_ALLOC(), DIM_ALLOC(128, 1), CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "qks");
	ccv_nnc_tensor_symbol_t s = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32 * 8 * 128, 128), "s");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(qks), TENSOR_SYMBOL_LIST(s), "softmax");
	ccv_nnc_tensor_symbol_t sa = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, s, DIM_ALLOC(), DIM_ALLOC(8 * 128 * 128, 128 * 128, 128, 1), CPU_TENSOR_NHWC(32F, 32, 8, 128, 128), "sa");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 8, 128, 96), "f");
	ccv_nnc_tensor_symbol_t tr = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "f");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), TENSOR_SYMBOL_LIST(sa, tv), TENSOR_SYMBOL_LIST(r), "final");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TRANSPOSE_FORWARD(1, 2), TENSOR_SYMBOL_LIST(r), TENSOR_SYMBOL_LIST(tr), "final");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(tr), TENSOR_SYMBOL_LIST(q, k, v), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
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
	ccv_nnc_tensor_symbol_t dr = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, tr);
	ccv_nnc_tensor_t* const dr_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dr);
	for (i = 0; i < 32 * 128 * 8 * 96; i++)
		dr_tensor->data.f32[i] = 1;
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bq = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t bk = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t bv = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "v");
	ccv_nnc_tensor_symbol_t br = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 96), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(bq, bk, bv), TENSOR_SYMBOL_LIST(br, NO_TENSOR_SYMBOL, NO_TENSOR_SYMBOL), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(sdp_symbolic_graph, TENSOR_SYMBOL_LIST(br), TENSOR_SYMBOL_LIST(bq, bk, bv), SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(sdp_symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	GRAPH_GEN(sdp_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bq);
	ccv_nnc_tensor_t* const bk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bk);
	ccv_nnc_tensor_t* const bv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bv);
	memcpy(bq_tensor->data.f32, q_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bk_tensor->data.f32, k_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 64);
	memcpy(bv_tensor->data.f32, v_tensor->data.f32, sizeof(float) * 32 * 8 * 128 * 96);
	ccv_nnc_tensor_symbol_t dbr = ccv_nnc_tensor_symbol_for_backward(sdp_symbolic_graph, br);
	ccv_nnc_tensor_t* const dbr_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, dbr);
	for (i = 0; i < 32 * 128 * 8 * 96; i++)
		dbr_tensor->data.f32[i] = 1;
	// CCV_CLI_SET_OUTPUT_LEVEL_AND_ABOVE(CCV_CLI_VERBOSE);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_symbol_t dq = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, q);
	ccv_nnc_tensor_symbol_t dk = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, k);
	ccv_nnc_tensor_symbol_t dv = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, v);
	ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dq);
	ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dk);
	ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dv);
	ccv_nnc_tensor_symbol_t dbq = ccv_nnc_tensor_symbol_for_backward(sdp_symbolic_graph, bq);
	ccv_nnc_tensor_symbol_t dbk = ccv_nnc_tensor_symbol_for_backward(sdp_symbolic_graph, bk);
	ccv_nnc_tensor_symbol_t dbv = ccv_nnc_tensor_symbol_for_backward(sdp_symbolic_graph, bv);
	ccv_nnc_tensor_t* const dbq_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, dbq);
	ccv_nnc_tensor_t* const dbk_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, dbk);
	ccv_nnc_tensor_t* const dbv_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, dbv);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dq_tensor->data.f32, dbq_tensor->data.f32, 32 * 128 * 8 * 64, 1e-5, "graph computed gradient should match scaled dot product attention op gradient");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dk_tensor->data.f32, dbk_tensor->data.f32, 32 * 128 * 8 * 64, 1e-5, "graph computed gradient should match scaled dot product attention op gradient");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dv_tensor->data.f32, dbv_tensor->data.f32, 32 * 128 * 8 * 96, 1e-5, "graph computed gradient should match scaled dot product attention op gradient");
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
