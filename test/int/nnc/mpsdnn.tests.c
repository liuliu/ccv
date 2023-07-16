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

#define INPUT_DIM (3)
#define OUTPUT_DIM (96)

#define INPUT_SIZE (224)
#define OUTPUT_SIZE (112)

#define KERNEL_SIZE (7)

#define BATCH_SIZE (64)

TEST_CASE("mps forward convolution")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);

	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(transform, ccv_nnc_no_hint, 0, TENSOR_LIST(gw), TENSOR_LIST(gwo), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_free(gw);

	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-5, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution in nchw format")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);

	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-5, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	ccv_nnc_tensor_t* a1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* w1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(a1, w1, bias1), 0);
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a1, w1, bias1), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);

	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_exec(transform, ccv_nnc_no_hint, 0, TENSOR_LIST(gw), TENSOR_LIST(gwo), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_free(gw);

	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c1), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c1), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 5e-3, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c1);
	ccv_nnc_tensor_free(bias1);
	ccv_nnc_tensor_free(w1);
	ccv_nnc_tensor_free(a1);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("compare softmax with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_TENSOR_EQ(ty, y_tensor, "softmax from mps should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare softmax with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "softmax");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty->data.f32, y_tensor->data.f32, 20 * 10, 1e-3, "softmax from mps should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare sigmoid with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_TENSOR_EQ(ty, y_tensor, "sigmoid from mps should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare sigmoid with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty->data.f32, y_tensor->data.f32, 20 * 10, 1e-3, "sigmoid from mps should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}


TEST_CASE("compare sigmoid gradient with mps")
{
	ccv_cli_set_output_levels(ccv_cli_output_level_and_above(CCV_CLI_VERBOSE));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SIGMOID_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SIGMOID_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_print_tensor_info(dxt);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, 0, ty_tensor), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_TENSOR_EQ(tdx_tensor, dx_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare relu with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RELU_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 7, 7, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "relu");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	ccv_nnc_cmd_exec(CMD_RELU_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y), 0);
	REQUIRE_TENSOR_EQ(y_tensor, cpu_y, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
}

TEST_CASE("compare relu with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RELU_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 7, 7, 10), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "relu");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 7, 7, 10), 0);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	ccv_nnc_cmd_exec(CMD_RELU_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 7, 7, 10), 0);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_y16), TENSOR_LIST(cpu_y), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cpu_y->data.f32, 7 * 7 * 10, 1e-3, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_y16);
}

TEST_CASE("compare layer norm with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, 10), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-6, 1, 2, 3), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(y), "transfer y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	float xdata[2 * 2 * 2 * 10];
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * 10; i++)
		x_tensor->data.f32[i] = xdata[i] = dsfmt_genrand_open_close(&dsfmt);
	float scaledata[1 * 2 * 2 * 10];
	float biasdata[1 * 2 * 2 * 10];
	for (i = 0; i < 1 * 2 * 2 * 10; i++)
	{
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);
	}
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-6, 1, 2, 3), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 2 * 2 * 10);
	ccv_nnc_tensor_t* const cbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cbias);
	memcpy(cbias_tensor->data.f32, biasdata, sizeof(float) * 1 * 2 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	// Note that MPS and my other implementations treat epsilon differently.
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cy_tensor->data.f32, 2 * 2 * 2 * 10, 1e-4, "layer norm result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
}

TEST_CASE("compare group norm with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 16, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 16, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 16, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 16, 2, 10), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(y), "transfer y");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	float xdata[2 * 16 * 2 * 10];
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 16 * 2 * 10; i++)
		x_tensor->data.f32[i] = xdata[i] = dsfmt_genrand_open_close(&dsfmt);
	float scaledata[1 * 16 * 2 * 10];
	float biasdata[1 * 16 * 2 * 10];
	for (i = 0; i < 1 * 16 * 2 * 10; i++)
	{
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);
	}
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 16, 2, 10), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, 16, 2, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 16, 2, 10), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 4, 2, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 4, 2, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 16 * 2 * 10);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 16 * 2 * 10);
	ccv_nnc_tensor_t* const cbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cbias);
	memcpy(cbias_tensor->data.f32, biasdata, sizeof(float) * 1 * 16 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	// Note that MPS and my other implementations treat epsilon differently.
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cy_tensor->data.f32, 2 * 16 * 2 * 10, 1e-3, "group norm result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
}

TEST_CASE("compare add with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), "y");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 1, 3), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "c");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "z");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(a, b), "transfer");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(0.5, 0.2), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "add");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(z), "transfer");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(x, x_tensor), KV(y, y_tensor)), TENSOR_SYMBOL_LIST(z), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 5 * 1 * 3; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* zt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(zt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_TENSOR_EQ(zt, z_tensor, "add should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(zt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare add with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), "y");
	ccv_nnc_tensor_symbol_t x16 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(16F, 10, 5, 5, 3), "x 16");
	ccv_nnc_tensor_symbol_t y16 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(16F, 10, 5, 1, 3), "y 16");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 5, 5, 3), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 5, 1, 3), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 5, 5, 3), "c");
	ccv_nnc_tensor_symbol_t z = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "z");
	ccv_nnc_tensor_symbol_t z16 = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(16F, 10, 5, 5, 3), "z 16");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATATYPE_CONVERSION_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(x16, y16), "convert");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x16, y16), TENSOR_SYMBOL_LIST(a, b), "transfer");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(0.5, 0.2), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "add");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(z16), "transfer");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATATYPE_CONVERSION_FORWARD(), TENSOR_SYMBOL_LIST(z16), TENSOR_SYMBOL_LIST(z), "convert");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(x, x_tensor), KV(y, y_tensor)), TENSOR_SYMBOL_LIST(z), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 5 * 1 * 3; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* zt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(zt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, z);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, zt->data.f32, z_tensor->data.f32, 10 * 5 * 5 * 3, 1e-3, "add should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(zt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare ewsum with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 100), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 100), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 100), 0);
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	int i;
	for (i = 0; i < 100; i++)
	{
		ha->data.f32[i] = 1;
		hb->data.f32[i] = 0.5;
		hc->data.f32[i] = 0.25;
		gd->data.f32[i] = 1.75;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hc), TENSOR_LIST(a, b, c), 0);
	ccv_nnc_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(d), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(hd), 0);
	REQUIRE_TENSOR_EQ(hd, gd, "ewsum result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("compare ewsum with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100), 0);
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100), 0);
	ccv_nnc_tensor_t* const hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100), 0);
	ccv_nnc_tensor_t* const hc16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100), 0);
	ccv_nnc_tensor_t* const hd16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100), 0);
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	int i;
	for (i = 0; i < 100; i++)
	{
		ha->data.f32[i] = 1;
		hb->data.f32[i] = 0.5;
		hc->data.f32[i] = 0.25;
		gd->data.f32[i] = 1.75;
	}
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hc), TENSOR_LIST(ha16, hb16, hc16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hb16, hc16), TENSOR_LIST(a, b, c), 0);
	ccv_nnc_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(d), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(hd16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hd16), TENSOR_LIST(hd), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hd->data.f32, gd->data.f32, 100, 1e-3, "ewsum result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(hc16);
	ccv_nnc_tensor_free(hd16);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("compare transpose two tensor views")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 6, 5, 4), 0);
	memset(ha->data.f32, 0, sizeof(float) * 7 * 6 * 5 * 4);
	ccv_nnc_tensor_view_t ha_view = ccv_nnc_tensor_view(ha, CPU_TENSOR_NHWC(32F, 4, 3, 2, 2), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(6 * 5 * 4, 5 * 4, 4, 1));
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 8, 7, 6, 5), 0);
	memset(hb->data.f32, 0, sizeof(float) * 8 * 7 * 6 * 5);
	ccv_nnc_tensor_view_t hb_view = ccv_nnc_tensor_view(hb, CPU_TENSOR_NHWC(32F, 4, 2, 2, 3), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(7 * 6 * 5, 6 * 5, 5, 1));
	int i, j, k, l;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 2; k++)
				for (l = 0; l < 2; l++)
					ha->data.f32[(i + 3) * 6 * 5 * 4 + (j + 2) * 5 * 4 + (k + 1) * 4 + l] = i * 3 * 2 * 2 + j * 2 * 2 + k * 2 + l;
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 3), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&ha_view), TENSOR_LIST((ccv_nnc_tensor_t*)&hb_view), 0);
	ccv_nnc_tensor_t* hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 6, 5, 4), 0);
	memset(hd->data.f32, 0, sizeof(float) * 7 * 6 * 5 * 4);
	ccv_nnc_tensor_view_t hd_view = ccv_nnc_tensor_view(hd, CPU_TENSOR_NHWC(32F, 4, 3, 2, 2), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(6 * 5 * 4, 5 * 4, 4, 1));
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 3), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&hb_view), TENSOR_LIST((ccv_nnc_tensor_t*)&hd_view), 0);
	REQUIRE_TENSOR_EQ(hd, ha, "4x3x2x2 tensor should be exactly the same.");
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_view_t a_view = ccv_nnc_tensor_view(a, GPU_TENSOR_NHWC(000, 32F, 4, 3, 2, 2), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(6 * 5 * 4, 5 * 4, 4, 1));
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 8, 7, 6, 5), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_view_t b_view = ccv_nnc_tensor_view(b, GPU_TENSOR_NHWC(000, 32F, 4, 2, 2, 3), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(7 * 6 * 5, 6 * 5, 5, 1));
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 3), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&a_view), TENSOR_LIST((ccv_nnc_tensor_t*)&b_view), 0);
	ccv_nnc_tensor_t* d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(d), 0);
	ccv_nnc_tensor_view_t d_view = ccv_nnc_tensor_view(d, GPU_TENSOR_NHWC(000, 32F, 4, 3, 2, 2), DIM_ALLOC(3, 2, 1, 0), DIM_ALLOC(6 * 5 * 4, 5 * 4, 4, 1));
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 3), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&b_view), TENSOR_LIST((ccv_nnc_tensor_t*)&d_view), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 8, 7, 6, 5), 0);
	ccv_nnc_tensor_t* const hdt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, d), TENSOR_LIST(hbt, hdt), 0);
	REQUIRE_TENSOR_EQ(hbt, hb, "4x2x2x3 tensor should be exactly the same.");
	REQUIRE_TENSOR_EQ(hdt, hd, "4x3x2x2 tensor should be exactly the same.");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(hbt);
	ccv_nnc_tensor_free(hdt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(d);
}

TEST_CASE("broadcasting semantics for add [[1, 2, 3], [4, 5, 6]] + [7, 8, 9]")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	b->data.f32[0] = 7;
	b->data.f32[1] = 8;
	b->data.f32[2] = 9;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		8, 10, 12,
		11, 13, 15
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("broadcasting semantics for add [[1], [2], [3], [4]] + [5, 6]")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("broadcasting semantics for mul [[1, 2, 3], [4, 5, 6]] * [7, 8, 9]")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	b->data.f32[0] = 7;
	b->data.f32[1] = 8;
	b->data.f32[2] = 9;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		7, 16, 27,
		28, 40, 54
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("broadcasting semantics for mul [[1], [2], [3], [4]] * [5, 6]")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		5, 6,
		10, 12,
		15, 18,
		20, 24
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("scalar mul [[1, 2, 3], [4, 5, 6]] * 0.3")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(0.3), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		0.3, 0.6, 0.9,
		1.2, 1.5, 1.8,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("compare average pooling with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_AVERAGE_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 3, 3, 10), "y");
	ccv_nnc_graph_exec_symbol_t avg_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(5, 5), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "avg_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg_pool, HINT((2, 2), (1, 1)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_FORWARD(5, 5), HINT((2, 2), (1, 1)), 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y), 0);
	REQUIRE_TENSOR_EQ(y_tensor, cpu_y, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
}

TEST_CASE("compare average pooling with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_AVERAGE_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 3, 3, 10), "y");
	ccv_nnc_graph_exec_symbol_t avg_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(5, 5), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "avg_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg_pool, HINT((2, 2), (1, 1)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 7, 7, 10), 0);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_FORWARD(5, 5), HINT((2, 2), (1, 1)), 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_tensor_t* const cpu_y16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_y16), TENSOR_LIST(cpu_y), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cpu_y->data.f32, 3 * 3 * 10, 1e-3, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_y16);
}

TEST_CASE("compare max pooling with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 3, 3, 10), "y");
	ccv_nnc_graph_exec_symbol_t max_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(5, 5), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "max_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, max_pool, HINT((2, 2), (1, 1)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_MAX_POOL_FORWARD(5, 5), HINT((2, 2), (1, 1)), 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y), 0);
	REQUIRE_TENSOR_EQ(y_tensor, cpu_y, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
}

TEST_CASE("compare max pooling with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 7, 7, 10), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 3, 3, 10), "y");
	ccv_nnc_graph_exec_symbol_t max_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(5, 5), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "max_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, max_pool, HINT((2, 2), (1, 1)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 10), 0);
	int i;
	for (i = 0; i < 7 * 7 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 7, 7, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_MAX_POOL_FORWARD(5, 5), HINT((2, 2), (1, 1)), 0, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 3, 10), 0);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_y16), TENSOR_LIST(cpu_y), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cpu_y->data.f32, 3 * 3 * 10, 1e-3, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_y16);
}

TEST_CASE("compare max pooling 2x2 with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 10, 6, 6), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 10, 3, 3), "y");
	ccv_nnc_graph_exec_symbol_t max_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "max_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, max_pool, HINT((2, 2), (0, 0)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 6, 6), 0);
	int i, j;
	for (i = 0; i < 6 * 6 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gt_x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 6, 10), 0);
	for (i = 0; i < 10; i++)
		for (j = 0; j < 6 * 6; j++)
			gt_x->data.f32[j * 10 + i] = x_tensor->data.f32[i * 6 * 6 + j];
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const gt_y= ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_MAX_POOL_FORWARD(2, 2), HINT((2, 2), (0, 0)), 0, TENSOR_LIST(gt_x), TENSOR_LIST(gt_y), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 3, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 3, 3), 0);
	for (i = 0; i < 10; i++)
		for (j = 0; j < 3 * 3; j++)
			y_tensor->data.f32[i * 3 * 3 + j] = gt_y->data.f32[j * 10 + i];
	REQUIRE_TENSOR_EQ(y_tensor, cpu_y, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
}

TEST_CASE("compare max pooling 2x2 with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MAX_POOL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 10, 6, 6), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 10, 3, 3), "y");
	ccv_nnc_graph_exec_symbol_t max_pool = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "max_pool");
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, max_pool, HINT((2, 2), (0, 0)));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 6, 6), 0);
	int i, j;
	for (i = 0; i < 6 * 6 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gt_x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 6, 10), 0);
	for (i = 0; i < 10; i++)
		for (j = 0; j < 6 * 6; j++)
			gt_x->data.f32[j * 10 + i] = x_tensor->data.f32[i * 6 * 6 + j];
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 10, 6, 6), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const gt_y= ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 10), 0);
	ccv_nnc_cmd_exec(CMD_MAX_POOL_FORWARD(2, 2), HINT((2, 2), (0, 0)), 0, TENSOR_LIST(gt_x), TENSOR_LIST(gt_y), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_tensor_t* const cpu_y16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 10, 3, 3), 0);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 3, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(cpu_y16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_y16), TENSOR_LIST(cpu_y), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 3, 3), 0);
	for (i = 0; i < 10; i++)
		for (j = 0; j < 3 * 3; j++)
			y_tensor->data.f32[i * 3 * 3 + j] = gt_y->data.f32[j * 10 + i];
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cpu_y->data.f32, 10 * 3 * 3, 1e-3, "mps result should equal to cpu result");
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_y16);
}

#include "case_main.h"
