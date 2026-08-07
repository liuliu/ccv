#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <nnc/ccv_nnc_internal.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

#define INPUT_DIM (3)
#define OUTPUT_DIM (96)

#define INPUT_SIZE (224)
#define OUTPUT_SIZE (112)

#define KERNEL_SIZE (7)

#define BATCH_SIZE (16)

#define LN_DIM (10)
#define GN_C_DIM (16)
#define GN_RC_DIM (4)

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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 5e-4, "output from mps should match from CPU");
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-3, "output from mps should match from CPU");
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

TEST_CASE("mps forward grouped convolution falls back with mfa enabled")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int groups = 2;
	const int input_dim = 4;
	const int output_dim = 6;
	const int batch_size = 2;
	const int input_size = 8;
	const int kernel_size = 3;
	const int output_size = input_size - kernel_size + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, input_dim, input_size, input_size), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_size, output_size), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, output_dim, input_dim / groups, kernel_size, kernel_size), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(groups, output_dim, kernel_size, kernel_size, input_dim / groups);
	ccv_nnc_hint_t hint = HINT((1, 1), (0, 0));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < batch_size * input_dim * input_size * input_size; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < output_dim * (input_dim / groups) * kernel_size * kernel_size; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (input_dim / groups * kernel_size * kernel_size);
	ccv_nnc_cmd_t cpu_cmd = cmd;
	cpu_cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cpu_cmd.backend >= 0);
	ccv_nnc_cmd_exec(cpu_cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, input_dim, input_size, input_size), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_dim, input_dim / groups, kernel_size, kernel_size), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, output_dim, output_size, output_size), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
	ccv_nnc_cmd_t mps_cmd = cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(mps_cmd.backend >= 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(mps_cmd, hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_size, output_size), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_dim * output_size * output_size, 1e-5, "grouped convolution output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("mps forward convolution with 1x1 kernel")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, 1, 1, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, 1, 1, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * 1 * 1 * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * 1 * 1);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, 1, 1, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, 1, 1), 0);
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-3, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution in nchw format with 1x1 kernel")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, 1, 1, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, INPUT_DIM, 1, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * 1 * 1 * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * 1 * 1);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, 1, 1), 0);
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-3, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution in nchw format with row-wise 8i weight")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_dim = 8;
	const int output_dim = 12;
	const int spatial = 9;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, input_dim, spatial, spatial), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, spatial, spatial), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_dim, 1, 1, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, output_dim, input_dim, 1, 1), 0);
	ccv_nnc_tensor_t* wq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NCHW(32F, output_dim, input_dim, 1, 1)), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < batch_size * input_dim * spatial * spatial; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < output_dim * input_dim; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5f;
	for (i = 0; i < output_dim; i++)
		bias->data.f32[i] = (float)i / output_dim;
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(w->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, output_dim * input_dim, 1, 0, 0, wq->data.u8, ccv_nnc_tensor_data_size_without_padding(wq->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(wq->info), "row-wise 8i convolution weight should fit the tensor exactly");
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, input_dim, spatial, spatial), 0);
	ccv_nnc_tensor_t* gwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NCHW(000, 32F, output_dim, input_dim, 1, 1)), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_dim), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, output_dim, spatial, spatial), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, wq, bias), TENSOR_LIST(ga, gwq, gbias), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwq, gbias), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwq, gbias), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, spatial, spatial), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_dim * spatial * spatial, 2e-3, "output from mps should match CPU when row-wise 8i weights are dequantized to dense scratch");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gwq);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(wq);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("mps forward convolution in nchw format with 1x1 kernel and no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, 1, 1, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, INPUT_DIM, 1, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < INPUT_DIM * 1 * 1 * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * 1 * 1);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, 1, 1), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);

	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-3, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution in nchw format with 1x1 kernel on edge tiles")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_dim = 17;
	const int output_dim = 77;
	const int output_h = 17;
	const int output_w = 19;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, input_dim, output_h, output_w), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_h, output_w), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_dim, 1, 1, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, input_dim, 1, 1), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i;
	for (i = 0; i < input_dim * output_dim; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / input_dim;
	for (i = 0; i < batch_size * input_dim * output_h * output_w; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < output_dim; i++)
		bias->data.f32[i] = (float)i / output_dim;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, input_dim, output_h, output_w), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_dim, input_dim, 1, 1), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_dim), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, output_dim, output_h, output_w), 0);

	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_h, output_w), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_dim * output_h * output_w, 1e-3, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution in nchw format with 1x1 kernel on edge tiles and no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_dim = 17;
	const int output_dim = 77;
	const int output_h = 17;
	const int output_w = 19;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, input_dim, output_h, output_w), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_h, output_w), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_dim, 1, 1, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, input_dim, 1, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i;
	for (i = 0; i < input_dim * output_dim; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / input_dim;
	for (i = 0; i < batch_size * input_dim * output_h * output_w; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, input_dim, output_h, output_w), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_dim, input_dim, 1, 1), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, batch_size, output_dim, output_h, output_w), 0);

	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, output_dim, output_h, output_w), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_dim * output_h * output_w, 1e-3, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
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

TEST_CASE("mps forward convolution in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_size = 17;
	const int kernel_size = 3;
	const int output_size = 15;
	const int input_dim = 8;
	const int output_dim = 16;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_dim, kernel_size, kernel_size, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	int i;
	for (i = 0; i < input_dim * kernel_size * kernel_size * output_dim; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) / (input_dim * kernel_size * kernel_size);
	for (i = 0; i < input_size * input_size * input_dim * batch_size; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_dim; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	const int input_count = batch_size * input_size * input_size * input_dim;
	const int weight_count = output_dim * kernel_size * kernel_size * input_dim;
	const int output_count = batch_size * output_size * output_size * output_dim;
	ccv_nnc_tensor_t* a16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* w16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* bias16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim), 0);
	ccv_float_to_bfloat(a->data.f32, (uint16_t*)a16bf->data.f16, input_count);
	ccv_float_to_bfloat(w->data.f32, (uint16_t*)w16bf->data.f16, weight_count);
	ccv_float_to_bfloat(bias->data.f32, (uint16_t*)bias16bf->data.f16, output_dim);
	ccv_bfloat_to_float((uint16_t*)a16bf->data.f16, a->data.f32, input_count);
	ccv_bfloat_to_float((uint16_t*)w16bf->data.f16, w->data.f32, weight_count);
	ccv_bfloat_to_float((uint16_t*)bias16bf->data.f16, bias->data.f32, output_dim);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* b16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_float_to_bfloat(b->data.f32, (uint16_t*)b16bf->data.f16, output_count);
	ccv_bfloat_to_float((uint16_t*)b16bf->data.f16, b->data.f32, output_count);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, output_dim, input_dim, kernel_size, kernel_size), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a16bf, w16bf, bias16bf), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c16bf), 0);
	ccv_bfloat_to_float((uint16_t*)c16bf->data.f16, c->data.f32, output_count);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, output_count, 5e-2, "bfloat output from mps should match CPU reference rounded to bfloat");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(c16bf);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(b16bf);
	ccv_nnc_tensor_free(bias16bf);
	ccv_nnc_tensor_free(w16bf);
	ccv_nnc_tensor_free(a16bf);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("mps forward convolution in bfloat precision with 1x1 kernel")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_size = 17;
	const int output_size = 17;
	const int input_dim = 16;
	const int output_dim = 32;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_dim, 1, 1, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, 1, 1, input_dim), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 11);
	int i;
	for (i = 0; i < input_dim * output_dim; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) / input_dim;
	for (i = 0; i < input_size * input_size * input_dim * batch_size; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_dim; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	const int input_count = batch_size * input_size * input_size * input_dim;
	const int weight_count = output_dim * input_dim;
	const int output_count = batch_size * output_size * output_size * output_dim;
	ccv_nnc_tensor_t* a16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* w16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim, 1, 1, input_dim), 0);
	ccv_nnc_tensor_t* bias16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim), 0);
	ccv_float_to_bfloat(a->data.f32, (uint16_t*)a16bf->data.f16, input_count);
	ccv_float_to_bfloat(w->data.f32, (uint16_t*)w16bf->data.f16, weight_count);
	ccv_float_to_bfloat(bias->data.f32, (uint16_t*)bias16bf->data.f16, output_dim);
	ccv_bfloat_to_float((uint16_t*)a16bf->data.f16, a->data.f32, input_count);
	ccv_bfloat_to_float((uint16_t*)w16bf->data.f16, w->data.f32, weight_count);
	ccv_bfloat_to_float((uint16_t*)bias16bf->data.f16, bias->data.f32, output_dim);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* b16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_float_to_bfloat(b->data.f32, (uint16_t*)b16bf->data.f16, output_count);
	ccv_bfloat_to_float((uint16_t*)b16bf->data.f16, b->data.f32, output_count);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim, 1, 1, input_dim), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, output_dim, input_dim, 1, 1), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a16bf, w16bf, bias16bf), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c16bf), 0);
	ccv_bfloat_to_float((uint16_t*)c16bf->data.f16, c->data.f32, output_count);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, output_count, 5e-2, "bfloat output from mps 1x1 convolution should match CPU reference rounded to bfloat");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(c16bf);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(b16bf);
	ccv_nnc_tensor_free(bias16bf);
	ccv_nnc_tensor_free(w16bf);
	ccv_nnc_tensor_free(a16bf);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("mps forward convolution with dilation 2, 3")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.info.convolution.dilation[0] = 2;
	cmd.info.convolution.dilation[1] = 3;
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_param_t modified_cmd = cmd.info;
	modified_cmd.size.dim[0] = (cmd.info.size.dim[0] - 1) * ccv_max(cmd.info.convolution.dilation[0], 1) + 1;
	modified_cmd.size.dim[1] = (cmd.info.size.dim[1] - 1) * ccv_max(cmd.info.convolution.dilation[1], 1) + 1;
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(modified_cmd, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, modified_cmd, a->info, b->info) == 0);
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

TEST_CASE("mps forward convolution 3d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, 5, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, 3, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 2;
	hint.border.begin[0] = 1;
	hint.border.end[0] = 1;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * 3 * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < 5 * INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, 5, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, 3, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, 3, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, BATCH_SIZE * 3 * OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d via mfa conv3d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 10;
	const int input_width = 10;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int output_depth = 3;
	const int output_height = 8;
	const int output_width = 8;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 10;
	const int input_width = 10;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int output_depth = 3;
	const int output_height = 8;
	const int output_width = 8;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	for (i = 0; i < output_channels; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d via mfa conv3d with no padding on 17x17 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 17;
	const int input_width = 17;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int output_depth = 3;
	const int output_height = 15;
	const int output_width = 15;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 21);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with no padding on 65x65 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 65;
	const int input_width = 65;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int output_depth = 3;
	const int output_height = 63;
	const int output_width = 63;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 22);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with no padding on 129x129 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 129;
	const int input_width = 129;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int output_depth = 3;
	const int output_height = 127;
	const int output_width = 127;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 23);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d 5x5")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 12;
	const int input_width = 12;
	const int kernel_depth = 3;
	const int kernel_height = 5;
	const int kernel_width = 5;
	const int output_depth = 3;
	const int output_height = 8;
	const int output_width = 8;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d 7x7")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 14;
	const int input_width = 14;
	const int kernel_depth = 3;
	const int kernel_height = 7;
	const int kernel_width = 7;
	const int output_depth = 3;
	const int output_height = 8;
	const int output_width = 8;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with padding")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 10;
	const int input_width = 10;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 5);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with asymmetric padding")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 10;
	const int input_width = 11;
	const int kernel_depth = 3;
	const int kernel_height = 5;
	const int kernel_width = 5;
	const int padding_top = 1;
	const int padding_bottom = 0;
	const int padding_left = 2;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 6);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with padding on small spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 5;
	const int input_width = 5;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 7);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with padding and bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 9;
	const int input_width = 9;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 19);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	for (i = 0; i < output_channels; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on tile boundary and bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 8;
	const int input_width = 8;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 20);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	for (i = 0; i < output_channels; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on tile boundary")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 8;
	const int input_width = 8;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 9);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on small rectangular spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 6;
	const int input_width = 7;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 9x9 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 9;
	const int input_width = 9;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 11);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 8x9 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 8;
	const int input_width = 9;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 12);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 9x8 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 9;
	const int input_width = 8;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 13);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 17x19 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 5;
	const int input_height = 17;
	const int input_width = 19;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 17);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 33x35 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 4;
	const int input_height = 33;
	const int input_width = 35;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 18);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with one point padding on 9x65 spatial dimensions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 32;
	const int input_depth = 4;
	const int input_height = 9;
	const int input_width = 65;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 19);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(ga, gw), 0);
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
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("mps forward convolution 3d via mfa conv3d with partial output channel tile and bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 16;
	const int output_channels = 16;
	const int input_depth = 4;
	const int input_height = 33;
	const int input_width = 35;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 21);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	for (i = 0; i < output_channels; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d via mfa conv3d with 48 channels and partial output channel tile")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_channels = 48;
	const int output_channels = 48;
	const int input_depth = 4;
	const int input_height = 33;
	const int input_width = 35;
	const int kernel_depth = 3;
	const int kernel_height = 3;
	const int kernel_width = 3;
	const int padding_top = 1;
	const int padding_bottom = 1;
	const int padding_left = 1;
	const int padding_right = 1;
	const int output_depth = input_depth - kernel_depth + 1;
	const int output_height = input_height + padding_top + padding_bottom - kernel_height + 1;
	const int output_width = input_width + padding_left + padding_right - kernel_width + 1;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, output_channels, kernel_depth, kernel_height, kernel_width, input_channels);
	ccv_nnc_hint_t hint = ccv_nnc_no_hint;
	hint.stride.dim[0] = 1;
	hint.stride.dim[1] = 1;
	hint.stride.dim[2] = 1;
	hint.border.begin[0] = 0;
	hint.border.end[0] = 0;
	hint.border.begin[1] = padding_top;
	hint.border.end[1] = padding_bottom;
	hint.border.begin[2] = padding_left;
	hint.border.end[2] = padding_right;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 23);
	int i;
	for (i = 0; i < batch_size * input_depth * input_height * input_width * input_channels; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_channels * kernel_depth * kernel_height * kernel_width * input_channels; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	for (i = 0; i < output_channels; i++)
		bias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) * 0.1;
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, input_depth, input_height, input_width, input_channels), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels, kernel_depth, kernel_height, kernel_width, input_channels), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, output_channels, input_channels, kernel_depth, kernel_height, kernel_width), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, output_channels), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_depth, output_height, output_width, output_channels), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, batch_size * output_depth * output_height * output_width * output_channels, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution 3d in nchw format")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, 5, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, 3, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, OUTPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, b->info);
	hint.stride.dim[0] = 2;
	hint.border.begin[0] = 1;
	hint.border.end[0] = 1;
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, b->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, OUTPUT_DIM, INPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 3 * INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < 5 * INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_DIM; i++)
		bias->data.f32[i] = (float)i / OUTPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, 5, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, 3, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, 3, OUTPUT_SIZE, OUTPUT_SIZE), 0);

	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, 3, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, BATCH_SIZE * 3 * OUTPUT_DIM * OUTPUT_SIZE * OUTPUT_SIZE, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("compare softmax gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SOFTMAX_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "softmax");
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
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, 0, ty_tensor), TENSOR_LIST(tdx_tensor), 0);
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

TEST_CASE("compare sigmoid with mps and more vectorization cases")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* const a0 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 32, 32), 0);
	ccv_nnc_tensor_t* const b0 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 32, 32), 0);
	ccv_nnc_tensor_t* const ha0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 32), 0);
	ccv_nnc_tensor_t* const hb0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 32), 0);
	ccv_nnc_tensor_t* const tb0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 32), 0);
	for (i = 0; i < 32 * 32; i++)
		switch (i % 6)
		{
			case 0:
				ha0->data.f32[i] = -80;
				break;
			case 1:
				ha0->data.f32[i] = 80;
				break;
			case 2:
				ha0->data.f32[i] = -20;
				break;
			case 3:
				ha0->data.f32[i] = 20;
				break;
			default:
				ha0->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 12 - 6;
				break;
		}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha0), TENSOR_LIST(a0), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a0), TENSOR_LIST(b0), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha0), TENSOR_LIST(tb0), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b0), TENSOR_LIST(hb0), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb0->data.f32, hb0->data.f32, 32 * 32, 1e-6, "sigmoid from mps should match from CPU for the length %% 1024 == 0 case");
	ccv_nnc_tensor_free(a0);
	ccv_nnc_tensor_free(b0);
	ccv_nnc_tensor_free(ha0);
	ccv_nnc_tensor_free(hb0);
	ccv_nnc_tensor_free(tb0);

	ccv_nnc_tensor_t* const a1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 61), 0);
	ccv_nnc_tensor_t* const b1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 17, 61), 0);
	ccv_nnc_tensor_t* const ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 61), 0);
	ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 61), 0);
	ccv_nnc_tensor_t* const tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 61), 0);
	for (i = 0; i < 17 * 61; i++)
		switch (i % 6)
		{
			case 0:
				ha1->data.f32[i] = -80;
				break;
			case 1:
				ha1->data.f32[i] = 80;
				break;
			case 2:
				ha1->data.f32[i] = -20;
				break;
			case 3:
				ha1->data.f32[i] = 20;
				break;
			default:
				ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 12 - 6;
				break;
		}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1), TENSOR_LIST(a1), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a1), TENSOR_LIST(b1), 0);
	ccv_nnc_cmd_exec(CMD_SIGMOID_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1), TENSOR_LIST(tb1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b1), TENSOR_LIST(hb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb1->data.f32, 17 * 61, 1e-6, "sigmoid from mps should match from CPU for the tail case");
	ccv_nnc_tensor_free(a1);
	ccv_nnc_tensor_free(b1);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(hb1);
	ccv_nnc_tensor_free(tb1);
}


TEST_CASE("compare sigmoid gradient with mps")
{
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
	const ccv_nnc_cmd_t layer_norm = ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=1e-6,.scale=0.75,.elementwise_affine=1,.count=3,.axis={1,2,3}}}), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, layer_norm, TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
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
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, layer_norm, TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
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

TEST_CASE("compare layer norm with mps without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-6, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
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
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-6, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 2 * 2 * 10);
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
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7, 1), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
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
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7, 1), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
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

TEST_CASE("compare group norm with mps without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 16, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 16, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7, 0), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
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
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 16, 2, 10), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 4, 2, 10), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 4, 2, 10), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-7, 0), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 16 * 2 * 10);
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

TEST_CASE("compare rmsnorm with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	const ccv_nnc_cmd_t rmsnorm = ccv_nnc_cmd(CCV_NNC_RMSNORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.rmsnorm={.epsilon=1e-6,.scale=0.75,.elementwise_affine=1,.count=3,.axis={1,2,3}}}), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, rmsnorm, TENSOR_SYMBOL_LIST(bx, scale), TENSOR_SYMBOL_LIST(by, saved_inv_std), "rmsnorm");
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
	for (i = 0; i < 1 * 2 * 2 * 10; i++)
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale)), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, 10), "scale");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, rmsnorm, TENSOR_SYMBOL_LIST(cx, cscale), TENSOR_SYMBOL_LIST(cy, csaved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 2 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	// Note that MPS and my other implementations treat epsilon differently.
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cy_tensor->data.f32, 2 * 2 * 2 * 10, 1e-4, "rmsnorm result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
}

TEST_CASE("compare rmsnorm with mps without scale")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host x");
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "host y");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(bx), "transfer x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RMSNORM_FORWARD(1e-6, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_inv_std), "rmsnorm");
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
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, 10), "y");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_RMSNORM_FORWARD(1e-6, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, xdata, sizeof(float) * 2 * 2 * 2 * 10);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	// Note that MPS and my other implementations treat epsilon differently.
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cy_tensor->data.f32, 2 * 2 * 2 * 10, 1e-4, "rmsnorm result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
}

TEST_CASE("compare rmsnorm with mps in half precision for qwen shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 1;
	const int sequence_length = 9;
	const int channels = 3584;
	const int element_count = batch_size * sequence_length * channels;
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, channels), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, channels), "b");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 1, 1, channels), "scale");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RMSNORM_FORWARD(1e-6, 1, 2), TENSOR_SYMBOL_LIST(a, scale), TENSOR_SYMBOL_LIST(b, saved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	for (i = 0; i < element_count; i++)
		x_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 200;
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const scale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	for (i = 0; i < channels; i++)
		scale_tensor->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	ccv_nnc_tensor_t* const scale16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, channels), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(scale_tensor), TENSOR_LIST(scale16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(scale16_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale)), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 1, channels), "scale");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_RMSNORM_FORWARD(1e-6, 1, 2), TENSOR_SYMBOL_LIST(cx, cscale), TENSOR_SYMBOL_LIST(cy, csaved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * element_count);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scale_tensor->data.f32, sizeof(float) * channels);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const cy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cy);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y_tensor->data.f32, cy_tensor->data.f32, element_count, 1e-2, "qwen-like half precision rmsnorm result from mps should match the CPU reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(scale_tensor);
	ccv_nnc_tensor_free(scale16_tensor);
}

TEST_CASE("compare layer norm with mps in bfloat precision through mfa and graph")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int sequence_length = 3;
	const int channels = 257;
	const int rows = batch_size * sequence_length;
	const int element_count = rows * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const saved_mean = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const saved_inv_std = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const hbias16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hmean = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const hinv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const hy16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hmean16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const hinv16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected_mean = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_inv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_y16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected_mean16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_inv16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 21);
	int i, j;
	for (i = 0; i < element_count; i++)
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
	for (i = 0; i < channels; i++)
	{
		hscale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
		hbias->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 0.25;
	}
	ccv_float_to_bfloat(ha->data.f32, (uint16_t*)ha16bf->data.f16, element_count);
	ccv_float_to_bfloat(hscale->data.f32, (uint16_t*)hscale16bf->data.f16, channels);
	ccv_float_to_bfloat(hbias->data.f32, (uint16_t*)hbias16bf->data.f16, channels);
	ccv_bfloat_to_float((uint16_t*)ha16bf->data.f16, ha->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hscale16bf->data.f16, hscale->data.f32, channels);
	ccv_bfloat_to_float((uint16_t*)hbias16bf->data.f16, hbias->data.f32, channels);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16bf, hscale16bf, hbias16bf), TENSOR_LIST(a, scale, bias), 0);
	for (i = 0; i < rows; i++)
	{
		float mean = 0;
		for (j = 0; j < channels; j++)
			mean += ha->data.f32[i * channels + j];
		mean = mean / channels;
		float variance = 0;
		for (j = 0; j < channels; j++)
		{
			const float centered = ha->data.f32[i * channels + j] - mean;
			variance += centered * centered;
		}
		const float inv_std = 1.0f / sqrtf(variance / channels + 1e-6f);
		expected_mean->data.f32[i] = mean;
		expected_inv->data.f32[i] = inv_std;
		for (j = 0; j < channels; j++)
			expected_y->data.f32[i * channels + j] = (ha->data.f32[i * channels + j] - mean) * inv_std * hscale->data.f32[j] + hbias->data.f32[j];
	}
	ccv_float_to_bfloat(expected_y->data.f32, (uint16_t*)expected_y16bf->data.f16, element_count);
	ccv_float_to_bfloat(expected_mean->data.f32, (uint16_t*)expected_mean16bf->data.f16, rows);
	ccv_float_to_bfloat(expected_inv->data.f32, (uint16_t*)expected_inv16bf->data.f16, rows);
	ccv_bfloat_to_float((uint16_t*)expected_y16bf->data.f16, expected_y->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)expected_mean16bf->data.f16, expected_mean->data.f32, rows);
	ccv_bfloat_to_float((uint16_t*)expected_inv16bf->data.f16, expected_inv->data.f32, rows);
	ccv_nnc_cmd_t cmd = CMD_LAYER_NORM_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, scale, bias), TENSOR_LIST(b, saved_mean, saved_inv_std), 0), "bfloat layer norm mfa should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, saved_mean, saved_inv_std), TENSOR_LIST(hy16bf, hmean16bf, hinv16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hmean16bf->data.f16, hmean->data.f32, rows);
	ccv_bfloat_to_float((uint16_t*)hinv16bf->data.f16, hinv->data.f32, rows);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected_y->data.f32, element_count, 2e-2, "bfloat layer norm result from mfa should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hmean->data.f32, expected_mean->data.f32, rows, 2e-2, "bfloat layer norm saved mean from mfa should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hinv->data.f32, expected_inv->data.f32, rows, 2e-2, "bfloat layer norm saved inv std from mfa should match fp32 reference rounded to bfloat");
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, scale, bias), TENSOR_LIST(b, saved_mean, saved_inv_std), 0), "bfloat layer norm graph fallback should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, saved_mean, saved_inv_std), TENSOR_LIST(hy16bf, hmean16bf, hinv16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hmean16bf->data.f16, hmean->data.f32, rows);
	ccv_bfloat_to_float((uint16_t*)hinv16bf->data.f16, hinv->data.f32, rows);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected_y->data.f32, element_count, 2e-2, "bfloat layer norm result from graph fallback should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hmean->data.f32, expected_mean->data.f32, rows, 2e-2, "bfloat layer norm saved mean from graph fallback should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hinv->data.f32, expected_inv->data.f32, rows, 2e-2, "bfloat layer norm saved inv std from graph fallback should match fp32 reference rounded to bfloat");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(saved_mean);
	ccv_nnc_tensor_free(saved_inv_std);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(ha16bf);
	ccv_nnc_tensor_free(hscale16bf);
	ccv_nnc_tensor_free(hbias16bf);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hmean);
	ccv_nnc_tensor_free(hinv);
	ccv_nnc_tensor_free(hy16bf);
	ccv_nnc_tensor_free(hmean16bf);
	ccv_nnc_tensor_free(hinv16bf);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_mean);
	ccv_nnc_tensor_free(expected_inv);
	ccv_nnc_tensor_free(expected_y16bf);
	ccv_nnc_tensor_free(expected_mean16bf);
	ccv_nnc_tensor_free(expected_inv16bf);
}

TEST_CASE("compare rmsnorm with mps in bfloat precision through mfa and graph")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int sequence_length = 3;
	const int channels = 257;
	const int rows = batch_size * sequence_length;
	const int element_count = rows * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const saved_inv_std = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hinv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const hy16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hinv16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected_inv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, 1), 0);
	ccv_nnc_tensor_t* const expected_y16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected_inv16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 22);
	int i, j;
	for (i = 0; i < element_count; i++)
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
	for (i = 0; i < channels; i++)
		hscale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	ccv_float_to_bfloat(ha->data.f32, (uint16_t*)ha16bf->data.f16, element_count);
	ccv_float_to_bfloat(hscale->data.f32, (uint16_t*)hscale16bf->data.f16, channels);
	ccv_bfloat_to_float((uint16_t*)ha16bf->data.f16, ha->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hscale16bf->data.f16, hscale->data.f32, channels);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16bf, hscale16bf), TENSOR_LIST(a, scale), 0);
	for (i = 0; i < rows; i++)
	{
		float variance = 0;
		for (j = 0; j < channels; j++)
		{
			const float v = ha->data.f32[i * channels + j];
			variance += v * v;
		}
		const float inv_std = 1.0f / sqrtf(variance / channels + 1e-6f);
		expected_inv->data.f32[i] = inv_std;
		for (j = 0; j < channels; j++)
			expected_y->data.f32[i * channels + j] = ha->data.f32[i * channels + j] * inv_std * hscale->data.f32[j];
	}
	ccv_float_to_bfloat(expected_y->data.f32, (uint16_t*)expected_y16bf->data.f16, element_count);
	ccv_float_to_bfloat(expected_inv->data.f32, (uint16_t*)expected_inv16bf->data.f16, rows);
	ccv_bfloat_to_float((uint16_t*)expected_y16bf->data.f16, expected_y->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)expected_inv16bf->data.f16, expected_inv->data.f32, rows);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, scale), TENSOR_LIST(b, saved_inv_std), 0), "bfloat rmsnorm mfa should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, saved_inv_std), TENSOR_LIST(hy16bf, hinv16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hinv16bf->data.f16, hinv->data.f32, rows);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected_y->data.f32, element_count, 2e-2, "bfloat rmsnorm result from mfa should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hinv->data.f32, expected_inv->data.f32, rows, 2e-2, "bfloat rmsnorm saved inv std from mfa should match fp32 reference rounded to bfloat");
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, scale), TENSOR_LIST(b, saved_inv_std), 0), "bfloat rmsnorm graph fallback should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, saved_inv_std), TENSOR_LIST(hy16bf, hinv16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	ccv_bfloat_to_float((uint16_t*)hinv16bf->data.f16, hinv->data.f32, rows);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected_y->data.f32, element_count, 2e-2, "bfloat rmsnorm result from graph fallback should match fp32 reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hinv->data.f32, expected_inv->data.f32, rows, 2e-2, "bfloat rmsnorm saved inv std from graph fallback should match fp32 reference rounded to bfloat");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(saved_inv_std);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(ha16bf);
	ccv_nnc_tensor_free(hscale16bf);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hinv);
	ccv_nnc_tensor_free(hy16bf);
	ccv_nnc_tensor_free(hinv16bf);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_inv);
	ccv_nnc_tensor_free(expected_y16bf);
	ccv_nnc_tensor_free(expected_inv16bf);
}

TEST_CASE("compare rmsnorm cmul with affine weight using cpu ref")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_CPU_REF));
	const int token_count = 3;
	const int head_count = 4;
	const int channels = 18;
	const int row_count = token_count * head_count;
	const int element_count = row_count * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const rotation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, 1, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 31);
	int i, j;
	for (i = 0; i < element_count; i++)
		a->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
	for (i = 0; i < token_count * channels; i++)
		rotation->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	for (i = 0; i < channels; i++)
		scale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	for (i = 0; i < row_count; i++)
	{
		float square_sum = 0;
		for (j = 0; j < channels; j++)
		{
			const float v = a->data.f32[i * channels + j];
			square_sum += v * v;
		}
		const float inv_rms = 1.0f / sqrtf(square_sum / channels + 1e-6f);
		const int rotation_offset = (i / head_count) * channels;
		for (j = 0; j < channels; j += 2)
		{
			const float real = a->data.f32[i * channels + j] * inv_rms * scale->data.f32[j];
			const float imag = a->data.f32[i * channels + j + 1] * inv_rms * scale->data.f32[j + 1];
			const float rotation_real = rotation->data.f32[rotation_offset + j];
			const float rotation_imag = rotation->data.f32[rotation_offset + j + 1];
			expected->data.f32[i * channels + j] = real * rotation_real - imag * rotation_imag;
			expected->data.f32[i * channels + j + 1] = real * rotation_imag + imag * rotation_real;
		}
	}
	memcpy(inplace->data.f32, a->data.f32, sizeof(float) * element_count);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_CMUL_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation, scale), TENSOR_LIST(b), 0), "rmsnorm cmul cpu ref should run with affine weight");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, expected->data.f32, element_count, 1e-6, "rmsnorm cmul cpu ref should match expected result");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(inplace, rotation, scale), TENSOR_LIST(inplace), 0), "rmsnorm cmul cpu ref should run in-place on first input with affine weight");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, inplace->data.f32, expected->data.f32, element_count, 1e-6, "rmsnorm cmul cpu ref in-place result should match expected result");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(rotation);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(inplace);
	ccv_nnc_tensor_free(expected);
}

TEST_CASE("compare rmsnorm cmul with affine weight at production shape through mfa and graph")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	const int token_count = 17;
	const int head_count = 64;
	const int channels = 512;
	const int element_count = token_count * head_count * channels;
	const int rotation_count = token_count * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const rotation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, 1, channels), 0);
	ccv_nnc_tensor_t* const exact_rotation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const shared_rotation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hrotation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, 1, channels), 0);
	ccv_nnc_tensor_t* const hexact_rotation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hshared_rotation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hy16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const expected16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const exact_expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const exact_expected16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const shared_expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const shared_expected16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 32);
	int i;
	for (i = 0; i < element_count; i++)
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
	for (i = 0; i < rotation_count; i++)
		hrotation->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	for (i = 0; i < element_count; i++)
		hexact_rotation->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	for (i = 0; i < channels; i++)
		hscale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	memcpy(hshared_rotation->data.f32, hrotation->data.f32, sizeof(float) * channels);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ha16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_CMUL_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hrotation, hscale), TENSOR_LIST(expected), 0), "rmsnorm cmul cpu ref should produce the affine production-shape reference");
	ccv_nnc_cmd_t exact_cmd = CMD_RMSNORM_CMUL_FORWARD(1e-4, 1, 2);
	exact_cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(exact_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hexact_rotation, hscale), TENSOR_LIST(exact_expected), 0), "rmsnorm cmul cpu ref should produce the affine exact-rotation reference");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hshared_rotation, hscale), TENSOR_LIST(shared_expected), 0), "rmsnorm cmul cpu ref should support affine weight with standard rotation broadcasting");
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected), TENSOR_LIST(expected16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected16), TENSOR_LIST(expected), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(exact_expected), TENSOR_LIST(exact_expected16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(exact_expected16), TENSOR_LIST(exact_expected), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(shared_expected), TENSOR_LIST(shared_expected16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(shared_expected16), TENSOR_LIST(shared_expected), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hrotation, hexact_rotation, hshared_rotation, hscale), TENSOR_LIST(a, rotation, exact_rotation, shared_rotation, scale), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	exact_cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation, scale), TENSOR_LIST(b), 0), "mixed-precision affine rmsnorm cmul mfa should run at the production shape");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "mixed-precision affine rmsnorm cmul result from mfa should match fp32 reference rounded to half");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation, scale), TENSOR_LIST(a), 0), "mixed-precision affine rmsnorm cmul mfa should run in-place on first input");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "mixed-precision affine in-place rmsnorm cmul result from mfa should match fp32 reference rounded to half");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(exact_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, exact_rotation, scale), TENSOR_LIST(b), 0), "mixed-precision affine rmsnorm cmul mfa should specialize a second pipeline for exact rotation and a different epsilon");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, exact_expected->data.f32, element_count, 1e-2, "affine exact-rotation rmsnorm cmul result from the second mfa pipeline should match fp32 reference rounded to half");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, shared_rotation, scale), TENSOR_LIST(b), 0), "affine rmsnorm cmul should use the graph fallback for a standard broadcast outside the native surface");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, shared_expected->data.f32, element_count, 1e-2, "standard-broadcast affine rmsnorm cmul result from graph fallback should match fp32 reference rounded to half");
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation, scale), TENSOR_LIST(b), 0), "mixed-precision affine rmsnorm cmul graph fallback should run at the production shape");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "mixed-precision affine rmsnorm cmul result from graph fallback should match fp32 reference rounded to half");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation, scale), TENSOR_LIST(a), 0), "mixed-precision affine rmsnorm cmul graph fallback should run in-place on first input");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "mixed-precision affine in-place rmsnorm cmul result from graph fallback should match fp32 reference rounded to half");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(rotation);
	ccv_nnc_tensor_free(exact_rotation);
	ccv_nnc_tensor_free(shared_rotation);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hrotation);
	ccv_nnc_tensor_free(hexact_rotation);
	ccv_nnc_tensor_free(hshared_rotation);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy16);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(expected16);
	ccv_nnc_tensor_free(exact_expected);
	ccv_nnc_tensor_free(exact_expected16);
	ccv_nnc_tensor_free(shared_expected);
	ccv_nnc_tensor_free(shared_expected16);
}

TEST_CASE("compare rmsnorm cmul with mps through two-row reuse")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_CPU_REF));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	const int token_count = 4096;
	const int head_count = 2;
	const int channels = 18;
	const int element_count = token_count * head_count * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const rotation = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, token_count, 1, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hrotation = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, 1, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, token_count, head_count, channels), 0);
	ccv_nnc_tensor_t* const hy16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, token_count, head_count, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 33);
	int i;
	for (i = 0; i < element_count; i++)
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
	for (i = 0; i < token_count * channels; i++)
		hrotation->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ha16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_CMUL_FORWARD(1e-6, 0, 2);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hrotation), TENSOR_LIST(expected), 0), "rmsnorm cmul cpu ref should produce the two-row reuse reference");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hrotation), TENSOR_LIST(a, rotation), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, rotation), TENSOR_LIST(b), 0), "mixed-precision rmsnorm cmul mfa should run through two-row reuse");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 5e-3, "two-row reuse result should match fp32 reference rounded to half");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(rotation);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hrotation);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy16);
}

TEST_CASE("compare rmsnorm gated with cpu ref")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_CPU_REF));
	const int rows = 4;
	const int channels = 17;
	const int element_count = rows * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const inplace = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 23);
	int i, j;
	for (i = 0; i < element_count; i++)
	{
		a->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
		gate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 5;
		inplace->data.f32[i] = a->data.f32[i];
	}
	for (i = 0; i < channels; i++)
		scale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	for (i = 0; i < rows; i++)
	{
		float variance = 0;
		for (j = 0; j < channels; j++)
		{
			const float v = a->data.f32[i * channels + j];
			variance += v * v;
		}
		const float inv_std = 1.0f / sqrtf(variance / channels + 1e-6f);
		for (j = 0; j < channels; j++)
		{
			const float z = gate->data.f32[i * channels + j];
			const float ez = expf(-fabsf(z));
			const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
			expected->data.f32[i * channels + j] = a->data.f32[i * channels + j] * inv_std * scale->data.f32[j] * z * sigmoid;
		}
	}
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_GATED_FORWARD(1e-6, 1, 1);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, scale), TENSOR_LIST(b), 0), "rmsnorm gated cpu ref should run");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, expected->data.f32, element_count, 1e-6, "rmsnorm gated cpu ref should match expected result");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(inplace, gate, scale), TENSOR_LIST(inplace), 0), "rmsnorm gated cpu ref should run in-place on first input");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, inplace->data.f32, expected->data.f32, element_count, 1e-6, "rmsnorm gated cpu ref in-place result should match expected result");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(inplace);
	ccv_nnc_tensor_free(expected);
}

TEST_CASE("compare rmsnorm gated with mps in mixed bfloat precision through mfa and graph")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int sequence_length = 3;
	const int channels = 257;
	const int rows = batch_size * sequence_length;
	const int element_count = rows * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hgate16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hy16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, sequence_length, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 24);
	int i, j;
	for (i = 0; i < element_count; i++)
	{
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
		hgate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 4;
	}
	for (i = 0; i < channels; i++)
		hscale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	ccv_float_to_bfloat(ha->data.f32, (uint16_t*)ha16bf->data.f16, element_count);
	ccv_bfloat_to_float((uint16_t*)ha16bf->data.f16, ha->data.f32, element_count);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgate), TENSOR_LIST(hgate16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgate16), TENSOR_LIST(hgate), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16bf, hgate16, hscale), TENSOR_LIST(a, gate, scale), 0);
	for (i = 0; i < rows; i++)
	{
		float variance = 0;
		for (j = 0; j < channels; j++)
		{
			const float v = ha->data.f32[i * channels + j];
			variance += v * v;
		}
		const float inv_std = 1.0f / sqrtf(variance / channels + 1e-6f);
		for (j = 0; j < channels; j++)
		{
			const float z = hgate->data.f32[i * channels + j];
			const float ez = expf(-fabsf(z));
			const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
			expected->data.f32[i * channels + j] = ha->data.f32[i * channels + j] * inv_std * hscale->data.f32[j] * z * sigmoid;
		}
	}
	ccv_float_to_bfloat(expected->data.f32, (uint16_t*)expected16bf->data.f16, element_count);
	ccv_bfloat_to_float((uint16_t*)expected16bf->data.f16, expected->data.f32, element_count);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_GATED_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, scale), TENSOR_LIST(b), 0), "mixed bfloat rmsnorm gated mfa should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-1, "mixed bfloat rmsnorm gated result from mfa should match fp32 reference rounded to bfloat");
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, scale), TENSOR_LIST(b), 0), "mixed bfloat rmsnorm gated graph fallback should run");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hy16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hy16bf->data.f16, hy->data.f32, element_count);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-1, "mixed bfloat rmsnorm gated result from graph fallback should match fp32 reference rounded to bfloat");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(ha16bf);
	ccv_nnc_tensor_free(hgate16);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy16bf);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(expected16bf);
}

TEST_CASE("compare rmsnorm gated with mps across non-last axis without scale")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int sequence_length = 5;
	const int channels = 7;
	const int element_count = batch_size * sequence_length * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 26);
	int i, j, k;
	for (i = 0; i < element_count; i++)
	{
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
		hgate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 4;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgate), TENSOR_LIST(a, gate), 0);
	for (i = 0; i < batch_size; i++)
		for (k = 0; k < channels; k++)
		{
			float variance = 0;
			for (j = 0; j < sequence_length; j++)
			{
				const float v = ha->data.f32[(i * sequence_length + j) * channels + k];
				variance += v * v;
			}
			const float inv_std = 1.0f / sqrtf(variance / sequence_length + 1e-6f);
			for (j = 0; j < sequence_length; j++)
			{
				const int offset = (i * sequence_length + j) * channels + k;
				const float z = hgate->data.f32[offset];
				const float ez = expf(-fabsf(z));
				const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
				expected->data.f32[offset] = ha->data.f32[offset] * inv_std * z * sigmoid;
			}
		}
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_GATED_FORWARD(1e-6, 0, 1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate), TENSOR_LIST(y), 0), "rmsnorm gated mps should run without scale across a non-last axis");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-5, "rmsnorm gated mps result without scale across a non-last axis should match fp32 reference");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(expected);
}

TEST_CASE("compare rmsnorm gated with mps in half precision in-place")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 1;
	const int sequence_length = 4;
	const int channels = 513;
	const int rows = batch_size * sequence_length;
	const int element_count = rows * channels;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const gate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hgate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hgate16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hscale16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, channels), 0);
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const hy16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, sequence_length, channels), 0);
	ccv_nnc_tensor_t* const expected16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, batch_size, sequence_length, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 25);
	int i, j;
	for (i = 0; i < element_count; i++)
	{
		ha->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 3;
		hgate->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 4;
	}
	for (i = 0; i < channels; i++)
		hscale->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 2;
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hgate, hscale), TENSOR_LIST(ha16, hgate16, hscale16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hgate16, hscale16), TENSOR_LIST(ha, hgate, hscale), 0);
	for (i = 0; i < rows; i++)
	{
		float variance = 0;
		for (j = 0; j < channels; j++)
		{
			const float v = ha->data.f32[i * channels + j];
			variance += v * v;
		}
		const float inv_std = 1.0f / sqrtf(variance / channels + 1e-6f);
		for (j = 0; j < channels; j++)
		{
			const float z = hgate->data.f32[i * channels + j];
			const float ez = expf(-fabsf(z));
			const float sigmoid = z >= 0 ? 1.f / (1.f + ez) : ez / (1.f + ez);
			expected->data.f32[i * channels + j] = ha->data.f32[i * channels + j] * inv_std * hscale->data.f32[j] * z * sigmoid;
		}
	}
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected), TENSOR_LIST(expected16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected16), TENSOR_LIST(expected), 0);
	ccv_nnc_cmd_t cmd = CMD_RMSNORM_GATED_FORWARD(1e-6, 1, 2);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hgate16, hscale16), TENSOR_LIST(a, gate, scale), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, scale), TENSOR_LIST(a), 0), "half rmsnorm gated mfa should run in-place on first input");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "half rmsnorm gated in-place result from mfa should match fp32 reference rounded to half");
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hgate16, hscale16), TENSOR_LIST(a, gate, scale), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, scale), TENSOR_LIST(a), 0), "half rmsnorm gated graph fallback should run in-place on first input");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(hy16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hy16), TENSOR_LIST(hy), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected->data.f32, element_count, 1e-2, "half rmsnorm gated in-place result from graph fallback should match fp32 reference rounded to half");
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgate);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hgate16);
	ccv_nnc_tensor_free(hscale16);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hy16);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(expected16);
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

TEST_CASE("compare add gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), "y");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 1, 3), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(a, b), "transfer");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(0.5, 0.2), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "add");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(x, y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dc = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, c);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(x, x_tensor), KV(y, y_tensor)), TENSOR_SYMBOL_LIST(dx, dy), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 5 * 1 * 3; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* dct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		dct->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dc_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dc);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dct), TENSOR_LIST(dc_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* zt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(zt), 0);
	ccv_nnc_tensor_t* dxt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_tensor_t* dyt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(dct, x_tensor, y_tensor, zt), TENSOR_LIST(dxt, dyt), 0);
	ccv_nnc_tensor_t* dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* dy_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dy);
	REQUIRE_TENSOR_EQ(dxt, dx_tensor, "backward pass should match");
	REQUIRE_TENSOR_EQ(dyt, dy_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dct);
	ccv_nnc_tensor_free(zt);
	ccv_nnc_tensor_free(dxt);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare add gradient with mps no dyt ")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), "y");
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 1, 3), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 5, 5, 3), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x, y), TENSOR_SYMBOL_LIST(a, b), "transfer");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_ADD_FORWARD(0.5, 0.2), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "add");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(c), TENSOR_SYMBOL_LIST(x, y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 1, 3), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_symbol_t dc = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, c);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(x, x_tensor), KV(y, y_tensor)), TENSOR_SYMBOL_LIST(dx, dy), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 5 * 1 * 3; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* dct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	for (i = 0; i < 10 * 5 * 5 * 3; i++)
		dct->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dc_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dc);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dct), TENSOR_LIST(dc_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* zt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(zt), 0);
	ccv_nnc_tensor_t* dxt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(dct, x_tensor, y_tensor, zt), TENSOR_LIST(dxt, 0), 0);
	ccv_nnc_tensor_t* dx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	REQUIRE_TENSOR_EQ(dxt, dx_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dct);
	ccv_nnc_tensor_free(zt);
	ccv_nnc_tensor_free(dxt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("broadcasting semantics for add backward mps (a,b)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(gda, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, gdb), TENSOR_LIST(da, db), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, b), TENSOR_LIST(dat, dbt), 0);
	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gda);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("broadcasting semantics for add backward mps (a, nil)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, ), TENSOR_LIST(gda, ), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, ), TENSOR_LIST(da, ), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, ), TENSOR_LIST(dat, ), 0);
	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gda);
}

TEST_CASE("broadcasting semantics for add backward mps (nil,b)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(0, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gdb), TENSOR_LIST(db), 0);
	ccv_nnc_cmd_exec(CMD_ADD_BACKWARD(0.5, 0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(0, dbt), 0);
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gdb);
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

TEST_CASE("compare ewsum with mps in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const ha16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hb16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hc16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hd16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	int i;
	for (i = 0; i < 100; i++)
	{
		ha->data.f32[i] = 1;
		hb->data.f32[i] = 0.5;
		hc->data.f32[i] = 0.25;
		gd->data.f32[i] = 1.75;
	}
	ccv_float_to_bfloat(ha->data.f32, (uint16_t*)ha16bf->data.f16, 100);
	ccv_float_to_bfloat(hb->data.f32, (uint16_t*)hb16bf->data.f16, 100);
	ccv_float_to_bfloat(hc->data.f32, (uint16_t*)hc16bf->data.f16, 100);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16bf, hb16bf, hc16bf), TENSOR_LIST(a, b, c), 0);
	ccv_nnc_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(d), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(hd16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hd16bf->data.f16, hd->data.f32, 100);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hd->data.f32, gd->data.f32, 100, 1e-3, "ewsum result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(ha16bf);
	ccv_nnc_tensor_free(hb16bf);
	ccv_nnc_tensor_free(hc16bf);
	ccv_nnc_tensor_free(hd16bf);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("compare ewsum with mps in bfloat precision with MFA disabled")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	ccv_nnc_tensor_t* const ha16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hb16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hc16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const hd16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 100), 0);
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100), 0);
	int i;
	for (i = 0; i < 100; i++)
	{
		ha->data.f32[i] = 1;
		hb->data.f32[i] = 0.5;
		hc->data.f32[i] = 0.25;
		gd->data.f32[i] = 1.75;
	}
	ccv_float_to_bfloat(ha->data.f32, (uint16_t*)ha16bf->data.f16, 100);
	ccv_float_to_bfloat(hb->data.f32, (uint16_t*)hb16bf->data.f16, 100);
	ccv_float_to_bfloat(hc->data.f32, (uint16_t*)hc16bf->data.f16, 100);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16bf, hb16bf, hc16bf), TENSOR_LIST(a, b, c), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(d), 0);
	if (!(old_flags & CCV_NNC_DISABLE_MFA))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(hd16bf), 0);
	ccv_bfloat_to_float((uint16_t*)hd16bf->data.f16, hd->data.f32, 100);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hd->data.f32, gd->data.f32, 100, 1e-3, "ewsum result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(ha16bf);
	ccv_nnc_tensor_free(hb16bf);
	ccv_nnc_tensor_free(hc16bf);
	ccv_nnc_tensor_free(hd16bf);
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

TEST_CASE("scalar mul [[1, 2, 3], [4, 5, 6]] * 0.5, int")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 3), 0);
	ccv_nnc_tensor_t* const ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 3), 0);
	a->data.i32[0] = 1;
	a->data.i32[1] = 2;
	a->data.i32[2] = 3;
	a->data.i32[3] = 4;
	a->data.i32[4] = 5;
	a->data.i32[5] = 6;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2, 3), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(ga), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ct), 0);
	REQUIRE_TENSOR_EQ(c, ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ct);
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


TEST_CASE("mps mse mean loss forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 1), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* tc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(tc), 0);
	REQUIRE_TENSOR_EQ(tc, hc, "MPS computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(tc);
}

TEST_CASE("mps mse sum loss forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 1), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* tc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(tc), 0);

	REQUIRE_TENSOR_EQ(tc, hc, "MPS computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(tc);
}

TEST_CASE("mps mse mean loss backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS) &&
	ccv_nnc_cmd_ok(CCV_NNC_MSE_BACKWARD, CCV_NNC_BACKEND_MPS));

	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* db = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hg->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hda, hdb), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_MEAN), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(da, db), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* tdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, db), TENSOR_LIST(tda, tdb), 0);

	REQUIRE_TENSOR_EQ(tda, hda, "MPS computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdb, hdb, "MPS computed output should be the same as CPU computed ones");

	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdb);
}

TEST_CASE("mps mse sum loss backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MSE_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* db = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hg->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hda, hdb), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(da, db), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* tdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, db), TENSOR_LIST(tda, tdb), 0);
	REQUIRE_TENSOR_EQ(tda, hda, "MPS computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdb, hdb, "MPS computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdb);
}


TEST_CASE("mps mse sum loss backward (no output db)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MSE_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hg->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hda, 0), 0);
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(CCV_NNC_MSE_REDUCE_SUM), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(da, 0), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* tdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, 0), TENSOR_LIST(tda, 0), 0);
	REQUIRE_TENSOR_EQ(tda, hda, "MPS computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdb);
}

TEST_CASE("mps leaky relu gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LEAKY_RELU_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_LEAKY_RELU_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LEAKY_RELU_FORWARD(0.2), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "leaky relu");
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
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_LEAKY_RELU_FORWARD(0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_LEAKY_RELU_BACKWARD(0.2), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, 0, y_tensor), TENSOR_LIST(tdx_tensor), 0);
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

TEST_CASE("compare layer norm gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	const ccv_nnc_cmd_t layer_norm = ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=1e-4,.scale=0.75,.elementwise_affine=1,.count=3,.axis={1,2,3}}}), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, layer_norm, TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * 2 * 2 * LN_DIM];
	float biasdata[1 * 2 * 2 * LN_DIM];
	for (i = 0; i < 1 * 2 * 2 * LN_DIM; i++)
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 2 * 2 * LN_DIM; i++)
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);

	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dbbias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bias));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dbias_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor, dbbias_tensor), TENSOR_LIST(dscale_tensor, dbias_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, layer_norm, TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale, cbias), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_tensor_symbol_t dcbias = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cbias);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cbias);
	memcpy(cbias_tensor->data.f32, biasdata, sizeof(float) * 1 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);
	ccv_nnc_tensor_t* const dcbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcbias);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	REQUIRE_TENSOR_EQ(dscale_tensor, dcscale_tensor, "layer norm scale gradient result from mps should match the one from reference implementation");
	REQUIRE_TENSOR_EQ(dbias_tensor, dcbias_tensor, "layer norm bias gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
	ccv_nnc_tensor_free(dbias_tensor);
}

TEST_CASE("compare layer norm gradient with mps (no bias)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 1, 1, 2, 3), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * 2 * 2 * LN_DIM];
	float biasdata[1 * 2 * 2 * LN_DIM];
	for (i = 0; i < 1 * 2 * 2 * LN_DIM; i++)
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 2 * 2 * LN_DIM; i++)
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);

	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor), TENSOR_LIST(dscale_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 1, 1, 2, 3), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	REQUIRE_TENSOR_EQ(dscale_tensor, dcscale_tensor, "layer norm scale gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
}

TEST_CASE("compare layer norm gradient with mps without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("compare layer norm gradient with mps (no bias) without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_LAYER_NORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("compare rmsnorm gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	const ccv_nnc_cmd_t rmsnorm = ccv_nnc_cmd(CCV_NNC_RMSNORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.rmsnorm={.epsilon=1e-4,.scale=0.75,.elementwise_affine=1,.count=3,.axis={1,2,3}}}), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, rmsnorm, TENSOR_SYMBOL_LIST(bx, scale), TENSOR_SYMBOL_LIST(by, saved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * 2 * 2 * LN_DIM];
	for (i = 0; i < 1 * 2 * 2 * LN_DIM; i++)
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);

	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale)), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor), TENSOR_LIST(dscale_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, 2, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, rmsnorm, TENSOR_SYMBOL_LIST(cx, cscale), TENSOR_SYMBOL_LIST(cy, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	REQUIRE_TENSOR_EQ(dscale_tensor, dcscale_tensor, "layer norm scale gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
}

TEST_CASE("compare rmsnorm gradient with mps without scale")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RMSNORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_inv_std), "rmsnorm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * 2 * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 2, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, 1, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_RMSNORM_FORWARD(1e-4, 0, 1, 2, 3), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_inv_std), "layer_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * 2 * 2 * LN_DIM);
	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_TENSOR_EQ(dx_tensor, dcx_tensor, "layer norm gradient result from mps should match the one from reference implementation");
	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("mps backward convolution in nchw format")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_BACKWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, g->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, g->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / OUTPUT_DIM; // (OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM);
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gdw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gao = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* ggo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* gho = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbiaso = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, OUTPUT_DIM, 1, 1), 0);
	ccv_nnc_tensor_t* gdwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gdbiaso = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, OUTPUT_DIM, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, g), TENSOR_LIST(ga, gw, gg), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gw, gg), TENSOR_LIST(gao, gwo, ggo), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;

	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);

	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ggo, gao, gwo), TENSOR_LIST(gho, gdwo, gdbiaso), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ggo, gao, gwo), TENSOR_LIST(gho, gdwo, gdbiaso), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, OUTPUT_DIM, 1, 1), 0);

	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gho, gdwo, gdbiaso), TENSOR_LIST(gh, gdw, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdw, gdbias), TENSOR_LIST(ch, cdw, cdbias), 0);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw->data.f32, cdw->data.f32, INPUT_DIM * OUTPUT_DIM * KERNEL_SIZE * KERNEL_SIZE, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias->data.f32, cdbias->data.f32, OUTPUT_DIM, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, ch->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(gao);
	ccv_nnc_tensor_free(ggo);
	ccv_nnc_tensor_free(gho);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(gbiaso);
	ccv_nnc_tensor_free(gdwo);
	ccv_nnc_tensor_free(gdbiaso);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gdbias);
	ccv_nnc_tensor_free(gdw);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ch);
	ccv_nnc_tensor_free(cdw);
	ccv_nnc_tensor_free(cdbias);
}

TEST_CASE("mps backward convolution in nhwc format")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_BACKWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, g->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, g->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / OUTPUT_DIM; // (OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM);
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gdw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gdwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, g), TENSOR_LIST(ga, gw, gg), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gw), TENSOR_LIST(gwo), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;

	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);

	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(gg, ga, gwo), TENSOR_LIST(gh, gdwo, gdbias), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(gg, ga, gwo), TENSOR_LIST(gh, gdwo, gdbias), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1,  OUTPUT_DIM), 0);
	
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gdwo), TENSOR_LIST(gdw), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdw, gdbias), TENSOR_LIST(ch, cdw, cdbias), 0);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw->data.f32, cdw->data.f32, INPUT_DIM * OUTPUT_DIM * KERNEL_SIZE * KERNEL_SIZE, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias->data.f32, cdbias->data.f32, OUTPUT_DIM, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, ch->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 1e-4, "output from mps should match from CPU");

	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(gdwo);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gdbias);
	ccv_nnc_tensor_free(gdw);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ch);
	ccv_nnc_tensor_free(cdw);
	ccv_nnc_tensor_free(cdbias);
}

TEST_CASE("mps backward convolution in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int batch_size = 2;
	const int input_size = 8;
	const int kernel_size = 3;
	const int output_size = 6;
	const int input_dim = 4;
	const int output_dim = 8;
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_BACKWARD(1, output_dim, kernel_size, kernel_size, input_dim);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, a->info, g->info);
	assert(ccv_nnc_hint_verify(hint, cmd.info, a->info, g->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 12);
	int i;
	for (i = 0; i < input_dim * kernel_size * kernel_size * output_dim; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) / (input_dim * kernel_size * kernel_size);
	for (i = 0; i < input_size * input_size * input_dim * batch_size; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (i = 0; i < output_size * output_size * output_dim * batch_size; i++)
		g->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) - 0.5) / output_dim;
	const int input_count = batch_size * input_size * input_size * input_dim;
	const int gradient_count = batch_size * output_size * output_size * output_dim;
	const int weight_count = output_dim * kernel_size * kernel_size * input_dim;
	ccv_nnc_tensor_t* a16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* w16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* g16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_float_to_bfloat(a->data.f32, (uint16_t*)a16bf->data.f16, input_count);
	ccv_float_to_bfloat(w->data.f32, (uint16_t*)w16bf->data.f16, weight_count);
	ccv_float_to_bfloat(g->data.f32, (uint16_t*)g16bf->data.f16, gradient_count);
	ccv_bfloat_to_float((uint16_t*)a16bf->data.f16, a->data.f32, input_count);
	ccv_bfloat_to_float((uint16_t*)w16bf->data.f16, w->data.f32, weight_count);
	ccv_bfloat_to_float((uint16_t*)g16bf->data.f16, g->data.f32, gradient_count);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* h16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* dw16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* dbias16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim), 0);
	ccv_float_to_bfloat(h->data.f32, (uint16_t*)h16bf->data.f16, input_count);
	ccv_float_to_bfloat(dw->data.f32, (uint16_t*)dw16bf->data.f16, weight_count);
	ccv_float_to_bfloat(dbias->data.f32, (uint16_t*)dbias16bf->data.f16, output_dim);
	ccv_bfloat_to_float((uint16_t*)h16bf->data.f16, h->data.f32, input_count);
	ccv_bfloat_to_float((uint16_t*)dw16bf->data.f16, dw->data.f32, weight_count);
	ccv_bfloat_to_float((uint16_t*)dbias16bf->data.f16, dbias->data.f32, output_dim);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, output_size, output_size, output_dim), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, output_dim, input_dim, kernel_size, kernel_size), 0);
	ccv_nnc_tensor_t* gdw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* gdwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, output_dim, input_dim, kernel_size, kernel_size), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 1, 1, output_dim), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a16bf, w16bf, g16bf), TENSOR_LIST(ga, gw, gg), 0);
	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	ccv_nnc_cmd_exec(transform, ccv_nnc_no_hint, 0, TENSOR_LIST(gw), TENSOR_LIST(gwo), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(gg, ga, gwo), TENSOR_LIST(gh, gdwo, gdbias), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(gg, ga, gwo), TENSOR_LIST(gh, gdwo, gdbias), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_cmd_exec(transform, ccv_nnc_no_hint, 0, TENSOR_LIST(gdwo), TENSOR_LIST(gdw), 0);
	ccv_nnc_tensor_t* ch16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* cdw16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* cdbias16bf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 1, 1, output_dim), 0);
	ccv_nnc_tensor_t* ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batch_size, input_size, input_size, input_dim), 0);
	ccv_nnc_tensor_t* cdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, output_dim, kernel_size, kernel_size, input_dim), 0);
	ccv_nnc_tensor_t* cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, output_dim), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdw, gdbias), TENSOR_LIST(ch16bf, cdw16bf, cdbias16bf), 0);
	ccv_bfloat_to_float((uint16_t*)ch16bf->data.f16, ch->data.f32, input_count);
	ccv_bfloat_to_float((uint16_t*)cdw16bf->data.f16, cdw->data.f32, weight_count);
	ccv_bfloat_to_float((uint16_t*)cdbias16bf->data.f16, cdbias->data.f32, output_dim);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, ch->data.f32, input_count, 5e-2, "bfloat input gradient from mps should match CPU reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw->data.f32, cdw->data.f32, weight_count, 1e-1, "bfloat weight gradient from mps should match CPU reference rounded to bfloat");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias->data.f32, cdbias->data.f32, output_dim, 1e-1, "bfloat bias gradient from mps should match CPU reference rounded to bfloat");
	ccv_nnc_tensor_free(cdbias);
	ccv_nnc_tensor_free(cdw);
	ccv_nnc_tensor_free(ch);
	ccv_nnc_tensor_free(cdbias16bf);
	ccv_nnc_tensor_free(cdw16bf);
	ccv_nnc_tensor_free(ch16bf);
	ccv_nnc_tensor_free(gdbias);
	ccv_nnc_tensor_free(gdwo);
	ccv_nnc_tensor_free(gdw);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(dbias16bf);
	ccv_nnc_tensor_free(dw16bf);
	ccv_nnc_tensor_free(h16bf);
	ccv_nnc_tensor_free(g16bf);
	ccv_nnc_tensor_free(w16bf);
	ccv_nnc_tensor_free(a16bf);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("mps backward convolution in nchw format with dilation 2, 3")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_BACKWARD(1, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM);
	cmd.info.convolution.dilation[0] = 2;
	cmd.info.convolution.dilation[1] = 3;
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_param_t modified_cmd = cmd.info;
	modified_cmd.size.dim[0] = (cmd.info.size.dim[0] - 1) * ccv_max(cmd.info.convolution.dilation[0], 1) + 1;
	modified_cmd.size.dim[1] = (cmd.info.size.dim[1] - 1) * ccv_max(cmd.info.convolution.dilation[1], 1) + 1;
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(modified_cmd, a->info, g->info);
	assert(ccv_nnc_hint_verify(hint, modified_cmd, a->info, g->info) == 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 1, OUTPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < INPUT_SIZE * INPUT_SIZE * INPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / OUTPUT_DIM; // (OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM);
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gdw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 1, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gao = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* ggo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* gho = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbiaso = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, OUTPUT_DIM, 1, 1), 0);
	ccv_nnc_tensor_t* gdwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gdbiaso = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, OUTPUT_DIM, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, g), TENSOR_LIST(ga, gw, gg), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gw, gg), TENSOR_LIST(gao, gwo, ggo), 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;

	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);

	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ggo, gao, gwo), TENSOR_LIST(gho, gdwo, gdbiaso), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ggo, gao, gwo), TENSOR_LIST(gho, gdwo, gdbiaso), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, OUTPUT_DIM, 1, 1), 0);

	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gho, gdwo, gdbiaso), TENSOR_LIST(gh, gdw, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdw, gdbias), TENSOR_LIST(ch, cdw, cdbias), 0);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw->data.f32, cdw->data.f32, INPUT_DIM * OUTPUT_DIM * KERNEL_SIZE * KERNEL_SIZE, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias->data.f32, cdbias->data.f32, OUTPUT_DIM, 5e-1, "output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, h->data.f32, ch->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 1e-4, "output from mps should match from CPU");
	ccv_nnc_tensor_free(gao);
	ccv_nnc_tensor_free(ggo);
	ccv_nnc_tensor_free(gho);
	ccv_nnc_tensor_free(gwo);
	ccv_nnc_tensor_free(gbiaso);
	ccv_nnc_tensor_free(gdwo);
	ccv_nnc_tensor_free(gdbiaso);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gdbias);
	ccv_nnc_tensor_free(gdw);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ch);
	ccv_nnc_tensor_free(cdw);
	ccv_nnc_tensor_free(cdbias);
}

TEST_CASE("compare group norm gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 1), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * GN_C_DIM * 2 * LN_DIM];
	float biasdata[1 * GN_C_DIM * 2 * LN_DIM];
	for (i = 0; i < 1 * GN_C_DIM * 2 * LN_DIM; i++)
	{
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);
	}
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dbbias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bias));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dbias_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor, dbbias_tensor), TENSOR_LIST(dscale_tensor, dbias_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 1), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale, cbias), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_tensor_symbol_t dcbias = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cbias);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cbias);
	memcpy(cbias_tensor->data.f32, biasdata, sizeof(float) * 1 * GN_C_DIM * 2 * LN_DIM);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);
	ccv_nnc_tensor_t* const dcbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcbias);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias_tensor->data.f32, dcbias_tensor->data.f32, 1 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dscale_tensor->data.f32, dcscale_tensor->data.f32, 1 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
	ccv_nnc_tensor_free(dbias_tensor);
}

TEST_CASE("compare group norm gradient with mps, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 1, 1), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 1, 1), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 1), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale, bias), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * GN_C_DIM * 1 * 1];
	float biasdata[1 * GN_C_DIM * 1 * 1];
	for (i = 0; i < 1 * GN_C_DIM * 1 * 1; i++)
	{
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);
	}
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dbbias_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bias));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), 0);
	ccv_nnc_tensor_t* const dbias_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor, dbbias_tensor), TENSOR_LIST(dscale_tensor, dbias_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 1, 1), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 1), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale, cbias), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_tensor_symbol_t dcbias = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cbias);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * GN_C_DIM * 1 * 1);
	ccv_nnc_tensor_t* const cbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cbias);
	memcpy(cbias_tensor->data.f32, biasdata, sizeof(float) * 1 * GN_C_DIM * 1 * 1);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);
	ccv_nnc_tensor_t* const dcbias_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcbias);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias_tensor->data.f32, dcbias_tensor->data.f32, 1 * GN_C_DIM * 1 * 1, 1e-5, "group norm output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dscale_tensor->data.f32, dcscale_tensor->data.f32, 1 * GN_C_DIM * 1 * 1, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
	ccv_nnc_tensor_free(dbias_tensor);
}

TEST_CASE("compare group norm gradient with mps (no dbias)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t scale = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 1, GN_C_DIM, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 1), TENSOR_SYMBOL_LIST(bx, scale, bias), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx, scale), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	float scaledata[1 * GN_C_DIM * 2 * LN_DIM];
	float biasdata[1 * GN_C_DIM * 2 * LN_DIM];
	for (i = 0; i < 1 * GN_C_DIM * 2 * LN_DIM; i++)
	{
		scaledata[i] = dsfmt_genrand_open_close(&dsfmt);
		biasdata[i] = dsfmt_genrand_open_close(&dsfmt);
	}
	ccv_nnc_tensor_t scale_tensor = ccv_nnc_tensor(scaledata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t bias_tensor = ccv_nnc_tensor(biasdata, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&scale_tensor, &bias_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, scale), ccv_nnc_tensor_from_symbol(tensor_arena, bias)), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_tensor_t* const dbscale_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_for_backward(symbolic_graph, scale));
	ccv_nnc_tensor_t* const dscale_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbscale_tensor), TENSOR_LIST(dscale_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t cscale = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), "scale");
	ccv_nnc_tensor_symbol_t cbias = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 1, GN_C_DIM, 2, LN_DIM), "bias");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 1), TENSOR_SYMBOL_LIST(cx, cscale, cbias), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx, cscale), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_tensor_symbol_t dcscale = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cscale);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const cscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cscale);
	memcpy(cscale_tensor->data.f32, scaledata, sizeof(float) * 1 * GN_C_DIM * 2 * LN_DIM);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);
	ccv_nnc_tensor_t* const dcscale_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcscale);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dscale_tensor->data.f32, dcscale_tensor->data.f32, 1 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dscale_tensor);
}

TEST_CASE("compare group norm gradient with mps without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 0), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 0), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("compare group norm gradient with mps, variant 1 without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 0), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 1, 1), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 1, 1), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 0), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("compare group norm gradient with mps (no dbias) without scale / bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) || ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t bx = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t by = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t saved_mean = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t saved_inv_std = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GROUP_NORM_FORWARD(1, 4, 1e-5, 0), TENSOR_SYMBOL_LIST(bx), TENSOR_SYMBOL_LIST(by, saved_mean, saved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(by), TENSOR_SYMBOL_LIST(bx), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dby = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, by);
	ccv_nnc_tensor_symbol_t dbx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, bx);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const bx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, bx);
	dsfmt_t dsfmt;
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(bx_tensor), 0);
	// ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_tensor_t* const dby_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dby);
	for (i = 0; i < 2 * GN_C_DIM * 2 * LN_DIM; i++)
		dy_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dby_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dbx_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dbx);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dbx_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_graph_free(graph);
	ccv_nnc_symbolic_graph_t* const cpu_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t cx = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "x");
	ccv_nnc_tensor_symbol_t cy = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_C_DIM, 2, LN_DIM), "y");
	ccv_nnc_tensor_symbol_t csaved_mean = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_mean");
	ccv_nnc_tensor_symbol_t csaved_inv_std = ccv_nnc_tensor_symbol_new(cpu_symbolic_graph, CPU_TENSOR_NHWC(32F, 2, GN_RC_DIM, 2, LN_DIM), "saved_inv_std");
	ccv_nnc_graph_exec_symbol_new(cpu_symbolic_graph, CMD_GROUP_NORM_FORWARD(1, GN_RC_DIM, 1e-5, 0), TENSOR_SYMBOL_LIST(cx), TENSOR_SYMBOL_LIST(cy, csaved_mean, csaved_inv_std), "group_norm");
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(cpu_symbolic_graph, TENSOR_SYMBOL_LIST(cy), TENSOR_SYMBOL_LIST(cx), SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(cpu_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_tensor_symbol_t dcy = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cy);
	ccv_nnc_tensor_symbol_t dcx = ccv_nnc_tensor_symbol_for_backward(cpu_symbolic_graph, cx);
	ccv_nnc_graph_t* cpu_graph = 0;
	ccv_nnc_tensor_arena_t* cpu_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* cpu_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(cpu_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(cpu_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(cpu_symbolic_graph), &cpu_graph, &cpu_tensor_arena, &cpu_graph_exec_arena);
	ccv_nnc_tensor_t* const cx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, cx);
	memcpy(cx_tensor->data.f32, x_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);
	ccv_nnc_tensor_t* const dcy_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcy);
	memcpy(dcy_tensor->data.f32, dy_tensor->data.f32, sizeof(float) * 2 * GN_C_DIM * 2 * LN_DIM);

	ccv_nnc_graph_run(cpu_graph, 0, TRAVERSE_FULL, 0, 0);

	ccv_nnc_tensor_t* const dcx_tensor = ccv_nnc_tensor_from_symbol(cpu_tensor_arena, dcx);

	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dx_tensor->data.f32, dcx_tensor->data.f32, 2 * GN_C_DIM * 2 * LN_DIM, 1e-5, "group norm output from mps should match from CPU");

	ccv_nnc_symbolic_graph_free(cpu_symbolic_graph);
	ccv_nnc_tensor_arena_free(cpu_tensor_arena);
	ccv_nnc_graph_exec_arena_free(cpu_graph_exec_arena);
	ccv_nnc_graph_free(cpu_graph);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dx_tensor);
}

TEST_CASE("broadcasting semantics for mul backward (a,b)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(gda, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, gdb), TENSOR_LIST(da, db), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, b), TENSOR_LIST(dat, dbt), 0);

	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gda);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("broadcasting semantics for mul backward (a, nil)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(gda, 0), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, 0), TENSOR_LIST(da, 0), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, b), TENSOR_LIST(dat, 0), 0);

	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gda);
}

TEST_CASE("broadcasting semantics for mul backward (nil,b)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(0, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, gdb), TENSOR_LIST(0, db), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, b), TENSOR_LIST(0, dbt), 0);

	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("broadcasting semantics for mul backward (no output db)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	memcpy(c->data.f32, ctp, sizeof(ctp));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, c), TENSOR_LIST(ga, gb, gc), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, ga, gb), TENSOR_LIST(gda, 0), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, 0), TENSOR_LIST(da, 0), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(c, a, b), TENSOR_LIST(dat, 0), 0);

	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gda);
}

TEST_CASE("broadcasting semantics for mul backward (no input grad)")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 1), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ga, gb), TENSOR_LIST(gda, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, gdb), TENSOR_LIST(da, db), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a, b), TENSOR_LIST(dat, dbt), 0);


	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gda);
	ccv_nnc_tensor_free(gdb);
}


TEST_CASE("broadcasting semantics for mul backward (no input grad) for b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
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
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ga, gb), TENSOR_LIST(gda, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, gdb), TENSOR_LIST(da, db), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a, b), TENSOR_LIST(dat, dbt), 0);

	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gda);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("broadcasting semantics for mul backward (no input grad) for a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	b->data.f32[0] = 1;
	b->data.f32[1] = 2;
	b->data.f32[2] = 3;
	b->data.f32[3] = 4;
	b->data.f32[4] = 5;
	b->data.f32[5] = 6;
	a->data.f32[0] = 7;
	a->data.f32[1] = 8;
	a->data.f32[2] = 9;
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gda = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ga, gb), TENSOR_LIST(gda, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gda, gdb), TENSOR_LIST(da, db), 0);
	ccv_nnc_cmd_exec(CMD_MUL_BACKWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a, b), TENSOR_LIST(dat, dbt), 0);

	REQUIRE_TENSOR_EQ(dat, da, "gradient of a should be equal");
	REQUIRE_TENSOR_EQ(dbt, db, "gradient of b should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gda);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("mps scalar mul forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_BACKWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));

	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const gx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 4; i++)
  		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x), TENSOR_LIST(gx), 0);

	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);

	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_LIST(gx), TENSOR_LIST(gy), 0);

	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gy), TENSOR_LIST(y), 0);
	for (i = 0; i < 4; i++) {
  		REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[i] * 1.1, y->data.f32[i], 1e-5, "scalarmul forward cy has to be 1.1 * x");
	}

	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(gx);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(y);
}

TEST_CASE("mps scalar mul backward")
{
	GUARD_ELSE_RETURN(
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS));

	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 4; i++)
  		y->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const gdx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y), TENSOR_LIST(gy), 0);
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_BACKWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_LIST(gy), TENSOR_LIST(gdx), 0);
	
	ccv_nnc_tensor_t* const dx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gdx), TENSOR_LIST(dx), 0);

	for (i = 0; i < 4; i++) {
  		REQUIRE_EQ_WITH_TOLERANCE(dx->data.f32[i], y->data.f32[i] * 1.1, 1e-5, "scalarmul backward dx has to be 1.1 * dy");
	}

	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(gdx);
	ccv_nnc_tensor_free(dx);
}

TEST_CASE("mps scalar mul backward, no input")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_BACKWARD, CCV_NNC_BACKEND_MPS));

	ccv_nnc_tensor_t* const gdx = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_SCALAR_MUL_BACKWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_LIST(0), TENSOR_LIST(gdx), 0);
	ccv_nnc_tensor_t* const dx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gdx), TENSOR_LIST(dx), 0);

	for (int i = 0; i < 4; i++)
  		REQUIRE_EQ_WITH_TOLERANCE(dx->data.f32[i], 1.1, 1e-5, "scalar mul backward without input should be 1.1 ");
	ccv_nnc_tensor_free(gdx);
	ccv_nnc_tensor_free(dx);
}

TEST_CASE("mps forward convolution transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(1, INPUT_DIM, 0, KERNEL_SIZE, KERNEL_SIZE, OUTPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, b->info, a->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, INPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < INPUT_DIM; i++)
		bias->data.f32[i] = (float)i / INPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, INPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);

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
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 1e-4, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution transpose in nchw format")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(1, INPUT_DIM, 0, KERNEL_SIZE, KERNEL_SIZE, OUTPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, b->info, a->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, INPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < INPUT_DIM; i++)
		bias->data.f32[i] = (float)i / INPUT_DIM;
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, OUTPUT_DIM, OUTPUT_SIZE, OUTPUT_SIZE), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, INPUT_DIM), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	assert(move.backend >= 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);

	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	assert(transform.backend >= 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	cmd.algorithm = -1;
	cmd = ccv_nnc_cmd_autotune(cmd, 1 * 1024 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gc), 0));
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, BATCH_SIZE, INPUT_DIM, INPUT_SIZE, INPUT_SIZE), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 1e-5, "output from mps should match from CPU");
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

TEST_CASE("mps forward convolution transpose in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_TRANSPOSE_FORWARD(1, INPUT_DIM, 0, KERNEL_SIZE, KERNEL_SIZE, OUTPUT_DIM);
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(cmd.info, b->info, a->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, INPUT_DIM), 0);
	// configure the inlets.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE * OUTPUT_DIM; i++)
		w->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (INPUT_DIM * KERNEL_SIZE * KERNEL_SIZE);
	for (i = 0; i < OUTPUT_SIZE * OUTPUT_SIZE * OUTPUT_DIM * ccv_max(1, BATCH_SIZE); i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < INPUT_DIM; i++)
		bias->data.f32[i] = (float)i / INPUT_DIM;
	ccv_nnc_tensor_t* a1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* w1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* bias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, INPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(a1, w1, bias1), 0);
	// Copy generated matrix values over to GPU.
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_DIM), 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, OUTPUT_DIM, KERNEL_SIZE, KERNEL_SIZE, INPUT_DIM), 0);
	ccv_nnc_tensor_t* gwo = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, OUTPUT_DIM, INPUT_DIM, KERNEL_SIZE, KERNEL_SIZE), 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, INPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a1, w1, bias1), TENSOR_LIST(ga, gw, gbias), 0);
	ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);

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
	cmd = ccv_nnc_cmd_autotune(cmd, 512 * 1024 * 1024, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context);
	assert(CCV_NNC_EXEC_SUCCESS == ccv_nnc_cmd_exec(cmd, hint, 0, TENSOR_LIST(ga, gwo, gbias), TENSOR_LIST(gc), stream_context));
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_t* c1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c1), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_DIM), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c1), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, c->data.f32, BATCH_SIZE * INPUT_DIM * INPUT_SIZE * INPUT_SIZE, 5e-3, "output from mps should match from CPU");
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

TEST_CASE("compare tanh with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TANH_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "tanh");
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
	ccv_nnc_cmd_exec(CMD_TANH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_TENSOR_EQ(ty, y_tensor, "tanh from mps should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("compare tanh with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TANH_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "tanh");
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
	ccv_nnc_cmd_exec(CMD_TANH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty->data.f32, y_tensor->data.f32, 20 * 10, 1e-3, "tanh from mps should match from CPU");
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

TEST_CASE("compare tanh gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_TANH_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TANH_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "tanh");
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
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_TANH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_TANH_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, 0, ty_tensor), TENSOR_LIST(tdx_tensor), 0);
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

TEST_CASE("compare tanh gradient with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_TANH_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_TANH_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "tanh");
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
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_TANH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_TANH_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, 0, ty_tensor), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

#include "case_main.h"
