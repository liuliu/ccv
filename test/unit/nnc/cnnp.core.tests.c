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

static ccv_cnnp_model_t* simple_cifar_10(void)
{
	return ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_flatten(0),
		ccv_cnnp_dense(256, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
		}, 0),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		}, 0)
	), 0);
}

TEST_CASE("compile simple cifar-10 model")
{
	ccv_cnnp_model_t* const sequential = simple_cifar_10();
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(32F, 1, 31, 31, 3);
	ccv_cnnp_model_compile(sequential, &input, 1, CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 31, 31, 3), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 31 * 31 * 3; i++)
		input_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	int t = 0;
	float max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	const int target = (t + 1) % 10;
	REQUIRE_NOT_EQ(target, t, "should not fit");
	// Doing training.
	ccv_nnc_tensor_t* const fit_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	fit_tensor->data.f32[0] = target;
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(fit_tensor), TENSOR_LIST(output_tensor), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	remove("/tmp/compile_simple_cifar_10_model.checkpoint");
	ccv_cnnp_model_checkpoint(sequential, "/tmp/compile_simple_cifar_10_model.checkpoint", 0);
	CNNP_MODEL_GEN(sequential, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(sequential);
	ccv_cnnp_model_t* const sequential2 = simple_cifar_10();
	ccv_cnnp_model_compile(sequential2, &input, 1, CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	// Load from the checkpoint file.
	ccv_cnnp_model_checkpoint(sequential2, "/tmp/compile_simple_cifar_10_model.checkpoint", 0);
	remove("/tmp/compile_simple_cifar_10_model.checkpoint");
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_cnnp_model_evaluate(sequential2, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	ccv_cnnp_model_free(sequential2);
	ccv_nnc_tensor_free(input_tensor);
	ccv_nnc_tensor_free(fit_tensor);
	ccv_nnc_tensor_free(output_tensor);
}

TEST_CASE("inception layer for model")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_io_t tower_1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}, 0), MODEL_IO_LIST(x));
	tower_1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	}, 0), MODEL_IO_LIST(tower_1));

	ccv_cnnp_model_io_t tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}, 0), MODEL_IO_LIST(x));
	tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (2, 2)),
	}, 0), MODEL_IO_LIST(tower_2));

	ccv_cnnp_model_io_t tower_3 = ccv_cnnp_model_apply(ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.hint = HINT((1, 1), (1, 1)),
	}, 0), MODEL_IO_LIST(x));
	tower_3 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}, 0), MODEL_IO_LIST(tower_3));

	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(tower_1, tower_2, tower_3));

	ccv_cnnp_model_t* inception = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(output), 0);
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	ccv_cnnp_model_compile(inception, &input, 1, CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(inception, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(inception);
}

TEST_CASE("functional model's IO can represent multiple outputs")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}, 0), MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	}, 0), MODEL_IO_LIST(input1));
	ccv_cnnp_model_t* model0 = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(output0, output1), 0);
	input0 = ccv_cnnp_input();
	input1 = ccv_cnnp_input();
	output0 = ccv_cnnp_model_apply(model0, MODEL_IO_LIST(input0, input1));
	ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	output1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (2, 2)),
	}, 0), MODEL_IO_LIST(input2));
	ccv_cnnp_model_t* interim = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2), MODEL_IO_LIST(output0, output1), 0);
	input0 = ccv_cnnp_input();
	input1 = ccv_cnnp_input();
	input2 = ccv_cnnp_input();
	output0 = ccv_cnnp_model_apply(interim, MODEL_IO_LIST(input0, input1, input2));
	output0 = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(output0));
	ccv_cnnp_model_t* final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2), MODEL_IO_LIST(output0), 0);
	const ccv_nnc_tensor_param_t a0 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	const ccv_nnc_tensor_param_t a1 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	const ccv_nnc_tensor_param_t a2 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, a2), CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(final);
}

TEST_CASE("make sure reuse model enables share weights")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, (ccv_cnnp_param_t){}, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(output0, output1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(final_output), 0);
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1), CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(final);
}

TEST_CASE("train model with share weights and L2 loss")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, (ccv_cnnp_param_t){}, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t fit0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t fit1 = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(output0, fit0));
	ccv_cnnp_model_io_t sqr0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(diff0, diff0));
	ccv_cnnp_model_io_t diff1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(output1, fit1));
	ccv_cnnp_model_io_t sqr1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(diff1, diff1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(sqr0, sqr1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, fit0, fit1), MODEL_IO_LIST(final_output), 0);
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t b0 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t b1 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, b0, b1), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* a1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* b0_tensor = ccv_nnc_tensor_new(0, b0, 0);
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_new(0, b1, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	a0_tensor->data.f32[0] = 1;
	a1_tensor->data.f32[0] = 3;
	b0_tensor->data.f32[0] = 2;
	b1_tensor->data.f32[0] = 3;
	int i;
	for (i = 0; i < 10; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0);
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0);
	for (i = 0; i < 1000; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 0.5, 2 * 1e-2, "We should linear regressed this.");
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(a1_tensor);
	ccv_nnc_tensor_free(b0_tensor);
	ccv_nnc_tensor_free(b1_tensor);
	ccv_nnc_tensor_free(o0_tensor);
	ccv_cnnp_model_free(final);
}

static ccv_cnnp_model_t* simple_cifar_10_no_softmax(void)
{
	return ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}, 0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}, 0),
		ccv_cnnp_flatten(0),
		ccv_cnnp_dense(256, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
		}, 0),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_NONE,
		}, 0)
	), 0);
}

TEST_CASE("evaluate cifar-10 model in multi-stage mode")
{
	ccv_cnnp_model_t* const sequential = simple_cifar_10_no_softmax();
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(32F, 1, 31, 31, 3);
	ccv_cnnp_model_compile(sequential, &input, 1, CMD_SGD_FORWARD(0, 0.001, 1, 0.99, 0.9, 0.9), CMD_NOOP());
	ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 31, 31, 3), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 31 * 31 * 3; i++)
		input_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	int t = 0;
	float max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	const int target = (t + 1) % 10;
	REQUIRE_NOT_EQ(target, t, "should not fit");
	// Doing training.
	ccv_nnc_tensor_t* const fit_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	fit_tensor->data.f32[0] = target;
	ccv_nnc_tensor_t* const softmax_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	ccv_nnc_tensor_t* const loss_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const ingrad_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	for (i = 0; i < 100; i++)
	{
		ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1
		}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensor, fit_tensor), TENSOR_LIST(loss_tensor, softmax_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, 0, output_tensor, fit_tensor, loss_tensor, softmax_tensor), TENSOR_LIST(ingrad_tensor), 0);
		ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0);
		ccv_cnnp_model_apply_gradients(sequential, 0);
	}
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	ccv_nnc_tensor_free(ingrad_tensor);
	ccv_nnc_tensor_free(fit_tensor);
	ccv_nnc_tensor_free(softmax_tensor);
	ccv_nnc_tensor_free(loss_tensor);
	ccv_nnc_tensor_free(input_tensor);
	ccv_nnc_tensor_free(output_tensor);
	ccv_cnnp_model_free(sequential);
}

TEST_CASE("evaluate cifar-10 model in multi-stage mode with gradient accumulated")
{
	ccv_cnnp_model_t* const sequential = simple_cifar_10_no_softmax();
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(32F, 1, 31, 31, 3);
	ccv_cnnp_model_compile(sequential, &input, 1, CMD_SGD_FORWARD(0, 0.00033, 1, 0.99, 0.9, 0.9), CMD_NOOP());
	ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 31, 31, 3), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 31 * 31 * 3; i++)
		input_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	int t = 0;
	float max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	const int target = (t + 1) % 10;
	REQUIRE_NOT_EQ(target, t, "should not fit");
	// Doing training.
	ccv_nnc_tensor_t* const fit_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	fit_tensor->data.f32[0] = target;
	ccv_nnc_tensor_t* const softmax_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	ccv_nnc_tensor_t* const loss_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const ingrad_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 10), 0);
	for (i = 0; i < 100; i++)
	{
		ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1
		}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensor, fit_tensor), TENSOR_LIST(loss_tensor, softmax_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, 0, output_tensor, fit_tensor, loss_tensor, softmax_tensor), TENSOR_LIST(ingrad_tensor), 0);
		ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0);
		// Backward again to accumulate gradient.
		if (i % 2 == 0)
		{
			ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0);
			// Backward again to accumulate gradient.
			if (i % 3 == 0)
				ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0);
		}
		ccv_cnnp_model_apply_gradients(sequential, 0);
	}
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	ccv_nnc_tensor_free(ingrad_tensor);
	ccv_nnc_tensor_free(fit_tensor);
	ccv_nnc_tensor_free(softmax_tensor);
	ccv_nnc_tensor_free(loss_tensor);
	ccv_nnc_tensor_free(input_tensor);
	ccv_nnc_tensor_free(output_tensor);
	ccv_cnnp_model_free(sequential);
}

TEST_CASE("train model with share weights and L2 loss and check out gradients")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, (ccv_cnnp_param_t){}, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t fit0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t fit1 = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(output0, fit0));
	ccv_cnnp_model_io_t sqr0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(diff0, diff0));
	ccv_cnnp_model_io_t diff1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(output1, fit1));
	ccv_cnnp_model_io_t sqr1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(diff1, diff1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(sqr0, sqr1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, fit0, fit1), MODEL_IO_LIST(final_output), 0);
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t b0 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_nnc_tensor_param_t b1 = CPU_TENSOR_NCHW(32F, 1, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, b0, b1), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* a1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* b0_tensor = ccv_nnc_tensor_new(0, b0, 0);
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_new(0, b1, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	// It should fit to 1*0.5+1.5=2, 3*0.5+1.5=3
	a0_tensor->data.f32[0] = 1;
	a1_tensor->data.f32[0] = 3;
	b0_tensor->data.f32[0] = 2;
	b1_tensor->data.f32[0] = 3;
	int i;
	for (i = 0; i < 10; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0);
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0);
	for (i = 0; i < 1000; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0);
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 0.5, 2 * 1e-2, "We should linear regressed this.");
	// Figure out the actual weight and bias term in the model.
	a0_tensor->data.f32[0] = 0;
	a1_tensor->data.f32[0] = 0;
	b0_tensor->data.f32[0] = 0;
	b1_tensor->data.f32[0] = 0;
	// The output will be 2*bias^2
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0);
	const float bias = sqrtf(o0_tensor->data.f32[0] * 0.5);
	a0_tensor->data.f32[0] = 1;
	a1_tensor->data.f32[0] = 1;
	b0_tensor->data.f32[0] = 0;
	b1_tensor->data.f32[0] = 0;
	// The output will be 2*(w+bias)^2
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0);
	const float w = sqrt(o0_tensor->data.f32[0] * 0.5) - bias;
	// Compute the out gradient to verify.
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
		.enable_outgrad = 1,
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0);
	// Note that I have to use new tensors and have to keep these tensors around since they were binded to the model when evaluate.
	ccv_nnc_tensor_t* da0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* da1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* db0_tensor = ccv_nnc_tensor_new(0, b0, 0);
	ccv_nnc_tensor_t* db1_tensor = ccv_nnc_tensor_new(0, b1, 0);
	ccv_nnc_tensor_t* do0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	do0_tensor->data.f32[0] = 1;
	ccv_cnnp_model_backward(final, TENSOR_LIST(do0_tensor), TENSOR_LIST(da0_tensor, da1_tensor, db0_tensor, db1_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(da0_tensor->data.f32[0], 2 * w * (w * 2 + bias - 2), 1e-5, "da0=2*w*(w*a0+bias-b0), thus, 0.5");
	REQUIRE_EQ_WITH_TOLERANCE(da1_tensor->data.f32[0], 2 * w * (w * 2 + bias - 3), 1e-5, "da1=2*w*(w*a1+bias-b1), thus, -0.5");
	REQUIRE_EQ_WITH_TOLERANCE(db0_tensor->data.f32[0], -2 * (w * 2 + bias - 2), 1e-5, "db0=-2*(w*a0+bias-b0), thus, -1");
	REQUIRE_EQ_WITH_TOLERANCE(db1_tensor->data.f32[0], -2 * (w * 2 + bias - 3), 1e-5, "db1=-2*(w*a1+bias-b1), thus, 1");
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(a1_tensor);
	ccv_nnc_tensor_free(b0_tensor);
	ccv_nnc_tensor_free(b1_tensor);
	ccv_nnc_tensor_free(o0_tensor);
	ccv_nnc_tensor_free(da0_tensor);
	ccv_nnc_tensor_free(da1_tensor);
	ccv_nnc_tensor_free(db0_tensor);
	ccv_nnc_tensor_free(db1_tensor);
	ccv_nnc_tensor_free(do0_tensor);
	ccv_cnnp_model_free(final);
}

TEST_CASE("apply functional model as forward pass")
{
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_set_by(CMD_SET_FORWARD(2.12), ccv_nnc_no_hint, 0, CPU_TENSOR_NCHW(32F, 1)))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "mul");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(mul, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(mul, MODEL_IO_LIST(output));
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = -1;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "add");
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), "final");
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 1.12;
	o0_tensor->data.f32[0] = 0;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor), TENSOR_LIST(o0_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 1.12 * 2.12 * 2.12 - 1, 1e-5, "all the model building is to compute 1.12 * 2.12 * 2.12 - 1");
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(o0_tensor);
	ccv_cnnp_model_free(final);
}

TEST_CASE("apply sequential model as forward pass")
{
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_set_by(CMD_SET_FORWARD(2.12), ccv_nnc_no_hint, 0, CPU_TENSOR_NCHW(32F, 1)))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "mul");
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = -1;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "add");
	ccv_cnnp_model_t* const final = ccv_cnnp_sequential_new(MODEL_LIST(mul, mul, add), "seq");
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 1.12;
	o0_tensor->data.f32[0] = 0;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor), TENSOR_LIST(o0_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 1.12 * 2.12 * 2.12 - 1, 1e-5, "all the model building is to compute 1.12 * 2.12 * 2.12 - 1");
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(o0_tensor);
	ccv_cnnp_model_free(final);
}

TEST_CASE("learn simple math of 2 * x + 1 + 1 = 10, x = 4")
{
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(
			KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_set_by(CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, CPU_TENSOR_NCHW(32F, 1))),
		),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "mul");
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = 1;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), "add");
	ccv_cnnp_model_t* const left = ccv_cnnp_sequential_new(MODEL_LIST(mul, add, add), "seq");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t left_out = ccv_cnnp_model_apply(left, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(left_out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 0),
		MODEL_IO_LIST(diff, diff));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 0);
	ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_tensor_t* ingrad = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_tensor_t* outgrad0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_tensor_t* outgrad1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ingrad->data.f32[0] = 1;
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 10;
	int i;
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0);
	for (i = 0; i < 100; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.enable_outgrad = 1,
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(ingrad), TENSOR_LIST(outgrad0, outgrad1), 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0);
	for (i = 0; i < 1000; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.enable_outgrad = 1,
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(ingrad), TENSOR_LIST(outgrad0, outgrad1), 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	o_tensor->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1,
	}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 0, 1e-5, "(2 * x + 1 + 1 - 10) ^ 2 should equal to 0");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(ingrad);
	ccv_nnc_tensor_free(outgrad0);
	ccv_nnc_tensor_free(outgrad1);
	ccv_cnnp_model_free(final);
}

#include "case_main.h"
