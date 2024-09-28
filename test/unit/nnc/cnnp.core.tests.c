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
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_flatten(0),
		ccv_cnnp_dense(256, 0, 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_dense(10, 0, 0, 1, 0),
		ccv_cnnp_softmax(0)
	), 1, 0);
}

TEST_CASE("compile simple cifar-10 model")
{
	ccv_cnnp_model_t* const sequential0 = simple_cifar_10();
	ccv_cnnp_model_t* const sequential = ccv_cnnp_model_copy(sequential0, 1);
	ccv_cnnp_model_free(sequential0);
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
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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
		ccv_cnnp_model_fit(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(fit_tensor), TENSOR_LIST(output_tensor), 0, 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	remove("/tmp/compile_simple_cifar_10_model.checkpoint");
	ccv_cnnp_model_write_to_file(sequential, "/tmp/compile_simple_cifar_10_model.checkpoint", 0);
	CNNP_MODEL_GEN(sequential, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(sequential);
	ccv_cnnp_model_t* const sequential2 = simple_cifar_10();
	ccv_cnnp_model_compile(sequential2, &input, 1, CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	// Load from the checkpoint file.
	ccv_cnnp_model_read_from_file("/tmp/compile_simple_cifar_10_model.checkpoint", 0, sequential2);
	remove("/tmp/compile_simple_cifar_10_model.checkpoint");
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	ccv_cnnp_model_evaluate(sequential2, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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

static int _ccv_cnnp_model_notified = 0;

static void _ccv_cnnp_model_hook(const ccv_cnnp_model_t* const model, const int tag, void* const payload, void* const context)
{
	if (payload)
		++_ccv_cnnp_model_notified;
}

TEST_CASE("inception layer for model")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	_ccv_cnnp_model_notified = 0;
	ccv_cnnp_model_t* const conv_1 = ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0);
	ccv_cnnp_model_notify_hook(conv_1, _ccv_cnnp_model_hook, 0);
	ccv_cnnp_model_io_t tower_1 = ccv_cnnp_model_apply(conv_1, MODEL_IO_LIST(x));
	ccv_cnnp_model_t* const relu_1 = ccv_cnnp_relu(0);
	ccv_cnnp_model_notify_hook(relu_1, _ccv_cnnp_model_hook, 0);
	tower_1 = ccv_cnnp_model_apply(relu_1, MODEL_IO_LIST(tower_1));
	tower_1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), DIM_ALLOC(), 0, HINT((1, 1), (1, 1)), 0, 1, 0), MODEL_IO_LIST(tower_1));
	tower_1 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(tower_1));

	ccv_cnnp_model_io_t tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0), MODEL_IO_LIST(x));
	tower_2 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(tower_2));
	tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0), MODEL_IO_LIST(tower_2));
	tower_2 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(tower_2));

	ccv_cnnp_model_io_t tower_3 = ccv_cnnp_model_apply(ccv_cnnp_max_pool(DIM_ALLOC(3, 3), HINT((1, 1), (1, 1)), 0), MODEL_IO_LIST(x));
	tower_3 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0), MODEL_IO_LIST(tower_3));
	tower_3 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(tower_3));
	ccv_cnnp_model_t* const add_1 = ccv_cnnp_sum(0);
	ccv_cnnp_model_notify_hook(add_1, _ccv_cnnp_model_hook, 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(add_1, MODEL_IO_LIST(tower_1, tower_2, tower_3));
	REQUIRE_EQ(0, _ccv_cnnp_model_notified, "haven't notified");
	ccv_cnnp_model_t* const inception0 = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(output), 1, 0);
	ccv_cnnp_model_notify(inception0, 0, inception0);
	ccv_cnnp_model_t* const inception = ccv_cnnp_model_copy(inception0, 1);
	REQUIRE_EQ(3, _ccv_cnnp_model_notified, "3 models changed owner");
	ccv_cnnp_model_free(inception0);
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	ccv_cnnp_model_compile(inception, &input, 1, CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(inception, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(inception);
}

static ccv_cnnp_model_t* _ccv_multiple_outputs_functional_model(const ccv_nnc_tensor_param_t* const inputs, const int input_size, void* const context)
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0), MODEL_IO_LIST(input0));
	output0 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(output0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), DIM_ALLOC(), 0, HINT((1, 1), (1, 1)), 0, 1, 0), MODEL_IO_LIST(input1));
	output1 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(output1));
	ccv_cnnp_model_t* model0 = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(output0, output1), 1, 0);
	input0 = ccv_cnnp_input();
	input1 = ccv_cnnp_input();
	output0 = ccv_cnnp_model_apply(model0, MODEL_IO_LIST(input0, input1));
	ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	output1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0), MODEL_IO_LIST(input2));
	output1 = ccv_cnnp_model_apply(ccv_cnnp_relu(0), MODEL_IO_LIST(output1));
	ccv_cnnp_model_t* interim = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2), MODEL_IO_LIST(output0, output1), 1, 0);
	input0 = ccv_cnnp_input();
	input1 = ccv_cnnp_input();
	input2 = ccv_cnnp_input();
	output0 = ccv_cnnp_model_apply(interim, MODEL_IO_LIST(input0, input1, input2));
	output0 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output0));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2), MODEL_IO_LIST(output0), 1, 0);
}

TEST_CASE("functional model's IO can represent multiple outputs")
{
	ccv_cnnp_model_t* const final = ccv_cnnp_dynamic_new(_ccv_multiple_outputs_functional_model, 0, 0);
	const ccv_nnc_tensor_param_t a0 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	const ccv_nnc_tensor_param_t a1 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	const ccv_nnc_tensor_param_t a2 = GPU_TENSOR_NCHW(000, 32F, 1, 3, 256, 256);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, a2), CMD_SGD_FORWARD(1, 0.001, 1, 0.99, 0.9, 0), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(final);
}

TEST_CASE("functional model's IO outputs can be non-terminal")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input3 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(input0, input1));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(ccv_cnnp_mul(1, 0), MODEL_IO_LIST(output0, input2));
	output1 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output1, input3));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2, input3), MODEL_IO_LIST(output0, output1), 1, 0);
	const ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a2 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a3 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, a2, a3), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* const a1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* const a2_tensor = ccv_nnc_tensor_new(0, a2, 0);
	ccv_nnc_tensor_t* const a3_tensor = ccv_nnc_tensor_new(0, a3, 0);
	ccv_nnc_tensor_t* const b0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* const b1_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 0.5;
	a1_tensor->data.f32[0] = 0.75;
	a2_tensor->data.f32[0] = 1.75;
	a3_tensor->data.f32[0] = 2.5;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, a2_tensor, a3_tensor), TENSOR_LIST(b0_tensor, b1_tensor), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(b0_tensor->data.f32[0], 0.5 + 0.75, 1e-5, "should match the intermediate result");
	REQUIRE_EQ_WITH_TOLERANCE(b1_tensor->data.f32[0], (0.5 + 0.75) * 1.75 + 2.5, 1e-5, "should match the final result");
	ccv_cnnp_model_free(final);
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(a1_tensor);
	ccv_nnc_tensor_free(a2_tensor);
	ccv_nnc_tensor_free(a3_tensor);
	ccv_nnc_tensor_free(b0_tensor);
	ccv_nnc_tensor_free(b1_tensor);
}

TEST_CASE("functional model's IO can introduce non-functional dependencies")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input3 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(input0, input1));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(ccv_cnnp_mul(1, 0), MODEL_IO_LIST(input2, input3));
	// non-functional dependency.
	ccv_cnnp_model_add_dependencies(output1, MODEL_IO_LIST(output0));
	output1 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output0, output1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2, input3), MODEL_IO_LIST(output0, output1), 1, 0);
	const ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a2 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a3 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, a2, a3), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* const a1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* const a2_tensor = ccv_nnc_tensor_new(0, a2, 0);
	ccv_nnc_tensor_t* const a3_tensor = ccv_nnc_tensor_new(0, a3, 0);
	ccv_nnc_tensor_t* const b0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* const b1_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 0.5;
	a1_tensor->data.f32[0] = 0.75;
	a2_tensor->data.f32[0] = 1.75;
	a3_tensor->data.f32[0] = 2.5;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, a2_tensor, a3_tensor), TENSOR_LIST(b0_tensor, b1_tensor), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(b0_tensor->data.f32[0], 0.5 + 0.75, 1e-5, "should match the intermediate result");
	REQUIRE_EQ_WITH_TOLERANCE(b1_tensor->data.f32[0], (0.5 + 0.75) + (1.75 * 2.5), 1e-5, "should match the final result");
	ccv_cnnp_model_free(final);
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(a1_tensor);
	ccv_nnc_tensor_free(a2_tensor);
	ccv_nnc_tensor_free(a3_tensor);
	ccv_nnc_tensor_free(b0_tensor);
	ccv_nnc_tensor_free(b1_tensor);
}

TEST_CASE("make sure reuse model enables share weights")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, 0, 0, 1, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output0, output1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(final_output), 1, 0);
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
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, 0, 0, 1, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t fit0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t fit1 = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(output0, fit0));
	ccv_cnnp_model_io_t sqr0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff0, diff0));
	ccv_cnnp_model_io_t diff1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(output1, fit1));
	ccv_cnnp_model_io_t sqr1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff1, diff1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(sqr0, sqr1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, fit0, fit1), MODEL_IO_LIST(final_output), 1, 0);
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
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0, 0);
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0, 0);
	for (i = 0; i < 1000; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0, 0);
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
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), DIM_ALLOC(), 0, HINT((1, 1), (2, 2)), 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), HINT((2, 2), (0, 0)), 0),
		ccv_cnnp_flatten(0),
		ccv_cnnp_dense(256, 0, 0, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_dense(10, 0, 0, 1, 0)
	), 1, 0);
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
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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
		}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensor, fit_tensor), TENSOR_LIST(loss_tensor, softmax_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, 0, output_tensor, fit_tensor, loss_tensor, softmax_tensor), TENSOR_LIST(ingrad_tensor), 0);
		ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0, 0);
		ccv_cnnp_model_apply_gradients(sequential, 0);
	}
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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
		}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output_tensor, fit_tensor), TENSOR_LIST(loss_tensor, softmax_tensor), 0);
		ccv_nnc_cmd_exec(CMD_SOFTMAX_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, 0, output_tensor, fit_tensor, loss_tensor, softmax_tensor), TENSOR_LIST(ingrad_tensor), 0);
		ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0, 0);
		// Backward again to accumulate gradient.
		if (i % 2 == 0)
		{
			ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0, 0);
			// Backward again to accumulate gradient.
			if (i % 3 == 0)
				ccv_cnnp_model_backward(sequential, TENSOR_LIST(ingrad_tensor), 0, 0, 0, 0);
		}
		ccv_cnnp_model_apply_gradients(sequential, 0);
	}
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0, 0);
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
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, 0, 0, 1, 0);
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t fit0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t fit1 = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(output0, fit0));
	ccv_cnnp_model_io_t sqr0 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff0, diff0));
	ccv_cnnp_model_io_t diff1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(output1, fit1));
	ccv_cnnp_model_io_t sqr1 = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff1, diff1));
	ccv_cnnp_model_io_t final_output = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(sqr0, sqr1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, fit0, fit1), MODEL_IO_LIST(final_output), 1, 0);
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
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0, 0);
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0, 0);
	for (i = 0; i < 1000; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), 0, 0, TENSOR_LIST(o0_tensor), 0, 0);
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 0.5, 2 * 1e-2, "We should linear regressed this.");
	// Figure out the actual weight and bias term in the model.
	a0_tensor->data.f32[0] = 0;
	a1_tensor->data.f32[0] = 0;
	b0_tensor->data.f32[0] = 0;
	b1_tensor->data.f32[0] = 0;
	// The output will be 2*bias^2
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0, 0);
	const float bias = sqrtf(o0_tensor->data.f32[0] * 0.5);
	a0_tensor->data.f32[0] = 1;
	a1_tensor->data.f32[0] = 1;
	b0_tensor->data.f32[0] = 0;
	b1_tensor->data.f32[0] = 0;
	// The output will be 2*(w+bias)^2
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0, 0);
	const float w = sqrt(o0_tensor->data.f32[0] * 0.5) - bias;
	// Compute the out gradient to verify.
	a0_tensor->data.f32[0] = 2;
	a1_tensor->data.f32[0] = 2; // The final result should be 4.
	b0_tensor->data.f32[0] = 2; // diff is 0.5
	b1_tensor->data.f32[0] = 3; // diff is 0.5, and 0.5^2 + 0.5^2 = 0.5.
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(a0_tensor, a1_tensor, b0_tensor, b1_tensor), TENSOR_LIST(o0_tensor), 0, 0);
	// Note that I have to use new tensors and have to keep these tensors around since they were binded to the model when evaluate.
	ccv_nnc_tensor_t* da0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* da1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* db0_tensor = ccv_nnc_tensor_new(0, b0, 0);
	ccv_nnc_tensor_t* db1_tensor = ccv_nnc_tensor_new(0, b1, 0);
	ccv_nnc_tensor_t* do0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	do0_tensor->data.f32[0] = 1;
	ccv_cnnp_model_backward(final, TENSOR_LIST(do0_tensor), TENSOR_LIST(da0_tensor, da1_tensor, db0_tensor, db1_tensor), 0, 0);
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
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "mul");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(mul, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(mul, MODEL_IO_LIST(output));
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = -1;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "add");
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 1, "final");
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 1.12;
	o0_tensor->data.f32[0] = 0;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor), TENSOR_LIST(o0_tensor), 0, 0);
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
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "mul");
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = -1;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "add");
	ccv_cnnp_model_t* const final = ccv_cnnp_sequential_new(MODEL_LIST(mul, mul, add), 1, "seq");
	ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* o0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 1.12;
	o0_tensor->data.f32[0] = 0;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor), TENSOR_LIST(o0_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o0_tensor->data.f32[0], 1.12 * 2.12 * 2.12 - 1, 1e-5, "all the model building is to compute 1.12 * 2.12 * 2.12 - 1");
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(o0_tensor);
	ccv_cnnp_model_free(final);
}

ccv_cnnp_model_t* _math_2_x_1_1_10(const ccv_nnc_tensor_t* const b)
{
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(
			KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_set_by(CMD_RANDOM_UNIFORM_FORWARD(-1, 1), ccv_nnc_no_hint, 0, CPU_TENSOR_NCHW(32F, 1))),
		),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "mul");
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR, ccv_cnnp_cmd_exec_io_copy(b))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "add");
	ccv_cnnp_model_t* const left = ccv_cnnp_sequential_new(MODEL_LIST(mul, add, add), 1, "seq");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t left_out = ccv_cnnp_model_apply(left, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(left_out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff, diff));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 1, 0);
}

TEST_CASE("learn simple math of 2 * x + 1 + 1 = 10, x = 4")
{
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = 1;
	ccv_cnnp_model_t* const final = _math_2_x_1_1_10(b);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* ingrad = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* outgrad0 = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* outgrad1 = ccv_nnc_tensor_new(0, f, 0);
	ingrad->data.f32[0] = 1;
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 10;
	int i;
	float old_o = 10;
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_NOT_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], old_o, 1e-5, "after 10 iterations, output should be different");
	old_o = o_tensor->data.f32[0];
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0, 0, 0), 0, 0, 0); // No decay.
	ingrad->data.f32[0] = 0; // ingrad is 0, no update at all.
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(ingrad), TENSOR_LIST(outgrad0, outgrad1), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], old_o, 1e-5, "after 10 iterations, output should be the same because the ingrad");
	old_o = o_tensor->data.f32[0];
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, 0, 0);
	for (i = 0; i < 100; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(0), TENSOR_LIST(outgrad0, outgrad1), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_NOT_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], old_o, 1e-5, "after 100 iterations, output should be different");
	old_o = o_tensor->data.f32[0];
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0, 0, 0), 0, 0, 0); // No decay.
	// Note we still use the old ingrad which is 0.
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(ingrad), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], old_o, 1e-5, "after 10 iterations, output should be the same because the ingrad");
	ingrad->data.f32[0] = 1; // ingrad reset to 1.
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, 0, 0);
	for (i = 0; i < 1000; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(ingrad), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_NOT_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], old_o, 1e-5, "after 1000 iterations, output should be different");
	o_tensor->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1,
	}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
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

static int _ccv_cnnp_model_clip_grad_norm_reduce_norm2(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_t* const old_norm2 = outputs[1];
	ccv_nnc_tensor_t* const norm2 = outputs[2];
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(), hint, flags, TENSOR_LIST(inputs[0]), TENSOR_LIST(norm2), stream_context);
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), hint, flags, TENSOR_LIST(old_norm2, norm2), TENSOR_LIST(old_norm2), stream_context);
	return CCV_NNC_EXEC_SUCCESS;
}

static ccv_nnc_cmd_vtab_t clip_grad_norm_reduce_norm2_vtab = {
	.exec = _ccv_cnnp_model_clip_grad_norm_reduce_norm2
};

TEST_CASE("learn simple math of 2 * x + 1 + 1 = 10, x = 4 and clip grad to max_norm = 0.5")
{
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = 1;
	ccv_cnnp_model_t* const final = _math_2_x_1_1_10(b);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* ingrad = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* outgrad0 = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* outgrad1 = ccv_nnc_tensor_new(0, f, 0);
	ingrad->data.f32[0] = 1;
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
	ccv_cnnp_model_parameters_clip_grad_norm(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), 2, 0.5, 0);
	ccv_nnc_tensor_t* old_norm2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* norm2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(old_norm2), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(norm2), 0);
	ccv_cnnp_model_apply_gradients(final, 0);
	ccv_nnc_cmd_t reduce_cmd = {
		.cmd = CCV_NNC_CUSTOM_FORWARD,
		.isa = &clip_grad_norm_reduce_norm2_vtab,
	};
	ccv_cnnp_model_parameter_gradients_map(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), reduce_cmd, ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(old_norm2, norm2), 0);
	REQUIRE(norm2->data.f32[0] < 0.5 + 1e-5, "norm2 should be smaller than max_norm");
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
	ccv_cnnp_model_parameters_clip_grad_norm(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), 2, 0.5, 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(old_norm2), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(norm2), 0);
	ccv_cnnp_model_parameter_gradients_map(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), reduce_cmd, ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(old_norm2, norm2), 0);
	REQUIRE(norm2->data.f32[0] < 0.5 + 1e-5, "norm2 should be smaller than max_norm");
	ccv_cnnp_model_apply_gradients(final, 0);
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(ingrad);
	ccv_nnc_tensor_free(outgrad0);
	ccv_nnc_tensor_free(outgrad1);
	ccv_cnnp_model_free(final);
	ccv_nnc_tensor_free(old_norm2);
	ccv_nnc_tensor_free(norm2);
}

TEST_CASE("train a simple math 2 * x + 1 + 1 = 10, x = 4 and copy parameter to a new model entirely")
{
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = 1;
	ccv_cnnp_model_t* const final = _math_2_x_1_1_10(b);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* ingrad = ccv_nnc_tensor_new(0, o, 0);
	ingrad->data.f32[0] = 1;
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 10;
	int i;
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	const float o_final = o_tensor->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_1_1_10(b);
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], o_final, 1e-5, "should match the previous output");
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 64, 1e-5, "should match the output when x is 0");
	ccv_cnnp_model_t* const final3 = ccv_cnnp_model_copy(final, 1);
	ccv_cnnp_model_set_parameters(final3, ccv_cnnp_model_parameters(final3, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	ccv_cnnp_model_evaluate(final3, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], o_final, 1e-5, "should match the previous output");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(ingrad);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
	ccv_cnnp_model_free(final3);
}

TEST_CASE("train a simple math 2 * x + 1 + 1 = 10, x = 4 and merge parameters with a model")
{
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	b->data.f32[0] = 1;
	ccv_cnnp_model_t* const final = _math_2_x_1_1_10(b);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	ccv_nnc_tensor_t* ingrad = ccv_nnc_tensor_new(0, o, 0);
	ingrad->data.f32[0] = 1;
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 10;
	int i;
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	const float o_final = o_tensor->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_1_1_10(b);
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], o_final, 1e-5, "should match the previous output");
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 36, 1e-5, "should match the output when x is 1");
	ccv_cnnp_model_parameters_zip_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_ADD_FORWARD(0.6, 0.4), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0, final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_new(0, a, 0);
	const ccv_nnc_tensor_param_t params = ccv_cnnp_model_parameter_tensor_params(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	REQUIRE_EQ(1, params.dim[0], "should match parameter shape");
	REQUIRE_EQ(0, params.dim[1], "should match parameter shape");
	ccv_cnnp_model_parameter_copy(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), x_tensor);
	const float x_final = x_tensor->data.f32[0] * 0.4 + 1 * 0.6;
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], (x_final * 2 + 1 + 1 - 10) * (x_final * 2 + 1 + 1 - 10), 1e-5, "should match the previous output");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(ingrad);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
}

TEST_CASE("learn 2 * x + y = 12, first learn x, and then learn y, evaluate convergence")
{
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	x->data.f32[0] = 1;
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(
			KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_copy(x))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "mul");
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	y->data.f32[0] = 2;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_copy(y))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "add");
	ccv_cnnp_model_t* const left = ccv_cnnp_sequential_new(MODEL_LIST(mul, add), 1, "seq");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t left_out = ccv_cnnp_model_apply(left, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(left_out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff, diff));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 1, 0);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1, 0.1, 0, 0), CMD_NOOP());
	// Train add exclusively.
	ccv_cnnp_model_set_minimizer(final, CMD_NOOP(), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(mul, ALL_PARAMETERS, ALL_PARAMETERS)));
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 12;
	o_tensor->data.f32[0] = 12;
	int i;
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_NOT_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 12, 1e-5, "after 10 iterations, output should not be the original");
	// Switch to train mul exclusively.
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(mul, ALL_PARAMETERS, ALL_PARAMETERS)));
	ccv_cnnp_model_set_minimizer(final, CMD_NOOP(), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(add, ALL_PARAMETERS, ALL_PARAMETERS)));
	float old_o = o_tensor->data.f32[0];
	for (i = 0; i < 10; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE(o_tensor->data.f32[0] < old_o, "we should be closer to 0 at this point");
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.001, 1, 0.001, 0, 0), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(mul, ALL_PARAMETERS, ALL_PARAMETERS)));
	for (i = 0; i < 1000; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 0, 1e-5, "the mean squared error should be 0 at this point");
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("learn 2 * x + y = 12, first learn x, and then learn y, evaluate learn-ability")
{
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	x->data.f32[0] = 1;
	ccv_cnnp_model_t* mul = ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(
			KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_copy(x))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "mul");
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	y->data.f32[0] = 2;
	ccv_cnnp_model_t* add = ccv_cnnp_cmd_exec(CMD_EWSUM_FORWARD(), ccv_nnc_no_hint, 0,
		MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO),
			KV(CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE, ccv_cnnp_cmd_exec_io_copy(y))),
		MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, "add");
	ccv_cnnp_model_t* const left = ccv_cnnp_sequential_new(MODEL_LIST(mul, add), 1, "seq");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t left_out = ccv_cnnp_model_apply(left, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(left_out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff, diff));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 1, 0);
	const ccv_nnc_tensor_param_t a = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t f = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(mul, 0, 0), x);
	// Train add exclusively.
	ccv_cnnp_model_set_minimizer(final, CMD_NOOP(), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(mul, ALL_PARAMETERS, ALL_PARAMETERS)));
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, a, 0);
	ccv_nnc_tensor_t* f_tensor = ccv_nnc_tensor_new(0, f, 0);
	ccv_nnc_tensor_t* o_tensor = ccv_nnc_tensor_new(0, o, 0);
	a_tensor->data.f32[0] = 2;
	f_tensor->data.f32[0] = 12;
	o_tensor->data.f32[0] = 12;
	int i;
	for (i = 0; i < 1000; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 0, 5e-3, "the mean squared error should be 0 at this point");
	ccv_cnnp_model_parameter_copy(final, ccv_cnnp_model_parameters(add, 0, 0), x);
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 10, 1e-1, "the weight on add should be 10");
	// Switch to train mul exclusively. Reset its value.
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(add, 0, 0), y);
	ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(mul, ALL_PARAMETERS, ALL_PARAMETERS)));
	ccv_cnnp_model_set_minimizer(final, CMD_NOOP(), 0, MODEL_IO_LIST(ccv_cnnp_model_parameters(add, ALL_PARAMETERS, ALL_PARAMETERS)));
	for (i = 0; i < 1000; i++)
	{
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(a_tensor, f_tensor), TENSOR_LIST(o_tensor), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	REQUIRE_EQ_WITH_TOLERANCE(o_tensor->data.f32[0], 0, 5e-3, "the mean squared error should be 0 at this point");
	ccv_cnnp_model_parameter_copy(final, ccv_cnnp_model_parameters(mul, 0, 0), x);
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[0], 5, 1e-2, "the weight on add should be 10");
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(o_tensor);
	ccv_nnc_tensor_free(f_tensor);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("a compiled model absorbs a new model with slightly different configuration")
{
	ccv_cnnp_model_t* const multi_layer = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(1, 0, 0, 1, 0)
	), 1, "multi_layer");
	ccv_nnc_tensor_param_t x = CPU_TENSOR_NHWC(32F, 2, 2);
	ccv_cnnp_model_compile(multi_layer, TENSOR_PARAM_LIST(x), CMD_SGD_FORWARD(0, 0.01, 1, 0.01, 0, 0), CMD_NOOP());
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, x, 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 4; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	ccv_cnnp_model_evaluate(multi_layer, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(x_tensor), TENSOR_LIST(y_tensor), 0, 0);
	ccv_cnnp_model_t* const small_model = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(1, 0, 0, 1, 0)
	), 1, "multi_layer");
	x = CPU_TENSOR_NHWC(32F, 1, 2);
	ccv_cnnp_model_absorb(multi_layer, small_model, TENSOR_PARAM_LIST(x));
	ccv_nnc_tensor_t* const small_x = ccv_nnc_tensor_new(0, x, 0);
	ccv_nnc_tensor_t* const small_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	memcpy(small_x->data.f32, x_tensor->data.f32, sizeof(float) * 2);
	ccv_cnnp_model_evaluate(multi_layer, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(small_x), TENSOR_LIST(small_y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(small_y->data.f32[0], y_tensor->data.f32[0], 1e-5, "the parameters retained, the value should be too");
	ccv_cnnp_model_t* const large_model = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(2, 0, 0, 1, 0),
		ccv_cnnp_dense(1, 0, 0, 1, 0)
	), 1, "multi_layer");
	x = CPU_TENSOR_NHWC(32F, 4, 2);
	ccv_cnnp_model_absorb(multi_layer, large_model, TENSOR_PARAM_LIST(x));
	ccv_nnc_tensor_t* const large_x = ccv_nnc_tensor_new(0, x, 0);
	ccv_nnc_tensor_t* const large_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1), 0);
	memcpy(large_x->data.f32, x_tensor->data.f32, sizeof(float) * 4);
	for (i = 4; i < 8; i++)
		large_x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_cnnp_model_evaluate(multi_layer, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(large_x), TENSOR_LIST(large_y), 0, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, large_y->data.f32, y_tensor->data.f32, 2, 1e-5, "the parameters retained, the value should be too");
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(small_y);
	ccv_nnc_tensor_free(small_x);
	ccv_nnc_tensor_free(large_y);
	ccv_nnc_tensor_free(large_x);
	ccv_cnnp_model_free(multi_layer);
}

TEST_CASE("use linear model's parameter as the input for more computation")
{
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 0, 0, 1, 0);
	ccv_cnnp_model_t* const multi_layer = ccv_cnnp_sequential_new(MODEL_LIST(
		linear,
	), 1, "multi_layer");
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(multi_layer, MODEL_IO_LIST(input));
	out = ccv_cnnp_model_apply(ccv_cnnp_matmul(NO_TRANSPOSE, NO_TRANSPOSE, 0, 0), MODEL_IO_LIST(out, ccv_cnnp_model_parameters(linear, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0)));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_EWPROD_FORWARD(), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff, diff));
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 1, 0);
	const ccv_nnc_tensor_param_t x_params = CPU_TENSOR_NHWC(32F, 1);
	const ccv_nnc_tensor_param_t t_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(x_params, t_params), CMD_SGD_FORWARD(0, 0.05, 1, 0, 0, 0), CMD_NOOP());
	ccv_cnnp_model_t* const final = ccv_cnnp_model_copy(model, 1);
	ccv_cnnp_model_free(model);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(x_params, t_params), CMD_SGD_FORWARD(0, 0.05, 1, 0, 0, 0), CMD_NOOP());
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	x->data.f32[0] = 1.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(final, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0), x);
	x->data.f32[0] = 0;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(final, CCV_CNNP_PARAMETER_SELECT_BIAS, 0), x);
	int i;
	for (i = 0; i < 1000; i++)
	{
		if (i % 2 == 0)
		{
			x->data.f32[0] = 1;
			t->data.f32[0] = 3;
		} else {
			x->data.f32[0] = 2;
			t->data.f32[0] = 4;
		}
		float lr = 0.05;
		if (i >= 100)
			lr = 0.01;
		else if (i >= 500)
			lr = 0.001;
		ccv_cnnp_model_set_minimizer(final, CMD_SGD_FORWARD(0, lr, 1, 0, 0, 0), 0, 0, 0);
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, TENSOR_LIST(x, t), TENSOR_LIST(y), 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	x->data.f32[0] = 1;
	t->data.f32[0] = 3;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x, t), TENSOR_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], 0, 1e-2, "the mean squared error should be 0 at this point");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("model can have multiple outputs and some of them can be used in the computation")
{
	ccv_cnnp_model_t* const linear1 = ccv_cnnp_dense(1, 1, 0, 1, 0);
	ccv_cnnp_model_t* const linear2 = ccv_cnnp_dense(1, 1, 0, 1, 0);
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear1, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(linear2, MODEL_IO_LIST(out1));
	ccv_cnnp_model_t* const multi_layer = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out1, out2), 1, 0);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(multi_layer, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(multi_layer, ccv_cnnp_model_parameters(linear1, ALL_PARAMETERS, 0), t);
	t->data.f32[0] = -1.5;
	ccv_cnnp_model_set_parameter(multi_layer, ccv_cnnp_model_parameters(linear2, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(multi_layer, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(t, y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(t->data.f32[0], 10 * 2.4, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], -10 * 2.4 * 1.5, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(multi_layer);
}

TEST_CASE("index select model can select a part from vocabulary")
{
	ccv_cnnp_model_t* const index_select = ccv_cnnp_index_select(0);
	const ccv_nnc_tensor_param_t v_params = CPU_TENSOR_NHWC(32F, 10, 8);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, v_params, 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10 * 8; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	const ccv_nnc_tensor_param_t x_params = CPU_TENSOR_NHWC(32S, 3);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, x_params, 0);
	ccv_cnnp_model_compile(index_select, TENSOR_PARAM_LIST(v_params, x_params), CMD_NOOP(), CMD_NOOP());
	x->data.i32[0] = 1;
	x->data.i32[1] = 0;
	x->data.i32[2] = 5;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 8), 0);
	ccv_cnnp_model_evaluate(index_select, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(v, x), TENSOR_LIST(y), 0, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 1 * 8, y->data.f32, 8, 1e-5, "index 1st vector");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 0 * 8, y->data.f32 + 8, 8, 1e-5, "index 0th vector");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 5 * 8, y->data.f32 + 8 * 2, 8, 1e-5, "index 5th vector");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(v);
	ccv_cnnp_model_free(index_select);
}

TEST_CASE("embedding model can generate vector embedding")
{
	ccv_cnnp_model_t* const embedding = ccv_cnnp_embedding(CCV_32F, 10, 8, 1, 0);
	const ccv_nnc_tensor_param_t x_params = CPU_TENSOR_NHWC(32S, 3);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, x_params, 0);
	ccv_cnnp_model_compile(embedding, TENSOR_PARAM_LIST(x_params), CMD_NOOP(), CMD_NOOP());
	x->data.i32[0] = 1;
	x->data.i32[1] = 0;
	x->data.i32[2] = 5;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 8), 0);
	ccv_cnnp_model_evaluate(embedding, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(y), 0, 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 8), 0);
	ccv_cnnp_model_parameter_copy(embedding, ccv_cnnp_model_parameters(embedding, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0), v);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 1 * 8, y->data.f32, 8, 1e-5, "index 1st vector");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 0 * 8, y->data.f32 + 8, 8, 1e-5, "index 0th vector");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, v->data.f32 + 5 * 8, y->data.f32 + 8 * 2, 8, 1e-5, "index 5th vector");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(v);
	ccv_cnnp_model_free(embedding);
}

TEST_CASE("model to get the internal name for parameters")
{
	ccv_cnnp_model_t* const linear1 = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_t* const linear2 = ccv_cnnp_dense(1, 1, 0, 1, 0);
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear1, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(linear2, MODEL_IO_LIST(out1));
	ccv_cnnp_model_t* const multi_layer = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out1, out2), 1, 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(multi_layer, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	const char* linear1p = "t-linear-0-0";
	REQUIRE(memcmp(linear1p, ccv_cnnp_model_parameter_name(multi_layer, ccv_cnnp_model_parameters(linear1, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0)), strlen(linear1p) + 1) == 0, "should be equal");
	const char* linear2p = "t-0-0";
	REQUIRE(memcmp(linear2p, ccv_cnnp_model_parameter_name(multi_layer, ccv_cnnp_model_parameters(linear2, CCV_CNNP_PARAMETER_SELECT_WEIGHT, 0)), strlen(linear2p) + 1) == 0, "should be equal");
	ccv_cnnp_model_free(multi_layer);
}

static ccv_cnnp_model_t* _resnet_block_new(const int filters, const int expansion, const int strides, const int projection_shortcut)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t shortcut = input;
	if (projection_shortcut)
	{
		ccv_cnnp_model_t* const avgdown = ccv_cnnp_average_pool(DIM_ALLOC(strides, strides), HINT((strides, strides), (0, 0)), 0);
		shortcut = ccv_cnnp_model_apply(avgdown, MODEL_IO_LIST(input));
		ccv_cnnp_model_t* const conv0 = ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), DIM_ALLOC(), 1, HINT((1, 1), (0, 0)), 0, 1, 0);
		shortcut = ccv_cnnp_model_apply(conv0, MODEL_IO_LIST(shortcut));
	}
	ccv_cnnp_model_t* const conv1 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0),
		ccv_cnnp_relu(0)
	), 1, 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(conv1, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const conv2 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), DIM_ALLOC(), 0, HINT((strides, strides), (1, 1)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0),
		ccv_cnnp_relu(0)
	), 1, 0);
	output = ccv_cnnp_model_apply(conv2, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const conv3 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0)
	), 1, 0);
	output = ccv_cnnp_model_apply(conv3, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum(0);
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output, shortcut));
	ccv_cnnp_model_t* const relu = ccv_cnnp_relu(0);
	output = ccv_cnnp_model_apply(relu, MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 1, 0);
}

static ccv_cnnp_model_t* _resnet_block_layer_new(const int filters, const int expansion, const int strides, const int blocks)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* first_block = _resnet_block_new(filters, expansion, strides, 1);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(first_block, MODEL_IO_LIST(input));
	int i;
	for (i = 1; i < blocks; i++)
	{
		ccv_cnnp_model_t* block = _resnet_block_new(filters, expansion, 1, 0);
		output = ccv_cnnp_model_apply(block, MODEL_IO_LIST(output));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 1, 0);
}

static void _fpn(const int d, const ccv_cnnp_model_io_t* const c, const int c_size, ccv_cnnp_model_io_t* const p)
{
	int i;
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0), MODEL_IO_LIST(c[c_size - 1]));
	p[c_size - 1] = output;
	for (i = c_size - 2; i >= 0; i--)
	{
		const ccv_cnnp_model_io_t lateral = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, 0), MODEL_IO_LIST(c[i]));
		const ccv_cnnp_model_io_t up = ccv_cnnp_model_apply(ccv_cnnp_upsample(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2, 0, 0), MODEL_IO_LIST(output));
		const ccv_cnnp_model_io_t sum = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(lateral, up));
		output = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(3, 3), DIM_ALLOC(), 1, HINT((1, 1), (1, 1)), 0, 1, 0), MODEL_IO_LIST(sum));
		p[i] = output;
	}
}

ccv_cnnp_model_t* _imagenet_resnet50_v1d_fpn(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), DIM_ALLOC(), 1, HINT((2, 2), (1, 1)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), DIM_ALLOC(), 1, HINT((1, 1), (1, 1)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), DIM_ALLOC(), 1, HINT((1, 1), (1, 1)), 0, 1, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 1, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), HINT((2, 2), (1, 1)), 0)
	), 1, 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(64, 4, 1, 3), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c2 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(128, 4, 2, 4), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c3 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(256, 4, 2, 6), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c4 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(512, 4, 2, 3), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c5 = output;
	const ccv_cnnp_model_io_t c[] = { c2, c3, c4, c5 };
	ccv_cnnp_model_io_t p[5];
	_fpn(256, c, 4, p);
	p[4] = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(2, 2), HINT((2, 2), (0, 0)), 0), MODEL_IO_LIST(p[3]));
	// 3 aspect ratios (1:2, 1:1, 2:1). Each has 4 + 2 (x, y, w, h, object, non-object), total 18.
	ccv_cnnp_model_t* const rpn_proposals = ccv_cnnp_convolution(1, 18, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, "rpn");
	ccv_cnnp_model_io_t proposals[5];
	int i;
	for (i = 0; i < 5; i++)
		proposals[i] = ccv_cnnp_model_apply(rpn_proposals, MODEL_IO_LIST(p[i]));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), proposals, 5, 1, 0);
}

TEST_CASE("FPN-RPN use cnnp model with multiple outputs")
{
	ccv_cnnp_model_t* rpn = _imagenet_resnet50_v1d_fpn();
	ccv_nnc_tensor_param_t input_params = GPU_TENSOR_NCHW(000, 32F, 4, 3, 835, 1146);
	ccv_cnnp_model_compile(rpn, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_param_t output_params[5];
	ccv_cnnp_model_tensor_auto(rpn, output_params, 5);
	REQUIRE_EQ(output_params[0].dim[2], 209, "should be equal");
	REQUIRE_EQ(output_params[0].dim[3], 287, "should be equal");
	REQUIRE_EQ(output_params[1].dim[2], 105, "should be equal");
	REQUIRE_EQ(output_params[1].dim[3], 144, "should be equal");
	REQUIRE_EQ(output_params[2].dim[2], 53, "should be equal");
	REQUIRE_EQ(output_params[2].dim[3], 72, "should be equal");
	REQUIRE_EQ(output_params[3].dim[2], 27, "should be equal");
	REQUIRE_EQ(output_params[3].dim[3], 36, "should be equal");
	REQUIRE_EQ(output_params[4].dim[2], 13, "should be equal");
	REQUIRE_EQ(output_params[4].dim[3], 18, "should be equal");
	ccv_cnnp_model_free(rpn);
}

TEST_CASE("extract one output each feed into different feed-forward")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const sigmoid = ccv_cnnp_sigmoid("sigmoid");
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(sigmoid, MODEL_IO_LIST(out1));
	ccv_cnnp_model_t* tiny = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out1, out2), 1, "tiny");
	const ccv_cnnp_model_io_t i0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t o0 = ccv_cnnp_model_apply(tiny, MODEL_IO_LIST(i0));
	ccv_cnnp_model_io_t o00 = ccv_cnnp_model_apply(ccv_cnnp_extract(0, "index0"), MODEL_IO_LIST(o0));
	ccv_cnnp_model_io_t o01 = ccv_cnnp_model_apply(ccv_cnnp_extract(1, "index1"), MODEL_IO_LIST(o0));
	ccv_cnnp_model_t* const l0 = ccv_cnnp_dense(1, 1, 0, 1, "l0");
	ccv_cnnp_model_io_t o10 = ccv_cnnp_model_apply(l0, MODEL_IO_LIST(o00));
	ccv_cnnp_model_t* const l1 = ccv_cnnp_dense(1, 1, 0, 1, "l1");
	ccv_cnnp_model_io_t o11 = ccv_cnnp_model_apply(l1, MODEL_IO_LIST(o01));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(i0), MODEL_IO_LIST(o10, o11), 1, "final");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), t);
	t->data.f32[0] = -1.5;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(l0, ALL_PARAMETERS, 0), t);
	t->data.f32[0] = 1.7;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(l1, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(t, y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(t->data.f32[0], 10 * 2.4 * -1.5, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], 1 / (1 + exp(-10 * 2.4)) * 1.7, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use parameter for values")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const sigmoid = ccv_cnnp_sigmoid("sigmoid");
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(sigmoid, MODEL_IO_LIST(out1));
	ccv_cnnp_model_t* const value = ccv_cnnp_parameter(CPU_TENSOR_NCHW(32F, 1), 0, 1, "value");
	ccv_cnnp_model_io_t out3 = ccv_cnnp_model_apply(value, 0, 0);
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out4 = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out2, out3));
	ccv_cnnp_model_t* final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out4), 1, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), t);
	t->data.f32[0] = -1.5;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(value, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], 1 / (1 + exp(-10 * 2.4)) - 1.5, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use scalar for values")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const sigmoid = ccv_cnnp_sigmoid("sigmoid");
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(sigmoid, MODEL_IO_LIST(out1));
	ccv_cnnp_model_io_t value = ccv_cnnp_model_apply(ccv_cnnp_scalar(CCV_TENSOR_CPU_MEMORY, CCV_TENSOR_FORMAT_NHWC, CCV_32F, 1.5, "value"), 0, 0);
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out4 = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out2, value));
	ccv_cnnp_model_t* final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out4), 1, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], 1 / (1 + exp(-10 * 2.4)) + 1.5, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use scalar for values and copy types from other inputs")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const sigmoid = ccv_cnnp_sigmoid("sigmoid");
	ccv_cnnp_model_io_t out2 = ccv_cnnp_model_apply(sigmoid, MODEL_IO_LIST(out1));
	ccv_cnnp_model_io_t value = ccv_cnnp_model_apply(ccv_cnnp_scalar(0, 0, 0, 1.5, "value"), MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out4 = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out2, value));
	ccv_cnnp_model_t* final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out4), 1, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x), TENSOR_LIST(y), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y->data.f32[0], 1 / (1 + exp(-10 * 2.4)) + 1.5, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("LoRA fine-tuning GEMM set is_trainable to false")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_t* const down = ccv_cnnp_dense(2, 1, 0, 1, "down");
	ccv_cnnp_model_t* const up = ccv_cnnp_dense(10, 1, 0, 1, "up");
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_down = ccv_cnnp_model_apply(down, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_up = ccv_cnnp_model_apply(up, MODEL_IO_LIST(out_down));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out_final = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out, out_up));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out_final), 0, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const tlinear = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	int i;
	for (i = 0; i < 10 * 10; i++)
		tlinear->data.f32[i] = (i / 10 == i % 10) ? 1 : 0;
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2), 0);
	for (i = 0; i < 10 * 2; i++)
		t->data.f32[i] = 0;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 10);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_SGD_FORWARD(1, 0.01, 1, 0.1, 0, 0), CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN));
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), tlinear);
	ccv_nnc_tensor_free(tlinear);
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(up, ALL_PARAMETERS, 0), t);
	ccv_nnc_tensor_free(t);
	for (i = 0; i < 10; i++)
		x->data.f32[i] = i;
	ccv_nnc_tensor_t* const target = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	for (i = 0; i < 10; i++)
		target->data.f32[i] = 10 - i;
	for (i = 0; i < 10; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y->data.f32, target->data.f32, 10, 1e-2, "should match the target after fine-tuning");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(final), 0, "should be marked as not trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up), 1, "should be marked as trainable");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(target);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("LoRA fine-tuning convolution set is_trainable to false")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const conv = ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), DIM_ALLOC(), 0, HINT((1, 1), (1, 1)), 0, -1, "conv");
	ccv_cnnp_model_t* const down = ccv_cnnp_convolution(1, 4, DIM_ALLOC(3, 3), DIM_ALLOC(), 0, HINT((1, 1), (1, 1)), 0, 1, "down");
	ccv_cnnp_model_t* const up = ccv_cnnp_convolution(1, 32, DIM_ALLOC(1, 1), DIM_ALLOC(), 0, HINT((1, 1), (0, 0)), 0, 1, "up");
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(conv, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_down = ccv_cnnp_model_apply(down, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_up = ccv_cnnp_model_apply(up, MODEL_IO_LIST(out_down));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out_final = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out, out_up));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out_final), 0, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 5, 10), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 5, 32), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 5, 5, 10);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 1,
	}, TENSOR_LIST(x), TENSOR_LIST(y), 0, 0);
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(final), 0, "should be marked as not trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up), 1, "should be marked as trainable");
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

static int _ccv_nnc_same_namer(void* context, const char* src_name, char* updated_name, const size_t provided_size)
{
	const size_t src_len = ccv_min(strnlen(src_name, provided_size - 1), provided_size - 1);
	memcpy(updated_name, src_name, src_len);
	updated_name[src_len] = '\0';
	return 0;
}

TEST_CASE("two models share the same parameters")
{
	const ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear0 = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_t* const down0 = ccv_cnnp_dense(2, 1, 0, 1, "down");
	ccv_cnnp_model_t* const up0 = ccv_cnnp_dense(10, 1, 0, 1, "up");
	ccv_cnnp_model_io_t out0 = ccv_cnnp_model_apply(linear0, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t out0_down = ccv_cnnp_model_apply(down0, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t out0_up = ccv_cnnp_model_apply(up0, MODEL_IO_LIST(out0_down));
	ccv_cnnp_model_t* const add0 = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out0_final = ccv_cnnp_model_apply(add0, MODEL_IO_LIST(out0, out0_up));
	ccv_cnnp_model_t* const final0 = ccv_cnnp_model_new(MODEL_IO_LIST(input0), MODEL_IO_LIST(out0_final), 0, "tiny0");

	const ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear1 = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_t* const down1 = ccv_cnnp_dense(2, 1, 0, 1, "down");
	ccv_cnnp_model_t* const up1 = ccv_cnnp_dense(10, 1, 0, 1, "up");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear1, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t out1_down = ccv_cnnp_model_apply(down1, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t out1_up = ccv_cnnp_model_apply(up1, MODEL_IO_LIST(out1_down));
	ccv_cnnp_model_t* const add1 = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out1_final = ccv_cnnp_model_apply(add1, MODEL_IO_LIST(out1, out1_up));
	ccv_cnnp_model_t* const final1 = ccv_cnnp_model_new(MODEL_IO_LIST(input1), MODEL_IO_LIST(out1_final), 0, "tiny1");

	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const y0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const y1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 10);
	ccv_cnnp_model_compile(final0, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(final0, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(x), TENSOR_LIST(y0), 0, 0);
	ccv_cnnp_model_compile(final1, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_share_parameters(final1, ccv_cnnp_model_parameters(final1, ALL_PARAMETERS, ALL_PARAMETERS), final0, ccv_cnnp_model_parameters(final0, ALL_PARAMETERS, ALL_PARAMETERS), 0, 0);
	ccv_cnnp_model_evaluate(final1, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(x), TENSOR_LIST(y1), 0, 0);
	REQUIRE_TENSOR_EQ(y0, y1, "two model now shares the weights, should have the same result");
	CNNP_MODEL_GEN(final0, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y0);
	ccv_nnc_tensor_free(y1);
	ccv_cnnp_model_free(final0);
	ccv_cnnp_model_free(final1);
}

TEST_CASE("two models, one with LoRA, one with not, share the same parameters")
{
	const ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear0 = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_io_t out0 = ccv_cnnp_model_apply(linear0, MODEL_IO_LIST(input0));
	ccv_cnnp_model_t* const final0 = ccv_cnnp_model_new(MODEL_IO_LIST(input0), MODEL_IO_LIST(out0), 0, "tiny");

	const ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear1 = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_t* const down1 = ccv_cnnp_dense(2, 1, 0, 1, "down");
	ccv_cnnp_model_t* const up1 = ccv_cnnp_dense(10, 1, 0, 1, "up");
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear1, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t out1_down = ccv_cnnp_model_apply(down1, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t out1_up = ccv_cnnp_model_apply(up1, MODEL_IO_LIST(out1_down));
	ccv_cnnp_model_t* const add1 = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out1_final = ccv_cnnp_model_apply(add1, MODEL_IO_LIST(out1, out1_up));
	ccv_cnnp_model_t* const final1 = ccv_cnnp_model_new(MODEL_IO_LIST(input1), MODEL_IO_LIST(out1_final), 0, "tiny");

	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 10; i++)
		x->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const y0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const y1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 10);
	ccv_cnnp_model_compile(final0, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(final0, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(x), TENSOR_LIST(y0), 0, 0);
	ccv_cnnp_model_compile(final1, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const up_weights = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10), 0);
	for (i = 0; i < 2 * 10; i++)
		up_weights->data.f32[i] = 0;
	ccv_cnnp_model_set_parameter(final1, ccv_cnnp_model_parameters(up1, ALL_PARAMETERS, ALL_PARAMETERS), up_weights);
	ccv_nnc_tensor_free(up_weights);
	ccv_cnnp_model_share_parameters(final1, ccv_cnnp_model_parameters(final1, ALL_PARAMETERS, ALL_PARAMETERS), final0, ccv_cnnp_model_parameters(final0, ALL_PARAMETERS, ALL_PARAMETERS), _ccv_nnc_same_namer, 0);
	ccv_cnnp_model_evaluate(final1, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(x), TENSOR_LIST(y1), 0, 0);
	REQUIRE_TENSOR_EQ(y0, y1, "two model now shares the weights, should have the same result");
	CNNP_MODEL_GEN(final0, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(y0);
	ccv_nnc_tensor_free(y1);
	ccv_cnnp_model_free(final0);
	ccv_cnnp_model_free(final1);
}

TEST_CASE("pad a tensor with padding")
{
	const ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	const ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const pad = ccv_cnnp_pad(CCV_NNC_PAD_ZERO, DIM_ALLOC(0, 2, 2, 0), DIM_ALLOC(0, 1, 2, 1), "pad");
	ccv_cnnp_model_io_t out0 = ccv_cnnp_model_apply(pad, MODEL_IO_LIST(input0));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out0, input1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(out), 0, "tiny");

	ccv_nnc_tensor_t* const x0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 3, 3, 10), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 3 * 3 * 10; i++)
		x0->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const x1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 6, 7, 11), 0);
	for (i = 0; i < 6 * 7 * 11; i++)
		x1->data.f32[i] = 1;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 6, 7, 11), 0);
	ccv_nnc_tensor_param_t input0_params = CPU_TENSOR_NHWC(32F, 1, 3, 3, 10);
	ccv_nnc_tensor_param_t input1_params = CPU_TENSOR_NHWC(32F, 1, 6, 7, 11);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input0_params, input1_params), CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
	}, TENSOR_LIST(x0, x1), TENSOR_LIST(y), 0, 0);
	int j, k;
	ccv_nnc_tensor_t* const y0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 6, 7, 11), 0);
	for (i = 0; i < 6; i++)
		for (j = 0; j < 7; j++)
			for (k = 0; k < 11; k++)
				y0->data.f32[i * 7 * 11 + j * 11 + k] = (i >= 2 && i < 5 && j >=2 && j < 5 && k < 10) ? 1 + x0->data.f32[(i - 2) * 3 * 10 + (j - 2) * 10 + k] : 1;
	REQUIRE_TENSOR_EQ(y, y0, "it should be padded");
	CNNP_MODEL_GEN(pad, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_free(x0);
	ccv_nnc_tensor_free(x1);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(y0);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use move semantics to write output to the empty space of the input tensor")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t input0 = ccv_cnnp_model_apply(ccv_cnnp_reshape(CCV_TENSOR_FORMAT_NHWC, DIM_ALLOC(1), DIM_ALLOC(0), DIM_ALLOC(1), "first reshape"), MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t input1 = ccv_cnnp_model_apply(ccv_cnnp_reshape(CCV_TENSOR_FORMAT_NHWC, DIM_ALLOC(1), DIM_ALLOC(1), DIM_ALLOC(1), "second reshape"), MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t move0 = ccv_cnnp_model_apply(ccv_cnnp_move("move"), MODEL_IO_LIST(out1, input1));
	const ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	ccv_cnnp_model_io_t out1_final = ccv_cnnp_model_apply(ccv_cnnp_sum("sum"), MODEL_IO_LIST(move0, input2));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input, input2), MODEL_IO_LIST(out1_final), 0, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const z = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 2);
	ccv_nnc_tensor_param_t input2_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params, input2_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	t->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), t);
	x->data.f32[0] = 10;
	x->data.f32[1] = 0;
	y->data.f32[0] = 3;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x, y), TENSOR_LIST(z), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], 2.4 * 10 + 3, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(x->data.f32[1], 2.4 * 10, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use variable and move semantics to co-locate input in the same tensor")
{
	const ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	const ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear0 = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out0 = ccv_cnnp_model_apply(linear0, MODEL_IO_LIST(input0));
	ccv_cnnp_model_io_t out1 = ccv_cnnp_model_apply(linear0, MODEL_IO_LIST(input1));
	ccv_cnnp_model_io_t var = ccv_cnnp_model_apply(ccv_cnnp_variable(CPU_TENSOR_NHWC(32F, 2), "var"), MODEL_IO_LIST());
	ccv_cnnp_model_io_t var0 = ccv_cnnp_model_apply(ccv_cnnp_reshape(CCV_TENSOR_FORMAT_NHWC, DIM_ALLOC(1), DIM_ALLOC(0), DIM_ALLOC(1), "first reshape"), MODEL_IO_LIST(var));
	ccv_cnnp_model_io_t var1 = ccv_cnnp_model_apply(ccv_cnnp_reshape(CCV_TENSOR_FORMAT_NHWC, DIM_ALLOC(1), DIM_ALLOC(1), DIM_ALLOC(1), "second reshape"), MODEL_IO_LIST(var));
	ccv_cnnp_model_io_t move0 = ccv_cnnp_model_apply(ccv_cnnp_move("move"), MODEL_IO_LIST(out0, var0));
	ccv_cnnp_model_io_t move1 = ccv_cnnp_model_apply(ccv_cnnp_move("move"), MODEL_IO_LIST(out1, var1));
	ccv_cnnp_model_t* const linear1 = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t out1_final = ccv_cnnp_model_apply(linear1, MODEL_IO_LIST(var));
	ccv_cnnp_model_add_dependencies(out1_final, MODEL_IO_LIST(move0, move1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1), MODEL_IO_LIST(out1_final), 0, "tiny");
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const z = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_nnc_tensor_param_t input2_params = CPU_TENSOR_NHWC(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params, input2_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const t0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	t0->data.f32[0] = 2.4;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear0, ALL_PARAMETERS, 0), t0);
	ccv_nnc_tensor_t* const t1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	t1->data.f32[0] = -1.1;
	t1->data.f32[1] = 1.2;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear1, ALL_PARAMETERS, 0), t1);
	x->data.f32[0] = 10;
	y->data.f32[0] = 3;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x, y), TENSOR_LIST(z), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(z->data.f32[0], -1.1 * 2.4 * 10 + 3 * 2.4 * 1.2, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(t0);
	ccv_nnc_tensor_free(t1);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(z);
	ccv_cnnp_model_free(final);
}

TEST_CASE("use contiguous to make certain tensor contiguous during model inference")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear0 = ccv_cnnp_dense(4, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t y = ccv_cnnp_model_apply(linear0, MODEL_IO_LIST(x));
	// Get the middle 2, and then apply GELU, which in Float32 / CPU, requires to be contiguous for now.
	ccv_cnnp_model_io_t y0 = ccv_cnnp_model_apply(ccv_cnnp_reshape(CCV_TENSOR_FORMAT_NHWC, DIM_ALLOC(2, 2), DIM_ALLOC(0, 2), DIM_ALLOC(4, 1), "reshape"), MODEL_IO_LIST(y));
	/* Using just data transfer is not enough.
	ccv_cnnp_model_io_t moved = ccv_cnnp_model_apply(ccv_cnnp_variable(CPU_TENSOR_NHWC(32F, 2, 2), 0), MODEL_IO_LIST());
	ccv_cnnp_model_io_t y_copied = ccv_cnnp_model_apply(ccv_cnnp_move(0), MODEL_IO_LIST(y0, moved));
	ccv_cnnp_model_io_t z = ccv_cnnp_model_apply(ccv_cnnp_sigmoid("sigmoid"), MODEL_IO_LIST(y_copied));
	*/
	// Have to use the new contiguous model.
	ccv_cnnp_model_io_t y_copied = ccv_cnnp_model_apply(ccv_cnnp_contiguous(0), MODEL_IO_LIST(y0));
	ccv_cnnp_model_io_t z = ccv_cnnp_model_apply(ccv_cnnp_sigmoid("sigmoid"), MODEL_IO_LIST(y_copied));
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(z), 0, "tiny");
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(x_tensor->info), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const t0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	t0->data.f32[0] = 2.4;
	t0->data.f32[1] = -0.4;
	t0->data.f32[2] = 1.2;
	t0->data.f32[3] = -3.6;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear0, ALL_PARAMETERS, 0), t0);
	x_tensor->data.f32[0] = 1;
	x_tensor->data.f32[1] = -1;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x_tensor), TENSOR_LIST(z_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[0], 1.0 / (1.0 + exp(-1.2)), 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[1], 1.0 / (1.0 + exp(3.6)), 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[2], 1.0 / (1.0 + exp(1.2)), 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(z_tensor->data.f32[3], 1.0 / (1.0 + exp(-3.6)), 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(t0);
	ccv_nnc_tensor_free(z_tensor);
	ccv_cnnp_model_free(final);
}

TEST_CASE("chunk a tensor into several smaller ones, variant 1")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_t* const chunk = ccv_cnnp_chunk(2, 1, "chunk");
	ccv_cnnp_model_io_t y = ccv_cnnp_model_apply(chunk, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t y0 = ccv_cnnp_model_apply(ccv_cnnp_extract(0, "index0"), MODEL_IO_LIST(y));
	ccv_cnnp_model_io_t o0 = ccv_cnnp_model_apply(ccv_cnnp_contiguous(0), MODEL_IO_LIST(y0));
	ccv_cnnp_model_io_t y1 = ccv_cnnp_model_apply(ccv_cnnp_extract(1, "index1"), MODEL_IO_LIST(y));
	ccv_cnnp_model_io_t o1 = ccv_cnnp_model_apply(ccv_cnnp_contiguous(0), MODEL_IO_LIST(y1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(o0, o1), 0, "tiny");
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(x_tensor->info), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	x_tensor->data.f32[0] = 1;
	x_tensor->data.f32[1] = -1;
	x_tensor->data.f32[2] = 2;
	x_tensor->data.f32[3] = 3;
	x_tensor->data.f32[4] = 4;
	x_tensor->data.f32[5] = 5;
	x_tensor->data.f32[6] = 6;
	x_tensor->data.f32[7] = 7;
	ccv_nnc_tensor_t* const y0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_tensor_t* const y1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x_tensor), TENSOR_LIST(y0_tensor, y1_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[0], 1, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[1], -1, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[2], 4, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[3], 5, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[0], 2, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[1], 3, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[2], 6, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[3], 7, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y0_tensor);
	ccv_nnc_tensor_free(y1_tensor);
	ccv_cnnp_model_free(final);
}

TEST_CASE("chunk a tensor into several smaller ones, variant 2")
{
	const ccv_cnnp_model_io_t x = ccv_cnnp_input();
	ccv_cnnp_model_t* const chunk = ccv_cnnp_chunk(2, 0, "chunk");
	ccv_cnnp_model_io_t y = ccv_cnnp_model_apply(chunk, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t y0 = ccv_cnnp_model_apply(ccv_cnnp_extract(0, "index0"), MODEL_IO_LIST(y));
	ccv_cnnp_model_io_t o0 = ccv_cnnp_model_apply(ccv_cnnp_contiguous(0), MODEL_IO_LIST(y0));
	ccv_cnnp_model_io_t y1 = ccv_cnnp_model_apply(ccv_cnnp_extract(1, "index1"), MODEL_IO_LIST(y));
	ccv_cnnp_model_io_t o1 = ccv_cnnp_model_apply(ccv_cnnp_contiguous(0), MODEL_IO_LIST(y1));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(o0, o1), 0, "tiny");
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(x_tensor->info), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	x_tensor->data.f32[0] = 1;
	x_tensor->data.f32[1] = -1;
	x_tensor->data.f32[2] = 2;
	x_tensor->data.f32[3] = 3;
	x_tensor->data.f32[4] = 4;
	x_tensor->data.f32[5] = 5;
	x_tensor->data.f32[6] = 6;
	x_tensor->data.f32[7] = 7;
	ccv_nnc_tensor_t* const y0_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 4), 0);
	ccv_nnc_tensor_t* const y1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 4), 0);
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(x_tensor), TENSOR_LIST(y0_tensor, y1_tensor), 0, 0);
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[0], 1, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[1], -1, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[2], 2, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y0_tensor->data.f32[3], 3, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[0], 4, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[1], 5, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[2], 6, 1e-5, "should be equal to expected value");
	REQUIRE_EQ_WITH_TOLERANCE(y1_tensor->data.f32[3], 7, 1e-5, "should be equal to expected value");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y0_tensor);
	ccv_nnc_tensor_free(y1_tensor);
	ccv_cnnp_model_free(final);
}

static float _debug_value = 0;

static void _debug_test(ccv_nnc_tensor_t* const *const inputs, const int input_size, ccv_nnc_stream_context_t* const stream_context, void* const context)
{
	_debug_value = inputs[0]->data.f32[0];
}

TEST_CASE("debug can intercept model execution")
{
	ccv_cnnp_model_io_t input0 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	ccv_cnnp_model_io_t input3 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output0 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(input0, input1));
	ccv_cnnp_model_io_t output1 = ccv_cnnp_model_apply(ccv_cnnp_debug(_debug_test, 0, 0, 0, 0), MODEL_IO_LIST(output0));
	ccv_cnnp_model_io_t output2 = ccv_cnnp_model_apply(ccv_cnnp_mul(1, 0), MODEL_IO_LIST(output1, input2));
	output2 = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output2, input3));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input0, input1, input2, input3), MODEL_IO_LIST(output2), 1, 0);
	const ccv_nnc_tensor_param_t a0 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a1 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a2 = CPU_TENSOR_NCHW(32F, 1);
	const ccv_nnc_tensor_param_t a3 = CPU_TENSOR_NCHW(32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a0, a1, a2, a3), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const a0_tensor = ccv_nnc_tensor_new(0, a0, 0);
	ccv_nnc_tensor_t* const a1_tensor = ccv_nnc_tensor_new(0, a1, 0);
	ccv_nnc_tensor_t* const a2_tensor = ccv_nnc_tensor_new(0, a2, 0);
	ccv_nnc_tensor_t* const a3_tensor = ccv_nnc_tensor_new(0, a3, 0);
	ccv_nnc_tensor_t* const b2_tensor = ccv_nnc_tensor_new(0, a0, 0);
	a0_tensor->data.f32[0] = 0.5;
	a1_tensor->data.f32[0] = 0.75;
	a2_tensor->data.f32[0] = 1.75;
	a3_tensor->data.f32[0] = 2.5;
	_debug_value = 0;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
		.is_test = 1
	}, TENSOR_LIST(a0_tensor, a1_tensor, a2_tensor, a3_tensor), TENSOR_LIST(b2_tensor), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_EQ_WITH_TOLERANCE(b2_tensor->data.f32[0], (0.5 + 0.75) * 1.75 + 2.5, 1e-5, "should match the final result");
	REQUIRE_EQ_WITH_TOLERANCE(_debug_value, 0.5 + 0.75, 1e-5, "should match the intermediate result");
	ccv_cnnp_model_free(final);
	ccv_nnc_tensor_free(a0_tensor);
	ccv_nnc_tensor_free(a1_tensor);
	ccv_nnc_tensor_free(a2_tensor);
	ccv_nnc_tensor_free(a3_tensor);
	ccv_nnc_tensor_free(b2_tensor);
}

TEST_CASE("LoRA fine-tuning GEMM set is_trainable to false and with gradient checkpointing")
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const linear = ccv_cnnp_dense(10, 1, 0, -1, "linear");
	ccv_cnnp_model_t* const down = ccv_cnnp_dense(2, 1, 0, 1, "down");
	ccv_cnnp_model_t* const up = ccv_cnnp_dense(10, 1, 0, 1, "up");
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(linear, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_down = ccv_cnnp_model_apply(down, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_up = ccv_cnnp_model_apply(up, MODEL_IO_LIST(out_down));
	ccv_cnnp_model_t* const add = ccv_cnnp_sum("sum");
	ccv_cnnp_model_io_t out_final = ccv_cnnp_model_apply(add, MODEL_IO_LIST(out, out_up));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out_final), 0, "tiny");
	ccv_cnnp_model_set_gradient_checkpointing(final, 1);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const tlinear = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	int i;
	for (i = 0; i < 10 * 10; i++)
		tlinear->data.f32[i] = (i / 10 == i % 10) ? 1 : 0;
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2), 0);
	for (i = 0; i < 10 * 2; i++)
		t->data.f32[i] = 0;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 10);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_SGD_FORWARD(1, 0.01, 1, 0.1, 0, 0), CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN));
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(linear, ALL_PARAMETERS, 0), tlinear);
	ccv_nnc_tensor_free(tlinear);
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(up, ALL_PARAMETERS, 0), t);
	ccv_nnc_tensor_free(t);
	for (i = 0; i < 10; i++)
		x->data.f32[i] = i;
	ccv_nnc_tensor_t* const target = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	for (i = 0; i < 10; i++)
		target->data.f32[i] = 10 - i;
	for (i = 0; i < 10; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y->data.f32, target->data.f32, 10, 1e-2, "should match the target after fine-tuning");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(final), 0, "should be marked as not trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up), 1, "should be marked as trainable");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(target);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

TEST_CASE("LoRA fine-tuning MLP with GELU, set is_trainable to false and with gradient checkpointing")
{
	ccv_nnc_stream_context_set_seed(0, 47);
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const fc1 = ccv_cnnp_dense(10, 1, 0, -1, "fc1");
	ccv_cnnp_model_t* const fc2 = ccv_cnnp_dense(10, 1, 0, -1, "fc2");
	ccv_cnnp_model_t* const down_fc1 = ccv_cnnp_dense(2, 1, 0, 1, "down_fc1");
	ccv_cnnp_model_t* const up_fc1 = ccv_cnnp_dense(10, 1, 0, 1, "up_fc1");
	ccv_cnnp_model_t* const down_fc2 = ccv_cnnp_dense(2, 1, 0, 1, "down_fc2");
	ccv_cnnp_model_t* const up_fc2 = ccv_cnnp_dense(10, 1, 0, 1, "up_fc2");
	ccv_cnnp_model_t* const fc3 = ccv_cnnp_dense(5, 1, 0, -1, "fc3");
	ccv_cnnp_model_t* const down_fc3 = ccv_cnnp_dense(2, 1, 0, 1, "down_fc3");
	ccv_cnnp_model_t* const up_fc3 = ccv_cnnp_dense(5, 1, 0, 1, "up_fc3");
	ccv_cnnp_model_io_t out_fc1 = ccv_cnnp_model_apply(fc1, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_fc2 = ccv_cnnp_model_apply(fc2, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_down_fc1 = ccv_cnnp_model_apply(down_fc1, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_up_fc1 = ccv_cnnp_model_apply(up_fc1, MODEL_IO_LIST(out_down_fc1));
	ccv_cnnp_model_io_t out_down_fc2 = ccv_cnnp_model_apply(down_fc2, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t out_up_fc2 = ccv_cnnp_model_apply(up_fc2, MODEL_IO_LIST(out_down_fc2));
	ccv_cnnp_model_io_t out_sum_fc1 = ccv_cnnp_model_apply(ccv_cnnp_sum("sum_fc1"), MODEL_IO_LIST(out_fc1, out_up_fc1));
	ccv_cnnp_model_io_t out_sum_fc2 = ccv_cnnp_model_apply(ccv_cnnp_sum("sum_fc2"), MODEL_IO_LIST(out_fc2, out_up_fc2));
	ccv_cnnp_model_io_t out_gelu_fc2 = ccv_cnnp_model_apply(ccv_cnnp_gelu(0, "gelu_fc2"), MODEL_IO_LIST(out_sum_fc2));
	ccv_cnnp_model_io_t out_mul_fc12 = ccv_cnnp_model_apply(ccv_cnnp_mul(1, "mul_fc12"), MODEL_IO_LIST(out_sum_fc1, out_gelu_fc2));
	ccv_cnnp_model_io_t out_fc3 = ccv_cnnp_model_apply(fc3, MODEL_IO_LIST(out_mul_fc12));
	ccv_cnnp_model_io_t out_down_fc3 = ccv_cnnp_model_apply(down_fc3, MODEL_IO_LIST(out_mul_fc12));
	ccv_cnnp_model_io_t out_up_fc3 = ccv_cnnp_model_apply(up_fc3, MODEL_IO_LIST(out_down_fc3));
	ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_sum("sum_fc3"), MODEL_IO_LIST(out_fc3, out_up_fc3));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(out), 0, "tiny");
	ccv_cnnp_model_set_gradient_checkpointing(final, 1);
	ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const tlinear = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	int i;
	for (i = 0; i < 10 * 10; i++)
		tlinear->data.f32[i] = (i / 10 == i % 10) ? 1 : 0;
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2), 0);
	for (i = 0; i < 10 * 2; i++)
		t->data.f32[i] = 0;
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NHWC(32F, 10);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(input_params), CMD_SGD_FORWARD(1, 0.001, 1, 0.1, 0, 0), CMD_MSE_FORWARD(CCV_NNC_MSE_REDUCE_MEAN));
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(fc1, ALL_PARAMETERS, 0), tlinear);
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(fc2, ALL_PARAMETERS, 0), tlinear);
	ccv_nnc_tensor_free(tlinear);
	ccv_nnc_tensor_t* const tlinear_fc3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 10), 0);
	for (i = 0; i < 5 * 10; i++)
		tlinear_fc3->data.f32[i] = (i / 10 == i % 10) ? 1 : 0;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(fc3, ALL_PARAMETERS, 0), tlinear_fc3);
	ccv_nnc_tensor_free(tlinear_fc3);
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(up_fc1, ALL_PARAMETERS, 0), t);
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(up_fc2, ALL_PARAMETERS, 0), t);
	ccv_nnc_tensor_free(t);
	ccv_nnc_tensor_t* const t_fc3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 2), 0);
	for (i = 0; i < 5 * 2; i++)
		t_fc3->data.f32[i] = 0;
	ccv_cnnp_model_set_parameter(final, ccv_cnnp_model_parameters(up_fc3, ALL_PARAMETERS, 0), t_fc3);
	ccv_nnc_tensor_free(t_fc3);
	for (i = 0; i < 10; i++)
		x->data.f32[i] = i;
	ccv_nnc_tensor_t* const target = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	for (i = 0; i < 5; i++)
		target->data.f32[i] = 5 - i;
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	ccv_cnnp_model_fit(final, TENSOR_LIST(x), TENSOR_LIST(target), TENSOR_LIST(y), 0, 0);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y->data.f32, target->data.f32, 5, 1e-1, "should match the target after fine-tuning");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(final), 0, "should be marked as not trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down_fc1), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up_fc1), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down_fc2), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up_fc2), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(down_fc3), 1, "should be marked as trainable");
	REQUIRE_EQ(ccv_cnnp_model_is_trainable(up_fc3), 1, "should be marked as trainable");
	ccv_nnc_tensor_free(x);
	ccv_nnc_tensor_free(target);
	ccv_nnc_tensor_free(y);
	ccv_cnnp_model_free(final);
}

#include "case_main.h"
