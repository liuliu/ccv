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

TEST_CASE("compile simple cifar-10 model")
{
	ccv_cnnp_model_t* const sequential = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_flatten(),
		ccv_cnnp_dense(256, (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
		}),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		})
	));
	const ccv_nnc_tensor_param_t input = CPU_TENSOR_NHWC(1, 31, 31, 3);
	ccv_cnnp_model_compile(sequential, &input, 1, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	ccv_nnc_tensor_t* const input_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(1, 31, 31, 3), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 31 * 31 * 3; i++)
		input_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const output_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(1, 10), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// Before training, it doesn't fit.
	ccv_cnnp_model_evaluate(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	int t = 0;
	float max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	const int target = (t + 1) % 10;
	REQUIRE_NOT_EQ(target, t, "should not fit");
	// Doing training.
	ccv_nnc_tensor_t* const fit_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(1), 0);
	fit_tensor->data.f32[0] = target;
	for (i = 0; i < 100; i++)
		ccv_cnnp_model_fit(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(fit_tensor), TENSOR_LIST(output_tensor), 0);
	memset(output_tensor->data.f32, 0, sizeof(float) * 10);
	// After training, it should fit.
	ccv_cnnp_model_evaluate(sequential, TENSOR_LIST(input_tensor), TENSOR_LIST(output_tensor), 0);
	t = 0;
	max = output_tensor->data.f32[0];
	for (i = 1; i < 10; i++)
		if (output_tensor->data.f32[i] > max)
			max = output_tensor->data.f32[i], t = i;
	REQUIRE_EQ(target, t, "should fit");
	CNNP_MODEL_GEN(sequential, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(sequential);
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
	}), MODEL_IO_LIST(x));
	tower_1 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	}), MODEL_IO_LIST(tower_1));

	ccv_cnnp_model_io_t tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}), MODEL_IO_LIST(x));
	tower_2 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (2, 2)),
	}), MODEL_IO_LIST(tower_2));

	ccv_cnnp_model_io_t tower_3 = ccv_cnnp_model_apply(ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.hint = HINT((1, 1), (1, 1)),
	}), MODEL_IO_LIST(x));
	tower_3 = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, 64, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (0, 0)),
	}), MODEL_IO_LIST(tower_3));

	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_add(), MODEL_IO_LIST(tower_1, tower_2, tower_3));

	ccv_cnnp_model_t* inception = ccv_cnnp_model_new(MODEL_IO_LIST(x), MODEL_IO_LIST(output));
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 1, 3, 256, 256);
	ccv_cnnp_model_compile(inception, &input, 1, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	CNNP_MODEL_GEN(inception, CCV_NNC_LONG_DOT_GRAPH);
	ccv_cnnp_model_free(inception);
}

#include "case_main.h"
