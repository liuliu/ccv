#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("compile simple CIFAR-10 model")
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
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(5, 5), (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (2, 2)),
		}),
		ccv_cnnp_average_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (0, 0)),
		}),
		ccv_cnnp_flatten(),
		ccv_cnnp_dense(10, (ccv_cnnp_param_t){
			.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
		})
	));
	const ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 128, 3, 31, 31);
	ccv_cnnp_model_compile(sequential, &input, 1);
	ccv_cnnp_model_free(sequential);
}

#include "case_main.h"
