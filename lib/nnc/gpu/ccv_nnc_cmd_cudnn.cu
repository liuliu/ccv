#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>

// The header for CUDA related objects.
#include "ccv_nnc_cmd.h"

// nvcc is a C++ compiler, need to specify this is a "C" function to avoid name mangling.
extern "C" void ccv_nnc_gpu_cudnn_init(ccv_nnc_cmd_api_t cmd_api[]);

#ifdef HAVE_CUDNN
static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	assert(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudnnTensorDescriptor_t a_desc, bias_desc, b_desc;
	cudnnFilterDescriptor_t w_desc;
	cudnnCreateTensorDescriptor(&a_desc);
	cudnnCreateFilterDescriptor(&w_desc);
	cudnnCreateTensorDescriptor(&bias_desc);
	cudnnCreateTensorDescriptor(&b_desc);

	cudnnDestroyTensorDescriptor(a_desc);
	cudnnDestroyFilterDescriptor(w_desc);
	cudnnDestroyTensorDescriptor(bias_desc);
	cudnnDestroyTensorDescriptor(b_desc);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_forw_autotune(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	return 0; // Return the most efficient algorithm, return -1 if cannot find one.
}
#endif

//@ccv_nnc_init CCV_NNC_BACKEND_GPU_CUDNN
void ccv_nnc_gpu_cudnn_init(ccv_nnc_cmd_api_t cmd_api[])
{
#ifdef HAVE_CUDNN
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_memory = CCV_TENSOR_GPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].algorithms = 0;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].autotune = _ccv_nnc_conv_forw_autotune;
	/* Full connect layer */
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
#endif
}
