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

#define checkCUDNN(status) {                                      \
	if (status != CUDNN_STATUS_SUCCESS) {                         \
		printf("%s:%d\nCUDNN failure\nError: %s\n",               \
				__FILE__, __LINE__, cudnnGetErrorString(status)); \
		cudaDeviceReset();                                        \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
}

static void _ccv_nnc_set_tensor_nd_desc(const ccv_nnc_tensor_view_t* tensor, cudnnTensorDescriptor_t desc)
{
	// N is the outer one nevertheless.
	assert(tensor->info.format == CCV_TENSOR_FORMAT_NCHW || tensor->info.format == CCV_TENSOR_FORMAT_NHWC);
	// Fill up dimensions with 1s.
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int i;
	const int nd = CCV_NNC_MAX_DIM + 2;
	// This has to follow NCHW
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
		for (i = 0; i < nd; i++)
			dim[i] = ccv_max(1, tensor->info.dim[nd - 1 - i]);
	else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = ccv_max(1, tensor->info.dim[nd - 1]);
		dim[1] = ccv_max(1, tensor->info.dim[0]);
		for (i = 2; i < nd; i++)
			dim[i] = ccv_max(1, tensor->info.dim[nd - i]);
	}
	printf("tensor %d %d %d %d\n", dim[0], dim[1], dim[2], dim[3]);
	const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int stride[CCV_NNC_MAX_DIM_ALLOC];
	stride[nd - 1] = 1;
	// Compute the stride from inc, so it will fit the tensor view.
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && dim[i] > 0; i++)
			stride[nd - 1 - i] = stride[nd - i] * ccv_max(1, inc[i - 1]);
	else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		stride[1] = 1;
		stride[nd - 1] = ccv_max(1, inc[0]);
		for (i = 0; i < nd - 3; i++)
			stride[nd - 2 - i] = stride[nd - 1 - i] * ccv_max(1, inc[i + 1]);
		stride[0] = stride[2] * ccv_max(1, inc[nd - 2]);
	}
	printf("tensor stride %d %d %d %d\n", stride[0], stride[1], stride[2], stride[3]);
	cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, nd, dim, stride);
}

static void _ccv_nnc_set_filter_nd_desc(const ccv_nnc_tensor_t* tensor, cudnnFilterDescriptor_t desc)
{
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	int nd = ccv_nnc_tensor_nd(tensor->info);
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int i;
	// Reorder since nnc have different idea about NCHW and NHWC (N is in 3, C is in 0).
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		for (i = 0; i < nd; i++)
			dim[i] = tensor->info.dim[nd - 1 - i];
		printf("filter %d %d %d %d\n", dim[0], dim[1], dim[2], dim[3]);
		cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nd, dim);
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = tensor->info.dim[nd - 1];
		dim[1] = tensor->info.dim[0];
		for (i = 2; i < nd; i++)
			dim[i] = tensor->info.dim[nd - i];
		printf("filter %d %d %d %d\n", dim[0], dim[1], dim[2], dim[3]);
		cudnnSetFilterNdDescriptor(desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nd, dim);
	}
}

enum {
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_DIRECT, // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT, // CUDNN_CONVOLUTION_FWD_ALGO_FFT
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_TILING, // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_WINOGRAD, // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT
};

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	assert(stream_context);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	cudnnTensorDescriptor_t a_desc, bias_desc, b_desc;
	cudnnFilterDescriptor_t w_desc;
	cudnnCreateTensorDescriptor(&a_desc);
	const ccv_nnc_tensor_t* a = inputs[0];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)a, a_desc);
	cudnnCreateFilterDescriptor(&w_desc);
	const ccv_nnc_tensor_t* w = inputs[1];
	_ccv_nnc_set_filter_nd_desc(w, w_desc);
	cudnnCreateTensorDescriptor(&bias_desc);
	const ccv_nnc_tensor_t* bias = inputs[2];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)bias, bias_desc);
	cudnnCreateTensorDescriptor(&b_desc);
	ccv_nnc_tensor_t* b = outputs[0];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)b, b_desc);
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnCreateConvolutionDescriptor(&conv_desc);
	int i;
	int p[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		p[i] = ccv_max(hint.border.begin[CCV_NNC_MAX_DIM - i], hint.border.end[CCV_NNC_MAX_DIM - i]);
	int v[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		v[i] = hint.stride.dim[CCV_NNC_MAX_DIM - i];
	int u[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		u[i] = 1;
	cudnnSetConvolutionNdDescriptor(conv_desc, CCV_NNC_MAX_DIM, p, v, u, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	cudnnConvolutionFwdAlgo_t algo;
	// Choose an algorithm
	switch (cmd.algorithm)
	{
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_GEMM:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_GEMM:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_DIRECT:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_TILING:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_WINOGRAD:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
			break;
		default: // -1: Using preferences to find a suitable algorithm
			cudnnGetConvolutionForwardAlgorithm(cudnn, a_desc, w_desc, conv_desc, b_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
	}

	size_t workspace_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnn, a_desc, w_desc, conv_desc, b_desc, algo, &workspace_size);
	void* workspace = 0;
	// TODO: If error, return OOM
	if (workspace_size)
		cudaMalloc(&workspace, workspace_size);
	float one = 1, zero = 0;
	cudnnConvolutionForward(cudnn, &one, a_desc, a->data.u8, w_desc, w->data.u8, conv_desc, algo, workspace, workspace_size, &zero, b_desc, b->data.u8);
	cudnnAddTensor(cudnn, &one, bias_desc, bias->data.u8, &one, b_desc, b->data.u8);

	cudnnDestroyTensorDescriptor(a_desc);
	cudnnDestroyFilterDescriptor(w_desc);
	cudnnDestroyTensorDescriptor(bias_desc);
	cudnnDestroyTensorDescriptor(b_desc);
	cudnnDestroyConvolutionDescriptor(conv_desc);
	if (workspace)
		cudaFreeAsync(workspace, stream);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_forw_autotune(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	assert(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	cudnnTensorDescriptor_t a_desc, bias_desc, b_desc;
	cudnnFilterDescriptor_t w_desc;
	cudnnCreateTensorDescriptor(&a_desc);
	const ccv_nnc_tensor_t* a = inputs[0];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)a, a_desc);
	cudnnCreateFilterDescriptor(&w_desc);
	const ccv_nnc_tensor_t* w = inputs[1];
	_ccv_nnc_set_filter_nd_desc(w, w_desc);
	cudnnCreateTensorDescriptor(&bias_desc);
	const ccv_nnc_tensor_t* bias = inputs[2];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)bias, bias_desc);
	cudnnCreateTensorDescriptor(&b_desc);
	ccv_nnc_tensor_t* b = outputs[0];
	_ccv_nnc_set_tensor_nd_desc((const ccv_nnc_tensor_view_t*)b, b_desc);
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnCreateConvolutionDescriptor(&conv_desc);
	int i;
	int p[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		p[i] = ccv_max(hint.border.begin[CCV_NNC_MAX_DIM - i], hint.border.end[CCV_NNC_MAX_DIM - i]);
	int v[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		v[i] = hint.stride.dim[CCV_NNC_MAX_DIM - i];
	int u[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		u[i] = 1;
	cudnnSetConvolutionNdDescriptor(conv_desc, CCV_NNC_MAX_DIM, p, v, u, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	int returnedAlgoCount;
	cudnnConvolutionFwdAlgoPerf_t perfResults[CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT];
	cudnnFindConvolutionForwardAlgorithm(cudnn, a_desc, w_desc, conv_desc, b_desc, CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT, &returnedAlgoCount, perfResults);
	for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
		printf("^^^^ %s for Algo %d: %f time requiring %llu memory\n", cudnnGetErrorString(perfResults[algoIndex].status), perfResults[algoIndex].algo, perfResults[algoIndex].time, (unsigned long long)perfResults[algoIndex].memory);
	}

	cudnnDestroyTensorDescriptor(a_desc);
	cudnnDestroyFilterDescriptor(w_desc);
	cudnnDestroyTensorDescriptor(bias_desc);
	cudnnDestroyTensorDescriptor(b_desc);
	cudnnDestroyConvolutionDescriptor(conv_desc);
	return -1; // Return the most efficient algorithm, return -1 if cannot find one.
}
#endif

//@ccv_nnc_init CCV_NNC_BACKEND_GPU_CUDNN
void ccv_nnc_gpu_cudnn_init(ccv_nnc_cmd_api_t cmd_api[])
{
#ifdef HAVE_CUDNN
	/* Convolutional layer */
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].tensor_memory = CCV_TENSOR_GPU_MEMORY;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].algorithms = CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].exec = _ccv_nnc_conv_forw;
	cmd_api[CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD].autotune = _ccv_nnc_conv_forw_autotune;
	/* Full connect layer */
	/* Max pool layer */
	/* Average pool layer */
	/* Softmax layer */
	/* ReLU activation */
#endif
}
