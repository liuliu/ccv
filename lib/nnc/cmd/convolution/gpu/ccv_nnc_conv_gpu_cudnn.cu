extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

enum {
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_GEMM, // CUDNN_CONVOLUTION_FWD_ALGO_GEMM
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_DIRECT, // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT, // CUDNN_CONVOLUTION_FWD_ALGO_FFT
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_TILING, // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD, // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD_NONFUSED, // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
	CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT
};

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, hint);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution.groups);

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
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD_NONFUSED:
			algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
			break;
		default: // -1: Using preferences to find a suitable algorithm
			CUDNN_ENFORCE(cudnnGetConvolutionForwardAlgorithm(cudnn, a.descriptor, w.descriptor, conv.descriptor, b.descriptor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
	}

	size_t workspace_size = 0;
	CUDNN_ENFORCE(cudnnGetConvolutionForwardWorkspaceSize(cudnn, a.descriptor, w.descriptor, conv.descriptor, b.descriptor, algo, &workspace_size));
	void* workspace = 0;
	// TODO: If error, return OOM
	if (workspace_size)
		cudaMalloc(&workspace, workspace_size);
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnConvolutionForward(cudnn, &one, a.descriptor, a.data.u8, w.descriptor, w.data.u8, conv.descriptor, algo, workspace, workspace_size, &zero, b.descriptor, b.data.u8));
	if (input_size > 2 && inputs[2])
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
		CUDNN_ENFORCE(cudnnAddTensor(cudnn, &one, bias.descriptor, bias.data.u8, &one, b.descriptor, b.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
	}
	if (workspace)
		cudaFreeAsync(workspace, stream);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_filter_descriptor(w);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_forw_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	void* workmem = 0;
	cudaMalloc(&workmem, max_workspace_size);
	if (!workmem)
		return -1;
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, hint);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution.groups);
	int count = 0;
	cudnnConvolutionFwdAlgoPerf_t perfs[CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT];
	CUDNN_ENFORCE(cudnnFindConvolutionForwardAlgorithmEx(cudnn, a.descriptor, a.data.u8, w.descriptor, w.data.u8, conv.descriptor, b.descriptor, b.data.u8, CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT, &count, perfs, workmem, max_workspace_size));
	cudaFree(workmem);
	int i;
	cudnnConvolutionFwdAlgo_t algorithm;
	for(i = 0; i < count; i++)
		if ((size_t)perfs[i].memory <= max_workspace_size && perfs[i].status == CUDNN_STATUS_SUCCESS)
		{
			algorithm = perfs[i].algo;
			break;
		}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_filter_descriptor(w);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	switch (algorithm)
	{
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_GEMM;
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
		case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_GEMM;
		case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_DIRECT;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_FFT_TILING;
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD;
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
			return CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_WINOGRAD_NONFUSED;
		case CUDNN_CONVOLUTION_FWD_ALGO_COUNT:
			break;
	}
	return -1; // Return the most efficient algorithm, return -1 if cannot find one.
}

enum {
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_0, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_1, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_3, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT_TILING, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD_NONFUSED, // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
	CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT
};

enum {
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_0, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_1, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT_TILING, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD_NONFUSED, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT
};

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert((input_size >= 2 && output_size >= 2));
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_filter_descriptor_t dw = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)outputs[1]);
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, hint);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution.groups);

	cudnnConvolutionBwdFilterAlgo_t filter_algo;
	// Choose an algorithm
	switch (cmd.algorithm % CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT)
	{
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_0:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_1:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_3:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT_TILING:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
			break;
		case CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
			filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
			break;
		default: // -1: Using preferences to find a suitable algorithm
			CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, a.descriptor, g.descriptor, conv.descriptor, dw.descriptor, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &filter_algo));
	}

	size_t workspace_size = 0;
	CUDNN_ENFORCE(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, a.descriptor, g.descriptor, conv.descriptor, dw.descriptor, filter_algo, &workspace_size));
	void* workspace = 0;
	// TODO: If error, return OOM
	if (workspace_size)
		cudaMalloc(&workspace, workspace_size);
	static const float one = 1, zero = 0;
	if ((flags & CCV_NNC_ACCUMULATE_OUTPUT)) // accumulating results to bias and dw
	{
		CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(cudnn, &one, a.descriptor, a.data.u8, g.descriptor, g.data.u8, conv.descriptor, filter_algo, workspace, workspace_size, &one, dw.descriptor, dw.data.u8));
		if (output_size > 2 && outputs[2])
		{
			const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[2]);
			CUDNN_ENFORCE(cudnnConvolutionBackwardBias(cudnn, &one, g.descriptor, g.data.u8, &one, bias.descriptor, bias.data.u8));
			ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
		}
	} else {
		CUDNN_ENFORCE(cudnnConvolutionBackwardFilter(cudnn, &one, a.descriptor, a.data.u8, g.descriptor, g.data.u8, conv.descriptor, filter_algo, workspace, workspace_size, &zero, dw.descriptor, dw.data.u8));
		if (output_size > 2 && outputs[2])
		{
			const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[2]);
			CUDNN_ENFORCE(cudnnConvolutionBackwardBias(cudnn, &one, g.descriptor, g.data.u8, &zero, bias.descriptor, bias.data.u8));
			ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
		}
	}
	if (workspace)
		cudaFreeAsync(workspace, stream);
	// If h is available, therefore, we need to propagate the gradients back
	if (outputs[0])
	{
		assert(input_size >= 3);
		const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[2]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		cudnnConvolutionBwdDataAlgo_t data_algo;
		switch (cmd.algorithm / CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT)
		{
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_0:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
				break;
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_1:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
				break;
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
				break;
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT_TILING:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
				break;
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
				break;
			case CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
				data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
				break;
			default: // -1: Using preferences to find a suitable algorithm
				CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, w.descriptor, g.descriptor, conv.descriptor, h.descriptor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &data_algo));
		}
		size_t workspace_size = 0;
		CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, w.descriptor, g.descriptor, conv.descriptor, h.descriptor, data_algo, &workspace_size));
		void* workspace = 0;
		// TODO: If error, return OOM
		if (workspace_size)
			cudaMalloc(&workspace, workspace_size);
		CUDNN_ENFORCE(cudnnConvolutionBackwardData(cudnn, &one, w.descriptor, w.data.u8, g.descriptor, g.data.u8, conv.descriptor, data_algo, workspace, workspace_size, &zero, h.descriptor, h.data.u8));
		if (workspace)
			cudaFreeAsync(workspace, stream);
		ccv_nnc_cudnn_deinit_filter_descriptor(w);
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_filter_descriptor(dw);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, w
	// outputs:  output gradient, weight updates, bias updates [unused]
	assert(input_size >= 2 && output_size >= 2);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	void* workmem = 0;
	cudaMalloc(&workmem, max_workspace_size);
	if (!workmem)
		return -1;
	const ccv_nnc_cudnn_tensor_view_descriptor_t g = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
	const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[2]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t h = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const ccv_nnc_cudnn_filter_descriptor_t dw = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)outputs[1]);
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, hint);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution.groups);
	int count = 0;
	cudnnConvolutionBwdFilterAlgoPerf_t filter_perfs[CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT];
	CUDNN_ENFORCE(cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnn, a.descriptor, a.data.u8, g.descriptor, g.data.u8, conv.descriptor, dw.descriptor, dw.data.u8, CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT, &count, filter_perfs, workmem, max_workspace_size));
	int i;
	cudnnConvolutionBwdFilterAlgo_t filter_algorithm;
	for(i = 0; i < count; i++)
		if ((size_t)filter_perfs[i].memory <= max_workspace_size && filter_perfs[i].status == CUDNN_STATUS_SUCCESS)
		{
			filter_algorithm = filter_perfs[i].algo;
			break;
		}
	cudnnConvolutionBwdDataAlgoPerf_t data_perfs[CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT];
	CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(cudnn, w.descriptor, w.data.u8, g.descriptor, g.data.u8, conv.descriptor, h.descriptor, h.data.u8, CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT, &count, data_perfs, workmem, max_workspace_size));
	cudnnConvolutionBwdDataAlgo_t data_algorithm;
	for(i = 0; i < count; i++)
		if ((size_t)data_perfs[i].memory <= max_workspace_size && data_perfs[i].status == CUDNN_STATUS_SUCCESS)
		{
			data_algorithm = data_perfs[i].algo;
			break;
		}
	cudaFree(workmem);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(g);
	ccv_nnc_cudnn_deinit_filter_descriptor(w);
	ccv_nnc_cudnn_deinit_filter_descriptor(dw);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(h);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	int filter = -1, data = -1;
	switch (filter_algorithm)
	{
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_0;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_1;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_3;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_FFT_TILING;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
			filter = CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
			break;
		case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT:
			break;
	}
	switch (data_algorithm)
	{
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_0;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_1;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT_TILING;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
			data = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
			break;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
			break;
	}
	return data * CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT + filter;
}
#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_CUDNN_CONV_FWD_ALGO_COUNT;
	registry->exec = _ccv_nnc_conv_forw;
	registry->autotune = _ccv_nnc_conv_forw_autotune;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT * CCV_NNC_CMD_CUDNN_CONV_BWD_FILTER_ALGO_COUNT;
	registry->exec = _ccv_nnc_conv_back;
	registry->autotune = _ccv_nnc_conv_back_autotune;
#endif
}
