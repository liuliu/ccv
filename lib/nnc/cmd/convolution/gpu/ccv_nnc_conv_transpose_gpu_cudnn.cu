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
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_0, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_1, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT_TILING, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD_NONFUSED, // CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
	CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT
};

static int _ccv_nnc_conv_transpose_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const int is_w_nhwc = inputs[1]->info.format == CCV_TENSOR_FORMAT_NHWC;
	const int w_datatype = inputs[1]->info.datatype;
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, cmd.info, hint, (is_w_nhwc && w_datatype == CCV_16F) ? CCV_32F : w_datatype);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution_transpose.groups);

	cudnnConvolutionBwdDataAlgo_t data_algo;
	const int data_algorithm = cmd.algorithm < 0 ? -1 : cmd.algorithm;
	switch (data_algorithm)
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
#if CUDNN_VERSION >= 7000
			int data_algo_count;
			cudnnConvolutionBwdDataAlgoPerf_t data_perf;
			CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn, w.descriptor, a.descriptor, conv.descriptor, b.descriptor, 1, &data_algo_count, &data_perf));
			assert(data_algo_count > 0);
			data_algo = data_perf.algo;
#else
			CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, w.descriptor, a.descriptor, conv.descriptor, b.descriptor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &data_algo));
#endif
	}
	size_t workspace_size = 0;
	CUDNN_ENFORCE(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, w.descriptor, a.descriptor, conv.descriptor, b.descriptor, data_algo, &workspace_size));
	void* workspace = 0;
	void* weight_data = w.data.u8;
	if (CCV_GET_DATA_TYPE(inputs[2]->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t weight_params = inputs[2]->info;
		const size_t count = ccv_nnc_tensor_count(weight_params);
		const int palette_datatype = (weight_params.datatype & 0xff) << 12;
		const int qbits = (weight_params.datatype & 0xf00) >> 8;
		const int number_in_blocks = weight_params.reserved;
		ccv_nnc_tensor_param_t depalettize_weight_params = weight_params;
		depalettize_weight_params.datatype = palette_datatype;
		depalettize_weight_params.reserved = 0;
		const size_t data_size = ccv_nnc_tensor_data_size(depalettize_weight_params);
		workspace_size = ((ssize_t)workspace_size + 1023) & -1024; // Somehow the workspace size is not padded. We need to pad it for weight_data to be aligned.
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size + data_size, CCV_TENSOR_GPU_MEMORY);
		weight_data = (uint8_t*)workspace + workspace_size;
		ccv_nnc_compat_depalettize(w.data.u8, palette_datatype, ccv_nnc_tensor_data_size_without_padding(weight_params), qbits, number_in_blocks, weight_data, count, stream_context);
		if (workspace_size == 0)
			workspace = 0;
	} else {
		// TODO: If error, return OOM
		if (workspace_size)
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, workspace_size, CCV_TENSOR_GPU_MEMORY);
	}
	static const float one = 1, zero = 0;
	CUDNN_ENFORCE(cudnnConvolutionBackwardData(cudnn, &one, w.descriptor, weight_data, a.descriptor, a.data.u8, conv.descriptor, data_algo, workspace, workspace_size, &zero, b.descriptor, b.data.u8));
	if (input_size > 2 && inputs[2])
	{
		const ccv_nnc_cudnn_tensor_view_descriptor_t bias = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[2]);
		CUDNN_ENFORCE(cudnnAddTensor(cudnn, &one, bias.descriptor, bias.data.u8, &one, b.descriptor, b.data.u8));
		ccv_nnc_cudnn_deinit_tensor_view_descriptor(bias);
	}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_filter_descriptor(w);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_transpose_forw_autotune(const ccv_nnc_cmd_t cmd, size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	void* workmem = ccv_nnc_stream_context_get_workspace(stream_context, max_workspace_size, CCV_TENSOR_GPU_MEMORY);
	if (max_workspace_size && !workmem)
		return -1;
	const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
	const ccv_nnc_cudnn_filter_descriptor_t w = ccv_nnc_cudnn_get_filter_descriptor(stream_context, (const ccv_nnc_tensor_t*)inputs[1]);
	const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
	const int is_w_nhwc = inputs[1]->info.format == CCV_TENSOR_FORMAT_NHWC;
	const int w_datatype = inputs[1]->info.datatype;
	const ccv_nnc_cudnn_convolution_descriptor_t conv = ccv_nnc_cudnn_get_convolution_descriptor(stream_context, cmd.info, hint, (is_w_nhwc && w_datatype == CCV_16F) ? CCV_32F : w_datatype);
	cudnnSetConvolutionGroupCount(conv.descriptor, cmd.info.convolution.groups);
	int count = 0;
	cudnnConvolutionBwdDataAlgoPerf_t data_perfs[CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT];
	void* weight_data = w.data.u8;
	if (CCV_GET_DATA_TYPE(inputs[1]->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t weight_params = inputs[1]->info;
		const int palette_datatype = (weight_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_weight_params = weight_params;
		depalettize_weight_params.datatype = palette_datatype;
		depalettize_weight_params.reserved = 0;
		const size_t data_size = ccv_nnc_tensor_data_size(depalettize_weight_params);
		max_workspace_size = ((ssize_t)max_workspace_size + 1023) & -1024; // Somehow the workspace size is not padded. We need to pad it for weight_data to be aligned.
		workmem = ccv_nnc_stream_context_get_workspace(stream_context, max_workspace_size + data_size, CCV_TENSOR_GPU_MEMORY);
		weight_data = (uint8_t*)workmem + max_workspace_size;
		if (max_workspace_size == 0)
			workmem = 0;
	}
	CUDNN_ENFORCE(cudnnFindConvolutionBackwardDataAlgorithmEx(cudnn, w.descriptor, weight_data, a.descriptor, a.data.u8, conv.descriptor, b.descriptor, b.data.u8, CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT, &count, data_perfs, workmem, max_workspace_size));
	int i;
	cudnnConvolutionBwdDataAlgo_t data_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
	for(i = 0; i < count; i++)
		if ((size_t)data_perfs[i].memory <= max_workspace_size && data_perfs[i].status == CUDNN_STATUS_SUCCESS)
		{
			data_algorithm = data_perfs[i].algo;
			break;
		}
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(a);
	ccv_nnc_cudnn_deinit_filter_descriptor(w);
	ccv_nnc_cudnn_deinit_tensor_view_descriptor(b);
	ccv_nnc_cudnn_deinit_convolution_descriptor(conv);
	switch (data_algorithm)
	{
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_0;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_1;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_FFT_TILING;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
			return CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
		case CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT:
			break;
	}
	return -1; // Return the most efficient algorithm, return -1 if cannot find one.
}

static int _ccv_nnc_conv_transpose_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = CCV_NNC_CMD_CUDNN_CONV_BWD_DATA_ALGO_COUNT;
	registry->exec = _ccv_nnc_conv_transpose_forw;
	registry->autotune = _ccv_nnc_conv_transpose_forw_autotune;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->exec = _ccv_nnc_conv_transpose_back;
#endif
}
