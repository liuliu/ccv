extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

static int _ccv_nnc_batch_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	cudnnBatchNormMode_t mode;
	if (cmd.info.bnorm.count == 1)
	{
		mode = CUDNN_BATCHNORM_PER_ACTIVATION;
		assert((inputs[0]->info.format == CCV_TENSOR_FORMAT_NCHW || inputs[0]->info.format == CCV_TENSOR_FORMAT_NHWC) && cmd.info.bnorm.axis[0] == 0);
	} else {
		mode = CUDNN_BATCHNORM_SPATIAL;
		assert(cmd.info.bnorm.count == 3);
		assert(CCV_NNC_MAX_DIM == 2);
	}
	const float one = 1, zero = 0;
	if (!cmd.info.bnorm.is_test)
	{
		assert(output_size == 5);
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
		assert(!CCV_IS_TENSOR_VIEW(inputs[1]));
		ccv_nnc_tensor_t* const bias = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(bias));
		ccv_nnc_tensor_t* const mean = inputs[3];
		assert(!CCV_IS_TENSOR_VIEW(mean));
		ccv_nnc_tensor_t* const var = inputs[4];
		assert(!CCV_IS_TENSOR_VIEW(var));
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		ccv_nnc_tensor_t* const saved_mean = outputs[3];
		assert(!CCV_IS_TENSOR_VIEW(saved_mean));
		ccv_nnc_tensor_t* const saved_inv_std = outputs[4];
		assert(!CCV_IS_TENSOR_VIEW(saved_inv_std));
		assert_cudnn(cudnnBatchNormalizationForwardTraining(cudnn, mode, &one, &zero, a.descriptor, a.data.u8, b.descriptor, b.data.u8, scale.descriptor, scale.data.u8, bias->data.u8, cmd.info.bnorm.momentum, mean->data.u8, var->data.u8, cmd.info.bnorm.epsilon, saved_mean->data.u8, saved_inv_std->data.u8));
	} else {
		assert(output_size == 1);
		const ccv_nnc_cudnn_tensor_view_descriptor_t a = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[0]);
		const ccv_nnc_cudnn_tensor_view_descriptor_t scale = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)inputs[1]);
		assert(!CCV_IS_TENSOR_VIEW(inputs[1]));
		ccv_nnc_tensor_t* const bias = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(bias));
		ccv_nnc_tensor_t* const mean = inputs[3];
		assert(!CCV_IS_TENSOR_VIEW(mean));
		ccv_nnc_tensor_t* const var = inputs[4];
		assert(!CCV_IS_TENSOR_VIEW(var));
		const ccv_nnc_cudnn_tensor_view_descriptor_t b = ccv_nnc_cudnn_get_tensor_view_descriptor(stream_context, (const ccv_nnc_tensor_view_t*)outputs[0]);
		assert_cudnn(cudnnBatchNormalizationForwardInference(cudnn, mode, &one, &zero, a.descriptor, a.data.u8, b.descriptor, b.data.u8, scale.descriptor, scale.data.u8, bias->data.u8, mean->data.u8, var->data.u8, cmd.info.bnorm.epsilon));
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_batch_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_BATCH_NORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_batch_norm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_BATCH_NORM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_batch_norm_back;
#endif
}
