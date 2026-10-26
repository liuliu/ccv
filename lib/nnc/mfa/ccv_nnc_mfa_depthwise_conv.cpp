#include "ccv_nnc_mfa.hpp"
#include "kernels/DepthwiseConvDescriptor.hpp"
#include "kernels/DepthwiseConvKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_encode_depthwise_conv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depthwise_conv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.batch_size > 0 && params.channels > 0);
	CCV_NNC_MFA_PRECONDITION(params.output_dimensions[0] > 0 && params.output_dimensions[1] > 0);
	CCV_NNC_MFA_PRECONDITION(params.filter_dimensions[0] > 0 && params.filter_dimensions[1] > 0);
	CCV_NNC_MFA_PRECONDITION(params.stride_dimensions[0] > 0 && params.stride_dimensions[1] > 0);
	CCV_NNC_MFA_PRECONDITION(params.dilation_dimensions[0] > 0 && params.dilation_dimensions[1] > 0);
	const int tensor_count = params.fused_bias ? 4 : 3;
	int i;
	for (i = 0; i < tensor_count; i++)
		CCV_NNC_MFA_PRECONDITION(tensors[i] != nullptr);

	DepthwiseConvDescriptor descriptor;
	if (params.data_type == MTL::DataTypeHalf) {
		descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
	} else if (params.data_type == MTL::DataTypeBFloat) {
		descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
	} else {
		CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat);
		descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
	}
	const size_t dataTypeSize = params.data_type == MTL::DataTypeFloat ? 4 : 2;
	// Four channels per thread when every channel row starts on a vector boundary.
	descriptor.vectorized = (params.channels % 4 == 0 &&
		tensor_offsets[0] % (dataTypeSize * 4) == 0 &&
		tensor_offsets[2] % (dataTypeSize * 4) == 0) ? 1 : 0;
	descriptor.fusedBias = params.fused_bias ? 1 : 0;

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<DepthwiseConvKernel, DepthwiseConvDescriptor, DepthwiseConvKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	auto encoder = command_batch->startCommand();
	encoder->setComputePipelineState(pipeline.get());
	for (i = 0; i < tensor_count; i++)
		encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
	if (params.fused_bias)
		encoder->useResource(tensors[3], MTL::ResourceUsageRead);

	// Geometry is a runtime argument, so new spatial sizes (e.g. prefill lengths) reuse one pipeline.
	const uint32_t channelUnits = descriptor.vectorized ? params.channels / 4 : params.channels;
	const uint64_t elementCount = (uint64_t)params.batch_size * params.output_dimensions[0] * params.output_dimensions[1] * channelUnits;
	CCV_NNC_MFA_PRECONDITION(elementCount <= UINT32_MAX);
	const DepthwiseConvArguments arguments = {
		.channel_units = channelUnits,
		.input_height = params.input_dimensions[0],
		.input_width = params.input_dimensions[1],
		.output_height = params.output_dimensions[0],
		.output_width = params.output_dimensions[1],
		.kernel_height = params.filter_dimensions[0],
		.kernel_width = params.filter_dimensions[1],
		.stride_height = params.stride_dimensions[0],
		.stride_width = params.stride_dimensions[1],
		.dilation_height = params.dilation_dimensions[0],
		.dilation_width = params.dilation_dimensions[1],
		.padding_top = params.padding_top,
		.padding_left = params.padding_left,
		.element_count = (uint32_t)elementCount,
	};
	encoder->setBytes(&arguments, sizeof(arguments), 4);

	const MTL::Size gridSize = kernel->gridSize((uint32_t)elementCount);
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

	command_batch->finishCommand(encoder);
}
