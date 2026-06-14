#include "ccv_nnc_mfa.hpp"
#include "kernels/Dequantize8iRowwiseDescriptor.hpp"
#include "kernels/Dequantize8iRowwiseKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise(mfa::context* context, ccv_nnc_mfa_dequantize_8i_rowwise_params_t params)
{
	(void)context;
	(void)params;
}

void ccv_nnc_mfa_encode_dequantize_8i_rowwise(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr) {
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors += 1;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

	Dequantize8iRowwiseDescriptor descriptor;
	if (params.data_type == MTL::DataTypeHalf) {
		descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
	} else if (params.data_type == MTL::DataTypeBFloat) {
		descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
	} else {
		CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat);
		descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
	}
	descriptor.rowLength = (uint32_t)params.row_length;
	descriptor.length = (uint32_t)params.length;
	const size_t scale_buffer_offset = ((size_t)params.length + 127) & ~(size_t)127;
	encoder->setBuffer(tensors[0], tensor_offsets[0] + scale_buffer_offset, NS::UInteger(2));

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<Dequantize8iRowwiseKernel, Dequantize8iRowwiseDescriptor, Dequantize8iRowwiseKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	encoder->setComputePipelineState(pipeline.get());
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageWrite);

	MTL::Size gridSize = kernel->gridSize(descriptor.length);
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
	command_batch->finishCommand(encoder);
}
