#include "ccv_nnc_mfa.hpp"
#include "kernels/Dequantize8iRowwiseXDescriptor.hpp"
#include "kernels/Dequantize8iRowwiseXKernel.hpp"

using namespace ccv::nnc;

namespace {

static uint32_t scale_size_for_data_type(const uint64_t data_type) noexcept
{
	switch (data_type) {
		case MTL::DataTypeHalf:
		case MTL::DataTypeBFloat:
			return 2;
		case MTL::DataTypeFloat:
			return 4;
		default:
			CCV_NNC_MFA_PRECONDITION(false);
			return 0;
	}
}

}

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(mfa::context* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params)
{
	(void)context;
	(void)params;
}

void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr) {
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors += 1;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

	Dequantize8iRowwiseXDescriptor descriptor;
	descriptor.format = params.format;
	descriptor.scaleSize = scale_size_for_data_type(params.data_type);
	descriptor.rowLength = (uint32_t)params.row_length;
	descriptor.length = (uint32_t)params.length;

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<Dequantize8iRowwiseXKernel, Dequantize8iRowwiseXDescriptor, Dequantize8iRowwiseXKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	encoder->setComputePipelineState(pipeline.get());
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageWrite);

	MTL::Size gridSize = kernel->gridSize(descriptor.dispatchItems());
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
	command_batch->finishCommand(encoder);
}
