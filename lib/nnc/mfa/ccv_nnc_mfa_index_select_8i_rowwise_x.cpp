#include "ccv_nnc_mfa.hpp"
#include "kernels/IndexSelect8iRowwiseXDescriptor.hpp"
#include "kernels/IndexSelect8iRowwiseXKernel.hpp"

using namespace ccv::nnc;

namespace {

static GEMMOperandPrecision memory_precision_for_data_type(const uint64_t data_type) noexcept
{
	switch (data_type) {
		case MTL::DataTypeHalf:
			return GEMMOperandPrecision::FP16;
		case MTL::DataTypeBFloat:
			return GEMMOperandPrecision::BF16;
		case MTL::DataTypeFloat:
			return GEMMOperandPrecision::FP32;
		default:
			CCV_NNC_MFA_PRECONDITION(false);
			return GEMMOperandPrecision::FP32;
	}
}

}

void ccv_nnc_mfa_prepare_index_select_8i_rowwise_x(mfa::context* context, ccv_nnc_mfa_index_select_8i_rowwise_x_params_t params)
{
	(void)context;
	(void)params;
}

void ccv_nnc_mfa_encode_index_select_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_index_select_8i_rowwise_x_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr) {
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors += 1;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 3);

	IndexSelect8iRowwiseXDescriptor descriptor;
	descriptor.format = params.format;
	descriptor.memoryPrecision = memory_precision_for_data_type(params.data_type);
	CCV_NNC_MFA_PRECONDITION(params.row_length > 0);
	CCV_NNC_MFA_PRECONDITION(params.input_length <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.output_length <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.input_length % params.row_length == 0);
	CCV_NNC_MFA_PRECONDITION(params.output_length % params.row_length == 0);
	descriptor.rowLength = (uint32_t)params.row_length;
	descriptor.inputLength = (uint32_t)params.input_length;
	descriptor.outputLength = (uint32_t)params.output_length;

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<IndexSelect8iRowwiseXKernel, IndexSelect8iRowwiseXDescriptor, IndexSelect8iRowwiseXKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	encoder->setComputePipelineState(pipeline.get());
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageWrite);

	MTL::Size gridSize = kernel->gridSize(descriptor.outputGroups());
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
	command_batch->finishCommand(encoder);
}
