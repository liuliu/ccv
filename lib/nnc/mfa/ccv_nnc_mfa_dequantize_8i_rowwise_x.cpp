#include "ccv_nnc_mfa.hpp"
#include "kernels/Dequantize8iRowwiseXDescriptor.hpp"
#include "kernels/Dequantize8iRowwiseXKernel.hpp"

using namespace ccv::nnc;

namespace {

typedef struct {
	size_t active_experts_offset;
	size_t dispatch_offset;
	size_t scratch_bytes;
} ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_scratch_layout_t;

static size_t align_up(const size_t value, const size_t alignment) noexcept
{
	return (value + alignment - 1) & ~(alignment - 1);
}

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

static ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_scratch_layout_t selected_scratch_layout(ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params) noexcept
{
	const size_t active_experts = params.segment_count > 0 ? (size_t)params.segment_count : 1;
	const size_t active_experts_bytes = align_up(active_experts * sizeof(uint32_t), 256);
	const size_t dispatch_offset = active_experts_bytes;
	return (ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_scratch_layout_t){
		.active_experts_offset = 0,
		.dispatch_offset = dispatch_offset,
		.scratch_bytes = align_up(dispatch_offset + 4 * sizeof(uint32_t), 256),
	};
}

}

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(mfa::context* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params)
{
	(void)context;
	(void)params;
}

size_t ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params)
{
	return selected_scratch_layout(params).scratch_bytes;
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
	CCV_NNC_MFA_PRECONDITION(params.row_length <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.length <= UINT32_MAX);
	const uint64_t row_count = params.row_length > 0 ? params.length / params.row_length : 0;
	CCV_NNC_MFA_PRECONDITION(row_count <= UINT32_MAX / descriptor.scaleSize);
	descriptor.rowLength = (uint32_t)params.row_length;
	descriptor.length = (uint32_t)params.length;
	encoder->setBuffer(tensors[0], tensor_offsets[0] + (size_t)descriptor.inputScaleOffset(), NS::UInteger(2));
	encoder->setBuffer(tensors[1], tensor_offsets[1] + (size_t)descriptor.outputScaleOffset(), NS::UInteger(3));

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

void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
		num_tensors += 1;
	CCV_NNC_MFA_PRECONDITION(num_tensors == 5);

	Dequantize8iRowwiseXSelectedDescriptor descriptor;
	descriptor.format = params.format;
	descriptor.scaleSize = scale_size_for_data_type(params.data_type);
	CCV_NNC_MFA_PRECONDITION(params.row_length <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.rows_per_expert <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.expert_count <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.segment_count <= UINT32_MAX);
	CCV_NNC_MFA_PRECONDITION(params.rows_per_expert == 0 || params.expert_count <= UINT32_MAX / params.rows_per_expert);
	const uint64_t row_count = params.expert_count * params.rows_per_expert;
	CCV_NNC_MFA_PRECONDITION(params.row_length == 0 || row_count <= UINT32_MAX / params.row_length);
	CCV_NNC_MFA_PRECONDITION(row_count <= UINT32_MAX / descriptor.scaleSize);
	descriptor.rowLength = (uint32_t)params.row_length;
	descriptor.rowsPerExpert = (uint32_t)params.rows_per_expert;
	descriptor.expertCount = (uint32_t)params.expert_count;
	descriptor.segmentCount = (uint32_t)params.segment_count;

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<Dequantize8iRowwiseXKernel, Dequantize8iRowwiseXSelectedDescriptor, Dequantize8iRowwiseXKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;
	auto planPipeline = pipelineValue->second;
	const ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_scratch_layout_t scratch_layout = selected_scratch_layout(params);

	encoder->setComputePipelineState(planPipeline.get());
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
	encoder->setBuffer(tensors[1], tensor_offsets[1], NS::UInteger(0));
	encoder->setBuffer(tensors[2], tensor_offsets[2], NS::UInteger(1));
	encoder->setBuffer(tensors[4], tensor_offsets[4] + scratch_layout.active_experts_offset, NS::UInteger(2));
	encoder->setBuffer(tensors[4], tensor_offsets[4] + scratch_layout.dispatch_offset, NS::UInteger(3));
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), kernel->threadgroupSize);
	command_batch->finishCommand(encoder);

	encoder = command_batch->startCommand();
	encoder->setComputePipelineState(pipeline.get());
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
	encoder->useResource(tensors[4], MTL::ResourceUsageRead);
	encoder->setBuffer(tensors[0], tensor_offsets[0], NS::UInteger(0));
	encoder->setBuffer(tensors[4], tensor_offsets[4] + scratch_layout.active_experts_offset, NS::UInteger(1));
	encoder->setBuffer(tensors[3], tensor_offsets[3], NS::UInteger(2));
	encoder->setBuffer(tensors[0], tensor_offsets[0] + (size_t)descriptor.inputScaleOffset(), NS::UInteger(3));
	encoder->setBuffer(tensors[3], tensor_offsets[3] + (size_t)descriptor.outputScaleOffset(), NS::UInteger(4));

	encoder->dispatchThreadgroups(tensors[4], tensor_offsets[4] + scratch_layout.dispatch_offset, kernel->threadgroupSize);
	command_batch->finishCommand(encoder);
}
