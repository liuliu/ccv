#include "ccv_nnc_mfa.hpp"
#include "kernels/IndexSelectDescriptor.hpp"
#include "kernels/IndexSelectKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_index_select(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_index_select_params_t params)
{
	(void)context;
	(void)params;
}

void ccv_nnc_mfa_encode_index_select(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_index_select_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeBFloat || params.data_type == MTL::DataTypeInt);
	CCV_NNC_MFA_PRECONDITION(params.output_rows > 0);
	CCV_NNC_MFA_PRECONDITION(params.row_length > 0);
	CCV_NNC_MFA_PRECONDITION(tensors[0] && tensors[1] && tensors[2] && !tensors[3]);

	const uint32_t element_size = params.data_type == MTL::DataTypeInt ? 4 : 2;
	uint8_t vector_width = 1;
	if (params.row_length % 4 == 0 && tensor_offsets[0] % (element_size * 4) == 0 && tensor_offsets[2] % (element_size * 4) == 0)
		vector_width = 4;
	else if (params.row_length % 2 == 0 && tensor_offsets[0] % (element_size * 2) == 0 && tensor_offsets[2] % (element_size * 2) == 0)
		vector_width = 2;
	const uint32_t row_units = params.row_length / vector_width;
	uint16_t threads_per_row = 1;
	while (threads_per_row < row_units && threads_per_row < 256)
		threads_per_row <<= 1;

	IndexSelectDataType data_type = IndexSelectDataType::INT32;
	if (params.data_type == MTL::DataTypeHalf)
		data_type = IndexSelectDataType::FP16;
	else if (params.data_type == MTL::DataTypeBFloat)
		data_type = IndexSelectDataType::BF16;
	const IndexSelectDescriptor descriptor = {
		.dataType = data_type,
		.vectorWidth = vector_width,
		.threadsPerRow = threads_per_row,
		.outputRows = params.output_rows,
		.loadM = (params.loadM != 0 && params.output_rows > 1),
	};
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipeline_value = context->kernel_cache.findKernel<IndexSelectKernel, IndexSelectDescriptor, IndexSelectKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();

	struct {
		uint32_t row_units;
		uint32_t output_rows;
	} runtime_params = {
		row_units, params.output_rows,
	};
	auto encoder = command_batch->startCommand();
	encoder->setComputePipelineState(pipeline_value->pipeline.get());
	encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
	encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
	encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
	encoder->setBytes(&runtime_params, sizeof(runtime_params), 3);
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
	const MTL::Size threadgroup_size = pipeline_value->kernel->threadgroupSize(params.output_rows, threads_per_row);
	CCV_NNC_MFA_PRECONDITION(threadgroup_size.width <= pipeline_value->pipeline->maxTotalThreadsPerThreadgroup());
	const MTL::Size grid_size = pipeline_value->kernel->gridSize(params.output_rows, threads_per_row);
	CCV_NNC_MFA_PRECONDITION(grid_size.width > 0 && grid_size.height > 0 && grid_size.depth > 0);
	encoder->dispatchThreadgroups(grid_size, threadgroup_size);
	command_batch->finishCommand(encoder);
}
