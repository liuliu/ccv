#include "ccv_nnc_mfa.hpp"
#include "kernels/ScatterAddDescriptor.hpp"

using namespace ccv::nnc;

enum {
	CCV_NNC_MFA_SCATTER_ADD_SCAN,
	CCV_NNC_MFA_SCATTER_ADD_ATOMIC_MIN,
	CCV_NNC_MFA_SCATTER_ADD_ATOMIC_SORT,
};

void ccv_nnc_mfa_prepare_scatter_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scatter_add_params_t params)
{
	CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeFloat);
	CCV_NNC_MFA_PRECONDITION(params.input_rows > 0);
	CCV_NNC_MFA_PRECONDITION(params.output_rows > 0);
	CCV_NNC_MFA_PRECONDITION(params.columns > 0 && params.columns % 4 == 0);
	CCV_NNC_MFA_PRECONDITION(params.output_rows == 1 ||
		(params.count_per_output > 0 && (uint64_t)params.output_rows * params.count_per_output == params.input_rows));
	const ScatterAddDescriptor descriptor = {
		.memoryPrecision = params.data_type == MTL::DataTypeHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32,
	};
	auto pool = NS::AutoreleasePool::alloc()->init();
	context->kernel_cache.findKernel<ScatterAddKernel, ScatterAddDescriptor, ScatterAddKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
}

void ccv_nnc_mfa_encode_scatter_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scatter_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(tensors[0] && tensors[1] && tensors[2] && !tensors[3]);
	CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeFloat);
	CCV_NNC_MFA_PRECONDITION(params.input_rows > 0);
	CCV_NNC_MFA_PRECONDITION(params.output_rows > 0);
	CCV_NNC_MFA_PRECONDITION(params.columns > 0 && params.columns % 4 == 0);
	CCV_NNC_MFA_PRECONDITION(params.output_rows == 1 ||
		(params.count_per_output > 0 && (uint64_t)params.output_rows * params.count_per_output == params.input_rows));
	const ScatterAddDescriptor descriptor = {
		.memoryPrecision = params.data_type == MTL::DataTypeHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32,
	};
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipeline_value = context->kernel_cache.findKernel<ScatterAddKernel, ScatterAddDescriptor, ScatterAddKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
	struct {
		uint32_t input_rows;
		uint32_t output_rows;
		uint32_t column_vectors;
		uint32_t count_per_output;
		uint32_t strategy;
	} runtime_params = {
		params.input_rows, params.output_rows, params.columns / 4, params.count_per_output, 0,
	};
	if (params.output_rows == 1)
	{
		auto encoder = command_batch->startCommand();
		encoder->setComputePipelineState(pipeline_value->pipeline.get());
		encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
		encoder->setBuffer(tensors[2], tensor_offsets[2], 1);
		encoder->setBytes(&runtime_params, sizeof(runtime_params), 2);
		encoder->useResource(tensors[0], MTL::ResourceUsageRead);
		encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
		const NS::UInteger width = std::min<NS::UInteger>(256, pipeline_value->pipeline->maxTotalThreadsPerThreadgroup());
		encoder->dispatchThreadgroups(MTL::Size((runtime_params.column_vectors + width - 1) / width, 1, 1), MTL::Size(width, 1, 1));
		command_batch->finishCommand(encoder);
		return;
	}
	uint64_t sort_count = 1;
	while (sort_count < params.count_per_output)
		sort_count <<= 1;
	const bool can_sort = sort_count * sizeof(uint32_t) <= context->device->maxThreadgroupMemoryLength();
	runtime_params.strategy = params.input_rows <= 32 || !can_sort ? CCV_NNC_MFA_SCATTER_ADD_SCAN :
		(params.count_per_output <= 8 ? CCV_NNC_MFA_SCATTER_ADD_ATOMIC_MIN : CCV_NNC_MFA_SCATTER_ADD_ATOMIC_SORT);
	const uint64_t counts_bytes = runtime_params.strategy == CCV_NNC_MFA_SCATTER_ADD_ATOMIC_SORT ?
		((uint64_t)params.output_rows * sizeof(uint32_t) + 255) & ~UINT64_C(255) : 0;
	const uint64_t inverse_offset = counts_bytes;
	const uint64_t scratch_bytes = counts_bytes + (uint64_t)params.input_rows * sizeof(uint32_t);
	MTL::Buffer* const scratch = context->request_scratch(scratch_bytes);
	if (runtime_params.strategy != CCV_NNC_MFA_SCATTER_ADD_SCAN)
	{
		auto encoder = command_batch->startCommand();
		encoder->setComputePipelineState(pipeline_value->second.get());
		encoder->setBuffer(scratch, 0, 0);
		encoder->setBytes(&runtime_params, sizeof(runtime_params), 1);
		encoder->useResource(scratch, MTL::ResourceUsageWrite);
		const NS::UInteger width = std::min<NS::UInteger>(256, pipeline_value->second->maxTotalThreadsPerThreadgroup());
		const uint64_t count = runtime_params.strategy == CCV_NNC_MFA_SCATTER_ADD_ATOMIC_MIN ? params.input_rows : params.output_rows;
		encoder->dispatchThreadgroups(MTL::Size((count + width - 1) / width, 1, 1), MTL::Size(width, 1, 1));
		command_batch->finishCommand(encoder);
	}
	{
		auto encoder = command_batch->startCommand();
		encoder->setComputePipelineState(pipeline_value->third.get());
		encoder->setBuffer(tensors[1], tensor_offsets[1], 0);
		encoder->setBuffer(scratch, 0, 1);
		encoder->setBuffer(scratch, inverse_offset, 2);
		encoder->setBytes(&runtime_params, sizeof(runtime_params), 3);
		encoder->useResource(tensors[1], MTL::ResourceUsageRead);
		encoder->useResource(scratch, runtime_params.strategy == CCV_NNC_MFA_SCATTER_ADD_SCAN ? MTL::ResourceUsageWrite :
			MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
		const uint64_t count = runtime_params.strategy == CCV_NNC_MFA_SCATTER_ADD_SCAN ? params.output_rows : params.input_rows;
		const NS::UInteger width = std::min<NS::UInteger>(256, pipeline_value->third->maxTotalThreadsPerThreadgroup());
		encoder->dispatchThreadgroups(MTL::Size((count + width - 1) / width, 1, 1), MTL::Size(width, 1, 1));
		command_batch->finishCommand(encoder);
	}
	if (runtime_params.strategy == CCV_NNC_MFA_SCATTER_ADD_ATOMIC_SORT)
	{
		auto encoder = command_batch->startCommand();
		encoder->setComputePipelineState(pipeline_value->fourth.get());
		encoder->setBuffer(scratch, inverse_offset, 0);
		encoder->setBytes(&runtime_params, sizeof(runtime_params), 1);
		encoder->setThreadgroupMemoryLength(sort_count * sizeof(uint32_t), 0);
		encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
		const NS::UInteger width = std::min<NS::UInteger>(256,
			std::min<NS::UInteger>(sort_count, pipeline_value->fourth->maxTotalThreadsPerThreadgroup()));
		encoder->dispatchThreadgroups(MTL::Size(params.output_rows, 1, 1), MTL::Size(width, 1, 1));
		command_batch->finishCommand(encoder);
	}
	{
		auto encoder = command_batch->startCommand();
		encoder->setComputePipelineState(pipeline_value->fifth.get());
		encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
		encoder->setBuffer(scratch, inverse_offset, 1);
		encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
		encoder->setBytes(&runtime_params, sizeof(runtime_params), 3);
		encoder->useResource(tensors[0], MTL::ResourceUsageRead);
		encoder->useResource(scratch, MTL::ResourceUsageRead);
		encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
		const uint64_t count = (uint64_t)params.output_rows * runtime_params.column_vectors;
		const NS::UInteger width = std::min<NS::UInteger>(256, pipeline_value->fifth->maxTotalThreadsPerThreadgroup());
		encoder->dispatchThreadgroups(MTL::Size((count + width - 1) / width, 1, 1), MTL::Size(width, 1, 1));
		command_batch->finishCommand(encoder);
	}
}
