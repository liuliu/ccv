#include "ccv_nnc_mfa.hpp"
#include "kernels/HyperConnectionDescriptor.hpp"
#include "kernels/HyperConnectionKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_hyper_connection(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_hyper_connection_params_t)
{
}

static void _ccv_nnc_mfa_encode_hyper_connection_mix_stage(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();
	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
	{
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		encoder->useResource(tensors[num_tensors], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
		num_tensors++;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == (params.operation == 3 || params.operation == 4 ? 7 : params.operation == 5 ? 11 : 5));
	HyperConnectionDescriptor descriptor {
		0, params.count, params.hidden, params.sinkhorn_iterations,
		params.epsilon, params.operation, false, false
	};
	descriptor.normEpsilon = params.norm_epsilon;
	descriptor.dotSplits = params.dot_splits;
	descriptor.collapseSplits = params.collapse_splits;
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipelineValue = context->kernel_cache.findKernel<HyperConnectionKernel, HyperConnectionDescriptor, HyperConnectionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
	encoder->setComputePipelineState(pipelineValue->pipeline.get());
	const uint32_t mix_dim = 2 * params.count + params.count * params.count;
	switch (params.operation)
	{
		case 3:
			encoder->dispatchThreadgroups(MTL::Size((mix_dim + 1) / 2 * params.dot_splits + params.collapse_splits, params.row_count, 1), MTL::Size(CCV_NNC_MFA_HYPER_CONNECTION_MIX_THREADS, 1, 1));
			break;
		case 4:
			encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), MTL::Size(32, 1, 1));
			break;
		case 5:
			// The split's threadgroup, then the normalization's.
			encoder->dispatchThreadgroups(MTL::Size(1 + params.collapse_splits, params.row_count, 1), MTL::Size(CCV_NNC_MFA_HYPER_CONNECTION_MIX_THREADS, 1, 1));
			break;
		default:
			encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), MTL::Size(1024, 1, 1));
			break;
	}
	command_batch->finishCommand(encoder);
}

void ccv_nnc_mfa_encode_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	if (params.operation >= 3)
	{
		_ccv_nnc_mfa_encode_hyper_connection_mix_stage(context, params, command_batch, tensors, tensor_offsets);
		return;
	}
	auto encoder = command_batch->startCommand();
	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
	{
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors++;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 8);
	HyperConnectionDescriptor descriptor {
		params.row_count, params.count, params.hidden, params.sinkhorn_iterations,
		params.epsilon, params.operation, params.block_fp16 != 0, params.loadM != 0
	};
	descriptor.expandAddend = params.expand_addend != 0;
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipelineValue = context->kernel_cache.findKernel<HyperConnectionKernel, HyperConnectionDescriptor, HyperConnectionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
	encoder->setComputePipelineState(pipelineValue->pipeline.get());
	if (params.loadM)
		encoder->setBytes(&params.row_count, sizeof(params.row_count), 8);
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	if (params.operation == 2)
	{
		encoder->useResource(tensors[3], MTL::ResourceUsageRead);
		encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
		if (params.expand_addend)
			encoder->useResource(tensors[5], MTL::ResourceUsageRead);
	} else {
		if (params.operation == 0)
			encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
		encoder->useResource(tensors[5], MTL::ResourceUsageWrite);
		encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
	}
	if (params.operation == 1)
	{
		encoder->useResource(tensors[3], MTL::ResourceUsageRead);
		encoder->useResource(tensors[7], MTL::ResourceUsageWrite);
	}
	uint32_t threads = params.operation == 0 ? 32 : 256;
	if (params.row_count == 1 && params.operation == 1)
		threads = (uint32_t)pipelineValue->pipeline->maxTotalThreadsPerThreadgroup();
	encoder->dispatchThreadgroups(MTL::Size(params.row_count, params.operation == 2 && params.row_count == 1 ? (params.count * params.hidden + 255) / 256 : 1, 1), MTL::Size(threads, 1, 1));
	command_batch->finishCommand(encoder);
}
