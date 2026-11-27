#include "ccv_nnc_mfa.hpp"
#include "kernels/MoERoutingDescriptor.hpp"
#include "kernels/MoERoutingKernel.hpp"
using namespace ccv::nnc;

// Any token count past the few-token kernel's routes in three stages, so the pipelines do not depend on it.
static bool _ccv_nnc_mfa_moe_routing_many(const ccv_nnc_mfa_moe_routing_params_t& params)
{
	return params.token_count > 16;
}

static MoERoutingDescriptor _ccv_nnc_mfa_moe_routing_descriptor(const ccv_nnc_mfa_moe_routing_params_t& params, const uint32_t stage)
{
	return MoERoutingDescriptor {
		params.activation_data_type,
		params.routing_data_type,
		params.bias_data_type,
		params.expert_count,
		params.kth,
		params.hidden,
		params.weight_scale,
		params.preselected != 0,
		params.single_input_token != 0,
		params.normalization_epsilon,
		params.sigmoid != 0,
		stage == 0 ? params.token_count : 0,
		stage,
	};
}

void ccv_nnc_mfa_prepare_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params)
{
	auto pool = NS::AutoreleasePool::alloc()->init();
	if (_ccv_nnc_mfa_moe_routing_many(params))
		for (uint32_t stage = 1; stage <= 3; stage++)
			context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(_ccv_nnc_mfa_moe_routing_descriptor(params, stage), context->device.get(), DeviceProperties());
	else
		context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(_ccv_nnc_mfa_moe_routing_descriptor(params, 0), context->device.get(), DeviceProperties());
	pool->drain();
}

void ccv_nnc_mfa_encode_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.expert_count > 0 && params.expert_count <= 512);
	CCV_NNC_MFA_PRECONDITION(params.kth > 0 && params.kth <= 32 && params.kth <= params.expert_count);
	CCV_NNC_MFA_PRECONDITION(params.token_count > 0);
	CCV_NNC_MFA_PRECONDITION(params.hidden > 0);
	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
		++num_tensors;
	CCV_NNC_MFA_PRECONDITION(num_tensors == 8);
	if (_ccv_nnc_mfa_moe_routing_many(params))
	{
		CCV_NNC_MFA_PRECONDITION(params.kth <= 8 && params.single_input_token == 0);
		// The pairs' experts and weights pass through `gathered`, which holds hidden elements per pair.
		const uint32_t activation_size = params.activation_data_type == 3 ? 4 : 2;
		CCV_NNC_MFA_PRECONDITION(params.hidden * activation_size >= sizeof(int) + sizeof(float));
		const uint32_t pair_count = params.token_count * params.kth;
		for (uint32_t stage = 1; stage <= 3; stage++)
		{
			auto pool = NS::AutoreleasePool::alloc()->init();
			auto pipeline_value = context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(_ccv_nnc_mfa_moe_routing_descriptor(params, stage), context->device.get(), DeviceProperties());
			pool->drain();
			auto encoder = command_batch->startCommand();
			for (int i = 0; i < 8; i++)
				encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
			encoder->setBytes(&params.token_count, sizeof(params.token_count), 8);
			encoder->setComputePipelineState(pipeline_value->pipeline.get());
			for (int i = 0; i < 8; i++)
				encoder->useResource(tensors[i], i < 3 ? MTL::ResourceUsageRead : MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
			if (stage == 1) {
				const uint32_t simd_groups = 8;
				encoder->dispatchThreadgroups(MTL::Size((params.token_count + simd_groups - 1) / simd_groups, 1, 1), MTL::Size(simd_groups * 32, 1, 1));
			} else if (stage == 2) {
				const NS::UInteger threads = std::min<NS::UInteger>((params.expert_count + 31) / 32 * 32, pipeline_value->pipeline->maxTotalThreadsPerThreadgroup() / 32 * 32);
				encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(threads, 1, 1));
			} else {
				const NS::UInteger count = NS::UInteger(pair_count) * params.hidden;
				encoder->dispatchThreads(MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
			}
			command_batch->finishCommand(encoder);
		}
		return;
	}
	auto encoder = command_batch->startCommand();
	for (int i = 0; i < 8; i++)
		encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
	const MoERoutingDescriptor descriptor = _ccv_nnc_mfa_moe_routing_descriptor(params, 0);
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipeline_value = context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
	encoder->setComputePipelineState(pipeline_value->pipeline.get());
	encoder->setThreadgroupMemoryLength((NS::UInteger(params.expert_count * sizeof(float)) + 15) & ~NS::UInteger(15), 0);
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	int i;
	for (i = 3; i < 8; i++)
		encoder->useResource(tensors[i], MTL::ResourceUsageWrite);
	// The few-token kernel needs a SIMD group per token; the threads past those only gather.
	const NS::UInteger threads = std::min<NS::UInteger>(descriptor.executionSIMDGroups() * 32, pipeline_value->pipeline->maxTotalThreadsPerThreadgroup() / 32 * 32);
	CCV_NNC_MFA_PRECONDITION(threads >= params.token_count * 32);
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(threads, 1, 1));
	command_batch->finishCommand(encoder);
}
