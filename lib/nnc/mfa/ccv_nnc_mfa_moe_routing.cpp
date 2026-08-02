#include "ccv_nnc_mfa.hpp"
#include "kernels/MoERoutingDescriptor.hpp"
#include "kernels/MoERoutingKernel.hpp"
using namespace ccv::nnc;

typedef struct {
	uint32_t expert_count;
	uint32_t kth;
	uint32_t hidden;
	float weight_scale;
	uint32_t preselected;
} ccv_nnc_mfa_moe_routing_dispatch_params_t;

void ccv_nnc_mfa_prepare_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params)
{
	const MoERoutingDescriptor descriptor { params.data_type };
	auto pool = NS::AutoreleasePool::alloc()->init();
	context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
}

void ccv_nnc_mfa_encode_moe_routing(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_moe_routing_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.expert_count > 0 && params.expert_count <= 256);
	CCV_NNC_MFA_PRECONDITION(params.kth > 0 && params.kth <= 32 && params.kth <= params.expert_count);
	CCV_NNC_MFA_PRECONDITION(params.hidden > 0);
	auto encoder = command_batch->startCommand();
	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
	{
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		++num_tensors;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 8);
	const MoERoutingDescriptor descriptor { params.data_type };
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipeline_value = context->kernel_cache.findKernel<MoERoutingKernel, MoERoutingDescriptor, MoERoutingKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
	pool->drain();
	encoder->setComputePipelineState(pipeline_value->pipeline.get());
	const ccv_nnc_mfa_moe_routing_dispatch_params_t dispatch_params = {
		.expert_count = params.expert_count,
		.kth = params.kth,
		.hidden = params.hidden,
		.weight_scale = params.weight_scale,
		.preselected = params.preselected,
	};
	encoder->setBytes(&dispatch_params, sizeof(dispatch_params), 8);
	encoder->setThreadgroupMemoryLength(NS::UInteger(params.expert_count * sizeof(float)), 0);
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	int i;
	for (i = 3; i < 8; i++)
		encoder->useResource(tensors[i], MTL::ResourceUsageWrite);
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
	command_batch->finishCommand(encoder);
}
