#include "ccv_nnc_mfa.hpp"
#include "kernels/MoEWeightsStreamingDescriptor.hpp"
#include "kernels/MoEWeightsStreamingKernel.hpp"
using namespace ccv::nnc;

static MoEWeightsStreamingDescriptor _ccv_nnc_mfa_moe_weights_streaming_descriptor()
{
	return MoEWeightsStreamingDescriptor { 0 };
}

void ccv_nnc_mfa_prepare_moe_weights_streaming(ccv_nnc_mfa_context_t* const context)
{
	auto pool = NS::AutoreleasePool::alloc()->init();
	context->kernel_cache.findKernel<MoEWeightsStreamingKernel,
		MoEWeightsStreamingDescriptor, MoEWeightsStreamingKernelDescriptor>(
			_ccv_nnc_mfa_moe_weights_streaming_descriptor(),
			context->device.get(), DeviceProperties());
	pool->drain();
}

void ccv_nnc_mfa_encode_moe_weights_streaming(
	ccv_nnc_mfa_context_t* const context,
	const ccv_nnc_mfa_moe_weights_streaming_params_t params,
	mtl_command_batch_t* const command_batch, mtl_buffer_t** const tensors,
	size_t* const tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.generation > 0);
	CCV_NNC_MFA_PRECONDITION(params.index_count > 0);
	CCV_NNC_MFA_PRECONDITION(params.expert_count > 0);
	CCV_NNC_MFA_PRECONDITION(params.resident_slots > 0 &&
		params.resident_slots <= params.expert_count);
	CCV_NNC_MFA_PRECONDITION(params.routing_width > 0);
	CCV_NNC_MFA_PRECONDITION(params.route_weight_count > 0);
	CCV_NNC_MFA_PRECONDITION(params.route_weight_bytes > 0);
	auto encoder = command_batch->startCommand();
	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr)
	{
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors],
			NS::UInteger(num_tensors));
		++num_tensors;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 11);
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipeline_value = context->kernel_cache.findKernel<MoEWeightsStreamingKernel,
		MoEWeightsStreamingDescriptor, MoEWeightsStreamingKernelDescriptor>(
			_ccv_nnc_mfa_moe_weights_streaming_descriptor(),
			context->device.get(), DeviceProperties());
	pool->drain();
	encoder->setComputePipelineState(pipeline_value->pipeline.get());
	encoder->setBytes(&params, sizeof(params), NS::UInteger(11));
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
	encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
	encoder->useResource(tensors[5], MTL::ResourceUsageWrite);
	encoder->useResource(tensors[6], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
	encoder->useResource(tensors[7], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
	encoder->useResource(tensors[8], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
	encoder->useResource(tensors[9], MTL::ResourceUsageWrite);
	encoder->useResource(tensors[10], MTL::ResourceUsageWrite);
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
	command_batch->finishCommand(encoder);
}
