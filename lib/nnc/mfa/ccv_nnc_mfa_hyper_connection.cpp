#include "ccv_nnc_mfa.hpp"
#include "kernels/HyperConnectionDescriptor.hpp"
#include "kernels/HyperConnectionKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_hyper_connection(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_hyper_connection_params_t)
{
}

void ccv_nnc_mfa_encode_hyper_connection(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_hyper_connection_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
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
		params.epsilon, params.operation, params.loadM != 0
	};
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
	const uint32_t threads = params.operation == 0 ? 32 : 256;
	encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), MTL::Size(threads, 1, 1));
	command_batch->finishCommand(encoder);
}
