#include "ccv_nnc_mfa.hpp"
#include "kernels/DepalettizeDescriptor.hpp"
#include "kernels/DepalettizeKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_depalettize(mfa::context* context, ccv_nnc_mfa_depalettize_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

  DepalettizeDescriptor descriptor;
  descriptor.qbits = params.qbits;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.numberInBlocks = params.number_in_blocks;
  descriptor.length = params.length;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<DepalettizeKernel, DepalettizeDescriptor, DepalettizeKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageWrite);

  MTL::Size gridSize = kernel->gridSize(params.length, params.number_in_blocks);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
