#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include "v2/SwishDescriptor.hpp"
#include "v2/SwishKernel.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_swish(mfa::context* context, ccv_nnc_mfa_swish_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_swish(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_swish_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  if (params.gradient) {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 3);
  } else {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 2);
  }

  SwishDescriptor descriptor;
  descriptor.gradient = params.gradient ? 1 : 0;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.length = params.length;

  if (params.length % (4 * 256) == 0) {
    descriptor.value = 0;
  } else if (params.length % 4 == 0) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->v2_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<SwishKernel, SwishDescriptor, SwishKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());

  if (params.gradient) {
    if (tensors[0] == tensors[2]) {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    } else if (tensors[1] == tensors[2]) {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    }
  } else {
    if (tensors[0] == tensors[1]) {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    }
  }

  unsigned int count;
  if (params.length % 4 == 0) {
    count = params.length / 4;
  } else {
    count = params.length;
  }
  const int num_blocks = (count + 255) / 256;
  MTL::Size gridSize = MTL::Size(num_blocks, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

  command_batch->finishCommand(encoder);
}

