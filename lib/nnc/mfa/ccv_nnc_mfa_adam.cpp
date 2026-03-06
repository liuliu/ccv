#include "ccv_nnc_mfa.hpp"
#include "kernels/AdamDescriptor.hpp"
#include "kernels/AdamKernel.hpp"

#include <math.h>

using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_adam(mfa::context* context, ccv_nnc_mfa_adam_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_adam(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_adam_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  if (params.amsgrad) {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 9);
  } else {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 7);
  }

  CCV_NNC_MFA_PRECONDITION((params.data_type == MTL::DataTypeFloat) || (params.data_type == MTL::DataTypeHalf));

  AdamDescriptor descriptor;
  descriptor.adamw = params.adamw;
  descriptor.amsgrad = params.amsgrad;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16;
  descriptor.rate = params.rate;
  descriptor.scale = params.scale;
  descriptor.beta1 = params.beta1;
  descriptor.beta2 = params.beta2;
  descriptor.decay = params.decay;
  descriptor.epsilon = params.epsilon;
  descriptor.length = params.length;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<AdamKernel, AdamDescriptor, AdamKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  const float rate_inv_bias_correction1 = params.rate / (1 - powf(params.beta1, params.step));
  const float inv_bias_correction2 = 1. / (1 - powf(params.beta2, params.step));
  float values[2] = { rate_inv_bias_correction1, inv_bias_correction2 };
  encoder->setBytes(values, sizeof(float) * 2, 10);

  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  if (tensors[1] != tensors[2]) {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (tensors[3] != tensors[5]) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[5], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (tensors[4] != tensors[6]) {
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[4], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (num_tensors == 9) {
    if (tensors[7] != tensors[8]) {
      encoder->useResource(tensors[7], MTL::ResourceUsageRead);
      encoder->useResource(tensors[8], MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[7], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
  }

  MTL::Size gridSize = kernel->gridSize(params.length);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
