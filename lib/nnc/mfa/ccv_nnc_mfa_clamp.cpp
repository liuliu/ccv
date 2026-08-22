#include "ccv_nnc_mfa.hpp"
#include "kernels/ClampDescriptor.hpp"
#include "kernels/ClampKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_clamp(mfa::context* context, ccv_nnc_mfa_clamp_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_clamp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_clamp_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();
  encoder->setBuffer(tensors[0], tensor_offsets[0], NS::UInteger(0));
  encoder->setBuffer(tensors[1], tensor_offsets[1], NS::UInteger(1));
  encoder->setBytes(&params.min, sizeof(params.min), NS::UInteger(2));
  encoder->setBytes(&params.max, sizeof(params.max), NS::UInteger(3));

  ClampDescriptor descriptor;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16;
  descriptor.length = params.length;
  descriptor.bounds = params.bounds;
  descriptor.loadM = params.loadM;
  if (!params.loadM && params.length % (4 * 256) == 0)
    descriptor.value = 0;
  else if (params.length % 4 == 0)
    descriptor.value = 1;
  else
    descriptor.value = 2;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  auto pipelineValue = shaderCache.findKernel<ClampKernel, ClampDescriptor, ClampKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  if (tensors[0] == tensors[1])
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  }

  const uint32_t count = params.length % 4 == 0 ? params.length / 4 : params.length;
  if (params.loadM)
    encoder->setBytes(&count, sizeof(count), NS::UInteger(4));
  encoder->dispatchThreadgroups(MTL::Size((count + 255) / 256, 1, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
