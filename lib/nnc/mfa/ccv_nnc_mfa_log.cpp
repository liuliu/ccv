#include "ccv_nnc_mfa.hpp"
#include "kernels/LogDescriptor.hpp"
#include "kernels/LogKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_log(mfa::context* context, ccv_nnc_mfa_log_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_log(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_log_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    ++num_tensors;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);

  LogDescriptor descriptor;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16;
  descriptor.length = params.length;
  descriptor.loadM = params.loadM;
  if (!params.loadM && params.length % (4 * 256) == 0)
    descriptor.value = 0;
  else if (params.length % 4 == 0)
    descriptor.value = 1;
  else
    descriptor.value = 2;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  auto pipelineValue = shaderCache.findKernel<LogKernel, LogDescriptor, LogKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
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
    encoder->setBytes(&count, sizeof(count), NS::UInteger(2));
  encoder->dispatchThreadgroups(MTL::Size((count + 255) / 256, 1, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
