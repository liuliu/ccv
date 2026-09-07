#include "ccv_nnc_mfa.hpp"
#include "kernels/MulDescriptor.hpp"
#include "kernels/MulKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_mul(mfa::context* context, ccv_nnc_mfa_mul_params_t params)
{
  // Do nothing now.
}

void ccv_nnc_mfa_encode_mul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_mul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.length > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] && tensors[1] && tensors[2] && !tensors[3]);
  auto encoder = command_batch->startCommand();
  for (int i = 0; i < 3; i++)
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));

  // Larger 16-bit unary / binary workloads benefit from wider threadgroups.
  const unsigned int threadgroup_width = params.data_type != MTL::DataTypeFloat && params.length >= 262144 ? 512 : 256;

  MulDescriptor descriptor;
  if (params.data_type == MTL::DataTypeFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
  } else if (params.data_type == MTL::DataTypeBFloat) {
    descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
  } else {
    descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  }
  descriptor.length = params.length;
  descriptor.loadM = params.loadM;
  if (!params.loadM && params.length % (4 * threadgroup_width) == 0) {
    descriptor.value = 0;
  } else if (params.length % 4 == 0) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<MulKernel, MulDescriptor, MulKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  for (int i = 0; i < 2; i++) {
    if (tensors[i] == tensors[2]) {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    }
  }
  if (tensors[0] != tensors[2] && tensors[1] != tensors[2])
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);

  const uint32_t count = params.length % 4 == 0 ? params.length / 4 : params.length;
  if (params.loadM)
    encoder->setBytes(&count, sizeof(count), NS::UInteger(3));
  const MTL::Size gridSize = MTL::Size(((size_t)count + threadgroup_width - 1) / threadgroup_width, 1, 1);
  encoder->dispatchThreadgroups(gridSize, MTL::Size(threadgroup_width, 1, 1));
  command_batch->finishCommand(encoder);
}
