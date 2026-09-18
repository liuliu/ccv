#include "ccv_nnc_mfa.hpp"
#include "kernels/SignedSqrtDescriptor.hpp"
#include "kernels/SignedSqrtKernel.hpp"

void ccv_nnc_mfa_prepare_signed_sqrt(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_signed_sqrt_params_t params)
{
  // No eager preparation is needed.
}

void ccv_nnc_mfa_encode_signed_sqrt(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_signed_sqrt_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  if (params.length == 0)
    return;
  const int num_tensors = params.gradient ? 3 : 2;
  CCV_NNC_MFA_PRECONDITION(tensors[num_tensors] == nullptr);
  SignedSqrtDescriptor descriptor;
  descriptor.gradient = params.gradient != 0;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 :
    (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
  descriptor.length = params.length;
  descriptor.minimum_magnitude = params.minimum_magnitude;
  descriptor.loadM = params.loadM;
  bool vectorized = params.length % 4 == 0;
  const size_t vector_alignment = 4 * descriptor.memoryPrecision.size();
  for (int i = 0; i < num_tensors; i++)
    vectorized = vectorized && tensor_offsets[i] % vector_alignment == 0;
  if (vectorized && !params.loadM && params.length % (4 * 256) == 0) {
    descriptor.value = 0;
  } else if (vectorized) {
    descriptor.value = 1;
  } else {
    descriptor.value = 2;
  }
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<SignedSqrtKernel, SignedSqrtDescriptor, SignedSqrtKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  for (int i = 0; i < num_tensors; i++)
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
  const uint32_t count = vectorized ? params.length / 4 : params.length;
  if (params.loadM)
    encoder->setBytes(&count, sizeof(count), NS::UInteger(num_tensors));
  for (int i = 0; i < num_tensors - 1; i++)
    encoder->useResource(tensors[i], tensors[i] == tensors[num_tensors - 1] ? MTL::ResourceUsageRead | MTL::ResourceUsageWrite : MTL::ResourceUsageRead);
  if (tensors[num_tensors - 1] != tensors[0] && (!params.gradient || tensors[num_tensors - 1] != tensors[1]))
    encoder->useResource(tensors[num_tensors - 1], MTL::ResourceUsageWrite);
  const NS::UInteger num_blocks = (NS::UInteger(count) + 255) / 256;
  encoder->dispatchThreadgroups(MTL::Size(num_blocks, 1, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
