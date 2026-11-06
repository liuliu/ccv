#include "ccv_nnc_mfa.hpp"
#include "kernels/CausalConvolutionDescriptor.hpp"
#include "kernels/CausalConvolutionKernel.hpp"

void ccv_nnc_mfa_prepare_causal_convolution(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_causal_convolution_params_t params)
{
  // No eager preparation is needed.
}

void ccv_nnc_mfa_encode_causal_convolution(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_causal_convolution_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  // Buffers: x, previous rows, weights, y, next rows.
  CCV_NNC_MFA_PRECONDITION(tensors[5] == nullptr);
  if (params.batch_size == 0 || params.channels == 0)
    return;
  CausalConvolutionDescriptor descriptor;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 :
    (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
  descriptor.silu = params.silu != 0;
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<CausalConvolutionKernel, CausalConvolutionDescriptor, CausalConvolutionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  for (int i = 0; i < 5; i++)
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
  const uint32_t sizes[5] = { params.batch_size, params.sequence_length, params.channels, params.kernel_size, params.next_rows };
  encoder->setBytes(sizes, sizeof(sizes), NS::UInteger(5));
  for (int i = 0; i < 3; i++)
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  encoder->useResource(tensors[4], MTL::ResourceUsageWrite);
  // One thread per channel and context row: the first T rows convolve, the rest copy the next rows.
  const NS::UInteger threads = pipelineValue->kernel->threadgroupSize.width;
  const NS::UInteger rows = params.sequence_length + params.next_rows;
  encoder->dispatchThreadgroups(MTL::Size((params.channels + threads - 1) / threads, rows, params.batch_size), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
