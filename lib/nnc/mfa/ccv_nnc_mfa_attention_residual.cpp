#include "ccv_nnc_mfa.hpp"
#include "kernels/AttentionResidualDescriptor.hpp"
#include "kernels/AttentionResidualKernel.hpp"

void ccv_nnc_mfa_encode_attention_residual(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_residual_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  // Buffers: block states, partial state, queries, then an output for each query.
  CCV_NNC_MFA_PRECONDITION(params.query_count >= 1 && params.query_count <= CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_QUERIES);
  CCV_NNC_MFA_PRECONDITION(params.block_count + 1 <= CCV_NNC_MFA_ATTENTION_RESIDUAL_MAX_STATES);
  CCV_NNC_MFA_PRECONDITION(tensors[3 + params.query_count] == nullptr);
  if (params.rows == 0 || params.channels == 0)
    return;
  AttentionResidualDescriptor descriptor;
  descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 :
    (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
  descriptor.queryCount = params.query_count;
  descriptor.channels = params.channels;
  descriptor.blockCount = params.block_count;
  descriptor.epsilon = params.epsilon;
  descriptor.scale = params.scale;
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<AttentionResidualKernel, AttentionResidualDescriptor, AttentionResidualKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();
  // The kernel strides a row's channels by its full threadgroup.
  CCV_NNC_MFA_PRECONDITION(pipelineValue->kernel->threadgroupSize.width <= pipelineValue->pipeline->maxTotalThreadsPerThreadgroup());

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  for (int i = 0; i < 3; i++) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  }
  for (uint32_t i = 0; i < params.query_count; i++) {
    encoder->setBuffer(tensors[3 + i], tensor_offsets[3 + i], NS::UInteger(3 + i));
    encoder->useResource(tensors[3 + i], MTL::ResourceUsageWrite);
  }
  // One threadgroup mixes a row, and the kernel takes the row count from the grid.
  encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
