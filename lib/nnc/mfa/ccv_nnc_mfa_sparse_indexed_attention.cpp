#include "ccv_nnc_mfa.hpp"
#include "kernels/NASparseIndexedAttentionDescriptor.hpp"
#include "kernels/NASparseIndexedAttentionKernel.hpp"
#include "kernels/NASparseIndexedAttentionKernelDescriptor.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_sparse_indexed_attention(mfa::context*, ccv_nnc_mfa_sparse_indexed_attention_params_t)
{
}

void ccv_nnc_mfa_encode_sparse_indexed_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sparse_indexed_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.H == 64);
  CCV_NNC_MFA_PRECONDITION((params.D == NASparseIndexedAttentionKernel::headDimension && params.variant <= 3) || (params.D == 128 && params.K > 0 && params.variant == 4));
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[3] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[5] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[7] != nullptr);
  CCV_NNC_MFA_PRECONDITION(params.variant <= 4);
  if (params.attention_sinks) {
    CCV_NNC_MFA_PRECONDITION(tensors[6] != nullptr);
  }

  NASparseIndexedAttentionDescriptor descriptor;
  switch (params.data_type) {
    case MTL::DataTypeHalf:
      descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
      break;
    case MTL::DataTypeBFloat:
      descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
      break;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
  }
  descriptor.attentionSinks = params.attention_sinks != 0;
  descriptor.T = params.T;
  descriptor.denseRows = params.dense_rows;
  descriptor.sparseRows = params.sparse_rows;
  descriptor.H = params.H;
  descriptor.K = params.K;
  descriptor.isCausal = params.is_causal != 0;
  descriptor.sinkHeadStride = params.sink_head_stride;
  descriptor.scale = params.scale;
  descriptor.variant = static_cast<NASparseIndexedAttentionVariant>(params.variant);

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<NASparseIndexedAttentionKernel, NASparseIndexedAttentionDescriptor, NASparseIndexedAttentionKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();

  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;
  MTL::Buffer* scratch = nullptr;
  if (kernel->usesDeviceScratch()) {
    scratch = context->request_scratch(kernel->scratchMemoryAllocation(params.T, params.H));
  }

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(), 0);
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  encoder->useResource(tensors[5], MTL::ResourceUsageRead);
  encoder->useResource(tensors[7], MTL::ResourceUsageWrite);
  encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
  encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
  encoder->setBuffer(tensors[3], tensor_offsets[3], 2);
  encoder->setBuffer(tensors[5], tensor_offsets[5], 3);
  if (params.attention_sinks) {
    encoder->useResource(tensors[6], MTL::ResourceUsageRead);
    encoder->setBuffer(tensors[6], tensor_offsets[6], 4);
  }
  encoder->setBuffer(tensors[7], tensor_offsets[7], 5);
  if (scratch) {
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 6);
  }
  encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(params.T, params.H), kernel->threadgroupSize());
  command_batch->finishCommand(encoder);
}
