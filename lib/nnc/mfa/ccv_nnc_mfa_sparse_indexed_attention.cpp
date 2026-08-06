#include "ccv_nnc_mfa.hpp"
#include "kernels/NASparseIndexedAttentionDescriptor.hpp"
#include "kernels/NASparseIndexedAttentionKernel.hpp"
#include "kernels/NASparseIndexedAttentionKernelDescriptor.hpp"
#include "kernels/SparseIndexedAttentionDescriptor.hpp"
#include "kernels/SparseIndexedAttentionKernel.hpp"
#include "kernels/SparseIndexedAttentionKernelDescriptor.hpp"
#include "kernels/SparseIndexedAttentionR1Descriptor.hpp"
#include "kernels/SparseIndexedAttentionR1Kernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_prepare_sparse_indexed_attention(mfa::context*, ccv_nnc_mfa_sparse_indexed_attention_params_t)
{
}

void ccv_nnc_mfa_encode_sparse_indexed_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_sparse_indexed_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  if (params.use_r1) {
    CCV_NNC_MFA_PRECONDITION(params.T == 1);
    CCV_NNC_MFA_PRECONDITION(params.H > 0);
    CCV_NNC_MFA_PRECONDITION(params.D > 0);
    CCV_NNC_MFA_PRECONDITION(params.D <= SparseIndexedAttentionR1Kernel::maxHeadDimension);
  } else if (params.use_neural_accelerators) {
    CCV_NNC_MFA_PRECONDITION(params.H == 64);
    CCV_NNC_MFA_PRECONDITION((params.D == NASparseIndexedAttentionKernel::headDimension && params.variant <= 3) || (params.D == 128 && params.K > 0 && params.variant == 4));
    CCV_NNC_MFA_PRECONDITION(params.variant != 2 && params.variant <= 4);
  } else {
    CCV_NNC_MFA_PRECONDITION(params.D <= SparseIndexedAttentionKernel::maxHeadDimension);
  }
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[3] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[5] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[7] != nullptr);
  if (params.attention_sinks) {
    CCV_NNC_MFA_PRECONDITION(tensors[6] != nullptr);
  }

  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP16;
  switch (params.data_type) {
    case MTL::DataTypeFloat:
      memoryPrecision = GEMMOperandPrecision::FP32;
      break;
    case MTL::DataTypeHalf:
      memoryPrecision = GEMMOperandPrecision::FP16;
      break;
    case MTL::DataTypeBFloat:
      memoryPrecision = GEMMOperandPrecision::BF16;
      break;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
  }

  auto setDescriptor =
  [&](auto& descriptor) {
    descriptor.memoryPrecision = memoryPrecision;
    descriptor.attentionSinks = params.attention_sinks != 0;
    descriptor.T = params.T;
    descriptor.denseRows = params.dense_rows;
    descriptor.sparseRows = params.sparse_rows;
    descriptor.H = params.H;
    descriptor.K = params.K;
    descriptor.isCausal = params.is_causal != 0;
    descriptor.slidingWindow = params.sliding_window;
    descriptor.sinkHeadStride = params.sink_head_stride;
    descriptor.scale = params.scale;
  };

  auto encodeNAPipeline =
  [&](auto pipelineValue, const auto& descriptor) {
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
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
    if (descriptor.loadRows) {
      const uint32_t runtimeRows[2] = { params.dense_rows, params.sparse_rows };
      encoder->setBytes(runtimeRows, sizeof(runtimeRows), 6);
    }
    encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(params.T, params.H), kernel->threadgroupSize());
    command_batch->finishCommand(encoder);
  };

  auto encodeGenericPipeline =
  [&](auto pipelineValue, const auto& descriptor) {
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
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
    if (descriptor.loadRows) {
      const uint32_t runtimeRows[2] = { params.dense_rows, params.sparse_rows };
      encoder->setBytes(runtimeRows, sizeof(runtimeRows), 6);
    }
    encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(params.T, params.H), kernel->threadgroupSize());
    command_batch->finishCommand(encoder);
  };

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  if (params.use_r1) {
    SparseIndexedAttentionR1Descriptor descriptor =
        SparseIndexedAttentionR1Descriptor::select(
            memoryPrecision,
            params.dense_rows,
            params.sparse_rows,
            params.K,
            params.H,
            params.D,
            params.scale,
            true,
            params.attention_sinks != 0,
            params.sliding_window);
    auto pipelineValue = shaderCache.findKernel<
        SparseIndexedAttentionR1Kernel,
        SparseIndexedAttentionR1Descriptor,
        SparseIndexedAttentionR1KernelDescriptor>(
            descriptor, context->device.get(), dprops);
    pool->drain();
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
    const uint32_t threadgroupSize = kernel->threadgroupSize(descriptor);
    CCV_NNC_MFA_PRECONDITION(
        threadgroupSize <= pipeline->maxTotalThreadsPerThreadgroup());
    const uint32_t runtimeShape[3] = {
      params.dense_rows,
      params.sparse_rows,
      params.K,
    };
    if (descriptor.mode == SparseIndexedAttentionR1Descriptor::Mode::direct) {
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(
          kernel->threadgroupMemoryAllocation(descriptor), 0);
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
      if (descriptor.loadK) {
        encoder->setBytes(runtimeShape, sizeof(runtimeShape), 6);
      }
      if (params.attention_sinks) {
        encoder->setBytes(
            &params.sink_head_stride, sizeof(params.sink_head_stride), 7);
      }
      encoder->dispatchThreadgroups(
          MTL::Size(params.H, 1, 1),
          MTL::Size(threadgroupSize, 1, 1));
      command_batch->finishCommand(encoder);
      return;
    }

    const size_t partialBytes =
        (size_t)params.H * descriptor.workgroups *
        (params.D + 2) * sizeof(float);
    auto scratch = context->request_scratch(partialBytes);
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setThreadgroupMemoryLength(
        kernel->threadgroupMemoryAllocation(descriptor), 0);
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[5], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBuffer(tensors[3], tensor_offsets[3], 2);
    encoder->setBuffer(tensors[5], tensor_offsets[5], 3);
    if (params.attention_sinks) {
      encoder->useResource(tensors[6], MTL::ResourceUsageRead);
      encoder->setBuffer(tensors[6], tensor_offsets[6], 4);
    }
    encoder->setBuffer(scratch, 0, 5);
    if (descriptor.loadK) {
      encoder->setBytes(runtimeShape, sizeof(runtimeShape), 6);
    }
    if (params.attention_sinks) {
      encoder->setBytes(
          &params.sink_head_stride, sizeof(params.sink_head_stride), 7);
    }
    encoder->dispatchThreadgroups(
        MTL::Size(params.H, 1, descriptor.workgroups),
        MTL::Size(threadgroupSize, 1, 1));
    command_batch->finishCommand(encoder);

    auto reduceEncoder = command_batch->startCommand();
    reduceEncoder->setComputePipelineState(pipelineValue->second.get());
    reduceEncoder->useResource(scratch, MTL::ResourceUsageRead);
    reduceEncoder->useResource(tensors[7], MTL::ResourceUsageWrite);
    reduceEncoder->setBuffer(scratch, 0, 0);
    reduceEncoder->setBuffer(tensors[7], tensor_offsets[7], 1);
    reduceEncoder->dispatchThreadgroups(
        MTL::Size(params.H, 1, 1),
        MTL::Size(32, 1, 1));
    command_batch->finishCommand(reduceEncoder);
    return;
  } else if (params.use_neural_accelerators) {
    NASparseIndexedAttentionDescriptor descriptor;
    setDescriptor(descriptor);
    descriptor.loadRows = true;
    descriptor.variant = static_cast<NASparseIndexedAttentionVariant>(params.variant);
    auto pipelineValue = shaderCache.findKernel<NASparseIndexedAttentionKernel, NASparseIndexedAttentionDescriptor, NASparseIndexedAttentionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodeNAPipeline(pipelineValue, descriptor);
  } else {
    SparseIndexedAttentionDescriptor descriptor;
    setDescriptor(descriptor);
    descriptor.D = params.D;
    descriptor.loadRows = true;
    auto pipelineValue = shaderCache.findKernel<SparseIndexedAttentionKernel, SparseIndexedAttentionDescriptor, SparseIndexedAttentionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodeGenericPipeline(pipelineValue, descriptor);
  }
}
