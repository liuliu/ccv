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

// Two-pass prefill keeps a probability column for every visible dense row and
// virtual sparse row, for every head of a chunk of tokens.
static uint64_t two_pass_columns(const ccv_nnc_mfa_sparse_indexed_attention_params_t& params)
{
  const uint32_t block_size = std::max(params.block_size, 1u);
  const uint32_t dense_columns = !params.is_causal ? params.dense_rows :
    block_size > 1 ? std::min(params.dense_rows, block_size - 1) :
    params.sliding_window > 0 ? std::min(params.dense_rows, params.sliding_window) : params.dense_rows;
  return uint64_t(dense_columns) + uint64_t(params.K) * block_size;
}

// Every head's probabilities and normalizer, then the token's valid column count.
static uint64_t two_pass_token_bytes(const ccv_nnc_mfa_sparse_indexed_attention_params_t& params)
{
  return uint64_t(params.H) * (two_pass_columns(params) + 1) * sizeof(float) + sizeof(uint32_t);
}

static uint32_t two_pass_chunk_tokens(const ccv_nnc_mfa_sparse_indexed_attention_params_t& params)
{
  return (uint32_t)std::min<uint64_t>(params.T, NASparseIndexedAttentionKernel::twoPassScratchBytes / two_pass_token_bytes(params));
}

// Matches two_pass_chunk in the two-pass kernels.
struct two_pass_chunk {
  uint32_t token_base;
  uint32_t tokens;
  uint32_t columns;
};

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
    descriptor.loadM = params.loadM;
    descriptor.loadK = params.loadC;
    descriptor.denseRows = params.dense_rows;
    descriptor.sparseRows = params.sparse_rows;
    descriptor.H = params.H;
    descriptor.K = params.K;
    descriptor.isCausal = params.is_causal != 0;
    descriptor.slidingWindow = params.sliding_window;
    descriptor.blockSize = std::max(params.block_size, 1u);
    descriptor.sinkHeadStride = params.sink_head_stride;
    descriptor.scale = params.scale;
  };

  auto encodeNAPipeline =
  [&](auto pipelineValue, const auto& descriptor) {
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline.get());
    const NS::UInteger threadgroupMemoryAllocation = kernel->threadgroupMemoryAllocation();
    CCV_NNC_MFA_PRECONDITION(threadgroupMemoryAllocation + pipeline->staticThreadgroupMemoryLength() <= context->device->maxThreadgroupMemoryLength());
    encoder->setThreadgroupMemoryLength(threadgroupMemoryAllocation, 0);
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
    if (descriptor.loadRows || descriptor.loadM || descriptor.loadK) {
      const uint32_t runtimeRows[4] = { params.dense_rows, params.sparse_rows, params.T, params.K };
      encoder->setBytes(runtimeRows, sizeof(runtimeRows), 6);
    }
    encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(params.T, params.H), kernel->threadgroupSize());
    command_batch->finishCommand(encoder);
  };

  // Probabilities for a chunk of tokens, then their values, until every token is done.
  auto encodeNATwoPassPipeline =
  [&](auto pipelineValue, const auto& descriptor) {
    auto kernel = pipelineValue->kernel;
    const uint64_t columns = two_pass_columns(params);
    CCV_NNC_MFA_PRECONDITION(columns <= UINT32_MAX);
    const uint32_t chunkTokens = two_pass_chunk_tokens(params);
    CCV_NNC_MFA_PRECONDITION(chunkTokens > 0);
    // Scratch holds the chunk's probabilities, its normalizers, then its column counts.
    const uint64_t normalizersOffset = uint64_t(chunkTokens) * params.H * columns * sizeof(float);
    const uint64_t countsOffset = normalizersOffset + uint64_t(chunkTokens) * params.H * sizeof(float);
    auto scratch = context->request_scratch(countsOffset + uint64_t(chunkTokens) * sizeof(uint32_t));
    const uint32_t runtimeRows[4] = { params.dense_rows, params.sparse_rows, params.T, params.K };
    const bool loadRuntimeRows = descriptor.loadRows || descriptor.loadM || descriptor.loadK;
    const NS::UInteger threadgroupMemoryAllocation = kernel->threadgroupMemoryAllocation();
    CCV_NNC_MFA_PRECONDITION(threadgroupMemoryAllocation + pipelineValue->pipeline->staticThreadgroupMemoryLength() <= context->device->maxThreadgroupMemoryLength());
    for (uint32_t tokenBase = 0; tokenBase < params.T; tokenBase += chunkTokens) {
      const two_pass_chunk chunk = { tokenBase, std::min(chunkTokens, params.T - tokenBase), (uint32_t)columns };
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipelineValue->pipeline.get());
      encoder->setThreadgroupMemoryLength(threadgroupMemoryAllocation, 0);
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      encoder->useResource(tensors[5], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      encoder->setBuffer(tensors[3], tensor_offsets[3], 2);
      encoder->setBuffer(tensors[5], tensor_offsets[5], 3);
      if (params.attention_sinks) {
        encoder->useResource(tensors[6], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[6], tensor_offsets[6], 4);
      }
      encoder->setBuffer(scratch, 0, 5);
      if (loadRuntimeRows) {
        encoder->setBytes(runtimeRows, sizeof(runtimeRows), 6);
      }
      encoder->setBytes(&chunk, sizeof(chunk), 7);
      encoder->setBuffer(scratch, countsOffset, 8);
      encoder->setBuffer(scratch, normalizersOffset, 9);
      encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(chunk.tokens, params.H), kernel->threadgroupSize());
      command_batch->finishCommand(encoder);

      auto valuesEncoder = command_batch->startCommand();
      valuesEncoder->setComputePipelineState(pipelineValue->second.get());
      valuesEncoder->setThreadgroupMemoryLength(kernel->valuesThreadgroupMemoryAllocation(), 0);
      valuesEncoder->useResource(scratch, MTL::ResourceUsageRead);
      valuesEncoder->useResource(tensors[1], MTL::ResourceUsageRead);
      valuesEncoder->useResource(tensors[3], MTL::ResourceUsageRead);
      valuesEncoder->useResource(tensors[5], MTL::ResourceUsageRead);
      valuesEncoder->useResource(tensors[7], MTL::ResourceUsageWrite);
      valuesEncoder->setBuffer(scratch, 0, 0);
      valuesEncoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      valuesEncoder->setBuffer(tensors[3], tensor_offsets[3], 2);
      valuesEncoder->setBuffer(tensors[5], tensor_offsets[5], 3);
      valuesEncoder->setBuffer(tensors[7], tensor_offsets[7], 5);
      if (loadRuntimeRows) {
        valuesEncoder->setBytes(runtimeRows, sizeof(runtimeRows), 6);
      }
      valuesEncoder->setBytes(&chunk, sizeof(chunk), 7);
      valuesEncoder->setBuffer(scratch, countsOffset, 8);
      valuesEncoder->setBuffer(scratch, normalizersOffset, 9);
      valuesEncoder->dispatchThreadgroups(kernel->valuesThreadgroupsPerGrid(chunk.tokens, params.H), kernel->valuesThreadgroupSize());
      command_batch->finishCommand(valuesEncoder);
    }
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
    if (descriptor.loadRows || descriptor.loadM || descriptor.loadK) {
      const uint32_t runtimeRows[4] = { params.dense_rows, params.sparse_rows, params.T, params.K };
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
            params.sliding_window,
            std::max(params.block_size, 1u));
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
        MTL::Size((params.H + descriptor.simdgroups - 1) / descriptor.simdgroups, 1, descriptor.workgroups),
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
        MTL::Size(kernel->reduceThreadgroupSize(descriptor), 1, 1));
    command_batch->finishCommand(reduceEncoder);
    return;
  } else if (params.use_neural_accelerators) {
    NASparseIndexedAttentionDescriptor descriptor;
    setDescriptor(descriptor);
    descriptor.loadRows = true;
    descriptor.variant = static_cast<NASparseIndexedAttentionVariant>(params.variant);
    // D=512 prefill stores probabilities while a bounded scratch chunk still
    // gives every core enough workgroups; wider rows stay in one pass.
    const uint64_t twoPassMinimumTokens = uint64_t(context->device_properties.coreCount) * NASparseIndexedAttentionKernel::twoPassMinimumTokensPerCore;
    if (descriptor.variant == NASparseIndexedAttentionVariant::Threadgroup64 && params.K > 0 &&
        params.H % NASparseIndexedAttentionKernel::twoPassHeadGroup == 0 &&
        two_pass_chunk_tokens(params) >= std::min<uint64_t>(params.T, twoPassMinimumTokens))
      descriptor.variant = NASparseIndexedAttentionVariant::Threadgroup64TwoPass;
    auto pipelineValue = shaderCache.findKernel<NASparseIndexedAttentionKernel, NASparseIndexedAttentionDescriptor, NASparseIndexedAttentionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    if (descriptor.variant == NASparseIndexedAttentionVariant::Threadgroup64TwoPass)
      encodeNATwoPassPipeline(pipelineValue, descriptor);
    else
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
