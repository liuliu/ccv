#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

#include "kernels/ShaderCache.hpp"
#include "kernels/AttentionKernel.hpp"
#include "kernels/AttentionKernelDescriptor.hpp"
#include "kernels/AttentionDescriptor.hpp"
#include "kernels/NAAttentionKernel.hpp"
#include "kernels/NAAttentionKernelDescriptor.hpp"
#include "kernels/NAAttentionDescriptor.hpp"
#include "kernels/NAInt8AttentionKernel.hpp"
#include "kernels/NAInt8AttentionKernelDescriptor.hpp"
#include "kernels/NAInt8AttentionDescriptor.hpp"

static uint32_t ccv_nnc_mfa_ceil_log2_u32(uint32_t x)
{
  if (x <= 1) {
    return 0;
  }
  --x;
  uint32_t bits = 0;
  while (x > 0) {
    x >>= 1;
    ++bits;
  }
  return bits;
}

// MARK: - C

void ccv_nnc_mfa_prepare_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params)
{
  if (params.masked &&
      !(params.type == 0 && params.use_neural_accelerators)) {
    context->attention_cache.prepare(context, mfa::attention::hash(params));
  }
}

void ccv_nnc_mfa_encode_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  mfa::attention::hash hash(params);

  if (!params.masked ||
      (params.type == 0 && params.use_neural_accelerators)) {
    simd::ushort2 num_batch_dims(0);
    simd::uint2 batch_sizes(1);
    if (params.batched) {
      for (uint16_t operand = 0; operand < 2; ++operand) {
        uint32_t* batch_dims;
        if (operand == 0) {
          batch_dims = params.batch_dims_q;
        } else if (operand == 1) {
          batch_dims = params.batch_dims_mask;
        }
        
        for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
          if (batch_dims[i] == 0) {
            break;
          }
          num_batch_dims[operand] += 1;
          batch_sizes[operand] *= batch_dims[i];
        }
        
        bool dims_match_q = true;
        if (num_batch_dims[0] != num_batch_dims[operand]) {
          dims_match_q = false;
        } else if (batch_sizes[0] != batch_sizes[operand]) {
          dims_match_q = false;
        } else {
          for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
            if (params.batch_dims_q[i] != batch_dims[i]) {
              dims_match_q = false;
            }
          }
        }
        
        if (!dims_match_q) {
          CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
        }
      }
    }
    if (params.type == 0 && params.use_neural_accelerators && params.use_quantized_attention) {
      NAInt8AttentionDescriptor attentionDesc;
      switch (params.data_type) {
      case MTL::DataTypeHalf:
        attentionDesc.ioPrecision = GEMMOperandPrecision::FP16;
        break;
      case MTL::DataTypeBFloat:
        attentionDesc.ioPrecision = GEMMOperandPrecision::BF16;
        break;
      case MTL::DataTypeFloat:
        attentionDesc.ioPrecision = GEMMOperandPrecision::FP32;
        break;
      default:
        CCV_NNC_MFA_PRECONDITION(false);
      }
      attentionDesc.matrixDimensions[0] = hash.R;
      attentionDesc.matrixDimensions[1] = hash.C;
      attentionDesc.matrixDimensions[2] = hash.D;
      attentionDesc.Hq = hash.Hq;
      attentionDesc.Hk = hash.Hk;
      attentionDesc.batchDimension = batch_sizes[0];
      attentionDesc.scale = hash.alpha;
      attentionDesc.isCausal = hash.is_causal;
      attentionDesc.masked = hash.masked;
      attentionDesc.isVarlen = hash.is_varlen;
      if (hash.masked && batch_sizes[1] > 1) {
        attentionDesc.maskBatchStride = hash.R * hash.C;
      }
      attentionDesc.lowPrecisionIntermediates =
          (params.data_type != MTL::DataTypeFloat && !hash.upcast) ? true : false;
      if (params.batched) {
        attentionDesc.batchStrides[AttentionOperand::Q] = hash.R * hash.D * hash.Hq;
        attentionDesc.batchStrides[AttentionOperand::K] = hash.C * hash.D * hash.Hk;
        attentionDesc.batchStrides[AttentionOperand::V] = hash.C * hash.D * hash.Hk;
        attentionDesc.batchStrides[AttentionOperand::O] = hash.R * hash.D * hash.Hq;
      }
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto &shaderCache = context->kernel_cache;
      DeviceProperties dprops = DeviceProperties();
      auto pipelineValue = shaderCache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor, NAInt8AttentionKernelDescriptor>(attentionDesc, context->device.get(), dprops);
      pool->drain();
      auto kernel = pipelineValue->kernel;
      auto pipeline = pipelineValue->pipeline;
      auto quantizeQPipeline = pipelineValue->second;
      auto quantizeKPipeline = pipelineValue->third;
      auto quantizeVPipeline = pipelineValue->fourth;
      auto computeVMeanPipeline = pipelineValue->fifth;
      auto blockMaskPipeline = pipelineValue->sixth;
      auto align_up =
      [&](size_t value) -> size_t {
        return (value + 255) & ~((size_t)255);
      };
      auto reserve =
      [&](size_t* total, size_t size) -> size_t {
        const size_t offset = *total;
        *total = align_up(*total + size);
        return offset;
      };

      const uint32_t batchDimension = attentionDesc.batchDimension;
      const uint32_t qTiles = (hash.R + kernel->blockDimensions[0] - 1) / kernel->blockDimensions[0];
      const uint32_t kTiles = (hash.C + kernel->blockDimensions[1] - 1) / kernel->blockDimensions[1];
      const uint32_t qBatchStride = hash.R * hash.D * hash.Hq;
      const uint32_t kvBatchStride = hash.C * hash.D * hash.Hk;
      const uint32_t qScaleBatchStride = hash.Hq * qTiles;
      const uint32_t kvScaleBatchStride = hash.Hk * kTiles;
      const size_t qInt8Bytes = (size_t)batchDimension * qBatchStride * sizeof(int8_t);
      const size_t kInt8Bytes = (size_t)batchDimension * kvBatchStride * sizeof(int8_t);
      const size_t vInt8Bytes = (size_t)batchDimension * kvBatchStride * sizeof(int8_t);
      const size_t qScaleBytes = (size_t)batchDimension * qScaleBatchStride * sizeof(float);
      const size_t kScaleBytes = (size_t)batchDimension * kvScaleBatchStride * sizeof(float);
      const size_t vScaleBytes = (size_t)batchDimension * kvScaleBatchStride * sizeof(float);
      const size_t vMeanBytes =
          (size_t)batchDimension * hash.Hk * hash.D * sizeof(float);
      const size_t blockMaskBytes =
          hash.masked ? (size_t)batch_sizes[1] * qTiles * kTiles * sizeof(uint8_t) : 0;
      const GEMMOperandPrecision lPrecision =
          attentionDesc.lowPrecisionIntermediates ?
          (attentionDesc.ioPrecision == GEMMOperandPrecision::BF16 ? GEMMOperandPrecision::BF16 :
              (attentionDesc.ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::FP16)) :
          GEMMOperandPrecision::FP32;
      const size_t lBytes = (size_t)batchDimension * hash.Hq * hash.R * lPrecision.size();
      size_t scratchSize = 0;
      const size_t qInt8Offset = reserve(&scratchSize, qInt8Bytes);
      const size_t kInt8Offset = reserve(&scratchSize, kInt8Bytes);
      const size_t vInt8Offset = reserve(&scratchSize, vInt8Bytes);
      const size_t qScaleOffset = reserve(&scratchSize, qScaleBytes);
      const size_t kScaleOffset = reserve(&scratchSize, kScaleBytes);
      const size_t vScaleOffset = reserve(&scratchSize, vScaleBytes);
      const size_t vMeanOffset = reserve(&scratchSize, vMeanBytes);
      const size_t blockMaskOffset = hash.masked ? reserve(&scratchSize, blockMaskBytes) : 0;
      const bool needsScratchL = !tensors[5];
      const size_t lOffset = needsScratchL ? reserve(&scratchSize, lBytes) : 0;
      auto scratch = context->request_scratch(scratchSize);
      auto lBuffer = tensors[5] ? tensors[5] : scratch;
      const size_t lBufferOffset = tensors[5] ? tensor_offsets[5] : lOffset;

      auto encodeQuantize =
      [&](NS::SharedPtr<MTL::ComputePipelineState> quantizePipeline, uint16_t threads, MTL::Buffer* source, size_t sourceOffset, size_t int8Offset, size_t scaleOffset, uint32_t scaleTiles, uint32_t heads, MTL::Buffer* seqOffsets, size_t seqOffsetsOffset) {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(quantizePipeline.get());
        encoder->useResource(source, MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (hash.is_varlen) {
          encoder->useResource(seqOffsets, MTL::ResourceUsageRead);
        }
        encoder->setBuffer(source, sourceOffset, 0);
        encoder->setBuffer(scratch, int8Offset, 1);
        encoder->setBuffer(scratch, scaleOffset, 2);
        if (hash.is_varlen) {
          encoder->setBuffer(seqOffsets, seqOffsetsOffset, 17);
        }
        encoder->dispatchThreadgroups(MTL::Size(scaleTiles, heads, batchDimension), MTL::Size(threads, 1, 1));
        command_batch->finishCommand(encoder);
      };

      encodeQuantize(quantizeQPipeline, NAInt8AttentionKernel::qQuantizeThreads, tensors[0], tensor_offsets[0], qInt8Offset, qScaleOffset, qTiles, hash.Hq, tensors[6], tensor_offsets[6]);
      encodeQuantize(quantizeKPipeline, NAInt8AttentionKernel::kvQuantizeThreads, tensors[1], tensor_offsets[1], kInt8Offset, kScaleOffset, kTiles, hash.Hk, tensors[7], tensor_offsets[7]);
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(computeVMeanPipeline.get());
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (hash.is_varlen) {
          encoder->useResource(tensors[7], MTL::ResourceUsageRead);
        }
        encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
        encoder->setBuffer(scratch, vMeanOffset, 1);
        if (hash.is_varlen) {
          encoder->setBuffer(tensors[7], tensor_offsets[7], 17);
        }
        const uint32_t meanTiles = (hash.D % 4) == 0 ? (hash.D / 4) : hash.D;
        const uint32_t meanTileBits = ccv_nnc_mfa_ceil_log2_u32(meanTiles);
        const uint32_t headBits = ccv_nnc_mfa_ceil_log2_u32(hash.Hk);
        const uint32_t mortonCodes = 1u << (meanTileBits + headBits);
        encoder->dispatchThreadgroups(
            MTL::Size(mortonCodes, 1, batchDimension),
            MTL::Size(kernel->vMeanThreads, 1, 1));
        command_batch->finishCommand(encoder);
      }
      {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(quantizeVPipeline.get());
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
        encoder->setBuffer(scratch, vInt8Offset, 1);
        encoder->setBuffer(scratch, vScaleOffset, 2);
        encoder->setBuffer(scratch, vMeanOffset, 3);
        if (hash.is_varlen) {
          encoder->useResource(tensors[7], MTL::ResourceUsageRead);
          encoder->setBuffer(tensors[7], tensor_offsets[7], 17);
        }
        encoder->dispatchThreadgroups(MTL::Size(kTiles, hash.Hk, batchDimension), MTL::Size(NAInt8AttentionKernel::kvQuantizeThreads, 1, 1));
        command_batch->finishCommand(encoder);
      }
      if (hash.masked) {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(blockMaskPipeline.get());
        encoder->setThreadgroupMemoryLength(NAInt8AttentionKernel::blockMaskThreads * sizeof(uint32_t) * 2, 0);
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 15);
        encoder->setBuffer(scratch, blockMaskOffset, 16);
        encoder->dispatchThreadgroups(
            MTL::Size(qTiles, kTiles, batch_sizes[1]),
            MTL::Size(NAInt8AttentionKernel::blockMaskThreads, 1, 1));
        command_batch->finishCommand(encoder);
      }

      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(), 0);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      encoder->useResource(lBuffer, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBuffer(scratch, qInt8Offset, 0);
      encoder->setBuffer(scratch, kInt8Offset, 1);
      encoder->setBuffer(scratch, vInt8Offset, 2);
      encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
      encoder->setBuffer(lBuffer, lBufferOffset, 4);
      encoder->setBuffer(scratch, qScaleOffset, 10);
      encoder->setBuffer(scratch, kScaleOffset, 11);
      encoder->setBuffer(scratch, vScaleOffset, 12);
      encoder->setBuffer(scratch, vMeanOffset, 14);
      if (hash.masked) {
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 15);
        encoder->setBuffer(scratch, blockMaskOffset, 16);
      }
      if (hash.is_varlen) {
        encoder->useResource(tensors[6], MTL::ResourceUsageRead);
        encoder->useResource(tensors[7], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[6], tensor_offsets[6], 17);
        encoder->setBuffer(tensors[7], tensor_offsets[7], 18);
      }
      encoder->dispatchThreadgroups(
          kernel->threadgroupsPerGrid(batchDimension, hash.R),
          MTL::Size(kernel->threadgroupSize(pipeline.get()), 1, 1));
      command_batch->finishCommand(encoder);
      return;
    }
    if (params.type == 0 && params.use_neural_accelerators) {
      NAAttentionDescriptor attentionDesc;
      attentionDesc.lowPrecisionInputs = (params.data_type != MTL::DataTypeFloat) ? true : false;
      attentionDesc.isBF16 = params.data_type == MTL::DataTypeBFloat;
      attentionDesc.lowPrecisionIntermediates = (params.data_type != MTL::DataTypeFloat && !hash.upcast) ? true : false;
      attentionDesc.matrixDimensions[0] = hash.R;
      attentionDesc.matrixDimensions[1] = hash.C;
      attentionDesc.matrixDimensions[2] = hash.D;
      attentionDesc.Hq = hash.Hq;
      attentionDesc.Hk = hash.Hk;
      attentionDesc.batchDimension = batch_sizes[0];
      attentionDesc.scale = hash.alpha;
      attentionDesc.isCausal = hash.is_causal;
      attentionDesc.masked = hash.masked;
      attentionDesc.isVarlen = hash.is_varlen;
      if (hash.masked && batch_sizes[1] > 1) {
        attentionDesc.maskBatchStride = hash.R * hash.C;
      }
      if (params.batched) {
        attentionDesc.batchStrides[AttentionOperand::Q] = hash.R * hash.D * hash.Hq;
        attentionDesc.batchStrides[AttentionOperand::K] = hash.C * hash.D * hash.Hk;
        attentionDesc.batchStrides[AttentionOperand::V] = hash.C * hash.D * hash.Hk;
        attentionDesc.batchStrides[AttentionOperand::O] = hash.R * hash.D * hash.Hq;
      }
      attentionDesc.type = AttentionKernelType::forward;
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto &shaderCache = context->kernel_cache;
      DeviceProperties dprops = DeviceProperties();
      auto pipelineValue = shaderCache.findKernel<NAAttentionKernel, NAAttentionDescriptor, NAAttentionKernelDescriptor>(attentionDesc, context->device.get(), dprops);
      pool->drain();
      auto kernel = pipelineValue->kernel;
      auto pipeline = pipelineValue->pipeline;
      auto blockMaskPipeline = pipelineValue->second;
      auto align_up =
      [&](size_t value) -> size_t {
        return (value + 255) & ~((size_t)255);
      };
      auto reserve =
      [&](size_t* total, size_t size) -> size_t {
        const size_t offset = *total;
        *total = align_up(*total + size);
        return offset;
      };
      const uint32_t qTiles = (hash.R + kernel->blockDimensions[0] - 1) / kernel->blockDimensions[0];
      const uint32_t kTiles = (hash.C + kernel->blockDimensions[1] - 1) / kernel->blockDimensions[1];
      size_t scratchSize = 0;
      const size_t blockMaskBytes =
          hash.masked ? (size_t)batch_sizes[1] * qTiles * kTiles * sizeof(uint8_t) : 0;
      const size_t blockMaskOffset = hash.masked ? reserve(&scratchSize, blockMaskBytes) : 0;
      const size_t lBytes = sizeof(float) * hash.R * hash.Hq * attentionDesc.batchDimension;
      const size_t lOffset = !tensors[5] ? reserve(&scratchSize, lBytes) : 0;
      auto scratch = scratchSize > 0 ? context->request_scratch(scratchSize) : NULL;
      if (hash.masked) {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(blockMaskPipeline.get());
        encoder->setThreadgroupMemoryLength(NAAttentionKernel::blockMaskThreads * sizeof(uint32_t) * 2, 0);
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 15);
        encoder->setBuffer(scratch, blockMaskOffset, 16);
        encoder->dispatchThreadgroups(
            MTL::Size(qTiles, kTiles, batch_sizes[1]),
            MTL::Size(NAAttentionKernel::blockMaskThreads, 1, 1));
        command_batch->finishCommand(encoder);
      }
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(pipeline.get(), attentionDesc), 0);

      // Bind the function arguments.
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      if (tensors[5]) {
        encoder->useResource(tensors[5], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      } else {
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      }
      encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
      encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
      encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
      encoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
      if (tensors[5]) {
        encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
      } else {
        encoder->setBuffer(scratch, lOffset, AttentionOperand(AttentionOperand::L).bufferIndex());
      }
      if (hash.masked) {
        encoder->useResource(tensors[4], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[4], tensor_offsets[4], 15);
        encoder->setBuffer(scratch, blockMaskOffset, 16);
      }
      if (hash.is_varlen) {
        encoder->useResource(tensors[6], MTL::ResourceUsageRead);
        encoder->useResource(tensors[7], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[6], tensor_offsets[6], 17);
        encoder->setBuffer(tensors[7], tensor_offsets[7], 18);
      }

      // Calculate the grid size.
      MTL::Size gridSize = kernel->threadgroupsPerGrid(attentionDesc);
      MTL::Size groupSize(int64_t(kernel->threadgroupSize(pipeline.get(), attentionDesc)), 1, 1);

      // Dispatch the required number of threads.
      encoder->dispatchThreadgroups(gridSize, groupSize);

      // Finish the command.
      command_batch->finishCommand(encoder);
      return;
    }
    CCV_NNC_MFA_PRECONDITION(!params.is_varlen);
    CCV_NNC_MFA_PRECONDITION(!params.is_causal);
    AttentionDescriptor attentionDesc;
    attentionDesc.lowPrecisionInputs = (params.data_type != MTL::DataTypeFloat) ? true : false;
    attentionDesc.isBF16 = params.data_type == MTL::DataTypeBFloat;
    attentionDesc.lowPrecisionIntermediates = (params.data_type != MTL::DataTypeFloat && !hash.upcast) ? true : false;
    attentionDesc.matrixDimensions[0] = hash.R;
    attentionDesc.matrixDimensions[1] = hash.C;
    attentionDesc.matrixDimensions[2] = hash.D;
    attentionDesc.transposeState[0] = false;
    attentionDesc.transposeState[1] = false;
    attentionDesc.transposeState[2] = false;
    attentionDesc.transposeState[3] = false;
    attentionDesc.Hq = hash.Hq;
    attentionDesc.Hk = hash.Hk;
    attentionDesc.batchDimension = batch_sizes[0];
    attentionDesc.scale = hash.alpha;
    if (params.batched) {
      attentionDesc.batchStrides[AttentionOperand::Q] = hash.R * hash.D * hash.Hq;
      attentionDesc.batchStrides[AttentionOperand::K] = hash.C * hash.D * hash.Hk;
      attentionDesc.batchStrides[AttentionOperand::V] = hash.C * hash.D * hash.Hk;
      attentionDesc.batchStrides[AttentionOperand::O] = hash.R * hash.D * hash.Hq;
    }
    simd::uint4 leadingDimensions;
    leadingDimensions[0] = hash.Hq * hash.D;
    leadingDimensions[1] = hash.Hk * hash.D;
    leadingDimensions[2] = hash.Hk * hash.D;
    leadingDimensions[3] = hash.Hq * hash.D;
    attentionDesc.leadingDimensions = leadingDimensions;
    // Calculate the grid size.
    auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
    if (params.type == 0) {
      attentionDesc.type = AttentionKernelType::forward;
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto &shaderCache = context->kernel_cache;
      DeviceProperties dprops = DeviceProperties();
      auto pipelineValue = shaderCache.findKernel<AttentionKernel, AttentionDescriptor, AttentionKernelDescriptor>(attentionDesc, context->device.get(), dprops);
      pool->drain();
      auto kernel = pipelineValue->kernel;
      auto pipeline = pipelineValue->pipeline;
        // Allocate a new command.
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    
      // Bind the function arguments.
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      uint64_t scratch_size = 0;
      if (attentionDesc.lowPrecisionInputs) {
        // Need scratch space for FP16 output.
        scratch_size += sizeof(float) * hash.R * hash.D * hash.Hq * attentionDesc.batchDimension;
      }
      if (!tensors[5]) {
        // Need scratch space for LSE.
        scratch_size += sizeof(float) * hash.R * hash.Hq * attentionDesc.batchDimension;
      }
      auto scratch = scratch_size > 0 ? context->request_scratch(scratch_size) : NULL;
      if (attentionDesc.lowPrecisionInputs) {
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      } else {
        encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
        if (!tensors[5]) {
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        }
      }
      if (tensors[5]) {
        encoder->useResource(tensors[5], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      }
      encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
      encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
      encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
      if (attentionDesc.lowPrecisionInputs) {
        encoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
        if (tensors[5]) {
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
        } else {
          encoder->setBuffer(scratch, sizeof(float) * hash.R * hash.D * hash.Hq * attentionDesc.batchDimension, AttentionOperand(AttentionOperand::L).bufferIndex());
        }
      } else {
        encoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
        if (tensors[5]) {
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
        } else {
          encoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
        }
      }
    
      MTL::Size gridSize
      (ceilDivide(int64_t(hash.R), kernel->blockDimensions[0]) * hash.Hq * attentionDesc.batchDimension, 1, 1);
      MTL::Size groupSize
      (int64_t(kernel->threadgroupSize), 1, 1);
    
      // Dispatch the required number of threads.
      encoder->dispatchThreadgroups(gridSize, groupSize);
    
      // Finish the command.
      command_batch->finishCommand(encoder);
      if (attentionDesc.lowPrecisionInputs) {
        // Need to dispatch to cast.
        ccv_nnc_mfa_cast_params_t cast_params = {
          .original_data_type = MTL::DataTypeFloat,
          .data_type = attentionDesc.isBF16 ? MTL::DataTypeBFloat : MTL::DataTypeHalf,
          .length = hash.R * hash.D * hash.Hq * attentionDesc.batchDimension
        };
        ccv_nnc_mfa_prepare_cast(context, cast_params);
        mtl_buffer_t* cast_tensors[3] = {
          scratch, // gradient
          tensors[3], // destination
          NULL
        };
        size_t cast_tensor_offsets[2] = {
          0,
          tensor_offsets[3]
        };
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
      }
    } else {
      const bool use_na_int8_backward =
        params.use_neural_accelerators &&
        params.use_quantized_attention &&
        hash.Hq == hash.Hk &&
        hash.D <= 128 &&
        (hash.D % 8) == 0 &&
        tensors[3] &&
        tensors[4];
      if (use_na_int8_backward) {
        NAInt8AttentionDescriptor attentionDesc;
        switch (params.data_type) {
        case MTL::DataTypeHalf:
          attentionDesc.ioPrecision = GEMMOperandPrecision::FP16;
          break;
        case MTL::DataTypeBFloat:
          attentionDesc.ioPrecision = GEMMOperandPrecision::BF16;
          break;
        case MTL::DataTypeFloat:
          attentionDesc.ioPrecision = GEMMOperandPrecision::FP32;
          break;
        default:
          CCV_NNC_MFA_PRECONDITION(false);
        }
        attentionDesc.matrixDimensions[0] = hash.R;
        attentionDesc.matrixDimensions[1] = hash.C;
        attentionDesc.matrixDimensions[2] = hash.D;
        attentionDesc.Hq = hash.Hq;
        attentionDesc.Hk = hash.Hk;
        attentionDesc.batchDimension = batch_sizes[0];
        attentionDesc.scale = hash.alpha;
        attentionDesc.lowPrecisionIntermediates =
            (params.data_type != MTL::DataTypeFloat && !hash.upcast) ? true : false;
        if (params.batched) {
          attentionDesc.batchStrides[AttentionOperand::Q] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::K] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::V] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::O] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dO] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dQ] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dK] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::dV] = hash.C * hash.D * hash.Hk;
        }
        auto forwardDesc = attentionDesc;
        forwardDesc.type = AttentionKernelType::forward;
        auto backwardQueryDesc = attentionDesc;
        backwardQueryDesc.type = AttentionKernelType::backwardQuery;
        auto backwardKeyValueDesc = attentionDesc;
        backwardKeyValueDesc.type = AttentionKernelType::backwardKeyValue;

        auto pool = NS::AutoreleasePool::alloc()->init();
        auto &shaderCache = context->kernel_cache;
        DeviceProperties dprops = DeviceProperties();
        auto forwardPipelineValue = shaderCache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor, NAInt8AttentionKernelDescriptor>(forwardDesc, context->device.get(), dprops);
        auto backwardQueryPipelineValue = shaderCache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor, NAInt8AttentionKernelDescriptor>(backwardQueryDesc, context->device.get(), dprops);
        auto backwardKeyValuePipelineValue = shaderCache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor, NAInt8AttentionKernelDescriptor>(backwardKeyValueDesc, context->device.get(), dprops);
        pool->drain();

        auto forwardKernel = forwardPipelineValue->kernel;
        auto quantizeQPipeline = backwardQueryPipelineValue->third;
        auto quantizeKPipeline = forwardPipelineValue->third;
        auto quantizeVPipeline = forwardPipelineValue->fourth;
        auto computeVMeanPipeline = forwardPipelineValue->fifth;
        auto backwardQueryKernel = backwardQueryPipelineValue->kernel;
        auto backwardQueryPipeline = backwardQueryPipelineValue->pipeline;
        auto computeDPipeline = backwardQueryPipelineValue->second;
        auto backwardKeyValueKernel = backwardKeyValuePipelineValue->kernel;
        auto backwardKeyValuePipeline = backwardKeyValuePipelineValue->pipeline;

        auto align_up =
        [&](size_t value) -> size_t {
          return (value + 255) & ~((size_t)255);
        };
        auto reserve =
        [&](size_t* total, size_t size) -> size_t {
          const size_t offset = *total;
          *total = align_up(*total + size);
          return offset;
        };

        const uint32_t batchDimension = attentionDesc.batchDimension;
        const uint32_t qTiles = (hash.R + backwardQueryKernel->qScaleTileSize - 1) / backwardQueryKernel->qScaleTileSize;
        const uint32_t kTiles = (hash.C + forwardKernel->kvScaleTileSize - 1) / forwardKernel->kvScaleTileSize;
        const uint32_t qBatchStride = hash.R * hash.D * hash.Hq;
        const uint32_t kvBatchStride = hash.C * hash.D * hash.Hk;
        const uint32_t qScaleBatchStride = hash.Hq * qTiles;
        const uint32_t kvScaleBatchStride = hash.Hk * kTiles;
        const size_t qInt8Bytes = (size_t)batchDimension * qBatchStride * sizeof(int8_t);
        const size_t kInt8Bytes = (size_t)batchDimension * kvBatchStride * sizeof(int8_t);
        const size_t vInt8Bytes = (size_t)batchDimension * kvBatchStride * sizeof(int8_t);
        const size_t dOInt8Bytes = (size_t)batchDimension * qBatchStride * sizeof(int8_t);
        const size_t qScaleBytes = (size_t)batchDimension * qScaleBatchStride * sizeof(float);
        const size_t kScaleBytes = (size_t)batchDimension * kvScaleBatchStride * sizeof(float);
        const size_t vScaleBytes = (size_t)batchDimension * kvScaleBatchStride * sizeof(float);
        const size_t dOScaleBytes = (size_t)batchDimension * qScaleBatchStride * sizeof(float);
        const size_t vMeanBytes = (size_t)batchDimension * hash.Hk * hash.D * sizeof(float);
        const GEMMOperandPrecision dPrecision =
            attentionDesc.lowPrecisionIntermediates ?
            (attentionDesc.ioPrecision == GEMMOperandPrecision::FP32 ? GEMMOperandPrecision::FP32 : GEMMOperandPrecision::BF16) :
            GEMMOperandPrecision::FP32;
        const size_t dBytes = (size_t)batchDimension * hash.Hq * hash.R * dPrecision.size();
        size_t scratchSize = 0;
        const size_t qInt8Offset = reserve(&scratchSize, qInt8Bytes);
        const size_t kInt8Offset = reserve(&scratchSize, kInt8Bytes);
        const size_t vInt8Offset = reserve(&scratchSize, vInt8Bytes);
        const size_t dOInt8Offset = reserve(&scratchSize, dOInt8Bytes);
        const size_t qScaleOffset = reserve(&scratchSize, qScaleBytes);
        const size_t kScaleOffset = reserve(&scratchSize, kScaleBytes);
        const size_t vScaleOffset = reserve(&scratchSize, vScaleBytes);
        const size_t dOScaleOffset = reserve(&scratchSize, dOScaleBytes);
        const size_t vMeanOffset = reserve(&scratchSize, vMeanBytes);
        const size_t dOffset = reserve(&scratchSize, dBytes);
        auto scratch = context->request_scratch(scratchSize);

        auto encodeQuantize =
        [&](NS::SharedPtr<MTL::ComputePipelineState> quantizePipeline, uint16_t threads, MTL::Buffer* source, size_t sourceOffset, size_t int8Offset, size_t scaleOffset, uint32_t scaleTiles, uint32_t heads) {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(quantizePipeline.get());
          encoder->useResource(source, MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->setBuffer(source, sourceOffset, 0);
          encoder->setBuffer(scratch, int8Offset, 1);
          encoder->setBuffer(scratch, scaleOffset, 2);
          encoder->dispatchThreadgroups(MTL::Size(scaleTiles, heads, batchDimension), MTL::Size(threads, 1, 1));
          command_batch->finishCommand(encoder);
        };

        encodeQuantize(quantizeQPipeline, NAInt8AttentionKernel::qQuantizeThreads, tensors[0], tensor_offsets[0], qInt8Offset, qScaleOffset, qTiles, hash.Hq);
        encodeQuantize(quantizeKPipeline, NAInt8AttentionKernel::kvQuantizeThreads, tensors[1], tensor_offsets[1], kInt8Offset, kScaleOffset, kTiles, hash.Hk);
        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(computeVMeanPipeline.get());
          encoder->useResource(tensors[2], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
          encoder->setBuffer(scratch, vMeanOffset, 1);
          const uint32_t meanTiles = (hash.D % 4) == 0 ? (hash.D / 4) : hash.D;
          const uint32_t meanTileBits = ccv_nnc_mfa_ceil_log2_u32(meanTiles);
          const uint32_t headBits = ccv_nnc_mfa_ceil_log2_u32(hash.Hk);
          const uint32_t mortonCodes = 1u << (meanTileBits + headBits);
          encoder->dispatchThreadgroups(
              MTL::Size(mortonCodes, 1, batchDimension),
              MTL::Size(forwardKernel->vMeanThreads, 1, 1));
          command_batch->finishCommand(encoder);
        }
        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(quantizeVPipeline.get());
          encoder->useResource(tensors[2], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[2], tensor_offsets[2], 0);
          encoder->setBuffer(scratch, vInt8Offset, 1);
          encoder->setBuffer(scratch, vScaleOffset, 2);
          encoder->setBuffer(scratch, vMeanOffset, 3);
          encoder->dispatchThreadgroups(MTL::Size(kTiles, hash.Hk, batchDimension), MTL::Size(NAInt8AttentionKernel::kvQuantizeThreads, 1, 1));
          command_batch->finishCommand(encoder);
        }
        encodeQuantize(quantizeQPipeline, NAInt8AttentionKernel::qQuantizeThreads, tensors[5], tensor_offsets[5], dOInt8Offset, dOScaleOffset, qTiles, hash.Hq);

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(computeDPipeline.get());
          encoder->useResource(tensors[3], MTL::ResourceUsageRead);
          encoder->useResource(tensors[5], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
          encoder->setBuffer(tensors[5], tensor_offsets[5], 6);
          encoder->setBuffer(scratch, dOffset, 5);
          encoder->setBuffer(scratch, vMeanOffset, 14);
          encoder->dispatchThreadgroups(MTL::Size((uint64_t)hash.R * hash.Hq, 1, batchDimension), MTL::Size(NAInt8AttentionKernel::computeDThreads, 1, 1));
          command_batch->finishCommand(encoder);
        }

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(backwardQueryPipeline.get());
          encoder->setThreadgroupMemoryLength(backwardQueryKernel->threadgroupMemoryAllocation(), 0);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->useResource(tensors[4], MTL::ResourceUsageRead);
          encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
          encoder->setBuffer(scratch, qInt8Offset, 0);
          encoder->setBuffer(scratch, kInt8Offset, 1);
          encoder->setBuffer(scratch, vInt8Offset, 2);
          encoder->setBuffer(tensors[4], tensor_offsets[4], 4);
          encoder->setBuffer(scratch, dOffset, 5);
          encoder->setBuffer(scratch, dOInt8Offset, 6);
          encoder->setBuffer(tensors[6], tensor_offsets[6], 9);
          encoder->setBuffer(scratch, qScaleOffset, 10);
          encoder->setBuffer(scratch, kScaleOffset, 11);
          encoder->setBuffer(scratch, vScaleOffset, 12);
          encoder->setBuffer(scratch, dOScaleOffset, 13);
          encoder->dispatchThreadgroups(
              backwardQueryKernel->threadgroupsPerGrid(batchDimension, hash.R),
              MTL::Size(backwardQueryKernel->threadgroupSize(backwardQueryPipeline.get()), 1, 1));
          command_batch->finishCommand(encoder);
        }

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(backwardKeyValuePipeline.get());
          encoder->setThreadgroupMemoryLength(backwardKeyValueKernel->threadgroupMemoryAllocation(), 0);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->useResource(tensors[4], MTL::ResourceUsageRead);
          encoder->useResource(tensors[7], MTL::ResourceUsageWrite);
          encoder->useResource(tensors[8], MTL::ResourceUsageWrite);
          encoder->setBuffer(scratch, qInt8Offset, 0);
          encoder->setBuffer(scratch, kInt8Offset, 1);
          encoder->setBuffer(scratch, vInt8Offset, 2);
          encoder->setBuffer(tensors[4], tensor_offsets[4], 4);
          encoder->setBuffer(scratch, dOffset, 5);
          encoder->setBuffer(scratch, dOInt8Offset, 6);
          encoder->setBuffer(tensors[8], tensor_offsets[8], 7);
          encoder->setBuffer(tensors[7], tensor_offsets[7], 8);
          encoder->setBuffer(scratch, qScaleOffset, 10);
          encoder->setBuffer(scratch, kScaleOffset, 11);
          encoder->setBuffer(scratch, vScaleOffset, 12);
          encoder->setBuffer(scratch, dOScaleOffset, 13);
          encoder->dispatchThreadgroups(
              backwardKeyValueKernel->threadgroupsPerGrid(batchDimension, hash.C),
              MTL::Size(backwardKeyValueKernel->threadgroupSize(backwardKeyValuePipeline.get()), 1, 1));
          command_batch->finishCommand(encoder);
        }
        return;
      }
      const bool use_na_backward =
        params.use_neural_accelerators &&
        hash.Hq == hash.Hk &&
        hash.D <= 128 &&
        (hash.D % 8) == 0;
      if (use_na_backward) {
        NAAttentionDescriptor attentionDesc;
        attentionDesc.lowPrecisionInputs = (params.data_type != MTL::DataTypeFloat) ? true : false;
        attentionDesc.isBF16 = params.data_type == MTL::DataTypeBFloat;
        attentionDesc.lowPrecisionIntermediates =
          (params.data_type != MTL::DataTypeFloat && !hash.upcast) ? true : false;
        attentionDesc.matrixDimensions[0] = hash.R;
        attentionDesc.matrixDimensions[1] = hash.C;
        attentionDesc.matrixDimensions[2] = hash.D;
        attentionDesc.Hq = hash.Hq;
        attentionDesc.Hk = hash.Hk;
        attentionDesc.batchDimension = batch_sizes[0];
        attentionDesc.scale = hash.alpha;
        if (params.batched) {
          attentionDesc.batchStrides[AttentionOperand::Q] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::K] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::V] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::O] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dO] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dQ] = hash.R * hash.D * hash.Hq;
          attentionDesc.batchStrides[AttentionOperand::dK] = hash.C * hash.D * hash.Hk;
          attentionDesc.batchStrides[AttentionOperand::dV] = hash.C * hash.D * hash.Hk;
        }
        auto backwardQueryDesc = attentionDesc;
        backwardQueryDesc.type = AttentionKernelType::backwardQuery;
        auto backwardKeyValueDesc = attentionDesc;
        backwardKeyValueDesc.type = AttentionKernelType::backwardKeyValue;
        auto pool = NS::AutoreleasePool::alloc()->init();
        auto &shaderCache = context->kernel_cache;
        DeviceProperties dprops = DeviceProperties();
        auto backwardQueryPipelineValue = shaderCache.findKernel<NAAttentionKernel, NAAttentionDescriptor, NAAttentionKernelDescriptor>(backwardQueryDesc, context->device.get(), dprops);
        auto backwardKeyValuePipelineValue = shaderCache.findKernel<NAAttentionKernel, NAAttentionDescriptor, NAAttentionKernelDescriptor>(backwardKeyValueDesc, context->device.get(), dprops);
        pool->drain();

        auto backwardQueryKernel = backwardQueryPipelineValue->kernel;
        auto computeDPipeline = backwardQueryPipelineValue->second;
        auto backwardQueryPipeline = backwardQueryPipelineValue->pipeline;
        auto backwardKeyValueKernel = backwardKeyValuePipelineValue->kernel;
        auto backwardKeyValuePipeline = backwardKeyValuePipelineValue->pipeline;

        const size_t dBytes = sizeof(float) * hash.R * hash.Hq * attentionDesc.batchDimension;
        size_t scratchSize = dBytes;
        const size_t dOffset = 0;
        auto scratch = context->request_scratch(scratchSize);

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(computeDPipeline.get());
          encoder->useResource(tensors[3], MTL::ResourceUsageRead);
          encoder->useResource(tensors[5], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::dO).bufferIndex());
          encoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
          encoder->dispatchThreadgroups(MTL::Size((uint64_t)hash.R * hash.Hq, 1, attentionDesc.batchDimension), MTL::Size(NAAttentionKernel::computeDThreads, 1, 1));
          command_batch->finishCommand(encoder);
        }

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(backwardQueryPipeline.get());
          encoder->setThreadgroupMemoryLength(backwardQueryKernel->threadgroupMemoryAllocation(backwardQueryPipeline.get(), backwardQueryDesc), 0);
          encoder->useResource(tensors[0], MTL::ResourceUsageRead);
          encoder->useResource(tensors[1], MTL::ResourceUsageRead);
          encoder->useResource(tensors[2], MTL::ResourceUsageRead);
          encoder->useResource(tensors[4], MTL::ResourceUsageRead);
          encoder->useResource(tensors[5], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
          encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
          encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
          encoder->setBuffer(tensors[4], tensor_offsets[4], AttentionOperand(AttentionOperand::L).bufferIndex());
          encoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::dO).bufferIndex());
          encoder->setBuffer(tensors[6], tensor_offsets[6], AttentionOperand(AttentionOperand::dQ).bufferIndex());
          encoder->dispatchThreadgroups(backwardQueryKernel->threadgroupsPerGrid(backwardQueryDesc), MTL::Size(backwardQueryKernel->threadgroupSize(backwardQueryPipeline.get(), backwardQueryDesc), 1, 1));
          command_batch->finishCommand(encoder);
        }

        {
          auto encoder = command_batch->startCommand();
          encoder->setComputePipelineState(backwardKeyValuePipeline.get());
          encoder->setThreadgroupMemoryLength(backwardKeyValueKernel->threadgroupMemoryAllocation(backwardKeyValuePipeline.get(), backwardKeyValueDesc), 0);
          encoder->useResource(tensors[0], MTL::ResourceUsageRead);
          encoder->useResource(tensors[1], MTL::ResourceUsageRead);
          encoder->useResource(tensors[2], MTL::ResourceUsageRead);
          encoder->useResource(tensors[4], MTL::ResourceUsageRead);
          encoder->useResource(tensors[5], MTL::ResourceUsageRead);
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
          encoder->useResource(tensors[7], MTL::ResourceUsageWrite);
          encoder->useResource(tensors[8], MTL::ResourceUsageWrite);
          encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
          encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
          encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
          encoder->setBuffer(tensors[4], tensor_offsets[4], AttentionOperand(AttentionOperand::L).bufferIndex());
          encoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::dO).bufferIndex());
          encoder->setBuffer(tensors[8], tensor_offsets[8], AttentionOperand(AttentionOperand::dV).bufferIndex());
          encoder->setBuffer(tensors[7], tensor_offsets[7], AttentionOperand(AttentionOperand::dK).bufferIndex());
          encoder->dispatchThreadgroups(backwardKeyValueKernel->threadgroupsPerGrid(backwardKeyValueDesc), MTL::Size(backwardKeyValueKernel->threadgroupSize(backwardKeyValuePipeline.get(), backwardKeyValueDesc), 1, 1));
          command_batch->finishCommand(encoder);
        }
        return;
      }
      if (params.batched) {
        attentionDesc.batchStrides[AttentionOperand::dQ] = hash.R * hash.D * hash.Hq;
        attentionDesc.batchStrides[AttentionOperand::dK] = hash.C * hash.D * hash.Hq;
        attentionDesc.batchStrides[AttentionOperand::dV] = hash.C * hash.D * hash.Hq; // We use Hq, in case supporting grouped kv later, we will do collection in another kernel.
        attentionDesc.batchStrides[AttentionOperand::dO] = hash.R * hash.D * hash.Hq;
      }
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto &shaderCache = context->kernel_cache;
      DeviceProperties dprops = DeviceProperties();
      attentionDesc.type = AttentionKernelType::backwardQuery;
      auto backwardQueryPipelineValue = shaderCache.findKernel<AttentionKernel, AttentionDescriptor, AttentionKernelDescriptor>(attentionDesc, context->device.get(), dprops);
      attentionDesc.type = AttentionKernelType::backwardKeyValue;
      auto backwardKeyValuePipelineValue = shaderCache.findKernel<AttentionKernel, AttentionDescriptor, AttentionKernelDescriptor>(attentionDesc, context->device.get(), dprops);
      pool->drain();
      auto backwardQueryKernel = backwardQueryPipelineValue->kernel;
      auto backwardQueryPipeline = backwardQueryPipelineValue->pipeline;
      auto backwardKeyValueKernel = backwardKeyValuePipelineValue->kernel;
      auto backwardKeyValuePipeline = backwardKeyValuePipelineValue->pipeline;

      uint64_t scratch_size = 0;
      if (attentionDesc.lowPrecisionInputs) {
        // Need scratch space for FP16 output.
        scratch_size += sizeof(float) * (hash.R + hash.C * 2) * hash.D * hash.Hq * attentionDesc.batchDimension;
      }
      // Need scratch space for D.
      scratch_size += sizeof(float) * hash.R * hash.Hq * attentionDesc.batchDimension;
      auto scratch = context->request_scratch(scratch_size);

      // Allocate a new command.
      auto backwardQueryEncoder = command_batch->startCommand();
      backwardQueryEncoder->setComputePipelineState(backwardQueryPipeline.get());
      backwardQueryEncoder->setThreadgroupMemoryLength(backwardQueryKernel->threadgroupMemoryAllocation, 0);

      // Bind the function arguments.
      backwardQueryEncoder->useResource(tensors[0], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(tensors[1], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(tensors[2], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(tensors[3], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(tensors[4], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(tensors[5], MTL::ResourceUsageRead);
      backwardQueryEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      if (!attentionDesc.lowPrecisionInputs) {
        backwardQueryEncoder->useResource(tensors[6], MTL::ResourceUsageWrite);
      }

      backwardQueryEncoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
      backwardQueryEncoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
      backwardQueryEncoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
      backwardQueryEncoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
      backwardQueryEncoder->setBuffer(tensors[4], tensor_offsets[4], AttentionOperand(AttentionOperand::L).bufferIndex());
      backwardQueryEncoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::dO).bufferIndex());
      if (attentionDesc.lowPrecisionInputs) {
        backwardQueryEncoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::dQ).bufferIndex());
        backwardQueryEncoder->setBuffer(scratch, sizeof(float) * (hash.R + hash.C * 2) * hash.D * hash.Hq * attentionDesc.batchDimension, AttentionOperand(AttentionOperand::D).bufferIndex());
      } else {
        backwardQueryEncoder->setBuffer(tensors[6], tensor_offsets[6], AttentionOperand(AttentionOperand::dQ).bufferIndex());
        backwardQueryEncoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::D).bufferIndex());
      }

      MTL::Size backwardQueryGridSize
      (ceilDivide(int64_t(hash.R), backwardQueryKernel->blockDimensions[0]) * hash.Hq * attentionDesc.batchDimension, 1, 1);
      MTL::Size backwardQueryGroupSize
      (int64_t(backwardQueryKernel->threadgroupSize), 1, 1);

      // Dispatch the required number of threads.
      backwardQueryEncoder->dispatchThreadgroups(backwardQueryGridSize, backwardQueryGroupSize);

      // Finish the command.
      command_batch->finishCommand(backwardQueryEncoder);

      // Allocate a new command.
      auto backwardKeyValueEncoder = command_batch->startCommand();
      backwardKeyValueEncoder->setComputePipelineState(backwardKeyValuePipeline.get());
      backwardKeyValueEncoder->setThreadgroupMemoryLength(backwardKeyValueKernel->threadgroupMemoryAllocation, 0);

      // Bind the function arguments.
      backwardKeyValueEncoder->useResource(tensors[0], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(tensors[1], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(tensors[2], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(tensors[3], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(tensors[4], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(tensors[5], MTL::ResourceUsageRead);
      backwardKeyValueEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      if (!attentionDesc.lowPrecisionInputs) {
        backwardKeyValueEncoder->useResource(tensors[7], MTL::ResourceUsageWrite);
        backwardKeyValueEncoder->useResource(tensors[8], MTL::ResourceUsageWrite);
      }

      backwardKeyValueEncoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
      backwardKeyValueEncoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
      backwardKeyValueEncoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
      backwardKeyValueEncoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
      backwardKeyValueEncoder->setBuffer(tensors[4], tensor_offsets[4], AttentionOperand(AttentionOperand::L).bufferIndex());
      backwardKeyValueEncoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::dO).bufferIndex());
      if (attentionDesc.lowPrecisionInputs) {
        backwardKeyValueEncoder->setBuffer(scratch, sizeof(float) * hash.R * hash.D * hash.Hq * attentionDesc.batchDimension, AttentionOperand(AttentionOperand::dK).bufferIndex());
        backwardKeyValueEncoder->setBuffer(scratch, sizeof(float) * (hash.R + hash.C) * hash.D * hash.Hq * attentionDesc.batchDimension, AttentionOperand(AttentionOperand::dV).bufferIndex());
        backwardKeyValueEncoder->setBuffer(scratch, sizeof(float) * (hash.R + hash.C * 2) * hash.D * hash.Hq * attentionDesc.batchDimension, AttentionOperand(AttentionOperand::D).bufferIndex());
      } else {
        backwardKeyValueEncoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::D).bufferIndex());
        backwardKeyValueEncoder->setBuffer(tensors[7], tensor_offsets[7], AttentionOperand(AttentionOperand::dK).bufferIndex());
        backwardKeyValueEncoder->setBuffer(tensors[8], tensor_offsets[8], AttentionOperand(AttentionOperand::dV).bufferIndex());
      }

      MTL::Size backwardKeyValueGridSize
      (ceilDivide(int64_t(hash.C), backwardKeyValueKernel->blockDimensions[0]) * hash.Hq * attentionDesc.batchDimension, 1, 1);
      MTL::Size backwardKeyValueGroupSize
      (int64_t(backwardKeyValueKernel->threadgroupSize), 1, 1);

      // Dispatch the required number of threads.
      backwardKeyValueEncoder->dispatchThreadgroups(backwardKeyValueGridSize, backwardKeyValueGroupSize);
    
      // Finish the command.
      command_batch->finishCommand(backwardKeyValueEncoder);

      if (attentionDesc.lowPrecisionInputs) {
        // Need to dispatch to cast.
        ccv_nnc_mfa_cast_params_t cast_params = {
          .original_data_type = MTL::DataTypeFloat,
          .data_type = attentionDesc.isBF16 ? MTL::DataTypeBFloat : MTL::DataTypeHalf,
          .length = hash.R * hash.D * hash.Hq * attentionDesc.batchDimension
        };
        ccv_nnc_mfa_prepare_cast(context, cast_params);
        mtl_buffer_t* cast_tensors[3] = {
          scratch, // gradient
          tensors[6], // destination
          NULL
        };
        size_t cast_tensor_offsets[2] = {
          0,
          tensor_offsets[6]
        };
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
        cast_params.length = hash.C * hash.D * hash.Hq * attentionDesc.batchDimension;
        ccv_nnc_mfa_prepare_cast(context, cast_params);
        cast_tensors[1] = tensors[7];
        cast_tensor_offsets[0] = sizeof(float) * hash.R * hash.D * hash.Hq * attentionDesc.batchDimension;
        cast_tensor_offsets[1] = tensor_offsets[7];
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
        cast_tensors[1] = tensors[8];
        cast_tensor_offsets[0] = sizeof(float) * (hash.R + hash.C) * hash.D * hash.Hq * attentionDesc.batchDimension;
        cast_tensor_offsets[1] = tensor_offsets[8];
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
      }
    }
    return;
  }

  if (params.type != 0) { // Cannot run backward pass for old kernels.
    CCV_NNC_MFA_PRECONDITION(false);
  }

  auto iterator = context->attention_cache.map.find(hash);
  if (iterator == context->attention_cache.map.end()) {
    mfa::precondition_failure("Attention hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  uint16_t data_type_size = 0;
  switch (params.data_type) {
    case MTL::DataTypeHalf: {
      data_type_size = 2;
      break;
    }
    case MTL::DataTypeFloat: {
      data_type_size = 4;
      break;
    }
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
  }
  
  // Simple broadcasting rules; not yet support for NumPy broadcasting rules.
  simd::ushort2 num_batch_dims(0);
  simd::ulong2 batch_sizes(1);
  if (params.batched) {
    for (uint16_t operand = 0; operand < 2; ++operand) {
      uint32_t* batch_dims;
      if (operand == 0) {
        batch_dims = params.batch_dims_q;
      } else if (operand == 1) {
        batch_dims = params.batch_dims_mask;
      }
      
      for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
        if (batch_dims[i] == 0) {
          break;
        }
        num_batch_dims[operand] += 1;
        batch_sizes[operand] *= batch_dims[i];
      }
      
      bool dims_match_q = true;
      if (num_batch_dims[0] != num_batch_dims[operand]) {
        dims_match_q = false;
      } else if (batch_sizes[0] != batch_sizes[operand]) {
        dims_match_q = false;
      } else {
        for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
          if (params.batch_dims_q[i] != batch_dims[i]) {
            dims_match_q = false;
          }
        }
      }
      
      if (!dims_match_q) {
        CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
      }
    }
    
    uint64_t byte_stride_mask = 0;
    uint64_t byte_stride_block_mask = 0;
    if (batch_sizes[1] > 1) {
      byte_stride_mask = hash.R * hash.C * data_type_size;
      auto grid_size = pipeline->grid_size;
      byte_stride_block_mask = grid_size.width * grid_size.height * 1;
    }
    
    if (hash.masked) {
      simd::ulong4 matrix_offsets[batch_sizes[0]];
      for (int i = 0; i < batch_sizes[0]; ++i) {
        matrix_offsets[i] = simd::ulong4 {
          i * byte_stride_mask,
          i * byte_stride_block_mask,
          0,
          0,
        };
      }
      encoder->setBytes(matrix_offsets, batch_sizes[0] * 32, 10);
    }
  }
  
  if (params.masked) {
    encoder->setComputePipelineState(pipeline->generate_block_mask_pso.get());
    encoder->setThreadgroupMemoryLength(48, 0);
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->setBuffer(tensors[4], tensor_offsets[4], 12);
    
    auto grid_size = pipeline->grid_size;
    auto scratch_size = grid_size.width * grid_size.height * 1;
    grid_size.depth = batch_sizes[1];
    scratch_size *= batch_sizes[1];
    
    auto scratch = context->request_scratch(scratch_size);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 13);
    encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  }
  
  encoder->setComputePipelineState(pipeline->attention_pso.get());
  encoder->setThreadgroupMemoryLength(pipeline->threadgroup_memory_length, 0);
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  for (int i = 0; i < 4; ++i) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], i);
  }
  // grouped query is on, need to pass query_offsets
  if (params.Hq != params.Hk) {
    CCV_NNC_MFA_PRECONDITION(params.Hq > params.Hk);
    CCV_NNC_MFA_PRECONDITION((params.Hq % params.Hk) == 0);
    uint32_t query_offsets[params.Hq * 4];
    const int h_h_k_ratio = params.Hq / params.Hk;
    for (int i = 0; i < params.Hq; i++) {
      query_offsets[i * 4] = i;
      query_offsets[i * 4 + 1] = i / h_h_k_ratio;
      query_offsets[i * 4 + 2] = i / h_h_k_ratio;
      query_offsets[i * 4 + 3] = i;
    }
    encoder->setBytes(query_offsets, params.Hq * 16, 11);
  }
  
  auto grid_size = pipeline->grid_size;
  grid_size.height = params.Hq;
  grid_size.depth = batch_sizes[0];
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::attention::hash::hash(ccv_nnc_mfa_attention_params_t params) {
  data_type = params.data_type;
  R = params.R;
  C = params.C;
  Hq = params.Hq;
  Hk = params.Hk;
  D = params.D;
  Q_trans = params.Q_trans;
  K_trans = params.K_trans;
  V_trans = params.V_trans;
  O_trans = params.O_trans;
  alpha = params.alpha;
  batched = params.batched;
  masked = params.masked;
  is_causal = params.is_causal;
  is_varlen = params.is_varlen;
  upcast = params.upcast;
  type = params.type;
  use_quantized_attention = params.use_quantized_attention;
}

bool mfa::attention::hash::operator==(const mfa::attention::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (R == hash.R) &&
  (C == hash.C) &&
  (Hq == hash.Hq) &&
  (Hk == hash.Hk) &&
  (D == hash.D) &&
  (Q_trans == hash.Q_trans) &&
  (K_trans == hash.K_trans) &&
  (V_trans == hash.V_trans) &&
  (O_trans == hash.O_trans) &&
  (alpha == hash.alpha) &&
  (batched == hash.batched) &&
  (masked == hash.masked) &&
  (is_causal == hash.is_causal) &&
  (is_varlen == hash.is_varlen) &&
  (upcast == hash.upcast) &&
  (type == hash.type) &&
  (use_quantized_attention == hash.use_quantized_attention);
}

std::ostream& operator<<(std::ostream& os, const mfa::attention::hash& hash) {
  os << "mfa::attention::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .R = " << hash.R << ',';
  os << " .C = " << hash.C << ',';
  os << " .Hq = " << hash.Hq << ',';
  os << " .Hk = " << hash.Hk << ',';
  os << " .D = " << hash.D << ',';
  os << " .Q_trans = " << bool(hash.Q_trans) << ',';
  os << " .K_trans = " << bool(hash.K_trans) << ',';
  os << " .V_trans = " << bool(hash.V_trans) << ',';
  os << " .O_trans = " << bool(hash.O_trans) << ',';
  os << " .alpha = " << double(hash.alpha) << ',';
  os << " .batched = " << bool(hash.batched) << ',';
  os << " .masked = " << bool(hash.masked) << ", ";
  os << " .is_causal = " << bool(hash.is_causal) << ", ";
  os << " .is_varlen = " << bool(hash.is_varlen) << ", ";
  os << " .upcast = " << bool(hash.upcast) << " ";
  os << " .use_quantized_attention = " << bool(hash.use_quantized_attention) << " ";
  os << " .type = " << hash.type << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::attention::hash>::operator()(const mfa::attention::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.R, hash.C }));
  combine_64(seed, pack_64(simd::uint2 { hash.Hq, hash.Hk }));
  combine_64(seed, pack_64(simd::uint2 { hash.D, pack_32(simd::uchar4 { hash.Q_trans, hash.K_trans, hash.V_trans, hash.O_trans })}));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.alpha), pack_32(simd::uchar4 { hash.batched, hash.masked, hash.is_causal, hash.is_varlen })}));
  combine_32(seed, hash.type);
  combine_32(seed, hash.use_quantized_attention);
  combine_32(seed, hash.upcast);
  return seed;
}

mfa::attention::pipeline::pipeline(mfa::context* context, mfa::attention::hash hash) {
  if (!hash.masked) { // Avoid pipeline setup if we use kernels.
    return;
  }
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&hash.R, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&hash.C, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&hash.Hq, MTL::DataTypeUInt, 2);
  constants->setConstantValue(&hash.D, MTL::DataTypeUInt, 3);
  constants->setConstantValue(&hash.Q_trans, MTL::DataTypeBool, 10);
  constants->setConstantValue(&hash.K_trans, MTL::DataTypeBool, 11);
  constants->setConstantValue(&hash.V_trans, MTL::DataTypeBool, 12);
  constants->setConstantValue(&hash.O_trans, MTL::DataTypeBool, 13);
  constants->setConstantValue(&hash.alpha, MTL::DataTypeFloat, 20);
  constants->setConstantValue(&hash.data_type, MTL::DataTypeUInt, 30);
  constants->setConstantValue(&hash.batched, MTL::DataTypeBool, 100);
  constants->setConstantValue(&hash.masked, MTL::DataTypeBool, 50000);
  constants->setConstantValue(&hash.upcast, MTL::DataTypeBool, 114);
  
  {
    bool block_sparse = hash.masked;
    bool triangular = false;
    bool forward = true;
    bool backward = false;
    bool generate_block_mask = false;
    bool grouped_query = (hash.Hq != hash.Hk);
    uint32_t H_Hk_ratio = hash.Hq / hash.Hk;
    constants->setConstantValue(&H_Hk_ratio, MTL::DataTypeUInt, 4);
    constants->setConstantValue(&block_sparse, MTL::DataTypeBool, 102);
    constants->setConstantValue(&triangular, MTL::DataTypeBool, 103);
    constants->setConstantValue(&forward, MTL::DataTypeBool, 110);
    constants->setConstantValue(&backward, MTL::DataTypeBool, 111);
    constants->setConstantValue(&generate_block_mask, MTL::DataTypeBool, 112);
    constants->setConstantValue(&grouped_query, MTL::DataTypeBool, 113);
  }

  simd::ulong4 garbage(0);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 101);
  
  uint16_t R_simd;
  uint16_t C_simd;
  uint16_t R_splits;
  bool fuse_async_loads = false;
  if (hash.data_type == MTL::DataTypeFloat) {
    R_simd = 8;
    C_simd = 32;
    R_splits = 4;
  } else {
    uint32_t D = hash.D;
    if (hash.masked) {
      if (D <= 16) {
        R_simd = 16;
        C_simd = 64;
        R_splits = 4;
      } else if (D <= 24) {
        R_simd = 8;
        C_simd = 64;
        R_splits = 8;
      } else if (D <= 80) {
        R_simd = 8;
        C_simd = 64;
        R_splits = 4;
      } else {
        R_simd = 8;
        C_simd = 32;
        R_splits = 4;
      }
    } else {
      R_simd = 8;
      R_splits = 8;
      
      if (D <= 8) {
        R_simd = 16;
        C_simd = 64;
      } else if (D <= 16) {
        C_simd = 72;
        fuse_async_loads = true;
      } else if (D <= 24) {
        C_simd = 56;
        fuse_async_loads = true;
      } else if (D <= 56) {
        C_simd = 64;
      } else if (D <= 64) {
        C_simd = 40;
        fuse_async_loads = true;
      } else if (D <= 96) {
        C_simd = 64;
      } else if (D <= 304) {
        C_simd = 32;
        R_splits = 4;
      } else {
        C_simd = 40;
        R_splits = 8;
      }
    }
  }
  
  constants->setConstantValue(&R_simd, MTL::DataTypeUShort, 200);
  constants->setConstantValue(&C_simd, MTL::DataTypeUShort, 201);
  constants->setConstantValue(&R_splits, MTL::DataTypeUShort, 210);
  constants->setConstantValue(&fuse_async_loads, MTL::DataTypeBool, 213);
  
  uint16_t data_type_size = UINT16_MAX;
  switch (hash.data_type) {
    case MTL::DataTypeHalf: {
      data_type_size = 2;
      break;
    }
    case MTL::DataTypeFloat: {
      data_type_size = 4;
      break;
    }
    default: {
      CCV_NNC_MFA_PRECONDITION(false)
      break;
    }
  }
  
  uint16_t D_simd = (hash.D + 7) / 8 * 8;
  uint16_t R_group = R_simd * R_splits;
  uint16_t R_block_dim = R_group;
  uint16_t C_block_dim = C_simd;
  uint16_t D_block_dim = D_simd;
  std::function<void(uint16_t*, NS::UInteger)> set_bank_offset = [=](uint16_t* dim, NS::UInteger index) {
    CCV_NNC_MFA_PRECONDITION(*dim % 8 == 0);
    
    uint16_t dim_bytes = *dim * data_type_size;
    uint16_t dim_bytes_modulo = dim_bytes % 64;
    if (dim_bytes_modulo == 16 || dim_bytes_modulo == 48) {
      return;
    } else if (dim_bytes_modulo == 0 || dim_bytes_modulo == 32) {
      constexpr uint16_t bank_offset_bytes = 16;
      uint16_t bank_offset = bank_offset_bytes / data_type_size;
      constants->setConstantValue(&bank_offset, MTL::DataTypeUShort, index);
      *dim += bank_offset;
    } else {
      CCV_NNC_MFA_PRECONDITION(false)
    }
  };
  set_bank_offset(&R_block_dim, 220);
  set_bank_offset(&C_block_dim, 221);
  set_bank_offset(&D_block_dim, 222);
  
  uint16_t Q_block_length;
  uint16_t K_block_length;
  uint16_t V_block_length;
  uint16_t O_block_length;
  if (hash.Q_trans) {
    Q_block_length = D_simd * R_block_dim;
  } else {
    Q_block_length = R_group * D_block_dim;
  }
  if (hash.K_trans) {
    K_block_length = C_simd * D_block_dim;
  } else {
    K_block_length = D_simd * C_block_dim;
  }
  if (hash.V_trans) {
    V_block_length = D_simd * C_block_dim;
  } else {
    V_block_length = C_simd * D_block_dim;
  }
  if (hash.O_trans) {
    O_block_length = D_simd * R_block_dim;
  } else {
    O_block_length = R_group * D_block_dim;
  }
  
  uint16_t block_elements;
  if (fuse_async_loads) {
    block_elements = K_block_length + V_block_length;
  } else {
    block_elements = std::max(K_block_length, V_block_length);
  }
  block_elements = std::max(block_elements, Q_block_length);
  block_elements = std::max(block_elements, O_block_length);
  this->threadgroup_memory_length = block_elements * data_type_size;
  
  std::function<size_t(size_t, uint16_t)> ceil_divide = [](size_t original, uint16_t granularity) {
    return (original + size_t(granularity) - 1) / size_t(granularity);
  };
  this->grid_size = MTL::Size(ceil_divide(hash.R, R_group), ceil_divide(hash.C, C_simd), 1);
  this->group_size = MTL::Size(32 * R_splits, 1, 1);
  
  auto swift_name = NS::String::string("attention", NS::UTF8StringEncoding);
  for (int i = 0; i < 2; ++i) {
    NS::SharedPtr<MTL::ComputePipelineState>* pso;
    if (i == 0) {
      pso = &attention_pso;
    } else {
      pso = &generate_block_mask_pso;
      if (hash.masked) {
        bool generate_block_mask = true;
        constants->setConstantValue(&generate_block_mask, MTL::DataTypeBool, 112);
      } else {
        continue;
      }
    }
    
    NS::Error *error = nullptr;
    auto function = NS::TransferPtr(context->library->newFunction(swift_name, constants.get(), &error));
    if (!function) {
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
    
    *pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
    if (!*pso) {
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
  }
  
  pool->drain();
}
