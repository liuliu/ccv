#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

#include "kernels/ShaderCache.hpp"
#include "kernels/AttentionKernel.hpp"
#include "kernels/AttentionKernelDescriptor.hpp"
#include "kernels/AttentionDescriptor.hpp"
#include "kernels/AttentionR1Descriptor.hpp"
#include "kernels/AttentionR1Kernel.hpp"
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
  (void)context;
  (void)params;
  // Generated attention kernels are compiled lazily through the shader cache.
}

void ccv_nnc_mfa_encode_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  mfa::attention::hash hash(params);

  if (params.type != 0) {
    CCV_NNC_MFA_PRECONDITION(!params.masked);
    CCV_NNC_MFA_PRECONDITION(!params.is_varlen);
  }
  CCV_NNC_MFA_PRECONDITION(!(params.masked && params.is_varlen));
  if (params.sliding_window > 0) {
    CCV_NNC_MFA_PRECONDITION(params.type == 0);
    CCV_NNC_MFA_PRECONDITION(params.is_causal);
    CCV_NNC_MFA_PRECONDITION(!params.is_varlen);
    CCV_NNC_MFA_PRECONDITION(!params.use_quantized_attention);
  }
  {
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
    GEMMOperandPrecision attentionR1Precision = GEMMOperandPrecision::FP16;
    bool attentionR1DataType = true;
    switch (params.data_type) {
    case MTL::DataTypeHalf:
      attentionR1Precision = GEMMOperandPrecision::FP16;
      break;
    case MTL::DataTypeBFloat:
      attentionR1Precision = GEMMOperandPrecision::BF16;
      break;
    default:
      attentionR1DataType = false;
      break;
    }
    const bool useAttentionR1 =
        params.type == 0 &&
        !params.use_quantized_attention &&
        !hash.masked &&
        !hash.is_varlen &&
        hash.sliding_window == 0 &&
        attentionR1DataType &&
        hash.R == 1 &&
        hash.C > 0 &&
        (hash.D == 128 || hash.D == 256) &&
        hash.Hk > 0 &&
        (hash.Hq % hash.Hk) == 0;
    if (useAttentionR1) {
      AttentionR1Descriptor attentionDesc = AttentionR1Descriptor::select(
          attentionR1Precision, hash.C, hash.Hq, hash.Hk, hash.D, hash.alpha, true, hash.attention_sinks);
      auto pool = NS::AutoreleasePool::alloc()->init();
      auto &shaderCache = context->kernel_cache;
      DeviceProperties dprops = DeviceProperties();
      auto pipelineValue = shaderCache.findKernel<AttentionR1Kernel, AttentionR1Descriptor, AttentionR1KernelDescriptor>(attentionDesc, context->device.get(), dprops);
      pool->drain();
      auto kernel = pipelineValue->kernel;
      auto pipeline = pipelineValue->pipeline;
      const uint32_t threadgroupSize = kernel->threadgroupSize(attentionDesc);
      CCV_NNC_MFA_PRECONDITION(threadgroupSize <= pipeline->maxTotalThreadsPerThreadgroup());
      if (attentionDesc.mode == AttentionR1Descriptor::Mode::direct) {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(pipeline.get());
        encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(attentionDesc), 0);
        encoder->useResource(tensors[0], MTL::ResourceUsageRead);
        encoder->useResource(tensors[1], MTL::ResourceUsageRead);
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
        if (hash.attention_sinks) {
          encoder->useResource(tensors[8], MTL::ResourceUsageRead);
        }
        encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
        encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
        encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
        encoder->setBuffer(tensors[3], tensor_offsets[3], 3);
        encoder->setBytes(&hash.C, sizeof(hash.C), 4);
        if (hash.attention_sinks) {
          encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
          encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
        }
        encoder->dispatchThreadgroups(
            MTL::Size(hash.Hq, batch_sizes[0], 1),
            MTL::Size(threadgroupSize, 1, 1));
        command_batch->finishCommand(encoder);
        return;
      }

      const size_t partialBytes =
          (size_t)batch_sizes[0] * hash.Hq * attentionDesc.workgroups * (hash.D + 2) * sizeof(float);
      auto scratch = context->request_scratch(partialBytes);
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(attentionDesc), 0);
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
      if (hash.attention_sinks) {
        encoder->useResource(tensors[8], MTL::ResourceUsageRead);
      }
      encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
      encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      encoder->setBuffer(tensors[2], tensor_offsets[2], 2);
      encoder->setBuffer(scratch, 0, 3);
      encoder->setBytes(&hash.C, sizeof(hash.C), 4);
      if (hash.attention_sinks) {
        encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
        encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
      }
      encoder->dispatchThreadgroups(
          MTL::Size(hash.Hq, batch_sizes[0], attentionDesc.workgroups),
          MTL::Size(threadgroupSize, 1, 1));
      command_batch->finishCommand(encoder);

      auto reduceEncoder = command_batch->startCommand();
      reduceEncoder->setComputePipelineState(pipelineValue->second.get());
      reduceEncoder->useResource(scratch, MTL::ResourceUsageRead);
      reduceEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      reduceEncoder->setBuffer(scratch, 0, 0);
      reduceEncoder->setBuffer(tensors[3], tensor_offsets[3], 1);
      reduceEncoder->dispatchThreadgroups(
          MTL::Size(hash.Hq, batch_sizes[0], 1),
          MTL::Size(32, 1, 1));
      command_batch->finishCommand(reduceEncoder);
      return;
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
      attentionDesc.attentionSinks = hash.attention_sinks;
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
      if (hash.attention_sinks) {
        encoder->useResource(tensors[8], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
        encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
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
      attentionDesc.attentionSinks = hash.attention_sinks;
      attentionDesc.slidingWindow = hash.sliding_window;
      attentionDesc.loadC = !hash.masked && !hash.is_varlen && hash.R <= 4;
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
      const bool useSplitKV = kernel->splitKV > 1;
      size_t scratchSize = 0;
      const size_t blockMaskBytes =
          hash.masked ? (size_t)batch_sizes[1] * qTiles * kTiles * sizeof(uint8_t) : 0;
      const size_t blockMaskOffset = hash.masked ? reserve(&scratchSize, blockMaskBytes) : 0;
      const size_t splitKVPartialOBytes =
          useSplitKV ? (size_t)attentionDesc.batchDimension * hash.Hq * kernel->splitKV * hash.R * hash.D * sizeof(float) : 0;
      const size_t splitKVPartialLBytes =
          useSplitKV ? (size_t)attentionDesc.batchDimension * hash.Hq * kernel->splitKV * hash.R * sizeof(float) : 0;
      const size_t splitKVPartialOOffset = useSplitKV ? reserve(&scratchSize, splitKVPartialOBytes) : 0;
      const size_t splitKVPartialLOffset = useSplitKV ? reserve(&scratchSize, splitKVPartialLBytes) : 0;
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
      if (useSplitKV) {
        auto encoder = command_batch->startCommand();
        encoder->setComputePipelineState(pipeline.get());
        encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(pipeline.get(), attentionDesc), 0);
        encoder->useResource(tensors[0], MTL::ResourceUsageRead);
        encoder->useResource(tensors[1], MTL::ResourceUsageRead);
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        if (hash.attention_sinks) {
          encoder->useResource(tensors[8], MTL::ResourceUsageRead);
        }
        encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
        encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
        encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
        encoder->setBuffer(scratch, splitKVPartialOOffset, 5);
        encoder->setBuffer(scratch, splitKVPartialLOffset, 6);
        if (attentionDesc.loadC) {
          encoder->setBytes(&hash.C, sizeof(hash.C), 21);
        }
        if (hash.attention_sinks) {
          encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
          encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
        }
        encoder->dispatchThreadgroups(
            kernel->threadgroupsPerGrid(attentionDesc),
            MTL::Size(kernel->threadgroupSize(pipeline.get(), attentionDesc), 1, 1));
        command_batch->finishCommand(encoder);

        auto combineEncoder = command_batch->startCommand();
        combineEncoder->setComputePipelineState(blockMaskPipeline.get());
        combineEncoder->useResource(scratch, MTL::ResourceUsageRead);
        combineEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
        if (tensors[5]) {
          combineEncoder->useResource(tensors[5], MTL::ResourceUsageWrite);
        } else {
          combineEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        }
        combineEncoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
        if (tensors[5]) {
          combineEncoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
        } else {
          combineEncoder->setBuffer(scratch, lOffset, AttentionOperand(AttentionOperand::L).bufferIndex());
        }
        combineEncoder->setBuffer(scratch, splitKVPartialOOffset, 5);
        combineEncoder->setBuffer(scratch, splitKVPartialLOffset, 6);
        const size_t combineThreadgroups =
            ((size_t)attentionDesc.batchDimension * hash.Hq * hash.R * hash.D + NAAttentionKernel::splitKVCombineThreads - 1) /
            NAAttentionKernel::splitKVCombineThreads;
        combineEncoder->dispatchThreadgroups(
            MTL::Size(combineThreadgroups, 1, 1),
            MTL::Size(NAAttentionKernel::splitKVCombineThreads, 1, 1));
        command_batch->finishCommand(combineEncoder);
        return;
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
      if (attentionDesc.loadC) {
        encoder->setBytes(&hash.C, sizeof(hash.C), 21);
      }
      if (hash.attention_sinks) {
        encoder->useResource(tensors[8], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
        encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
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
    if (params.type != 0) {
      CCV_NNC_MFA_PRECONDITION(!params.is_causal);
      CCV_NNC_MFA_PRECONDITION(!params.masked);
    }
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
    attentionDesc.isCausal = hash.is_causal;
    attentionDesc.masked = hash.masked;
    attentionDesc.isVarlen = hash.is_varlen;
    attentionDesc.attentionSinks = hash.attention_sinks;
    attentionDesc.slidingWindow = hash.sliding_window;
    if (hash.masked && batch_sizes[1] > 1) {
      attentionDesc.maskBatchStride = hash.R * hash.C;
    }
    if (params.batched && !hash.is_varlen) {
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
      auto blockMaskPipeline = pipelineValue->second;
      const uint64_t outputElements = hash.is_varlen ?
        (uint64_t)params.output_rows * hash.D * hash.Hq :
        (uint64_t)hash.R * hash.D * hash.Hq * attentionDesc.batchDimension;
      if (hash.is_varlen) {
        CCV_NNC_MFA_PRECONDITION(params.output_rows > 0);
        CCV_NNC_MFA_PRECONDITION(params.output_rows <= hash.R * attentionDesc.batchDimension);
      }
      CCV_NNC_MFA_PRECONDITION(outputElements <= (uint64_t)UINT32_MAX);
      const bool castsOutput = attentionDesc.lowPrecisionInputs;
      const uint64_t outputScratchBytes = castsOutput ?
        sizeof(float) * outputElements : 0;
      const uint64_t lScratchBytes = !tensors[5] ?
        sizeof(float) * hash.R * hash.Hq * attentionDesc.batchDimension : 0;
      const uint32_t qTiles = (hash.R + kernel->blockDimensions[0] - 1) / kernel->blockDimensions[0];
      const uint32_t kTiles = (hash.C + kernel->blockDimensions[1] - 1) / kernel->blockDimensions[1];
      auto align_up =
      [&](uint64_t value) -> uint64_t {
        return (value + 255) & ~((uint64_t)255);
      };
      uint64_t scratch_size = outputScratchBytes + lScratchBytes;
      const uint64_t blockMaskOffset = hash.masked ? align_up(scratch_size) : scratch_size;
      if (hash.masked) {
        scratch_size = blockMaskOffset + (uint64_t)batch_sizes[1] * qTiles * kTiles * sizeof(uint8_t);
      }
      auto scratch = scratch_size > 0 ? context->request_scratch(scratch_size) : NULL;
      if (hash.masked) {
        auto blockMaskEncoder = command_batch->startCommand();
        blockMaskEncoder->setComputePipelineState(blockMaskPipeline.get());
        blockMaskEncoder->setThreadgroupMemoryLength(AttentionKernel::blockMaskThreads * sizeof(uint32_t) * 2, 0);
        blockMaskEncoder->useResource(tensors[4], MTL::ResourceUsageRead);
        blockMaskEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        blockMaskEncoder->setBuffer(tensors[4], tensor_offsets[4], 15);
        blockMaskEncoder->setBuffer(scratch, blockMaskOffset, 16);
        blockMaskEncoder->dispatchThreadgroups(
            MTL::Size(qTiles, kTiles, batch_sizes[1]),
            MTL::Size(AttentionKernel::blockMaskThreads, 1, 1));
        command_batch->finishCommand(blockMaskEncoder);
      }

      // Allocate a new command.
      auto encoder = command_batch->startCommand();
      encoder->setComputePipelineState(pipeline.get());
      encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    
      // Bind the function arguments.
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      if (castsOutput) {
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      } else {
        encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
        if (!tensors[5] || hash.masked) {
          encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        }
      }
      if (tensors[5]) {
        encoder->useResource(tensors[5], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      }
      encoder->setBuffer(tensors[0], tensor_offsets[0], AttentionOperand(AttentionOperand::Q).bufferIndex());
      encoder->setBuffer(tensors[1], tensor_offsets[1], AttentionOperand(AttentionOperand::K).bufferIndex());
      encoder->setBuffer(tensors[2], tensor_offsets[2], AttentionOperand(AttentionOperand::V).bufferIndex());
      if (castsOutput) {
        encoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::O).bufferIndex());
        if (tensors[5]) {
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
        } else {
          encoder->setBuffer(scratch, outputScratchBytes, AttentionOperand(AttentionOperand::L).bufferIndex());
        }
      } else {
        encoder->setBuffer(tensors[3], tensor_offsets[3], AttentionOperand(AttentionOperand::O).bufferIndex());
        if (tensors[5]) {
          encoder->setBuffer(tensors[5], tensor_offsets[5], AttentionOperand(AttentionOperand::L).bufferIndex());
        } else {
          encoder->setBuffer(scratch, 0, AttentionOperand(AttentionOperand::L).bufferIndex());
        }
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
      if (hash.attention_sinks) {
        encoder->useResource(tensors[8], MTL::ResourceUsageRead);
        encoder->setBuffer(tensors[8], tensor_offsets[8], 19);
        encoder->setBytes(&params.sink_head_stride, sizeof(params.sink_head_stride), 20);
      }
    
      MTL::Size gridSize
      (ceilDivide(int64_t(hash.R), kernel->blockDimensions[0]) * hash.Hq * attentionDesc.batchDimension, 1, 1);
      MTL::Size groupSize
      (int64_t(kernel->threadgroupSize), 1, 1);
    
      // Dispatch the required number of threads.
      encoder->dispatchThreadgroups(gridSize, groupSize);
    
      // Finish the command.
      command_batch->finishCommand(encoder);
      if (castsOutput) {
        // Need to dispatch to cast.
        ccv_nnc_mfa_cast_params_t cast_params = {
          .original_data_type = MTL::DataTypeFloat,
          .data_type = attentionDesc.isBF16 ? MTL::DataTypeBFloat : MTL::DataTypeHalf,
          .length = (uint32_t)outputElements,
          .loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
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
        hash.Hk > 0 &&
        hash.Hq >= hash.Hk &&
        (hash.Hq % hash.Hk) == 0 &&
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
        hash.Hk > 0 &&
        hash.Hq >= hash.Hk &&
        (hash.Hq % hash.Hk) == 0 &&
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
        attentionDesc.batchStrides[AttentionOperand::dK] = hash.C * hash.D * hash.Hk;
        attentionDesc.batchStrides[AttentionOperand::dV] = hash.C * hash.D * hash.Hk;
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

      const size_t dQBytes = sizeof(float) * (size_t)hash.R * hash.D * hash.Hq * attentionDesc.batchDimension;
      const size_t dKBytes = sizeof(float) * (size_t)hash.C * hash.D * hash.Hk * attentionDesc.batchDimension;
      const size_t dVBytes = sizeof(float) * (size_t)hash.C * hash.D * hash.Hk * attentionDesc.batchDimension;
      const size_t gradientBytes = attentionDesc.lowPrecisionInputs ? dQBytes + dKBytes + dVBytes : 0;
      const size_t dOffset = gradientBytes;
      const size_t dBytes = sizeof(float) * (size_t)hash.R * hash.Hq * attentionDesc.batchDimension;
      const size_t scratch_size = gradientBytes + dBytes;
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
        backwardQueryEncoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
      } else {
        backwardQueryEncoder->setBuffer(tensors[6], tensor_offsets[6], AttentionOperand(AttentionOperand::dQ).bufferIndex());
        backwardQueryEncoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
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
        backwardKeyValueEncoder->setBuffer(scratch, dQBytes, AttentionOperand(AttentionOperand::dK).bufferIndex());
        backwardKeyValueEncoder->setBuffer(scratch, dQBytes + dKBytes, AttentionOperand(AttentionOperand::dV).bufferIndex());
        backwardKeyValueEncoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
      } else {
        backwardKeyValueEncoder->setBuffer(scratch, dOffset, AttentionOperand(AttentionOperand::D).bufferIndex());
        backwardKeyValueEncoder->setBuffer(tensors[7], tensor_offsets[7], AttentionOperand(AttentionOperand::dK).bufferIndex());
        backwardKeyValueEncoder->setBuffer(tensors[8], tensor_offsets[8], AttentionOperand(AttentionOperand::dV).bufferIndex());
      }

      MTL::Size backwardKeyValueGridSize
      (ceilDivide(int64_t(hash.C), backwardKeyValueKernel->blockDimensions[0]) * hash.Hk * attentionDesc.batchDimension, 1, 1);
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
          .length = hash.R * hash.D * hash.Hq * attentionDesc.batchDimension,
          .loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
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
        cast_params.length = hash.C * hash.D * hash.Hk * attentionDesc.batchDimension;
        ccv_nnc_mfa_prepare_cast(context, cast_params);
        cast_tensors[1] = tensors[7];
        cast_tensor_offsets[0] = dQBytes;
        cast_tensor_offsets[1] = tensor_offsets[7];
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
        cast_tensors[1] = tensors[8];
        cast_tensor_offsets[0] = dQBytes + dKBytes;
        cast_tensor_offsets[1] = tensor_offsets[8];
        ccv_nnc_mfa_encode_cast(context, cast_params, command_batch, cast_tensors, cast_tensor_offsets);
      }
    }
    return;
  }

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
  attention_sinks = params.attention_sinks;
  sliding_window = params.sliding_window;
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
  (use_quantized_attention == hash.use_quantized_attention) &&
  (attention_sinks == hash.attention_sinks) &&
  (sliding_window == hash.sliding_window);
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
  os << " .attention_sinks = " << bool(hash.attention_sinks) << " ";
  os << " .sliding_window = " << hash.sliding_window << " ";
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
  combine_32(seed, hash.attention_sinks);
  combine_32(seed, hash.sliding_window);
  combine_32(seed, hash.upcast);
  return seed;
}
