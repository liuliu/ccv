#include "ccv_nnc_mfa.hpp"
#include "kernels/NAScaledDotProductArgPartitionDescriptor.hpp"
#include "kernels/NAScaledDotProductArgPartitionKernel.hpp"
#include "kernels/NAScaledDotProductArgPartitionKernelDescriptor.hpp"
#include "kernels/ScaledDotProductArgPartitionDescriptor.hpp"
#include "kernels/ScaledDotProductArgPartitionEnumerateDescriptor.hpp"
#include "kernels/ScaledDotProductArgPartitionEnumerateKernel.hpp"
#include "kernels/ScaledDotProductArgPartitionEnumerateKernelDescriptor.hpp"
#include "kernels/ScaledDotProductArgPartitionKernel.hpp"
#include "kernels/ScaledDotProductArgPartitionKernelDescriptor.hpp"
#include <algorithm>
using namespace ccv::nnc;

static size_t _ccv_nnc_mfa_sdpap_align_up(const size_t value, const size_t alignment)
{
  return (value + alignment - 1) / alignment * alignment;
}

static void _ccv_nnc_mfa_sdpap_score_tile(ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, uint16_t* const block_m, uint16_t* const block_n, uint16_t* const simdgroups)
{
  (void)params;
  *block_m = 16;
  *block_n = 32;
  *simdgroups = 4;
}

void ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition(mfa::context* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition_enumerate(mfa::context* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_scaled_dot_product_arg_partition_enumerate(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.T > 0);
  CCV_NNC_MFA_PRECONDITION(params.C <= params.kth);
  CCV_NNC_MFA_PRECONDITION(params.kth > 0);
  CCV_NNC_MFA_PRECONDITION(params.compression_ratio > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] == nullptr);

  ScaledDotProductArgPartitionEnumerateDescriptor descriptor;
  descriptor.T = params.T;
  descriptor.C = params.C;
  descriptor.kth = params.kth;
  descriptor.compressionRatio = params.compression_ratio;
  descriptor.isCausal = params.is_causal != 0;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<ScaledDotProductArgPartitionEnumerateKernel, ScaledDotProductArgPartitionEnumerateDescriptor, ScaledDotProductArgPartitionEnumerateKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;

  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageWrite);
  encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
  const MTL::Size gridSize = kernel->gridSize(params.T, params.kth);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}

void ccv_nnc_mfa_encode_scaled_dot_product_arg_partition(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.kth > 0);
  CCV_NNC_MFA_PRECONDITION(params.kth <= 1024);
  CCV_NNC_MFA_PRECONDITION(params.compression_ratio > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[2] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[3] != nullptr);

  auto setDescriptor =
  [&](auto& descriptor) {
    switch (params.data_type) {
      case MTL::DataTypeFloat:
        descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
        break;
      case MTL::DataTypeHalf:
        descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
        break;
      case MTL::DataTypeBFloat:
        descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
        break;
      default:
        CCV_NNC_MFA_PRECONDITION(false);
    }
    descriptor.T = params.T;
    descriptor.C = params.C;
    descriptor.H = params.H;
    descriptor.D = params.D;
    descriptor.kth = params.kth;
    descriptor.compressionRatio = params.compression_ratio;
    descriptor.scale = params.scale;
    descriptor.isCausal = params.is_causal != 0;
    _ccv_nnc_mfa_sdpap_score_tile(params, &descriptor.scoreBlockM, &descriptor.scoreBlockN, &descriptor.scoreSIMDGroups);
  };

  auto encodePipeline =
  [&](auto pipelineValue, const auto& descriptor) {
    auto kernel = pipelineValue->kernel;
    auto scorePipeline = pipelineValue->pipeline;
    auto topKSerialPipeline = pipelineValue->second;
    auto topKTilePipeline = pipelineValue->third;
    auto topKMergePipeline = pipelineValue->fourth;

    const uint32_t topKTileC = 2048;
    const uint32_t topKMaxTiles = 4;
    const uint32_t topKTiles = (params.C + topKTileC - 1) / topKTileC;
    const bool useTiledTopK = params.kth <= 512 && topKTiles <= topKMaxTiles;
    const size_t scoreBytes = std::max<size_t>((size_t)params.T * params.C * sizeof(float), sizeof(float));
    const size_t candidateCount = useTiledTopK ? (size_t)params.T * topKTiles * params.kth : 0;
    const size_t candidateScoreOffset = _ccv_nnc_mfa_sdpap_align_up(scoreBytes, 16);
    const size_t candidateIndexOffset = _ccv_nnc_mfa_sdpap_align_up(candidateScoreOffset + candidateCount * sizeof(float), 16);
    const size_t scratchBytes = useTiledTopK ? candidateIndexOffset + candidateCount * sizeof(int32_t) : scoreBytes;
    auto scratch = context->request_scratch(scratchBytes);
    if (params.C > 0) {
      auto scoreEncoder = command_batch->startCommand();
      scoreEncoder->setComputePipelineState(scorePipeline.get());
      scoreEncoder->useResource(tensors[0], MTL::ResourceUsageRead);
      scoreEncoder->useResource(tensors[1], MTL::ResourceUsageRead);
      scoreEncoder->useResource(tensors[2], MTL::ResourceUsageRead);
      scoreEncoder->useResource(scratch, MTL::ResourceUsageWrite);
      scoreEncoder->setBuffer(tensors[0], tensor_offsets[0], 0);
      scoreEncoder->setBuffer(tensors[1], tensor_offsets[1], 1);
      scoreEncoder->setBuffer(tensors[2], tensor_offsets[2], 2);
      scoreEncoder->setBuffer(scratch, 0, 3);
      const uint32_t xBlocks = (params.C + descriptor.scoreBlockN - 1) / descriptor.scoreBlockN;
      const uint32_t yBlocks = (params.T + descriptor.scoreBlockM - 1) / descriptor.scoreBlockM;
      scoreEncoder->dispatchThreadgroups(MTL::Size(xBlocks, yBlocks, 1), kernel->scoreThreadgroupSize);
      command_batch->finishCommand(scoreEncoder);
    }

    if (useTiledTopK) {
      auto topKTileEncoder = command_batch->startCommand();
      topKTileEncoder->setComputePipelineState(topKTilePipeline.get());
      topKTileEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      topKTileEncoder->setBuffer(scratch, 0, 0);
      topKTileEncoder->setBuffer(scratch, candidateScoreOffset, 1);
      topKTileEncoder->setBuffer(scratch, candidateIndexOffset, 2);
      topKTileEncoder->dispatchThreadgroups(MTL::Size(topKTiles, params.T, 1), kernel->topKTileThreadgroupSize);
      command_batch->finishCommand(topKTileEncoder);

      auto topKMergeEncoder = command_batch->startCommand();
      topKMergeEncoder->setComputePipelineState(topKMergePipeline.get());
      topKMergeEncoder->useResource(scratch, MTL::ResourceUsageRead);
      topKMergeEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      topKMergeEncoder->setBuffer(scratch, candidateScoreOffset, 0);
      topKMergeEncoder->setBuffer(scratch, candidateIndexOffset, 1);
      topKMergeEncoder->setBuffer(tensors[3], tensor_offsets[3], 2);
      topKMergeEncoder->dispatchThreadgroups(MTL::Size(params.T, 1, 1), kernel->topKMergeThreadgroupSize);
      command_batch->finishCommand(topKMergeEncoder);
    } else {
      auto topKEncoder = command_batch->startCommand();
      topKEncoder->setComputePipelineState(topKSerialPipeline.get());
      topKEncoder->useResource(scratch, MTL::ResourceUsageRead);
      topKEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      topKEncoder->setBuffer(scratch, 0, 0);
      topKEncoder->setBuffer(tensors[3], tensor_offsets[3], 1);
      topKEncoder->dispatchThreadgroups(MTL::Size(params.T, 1, 1), kernel->topKThreadgroupSize);
      command_batch->finishCommand(topKEncoder);
    }
  };

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto &shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  if (params.use_neural_accelerators) {
    NAScaledDotProductArgPartitionDescriptor descriptor;
    setDescriptor(descriptor);
    auto pipelineValue = shaderCache.findKernel<NAScaledDotProductArgPartitionKernel, NAScaledDotProductArgPartitionDescriptor, NAScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodePipeline(pipelineValue, descriptor);
  } else {
    ScaledDotProductArgPartitionDescriptor descriptor;
    setDescriptor(descriptor);
    auto pipelineValue = shaderCache.findKernel<ScaledDotProductArgPartitionKernel, ScaledDotProductArgPartitionDescriptor, ScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodePipeline(pipelineValue, descriptor);
  }
}
