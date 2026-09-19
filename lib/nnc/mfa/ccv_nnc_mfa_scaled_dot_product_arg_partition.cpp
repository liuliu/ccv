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

static void _ccv_nnc_mfa_sdpap_na_score_tile(ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, uint16_t* const block_m, uint16_t* const block_n, uint16_t* const simdgroups)
{
  *block_m = 16;
  if (params.T == 1) {
    *block_n = 32;
    *simdgroups = 8;
  } else if (params.T >= 1024) {
    *block_n = 64;
    *simdgroups = params.C <= 1024 ? 4 : 2;
  } else {
    *block_n = 32;
    *simdgroups = 4;
  }
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
  descriptor.loadM = params.loadM;
  descriptor.T = params.T;
  descriptor.C = params.C;
  descriptor.kth = params.kth;
  descriptor.compressionRatio = params.compression_ratio;
  descriptor.queryOffset = params.query_offset;
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
  if (params.loadM) {
    const uint32_t dimensions[] = { params.T, params.C, static_cast<uint32_t>(params.query_offset) };
    encoder->setBytes(dimensions, sizeof(dimensions), 1);
  }
  const MTL::Size gridSize = kernel->gridSize(params.T, params.kth);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}

struct SDPAPCandidatePipeline {
  MTL::ComputePipelineState* score;
  MTL::ComputePipelineState* tile;
  MTL::ComputePipelineState* merge;
  MTL::ComputePipelineState* ids;
  MTL::Size scoreThreads;
  uint32_t blockM;
  uint32_t blockN;
};

static SDPAPCandidatePipeline _ccv_nnc_mfa_candid_aware_sdpap_pipeline(ccv_nnc_mfa_context_t* context, const ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, const uint8_t mode, const uint32_t width, const uint32_t kth, const uint32_t totalT)
{
  auto configure = [&](auto& descriptor) {
    descriptor.memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 : (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
    descriptor.T = params.T;
    descriptor.C = width;
    descriptor.H = params.H;
    descriptor.D = params.D;
    descriptor.kth = kth;
    descriptor.compressionRatio = params.compression_ratio;
    descriptor.queryOffset = params.query_offset;
    descriptor.scale = params.scale;
    descriptor.isCausal = params.is_causal;
    descriptor.loadC = true;
    descriptor.loadM = params.loadM;
    // Keep the original dense shader separate from variants that need index utilities.
    descriptor.scoreMode = mode == 0 ? 4 : mode;
    descriptor.candidateBlockSize = params.candidate_block_size;
    descriptor.candidateCount = params.candidate_count;
  };
  auto result = [](auto p) -> SDPAPCandidatePipeline {
    return { p->pipeline.get(), p->third.get(), p->fourth.get(), p->fifth.get(), p->kernel->scoreThreadgroupSize, p->kernel->scoreBlockM, p->kernel->scoreBlockN };
  };
  if (params.use_neural_accelerators && mode != 2) {
    NAScaledDotProductArgPartitionDescriptor descriptor;
    configure(descriptor);
    auto scoreParams = params;
    scoreParams.T = totalT;
    _ccv_nnc_mfa_sdpap_na_score_tile(scoreParams, &descriptor.scoreBlockM, &descriptor.scoreBlockN, &descriptor.scoreSIMDGroups);
    return result(context->kernel_cache.findKernel<NAScaledDotProductArgPartitionKernel, NAScaledDotProductArgPartitionDescriptor, NAScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties()));
  }
  ScaledDotProductArgPartitionDescriptor descriptor;
  configure(descriptor);
  return result(context->kernel_cache.findKernel<ScaledDotProductArgPartitionKernel, ScaledDotProductArgPartitionDescriptor, ScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), DeviceProperties()));
}

static void _ccv_nnc_mfa_encode_candid_aware_sdpap(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, MTL::CommandBatch* batch, MTL::Buffer** tensors, size_t* offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.candidate_count <= 2048);
  CCV_NNC_MFA_PRECONDITION(!params.has_candidates || !params.output_candidates);
  const uint32_t totalT = params.T;
  const int32_t queryOffset = params.query_offset;
  const size_t elementSize = params.data_type == MTL::DataTypeFloat ? 4 : 2;
  // Amortize short-context prefill dispatches within an 8 MiB score budget.
  // Long contexts retain 32-row tiles; scratch stays bounded in query length.
  // Enumeration needs no score scratch and can cover the entire query batch.
  const bool enumerateAll = params.C <= params.kth && (!params.output_candidates || params.C <= (uint64_t)params.candidate_count * params.candidate_block_size);
  const uint32_t queryTile = enumerateAll ? totalT : std::min(256u, std::max(32u, (2 * 1024 * 1024 / std::max(params.C, 1u)) / 16 * 16));
  for (uint32_t baseT = 0; baseT < totalT; baseT += queryTile) {
    params.T = std::min(queryTile, totalT - baseT);
    params.query_offset = queryOffset + baseT;
    // Indirect scoring wins once a fixed pool excludes enough history. At short
    // contexts retain dense query/key tiling and filter through a compact bitset.
    const bool denseCandidates = params.has_candidates && params.C <= (uint64_t)params.candidate_count * params.candidate_block_size * (params.T == 1 ? 4 : 8);
    const uint8_t scoreMode = params.has_candidates ? (denseCandidates ? 3 : 1) : 0;
    const uint32_t width = params.has_candidates && !denseCandidates ? params.candidate_count * params.candidate_block_size : params.C;
    const uint32_t blocks = params.output_candidates ? (params.C + params.candidate_block_size - 1) / params.candidate_block_size : 0;
    const bool selectBlocks = blocks > params.candidate_count;
    const bool enumerateRows = params.C <= params.kth && !selectBlocks;
    if (METAL_LOG_LEVEL(context) >= 3) {
      if (enumerateRows)
        ccv_nnc_mfa_log_message(params.has_candidates ? "SDPAP: candidate-restricted row enumeration." : "SDPAP: row and optional pool enumeration.");
      else if (params.has_candidates)
        ccv_nnc_mfa_log_message(denseCandidates ? "SDPAP: dense scores with candidate bitset." : "SDPAP: candidate-only scores.");
      else if (params.output_candidates)
        ccv_nnc_mfa_log_message(selectBlocks ? "SDPAP: dense scores and candidate pool output." : "SDPAP: dense scores and candidate pool enumeration.");
      else
        ccv_nnc_mfa_log_message("SDPAP: dense scores with sorted indices.");
    }
    auto pool = NS::AutoreleasePool::alloc()->init();
    // Enumeration passes its output width to index_ids at runtime. Keep the
    // unused ranking specialization fixed so growing widths reuse the pipeline.
    const auto position = _ccv_nnc_mfa_candid_aware_sdpap_pipeline(context, params, scoreMode, width, enumerateRows ? 1 : params.kth, totalT);
    SDPAPCandidatePipeline block = {};
    if (selectBlocks)
      block = _ccv_nnc_mfa_candid_aware_sdpap_pipeline(context, params, 2, blocks, params.candidate_count, totalT);
    pool->drain();
    const struct Runtime { uint32_t C; int32_t query_offset; uint32_t T; uint32_t key_count; } runtime = { width, params.query_offset, params.T, params.C };
    const size_t selectedOffset = offsets[3] + (size_t)baseT * params.kth * sizeof(int32_t);
    auto encodeIDs = [&](const SDPAPCandidatePipeline& pipeline, MTL::Buffer* input, size_t inputOffset, MTL::Buffer* output, size_t outputOffset, uint32_t length, uint32_t mode) {
      auto encoder = batch->startCommand();
      encoder->setComputePipelineState(pipeline.ids);
      encoder->useResource(input, MTL::ResourceUsageRead);
      encoder->useResource(output, MTL::ResourceUsageWrite);
      encoder->setBuffer(input, inputOffset, 0);
      encoder->setBuffer(output, outputOffset, 1);
      encoder->setBytes(&runtime, sizeof(runtime), 2);
      const uint32_t options[] = { length, mode };
      encoder->setBytes(options, sizeof(options), 3);
      encoder->dispatchThreadgroups(MTL::Size(params.T, 1, 1), MTL::Size(256, 1, 1));
      batch->finishCommand(encoder);
    };
    // If every block fits, the causal pool is already known without ranking.
    // When every row also fits, neither output needs any dot products.
    if (enumerateRows) {
      if (params.has_candidates)
        encodeIDs(position, tensors[4], offsets[4] + (size_t)baseT * params.candidate_count * sizeof(int32_t), tensors[3], selectedOffset, params.kth, 5);
      else
        encodeIDs(position, tensors[3], selectedOffset, tensors[3], selectedOffset, params.kth, 2);
      if (params.output_candidates)
        encodeIDs(position, tensors[5], offsets[5] + (size_t)baseT * params.candidate_count * sizeof(int32_t), tensors[5], offsets[5] + (size_t)baseT * params.candidate_count * sizeof(int32_t), params.candidate_count, 4);
      continue;
    }
    size_t bytes = 0;
    auto reserve = [&](size_t size) {
      const size_t offset = bytes;
      bytes += _ccv_nnc_mfa_sdpap_align_up(std::max(size, size_t(16)), 16);
      return offset;
    };
    const size_t inputIDs = reserve(params.has_candidates ? (size_t)params.T * (denseCandidates ? ((params.C + params.candidate_block_size - 1) / params.candidate_block_size + 31) / 32 : params.candidate_count) * sizeof(int32_t) : 0);
    const size_t scores = reserve((size_t)params.T * width * sizeof(float));
    const size_t blockScores = reserve(selectBlocks ? (size_t)params.T * blocks * sizeof(float) : 0);
    const uint32_t positionTiles = (width + 2047) / 2048;
    const uint32_t blockTiles = selectBlocks ? (blocks + 2047) / 2048 : 0;
    const size_t candidates = (size_t)params.T * std::max((size_t)positionTiles * params.kth, (size_t)blockTiles * params.candidate_count);
    const size_t mergeCandidates = (size_t)params.T * std::max((size_t)((positionTiles + 1) / 2) * params.kth, (size_t)((blockTiles + 1) / 2) * params.candidate_count);
    const size_t candidateScores = reserve(candidates * sizeof(float));
    const size_t candidateIDs = reserve(candidates * sizeof(int32_t));
    const size_t reducedScores = reserve(mergeCandidates * sizeof(float));
    const size_t reducedIDs = reserve(mergeCandidates * sizeof(int32_t));
    auto scratch = context->request_scratch(bytes);
    if (params.has_candidates)
      encodeIDs(position, tensors[4], offsets[4] + (size_t)baseT * params.candidate_count * sizeof(int32_t), scratch, inputIDs, params.candidate_count, denseCandidates ? 3 : 0);
    auto encoder = batch->startCommand();
    encoder->setComputePipelineState(position.score);
    for (int i = 0; i < 3; ++i) {
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
      const size_t offset = offsets[i] + (i == 1 ? 0 : (size_t)baseT * params.H * (i == 0 ? params.D : 1) * elementSize);
      encoder->setBuffer(tensors[i], offset, i);
    }
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, scores, 3);
    encoder->setBytes(&runtime, sizeof(runtime), 4);
    if (params.has_candidates) { encoder->setBuffer(scratch, inputIDs, 5); }
    if (params.has_candidates && !denseCandidates) {
      const uint32_t columns = params.use_neural_accelerators ? 32 : 4;
      encoder->dispatchThreadgroups(MTL::Size((width + columns - 1) / columns, params.T, 1), MTL::Size(params.use_neural_accelerators ? 32 : 128, 1, 1));
    } else {
      encoder->dispatchThreadgroups(MTL::Size((width + position.blockN - 1) / position.blockN, (params.T + position.blockM - 1) / position.blockM, 1), position.scoreThreads);
    }
    batch->finishCommand(encoder);
    if (selectBlocks) {
      auto encoder = batch->startCommand();
      encoder->setComputePipelineState(block.score);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBuffer(scratch, scores, 0);
      encoder->setBuffer(scratch, blockScores, 3);
      const Runtime blockRuntime = { blocks, params.query_offset, params.T, params.C };
      encoder->setBytes(&blockRuntime, sizeof(blockRuntime), 4);
      encoder->dispatchThreadgroups(MTL::Size((blocks + 255) / 256, params.T, 1), MTL::Size(256, 1, 1));
      batch->finishCommand(encoder);
    }
    auto select = [&](const SDPAPCandidatePipeline& pipeline, size_t scoreOffset, uint32_t count, uint32_t kth, bool hasCandidates, MTL::Buffer* output, size_t outputOffset) {
      const Runtime selectionRuntime = { count, params.query_offset, params.T, params.C };
      const uint32_t tiles = (count + 2047) / 2048;
      auto encoder = batch->startCommand();
      encoder->setComputePipelineState(pipeline.tile);
      encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
      encoder->setBuffer(scratch, scoreOffset, 0);
      encoder->setBuffer(scratch, candidateScores, 1);
      encoder->setBuffer(scratch, candidateIDs, 2);
      encoder->setBytes(&selectionRuntime, sizeof(selectionRuntime), 3);
      if (hasCandidates) { encoder->setBuffer(scratch, inputIDs, 5); }
      encoder->dispatchThreadgroups(MTL::Size(tiles, params.T, 1), MTL::Size(512, 1, 1));
      batch->finishCommand(encoder);
      uint32_t lists = tiles;
      const uint32_t mergeLists = kth > 512 ? 2 : 4;
      size_t inScore = candidateScores, inID = candidateIDs, outScore = reducedScores, outID = reducedIDs;
      for (;;) {
        const uint32_t outputLists = (lists + mergeLists - 1) / mergeLists;
        const bool final = outputLists == 1;
        auto encoder = batch->startCommand();
        encoder->setComputePipelineState(pipeline.merge);
        encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        encoder->setBuffer(scratch, inScore, 0);
        encoder->setBuffer(scratch, inID, 1);
        encoder->setBuffer(scratch, outScore, 2);
        encoder->setBuffer(final ? output : scratch, final ? outputOffset : outID, 3);
        if (final) { encoder->useResource(output, MTL::ResourceUsageWrite); }
        encoder->setBytes(&lists, sizeof(lists), 4);
        if (params.loadM) { encoder->setBytes(&selectionRuntime, sizeof(selectionRuntime), 5); }
        encoder->dispatchThreadgroups(MTL::Size(outputLists, params.T, 1), MTL::Size(512, 1, 1));
        batch->finishCommand(encoder);
        if (final) { break; }
        lists = outputLists;
        std::swap(inScore, outScore);
        std::swap(inID, outID);
      }
    };
    if (!params.has_candidates && params.C <= params.kth)
      encodeIDs(position, tensors[3], selectedOffset, tensors[3], selectedOffset, params.kth, 2);
    else {
      select(position, scores, width, params.kth, params.has_candidates, tensors[3], selectedOffset);
      if (params.sort_indices)
        encodeIDs(position, tensors[3], selectedOffset, tensors[3], selectedOffset, params.kth, 1);
    }
    if (params.output_candidates) {
      const size_t poolOffset = offsets[5] + (size_t)baseT * params.candidate_count * sizeof(int32_t);
      if (selectBlocks) {
        select(block, blockScores, blocks, params.candidate_count, false, tensors[5], poolOffset);
        encodeIDs(block, tensors[5], poolOffset, tensors[5], poolOffset, params.candidate_count, 1);
      } else {
        encodeIDs(position, tensors[5], poolOffset, tensors[5], poolOffset, params.candidate_count, 4);
      }
    }
  }
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

  if (params.has_candidates || params.output_candidates || params.sort_indices) {
    _ccv_nnc_mfa_encode_candid_aware_sdpap(context, params, command_batch, tensors, tensor_offsets);
    return;
  }

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
    descriptor.loadM = params.loadM;
    descriptor.C = params.C;
    descriptor.H = params.H;
    descriptor.D = params.D;
    descriptor.kth = params.kth;
    descriptor.compressionRatio = params.compression_ratio;
    descriptor.queryOffset = params.query_offset;
    descriptor.scale = params.scale;
    descriptor.isCausal = params.is_causal != 0;
    descriptor.scoreBlockM = 16;
    descriptor.scoreBlockN = 32;
    descriptor.scoreSIMDGroups = 4;
  };

  auto encodePipeline =
  [&](auto pipelineValue, const auto& descriptor, const bool loadC) {
    auto kernel = pipelineValue->kernel;
    auto scorePipeline = pipelineValue->pipeline;
    auto topKSerialPipeline = pipelineValue->second;
    auto topKTilePipeline = pipelineValue->third;
    auto topKMergePipeline = pipelineValue->fourth;

    const struct {
      uint32_t C;
      int32_t query_offset;
      uint32_t T;
    } runtimeParams = { params.C, params.query_offset, params.T };

    const uint32_t topKTileC = 2048;
    const uint32_t topKMergeLists = 4;
    const uint32_t topKTiles = (params.C + topKTileC - 1) / topKTileC;
    const bool useTiledTopK = params.kth <= 512;
    const size_t scoreBytes = std::max<size_t>((size_t)params.T * params.C * sizeof(float), sizeof(float));
    const size_t candidateCount = useTiledTopK ? (size_t)params.T * topKTiles * params.kth : 0;
    const uint32_t reducedTopKLists = (topKTiles + topKMergeLists - 1) / topKMergeLists;
    const size_t reducedCandidateCount = useTiledTopK ? (size_t)params.T * reducedTopKLists * params.kth : 0;
    const size_t reducedCandidateIndexOffset = _ccv_nnc_mfa_sdpap_align_up(reducedCandidateCount * sizeof(float), 16);
    if (topKTiles > topKMergeLists) {
      CCV_NNC_MFA_PRECONDITION(reducedCandidateIndexOffset + reducedCandidateCount * sizeof(int32_t) <= scoreBytes);
    }
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
      if (loadC) {
        scoreEncoder->setBytes(&runtimeParams, sizeof(runtimeParams), 4);
      }
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
      if (loadC) {
        topKTileEncoder->setBytes(&runtimeParams, sizeof(runtimeParams), 3);
      }
      topKTileEncoder->dispatchThreadgroups(MTL::Size(topKTiles, params.T, 1), kernel->topKTileThreadgroupSize);
      command_batch->finishCommand(topKTileEncoder);

      size_t inputScoreOffset = candidateScoreOffset;
      size_t inputIndexOffset = candidateIndexOffset;
      size_t outputScoreOffset = 0;
      size_t outputIndexOffset = reducedCandidateIndexOffset;
      uint32_t inputLists = topKTiles;
      for (;;) {
        const uint32_t outputLists = (inputLists + topKMergeLists - 1) / topKMergeLists;
        const bool isFinal = outputLists == 1;
        auto topKMergeEncoder = command_batch->startCommand();
        topKMergeEncoder->setComputePipelineState(topKMergePipeline.get());
        topKMergeEncoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        topKMergeEncoder->setBuffer(scratch, inputScoreOffset, 0);
        topKMergeEncoder->setBuffer(scratch, inputIndexOffset, 1);
        topKMergeEncoder->setBuffer(scratch, outputScoreOffset, 2);
        if (isFinal) {
          topKMergeEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
          topKMergeEncoder->setBuffer(tensors[3], tensor_offsets[3], 3);
        } else {
          topKMergeEncoder->setBuffer(scratch, outputIndexOffset, 3);
        }
        topKMergeEncoder->setBytes(&inputLists, sizeof(inputLists), 4);
        if (params.loadM)
          topKMergeEncoder->setBytes(&runtimeParams, sizeof(runtimeParams), 5);
        topKMergeEncoder->dispatchThreadgroups(MTL::Size(outputLists, params.T, 1), kernel->topKMergeThreadgroupSize);
        command_batch->finishCommand(topKMergeEncoder);
        if (isFinal) {
          break;
        }
        inputLists = outputLists;
        std::swap(inputScoreOffset, outputScoreOffset);
        std::swap(inputIndexOffset, outputIndexOffset);
      }
    } else {
      auto topKEncoder = command_batch->startCommand();
      topKEncoder->setComputePipelineState(topKSerialPipeline.get());
      topKEncoder->useResource(scratch, MTL::ResourceUsageRead);
      topKEncoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      topKEncoder->setBuffer(scratch, 0, 0);
      topKEncoder->setBuffer(tensors[3], tensor_offsets[3], 1);
      if (loadC) {
        topKEncoder->setBytes(&runtimeParams, sizeof(runtimeParams), 2);
      }
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
    _ccv_nnc_mfa_sdpap_na_score_tile(params, &descriptor.scoreBlockM, &descriptor.scoreBlockN, &descriptor.scoreSIMDGroups);
    descriptor.loadC = true;
    auto pipelineValue = shaderCache.findKernel<NAScaledDotProductArgPartitionKernel, NAScaledDotProductArgPartitionDescriptor, NAScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodePipeline(pipelineValue, descriptor, descriptor.loadC);
  } else {
    ScaledDotProductArgPartitionDescriptor descriptor;
    setDescriptor(descriptor);
    descriptor.loadC = true;
    auto pipelineValue = shaderCache.findKernel<ScaledDotProductArgPartitionKernel, ScaledDotProductArgPartitionDescriptor, ScaledDotProductArgPartitionKernelDescriptor>(descriptor, context->device.get(), dprops);
    pool->drain();
    encodePipeline(pipelineValue, descriptor, descriptor.loadC);
  }
}
