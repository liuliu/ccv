#include "NAInt8AttentionDescriptor.hpp"
#include "NAInt8AttentionKernelDescriptor.hpp"
#include "NAInt8AttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static void serializeBinaries(MTL::BinaryArchive *const binaryArchive, const std::string& pathToWrite) noexcept {
  NS::Error *error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
}

}

bool NAInt8AttentionDescriptor::operator==(const NAInt8AttentionDescriptor& rhs) const {
  auto lhsDimensions = matrixDimensions;
  auto rhsDimensions = rhs.matrixDimensions;
  if (loadR) lhsDimensions[0] = rhsDimensions[0] = 0;
  if (loadC) lhsDimensions[1] = rhsDimensions[1] = 0;
  return
    loadR == rhs.loadR && loadC == rhs.loadC &&
    splitQueryRanges() == rhs.splitQueryRanges() &&
    (!(loadR || loadC) || kernelDescriptor() == rhs.kernelDescriptor()) &&
    batchDimension == rhs.batchDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    type == rhs.type &&
    ioPrecision == rhs.ioPrecision &&
    partitioned == rhs.partitioned &&
    splitOutput == rhs.splitOutput &&
    lowPrecisionIntermediates == rhs.lowPrecisionIntermediates &&
    scale == rhs.scale &&
    isCausal == rhs.isCausal &&
    masked == rhs.masked &&
    isVarlen == rhs.isVarlen &&
    attentionSinks == rhs.attentionSinks &&
    ((loadR || loadC) || maskBatchStride == rhs.maskBatchStride) &&
    ((loadR || loadC) || batchStrides == rhs.batchStrides) &&
    simd_all(lhsDimensions == rhsDimensions);
}

std::size_t std::hash<NAInt8AttentionDescriptor>::operator()(const NAInt8AttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.Hq);
  combine_32(seed, hash.Hk);
  combine_32(seed, (uint16_t)hash.type.value);
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.ioPrecision.value,
      (uint16_t)(hash.lowPrecisionIntermediates ? 1 : 0) }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.isCausal ? 1 : 0),
      (uint16_t)(hash.masked ? 1 : 0) }));
  combine_32(seed, hash.isVarlen ? 1 : 0);
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, hash.partitioned ? 1 : 0);
  combine_32(seed, hash.splitOutput ? 1 : 0);
  combine_32(seed, hash.splitQueryRanges() ? 1 : 0);
  combine_32(seed, (hash.loadR || hash.loadC) ? 0 : hash.maskBatchStride);
  combine_32(seed, hash.loadR ? 0 : hash.matrixDimensions[0]);
  combine_32(seed, hash.loadC ? 0 : hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  combine_32(seed, (hash.loadR ? 1 : 0) | (hash.loadC ? 2 : 0));
  if (hash.loadR || hash.loadC)
    combine_64(seed, std::hash<NAInt8AttentionKernelDescriptor>{}(hash.kernelDescriptor()));
  return seed;
}

bool NAInt8AttentionDescriptor::splitQueryRanges() const noexcept {
  // Compiling the bulk without partial-row stores avoids the measured uneven-R
  // throughput penalty. The measured surface starts at 64 complete query
  // groups and 64 KV tiles; smaller workloads keep their single dispatch.
  return type == AttentionKernelType::forward &&
      ioPrecision == GEMMOperandPrecision::FP16 && lowPrecisionIntermediates &&
      matrixDimensions[2] == 128 && matrixDimensions[0] >= 4096 &&
      matrixDimensions[1] >= 4096 && matrixDimensions[0] % 16 != 0 &&
      !isCausal && !masked && !isVarlen && !attentionSinks;
}

NAInt8AttentionKernelDescriptor NAInt8AttentionDescriptor::kernelDescriptor() const noexcept {
  // Split outputs fold the KV tail into the main loop and omit the separate
  // short-sequence path. This is a support constraint, below the tuning cutoff.
  CCV_NNC_MFA_PRECONDITION(!splitOutput || matrixDimensions[1] >= 64);
  // A deep head increases register pressure. Use fewer SIMD groups and a
  // shallower D tile, amortizing Q loads across twice as many unmasked keys.
  // Quantization must use the same key tile: each traversal consumes one scale.
  // Preserve decode tuning when there are at most four query tiles.
  const bool wideForward = type == AttentionKernelType::forward &&
      matrixDimensions[2] == 256 && matrixDimensions[0] > 64 &&
      ioPrecision != GEMMOperandPrecision::FP32 &&
      lowPrecisionIntermediates && !isCausal && !masked && !isVarlen && !attentionSinks;
  const uint16_t qScaleTileSize =
      type == AttentionKernelType::forward ? 16 : 32;
  const uint16_t kvScaleTileSize = wideForward ? 128 : 64;
  const bool lowPrecisionBackward =
      type != AttentionKernelType::forward &&
      ioPrecision != GEMMOperandPrecision::FP32;
  const bool splitHeadBackward = lowPrecisionBackward && matrixDimensions[2] == 128;
  const bool splitHeadBackwardKeyValue =
      type == AttentionKernelType::backwardKeyValue && splitHeadBackward;
  const uint16_t blockD =
      splitOutput ? 64 : type == AttentionKernelType::forward ?
      (matrixDimensions[2] >= 192 && !wideForward ? 64 : 32) :
      (type == AttentionKernelType::backwardQuery && splitHeadBackward ?
          32 :
          (splitHeadBackward ? 64 : (uint16_t)matrixDimensions[2]));
  const uint16_t blockC =
      splitOutput ? 64 : type == AttentionKernelType::backwardKeyValue ?
      (splitHeadBackward ? 32 : 16) : (wideForward ? 128 : 64);
  const uint16_t queryBlockC =
      type == AttentionKernelType::backwardQuery && splitHeadBackward ? 32 : blockC;
  const simd::ushort3 blockDimensions { 16, queryBlockC, blockD };
  const uint16_t executionSIMDGroups =
      splitOutput ? 8 : type == AttentionKernelType::forward ?
      (matrixDimensions[2] > 192 && !wideForward ? 16 : 4) :
      (splitHeadBackwardKeyValue ?
          (Hq > Hk ? 8 : 16) :
          4);
  const uint16_t vMeanThreads =
      matrixDimensions[1] <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  const bool has_c_remainder =
      type == AttentionKernelType::forward &&
      (isVarlen || (matrixDimensions[1] % blockDimensions[1]) != 0);
  const bool has_causal_empty_rows =
      type == AttentionKernelType::forward &&
      isCausal &&
      !masked &&
      (isVarlen || matrixDimensions[0] > matrixDimensions[1]);
  auto descriptor = NAInt8AttentionKernelDescriptor(
      blockDimensions,
      (unsigned short)matrixDimensions[2],
      Hq,
      Hk,
      qScaleTileSize,
      kvScaleTileSize,
      executionSIMDGroups,
      vMeanThreads,
      has_c_remainder,
      isCausal ? 0 : 2,
      ioPrecision,
      lowPrecisionIntermediates,
      type,
      scale,
      isCausal,
      masked,
      has_causal_empty_rows,
      isVarlen,
      attentionSinks);
  descriptor.partitioned = partitioned;
  descriptor.splitOutput = splitOutput;
  descriptor.compactGrid = splitOutput;
  if (type == AttentionKernelType::forward &&
      ioPrecision == GEMMOperandPrecision::FP16 && lowPrecisionIntermediates &&
      matrixDimensions[2] == 128 && !isCausal && !masked && !isVarlen && !attentionSinks) {
    const uint64_t queryTile = blockDimensions[0] * executionSIMDGroups;
    const uint64_t queryGroups = splitQueryRanges() ? matrixDimensions[0] / queryTile :
        (uint64_t(matrixDimensions[0]) + queryTile - 1) / queryTile;
    uint64_t paddedQueries = 1, paddedHeads = 1;
    while (paddedQueries < queryGroups) paddedQueries <<= 1;
    while (paddedHeads < Hq) paddedHeads <<= 1;
    // Adjacent query groups for a head help long KV traversals. On shorter
    // sequences, retain Morton order unless at least a third of its padded
    // grid is inactive; dense short grids regressed in the shape sweep.
    // These are measured dispatch bounds, not assumptions about cache size.
    descriptor.compactGrid = matrixDimensions[1] >= 8192 ||
        2 * paddedQueries * paddedHeads >= 3 * queryGroups * Hq;
  }
  descriptor.loadR = loadR;
  descriptor.hasRRemainder = !loadR || isVarlen || matrixDimensions[0] % blockDimensions[0] != 0;
  descriptor.loadC = loadC;
  return descriptor;
}

std::pair<NAInt8AttentionKernelDescriptor, PipelineValue<NAInt8AttentionKernel> *> NAInt8AttentionDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties &dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NAInt8AttentionKernelDescriptor, std::unique_ptr<NAInt8AttentionKernel>> *const libraryCache) const noexcept
{
  (void)dprops;

  auto createKernel =
  [=](const NAInt8AttentionKernelDescriptor& descriptor) -> NAInt8AttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    NAInt8AttentionKernel* kernel = new NAInt8AttentionKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<NAInt8AttentionKernel>(kernel);
    return kernel;
  };

  auto createPipeline =
  [=](NAInt8AttentionKernel* kernel, MTL::FunctionConstantValues* constants, const char* functionNameString) -> MTL::ComputePipelineState* {
    NS::Error* error = nil;
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      pipelineDescriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      if (binaryArchiveToWrite != nullptr) {
        NS::Error* archiveError = nil;
        binaryArchiveToWrite->addComputePipelineFunctions(pipelineDescriptor.get(), &archiveError);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = kernelDescriptor();
  auto kernel = createKernel(kernelDesc);
  const uint32_t q_tiles = (matrixDimensions[0] + kernelDesc.qScaleTileSize - 1) / kernelDesc.qScaleTileSize;
  const uint32_t k_tiles = (matrixDimensions[1] + kernelDesc.kvScaleTileSize - 1) / kernelDesc.kvScaleTileSize;

  auto attentionConstants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t rowDimension = matrixDimensions[0];
  const uint32_t columnDimension = matrixDimensions[1];
  const uint32_t qBatchStride = batchStrides[AttentionOperand::Q].value_or(0);
  const uint32_t kBatchStride = batchStrides[AttentionOperand::K].value_or(0);
  const uint32_t vBatchStride = batchStrides[AttentionOperand::V].value_or(0);
  const uint32_t oBatchStride = batchStrides[AttentionOperand::O].value_or(0);
  const uint32_t dOBatchStride = batchStrides[AttentionOperand::dO].value_or(0);
  const uint32_t dVBatchStride = batchStrides[AttentionOperand::dV].value_or(0);
  const uint32_t dKBatchStride = batchStrides[AttentionOperand::dK].value_or(0);
  const uint32_t dQBatchStride = batchStrides[AttentionOperand::dQ].value_or(0);
  const uint32_t qScaleBatchStride = batchDimension > 1 ? Hq * q_tiles : 0;
  const uint32_t kScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  const uint32_t vScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  const uint32_t dOScaleBatchStride = batchDimension > 1 ? Hq * q_tiles : 0;
  const uint32_t vMeanBatchStride = batchDimension > 1 ? Hk * matrixDimensions[2] : 0;
  const uint32_t maskBatchStride = masked ? this->maskBatchStride : 0;
  const uint32_t blockMaskBatchStride = masked && maskBatchStride > 0 ? q_tiles * k_tiles : 0;
  if (!loadR)
    attentionConstants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::UInteger(0));
  if (!loadC)
    attentionConstants->setConstantValue(&columnDimension, MTL::DataTypeUInt, NS::UInteger(1));
  if (!loadR && !loadC) {
    attentionConstants->setConstantValue(&qBatchStride, MTL::DataTypeUInt, NS::UInteger(2));
    attentionConstants->setConstantValue(&kBatchStride, MTL::DataTypeUInt, NS::UInteger(3));
    attentionConstants->setConstantValue(&vBatchStride, MTL::DataTypeUInt, NS::UInteger(4));
    attentionConstants->setConstantValue(&oBatchStride, MTL::DataTypeUInt, NS::UInteger(5));
  }
  attentionConstants->setConstantValue(&dOBatchStride, MTL::DataTypeUInt, NS::UInteger(6));
  attentionConstants->setConstantValue(&dVBatchStride, MTL::DataTypeUInt, NS::UInteger(7));
  attentionConstants->setConstantValue(&dKBatchStride, MTL::DataTypeUInt, NS::UInteger(8));
  attentionConstants->setConstantValue(&dQBatchStride, MTL::DataTypeUInt, NS::UInteger(9));
  if (!loadR && !loadC) {
    attentionConstants->setConstantValue(&qScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(10));
    attentionConstants->setConstantValue(&kScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(11));
    attentionConstants->setConstantValue(&vScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(12));
  }
  attentionConstants->setConstantValue(&dOScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(13));
  attentionConstants->setConstantValue(&vMeanBatchStride, MTL::DataTypeUInt, NS::UInteger(14));
  if (masked && !loadR && !loadC) {
    attentionConstants->setConstantValue(&maskBatchStride, MTL::DataTypeUInt, NS::UInteger(15));
    attentionConstants->setConstantValue(&blockMaskBatchStride, MTL::DataTypeUInt, NS::UInteger(16));
  }

  auto quantizeConstants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t qSequence = matrixDimensions[0];
  const uint32_t kvSequence = matrixDimensions[1];
  const uint32_t qHeads = Hq;
  const uint32_t kvHeads = Hk;
  const uint32_t qTileSize = kernel->qScaleTileSize;
  const uint32_t kvTileSize = kernel->kvScaleTileSize;
  const uint32_t qBatchStrideQ = batchStrides[AttentionOperand::Q].value_or(0);
  const uint32_t kBatchStrideQ = batchStrides[AttentionOperand::K].value_or(0);
  const uint32_t vBatchStrideQ = batchStrides[AttentionOperand::V].value_or(0);
  const uint32_t kvScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  if (!loadR)
    quantizeConstants->setConstantValue(&qSequence, MTL::DataTypeUInt, NS::UInteger(900));
  if (!loadC)
    quantizeConstants->setConstantValue(&kvSequence, MTL::DataTypeUInt, NS::UInteger(901));
  quantizeConstants->setConstantValue(&qHeads, MTL::DataTypeUInt, NS::UInteger(902));
  quantizeConstants->setConstantValue(&kvHeads, MTL::DataTypeUInt, NS::UInteger(903));
  quantizeConstants->setConstantValue(&qTileSize, MTL::DataTypeUInt, NS::UInteger(904));
  quantizeConstants->setConstantValue(&kvTileSize, MTL::DataTypeUInt, NS::UInteger(905));
  if (!loadR && !loadC) {
    quantizeConstants->setConstantValue(&q_tiles, MTL::DataTypeUInt, NS::UInteger(906));
    quantizeConstants->setConstantValue(&k_tiles, MTL::DataTypeUInt, NS::UInteger(907));
    quantizeConstants->setConstantValue(&qBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(908));
    quantizeConstants->setConstantValue(&kBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(909));
    quantizeConstants->setConstantValue(&vBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(910));
    quantizeConstants->setConstantValue(&qScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(911));
    quantizeConstants->setConstantValue(&kvScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(912));
  }

  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> second;
  NS::SharedPtr<MTL::ComputePipelineState> third;
  NS::SharedPtr<MTL::ComputePipelineState> fourth;
  NS::SharedPtr<MTL::ComputePipelineState> fifth;
  NS::SharedPtr<MTL::ComputePipelineState> sixth;
  switch (type.value) {
  case AttentionKernelType::forward: {
    const uint16_t queryRange = splitQueryRanges() ? 1 : 0;
    attentionConstants->setConstantValue(&queryRange, MTL::DataTypeUShort, NS::UInteger(40));
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_attention"));
    second = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_q"));
    third = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_k"));
    fourth = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_v"));
    fifth = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "compute_v_mean"));
    CCV_NNC_MFA_PRECONDITION(fifth->staticThreadgroupMemoryLength() <= device->maxThreadgroupMemoryLength());
    CCV_NNC_MFA_PRECONDITION(kernel->vMeanThreadgroupSize() <= fifth->maxTotalThreadsPerThreadgroup());
    if (masked) {
      sixth = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "generate_int8_attention_block_mask"));
    } else if (splitQueryRanges()) {
      // Unmasked forward uses the otherwise-empty mask-pipeline slot for its
      // tail. Both pipelines retain the original quantization/batch strides.
      const uint16_t tailRange = 2;
      attentionConstants->setConstantValue(&tailRange, MTL::DataTypeUShort, NS::UInteger(40));
      sixth = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_attention"));
    }
    break;
  }
  case AttentionKernelType::backwardQuery:
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_backward_query"));
    second = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "compute_d"));
    third = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_q"));
    break;
  case AttentionKernelType::backwardKeyValue:
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_backward_keyvalue"));
    break;
  }

  PipelineValue<NAInt8AttentionKernel>* output = new PipelineValue<NAInt8AttentionKernel> { kernel, pipeline };
  output->second = second;
  output->third = third;
  output->fourth = fourth;
  output->fifth = fifth;
  output->sixth = sixth;
  if (partitioned)
    output->seventh = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "merge_int8_attention_partitions"));
  return std::make_pair(kernelDesc, output);
}
