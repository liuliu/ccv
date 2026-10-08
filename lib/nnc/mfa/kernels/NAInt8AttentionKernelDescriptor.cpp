#include "NAInt8AttentionKernelDescriptor.hpp"
#include "NAInt8AttentionDescriptor.hpp"
#include "NAInt8AttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8AttentionKernelDescriptor::operator==(const NAInt8AttentionKernelDescriptor& rhs) const {
  return
    loadR == rhs.loadR && loadC == rhs.loadC &&
    hasRRemainder == rhs.hasRRemainder &&
    simd_all(blockDimensions == rhs.blockDimensions) &&
    type == rhs.type &&
    headDimension == rhs.headDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    qScaleTileSize == rhs.qScaleTileSize &&
    kvScaleTileSize == rhs.kvScaleTileSize &&
    executionSIMDGroups == rhs.executionSIMDGroups &&
    vMeanThreads == rhs.vMeanThreads &&
    hasCRemainder == rhs.hasCRemainder &&
    threadBarrierEveryC == rhs.threadBarrierEveryC &&
    ioPrecision == rhs.ioPrecision &&
    lowPrecisionIntermediates == rhs.lowPrecisionIntermediates &&
    isCausal == rhs.isCausal &&
    masked == rhs.masked &&
    isVarlen == rhs.isVarlen &&
    hasCausalEmptyRows == rhs.hasCausalEmptyRows &&
    attentionSinks == rhs.attentionSinks &&
    qkHadamard == rhs.qkHadamard &&
    qkMeanCorrection == rhs.qkMeanCorrection &&
    outputTileSize == rhs.outputTileSize &&
    mortonTraversal == rhs.mortonTraversal &&
    scale == rhs.scale;
}

std::size_t std::hash<NAInt8AttentionKernelDescriptor>::operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  seed = combine_32(seed, (hash.loadR ? 1 : 0) | (hash.loadC ? 2 : 0) | (hash.hasRRemainder ? 4 : 0));
  seed = combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  seed = combine_32(seed, pack_32(simd::ushort2 {
      hash.headDimension,
      (uint16_t)hash.executionSIMDGroups }));
  seed = combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  seed = combine_32(seed, pack_32(simd::ushort2 { hash.qScaleTileSize, hash.kvScaleTileSize }));
  seed = combine_32(seed, hash.vMeanThreads);
  seed = combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.hasCRemainder ? 1 : 0),
      hash.threadBarrierEveryC }));
  seed = combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.ioPrecision.value,
      (uint16_t)(hash.lowPrecisionIntermediates ? 1 : 0) }));
  seed = combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.type.value,
      (uint16_t)(hash.isCausal ? 1 : 0) }));
  seed = combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.masked ? 1 : 0),
      (uint16_t)(hash.hasCausalEmptyRows ? 1 : 0) }));
  seed = combine_32(seed, hash.isVarlen ? 1 : 0);
  seed = combine_32(seed, hash.attentionSinks ? 1 : 0);
  seed = combine_32(seed, hash.qkHadamard ? 1 : 0);
  seed = combine_32(seed, hash.qkMeanCorrection ? 1 : 0);
  seed = combine_32(seed, hash.outputTileSize);
  seed = combine_32(seed, hash.mortonTraversal ? 1 : 0);
  seed = combine_32(seed, uint32_t(std::hash<float>{}(hash.scale)));
  return seed;
}

NAInt8AttentionKernelDescriptor::NAInt8AttentionKernelDescriptor(
    simd::ushort3 blockDimensions,
    unsigned short headDimension,
    unsigned short Hq,
    unsigned short Hk,
    uint16_t qScaleTileSize,
    uint16_t kvScaleTileSize,
    uint16_t executionSIMDGroups,
    uint16_t vMeanThreads,
    bool hasCRemainder,
    uint16_t threadBarrierEveryC,
    GEMMOperandPrecision ioPrecision,
    bool lowPrecisionIntermediates,
    AttentionKernelType type,
    float scale,
    bool isCausal,
    bool masked,
    bool hasCausalEmptyRows,
    bool isVarlen,
    bool mortonTraversal,
    bool attentionSinks) noexcept
{
  this->blockDimensions = blockDimensions;
  this->type = type;
  this->headDimension = headDimension;
  this->outputTileSize = headDimension;
  this->mortonTraversal = mortonTraversal;
  this->Hq = Hq;
  this->Hk = Hk;
  this->qScaleTileSize = qScaleTileSize;
  this->kvScaleTileSize = kvScaleTileSize;
  this->executionSIMDGroups = executionSIMDGroups;
  this->vMeanThreads = vMeanThreads;
  this->hasCRemainder = hasCRemainder;
  this->threadBarrierEveryC = threadBarrierEveryC;
  this->ioPrecision = ioPrecision;
  this->lowPrecisionIntermediates = lowPrecisionIntermediates;
  this->scale = scale;
  this->isCausal = isCausal;
  this->masked = masked;
  this->isVarlen = isVarlen;
  this->hasCausalEmptyRows = hasCausalEmptyRows;
  this->attentionSinks = attentionSinks;
}

NAInt8AttentionKernelDescriptor::NAInt8AttentionKernelDescriptor(
    const NAInt8AttentionDescriptor& descriptor, const DeviceProperties& dprops) noexcept
  : headDimension(descriptor.matrixDimensions[2]),
    Hq(descriptor.Hq),
    Hk(descriptor.Hk),
    qScaleTileSize(descriptor.type == AttentionKernelType::forward ? 16 : 32),
    kvScaleTileSize(64),
    threadBarrierEveryC(descriptor.isCausal ? 0 : 2),
    ioPrecision(descriptor.ioPrecision),
    lowPrecisionIntermediates(descriptor.lowPrecisionIntermediates),
    type(descriptor.type),
    scale(descriptor.scale),
    isCausal(descriptor.isCausal),
    masked(descriptor.masked),
    isVarlen(descriptor.isVarlen),
    hasCausalEmptyRows(type == AttentionKernelType::forward && isCausal && !masked &&
        (isVarlen || descriptor.matrixDimensions[0] > descriptor.matrixDimensions[1])),
    loadR(descriptor.loadR),
    loadC(descriptor.loadC),
    attentionSinks(descriptor.attentionSinks),
    qkHadamard(descriptor.qkHadamard),
    qkMeanCorrection(descriptor.qkMeanCorrection)
{
  (void)dprops;
  const uint32_t D = descriptor.matrixDimensions[2];
  const bool lowPrecisionBackward =
      type != AttentionKernelType::forward && ioPrecision != GEMMOperandPrecision::FP32;
  const bool splitHeadBackward = lowPrecisionBackward && D == 128;
  const bool splitHeadBackwardKeyValue =
      type == AttentionKernelType::backwardKeyValue && splitHeadBackward;
  const uint16_t blockD =
      type == AttentionKernelType::forward ? (D >= 192 ? 64 : 32) :
      (type == AttentionKernelType::backwardQuery && splitHeadBackward ?
          32 :
          (splitHeadBackward ? 64 : (uint16_t)D));
  const uint16_t blockC =
      type == AttentionKernelType::backwardKeyValue ?
      (splitHeadBackward ? 32 : 16) : 64;
  const uint16_t queryBlockC =
      type == AttentionKernelType::backwardQuery && splitHeadBackward ? 32 : blockC;
  blockDimensions = simd::ushort3 { 16, queryBlockC, blockD };
  // Wider heads benefit from limiting each group's FP32 output state to 128
  // channels. Each tile repeats QK over the complete head.
  outputTileSize = type == AttentionKernelType::forward && D > 192 ? 128 : D;
  executionSIMDGroups =
      type == AttentionKernelType::forward ?
      (D > 192 ? 8 : 4) :
      (splitHeadBackwardKeyValue ?
          (Hq > Hk ? 8 : 16) :
          4);
  vMeanThreads =
      descriptor.matrixDimensions[1] <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  hasCRemainder = type == AttentionKernelType::forward &&
      (isVarlen || (descriptor.matrixDimensions[1] % blockDimensions[1]) != 0);
  hasRRemainder = !loadR || isVarlen || descriptor.matrixDimensions[0] % blockDimensions[0] != 0;
  // Untiled dynamic causal kernels retain a measured advantage with Morton.
  // Output tiling keeps head-major order; backward keeps its existing Morton order.
  mortonTraversal = type != AttentionKernelType::forward ||
      (outputTileSize == headDimension && isCausal && (loadR || loadC));
}
