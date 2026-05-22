#include "NAInt8AttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8AttentionKernelDescriptor::operator==(const NAInt8AttentionKernelDescriptor& rhs) const {
  return
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
    scale == rhs.scale;
}

std::size_t std::hash<NAInt8AttentionKernelDescriptor>::operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 {
      hash.headDimension,
      (uint16_t)hash.executionSIMDGroups }));
  combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  combine_32(seed, pack_32(simd::ushort2 { hash.qScaleTileSize, hash.kvScaleTileSize }));
  combine_32(seed, hash.vMeanThreads);
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.hasCRemainder ? 1 : 0),
      hash.threadBarrierEveryC }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.ioPrecision.value,
      (uint16_t)(hash.lowPrecisionIntermediates ? 1 : 0) }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.type.value,
      (uint16_t)(hash.isCausal ? 1 : 0) }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.masked ? 1 : 0),
      (uint16_t)(hash.hasCausalEmptyRows ? 1 : 0) }));
  combine_32(seed, hash.isVarlen ? 1 : 0);
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
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
    bool attentionSinks) noexcept
{
  this->blockDimensions = blockDimensions;
  this->type = type;
  this->headDimension = headDimension;
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
