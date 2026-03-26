#include "NAInt8AttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8AttentionKernelDescriptor::operator==(const NAInt8AttentionKernelDescriptor& rhs) const {
  return
    simd_all(blockDimensions == rhs.blockDimensions) &&
    headDimension == rhs.headDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    executionSIMDGroups == rhs.executionSIMDGroups &&
    vMeanThreads == rhs.vMeanThreads &&
    hasCRemainder == rhs.hasCRemainder &&
    threadBarrierOverC == rhs.threadBarrierOverC &&
    ioPrecision == rhs.ioPrecision &&
    scale == rhs.scale;
}

std::size_t std::hash<NAInt8AttentionKernelDescriptor>::operator()(const NAInt8AttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 { hash.headDimension, hash.executionSIMDGroups }));
  combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  combine_32(seed, hash.vMeanThreads);
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.hasCRemainder ? 1 : 0),
      (uint16_t)(hash.threadBarrierOverC ? 1 : 0) }));
  combine_32(seed, (uint16_t)hash.ioPrecision.value);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

NAInt8AttentionKernelDescriptor::NAInt8AttentionKernelDescriptor(
    simd::ushort3 blockDimensions,
    unsigned short headDimension,
    unsigned short Hq,
    unsigned short Hk,
    uint16_t executionSIMDGroups,
    uint16_t vMeanThreads,
    bool hasCRemainder,
    bool threadBarrierOverC,
    GEMMOperandPrecision ioPrecision,
    float scale) noexcept
{
  this->blockDimensions = blockDimensions;
  this->headDimension = headDimension;
  this->Hq = Hq;
  this->Hk = Hk;
  this->executionSIMDGroups = executionSIMDGroups;
  this->vMeanThreads = vMeanThreads;
  this->hasCRemainder = hasCRemainder;
  this->threadBarrierOverC = threadBarrierOverC;
  this->ioPrecision = ioPrecision;
  this->scale = scale;
}
