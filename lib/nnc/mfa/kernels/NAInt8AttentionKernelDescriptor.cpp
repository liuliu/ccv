#include "NAInt8AttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAInt8AttentionKernelDescriptor::operator==(const NAInt8AttentionKernelDescriptor& rhs) const {
  return
    simd_all(blockDimensions == rhs.blockDimensions) &&
    headDimension == rhs.headDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    executionSIMDGroups == rhs.executionSIMDGroups &&
    checkCEdge1 == rhs.checkCEdge1 &&
    useInt8QK == rhs.useInt8QK &&
    useQKScales == rhs.useQKScales &&
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
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.checkCEdge1 ? 1 : 0),
      (uint16_t)((hash.useInt8QK ? 1 : 0) | (hash.useQKScales ? 2 : 0)) }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.threadBarrierOverC ? 1 : 0),
      (uint16_t)hash.ioPrecision.value }));
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

NAInt8AttentionKernelDescriptor::NAInt8AttentionKernelDescriptor(
    simd::ushort3 blockDimensions,
    unsigned short headDimension,
    unsigned short Hq,
    unsigned short Hk,
    uint16_t executionSIMDGroups,
    bool checkCEdge1,
    bool useInt8QK,
    bool useQKScales,
    bool threadBarrierOverC,
    GEMMOperandPrecision ioPrecision,
    float scale) noexcept
{
  this->blockDimensions = blockDimensions;
  this->headDimension = headDimension;
  this->Hq = Hq;
  this->Hk = Hk;
  this->executionSIMDGroups = executionSIMDGroups;
  this->checkCEdge1 = checkCEdge1;
  this->useInt8QK = useInt8QK;
  this->useQKScales = useQKScales;
  this->threadBarrierOverC = threadBarrierOverC;
  this->ioPrecision = ioPrecision;
  this->scale = scale;
}
