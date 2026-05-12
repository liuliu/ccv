#include "NAAttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

bool NAAttentionKernelDescriptor::operator==(const NAAttentionKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  headDimension == rhs.headDimension &&
  Hq == rhs.Hq && Hk == rhs.Hk &&
  memoryPrecisions == rhs.memoryPrecisions &&
  executionSIMDGroups == rhs.executionSIMDGroups &&
  checkCEdge1 == rhs.checkCEdge1 &&
  bypassThreadgroupMemory == rhs.bypassThreadgroupMemory &&
  isCausal == rhs.isCausal &&
  masked == rhs.masked &&
  isVarlen == rhs.isVarlen &&
  loadC == rhs.loadC &&
  splitKV == rhs.splitKV &&
  type == rhs.type &&
  scale == rhs.scale;
}

std::size_t std::hash<NAAttentionKernelDescriptor>::operator()(const NAAttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 { hash.headDimension, hash.type.value }));
  combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  combine_32(seed, pack_32(simd::uchar4 { (uint8_t)hash.executionSIMDGroups, (uint8_t)hash.checkCEdge1, (uint8_t)hash.bypassThreadgroupMemory, (uint8_t)hash.isCausal }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.masked ? 1 : 0),
      (uint16_t)(hash.isVarlen ? 1 : 0) }));
  combine_32(seed, hash.splitKV);
  combine_32(seed, hash.loadC ? 1 : 0);
  return seed;
}

// MARK: - Initializer

NAAttentionKernelDescriptor::NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale) noexcept
  : NAAttentionKernelDescriptor(blockDimensions, headDimension, Hq, Hk, executionSIMDGroups, checkCEdge1, memoryPrecisions, type, scale, false, false, false, false) {}

NAAttentionKernelDescriptor::NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory) noexcept
  : NAAttentionKernelDescriptor(blockDimensions, headDimension, Hq, Hk, executionSIMDGroups, checkCEdge1, memoryPrecisions, type, scale, bypassThreadgroupMemory, false, false, false) {}

NAAttentionKernelDescriptor::NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory, bool isCausal, bool masked) noexcept
  : NAAttentionKernelDescriptor(blockDimensions, headDimension, Hq, Hk, executionSIMDGroups, checkCEdge1, memoryPrecisions, type, scale, bypassThreadgroupMemory, isCausal, masked, false) {}

NAAttentionKernelDescriptor::NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale, bool bypassThreadgroupMemory, bool isCausal, bool masked, bool isVarlen, uint16_t splitKV, bool loadC) noexcept {
  this->blockDimensions = blockDimensions;
  this->headDimension = headDimension;
  this->Hq = Hq;
  this->Hk = Hk;
  this->executionSIMDGroups = executionSIMDGroups;
  this->checkCEdge1 = checkCEdge1;
  this->bypassThreadgroupMemory = bypassThreadgroupMemory;
  this->memoryPrecisions = memoryPrecisions;
  this->type = type;
  this->scale = scale;
  this->isCausal = isCausal;
  this->masked = masked;
  this->isVarlen = isVarlen;
  this->splitKV = splitKV;
  this->loadC = loadC;
}
