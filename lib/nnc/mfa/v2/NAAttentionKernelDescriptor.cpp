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
  type == rhs.type &&
  scale == rhs.scale;
}

std::size_t std::hash<NAAttentionKernelDescriptor>::operator()(const NAAttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 { hash.headDimension, hash.type.value }));
  combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  return seed;
}

// MARK: - Initializer

NAAttentionKernelDescriptor::NAAttentionKernelDescriptor(simd::ushort3 blockDimensions, unsigned short headDimension, unsigned short Hq, unsigned short Hk, uint16_t executionSIMDGroups, bool checkCEdge1, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, AttentionKernelType type, float scale) noexcept {
  this->blockDimensions = blockDimensions;
  this->headDimension = headDimension;
  this->Hq = Hq;
  this->Hk = Hk;
  this->executionSIMDGroups = executionSIMDGroups;
  this->checkCEdge1 = checkCEdge1;
  this->memoryPrecisions = memoryPrecisions;
  this->type = type;
  this->scale = scale;
}
