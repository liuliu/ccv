#include "AttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

bool AttentionKernelDescriptor::operator==(const AttentionKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  cacheState == rhs.cacheState &&
  headDimension == rhs.headDimension &&
  Hq == rhs.Hq && Hk == rhs.Hk &&
  memoryPrecisions == rhs.memoryPrecisions &&
  (preferAsyncCache == rhs.preferAsyncCache) &&
  (preferAsyncLoad == rhs.preferAsyncLoad) &&
  registerPrecisions == rhs.registerPrecisions &&
  transposeState == rhs.transposeState &&
  leadingDimensions == rhs.leadingDimensions &&
  type == rhs.type &&
  scale == rhs.scale;
}

std::size_t std::hash<AttentionKernelDescriptor>::operator()(const AttentionKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_32(seed, pack_32(simd::ushort2 { hash.headDimension, hash.type.value }));
  combine_32(seed, pack_32(simd::ushort2 { hash.Hq, hash.Hk }));
  combine_32(seed, pack_32(simd::uchar4 { hash.preferAsyncCache, hash.preferAsyncLoad, 0, 0 }));
  return 0;
}

// MARK: - Initializer

AttentionKernelDescriptor::AttentionKernelDescriptor(simd::ushort3 blockDimensions, AttentionOperands<bool> cacheState, unsigned short headDimension, unsigned short Hq, unsigned short Hk, AttentionOperands<GEMMOperandPrecision> memoryPrecisions, bool preferAsyncCache, bool preferAsyncLoad, AttentionOperands<GEMMOperandPrecision> registerPrecisions, AttentionOperands<bool> transposeState, AttentionOperands<unsigned short> leadingDimensions, AttentionKernelType type, float scale) noexcept {
  this->blockDimensions = blockDimensions;
  this->cacheState = cacheState;
  this->headDimension = headDimension;
  this->Hq = Hq;
  this->Hk = Hk;
  this->memoryPrecisions = memoryPrecisions;
  this->preferAsyncCache = preferAsyncCache;
  this->preferAsyncLoad = preferAsyncLoad;
  this->registerPrecisions = registerPrecisions;
  this->transposeState = transposeState;
  this->leadingDimensions = leadingDimensions;
  this->type = type;
  this->scale = scale;
}
