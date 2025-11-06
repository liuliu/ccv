#include "SegmentedGEMMPrologueKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

bool SegmentedGEMMPrologueKernelDescriptor::operator==(const SegmentedGEMMPrologueKernelDescriptor& rhs) const {
  return
  memoryPrecisions == rhs.memoryPrecisions &&
  splitK == rhs.splitK &&
  useBias == rhs.useBias;
}

std::size_t std::hash<SegmentedGEMMPrologueKernelDescriptor>::operator()(const SegmentedGEMMPrologueKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  combine_32(seed, pack_32(simd::uchar4 { hash.useBias, 0, 0, 0 }));
  return seed;
}

// MARK: - Initializer

SegmentedGEMMPrologueKernelDescriptor::SegmentedGEMMPrologueKernelDescriptor(GEMMOperandPrecisions memoryPrecisions, bool useBias, uint16_t splitK) noexcept {
  this->memoryPrecisions = memoryPrecisions;
  this->useBias = useBias;
  this->splitK = splitK;
}
