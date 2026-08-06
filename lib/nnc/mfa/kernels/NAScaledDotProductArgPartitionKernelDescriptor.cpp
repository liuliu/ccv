#include "NAScaledDotProductArgPartitionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool NAScaledDotProductArgPartitionKernelDescriptor::operator==(const NAScaledDotProductArgPartitionKernelDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  kth == rhs.kth &&
  scoreBlockM == rhs.scoreBlockM &&
  scoreBlockN == rhs.scoreBlockN &&
  scoreSIMDGroups == rhs.scoreSIMDGroups &&
  loadC == rhs.loadC;
}

std::size_t std::hash<NAScaledDotProductArgPartitionKernelDescriptor>::operator()(const NAScaledDotProductArgPartitionKernelDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.kth }));
  combine_32(seed, pack_32(simd::ushort2 { hash.scoreBlockM, hash.scoreBlockN }));
  combine_32(seed, pack_32(simd::ushort2 { hash.scoreSIMDGroups, (unsigned short)(hash.loadC ? 1 : 0) }));
  return seed;
}
