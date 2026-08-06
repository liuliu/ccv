#include "NASparseIndexedAttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool NASparseIndexedAttentionKernelDescriptor::operator==(const NASparseIndexedAttentionKernelDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  attentionSinks == rhs.attentionSinks &&
  denseOnly == rhs.denseOnly &&
  loadRows == rhs.loadRows &&
  variant == rhs.variant;
}

std::size_t std::hash<NASparseIndexedAttentionKernelDescriptor>::operator()(const NASparseIndexedAttentionKernelDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_32(seed, pack_32(simd::ushort2 { hash.memoryPrecision.value, (unsigned short)((hash.attentionSinks ? 1 : 0) | (hash.denseOnly ? 2 : 0) | (hash.loadRows ? 4 : 0)) }));
  combine_32(seed, static_cast<uint32_t>(hash.variant));
  return seed;
}
