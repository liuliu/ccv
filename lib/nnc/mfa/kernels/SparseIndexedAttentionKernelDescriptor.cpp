#include "SparseIndexedAttentionKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool SparseIndexedAttentionKernelDescriptor::operator==(const SparseIndexedAttentionKernelDescriptor& rhs) const {
  return memoryPrecision == rhs.memoryPrecision && attentionSinks == rhs.attentionSinks;
}

std::size_t std::hash<SparseIndexedAttentionKernelDescriptor>::operator()(const SparseIndexedAttentionKernelDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_32(seed, pack_32(simd::ushort2 { hash.memoryPrecision.value, (unsigned short)(hash.attentionSinks ? 1 : 0) }));
  return seed;
}
