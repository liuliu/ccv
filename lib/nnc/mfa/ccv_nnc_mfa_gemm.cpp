#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
using namespace ccv::nnc;

// MARK: - C++

std::size_t std::hash<mfa::gemm::hash>::operator()(mfa::gemm::hash const& hash) const noexcept {
  std::size_t seed = 0;
  mfa::hash::combine_64(seed, hash.datatype);
  mfa::hash::combine_32(seed, hash.M);
  mfa::hash::combine_32(seed, hash.N);
  mfa::hash::combine_32(seed, hash.K);
  mfa::hash::combine_32(seed, uint32_t(hash.A_trans));
  mfa::hash::combine_32(seed, uint32_t(hash.B_trans));
  mfa::hash::combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.alpha));
  mfa::hash::combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.beta));
  mfa::hash::combine_32(seed, uint32_t(hash.batched));
  mfa::hash::combine_32(seed, uint32_t(hash.fused_activation));
  return seed;
}

// MARK: - C

// TODO: Define the implementation of ccv_nnc_mfa_async_prepare_gemm
// TODO: Define the implementation of ccv_nnc_mfa_encode_gemm
