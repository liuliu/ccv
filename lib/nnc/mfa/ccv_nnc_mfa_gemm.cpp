#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
using namespace ccv::nnc;

// MARK: - C

void ccv_nnc_mfa_async_prepare_gemm(ccv_nnc_mfa_context_t* mfa_context, ccv_nnc_mfa_gemm_params_t params) {
  mfa::gemm::hash hash {
    .data_type = params.data_type,
    .M = params.M,
    .N = params.N,
    .K = params.K,
    .A_trans = params.A_trans,
    .B_trans = params.B_trans,
    .alpha = params.alpha,
    .beta = params.beta,
    .batched = params.batched,
    .fused_activation = params.fused_activation
  };
  
  auto* gemm_cache = &(mfa_context->gemm_cache);
  auto found = gemm_cache->find(hash);
  if (found != gemm_cache->end()) {
    // Already cached.
    return;
  }
  
  // TODO: Link this to the pipeline constructor, enter into the hash map.
}

// MARK: - C++

bool mfa::gemm::hash::operator==(const mfa::gemm::hash& hash) const {
  return (memcmp(this, &hash, sizeof(hash)) == 0);
}

std::size_t std::hash<mfa::gemm::hash>::operator()(const mfa::gemm::hash& hash) const noexcept {
  std::size_t seed = 0;
  mfa::hash::combine_64(seed, hash.data_type);
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

mfa::gemm::pipeline::pipeline(mfa::gemm::hash hash) : semaphore(0) {
  // TODO: Create the pipeline
  CCV_NNC_MFA_PRECONDITION(hash.data_type == MTL::DataTypeFloat || hash.data_type == MTL::DataTypeHalf)
}
