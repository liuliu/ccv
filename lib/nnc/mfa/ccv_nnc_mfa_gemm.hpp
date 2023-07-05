#ifndef GUARD_ccv_nnc_mfa_gemm_hpp
#define GUARD_ccv_nnc_mfa_gemm_hpp

namespace ccv {
namespace nnc {
namespace mfa {
namespace gemm {

class hash {
public:
  uint64_t datatype;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t A_trans;
  uint8_t B_trans;
  float alpha;
  float beta;
  uint8_t batched;
  uint8_t fused_activation;
};

class pipeline {
  MTL::ComputePipelineState pso;
  Dispatch::Semaphore semaphore;
  
public:
  pipeline(hash hash);
};

} // namespace gemm
} // namespace mfa
} // namespace nnc
} // namespace ccv

template<>
struct std::hash<ccv::nnc::mfa::gemm::hash>
{
  std::size_t operator()(ccv::nnc::mfa::gemm::hash const& hash) const noexcept;
};

#endif
