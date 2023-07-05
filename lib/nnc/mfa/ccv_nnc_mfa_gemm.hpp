#ifndef GUARD_ccv_nnc_mfa_gemm_hpp
#define GUARD_ccv_nnc_mfa_gemm_hpp

#ifdef __cplusplus
#include "3rdparty/metal-cpp/Dispatch.hpp"
#include "3rdparty/metal-cpp/Metal.hpp"

namespace ccv {
namespace nnc {
namespace mfa {
namespace gemm {

class hash {
public:
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t A_trans;
  uint8_t B_trans;
  float alpha;
  float beta;
  uint8_t batched;
  uint8_t fused_activation;
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pso;
  Dispatch::Semaphore semaphore;
  bool finished = false;
  
public:
  uint16_t threadgroup_memory_length;
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(hash hash);
};

} // namespace gemm
} // namespace mfa
} // namespace nnc
} // namespace ccv

template<>
struct std::hash<ccv::nnc::mfa::gemm::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::gemm::hash& hash) const noexcept;
};

extern "C" {
#endif // __cplusplus

typedef struct {
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t A_trans;
  uint8_t B_trans;
  float alpha;
  float beta;
  uint8_t batched;
  uint8_t fused_activation;
  
  uint32_t batch_dim_a[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dim_b[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mfa_gemm_params_t;

void ccv_nnc_mfa_async_prepare_gemm(ccv_nnc_mfa_context_t* mfa_context, ccv_nnc_mfa_gemm_params_t params);
void ccv_nnc_mfa_encode_gemm(ccv_nnc_mfa_context_t* mfa_context, ccv_nnc_mfa_gemm_params_t params, mtl_compute_command_encoder_t* compute_encoder, mtl_buffer_t** tensors, int64_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
