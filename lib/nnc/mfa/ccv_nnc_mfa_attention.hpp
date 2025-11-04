#ifndef GUARD_ccv_nnc_mfa_attention_hpp
#define GUARD_ccv_nnc_mfa_attention_hpp

typedef struct {
  uint8_t Q_trans;
  uint8_t K_trans;
  uint8_t V_trans;
  uint8_t O_trans;
  uint8_t batched;
  uint8_t masked;
  uint8_t upcast;
  uint8_t type;
  uint8_t use_neural_accelerators;
  uint32_t R;
  uint32_t C;
  uint32_t Hq;
  uint32_t Hk;
  uint32_t D;
  float alpha;
  uint64_t data_type;

  // Since grouped queries are not supported yet, assume Q, K, V, and O all have
  // the same batch dimensions.
  uint32_t batch_dims_q[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dims_mask[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mfa_attention_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace attention {

class hash {
public:
  uint8_t Q_trans;
  uint8_t K_trans;
  uint8_t V_trans;
  uint8_t O_trans;
  uint8_t batched;
  uint8_t masked;
  uint8_t upcast;
  uint8_t type;
  uint32_t R;
  uint32_t C;
  uint32_t Hq;
  uint32_t Hk;
  uint32_t D;
  float alpha;
  uint64_t data_type;
  
  hash(ccv_nnc_mfa_attention_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> attention_pso;
  NS::SharedPtr<MTL::ComputePipelineState> generate_block_mask_pso;
  
  uint16_t threadgroup_memory_length;
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace attention
} // namespace mfa
} // namespace nnc
} // namespace ccv

std::ostream& operator<<(std::ostream& os, const ccv::nnc::mfa::attention::hash& hash);

template<>
struct std::hash<ccv::nnc::mfa::attention::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::attention::hash& hash) const noexcept;
};

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_params_t params);
void ccv_nnc_mfa_encode_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
