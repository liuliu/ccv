#ifndef GUARD_ccv_nnc_mfa_attention_hpp
#define GUARD_ccv_nnc_mfa_attention_hpp

typedef struct {
  uint64_t data_type;
  uint32_t R;
  uint32_t C;
  uint32_t H;
  uint32_t D;
  uint8_t Q_trans;
  uint8_t K_trans;
  uint8_t V_trans;
  uint8_t O_trans;
  float alpha;
  uint8_t batched;
  uint8_t masked;
  
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
  uint64_t data_type;
  uint32_t R;
  uint32_t C;
  uint32_t H;
  uint32_t D;
  uint8_t Q_trans;
  uint8_t K_trans;
  uint8_t V_trans;
  uint8_t O_trans;
  float alpha;
  uint8_t batched;
  uint8_t masked;
  
  hash(ccv_nnc_mfa_attention_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
  Dispatch::Semaphore* semaphore;
  MTL::ComputePipelineState* attention_pso;
  MTL::ComputePipelineState* generate_mask_pso;
  
  simd::uchar4 flags;
  uint16_t threadgroup_memory_length; // use 48B for the mask-generating shader
  MTL::Size grid_size; // overwrite Y with the H dimension for the attention shader
  MTL::Size group_size;
  
public:
  pipeline(context* context, hash hash, bool async);
  ~pipeline();
  
  // This is a potentially blocking function. Call it before accessing any of
  // the property getters.
  void wait();
  
  MTL::ComputePipelineState* get_attention_pso() const;
  MTL::ComputePipelineState* get_generate_mask_pso() const;
  simd::uchar4 get_flags() const;
  uint16_t get_threadgroup_memory_length() const;
  MTL::Size get_grid_size() const;
  MTL::Size get_group_size() const;
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

void ccv_nnc_mfa_async_prepare_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_params_t params);
void ccv_nnc_mfa_sync_prepare_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_params_t params);
void ccv_nnc_mfa_encode_attention(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_attention_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
