#ifndef GUARD_ccv_nnc_mfa_normalization_hpp
#define GUARD_ccv_nnc_mfa_normalization_hpp

typedef struct {
  uint64_t data_type;
  uint32_t channel_count;
  uint32_t channel_groups;
  uint32_t sequence_count;
  float epsilon;
  uint8_t elementwise_affine;
  uint8_t scale_translation_batched;
  uint8_t normalization_type;
  uint8_t reuse_saved_statistics;
  
  uint32_t batch_dims_data[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dims_scale_translation[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t src_batch_stride;
  uint32_t dst_batch_stride;
} ccv_nnc_mfa_normalization_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace normalization {

  enum Type: uint8_t {
    layer_norm = 0,
    group_norm = 1,
    rmsnorm = 2
  };

class hash {
public:
  uint64_t data_type;
  uint32_t channel_count;
  uint32_t channel_groups;
  uint32_t sequence_count;
  float epsilon;
  uint8_t elementwise_affine;
  uint8_t scale_translation_batched;
  Type normalization_type;
  uint8_t reuse_saved_statistics;
  uint32_t src_batch_stride;
  uint32_t dst_batch_stride;
  
  hash(ccv_nnc_mfa_normalization_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> sampling_pso;
  NS::SharedPtr<MTL::ComputePipelineState> normalization_pso;
  
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace normalization
} // namespace mfa
} // namespace nnc
} // namespace ccv

std::ostream& operator<<(std::ostream& os, const ccv::nnc::mfa::normalization::hash& hash);

template<>
struct std::hash<ccv::nnc::mfa::normalization::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::normalization::hash& hash) const noexcept;
};

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params);
void ccv_nnc_mfa_encode_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
