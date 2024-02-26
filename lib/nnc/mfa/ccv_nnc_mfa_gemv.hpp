#ifndef GUARD_ccv_nnc_mfa_gemv_hpp
#define GUARD_ccv_nnc_mfa_gemv_hpp

typedef struct {
  uint64_t data_type;
  uint32_t nrows;
  uint32_t ncols;
  uint8_t fused_bias;
} ccv_nnc_mfa_gemv_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace gemv {

class hash {
public:
  uint64_t data_type;
  uint32_t nrows;
  uint32_t ncols;
  uint8_t fused_bias;

  hash(ccv_nnc_mfa_gemv_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> gemv_pso;
  
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace gemv
} // namespace mfa
} // namespace nnc
} // namespace ccv

std::ostream& operator<<(std::ostream& os, const ccv::nnc::mfa::gemv::hash& hash);

template<>
struct std::hash<ccv::nnc::mfa::gemv::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::gemv::hash& hash) const noexcept;
};

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemv_params_t params);
void ccv_nnc_mfa_encode_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
