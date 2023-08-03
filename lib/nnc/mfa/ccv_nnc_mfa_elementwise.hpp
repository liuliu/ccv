#ifndef GUARD_ccv_nnc_mfa_elementwise_hpp
#define GUARD_ccv_nnc_mfa_elementwise_hpp

typedef struct {
  const char* operation_name;
  uint64_t data_type;
  uint32_t reduction_dim;
  
  uint32_t batch_dims[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mfa_elementwise_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace elementwise {

class hash {
public:
  uint64_t data_type;
  uint32_t operation_id;
  uint32_t reduction_dim;
  
  hash(ccv_nnc_mfa_elementwise_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> pso;
  
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace elementwise
} // namespace mfa
} // namespace nnc
} // namespace ccv

std::ostream& operator<<(std::ostream& os, const ccv::nnc::mfa::elementwise::hash& hash);

template<>
struct std::hash<ccv::nnc::mfa::elementwise::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::elementwise::hash& hash) const noexcept;
};


extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_elementwise(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_elementwise_params_t params);
void ccv_nnc_mfa_encode_elementwise(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_elementwise_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
