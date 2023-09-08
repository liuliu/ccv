#ifndef GUARD_ccv_nnc_mfa_depalettize_hpp
#define GUARD_ccv_nnc_mfa_depalettize_hpp

typedef struct {
  uint64_t data_type;
  int qbits;
  int number_in_blocks;
  uint64_t length;
} ccv_nnc_mfa_depalettize_params_t;

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Dispatch.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

namespace ccv {
namespace nnc {
namespace mfa {
namespace depalettize {

class hash {
public:
  uint64_t data_type;
  int qbits;
  int number_in_blocks;
  uint64_t length;

  hash(ccv_nnc_mfa_depalettize_params_t);
  
  bool operator==(const hash& rhs) const;
};

class pipeline {
public:
  NS::SharedPtr<MTL::ComputePipelineState> depalettize_pso;
  
  MTL::Size grid_size;
  MTL::Size group_size;
  
  pipeline(context* context, hash hash);
};

} // namespace depalettize
} // namespace mfa
} // namespace nnc
} // namespace ccv

std::ostream& operator<<(std::ostream& os, const ccv::nnc::mfa::depalettize::hash& hash);

template<>
struct std::hash<ccv::nnc::mfa::depalettize::hash>
{
  std::size_t operator()(const ccv::nnc::mfa::depalettize::hash& hash) const noexcept;
};

extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params);
void ccv_nnc_mfa_encode_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
