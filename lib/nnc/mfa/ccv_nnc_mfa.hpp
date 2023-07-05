#ifndef GUARD_ccv_nnc_mfa_h
#define GUARD_ccv_nnc_mfa_h

#include "nnc/ccv_nnc.h"
#ifdef __cplusplus
#include "3rdparty/metal-cpp/Metal.hpp"

#define CCV_NNC_MFA_ASSERT(error) \
if (error) { ccv_nnc_mfa_fatal_error(error, __LINE__, __FILE__, __FUNCTION__); } \

void ccv_nnc_mfa_fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name);

class ccv_nnc_mfa_context {
public:
  bool supported;
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  
  ccv_nnc_mfa_context(MTL::Device* device);
  
  typedef struct {
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
  } gemm_hash;
};

typedef ccv_nnc_mfa_context ccv_nnc_mfa_context_t;
typedef MTL::Buffer mtl_buffer_t;
typedef MTL::ComputeCommandEncoder mtl_compute_command_encoder_t;

extern "C" {
#else // __cplusplus
typedef void ccv_nnc_mfa_context_t;
typedef void mtl_buffer_t;
typedef void mtl_compute_command_encoder_t;
#endif // __cplusplus

ccv_nnc_mfa_context_t* ccv_nnc_init_mfa_context(ccv_nnc_mfa_context_t* device);
void ccv_nnc_deinit_mfa_context(ccv_nnc_mfa_context_t* mfa_context);
int ccv_nnc_mfa_context_supported(ccv_nnc_mfa_context_t* mfa_context);

typedef struct {
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
  
  uint32_t batch_dim_a[CCV_NNC_MAX_DIM_ALLOC];
  uint32_t batch_dim_b[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mfa_gemm_params_t;

// It's okay to call these multiple times for the same set of parameters; MFA
// keeps an internal cache. What's not okay is calling it eagerly, just before
// dispatching the command. Doing so is technically supported for debugging, but
// only for debugging - shift such calls to NNC graph 'autotune' ASAP.
void ccv_nnc_mfa_async_prepare_attention(ccv_nnc_mfa_context_t* mfa_context /* arguments for tensors and other metadata */);
void ccv_nnc_mfa_async_prepare_gemm(ccv_nnc_mfa_context_t* mfa_context, ccv_nnc_mfa_gemm_params_t params);

// TODO: The encode parts should accept an array of MTL::Buffer* and offsets inside them.
// Requires that you already called "async_prepare_gemm" or the equivalent
// function for attention, forcing you to create them asynchronously or
// otherwise create some very wierd code.
void ccv_nnc_mfa_encode_attention(ccv_nnc_mfa_context_t* mfa_context, mtl_compute_command_encoder_t* compute_encoder /* arguments for tensors and other metadata */);
void ccv_nnc_mfa_encode_gemm(ccv_nnc_mfa_context_t* mfa_context, ccv_nnc_mfa_gemm_params_t params, mtl_compute_command_encoder_t* compute_encoder, mtl_buffer_t** tensors, int64_t* tensor_offsets);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
