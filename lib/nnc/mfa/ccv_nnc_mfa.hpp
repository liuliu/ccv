#include "3rdparty/metal-cpp/Metal.hpp"

#define CCV_NNC_MFA_ASSERT(error) \
if (error) { ccv_nnc_mfa_fatal_error(error, __LINE__, __FILE__, __FUNCTION__); } \

void ccv_nnc_mfa_fatal_error(NS::Error* error, int line, const char *file_name, const char *function_name);

// It's okay to call these multiple times for the same set of parameters; MFA
// keeps an internal cache. What's not okay is calling it eagerly, just before
// dispatching the command. Doing so is technically supported for debugging, but
// only for debugging - shift such calls to NNC graph 'autotune' ASAP.
void ccv_nnc_mfa_async_prepare_gemm(void* mfa_context /* arguments for tensors and other metadata */);
void ccv_nnc_mfa_async_prepare_multi_head_attention(void* mfa_context /* arguments for tensors and other metadata */);

// Requires that you already called "async_prepare_gemm" or the equivalent
// function for attention, forcing you to create them asynchronously or
// otherwise create some very wierd code.
void ccv_nnc_mfa_encode_gemm(void* mfa_context, void* compute_encoder /* arguments for tensors and other metadata */);
void ccv_nnc_mfa_encode_multi_head_attention(void* mfa_context, void* compute_encoder /* arguments for tensors and other metadata */);

class ccv_nnc_mfa_context {
public:
  bool supported;
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::Library> library;
  
  ccv_nnc_mfa_context(MTL::Device* device);
};
