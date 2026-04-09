#ifndef GUARD_ccv_nnc_mfa_ane_rowwise_internal_hpp
#define GUARD_ccv_nnc_mfa_ane_rowwise_internal_hpp

#include "ccv_nnc_mfa_defines.hpp"
#include "kernels/ANERowwiseTransformDescriptor.hpp"
#include "kernels/ANERowwiseTransformKernel.hpp"
#include "kernels/PipelineValue.hpp"

#ifdef __cplusplus
mtl_device_t* ccv_nnc_mfa_context_device(ccv_nnc_mfa_context_t* context);
void* ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(ccv_nnc_mfa_context_t* context);
void ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(ccv_nnc_mfa_context_t* context, void* cache);
PipelineValue<ANERowwiseTransformKernel>* ccv_nnc_mfa_prepare_ane_rowwise_transform(
    ccv_nnc_mfa_context_t* context,
    ANERowwiseTransformDescriptor descriptor);
#endif

#endif
