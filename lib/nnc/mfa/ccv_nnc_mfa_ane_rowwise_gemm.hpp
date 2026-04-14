#ifndef GUARD_ccv_nnc_mfa_ane_rowwise_gemm_hpp
#define GUARD_ccv_nnc_mfa_ane_rowwise_gemm_hpp

typedef struct {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t data_type;
  uint32_t fused_bias;
  uint32_t batch_dimension;
  uint32_t batch_stride_a;
  uint32_t batch_stride_c;
} ccv_nnc_mfa_ane_rowwise_gemm_params_t;

typedef struct {
  uint64_t data_type;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t fused_bias;
  uint32_t batch_dimension;
  uint32_t batch_stride_a;
  uint32_t batch_stride_b;
  uint32_t batch_stride_c;
  uint32_t batch_stride_d;
} ccv_nnc_mfa_ane_na_rowwise_gemm_params_t;

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

#ifdef __cplusplus
extern "C" {
#endif

int ccv_nnc_mfa_run_ane_rowwise_gemm(
    ccv_nnc_mfa_context_t* context,
    ccv_nnc_mfa_ane_rowwise_gemm_params_t params,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* stream_context);
int ccv_nnc_mfa_run_ane_na_rowwise_split_gemm(
    ccv_nnc_mfa_context_t* context,
    ccv_nnc_mfa_ane_na_rowwise_gemm_params_t params,
    mtl_buffer_t** tensors,
    size_t* tensor_offsets,
    ccv_nnc_stream_context_t* stream_context);
void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t* context);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
