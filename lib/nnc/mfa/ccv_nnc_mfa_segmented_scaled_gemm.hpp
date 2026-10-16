#ifndef GUARD_ccv_nnc_mfa_segmented_scaled_gemm_hpp
#define GUARD_ccv_nnc_mfa_segmented_scaled_gemm_hpp

typedef struct {
  uint64_t data_type;
  uint32_t format; // Packed 8i row-wise X format decoded per weight tile, or zero for unpacked INT8.
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t originalM;
  uint8_t fused_bias;
  uint8_t use_neural_accelerators;
  uint8_t loadM;
  uint32_t expert_count; // Number of expert weight matrices.
  uint32_t bincount; // Number of entries in indices and counts.
} ccv_nnc_mfa_segmented_scaled_gemm_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_segmented_scaled_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_segmented_scaled_gemm_params_t params);
size_t ccv_nnc_mfa_segmented_scaled_gemm_reserved_scratch_size(ccv_nnc_mfa_segmented_scaled_gemm_params_t params);
void ccv_nnc_mfa_encode_segmented_scaled_gemm(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_segmented_scaled_gemm_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
// Whether packed weights in this format should be decoded per tile rather than expanded into scratch.
int ccv_nnc_mfa_segmented_scaled_gemm_fuses_packed_format(uint32_t format, uint64_t M, uint32_t bincount);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
