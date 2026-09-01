#ifndef GUARD_ccv_nnc_mfa_segmented_scaled_swiglu_hpp
#define GUARD_ccv_nnc_mfa_segmented_scaled_swiglu_hpp

typedef struct {
  uint64_t data_type;
  uint32_t format;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t loadM;
  uint32_t expert_count;
  uint32_t bincount;
  uint8_t use_neural_accelerators;
  float clamp;
} ccv_nnc_mfa_segmented_scaled_swiglu_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_segmented_scaled_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_scaled_swiglu_params_t params);
// Tensors: gate weights, up weights, activation, indices, counts, route weight, output.
void ccv_nnc_mfa_encode_segmented_scaled_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_segmented_scaled_swiglu_params_t params,
  mtl_command_batch_t* command_batch,
  mtl_buffer_t** tensors,
  size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
