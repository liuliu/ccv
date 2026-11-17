#ifndef GUARD_ccv_nnc_mfa_swiglu_hpp
#define GUARD_ccv_nnc_mfa_swiglu_hpp

typedef struct {
  uint64_t data_type;
  // 0 for Q8_0 row-wise weights, or CCV_NNC_QX_8I_ROWWISE_Q5_K / Q6_K.
  uint32_t format;
  // Activation rows, at most 6; 0 means 1.
  uint32_t M;
  uint32_t N;
  uint32_t K;
  float clamp;
  float gate_soft_cap;
  float up_soft_cap;
} ccv_nnc_mfa_int8_swiglu_params_t;

typedef struct {
  uint64_t data_type;
  uint32_t format;
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint8_t loadM;
  uint8_t use_neural_accelerators;
  float clamp;
  float gate_soft_cap;
  float up_soft_cap;
  // The external codebook of a TMCQ format whose bits 16 - 23 hold its log2 entry count (ccv_nnc_mfa_codebook_format),
  // for both gate and up; NULL for the built-in table.
  mtl_buffer_t* codebook;
} ccv_nnc_mfa_scaled_swiglu_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_int8_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_int8_swiglu_params_t params);
// Tensors: gate weights, up weights, activation, output.
void ccv_nnc_mfa_encode_int8_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_int8_swiglu_params_t params,
  mtl_command_batch_t* command_batch,
  mtl_buffer_t** tensors,
  size_t* tensor_offsets);

void ccv_nnc_mfa_prepare_scaled_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_scaled_swiglu_params_t params);
// Tensors: gate weights, up weights, activation, output.
void ccv_nnc_mfa_encode_scaled_swiglu(
  ccv_nnc_mfa_context_t* context,
  ccv_nnc_mfa_scaled_swiglu_params_t params,
  mtl_command_batch_t* command_batch,
  mtl_buffer_t** tensors,
  size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
