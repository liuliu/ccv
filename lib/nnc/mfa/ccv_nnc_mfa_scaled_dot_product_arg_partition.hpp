#ifndef GUARD_ccv_nnc_mfa_scaled_dot_product_arg_partition_hpp
#define GUARD_ccv_nnc_mfa_scaled_dot_product_arg_partition_hpp

typedef struct {
  uint64_t data_type;
  uint32_t T;
  uint32_t C;
  uint32_t H;
  uint32_t D;
  uint32_t kth;
  uint32_t compression_ratio;
  float scale;
  uint8_t is_causal;
  uint8_t use_neural_accelerators;
} ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t;

typedef struct {
  uint32_t T;
  uint32_t C;
  uint32_t kth;
  uint32_t compression_ratio;
  uint8_t is_causal;
} ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params);
void ccv_nnc_mfa_encode_scaled_dot_product_arg_partition(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
void ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition_enumerate(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t params);
void ccv_nnc_mfa_encode_scaled_dot_product_arg_partition_enumerate(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
