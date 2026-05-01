#ifndef GUARD_ccv_nnc_mfa_cmul_hpp
#define GUARD_ccv_nnc_mfa_cmul_hpp

typedef struct {
  uint8_t conjugate;
  uint64_t data_type_a;
  uint64_t data_type_b;
  uint64_t data_type_c;
  uint32_t astride[3];
  uint32_t bstride[3];
  uint32_t cstride[3];
  uint32_t dim[4];
} ccv_nnc_mfa_cmul_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void ccv_nnc_mfa_prepare_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cmul_params_t params);
void ccv_nnc_mfa_encode_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cmul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
