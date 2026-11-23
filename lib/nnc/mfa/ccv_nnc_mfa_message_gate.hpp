#ifndef GUARD_ccv_nnc_mfa_message_gate_hpp
#define GUARD_ccv_nnc_mfa_message_gate_hpp

// The mix width a threadgroup holds for each of its rows.
#define CCV_NNC_MFA_MESSAGE_GATE_MAX_RANK (256)
// Up to this many rows, one kernel computes the mix (the normalizations and the down projection).
// More rows normalize into a buffer the down GEMM reads.
#define CCV_NNC_MFA_MESSAGE_GATE_FEW_ROWS (16)

typedef struct {
  uint64_t data_type; // The state, the message and the output.
  uint64_t down_data_type; // The down projection's, or its row scales' when quantized.
  uint64_t weight_data_type; // The forget and gain projections', or their row scales' when quantized.
  uint8_t down_quantized; // Whether the down projection is 8-bit rowwise.
  uint8_t quantized; // Whether the forget and gain projections are 8-bit rowwise.
  uint8_t mix_given; // Whether the mix buffer holds the mix already, rather than the down projection computing it.
  uint32_t rows;
  uint32_t channels;
  uint32_t rank;
  float epsilon;
  // From each quantized projection's offset to its row scales.
  uint64_t down_scale_offset;
  uint64_t forget_scale_offset;
  uint64_t gain_scale_offset;
} ccv_nnc_mfa_message_gate_params_t;

#ifdef __cplusplus
extern "C" {
#endif

// Buffers: the state, the message, the down, forget and gain projections, the output, then the mix
// [rows, R] in FP32.
void ccv_nnc_mfa_encode_message_gate(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_message_gate_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
// Buffers: the state, the message, then [rows, 2C] in FP32 for their RMS-normalized rows side by
// side.
void ccv_nnc_mfa_encode_message_gate_normalize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_message_gate_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
