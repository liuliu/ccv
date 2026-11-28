#ifndef GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_hpp
#define GUARD_ccv_nnc_mfa_dequantize_8i_rowwise_x_hpp

// TMCQ (CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 ... N32, see ccv_nnc.h): the rate n (n / 16 bits per weight) of a TMCQ format
// without modifier bits (no kernel takes them), 0 for every other format. Its data is the 1024-byte pair table, then
// (rows / 16) (cols / 16) tiles of 16 n bits, tile-row-major, then the row scales at the next 128 bytes.
static inline uint32_t ccv_nnc_mfa_tmcq_rate(const uint32_t format)
{
	return format >= CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 && format <= CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 ? 16 + 2 * (format - CCV_NNC_QX_8I_ROWWISE_TMCQ_N16) : 0;
}

// Bytes of the pair table before the tiles: 1024 for TMCQ, 0 for the other codecs.
static inline uint64_t ccv_nnc_mfa_tmcq_codebook_size(const uint32_t format)
{
	return ccv_nnc_mfa_tmcq_rate(format) ? 1024 : 0;
}

// Tiles of a rows x cols TMCQ weight (rows % 16 == 0, cols % 16 == 0), as ccv_nnc_tensor_data_size counts them.
static inline uint64_t ccv_nnc_mfa_tmcq_tile_count(const uint64_t rows, const uint64_t cols)
{
	return (rows / 16) * (cols / 16);
}

typedef struct {
	uint64_t data_type;
	uint32_t format;
	uint64_t row_length;
	uint64_t length;
} ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t;

typedef struct {
	uint64_t data_type;
	uint32_t format;
	uint64_t row_length;
	uint64_t rows_per_expert;
	uint64_t expert_count; // Number of expert weight matrices.
	uint64_t bincount; // Number of entries in indices and counts.
} ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t;

#ifdef __cplusplus
extern "C" {
#endif

void ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params);
void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);
size_t ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params);
void ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
}
#endif

#endif
