#ifndef GUARD_ccv_nnc_mfa_depthwise_conv_hpp
#define GUARD_ccv_nnc_mfa_depthwise_conv_hpp

typedef struct {
	uint64_t data_type;
	uint32_t batch_size;
	uint32_t channels;
	uint32_t input_dimensions[2]; // height, width
	uint32_t output_dimensions[2]; // height, width
	uint32_t filter_dimensions[2]; // height, width
	uint32_t stride_dimensions[2]; // height, width
	uint32_t dilation_dimensions[2]; // height, width
	uint32_t padding_top;
	uint32_t padding_left;
	uint8_t fused_bias;
} ccv_nnc_mfa_depthwise_conv_params_t;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Depthwise 2D convolution (groups == input channels == output channels) on NHWC tensors.
// Tensors: input [N, H, W, C], weights [C, 1, kH, kW], output [N, OH, OW, C], optional bias [C].
void ccv_nnc_mfa_encode_depthwise_conv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depthwise_conv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif
