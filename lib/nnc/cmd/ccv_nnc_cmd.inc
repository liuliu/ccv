static ccv_nnc_cmd_init_t init_map[] = {
	{.name = "CCV_NNC_SOFTMAX_FORWARD", .cmd = 0xc969a252},
	{.name = "CCV_NNC_SOFTMAX_BACKWARD", .cmd = 0xc969a253},
	{.name = "CCV_NNC_INDEX_SELECT_FORWARD", .cmd = 0x7ee7771e},
	{.name = "CCV_NNC_INDEX_SELECT_BACKWARD", .cmd = 0x7ee7771f},
	{.name = "CCV_NNC_NMS_FORWARD", .cmd = 0xdba26106},
	{.name = "CCV_NNC_NMS_BACKWARD", .cmd = 0xdba26107},
	{.name = "CCV_NNC_MIN_FORWARD", .cmd = 0x972fbd26},
	{.name = "CCV_NNC_MIN_BACKWARD", .cmd = 0x972fbd27},
	{.name = "CCV_NNC_REDUCE_MAX_FORWARD", .cmd = 0x80f1a506},
	{.name = "CCV_NNC_REDUCE_MAX_BACKWARD", .cmd = 0x80f1a507},
	{.name = "CCV_NNC_MASKED_FILL_FORWARD", .cmd = 0x7f992d84},
	{.name = "CCV_NNC_MASKED_FILL_BACKWARD", .cmd = 0x7f992d85},
	{.name = "CCV_NNC_SMOOTH_L1_FORWARD", .cmd = 0x4e428e},
	{.name = "CCV_NNC_SMOOTH_L1_BACKWARD", .cmd = 0x4e428f},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD", .cmd = 0xc26b7b5e},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD", .cmd = 0xc26b7b5f},
	{.name = "CCV_NNC_BINARY_CROSSENTROPY_FORWARD", .cmd = 0xcd2107ec},
	{.name = "CCV_NNC_BINARY_CROSSENTROPY_BACKWARD", .cmd = 0xcd2107ed},
	{.name = "CCV_NNC_LSTM_FORWARD", .cmd = 0xc5cb998c},
	{.name = "CCV_NNC_LSTM_BACKWARD", .cmd = 0xc5cb998d},
	{.name = "CCV_NNC_COMM_REDUCE_FORWARD", .cmd = 0x3434ead8},
	{.name = "CCV_NNC_COMM_REDUCE_BACKWARD", .cmd = 0x3434ead9},
	{.name = "CCV_NNC_GEMM_FORWARD", .cmd = 0x7e87d00c},
	{.name = "CCV_NNC_GEMM_BACKWARD", .cmd = 0x7e87d00d},
	{.name = "CCV_NNC_TRANSPOSE_FORWARD", .cmd = 0xb4d506e0},
	{.name = "CCV_NNC_TRANSPOSE_BACKWARD", .cmd = 0xb4d506e1},
	{.name = "CCV_NNC_MAX_POOL_FORWARD", .cmd = 0x7bec9360},
	{.name = "CCV_NNC_MAX_POOL_BACKWARD", .cmd = 0x7bec9361},
	{.name = "CCV_NNC_DROPOUT_FORWARD", .cmd = 0x7f2dc3e4},
	{.name = "CCV_NNC_DROPOUT_BACKWARD", .cmd = 0x7f2dc3e5},
	{.name = "CCV_NNC_REDUCE_NORM2_FORWARD", .cmd = 0xb3034e16},
	{.name = "CCV_NNC_REDUCE_NORM2_BACKWARD", .cmd = 0xb3034e17},
	{.name = "CCV_NNC_HISTOGRAM_FORWARD", .cmd = 0xc5473e44},
	{.name = "CCV_NNC_HISTOGRAM_BACKWARD", .cmd = 0xc5473e45},
	{.name = "CCV_NNC_SET_FORWARD", .cmd = 0x2b070804},
	{.name = "CCV_NNC_SET_BACKWARD", .cmd = 0x2b070805},
	{.name = "CCV_NNC_RELU_FORWARD", .cmd = 0xc51eaa80},
	{.name = "CCV_NNC_RELU_BACKWARD", .cmd = 0xc51eaa81},
	{.name = "CCV_NNC_GELU_FORWARD", .cmd = 0xb1527ab8},
	{.name = "CCV_NNC_GELU_BACKWARD", .cmd = 0xb1527ab9},
	{.name = "CCV_NNC_MAX_FORWARD", .cmd = 0xdf6f014c},
	{.name = "CCV_NNC_MAX_BACKWARD", .cmd = 0xdf6f014d},
	{.name = "CCV_NNC_ARGMAX_FORWARD", .cmd = 0x68af2804},
	{.name = "CCV_NNC_ARGMAX_BACKWARD", .cmd = 0x68af2805},
	{.name = "CCV_NNC_SCALAR_MUL_FORWARD", .cmd = 0x8b4d86aa},
	{.name = "CCV_NNC_SCALAR_MUL_BACKWARD", .cmd = 0x8b4d86ab},
	{.name = "CCV_NNC_ROI_ALIGN_FORWARD", .cmd = 0xfef55168},
	{.name = "CCV_NNC_ROI_ALIGN_BACKWARD", .cmd = 0xfef55169},
	{.name = "CCV_NNC_COMM_ALLREDUCE_FORWARD", .cmd = 0x75c8d340},
	{.name = "CCV_NNC_COMM_ALLREDUCE_BACKWARD", .cmd = 0x75c8d341},
	{.name = "CCV_NNC_MUL_FORWARD", .cmd = 0x24721a46},
	{.name = "CCV_NNC_MUL_BACKWARD", .cmd = 0x24721a47},
	{.name = "CCV_NNC_LAYER_NORM_FORWARD", .cmd = 0xbed3c264},
	{.name = "CCV_NNC_LAYER_NORM_BACKWARD", .cmd = 0xbed3c265},
	{.name = "CCV_NNC_CLAMP_FORWARD", .cmd = 0x2640d854},
	{.name = "CCV_NNC_CLAMP_BACKWARD", .cmd = 0x2640d855},
	{.name = "CCV_NNC_UPSAMPLE_FORWARD", .cmd = 0x73875556},
	{.name = "CCV_NNC_UPSAMPLE_BACKWARD", .cmd = 0x73875557},
	{.name = "CCV_NNC_TANH_FORWARD", .cmd = 0x6a62be30},
	{.name = "CCV_NNC_TANH_BACKWARD", .cmd = 0x6a62be31},
	{.name = "CCV_NNC_LAMB_FORWARD", .cmd = 0x450edb1a},
	{.name = "CCV_NNC_LAMB_BACKWARD", .cmd = 0x450edb1b},
	{.name = "CCV_NNC_CONVOLUTION_FORWARD", .cmd = 0x254d05f4},
	{.name = "CCV_NNC_CONVOLUTION_BACKWARD", .cmd = 0x254d05f5},
	{.name = "CCV_NNC_MSE_FORWARD", .cmd = 0x6904a9a2},
	{.name = "CCV_NNC_MSE_BACKWARD", .cmd = 0x6904a9a3},
	{.name = "CCV_NNC_AVERAGE_POOL_FORWARD", .cmd = 0x51267ab8},
	{.name = "CCV_NNC_AVERAGE_POOL_BACKWARD", .cmd = 0x51267ab9},
	{.name = "CCV_NNC_RMSPROP_FORWARD", .cmd = 0x9c886b1c},
	{.name = "CCV_NNC_RMSPROP_BACKWARD", .cmd = 0x9c886b1d},
	{.name = "CCV_NNC_EWSQRT_FORWARD", .cmd = 0x8870a61e},
	{.name = "CCV_NNC_EWSQRT_BACKWARD", .cmd = 0x8870a61f},
	{.name = "CCV_NNC_EWEXP_FORWARD", .cmd = 0xd784b170},
	{.name = "CCV_NNC_EWEXP_BACKWARD", .cmd = 0xd784b171},
	{.name = "CCV_NNC_SGD_FORWARD", .cmd = 0xe650ad26},
	{.name = "CCV_NNC_SGD_BACKWARD", .cmd = 0xe650ad27},
	{.name = "CCV_NNC_BATCH_NORM_FORWARD", .cmd = 0x5419819c},
	{.name = "CCV_NNC_BATCH_NORM_BACKWARD", .cmd = 0x5419819d},
	{.name = "CCV_NNC_EWSUM_FORWARD", .cmd = 0xe21a2c4c},
	{.name = "CCV_NNC_EWSUM_BACKWARD", .cmd = 0xe21a2c4d},
	{.name = "CCV_NNC_COMPRESSION_LSSC_FORWARD", .cmd = 0x17ea8f72},
	{.name = "CCV_NNC_COMPRESSION_LSSC_BACKWARD", .cmd = 0x17ea8f73},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_FORWARD", .cmd = 0xe4a2b192},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_BACKWARD", .cmd = 0xe4a2b193},
	{.name = "CCV_NNC_RANDOM_UNIFORM_FORWARD", .cmd = 0xa0cd1d5e},
	{.name = "CCV_NNC_RANDOM_UNIFORM_BACKWARD", .cmd = 0xa0cd1d5f},
	{.name = "CCV_NNC_REDUCE_SUM_FORWARD", .cmd = 0x52970f06},
	{.name = "CCV_NNC_REDUCE_SUM_BACKWARD", .cmd = 0x52970f07},
	{.name = "CCV_NNC_GROUP_NORM_FORWARD", .cmd = 0x17deb074},
	{.name = "CCV_NNC_GROUP_NORM_BACKWARD", .cmd = 0x17deb075},
	{.name = "CCV_NNC_DATA_TRANSFER_FORWARD", .cmd = 0x12d21e1a},
	{.name = "CCV_NNC_DATA_TRANSFER_BACKWARD", .cmd = 0x12d21e1b},
	{.name = "CCV_NNC_ADAM_FORWARD", .cmd = 0xe30099dc},
	{.name = "CCV_NNC_ADAM_BACKWARD", .cmd = 0xe30099dd},
	{.name = "CCV_NNC_RANDOM_NORMAL_FORWARD", .cmd = 0x7062c8b4},
	{.name = "CCV_NNC_RANDOM_NORMAL_BACKWARD", .cmd = 0x7062c8b5},
	{.name = "CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD", .cmd = 0xd9e0e4a},
	{.name = "CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD", .cmd = 0xd9e0e4b},
	{.name = "CCV_NNC_SWISH_FORWARD", .cmd = 0x583d90c2},
	{.name = "CCV_NNC_SWISH_BACKWARD", .cmd = 0x583d90c3},
	{.name = "CCV_NNC_REDUCE_MEAN_FORWARD", .cmd = 0xf23556c6},
	{.name = "CCV_NNC_REDUCE_MEAN_BACKWARD", .cmd = 0xf23556c7},
	{.name = "CCV_NNC_EWPROD_FORWARD", .cmd = 0xee07e8fe},
	{.name = "CCV_NNC_EWPROD_BACKWARD", .cmd = 0xee07e8ff},
	{.name = "CCV_NNC_ADD_FORWARD", .cmd = 0x58fb3664},
	{.name = "CCV_NNC_ADD_BACKWARD", .cmd = 0x58fb3665},
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD", .cmd = 0x1eb327a2},
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD", .cmd = 0x1eb327a3},
	{.name = "CCV_NNC_DATATYPE_CONVERSION_FORWARD", .cmd = 0xd873e38c},
	{.name = "CCV_NNC_DATATYPE_CONVERSION_BACKWARD", .cmd = 0xd873e38d},
	{.name = "CCV_NNC_SIGMOID_FORWARD", .cmd = 0xf2f69650},
	{.name = "CCV_NNC_SIGMOID_BACKWARD", .cmd = 0xf2f69651},
	{.name = "CCV_NNC_EWDIV_FORWARD", .cmd = 0x1cd2fa18},
	{.name = "CCV_NNC_EWDIV_BACKWARD", .cmd = 0x1cd2fa19},
	{.name = "CCV_NNC_COMM_BROADCAST_FORWARD", .cmd = 0x830eee},
	{.name = "CCV_NNC_COMM_BROADCAST_BACKWARD", .cmd = 0x830eef},
	{.name = "CCV_NNC_EWLOG_FORWARD", .cmd = 0xf4191bf2},
	{.name = "CCV_NNC_EWLOG_BACKWARD", .cmd = 0xf4191bf3},
};

static ccv_nnc_cmd_backend_init_t backend_init_map[] = {
	{.name = "CCV_NNC_BACKEND_GPU_CUBLAS", .backend = 0x9b8cfed},
	{.name = "CCV_NNC_BACKEND_GPU_NCCL", .backend = 0x7afed9c7},
	{.name = "CCV_NNC_BACKEND_CPU_REF", .backend = 0x3d9883e5},
	{.name = "CCV_NNC_BACKEND_GPU_CUDNN", .backend = 0x854b679a},
	{.name = "CCV_NNC_BACKEND_CPU_OPT", .backend = 0x46deb194},
	{.name = "CCV_NNC_BACKEND_GPU_REF", .backend = 0x5f19790a},
	{.name = "CCV_NNC_BACKEND_MPS", .backend = 0xb2f325e2},
};

static inline int _ccv_nnc_cmd_ph(const uint32_t cmd)
{
	switch ((cmd >> 27) % 9)
	{
		case 0:
			return ((((cmd >> 5) % 56) + 2) << 1) | (cmd & 1);
		case 1:
			return ((((cmd >> 1) % 19) + 30) << 1) | (cmd & 1);
		case 2:
			return ((((cmd >> 1) % 15) + 38) << 1) | (cmd & 1);
		case 3:
			return ((((cmd >> 1) % 13) + 46) << 1) | (cmd & 1);
		case 4:
			return ((((cmd >> 1) % 23) + 10) << 1) | (cmd & 1);
		case 5:
			return ((((cmd >> 6) % 38) + 13) << 1) | (cmd & 1);
		case 6:
			return ((((cmd >> 1) % 21) + 0) << 1) | (cmd & 1);
		case 7:
			return ((((cmd >> 9) % 9) + 0) << 1) | (cmd & 1);
		case 8:
		default:
			return ((((cmd >> 7) % 27) + 17) << 1) | (cmd & 1);
	}
}

static inline int _ccv_nnc_cmd_backend_ph(const uint32_t backend)
{
	switch ((backend >> 0) % 1)
	{
		case 0:
		default:
			return ((backend >> 15) % 7) + 0;
	}
}

void _register_command_CCV_NNC_SOFTMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD(ccv_nnc_cmd_registry_t* const registry);

void _register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
#ifdef HAVE_CUDA
void _register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
#endif
#ifdef HAVE_MPS
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
#endif

static inline void _ccv_nnc_cmd_init(void)
{
	_register_command_CCV_NNC_SOFTMAX_FORWARD(&init_map[0].registry);
	_register_command_CCV_NNC_SOFTMAX_BACKWARD(&init_map[1].registry);
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD(&init_map[2].registry);
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD(&init_map[3].registry);
	_register_command_CCV_NNC_NMS_FORWARD(&init_map[4].registry);
	_register_command_CCV_NNC_NMS_BACKWARD(&init_map[5].registry);
	_register_command_CCV_NNC_MIN_FORWARD(&init_map[6].registry);
	_register_command_CCV_NNC_MIN_BACKWARD(&init_map[7].registry);
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD(&init_map[8].registry);
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD(&init_map[9].registry);
	_register_command_CCV_NNC_MASKED_FILL_FORWARD(&init_map[10].registry);
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD(&init_map[11].registry);
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD(&init_map[12].registry);
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD(&init_map[13].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(&init_map[14].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(&init_map[15].registry);
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD(&init_map[16].registry);
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD(&init_map[17].registry);
	_register_command_CCV_NNC_LSTM_FORWARD(&init_map[18].registry);
	_register_command_CCV_NNC_LSTM_BACKWARD(&init_map[19].registry);
	_register_command_CCV_NNC_COMM_REDUCE_FORWARD(&init_map[20].registry);
	_register_command_CCV_NNC_COMM_REDUCE_BACKWARD(&init_map[21].registry);
	_register_command_CCV_NNC_GEMM_FORWARD(&init_map[22].registry);
	_register_command_CCV_NNC_GEMM_BACKWARD(&init_map[23].registry);
	_register_command_CCV_NNC_TRANSPOSE_FORWARD(&init_map[24].registry);
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD(&init_map[25].registry);
	_register_command_CCV_NNC_MAX_POOL_FORWARD(&init_map[26].registry);
	_register_command_CCV_NNC_MAX_POOL_BACKWARD(&init_map[27].registry);
	_register_command_CCV_NNC_DROPOUT_FORWARD(&init_map[28].registry);
	_register_command_CCV_NNC_DROPOUT_BACKWARD(&init_map[29].registry);
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD(&init_map[30].registry);
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD(&init_map[31].registry);
	_register_command_CCV_NNC_HISTOGRAM_FORWARD(&init_map[32].registry);
	_register_command_CCV_NNC_HISTOGRAM_BACKWARD(&init_map[33].registry);
	_register_command_CCV_NNC_SET_FORWARD(&init_map[34].registry);
	_register_command_CCV_NNC_SET_BACKWARD(&init_map[35].registry);
	_register_command_CCV_NNC_RELU_FORWARD(&init_map[36].registry);
	_register_command_CCV_NNC_RELU_BACKWARD(&init_map[37].registry);
	_register_command_CCV_NNC_GELU_FORWARD(&init_map[38].registry);
	_register_command_CCV_NNC_GELU_BACKWARD(&init_map[39].registry);
	_register_command_CCV_NNC_MAX_FORWARD(&init_map[40].registry);
	_register_command_CCV_NNC_MAX_BACKWARD(&init_map[41].registry);
	_register_command_CCV_NNC_ARGMAX_FORWARD(&init_map[42].registry);
	_register_command_CCV_NNC_ARGMAX_BACKWARD(&init_map[43].registry);
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD(&init_map[44].registry);
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD(&init_map[45].registry);
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD(&init_map[46].registry);
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD(&init_map[47].registry);
	_register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD(&init_map[48].registry);
	_register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD(&init_map[49].registry);
	_register_command_CCV_NNC_MUL_FORWARD(&init_map[50].registry);
	_register_command_CCV_NNC_MUL_BACKWARD(&init_map[51].registry);
	_register_command_CCV_NNC_LAYER_NORM_FORWARD(&init_map[52].registry);
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD(&init_map[53].registry);
	_register_command_CCV_NNC_CLAMP_FORWARD(&init_map[54].registry);
	_register_command_CCV_NNC_CLAMP_BACKWARD(&init_map[55].registry);
	_register_command_CCV_NNC_UPSAMPLE_FORWARD(&init_map[56].registry);
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD(&init_map[57].registry);
	_register_command_CCV_NNC_TANH_FORWARD(&init_map[58].registry);
	_register_command_CCV_NNC_TANH_BACKWARD(&init_map[59].registry);
	_register_command_CCV_NNC_LAMB_FORWARD(&init_map[60].registry);
	_register_command_CCV_NNC_LAMB_BACKWARD(&init_map[61].registry);
	_register_command_CCV_NNC_CONVOLUTION_FORWARD(&init_map[62].registry);
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD(&init_map[63].registry);
	_register_command_CCV_NNC_MSE_FORWARD(&init_map[64].registry);
	_register_command_CCV_NNC_MSE_BACKWARD(&init_map[65].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD(&init_map[66].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(&init_map[67].registry);
	_register_command_CCV_NNC_RMSPROP_FORWARD(&init_map[68].registry);
	_register_command_CCV_NNC_RMSPROP_BACKWARD(&init_map[69].registry);
	_register_command_CCV_NNC_EWSQRT_FORWARD(&init_map[70].registry);
	_register_command_CCV_NNC_EWSQRT_BACKWARD(&init_map[71].registry);
	_register_command_CCV_NNC_EWEXP_FORWARD(&init_map[72].registry);
	_register_command_CCV_NNC_EWEXP_BACKWARD(&init_map[73].registry);
	_register_command_CCV_NNC_SGD_FORWARD(&init_map[74].registry);
	_register_command_CCV_NNC_SGD_BACKWARD(&init_map[75].registry);
	_register_command_CCV_NNC_BATCH_NORM_FORWARD(&init_map[76].registry);
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD(&init_map[77].registry);
	_register_command_CCV_NNC_EWSUM_FORWARD(&init_map[78].registry);
	_register_command_CCV_NNC_EWSUM_BACKWARD(&init_map[79].registry);
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD(&init_map[80].registry);
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD(&init_map[81].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(&init_map[82].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(&init_map[83].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(&init_map[84].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(&init_map[85].registry);
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD(&init_map[86].registry);
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD(&init_map[87].registry);
	_register_command_CCV_NNC_GROUP_NORM_FORWARD(&init_map[88].registry);
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD(&init_map[89].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD(&init_map[90].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(&init_map[91].registry);
	_register_command_CCV_NNC_ADAM_FORWARD(&init_map[92].registry);
	_register_command_CCV_NNC_ADAM_BACKWARD(&init_map[93].registry);
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD(&init_map[94].registry);
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD(&init_map[95].registry);
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD(&init_map[96].registry);
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD(&init_map[97].registry);
	_register_command_CCV_NNC_SWISH_FORWARD(&init_map[98].registry);
	_register_command_CCV_NNC_SWISH_BACKWARD(&init_map[99].registry);
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD(&init_map[100].registry);
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD(&init_map[101].registry);
	_register_command_CCV_NNC_EWPROD_FORWARD(&init_map[102].registry);
	_register_command_CCV_NNC_EWPROD_BACKWARD(&init_map[103].registry);
	_register_command_CCV_NNC_ADD_FORWARD(&init_map[104].registry);
	_register_command_CCV_NNC_ADD_BACKWARD(&init_map[105].registry);
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(&init_map[106].registry);
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(&init_map[107].registry);
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD(&init_map[108].registry);
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD(&init_map[109].registry);
	_register_command_CCV_NNC_SIGMOID_FORWARD(&init_map[110].registry);
	_register_command_CCV_NNC_SIGMOID_BACKWARD(&init_map[111].registry);
	_register_command_CCV_NNC_EWDIV_FORWARD(&init_map[112].registry);
	_register_command_CCV_NNC_EWDIV_BACKWARD(&init_map[113].registry);
	_register_command_CCV_NNC_COMM_BROADCAST_FORWARD(&init_map[114].registry);
	_register_command_CCV_NNC_COMM_BROADCAST_BACKWARD(&init_map[115].registry);
	_register_command_CCV_NNC_EWLOG_FORWARD(&init_map[116].registry);
	_register_command_CCV_NNC_EWLOG_BACKWARD(&init_map[117].registry);

	_register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[60].backends[2]));
	_register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[61].backends[2]));
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[46].backends[2]));
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[47].backends[2]));
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[38].backends[2]));
	_register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[39].backends[2]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[56].backends[2]));
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[57].backends[2]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[22].backends[2]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[22].backends[4]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[23].backends[2]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[23].backends[4]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[104].backends[2]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[105].backends[2]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[50].backends[2]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[51].backends[2]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[44].backends[2]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[45].backends[2]));
	_register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[68].backends[2]));
	_register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[69].backends[2]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[34].backends[2]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[35].backends[2]));
	_register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[10].backends[2]));
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[11].backends[2]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[90].backends[2]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[91].backends[2]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[82].backends[2]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[83].backends[2]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[24].backends[2]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[25].backends[2]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[108].backends[2]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[109].backends[2]));
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[12].backends[2]));
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[13].backends[2]));
	_register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[64].backends[2]));
	_register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[65].backends[2]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[16].backends[2]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[17].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[106].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[107].backends[2]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[86].backends[2]));
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[87].backends[2]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[100].backends[2]));
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[101].backends[2]));
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[8].backends[2]));
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[9].backends[2]));
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[30].backends[2]));
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[31].backends[2]));
	_register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[42].backends[2]));
	_register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[43].backends[2]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[78].backends[2]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[79].backends[2]));
	_register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[102].backends[2]));
	_register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[103].backends[2]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[112].backends[2]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[113].backends[2]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[72].backends[2]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[73].backends[2]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[116].backends[2]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[117].backends[2]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[70].backends[2]));
	_register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[71].backends[2]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[54].backends[2]));
	_register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[55].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[14].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[15].backends[2]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[76].backends[2]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[77].backends[2]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[52].backends[2]));
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[53].backends[2]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[88].backends[2]));
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[89].backends[2]));
	_register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[58].backends[2]));
	_register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[59].backends[2]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[74].backends[2]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[75].backends[2]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[28].backends[2]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[29].backends[2]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[80].backends[2]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[81].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[0].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[1].backends[2]));
	_register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[6].backends[2]));
	_register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[7].backends[2]));
	_register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[40].backends[2]));
	_register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[41].backends[2]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[36].backends[2]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[37].backends[2]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[110].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[111].backends[2]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[2].backends[2]));
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[3].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[96].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[97].backends[2]));
	_register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[4].backends[2]));
	_register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[5].backends[2]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[84].backends[2]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[85].backends[2]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[94].backends[2]));
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[95].backends[2]));
	_register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[92].backends[2]));
	_register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[93].backends[2]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[26].backends[2]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[27].backends[2]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[66].backends[2]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[67].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[62].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[62].backends[4]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[63].backends[2]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[98].backends[2]));
	_register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[99].backends[2]));
	_register_command_CCV_NNC_HISTOGRAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[32].backends[2]));
	_register_command_CCV_NNC_HISTOGRAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[33].backends[2]));
#ifdef HAVE_CUDA
	_register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[60].backends[5]));
	_register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[61].backends[5]));
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[46].backends[5]));
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[47].backends[5]));
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[38].backends[5]));
	_register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[39].backends[5]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[56].backends[5]));
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[57].backends[5]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[22].backends[0]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[23].backends[0]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[104].backends[3]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[105].backends[3]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[50].backends[3]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[51].backends[3]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[44].backends[3]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[45].backends[3]));
	_register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[68].backends[5]));
	_register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[69].backends[5]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[34].backends[3]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[35].backends[3]));
	_register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[10].backends[5]));
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[11].backends[5]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[90].backends[5]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[91].backends[5]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[82].backends[3]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[83].backends[3]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[24].backends[3]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[25].backends[3]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[108].backends[5]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[109].backends[5]));
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[12].backends[5]));
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[13].backends[5]));
	_register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[64].backends[5]));
	_register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[65].backends[5]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[16].backends[5]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[17].backends[5]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[106].backends[5]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[107].backends[5]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[86].backends[3]));
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[87].backends[3]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[100].backends[3]));
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[101].backends[3]));
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[30].backends[3]));
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[31].backends[3]));
	_register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[42].backends[5]));
	_register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[43].backends[5]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[78].backends[3]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[79].backends[3]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[112].backends[5]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[113].backends[5]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[72].backends[5]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[73].backends[5]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[116].backends[5]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[117].backends[5]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[70].backends[5]));
	_register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[71].backends[5]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[54].backends[5]));
	_register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[55].backends[5]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[14].backends[3]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[15].backends[3]));
	_register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[48].backends[1]));
	_register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[49].backends[1]));
	_register_command_CCV_NNC_COMM_BROADCAST_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[114].backends[1]));
	_register_command_CCV_NNC_COMM_BROADCAST_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[115].backends[1]));
	_register_command_CCV_NNC_COMM_REDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[20].backends[1]));
	_register_command_CCV_NNC_COMM_REDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[21].backends[1]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[76].backends[3]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[77].backends[3]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[52].backends[3]));
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[53].backends[3]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[88].backends[3]));
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[89].backends[3]));
	_register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[58].backends[3]));
	_register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[59].backends[3]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[74].backends[5]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[75].backends[5]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[28].backends[3]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[29].backends[3]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[80].backends[5]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[81].backends[5]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[0].backends[3]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[1].backends[3]));
	_register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[6].backends[5]));
	_register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[7].backends[5]));
	_register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[40].backends[5]));
	_register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[41].backends[5]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[36].backends[3]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[37].backends[3]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[110].backends[3]));
	_register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[111].backends[3]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[2].backends[5]));
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[3].backends[5]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[96].backends[5]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[97].backends[5]));
	_register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[4].backends[5]));
	_register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[5].backends[5]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[84].backends[5]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[85].backends[5]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[94].backends[5]));
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[95].backends[5]));
	_register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[92].backends[5]));
	_register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[93].backends[5]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[26].backends[3]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[27].backends[3]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[66].backends[3]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[67].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[62].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[63].backends[3]));
	_register_command_CCV_NNC_LSTM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[18].backends[3]));
	_register_command_CCV_NNC_LSTM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[19].backends[3]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[98].backends[5]));
	_register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[99].backends[5]));
#endif
#ifdef HAVE_MPS
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[38].backends[6]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[56].backends[6]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[22].backends[6]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[23].backends[6]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[104].backends[6]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[50].backends[6]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[44].backends[6]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[34].backends[6]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[35].backends[6]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[90].backends[6]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[91].backends[6]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[82].backends[6]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[83].backends[6]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[24].backends[6]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[25].backends[6]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[108].backends[6]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[109].backends[6]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[100].backends[6]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[78].backends[6]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[112].backends[6]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[72].backends[6]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[116].backends[6]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[70].backends[6]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[54].backends[6]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[52].backends[6]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[88].backends[6]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[0].backends[6]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[36].backends[6]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[110].backends[6]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[2].backends[6]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[84].backends[6]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[94].backends[6]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[62].backends[6]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[98].backends[6]));
#endif
}
