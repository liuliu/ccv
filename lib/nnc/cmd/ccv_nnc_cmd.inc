static ccv_nnc_cmd_init_t init_map[] = {
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD", .cmd = 0x1eb327a2},
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD", .cmd = 0x1eb327a3},
	{.name = "CCV_NNC_NMS_FORWARD", .cmd = 0xdba26106},
	{.name = "CCV_NNC_NMS_BACKWARD", .cmd = 0xdba26107},
	{.name = "CCV_NNC_EWSIN_FORWARD", .cmd = 0xc87a926a},
	{.name = "CCV_NNC_EWSIN_BACKWARD", .cmd = 0xc87a926b},
	{.name = "CCV_NNC_GUMBEL_ARGMAX_FORWARD", .cmd = 0xa8f2f04c},
	{.name = "CCV_NNC_GUMBEL_ARGMAX_BACKWARD", .cmd = 0xa8f2f04d},
	{.name = "CCV_NNC_TRANSPOSE_FORWARD", .cmd = 0xb4d506e0},
	{.name = "CCV_NNC_TRANSPOSE_BACKWARD", .cmd = 0xb4d506e1},
	{.name = "CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD", .cmd = 0x5d575032},
	{.name = "CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD", .cmd = 0x5d575033},
	{.name = "CCV_NNC_HISTOGRAM_FORWARD", .cmd = 0xc5473e44},
	{.name = "CCV_NNC_HISTOGRAM_BACKWARD", .cmd = 0xc5473e45},
	{.name = "CCV_NNC_INDEX_SELECT_FORWARD", .cmd = 0x7ee7771e},
	{.name = "CCV_NNC_INDEX_SELECT_BACKWARD", .cmd = 0x7ee7771f},
	{.name = "CCV_NNC_GEMM_FORWARD", .cmd = 0x7e87d00c},
	{.name = "CCV_NNC_GEMM_BACKWARD", .cmd = 0x7e87d00d},
	{.name = "CCV_NNC_GRID_SAMPLE_FORWARD", .cmd = 0xd492a032},
	{.name = "CCV_NNC_GRID_SAMPLE_BACKWARD", .cmd = 0xd492a033},
	{.name = "CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD", .cmd = 0xd9e0e4a},
	{.name = "CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD", .cmd = 0xd9e0e4b},
	{.name = "CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD", .cmd = 0x7d835c92},
	{.name = "CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD", .cmd = 0x7d835c93},
	{.name = "CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD", .cmd = 0xfcd4f798},
	{.name = "CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD", .cmd = 0xfcd4f799},
	{.name = "CCV_NNC_LAMB_FORWARD", .cmd = 0x450edb1a},
	{.name = "CCV_NNC_LAMB_BACKWARD", .cmd = 0x450edb1b},
	{.name = "CCV_NNC_RMSNORM_CMUL_FORWARD", .cmd = 0x27c8454a},
	{.name = "CCV_NNC_RMSNORM_CMUL_BACKWARD", .cmd = 0x27c8454b},
	{.name = "CCV_NNC_ADAM_FORWARD", .cmd = 0xe30099dc},
	{.name = "CCV_NNC_ADAM_BACKWARD", .cmd = 0xe30099dd},
	{.name = "CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD", .cmd = 0xa89d75a0},
	{.name = "CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD", .cmd = 0xa89d75a1},
	{.name = "CCV_NNC_SEGMENTED_SWIGLU_FORWARD", .cmd = 0x1789f312},
	{.name = "CCV_NNC_SEGMENTED_SWIGLU_BACKWARD", .cmd = 0x1789f313},
	{.name = "CCV_NNC_HYPER_CONNECTION_FORWARD", .cmd = 0x584a5274},
	{.name = "CCV_NNC_HYPER_CONNECTION_BACKWARD", .cmd = 0x584a5275},
	{.name = "CCV_NNC_COMM_BROADCAST_FORWARD", .cmd = 0x830eee},
	{.name = "CCV_NNC_COMM_BROADCAST_BACKWARD", .cmd = 0x830eef},
	{.name = "CCV_NNC_GELU_FORWARD", .cmd = 0xb1527ab8},
	{.name = "CCV_NNC_GELU_BACKWARD", .cmd = 0xb1527ab9},
	{.name = "CCV_NNC_REDUCE_NORM2_FORWARD", .cmd = 0xb3034e16},
	{.name = "CCV_NNC_REDUCE_NORM2_BACKWARD", .cmd = 0xb3034e17},
	{.name = "CCV_NNC_RMSPROP_FORWARD", .cmd = 0x9c886b1c},
	{.name = "CCV_NNC_RMSPROP_BACKWARD", .cmd = 0x9c886b1d},
	{.name = "CCV_NNC_RMSNORM_FORWARD", .cmd = 0x6889e9d0},
	{.name = "CCV_NNC_RMSNORM_BACKWARD", .cmd = 0x6889e9d1},
	{.name = "CCV_NNC_CONVOLUTION_FORWARD", .cmd = 0x254d05f4},
	{.name = "CCV_NNC_CONVOLUTION_BACKWARD", .cmd = 0x254d05f5},
	{.name = "CCV_NNC_SOFTMAX_FORWARD", .cmd = 0xc969a252},
	{.name = "CCV_NNC_SOFTMAX_BACKWARD", .cmd = 0xc969a253},
	{.name = "CCV_NNC_COMM_ALLREDUCE_FORWARD", .cmd = 0x75c8d340},
	{.name = "CCV_NNC_COMM_ALLREDUCE_BACKWARD", .cmd = 0x75c8d341},
	{.name = "CCV_NNC_DATA_TRANSFER_FORWARD", .cmd = 0x12d21e1a},
	{.name = "CCV_NNC_DATA_TRANSFER_BACKWARD", .cmd = 0x12d21e1b},
	{.name = "CCV_NNC_ADAMW_FORWARD", .cmd = 0x4f5d4870},
	{.name = "CCV_NNC_ADAMW_BACKWARD", .cmd = 0x4f5d4871},
	{.name = "CCV_NNC_PARTITION_FORWARD", .cmd = 0x698f42fe},
	{.name = "CCV_NNC_PARTITION_BACKWARD", .cmd = 0x698f42ff},
	{.name = "CCV_NNC_BATCH_NORM_FORWARD", .cmd = 0x5419819c},
	{.name = "CCV_NNC_BATCH_NORM_BACKWARD", .cmd = 0x5419819d},
	{.name = "CCV_NNC_SEGMENTED_GEMM_FORWARD", .cmd = 0x58305730},
	{.name = "CCV_NNC_SEGMENTED_GEMM_BACKWARD", .cmd = 0x58305731},
	{.name = "CCV_NNC_SWIGLU_FORWARD", .cmd = 0x578dbf90},
	{.name = "CCV_NNC_SWIGLU_BACKWARD", .cmd = 0x578dbf91},
	{.name = "CCV_NNC_EWPOW_FORWARD", .cmd = 0xa9cd182},
	{.name = "CCV_NNC_EWPOW_BACKWARD", .cmd = 0xa9cd183},
	{.name = "CCV_NNC_SWISH_FORWARD", .cmd = 0x583d90c2},
	{.name = "CCV_NNC_SWISH_BACKWARD", .cmd = 0x583d90c3},
	{.name = "CCV_NNC_MOE_ROUTING_FORWARD", .cmd = 0xc1323630},
	{.name = "CCV_NNC_MOE_ROUTING_BACKWARD", .cmd = 0xc1323631},
	{.name = "CCV_NNC_EWDIV_FORWARD", .cmd = 0x1cd2fa18},
	{.name = "CCV_NNC_EWDIV_BACKWARD", .cmd = 0x1cd2fa19},
	{.name = "CCV_NNC_MIN_FORWARD", .cmd = 0x972fbd26},
	{.name = "CCV_NNC_MIN_BACKWARD", .cmd = 0x972fbd27},
	{.name = "CCV_NNC_LAYER_NORM_FORWARD", .cmd = 0xbed3c264},
	{.name = "CCV_NNC_LAYER_NORM_BACKWARD", .cmd = 0xbed3c265},
	{.name = "CCV_NNC_FILL_IF_LESS_THAN_FORWARD", .cmd = 0x2f5c2e7a},
	{.name = "CCV_NNC_FILL_IF_LESS_THAN_BACKWARD", .cmd = 0x2f5c2e7b},
	{.name = "CCV_NNC_AVERAGE_POOL_FORWARD", .cmd = 0x51267ab8},
	{.name = "CCV_NNC_AVERAGE_POOL_BACKWARD", .cmd = 0x51267ab9},
	{.name = "CCV_NNC_EWPROD_FORWARD", .cmd = 0xee07e8fe},
	{.name = "CCV_NNC_EWPROD_BACKWARD", .cmd = 0xee07e8ff},
	{.name = "CCV_NNC_MUL_FORWARD", .cmd = 0x24721a46},
	{.name = "CCV_NNC_MUL_BACKWARD", .cmd = 0x24721a47},
	{.name = "CCV_NNC_MAX_POOL_FORWARD", .cmd = 0x7bec9360},
	{.name = "CCV_NNC_MAX_POOL_BACKWARD", .cmd = 0x7bec9361},
	{.name = "CCV_NNC_REDUCE_MEAN_FORWARD", .cmd = 0xf23556c6},
	{.name = "CCV_NNC_REDUCE_MEAN_BACKWARD", .cmd = 0xf23556c7},
	{.name = "CCV_NNC_REDUCE_ISNAN_FORWARD", .cmd = 0xee0a4ade},
	{.name = "CCV_NNC_REDUCE_ISNAN_BACKWARD", .cmd = 0xee0a4adf},
	{.name = "CCV_NNC_ARGMAX_FORWARD", .cmd = 0x68af2804},
	{.name = "CCV_NNC_ARGMAX_BACKWARD", .cmd = 0x68af2805},
	{.name = "CCV_NNC_ADD_FORWARD", .cmd = 0x58fb3664},
	{.name = "CCV_NNC_ADD_BACKWARD", .cmd = 0x58fb3665},
	{.name = "CCV_NNC_RMSNORM_GATED_FORWARD", .cmd = 0x9b8af346},
	{.name = "CCV_NNC_RMSNORM_GATED_BACKWARD", .cmd = 0x9b8af347},
	{.name = "CCV_NNC_EWLOG_FORWARD", .cmd = 0xf4191bf2},
	{.name = "CCV_NNC_EWLOG_BACKWARD", .cmd = 0xf4191bf3},
	{.name = "CCV_NNC_SIGMOID_FORWARD", .cmd = 0xf2f69650},
	{.name = "CCV_NNC_SIGMOID_BACKWARD", .cmd = 0xf2f69651},
	{.name = "CCV_NNC_UPSAMPLE_FORWARD", .cmd = 0x73875556},
	{.name = "CCV_NNC_UPSAMPLE_BACKWARD", .cmd = 0x73875557},
	{.name = "CCV_NNC_LSTM_FORWARD", .cmd = 0xc5cb998c},
	{.name = "CCV_NNC_LSTM_BACKWARD", .cmd = 0xc5cb998d},
	{.name = "CCV_NNC_MSE_FORWARD", .cmd = 0x6904a9a2},
	{.name = "CCV_NNC_MSE_BACKWARD", .cmd = 0x6904a9a3},
	{.name = "CCV_NNC_PAD_FORWARD", .cmd = 0xd8aaca60},
	{.name = "CCV_NNC_PAD_BACKWARD", .cmd = 0xd8aaca61},
	{.name = "CCV_NNC_SORT_FORWARD", .cmd = 0x9b3f5868},
	{.name = "CCV_NNC_SORT_BACKWARD", .cmd = 0x9b3f5869},
	{.name = "CCV_NNC_SWISH_MUL_FORWARD", .cmd = 0x26467f24},
	{.name = "CCV_NNC_SWISH_MUL_BACKWARD", .cmd = 0x26467f25},
	{.name = "CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD", .cmd = 0x284ed926},
	{.name = "CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD", .cmd = 0x284ed927},
	{.name = "CCV_NNC_ROTATE_HALF_FORWARD", .cmd = 0xa831a22c},
	{.name = "CCV_NNC_ROTATE_HALF_BACKWARD", .cmd = 0xa831a22d},
	{.name = "CCV_NNC_GATED_DELTA_FORWARD", .cmd = 0x6de198ac},
	{.name = "CCV_NNC_GATED_DELTA_BACKWARD", .cmd = 0x6de198ad},
	{.name = "CCV_NNC_CLAMP_FORWARD", .cmd = 0x2640d854},
	{.name = "CCV_NNC_CLAMP_BACKWARD", .cmd = 0x2640d855},
	{.name = "CCV_NNC_COMM_REDUCE_FORWARD", .cmd = 0x3434ead8},
	{.name = "CCV_NNC_COMM_REDUCE_BACKWARD", .cmd = 0x3434ead9},
	{.name = "CCV_NNC_COMM_ALL_TO_ALL_FORWARD", .cmd = 0xa4a31844},
	{.name = "CCV_NNC_COMM_ALL_TO_ALL_BACKWARD", .cmd = 0xa4a31845},
	{.name = "CCV_NNC_EWSOFTPLUS_FORWARD", .cmd = 0x2b396320},
	{.name = "CCV_NNC_EWSOFTPLUS_BACKWARD", .cmd = 0x2b396321},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_FORWARD", .cmd = 0xe4a2b192},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_BACKWARD", .cmd = 0xe4a2b193},
	{.name = "CCV_NNC_DROPOUT_FORWARD", .cmd = 0x7f2dc3e4},
	{.name = "CCV_NNC_DROPOUT_BACKWARD", .cmd = 0x7f2dc3e5},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD", .cmd = 0xc26b7b5e},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD", .cmd = 0xc26b7b5f},
	{.name = "CCV_NNC_ROI_ALIGN_FORWARD", .cmd = 0xfef55168},
	{.name = "CCV_NNC_ROI_ALIGN_BACKWARD", .cmd = 0xfef55169},
	{.name = "CCV_NNC_TANH_FORWARD", .cmd = 0x6a62be30},
	{.name = "CCV_NNC_TANH_BACKWARD", .cmd = 0x6a62be31},
	{.name = "CCV_NNC_DATATYPE_CONVERSION_FORWARD", .cmd = 0xd873e38c},
	{.name = "CCV_NNC_DATATYPE_CONVERSION_BACKWARD", .cmd = 0xd873e38d},
	{.name = "CCV_NNC_REDUCE_MAX_FORWARD", .cmd = 0x80f1a506},
	{.name = "CCV_NNC_REDUCE_MAX_BACKWARD", .cmd = 0x80f1a507},
	{.name = "CCV_NNC_REDUCE_SUM_FORWARD", .cmd = 0x52970f06},
	{.name = "CCV_NNC_REDUCE_SUM_BACKWARD", .cmd = 0x52970f07},
	{.name = "CCV_NNC_COMPRESSION_LSSC_FORWARD", .cmd = 0x17ea8f72},
	{.name = "CCV_NNC_COMPRESSION_LSSC_BACKWARD", .cmd = 0x17ea8f73},
	{.name = "CCV_NNC_MAX_FORWARD", .cmd = 0xdf6f014c},
	{.name = "CCV_NNC_MAX_BACKWARD", .cmd = 0xdf6f014d},
	{.name = "CCV_NNC_RANDOM_NORMAL_FORWARD", .cmd = 0x7062c8b4},
	{.name = "CCV_NNC_RANDOM_NORMAL_BACKWARD", .cmd = 0x7062c8b5},
	{.name = "CCV_NNC_CONFORM_DATA_FORMAT_FORWARD", .cmd = 0x640dc06e},
	{.name = "CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD", .cmd = 0x640dc06f},
	{.name = "CCV_NNC_EWSUM_FORWARD", .cmd = 0xe21a2c4c},
	{.name = "CCV_NNC_EWSUM_BACKWARD", .cmd = 0xe21a2c4d},
	{.name = "CCV_NNC_BINARY_CROSSENTROPY_FORWARD", .cmd = 0xcd2107ec},
	{.name = "CCV_NNC_BINARY_CROSSENTROPY_BACKWARD", .cmd = 0xcd2107ed},
	{.name = "CCV_NNC_SMOOTH_L1_FORWARD", .cmd = 0x4e428e},
	{.name = "CCV_NNC_SMOOTH_L1_BACKWARD", .cmd = 0x4e428f},
	{.name = "CCV_NNC_EWEXP_FORWARD", .cmd = 0xd784b170},
	{.name = "CCV_NNC_EWEXP_BACKWARD", .cmd = 0xd784b171},
	{.name = "CCV_NNC_MASKED_FILL_FORWARD", .cmd = 0x7f992d84},
	{.name = "CCV_NNC_MASKED_FILL_BACKWARD", .cmd = 0x7f992d85},
	{.name = "CCV_NNC_LEAKY_RELU_FORWARD", .cmd = 0x507144e0},
	{.name = "CCV_NNC_LEAKY_RELU_BACKWARD", .cmd = 0x507144e1},
	{.name = "CCV_NNC_CMUL_FORWARD", .cmd = 0xead486e6},
	{.name = "CCV_NNC_CMUL_BACKWARD", .cmd = 0xead486e7},
	{.name = "CCV_NNC_EWCOS_FORWARD", .cmd = 0xbf13866c},
	{.name = "CCV_NNC_EWCOS_BACKWARD", .cmd = 0xbf13866d},
	{.name = "CCV_NNC_SCATTER_ADD_FORWARD", .cmd = 0xabddad90},
	{.name = "CCV_NNC_SCATTER_ADD_BACKWARD", .cmd = 0xabddad91},
	{.name = "CCV_NNC_EWABS_FORWARD", .cmd = 0x8a0db898},
	{.name = "CCV_NNC_EWABS_BACKWARD", .cmd = 0x8a0db899},
	{.name = "CCV_NNC_RANDOM_UNIFORM_FORWARD", .cmd = 0xa0cd1d5e},
	{.name = "CCV_NNC_RANDOM_UNIFORM_BACKWARD", .cmd = 0xa0cd1d5f},
	{.name = "CCV_NNC_RELU_FORWARD", .cmd = 0xc51eaa80},
	{.name = "CCV_NNC_RELU_BACKWARD", .cmd = 0xc51eaa81},
	{.name = "CCV_NNC_SET_FORWARD", .cmd = 0x2b070804},
	{.name = "CCV_NNC_SET_BACKWARD", .cmd = 0x2b070805},
	{.name = "CCV_NNC_REDUCE_LOGSUMEXP_FORWARD", .cmd = 0x3b6718b0},
	{.name = "CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD", .cmd = 0x3b6718b1},
	{.name = "CCV_NNC_REDUCE_MIN_FORWARD", .cmd = 0x6785ef96},
	{.name = "CCV_NNC_REDUCE_MIN_BACKWARD", .cmd = 0x6785ef97},
	{.name = "CCV_NNC_SGD_FORWARD", .cmd = 0xe650ad26},
	{.name = "CCV_NNC_SGD_BACKWARD", .cmd = 0xe650ad27},
	{.name = "CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD", .cmd = 0xd691f78e},
	{.name = "CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD", .cmd = 0xd691f78f},
	{.name = "CCV_NNC_GROUP_NORM_FORWARD", .cmd = 0x17deb074},
	{.name = "CCV_NNC_GROUP_NORM_BACKWARD", .cmd = 0x17deb075},
	{.name = "CCV_NNC_ARGMIN_FORWARD", .cmd = 0xeb8747f2},
	{.name = "CCV_NNC_ARGMIN_BACKWARD", .cmd = 0xeb8747f3},
	{.name = "CCV_NNC_SCALAR_MUL_FORWARD", .cmd = 0x8b4d86aa},
	{.name = "CCV_NNC_SCALAR_MUL_BACKWARD", .cmd = 0x8b4d86ab},
	{.name = "CCV_NNC_EWSQRT_FORWARD", .cmd = 0x8870a61e},
	{.name = "CCV_NNC_EWSQRT_BACKWARD", .cmd = 0x8870a61f},
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
	switch ((cmd >> 21) % 14)
	{
		case 0:
			return ((((cmd >> 1) % 30) + 28) << 1) | (cmd & 1);
		case 1:
			return ((((cmd >> 1) % 59) + 13) << 1) | (cmd & 1);
		case 2:
			return ((((cmd >> 1) % 21) + 70) << 1) | (cmd & 1);
		case 3:
			return ((((cmd >> 1) % 5) + 74) << 1) | (cmd & 1);
		case 4:
			return ((((cmd >> 1) % 30) + 0) << 1) | (cmd & 1);
		case 5:
			return ((((cmd >> 18) % 23) + 22) << 1) | (cmd & 1);
		case 6:
			return ((((cmd >> 1) % 56) + 0) << 1) | (cmd & 1);
		case 7:
			return ((((cmd >> 2) % 8) + 0) << 1) | (cmd & 1);
		case 8:
			return ((((cmd >> 1) % 9) + 87) << 1) | (cmd & 1);
		case 9:
			return ((((cmd >> 1) % 20) + 55) << 1) | (cmd & 1);
		case 10:
			return ((((cmd >> 4) % 28) + 6) << 1) | (cmd & 1);
		case 11:
			return ((((cmd >> 1) % 27) + 42) << 1) | (cmd & 1);
		case 12:
			return ((((cmd >> 2) % 40) + 47) << 1) | (cmd & 1);
		case 13:
		default:
			return ((((cmd >> 1) % 16) + 81) << 1) | (cmd & 1);
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

void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GUMBEL_ARGMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TRANSPOSE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GRID_SAMPLE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GRID_SAMPLE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_CMUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_CMUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MOE_ROUTING_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MOE_ROUTING_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_GATED_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_GATED_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_PAD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_PAD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SORT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SORT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GATED_DELTA_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GATED_DELTA_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MASKED_FILL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);

void _register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
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
void _register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GATED_DELTA_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HISTOGRAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MOE_ROUTING_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GUMBEL_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_CMUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_GATED_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SORT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
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
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
#ifdef HAVE_CUDA
void _register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_BROADCAST_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_REDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LSTM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SORT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
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
#endif
#ifdef HAVE_MPS
void _register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GATED_DELTA_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MOE_ROUTING_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_CMUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RMSNORM_GATED_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
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
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(ccv_nnc_cmd_backend_registry_t* const registry);
#endif

static inline void _ccv_nnc_cmd_init(void)
{
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(&init_map[0].registry);
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(&init_map[1].registry);
	_register_command_CCV_NNC_NMS_FORWARD(&init_map[2].registry);
	_register_command_CCV_NNC_NMS_BACKWARD(&init_map[3].registry);
	_register_command_CCV_NNC_EWSIN_FORWARD(&init_map[4].registry);
	_register_command_CCV_NNC_EWSIN_BACKWARD(&init_map[5].registry);
	_register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD(&init_map[6].registry);
	_register_command_CCV_NNC_GUMBEL_ARGMAX_BACKWARD(&init_map[7].registry);
	_register_command_CCV_NNC_TRANSPOSE_FORWARD(&init_map[8].registry);
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD(&init_map[9].registry);
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD(&init_map[10].registry);
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD(&init_map[11].registry);
	_register_command_CCV_NNC_HISTOGRAM_FORWARD(&init_map[12].registry);
	_register_command_CCV_NNC_HISTOGRAM_BACKWARD(&init_map[13].registry);
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD(&init_map[14].registry);
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD(&init_map[15].registry);
	_register_command_CCV_NNC_GEMM_FORWARD(&init_map[16].registry);
	_register_command_CCV_NNC_GEMM_BACKWARD(&init_map[17].registry);
	_register_command_CCV_NNC_GRID_SAMPLE_FORWARD(&init_map[18].registry);
	_register_command_CCV_NNC_GRID_SAMPLE_BACKWARD(&init_map[19].registry);
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD(&init_map[20].registry);
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD(&init_map[21].registry);
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(&init_map[22].registry);
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD(&init_map[23].registry);
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD(&init_map[24].registry);
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD(&init_map[25].registry);
	_register_command_CCV_NNC_LAMB_FORWARD(&init_map[26].registry);
	_register_command_CCV_NNC_LAMB_BACKWARD(&init_map[27].registry);
	_register_command_CCV_NNC_RMSNORM_CMUL_FORWARD(&init_map[28].registry);
	_register_command_CCV_NNC_RMSNORM_CMUL_BACKWARD(&init_map[29].registry);
	_register_command_CCV_NNC_ADAM_FORWARD(&init_map[30].registry);
	_register_command_CCV_NNC_ADAM_BACKWARD(&init_map[31].registry);
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD(&init_map[32].registry);
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD(&init_map[33].registry);
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD(&init_map[34].registry);
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD(&init_map[35].registry);
	_register_command_CCV_NNC_HYPER_CONNECTION_FORWARD(&init_map[36].registry);
	_register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD(&init_map[37].registry);
	_register_command_CCV_NNC_COMM_BROADCAST_FORWARD(&init_map[38].registry);
	_register_command_CCV_NNC_COMM_BROADCAST_BACKWARD(&init_map[39].registry);
	_register_command_CCV_NNC_GELU_FORWARD(&init_map[40].registry);
	_register_command_CCV_NNC_GELU_BACKWARD(&init_map[41].registry);
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD(&init_map[42].registry);
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD(&init_map[43].registry);
	_register_command_CCV_NNC_RMSPROP_FORWARD(&init_map[44].registry);
	_register_command_CCV_NNC_RMSPROP_BACKWARD(&init_map[45].registry);
	_register_command_CCV_NNC_RMSNORM_FORWARD(&init_map[46].registry);
	_register_command_CCV_NNC_RMSNORM_BACKWARD(&init_map[47].registry);
	_register_command_CCV_NNC_CONVOLUTION_FORWARD(&init_map[48].registry);
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD(&init_map[49].registry);
	_register_command_CCV_NNC_SOFTMAX_FORWARD(&init_map[50].registry);
	_register_command_CCV_NNC_SOFTMAX_BACKWARD(&init_map[51].registry);
	_register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD(&init_map[52].registry);
	_register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD(&init_map[53].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD(&init_map[54].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(&init_map[55].registry);
	_register_command_CCV_NNC_ADAMW_FORWARD(&init_map[56].registry);
	_register_command_CCV_NNC_ADAMW_BACKWARD(&init_map[57].registry);
	_register_command_CCV_NNC_PARTITION_FORWARD(&init_map[58].registry);
	_register_command_CCV_NNC_PARTITION_BACKWARD(&init_map[59].registry);
	_register_command_CCV_NNC_BATCH_NORM_FORWARD(&init_map[60].registry);
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD(&init_map[61].registry);
	_register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD(&init_map[62].registry);
	_register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD(&init_map[63].registry);
	_register_command_CCV_NNC_SWIGLU_FORWARD(&init_map[64].registry);
	_register_command_CCV_NNC_SWIGLU_BACKWARD(&init_map[65].registry);
	_register_command_CCV_NNC_EWPOW_FORWARD(&init_map[66].registry);
	_register_command_CCV_NNC_EWPOW_BACKWARD(&init_map[67].registry);
	_register_command_CCV_NNC_SWISH_FORWARD(&init_map[68].registry);
	_register_command_CCV_NNC_SWISH_BACKWARD(&init_map[69].registry);
	_register_command_CCV_NNC_MOE_ROUTING_FORWARD(&init_map[70].registry);
	_register_command_CCV_NNC_MOE_ROUTING_BACKWARD(&init_map[71].registry);
	_register_command_CCV_NNC_EWDIV_FORWARD(&init_map[72].registry);
	_register_command_CCV_NNC_EWDIV_BACKWARD(&init_map[73].registry);
	_register_command_CCV_NNC_MIN_FORWARD(&init_map[74].registry);
	_register_command_CCV_NNC_MIN_BACKWARD(&init_map[75].registry);
	_register_command_CCV_NNC_LAYER_NORM_FORWARD(&init_map[76].registry);
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD(&init_map[77].registry);
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD(&init_map[78].registry);
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD(&init_map[79].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD(&init_map[80].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(&init_map[81].registry);
	_register_command_CCV_NNC_EWPROD_FORWARD(&init_map[82].registry);
	_register_command_CCV_NNC_EWPROD_BACKWARD(&init_map[83].registry);
	_register_command_CCV_NNC_MUL_FORWARD(&init_map[84].registry);
	_register_command_CCV_NNC_MUL_BACKWARD(&init_map[85].registry);
	_register_command_CCV_NNC_MAX_POOL_FORWARD(&init_map[86].registry);
	_register_command_CCV_NNC_MAX_POOL_BACKWARD(&init_map[87].registry);
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD(&init_map[88].registry);
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD(&init_map[89].registry);
	_register_command_CCV_NNC_REDUCE_ISNAN_FORWARD(&init_map[90].registry);
	_register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD(&init_map[91].registry);
	_register_command_CCV_NNC_ARGMAX_FORWARD(&init_map[92].registry);
	_register_command_CCV_NNC_ARGMAX_BACKWARD(&init_map[93].registry);
	_register_command_CCV_NNC_ADD_FORWARD(&init_map[94].registry);
	_register_command_CCV_NNC_ADD_BACKWARD(&init_map[95].registry);
	_register_command_CCV_NNC_RMSNORM_GATED_FORWARD(&init_map[96].registry);
	_register_command_CCV_NNC_RMSNORM_GATED_BACKWARD(&init_map[97].registry);
	_register_command_CCV_NNC_EWLOG_FORWARD(&init_map[98].registry);
	_register_command_CCV_NNC_EWLOG_BACKWARD(&init_map[99].registry);
	_register_command_CCV_NNC_SIGMOID_FORWARD(&init_map[100].registry);
	_register_command_CCV_NNC_SIGMOID_BACKWARD(&init_map[101].registry);
	_register_command_CCV_NNC_UPSAMPLE_FORWARD(&init_map[102].registry);
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD(&init_map[103].registry);
	_register_command_CCV_NNC_LSTM_FORWARD(&init_map[104].registry);
	_register_command_CCV_NNC_LSTM_BACKWARD(&init_map[105].registry);
	_register_command_CCV_NNC_MSE_FORWARD(&init_map[106].registry);
	_register_command_CCV_NNC_MSE_BACKWARD(&init_map[107].registry);
	_register_command_CCV_NNC_PAD_FORWARD(&init_map[108].registry);
	_register_command_CCV_NNC_PAD_BACKWARD(&init_map[109].registry);
	_register_command_CCV_NNC_SORT_FORWARD(&init_map[110].registry);
	_register_command_CCV_NNC_SORT_BACKWARD(&init_map[111].registry);
	_register_command_CCV_NNC_SWISH_MUL_FORWARD(&init_map[112].registry);
	_register_command_CCV_NNC_SWISH_MUL_BACKWARD(&init_map[113].registry);
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(&init_map[114].registry);
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(&init_map[115].registry);
	_register_command_CCV_NNC_ROTATE_HALF_FORWARD(&init_map[116].registry);
	_register_command_CCV_NNC_ROTATE_HALF_BACKWARD(&init_map[117].registry);
	_register_command_CCV_NNC_GATED_DELTA_FORWARD(&init_map[118].registry);
	_register_command_CCV_NNC_GATED_DELTA_BACKWARD(&init_map[119].registry);
	_register_command_CCV_NNC_CLAMP_FORWARD(&init_map[120].registry);
	_register_command_CCV_NNC_CLAMP_BACKWARD(&init_map[121].registry);
	_register_command_CCV_NNC_COMM_REDUCE_FORWARD(&init_map[122].registry);
	_register_command_CCV_NNC_COMM_REDUCE_BACKWARD(&init_map[123].registry);
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD(&init_map[124].registry);
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD(&init_map[125].registry);
	_register_command_CCV_NNC_EWSOFTPLUS_FORWARD(&init_map[126].registry);
	_register_command_CCV_NNC_EWSOFTPLUS_BACKWARD(&init_map[127].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(&init_map[128].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(&init_map[129].registry);
	_register_command_CCV_NNC_DROPOUT_FORWARD(&init_map[130].registry);
	_register_command_CCV_NNC_DROPOUT_BACKWARD(&init_map[131].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(&init_map[132].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(&init_map[133].registry);
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD(&init_map[134].registry);
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD(&init_map[135].registry);
	_register_command_CCV_NNC_TANH_FORWARD(&init_map[136].registry);
	_register_command_CCV_NNC_TANH_BACKWARD(&init_map[137].registry);
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD(&init_map[138].registry);
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD(&init_map[139].registry);
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD(&init_map[140].registry);
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD(&init_map[141].registry);
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD(&init_map[142].registry);
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD(&init_map[143].registry);
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD(&init_map[144].registry);
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD(&init_map[145].registry);
	_register_command_CCV_NNC_MAX_FORWARD(&init_map[146].registry);
	_register_command_CCV_NNC_MAX_BACKWARD(&init_map[147].registry);
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD(&init_map[148].registry);
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD(&init_map[149].registry);
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD(&init_map[150].registry);
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD(&init_map[151].registry);
	_register_command_CCV_NNC_EWSUM_FORWARD(&init_map[152].registry);
	_register_command_CCV_NNC_EWSUM_BACKWARD(&init_map[153].registry);
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD(&init_map[154].registry);
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD(&init_map[155].registry);
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD(&init_map[156].registry);
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD(&init_map[157].registry);
	_register_command_CCV_NNC_EWEXP_FORWARD(&init_map[158].registry);
	_register_command_CCV_NNC_EWEXP_BACKWARD(&init_map[159].registry);
	_register_command_CCV_NNC_MASKED_FILL_FORWARD(&init_map[160].registry);
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD(&init_map[161].registry);
	_register_command_CCV_NNC_LEAKY_RELU_FORWARD(&init_map[162].registry);
	_register_command_CCV_NNC_LEAKY_RELU_BACKWARD(&init_map[163].registry);
	_register_command_CCV_NNC_CMUL_FORWARD(&init_map[164].registry);
	_register_command_CCV_NNC_CMUL_BACKWARD(&init_map[165].registry);
	_register_command_CCV_NNC_EWCOS_FORWARD(&init_map[166].registry);
	_register_command_CCV_NNC_EWCOS_BACKWARD(&init_map[167].registry);
	_register_command_CCV_NNC_SCATTER_ADD_FORWARD(&init_map[168].registry);
	_register_command_CCV_NNC_SCATTER_ADD_BACKWARD(&init_map[169].registry);
	_register_command_CCV_NNC_EWABS_FORWARD(&init_map[170].registry);
	_register_command_CCV_NNC_EWABS_BACKWARD(&init_map[171].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(&init_map[172].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(&init_map[173].registry);
	_register_command_CCV_NNC_RELU_FORWARD(&init_map[174].registry);
	_register_command_CCV_NNC_RELU_BACKWARD(&init_map[175].registry);
	_register_command_CCV_NNC_SET_FORWARD(&init_map[176].registry);
	_register_command_CCV_NNC_SET_BACKWARD(&init_map[177].registry);
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD(&init_map[178].registry);
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD(&init_map[179].registry);
	_register_command_CCV_NNC_REDUCE_MIN_FORWARD(&init_map[180].registry);
	_register_command_CCV_NNC_REDUCE_MIN_BACKWARD(&init_map[181].registry);
	_register_command_CCV_NNC_SGD_FORWARD(&init_map[182].registry);
	_register_command_CCV_NNC_SGD_BACKWARD(&init_map[183].registry);
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD(&init_map[184].registry);
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD(&init_map[185].registry);
	_register_command_CCV_NNC_GROUP_NORM_FORWARD(&init_map[186].registry);
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD(&init_map[187].registry);
	_register_command_CCV_NNC_ARGMIN_FORWARD(&init_map[188].registry);
	_register_command_CCV_NNC_ARGMIN_BACKWARD(&init_map[189].registry);
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD(&init_map[190].registry);
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD(&init_map[191].registry);
	_register_command_CCV_NNC_EWSQRT_FORWARD(&init_map[192].registry);
	_register_command_CCV_NNC_EWSQRT_BACKWARD(&init_map[193].registry);

	_register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[30].backends[2]));
	_register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[31].backends[2]));
	_register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[56].backends[2]));
	_register_command_CCV_NNC_ADAMW_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[57].backends[2]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[16].backends[2]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[16].backends[4]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[17].backends[2]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[17].backends[4]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[94].backends[2]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[95].backends[2]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[84].backends[2]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[85].backends[2]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[190].backends[2]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[191].backends[2]));
	_register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[164].backends[2]));
	_register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[165].backends[2]));
	_register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[62].backends[2]));
	_register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[63].backends[2]));
	_register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[116].backends[2]));
	_register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[117].backends[2]));
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[124].backends[2]));
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[125].backends[2]));
	_register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[74].backends[2]));
	_register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[75].backends[2]));
	_register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[146].backends[2]));
	_register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[147].backends[2]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[144].backends[2]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[145].backends[2]));
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[150].backends[2]));
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[151].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[48].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[48].backends[4]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[49].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[184].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[185].backends[2]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[130].backends[2]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[131].backends[2]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[152].backends[2]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[153].backends[2]));
	_register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[82].backends[2]));
	_register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[83].backends[2]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[72].backends[2]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[73].backends[2]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[158].backends[2]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[159].backends[2]));
	_register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[126].backends[2]));
	_register_command_CCV_NNC_EWSOFTPLUS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[127].backends[2]));
	_register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[66].backends[2]));
	_register_command_CCV_NNC_EWPOW_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[67].backends[2]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[98].backends[2]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[99].backends[2]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[192].backends[2]));
	_register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[193].backends[2]));
	_register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[4].backends[2]));
	_register_command_CCV_NNC_EWSIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[5].backends[2]));
	_register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[166].backends[2]));
	_register_command_CCV_NNC_EWCOS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[167].backends[2]));
	_register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[170].backends[2]));
	_register_command_CCV_NNC_EWABS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[171].backends[2]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[120].backends[2]));
	_register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[121].backends[2]));
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[78].backends[2]));
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[79].backends[2]));
	_register_command_CCV_NNC_GATED_DELTA_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[118].backends[2]));
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[40].backends[2]));
	_register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[41].backends[2]));
	_register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[18].backends[2]));
	_register_command_CCV_NNC_HISTOGRAM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[12].backends[2]));
	_register_command_CCV_NNC_HISTOGRAM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[13].backends[2]));
	_register_command_CCV_NNC_HYPER_CONNECTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[36].backends[2]));
	_register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[37].backends[2]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[14].backends[2]));
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[15].backends[2]));
	_register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[90].backends[2]));
	_register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[91].backends[2]));
	_register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[26].backends[2]));
	_register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[27].backends[2]));
	_register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[162].backends[2]));
	_register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[163].backends[2]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[154].backends[2]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[155].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[0].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[1].backends[2]));
	_register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[106].backends[2]));
	_register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[107].backends[2]));
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[156].backends[2]));
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[157].backends[2]));
	_register_command_CCV_NNC_MOE_ROUTING_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[70].backends[2]));
	_register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[2].backends[2]));
	_register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[3].backends[2]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[60].backends[2]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[61].backends[2]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[76].backends[2]));
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[77].backends[2]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[186].backends[2]));
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[187].backends[2]));
	_register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[46].backends[2]));
	_register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[47].backends[2]));
	_register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[108].backends[2]));
	_register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[109].backends[2]));
	_register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[58].backends[2]));
	_register_command_CCV_NNC_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[59].backends[2]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[86].backends[2]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[87].backends[2]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[80].backends[2]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[81].backends[2]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[172].backends[2]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[173].backends[2]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[148].backends[2]));
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[149].backends[2]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[142].backends[2]));
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[143].backends[2]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[88].backends[2]));
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[89].backends[2]));
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[178].backends[2]));
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[179].backends[2]));
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[140].backends[2]));
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[141].backends[2]));
	_register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[180].backends[2]));
	_register_command_CCV_NNC_REDUCE_MIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[181].backends[2]));
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[42].backends[2]));
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[43].backends[2]));
	_register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[92].backends[2]));
	_register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[93].backends[2]));
	_register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[6].backends[2]));
	_register_command_CCV_NNC_GUMBEL_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[7].backends[2]));
	_register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[188].backends[2]));
	_register_command_CCV_NNC_ARGMIN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[189].backends[2]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[174].backends[2]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[175].backends[2]));
	_register_command_CCV_NNC_RMSNORM_CMUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[28].backends[2]));
	_register_command_CCV_NNC_RMSNORM_GATED_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[96].backends[2]));
	_register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[44].backends[2]));
	_register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[45].backends[2]));
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[134].backends[2]));
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[135].backends[2]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[22].backends[2]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[23].backends[2]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[114].backends[2]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[115].backends[2]));
	_register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[168].backends[2]));
	_register_command_CCV_NNC_SCATTER_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[169].backends[2]));
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[34].backends[2]));
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[35].backends[2]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[182].backends[2]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[183].backends[2]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[100].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[101].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[20].backends[2]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[21].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[50].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[51].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[132].backends[2]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[133].backends[2]));
	_register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[110].backends[2]));
	_register_command_CCV_NNC_SORT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[111].backends[2]));
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[24].backends[2]));
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[25].backends[2]));
	_register_command_CCV_NNC_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[64].backends[2]));
	_register_command_CCV_NNC_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[65].backends[2]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[68].backends[2]));
	_register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[69].backends[2]));
	_register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[112].backends[2]));
	_register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[113].backends[2]));
	_register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[136].backends[2]));
	_register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[137].backends[2]));
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[10].backends[2]));
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[11].backends[2]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[102].backends[2]));
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[103].backends[2]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[176].backends[2]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[177].backends[2]));
	_register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[160].backends[2]));
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[161].backends[2]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[54].backends[2]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[55].backends[2]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[128].backends[2]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[129].backends[2]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[8].backends[2]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[9].backends[2]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[138].backends[2]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[139].backends[2]));
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[32].backends[2]));
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[33].backends[2]));
#ifdef HAVE_CUDA
	_register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[30].backends[5]));
	_register_command_CCV_NNC_ADAM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[31].backends[5]));
	_register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[56].backends[5]));
	_register_command_CCV_NNC_ADAMW_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[57].backends[5]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[16].backends[0]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[17].backends[0]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[94].backends[3]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[95].backends[3]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[84].backends[3]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[85].backends[3]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[190].backends[3]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[191].backends[3]));
	_register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[164].backends[5]));
	_register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[165].backends[5]));
	_register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[62].backends[0]));
	_register_command_CCV_NNC_SEGMENTED_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[63].backends[0]));
	_register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[116].backends[5]));
	_register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[117].backends[5]));
	_register_command_CCV_NNC_COMM_ALLREDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[52].backends[1]));
	_register_command_CCV_NNC_COMM_ALLREDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[53].backends[1]));
	_register_command_CCV_NNC_COMM_BROADCAST_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[38].backends[1]));
	_register_command_CCV_NNC_COMM_BROADCAST_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[39].backends[1]));
	_register_command_CCV_NNC_COMM_REDUCE_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[122].backends[1]));
	_register_command_CCV_NNC_COMM_REDUCE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[123].backends[1]));
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_FORWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[124].backends[1]));
	_register_command_CCV_NNC_COMM_ALL_TO_ALL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_NCCL(&(init_map[125].backends[1]));
	_register_command_CCV_NNC_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[74].backends[5]));
	_register_command_CCV_NNC_MIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[75].backends[5]));
	_register_command_CCV_NNC_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[146].backends[5]));
	_register_command_CCV_NNC_MAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[147].backends[5]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[144].backends[5]));
	_register_command_CCV_NNC_COMPRESSION_LSSC_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[145].backends[5]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[48].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[49].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[184].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[185].backends[3]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[130].backends[3]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[131].backends[3]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[152].backends[3]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[153].backends[3]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[72].backends[5]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[73].backends[5]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[158].backends[5]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[159].backends[5]));
	_register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[126].backends[5]));
	_register_command_CCV_NNC_EWSOFTPLUS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[127].backends[5]));
	_register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[66].backends[5]));
	_register_command_CCV_NNC_EWPOW_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[67].backends[5]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[98].backends[5]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[99].backends[5]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[192].backends[5]));
	_register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[193].backends[5]));
	_register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[4].backends[5]));
	_register_command_CCV_NNC_EWSIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[5].backends[5]));
	_register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[166].backends[5]));
	_register_command_CCV_NNC_EWCOS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[167].backends[5]));
	_register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[170].backends[5]));
	_register_command_CCV_NNC_EWABS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[171].backends[5]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[120].backends[5]));
	_register_command_CCV_NNC_CLAMP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[121].backends[5]));
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[40].backends[5]));
	_register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[41].backends[5]));
	_register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[18].backends[3]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[14].backends[5]));
	_register_command_CCV_NNC_INDEX_SELECT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[15].backends[5]));
	_register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[90].backends[3]));
	_register_command_CCV_NNC_REDUCE_ISNAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[91].backends[3]));
	_register_command_CCV_NNC_LAMB_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[26].backends[5]));
	_register_command_CCV_NNC_LAMB_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[27].backends[5]));
	_register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[162].backends[5]));
	_register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[163].backends[5]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[154].backends[5]));
	_register_command_CCV_NNC_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[155].backends[5]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[0].backends[5]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[1].backends[5]));
	_register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[106].backends[5]));
	_register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[107].backends[5]));
	_register_command_CCV_NNC_SMOOTH_L1_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[156].backends[5]));
	_register_command_CCV_NNC_SMOOTH_L1_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[157].backends[5]));
	_register_command_CCV_NNC_NMS_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[2].backends[5]));
	_register_command_CCV_NNC_NMS_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[3].backends[5]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[60].backends[3]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[61].backends[3]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[76].backends[3]));
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[77].backends[3]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[186].backends[3]));
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[187].backends[3]));
	_register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[46].backends[3]));
	_register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[47].backends[3]));
	_register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[108].backends[5]));
	_register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[109].backends[5]));
	_register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[58].backends[5]));
	_register_command_CCV_NNC_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[59].backends[5]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[86].backends[3]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[87].backends[3]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[80].backends[3]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[81].backends[3]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[172].backends[5]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[173].backends[5]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[148].backends[5]));
	_register_command_CCV_NNC_RANDOM_NORMAL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[149].backends[5]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[142].backends[3]));
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[143].backends[3]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[88].backends[3]));
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[89].backends[3]));
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[140].backends[3]));
	_register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[180].backends[3]));
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[42].backends[3]));
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[43].backends[3]));
	_register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[92].backends[5]));
	_register_command_CCV_NNC_ARGMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[93].backends[5]));
	_register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[188].backends[5]));
	_register_command_CCV_NNC_ARGMIN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[189].backends[5]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[174].backends[3]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[175].backends[3]));
	_register_command_CCV_NNC_RMSPROP_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[44].backends[5]));
	_register_command_CCV_NNC_RMSPROP_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[45].backends[5]));
	_register_command_CCV_NNC_LSTM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[104].backends[3]));
	_register_command_CCV_NNC_LSTM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[105].backends[3]));
	_register_command_CCV_NNC_ROI_ALIGN_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[134].backends[5]));
	_register_command_CCV_NNC_ROI_ALIGN_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[135].backends[5]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[114].backends[5]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[115].backends[5]));
	_register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[168].backends[5]));
	_register_command_CCV_NNC_SCATTER_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[169].backends[5]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[182].backends[5]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[183].backends[5]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[100].backends[3]));
	_register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[101].backends[3]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[20].backends[5]));
	_register_command_CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[21].backends[5]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[50].backends[3]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[51].backends[3]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[132].backends[3]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[133].backends[3]));
	_register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[110].backends[5]));
	_register_command_CCV_NNC_SORT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[111].backends[5]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[68].backends[5]));
	_register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[69].backends[5]));
	_register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[112].backends[5]));
	_register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[113].backends[5]));
	_register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[136].backends[3]));
	_register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[137].backends[3]));
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[10].backends[5]));
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[11].backends[5]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[102].backends[5]));
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[103].backends[5]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[176].backends[3]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[177].backends[3]));
	_register_command_CCV_NNC_MASKED_FILL_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[160].backends[5]));
	_register_command_CCV_NNC_MASKED_FILL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[161].backends[5]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[54].backends[5]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[55].backends[5]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[128].backends[3]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[129].backends[3]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[8].backends[3]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[9].backends[3]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[138].backends[5]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[139].backends[5]));
#endif
#ifdef HAVE_MPS
	_register_command_CCV_NNC_ADAM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[30].backends[6]));
	_register_command_CCV_NNC_ADAMW_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[56].backends[6]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[16].backends[6]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[17].backends[6]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[94].backends[6]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[95].backends[6]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[84].backends[6]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[85].backends[6]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[190].backends[6]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[191].backends[6]));
	_register_command_CCV_NNC_CMUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[164].backends[6]));
	_register_command_CCV_NNC_CMUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[165].backends[6]));
	_register_command_CCV_NNC_SEGMENTED_GEMM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[62].backends[6]));
	_register_command_CCV_NNC_ROTATE_HALF_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[116].backends[6]));
	_register_command_CCV_NNC_ROTATE_HALF_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[117].backends[6]));
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[150].backends[6]));
	_register_command_CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[151].backends[6]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[48].backends[6]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[49].backends[6]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[184].backends[6]));
	_register_command_CCV_NNC_CONVOLUTION_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[185].backends[6]));
	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[152].backends[6]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[72].backends[6]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[158].backends[6]));
	_register_command_CCV_NNC_EWSOFTPLUS_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[126].backends[6]));
	_register_command_CCV_NNC_EWPOW_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[66].backends[6]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[98].backends[6]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[192].backends[6]));
	_register_command_CCV_NNC_EWSIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[4].backends[6]));
	_register_command_CCV_NNC_EWCOS_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[166].backends[6]));
	_register_command_CCV_NNC_EWABS_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[170].backends[6]));
	_register_command_CCV_NNC_CLAMP_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[120].backends[6]));
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[78].backends[6]));
	_register_command_CCV_NNC_FILL_IF_LESS_THAN_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[79].backends[6]));
	_register_command_CCV_NNC_GATED_DELTA_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[118].backends[6]));
	_register_command_CCV_NNC_GELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[40].backends[6]));
	_register_command_CCV_NNC_GELU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[41].backends[6]));
	_register_command_CCV_NNC_GRID_SAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[18].backends[6]));
	_register_command_CCV_NNC_HYPER_CONNECTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[36].backends[6]));
	_register_command_CCV_NNC_HYPER_CONNECTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[37].backends[6]));
	_register_command_CCV_NNC_INDEX_SELECT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[14].backends[6]));
	_register_command_CCV_NNC_REDUCE_ISNAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[90].backends[6]));
	_register_command_CCV_NNC_LEAKY_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[162].backends[6]));
	_register_command_CCV_NNC_LEAKY_RELU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[163].backends[6]));
	_register_command_CCV_NNC_MSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[106].backends[6]));
	_register_command_CCV_NNC_MSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[107].backends[6]));
	_register_command_CCV_NNC_MOE_ROUTING_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[70].backends[6]));
	_register_command_CCV_NNC_LAYER_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[76].backends[6]));
	_register_command_CCV_NNC_LAYER_NORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[77].backends[6]));
	_register_command_CCV_NNC_GROUP_NORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[186].backends[6]));
	_register_command_CCV_NNC_GROUP_NORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[187].backends[6]));
	_register_command_CCV_NNC_RMSNORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[46].backends[6]));
	_register_command_CCV_NNC_RMSNORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[47].backends[6]));
	_register_command_CCV_NNC_PAD_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[108].backends[6]));
	_register_command_CCV_NNC_PAD_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[109].backends[6]));
	_register_command_CCV_NNC_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[58].backends[6]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[86].backends[6]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[80].backends[6]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[172].backends[6]));
	_register_command_CCV_NNC_RANDOM_NORMAL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[148].backends[6]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[142].backends[6]));
	_register_command_CCV_NNC_REDUCE_MEAN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[88].backends[6]));
	_register_command_CCV_NNC_REDUCE_MEAN_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[89].backends[6]));
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[178].backends[6]));
	_register_command_CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[179].backends[6]));
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[140].backends[6]));
	_register_command_CCV_NNC_REDUCE_MIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[180].backends[6]));
	_register_command_CCV_NNC_REDUCE_NORM2_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[42].backends[6]));
	_register_command_CCV_NNC_REDUCE_NORM2_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[43].backends[6]));
	_register_command_CCV_NNC_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[92].backends[6]));
	_register_command_CCV_NNC_GUMBEL_ARGMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[6].backends[6]));
	_register_command_CCV_NNC_ARGMIN_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[188].backends[6]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[174].backends[6]));
	_register_command_CCV_NNC_RMSNORM_CMUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[28].backends[6]));
	_register_command_CCV_NNC_RMSNORM_GATED_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[96].backends[6]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[22].backends[6]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[23].backends[6]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[114].backends[6]));
	_register_command_CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[115].backends[6]));
	_register_command_CCV_NNC_SCATTER_ADD_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[168].backends[6]));
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[34].backends[6]));
	_register_command_CCV_NNC_SEGMENTED_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[35].backends[6]));
	_register_command_CCV_NNC_SIGMOID_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[100].backends[6]));
	_register_command_CCV_NNC_SIGMOID_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[101].backends[6]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[50].backends[6]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[51].backends[6]));
	_register_command_CCV_NNC_SORT_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[110].backends[6]));
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[24].backends[6]));
	_register_command_CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[25].backends[6]));
	_register_command_CCV_NNC_SWIGLU_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[64].backends[6]));
	_register_command_CCV_NNC_SWIGLU_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[65].backends[6]));
	_register_command_CCV_NNC_SWISH_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[68].backends[6]));
	_register_command_CCV_NNC_SWISH_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[69].backends[6]));
	_register_command_CCV_NNC_SWISH_MUL_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[112].backends[6]));
	_register_command_CCV_NNC_SWISH_MUL_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[113].backends[6]));
	_register_command_CCV_NNC_TANH_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[136].backends[6]));
	_register_command_CCV_NNC_TANH_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[137].backends[6]));
	_register_command_CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[10].backends[6]));
	_register_command_CCV_NNC_UPSAMPLE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[102].backends[6]));
	_register_command_CCV_NNC_UPSAMPLE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[103].backends[6]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[176].backends[6]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[177].backends[6]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[54].backends[6]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[55].backends[6]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[128].backends[6]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[129].backends[6]));
	_register_command_CCV_NNC_TRANSPOSE_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[8].backends[6]));
	_register_command_CCV_NNC_TRANSPOSE_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[9].backends[6]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[138].backends[6]));
	_register_command_CCV_NNC_DATATYPE_CONVERSION_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[139].backends[6]));
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[32].backends[6]));
	_register_command_CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_MPS(&(init_map[33].backends[6]));
#endif
}
