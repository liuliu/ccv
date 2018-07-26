static ccv_nnc_cmd_init_t init_map[] = {
	{.name = "CCV_NNC_RELU_FORWARD", .cmd = 0xc51eaa80},
	{.name = "CCV_NNC_RELU_BACKWARD", .cmd = 0xc51eaa81},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_FORWARD", .cmd = 0xe4a2b192},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_BACKWARD", .cmd = 0xe4a2b193},
	{.name = "CCV_NNC_BATCH_NORM_FORWARD", .cmd = 0x5419819c},
	{.name = "CCV_NNC_BATCH_NORM_BACKWARD", .cmd = 0x5419819d},
	{.name = "CCV_NNC_REDUCE_MAX_FORWARD", .cmd = 0x80f1a506},
	{.name = "CCV_NNC_REDUCE_MAX_BACKWARD", .cmd = 0x80f1a507},
	{.name = "CCV_NNC_REDUCE_SUM_FORWARD", .cmd = 0x52970f06},
	{.name = "CCV_NNC_REDUCE_SUM_BACKWARD", .cmd = 0x52970f07},
	{.name = "CCV_NNC_SOFTMAX_FORWARD", .cmd = 0xc969a252},
	{.name = "CCV_NNC_SOFTMAX_BACKWARD", .cmd = 0xc969a253},
	{.name = "CCV_NNC_AVERAGE_POOL_FORWARD", .cmd = 0x51267ab8},
	{.name = "CCV_NNC_AVERAGE_POOL_BACKWARD", .cmd = 0x51267ab9},
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD", .cmd = 0x1eb327a2},
	{.name = "CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD", .cmd = 0x1eb327a3},
	{.name = "CCV_NNC_EWSUM_FORWARD", .cmd = 0xe21a2c4c},
	{.name = "CCV_NNC_EWSUM_BACKWARD", .cmd = 0xe21a2c4d},
	{.name = "CCV_NNC_MUL_FORWARD", .cmd = 0x24721a46},
	{.name = "CCV_NNC_MUL_BACKWARD", .cmd = 0x24721a47},
	{.name = "CCV_NNC_MAX_POOL_FORWARD", .cmd = 0x7bec9360},
	{.name = "CCV_NNC_MAX_POOL_BACKWARD", .cmd = 0x7bec9361},
	{.name = "CCV_NNC_ADD_FORWARD", .cmd = 0x58fb3664},
	{.name = "CCV_NNC_ADD_BACKWARD", .cmd = 0x58fb3665},
	{.name = "CCV_NNC_EWEXP_FORWARD", .cmd = 0xd784b170},
	{.name = "CCV_NNC_EWEXP_BACKWARD", .cmd = 0xd784b171},
	{.name = "CCV_NNC_SGD_FORWARD", .cmd = 0xe650ad26},
	{.name = "CCV_NNC_SGD_BACKWARD", .cmd = 0xe650ad27},
	{.name = "CCV_NNC_RANDOM_UNIFORM_FORWARD", .cmd = 0xa0cd1d5e},
	{.name = "CCV_NNC_RANDOM_UNIFORM_BACKWARD", .cmd = 0xa0cd1d5f},
	{.name = "CCV_NNC_EWSQRT_FORWARD", .cmd = 0x8870a61e},
	{.name = "CCV_NNC_EWSQRT_BACKWARD", .cmd = 0x8870a61f},
	{.name = "CCV_NNC_CONVOLUTION_FORWARD", .cmd = 0x254d05f4},
	{.name = "CCV_NNC_CONVOLUTION_BACKWARD", .cmd = 0x254d05f5},
	{.name = "CCV_NNC_SCALAR_MUL_FORWARD", .cmd = 0x8b4d86aa},
	{.name = "CCV_NNC_SCALAR_MUL_BACKWARD", .cmd = 0x8b4d86ab},
	{.name = "CCV_NNC_SET_FORWARD", .cmd = 0x2b070804},
	{.name = "CCV_NNC_SET_BACKWARD", .cmd = 0x2b070805},
	{.name = "CCV_NNC_DROPOUT_FORWARD", .cmd = 0x7f2dc3e4},
	{.name = "CCV_NNC_DROPOUT_BACKWARD", .cmd = 0x7f2dc3e5},
	{.name = "CCV_NNC_EWLOG_FORWARD", .cmd = 0xf4191bf2},
	{.name = "CCV_NNC_EWLOG_BACKWARD", .cmd = 0xf4191bf3},
	{.name = "CCV_NNC_EWDIV_FORWARD", .cmd = 0x1cd2fa18},
	{.name = "CCV_NNC_EWDIV_BACKWARD", .cmd = 0x1cd2fa19},
	{.name = "CCV_NNC_DATA_TRANSFER_FORWARD", .cmd = 0x12d21e1a},
	{.name = "CCV_NNC_DATA_TRANSFER_BACKWARD", .cmd = 0x12d21e1b},
	{.name = "CCV_NNC_GEMM_FORWARD", .cmd = 0x7e87d00c},
	{.name = "CCV_NNC_GEMM_BACKWARD", .cmd = 0x7e87d00d},
	{.name = "CCV_NNC_EWPROD_FORWARD", .cmd = 0xee07e8fe},
	{.name = "CCV_NNC_EWPROD_BACKWARD", .cmd = 0xee07e8ff},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD", .cmd = 0xc26b7b5e},
	{.name = "CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD", .cmd = 0xc26b7b5f},
};

static ccv_nnc_cmd_backend_init_t backend_init_map[] = {
	{.name = "CCV_NNC_BACKEND_GPU_CUBLAS", .backend = 0x9b8cfed},
	{.name = "CCV_NNC_BACKEND_CPU_REF", .backend = 0x3d9883e5},
	{.name = "CCV_NNC_BACKEND_GPU_REF", .backend = 0x5f19790a},
	{.name = "CCV_NNC_BACKEND_CPU_OPT", .backend = 0x46deb194},
	{.name = "CCV_NNC_BACKEND_GPU_CUDNN", .backend = 0x854b679a},
};

static inline int _ccv_nnc_cmd_ph(const uint32_t cmd)
{
	switch ((cmd >> 14) % 4)
	{
		case 0:
			return ((((cmd >> 21) % 22) + 4) << 1) | (cmd & 1);
		case 1:
			return ((((cmd >> 1) % 20) + 6) << 1) | (cmd & 1);
		case 2:
			return ((((cmd >> 1) % 18) + 0) << 1) | (cmd & 1);
		case 3:
		default:
			return ((((cmd >> 1) % 8) + 17) << 1) | (cmd & 1);
	}
}

static inline int _ccv_nnc_cmd_backend_ph(const uint32_t backend)
{
	switch ((backend >> 0) % 1)
	{
		case 0:
		default:
			return ((backend >> 12) % 5) + 0;
	}
}

void _register_command_CCV_NNC_RELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSQRT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SCALAR_MUL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);

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
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
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
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
#ifdef HAVE_CUDA
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
#endif

static inline void _ccv_nnc_cmd_init(void)
{
	_register_command_CCV_NNC_RELU_FORWARD(&init_map[0].registry);
	_register_command_CCV_NNC_RELU_BACKWARD(&init_map[1].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(&init_map[2].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(&init_map[3].registry);
	_register_command_CCV_NNC_BATCH_NORM_FORWARD(&init_map[4].registry);
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD(&init_map[5].registry);
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD(&init_map[6].registry);
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD(&init_map[7].registry);
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD(&init_map[8].registry);
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD(&init_map[9].registry);
	_register_command_CCV_NNC_SOFTMAX_FORWARD(&init_map[10].registry);
	_register_command_CCV_NNC_SOFTMAX_BACKWARD(&init_map[11].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD(&init_map[12].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(&init_map[13].registry);
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD(&init_map[14].registry);
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD(&init_map[15].registry);
	_register_command_CCV_NNC_EWSUM_FORWARD(&init_map[16].registry);
	_register_command_CCV_NNC_EWSUM_BACKWARD(&init_map[17].registry);
	_register_command_CCV_NNC_MUL_FORWARD(&init_map[18].registry);
	_register_command_CCV_NNC_MUL_BACKWARD(&init_map[19].registry);
	_register_command_CCV_NNC_MAX_POOL_FORWARD(&init_map[20].registry);
	_register_command_CCV_NNC_MAX_POOL_BACKWARD(&init_map[21].registry);
	_register_command_CCV_NNC_ADD_FORWARD(&init_map[22].registry);
	_register_command_CCV_NNC_ADD_BACKWARD(&init_map[23].registry);
	_register_command_CCV_NNC_EWEXP_FORWARD(&init_map[24].registry);
	_register_command_CCV_NNC_EWEXP_BACKWARD(&init_map[25].registry);
	_register_command_CCV_NNC_SGD_FORWARD(&init_map[26].registry);
	_register_command_CCV_NNC_SGD_BACKWARD(&init_map[27].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD(&init_map[28].registry);
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD(&init_map[29].registry);
	_register_command_CCV_NNC_EWSQRT_FORWARD(&init_map[30].registry);
	_register_command_CCV_NNC_EWSQRT_BACKWARD(&init_map[31].registry);
	_register_command_CCV_NNC_CONVOLUTION_FORWARD(&init_map[32].registry);
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD(&init_map[33].registry);
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD(&init_map[34].registry);
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD(&init_map[35].registry);
	_register_command_CCV_NNC_SET_FORWARD(&init_map[36].registry);
	_register_command_CCV_NNC_SET_BACKWARD(&init_map[37].registry);
	_register_command_CCV_NNC_DROPOUT_FORWARD(&init_map[38].registry);
	_register_command_CCV_NNC_DROPOUT_BACKWARD(&init_map[39].registry);
	_register_command_CCV_NNC_EWLOG_FORWARD(&init_map[40].registry);
	_register_command_CCV_NNC_EWLOG_BACKWARD(&init_map[41].registry);
	_register_command_CCV_NNC_EWDIV_FORWARD(&init_map[42].registry);
	_register_command_CCV_NNC_EWDIV_BACKWARD(&init_map[43].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD(&init_map[44].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(&init_map[45].registry);
	_register_command_CCV_NNC_GEMM_FORWARD(&init_map[46].registry);
	_register_command_CCV_NNC_GEMM_BACKWARD(&init_map[47].registry);
	_register_command_CCV_NNC_EWPROD_FORWARD(&init_map[48].registry);
	_register_command_CCV_NNC_EWPROD_BACKWARD(&init_map[49].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD(&init_map[50].registry);
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD(&init_map[51].registry);

	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[16].backends[1]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[17].backends[1]));
	_register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[48].backends[1]));
	_register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[49].backends[1]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[42].backends[1]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[43].backends[1]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[24].backends[1]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[25].backends[1]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[40].backends[1]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[41].backends[1]));
	_register_command_CCV_NNC_EWSQRT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[30].backends[1]));
	_register_command_CCV_NNC_EWSQRT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[31].backends[1]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[20].backends[1]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[21].backends[1]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[12].backends[1]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[13].backends[1]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[32].backends[1]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[32].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[33].backends[1]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[26].backends[1]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[27].backends[1]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[10].backends[1]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[11].backends[1]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[28].backends[1]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[29].backends[1]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[14].backends[1]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[15].backends[1]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[0].backends[1]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[1].backends[1]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[38].backends[1]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[39].backends[1]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[50].backends[1]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[51].backends[1]));
	_register_command_CCV_NNC_REDUCE_SUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[8].backends[1]));
	_register_command_CCV_NNC_REDUCE_SUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[9].backends[1]));
	_register_command_CCV_NNC_REDUCE_MAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[6].backends[1]));
	_register_command_CCV_NNC_REDUCE_MAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[7].backends[1]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[4].backends[1]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[5].backends[1]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[46].backends[1]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[46].backends[3]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[47].backends[1]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[47].backends[3]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[22].backends[1]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[23].backends[1]));
	_register_command_CCV_NNC_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[18].backends[1]));
	_register_command_CCV_NNC_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[19].backends[1]));
	_register_command_CCV_NNC_SCALAR_MUL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[34].backends[1]));
	_register_command_CCV_NNC_SCALAR_MUL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[35].backends[1]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[36].backends[1]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[37].backends[1]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[44].backends[1]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[45].backends[1]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[2].backends[1]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[3].backends[1]));
#ifdef HAVE_CUDA
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[20].backends[4]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[21].backends[4]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[12].backends[4]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[13].backends[4]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[32].backends[4]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[33].backends[4]));
	_register_command_CCV_NNC_SGD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[26].backends[4]));
	_register_command_CCV_NNC_SGD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[27].backends[4]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[10].backends[4]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[11].backends[4]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[28].backends[2]));
	_register_command_CCV_NNC_RANDOM_UNIFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[29].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[14].backends[2]));
	_register_command_CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[15].backends[2]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[0].backends[4]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[1].backends[4]));
	_register_command_CCV_NNC_DROPOUT_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[38].backends[4]));
	_register_command_CCV_NNC_DROPOUT_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[39].backends[4]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[50].backends[4]));
	_register_command_CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[51].backends[4]));
	_register_command_CCV_NNC_BATCH_NORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[4].backends[4]));
	_register_command_CCV_NNC_BATCH_NORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[5].backends[4]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[46].backends[0]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUBLAS(&(init_map[47].backends[0]));
	_register_command_CCV_NNC_ADD_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[22].backends[4]));
	_register_command_CCV_NNC_ADD_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[23].backends[4]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[36].backends[4]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[37].backends[4]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[44].backends[2]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[45].backends[2]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[2].backends[4]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[3].backends[4]));
#endif
}
