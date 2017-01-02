static ccv_nnc_cmd_init_t init_map[] = {
	{.name = "CCV_NNC_EWSUM_FORWARD", .cmd = 0xe21a2c4c},
	{.name = "CCV_NNC_EWSUM_BACKWARD", .cmd = 0xe21a2c4d},
	{.name = "CCV_NNC_MAX_POOL_FORWARD", .cmd = 0x7bec9360},
	{.name = "CCV_NNC_MAX_POOL_BACKWARD", .cmd = 0x7bec9361},
	{.name = "CCV_NNC_GEMM_FORWARD", .cmd = 0x7e87d00c},
	{.name = "CCV_NNC_GEMM_BACKWARD", .cmd = 0x7e87d00d},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_FORWARD", .cmd = 0xe4a2b192},
	{.name = "CCV_NNC_FORMAT_TRANSFORM_BACKWARD", .cmd = 0xe4a2b193},
	{.name = "CCV_NNC_EWDIV_FORWARD", .cmd = 0x1cd2fa18},
	{.name = "CCV_NNC_EWDIV_BACKWARD", .cmd = 0x1cd2fa19},
	{.name = "CCV_NNC_AXPY_FORWARD", .cmd = 0x8b86e274},
	{.name = "CCV_NNC_AXPY_BACKWARD", .cmd = 0x8b86e275},
	{.name = "CCV_NNC_AVERAGE_POOL_FORWARD", .cmd = 0x51267ab8},
	{.name = "CCV_NNC_AVERAGE_POOL_BACKWARD", .cmd = 0x51267ab9},
	{.name = "CCV_NNC_RELU_FORWARD", .cmd = 0xc51eaa80},
	{.name = "CCV_NNC_RELU_BACKWARD", .cmd = 0xc51eaa81},
	{.name = "CCV_NNC_CONVOLUTION_FORWARD", .cmd = 0x254d05f4},
	{.name = "CCV_NNC_CONVOLUTION_BACKWARD", .cmd = 0x254d05f5},
	{.name = "CCV_NNC_DATA_TRANSFER_FORWARD", .cmd = 0x12d21e1a},
	{.name = "CCV_NNC_DATA_TRANSFER_BACKWARD", .cmd = 0x12d21e1b},
	{.name = "CCV_NNC_EWEXP_FORWARD", .cmd = 0xd784b170},
	{.name = "CCV_NNC_EWEXP_BACKWARD", .cmd = 0xd784b171},
	{.name = "CCV_NNC_SOFTMAX_FORWARD", .cmd = 0xc969a252},
	{.name = "CCV_NNC_SOFTMAX_BACKWARD", .cmd = 0xc969a253},
	{.name = "CCV_NNC_EWLOG_FORWARD", .cmd = 0xf4191bf2},
	{.name = "CCV_NNC_EWLOG_BACKWARD", .cmd = 0xf4191bf3},
	{.name = "CCV_NNC_EWPROD_FORWARD", .cmd = 0xee07e8fe},
	{.name = "CCV_NNC_EWPROD_BACKWARD", .cmd = 0xee07e8ff},
	{.name = "CCV_NNC_SET_FORWARD", .cmd = 0x2b070804},
	{.name = "CCV_NNC_SET_BACKWARD", .cmd = 0x2b070805},
};

static ccv_nnc_cmd_backend_init_t backend_init_map[] = {
	{.name = "CCV_NNC_BACKEND_CPU_REF", .backend = 0x3d9883e5},
	{.name = "CCV_NNC_BACKEND_GPU_REF", .backend = 0x5f19790a},
	{.name = "CCV_NNC_BACKEND_CPU_OPT", .backend = 0x46deb194},
	{.name = "CCV_NNC_BACKEND_GPU_CUDNN", .backend = 0x854b679a},
};

static inline int _ccv_nnc_cmd_ph(const uint32_t cmd)
{
	switch ((cmd >> 7) % 2)
	{
		case 0:
			return ((((cmd >> 5) % 15) + 0) << 1) | (cmd & 1);
		case 1:
		default:
			return ((((cmd >> 15) % 12) + 2) << 1) | (cmd & 1);
	}
}

static inline int _ccv_nnc_cmd_backend_ph(const uint32_t backend)
{
	switch ((backend >> 0) % 1)
	{
		case 0:
		default:
			return ((backend >> 3) % 4) + 0;
	}
}

void _register_command_CCV_NNC_EWSUM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWSUM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWDIV_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AXPY_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AXPY_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWEXP_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWLOG_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_EWPROD_BACKWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD(ccv_nnc_cmd_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD(ccv_nnc_cmd_registry_t* const registry);

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
void _register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AXPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_AXPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
#ifdef HAVE_CUDA
void _register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
void _register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(ccv_nnc_cmd_backend_registry_t* const registry);
#endif

static inline void _ccv_nnc_cmd_init(void)
{
	_register_command_CCV_NNC_EWSUM_FORWARD(&init_map[0].registry);
	_register_command_CCV_NNC_EWSUM_BACKWARD(&init_map[1].registry);
	_register_command_CCV_NNC_MAX_POOL_FORWARD(&init_map[2].registry);
	_register_command_CCV_NNC_MAX_POOL_BACKWARD(&init_map[3].registry);
	_register_command_CCV_NNC_GEMM_FORWARD(&init_map[4].registry);
	_register_command_CCV_NNC_GEMM_BACKWARD(&init_map[5].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD(&init_map[6].registry);
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD(&init_map[7].registry);
	_register_command_CCV_NNC_EWDIV_FORWARD(&init_map[8].registry);
	_register_command_CCV_NNC_EWDIV_BACKWARD(&init_map[9].registry);
	_register_command_CCV_NNC_AXPY_FORWARD(&init_map[10].registry);
	_register_command_CCV_NNC_AXPY_BACKWARD(&init_map[11].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD(&init_map[12].registry);
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD(&init_map[13].registry);
	_register_command_CCV_NNC_RELU_FORWARD(&init_map[14].registry);
	_register_command_CCV_NNC_RELU_BACKWARD(&init_map[15].registry);
	_register_command_CCV_NNC_CONVOLUTION_FORWARD(&init_map[16].registry);
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD(&init_map[17].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD(&init_map[18].registry);
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD(&init_map[19].registry);
	_register_command_CCV_NNC_EWEXP_FORWARD(&init_map[20].registry);
	_register_command_CCV_NNC_EWEXP_BACKWARD(&init_map[21].registry);
	_register_command_CCV_NNC_SOFTMAX_FORWARD(&init_map[22].registry);
	_register_command_CCV_NNC_SOFTMAX_BACKWARD(&init_map[23].registry);
	_register_command_CCV_NNC_EWLOG_FORWARD(&init_map[24].registry);
	_register_command_CCV_NNC_EWLOG_BACKWARD(&init_map[25].registry);
	_register_command_CCV_NNC_EWPROD_FORWARD(&init_map[26].registry);
	_register_command_CCV_NNC_EWPROD_BACKWARD(&init_map[27].registry);
	_register_command_CCV_NNC_SET_FORWARD(&init_map[28].registry);
	_register_command_CCV_NNC_SET_BACKWARD(&init_map[29].registry);

	_register_command_CCV_NNC_EWSUM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[0].backends[0]));
	_register_command_CCV_NNC_EWSUM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[1].backends[0]));
	_register_command_CCV_NNC_EWPROD_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[26].backends[0]));
	_register_command_CCV_NNC_EWPROD_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[27].backends[0]));
	_register_command_CCV_NNC_EWDIV_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[8].backends[0]));
	_register_command_CCV_NNC_EWDIV_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[9].backends[0]));
	_register_command_CCV_NNC_EWEXP_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[20].backends[0]));
	_register_command_CCV_NNC_EWEXP_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[21].backends[0]));
	_register_command_CCV_NNC_EWLOG_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[24].backends[0]));
	_register_command_CCV_NNC_EWLOG_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[25].backends[0]));
	_register_command_CCV_NNC_MAX_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[2].backends[0]));
	_register_command_CCV_NNC_MAX_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[3].backends[0]));
	_register_command_CCV_NNC_AVERAGE_POOL_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[12].backends[0]));
	_register_command_CCV_NNC_AVERAGE_POOL_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[13].backends[0]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[16].backends[0]));
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[16].backends[2]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[17].backends[0]));
	_register_command_CCV_NNC_SOFTMAX_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[22].backends[0]));
	_register_command_CCV_NNC_SOFTMAX_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[23].backends[0]));
	_register_command_CCV_NNC_RELU_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[14].backends[0]));
	_register_command_CCV_NNC_RELU_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[15].backends[0]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[4].backends[0]));
	_register_command_CCV_NNC_GEMM_FORWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[4].backends[2]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[5].backends[0]));
	_register_command_CCV_NNC_GEMM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_OPT(&(init_map[5].backends[2]));
	_register_command_CCV_NNC_AXPY_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[10].backends[0]));
	_register_command_CCV_NNC_AXPY_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[11].backends[0]));
	_register_command_CCV_NNC_SET_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[28].backends[0]));
	_register_command_CCV_NNC_SET_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[29].backends[0]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[18].backends[0]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[19].backends[0]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[6].backends[0]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_CPU_REF(&(init_map[7].backends[0]));
#ifdef HAVE_CUDA
	_register_command_CCV_NNC_CONVOLUTION_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[16].backends[3]));
	_register_command_CCV_NNC_CONVOLUTION_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[17].backends[3]));
	_register_command_CCV_NNC_DATA_TRANSFER_FORWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[18].backends[1]));
	_register_command_CCV_NNC_DATA_TRANSFER_BACKWARD_backend_CCV_NNC_BACKEND_GPU_REF(&(init_map[19].backends[1]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_FORWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[6].backends[3]));
	_register_command_CCV_NNC_FORMAT_TRANSFORM_BACKWARD_backend_CCV_NNC_BACKEND_GPU_CUDNN(&(init_map[7].backends[3]));
#endif
}
