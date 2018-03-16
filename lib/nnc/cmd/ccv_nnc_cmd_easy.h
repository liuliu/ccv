// CCV_NNC_EWSUM_FORWARD
#define CMD_EWSUM_FORWARD() ccv_nnc_cmd(CCV_NNC_EWSUM_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWSUM_BACKWARD
#define CMD_EWSUM_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWSUM_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWPROD_FORWARD
#define CMD_EWPROD_FORWARD() ccv_nnc_cmd(CCV_NNC_EWPROD_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWPROD_BACKWARD
#define CMD_EWPROD_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWPROD_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWDIV_FORWARD
#define CMD_EWDIV_FORWARD() ccv_nnc_cmd(CCV_NNC_EWDIV_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWDIV_BACKWARD
#define CMD_EWDIV_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWDIV_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWEXP_FORWARD
#define CMD_EWEXP_FORWARD() ccv_nnc_cmd(CCV_NNC_EWEXP_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWEXP_BACKWARD
#define CMD_EWEXP_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWEXP_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWLOG_FORWARD
#define CMD_EWLOG_FORWARD() ccv_nnc_cmd(CCV_NNC_EWLOG_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWLOG_BACKWARD
#define CMD_EWLOG_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWLOG_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_MAX_POOL_FORWARD
#define CMD_MAX_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_MAX_POOL_BACKWARD
#define CMD_MAX_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_AVERAGE_POOL_FORWARD
#define CMD_AVERAGE_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_AVERAGE_POOL_BACKWARD
#define CMD_AVERAGE_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_CONVOLUTION_FORWARD
#define CMD_CONVOLUTION_FORWARD(_count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count}}), 0)
// CCV_NNC_CONVOLUTION_BACKWARD
#define CMD_CONVOLUTION_BACKWARD(_count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count}}), 0)
// CCV_NNC_SOFTMAX_FORWARD
#define CMD_SOFTMAX_FORWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SOFTMAX_BACKWARD
#define CMD_SOFTMAX_BACKWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_RELU_FORWARD
#define CMD_RELU_FORWARD() ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_RELU_BACKWARD
#define CMD_RELU_BACKWARD() ccv_nnc_cmd(CCV_NNC_RELU_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_GEMM_FORWARD
#define CMD_GEMM_FORWARD(_count) ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(_count), 0)
// CCV_NNC_GEMM_BACKWARD
#define CMD_GEMM_BACKWARD(_count) ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, 0, CMD_GEMM(_count), 0)
// CCV_NNC_AXPY_FORWARD
#define CMD_AXPY_FORWARD(...) ccv_nnc_cmd(CCV_NNC_AXPY_FORWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
// CCV_NNC_AXPY_BACKWARD
#define CMD_AXPY_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_AXPY_BACKWARD, 0, CMD_BLAS(__VA_ARGS__), 0)
// CCV_NNC_SET_FORWARD
#define CMD_SET_FORWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, CMD_BLAS(_val), 0)
// CCV_NNC_SET_BACKWARD
#define CMD_SET_BACKWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_BACKWARD, 0, CMD_BLAS(_val), 0)
// CCV_NNC_DATA_TRANSFER_FORWARD
#define CMD_DATA_TRANSFER_FORWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_DATA_TRANSFER_BACKWARD
#define CMD_DATA_TRANSFER_BACKWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_FORMAT_TRANSFORM_FORWARD
#define CMD_FORMAT_TRANSFORM_FORWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_FORMAT_TRANSFORM_BACKWARD
#define CMD_FORMAT_TRANSFORM_BACKWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, 0, ccv_nnc_cmd_auto, 0)

