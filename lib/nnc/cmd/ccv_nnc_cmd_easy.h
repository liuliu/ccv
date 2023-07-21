/**
 * @addtogroup available_commands Available Commands
 * @{
 */
// CCV_NNC_LAMB_FORWARD
#define CMD_LAMB_FORWARD(_step, _rate, _beta1, _beta2, _decay, _epsilon) ccv_nnc_cmd(CCV_NNC_LAMB_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lamb={.step=_step,.rate=_rate,.scale=1,.beta1=_beta1,.beta2=_beta2,.decay=_decay,.epsilon=_epsilon}}), 0)
// CCV_NNC_LEAKY_RELU_FORWARD
#define CMD_LEAKY_RELU_FORWARD(_negative_slope) ccv_nnc_cmd(CCV_NNC_LEAKY_RELU_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.leaky_relu={.negative_slope=_negative_slope}}, 0)
// CCV_NNC_LEAKY_RELU_BACKWARD
#define CMD_LEAKY_RELU_BACKWARD(_negative_slope) ccv_nnc_cmd(CCV_NNC_LEAKY_RELU_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.leaky_relu={.negative_slope=_negative_slope}}, 0)
// CCV_NNC_ROI_ALIGN_FORWARD
#define CMD_ROI_ALIGN_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_ROI_ALIGN_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_ROI_ALIGN_BACKWARD
#define CMD_ROI_ALIGN_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_ROI_ALIGN_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_GELU_FORWARD
#define CMD_GELU_FORWARD(_tanh) ccv_nnc_cmd(CCV_NNC_GELU_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gelu={.tanh=_tanh}}, 0)
// CCV_NNC_GELU_BACKWARD
#define CMD_GELU_BACKWARD(_tanh) ccv_nnc_cmd(CCV_NNC_GELU_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gelu={.tanh=_tanh}}, 0)
// CCV_NNC_UPSAMPLE_FORWARD
#define CMD_UPSAMPLE_FORWARD(_type, _width_scale, _height_scale) ccv_nnc_cmd(CCV_NNC_UPSAMPLE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.upsample={.type=_type,.width_scale=_width_scale,.height_scale=_height_scale}}), 0)
// CCV_NNC_UPSAMPLE_BACKWARD
#define CMD_UPSAMPLE_BACKWARD(_type, _width_scale, _height_scale) ccv_nnc_cmd(CCV_NNC_UPSAMPLE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.upsample={.type=_type,.width_scale=_width_scale,.height_scale=_height_scale}}), 0)
// CCV_NNC_GEMM_FORWARD
#define CMD_GEMM_FORWARD(...) ccv_nnc_cmd(CCV_NNC_GEMM_FORWARD, 0, CMD_GEMM(__VA_ARGS__), 0)
// CCV_NNC_GEMM_BACKWARD
#define CMD_GEMM_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_GEMM_BACKWARD, 0, CMD_GEMM(__VA_ARGS__), 0)
// CCV_NNC_ADD_FORWARD
#define CMD_ADD_FORWARD(_p, _q) ccv_nnc_cmd(CCV_NNC_ADD_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p, _q}}}, 0)
// CCV_NNC_ADD_BACKWARD
#define CMD_ADD_BACKWARD(_p, _q) ccv_nnc_cmd(CCV_NNC_ADD_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p, _q}}}, 0)
// CCV_NNC_MUL_FORWARD
#define CMD_MUL_FORWARD(_p) ccv_nnc_cmd(CCV_NNC_MUL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p,}}}, 0)
// CCV_NNC_MUL_BACKWARD
#define CMD_MUL_BACKWARD(_p) ccv_nnc_cmd(CCV_NNC_MUL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_p,}}}, 0)
// CCV_NNC_SCALAR_MUL_FORWARD
#define CMD_SCALAR_MUL_FORWARD(_a) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_a,}}}, 0)
// CCV_NNC_SCALAR_MUL_BACKWARD
#define CMD_SCALAR_MUL_BACKWARD(_a) ccv_nnc_cmd(CCV_NNC_SCALAR_MUL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_a,}}}, 0)
// CCV_NNC_RMSPROP_FORWARD
#define CMD_RMSPROP_FORWARD(_rate, _decay, _alpha, _momentum, _epsilon) ccv_nnc_cmd(CCV_NNC_RMSPROP_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.rmsprop={.rate=_rate,.scale=1,.decay=_decay,.alpha=_alpha,.momentum=_momentum,.epsilon=_epsilon}}), 0)
// CCV_NNC_SET_FORWARD
#define CMD_SET_FORWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_val,}}}, 0)
// CCV_NNC_SET_BACKWARD
#define CMD_SET_BACKWARD(_val) ccv_nnc_cmd(CCV_NNC_SET_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_val,}}}, 0)
// CCV_NNC_MASKED_FILL_FORWARD
#define CMD_MASKED_FILL_FORWARD(_eq, _fill) ccv_nnc_cmd(CCV_NNC_MASKED_FILL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_eq, _fill}}}, 0)
// CCV_NNC_MASKED_FILL_BACKWARD
#define CMD_MASKED_FILL_BACKWARD(_eq, _fill) ccv_nnc_cmd(CCV_NNC_MASKED_FILL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_eq, _fill}}}, 0)
// CCV_NNC_DATA_TRANSFER_FORWARD
#define CMD_DATA_TRANSFER_FORWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_DATA_TRANSFER_BACKWARD
#define CMD_DATA_TRANSFER_BACKWARD() ccv_nnc_cmd(CCV_NNC_DATA_TRANSFER_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_FORMAT_TRANSFORM_FORWARD
#define CMD_FORMAT_TRANSFORM_FORWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_FORMAT_TRANSFORM_BACKWARD
#define CMD_FORMAT_TRANSFORM_BACKWARD() ccv_nnc_cmd(CCV_NNC_FORMAT_TRANSFORM_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_TRANSPOSE_FORWARD
#define CMD_TRANSPOSE_FORWARD(_axis_a, _axis_b) ccv_nnc_cmd(CCV_NNC_TRANSPOSE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.transpose={.axis={_axis_a, _axis_b}}}), 0)
// CCV_NNC_TRANSPOSE_BACKWARD
#define CMD_TRANSPOSE_BACKWARD(_axis_a, _axis_b) ccv_nnc_cmd(CCV_NNC_TRANSPOSE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.transpose={.axis={_axis_a, _axis_b}}}), 0)
// CCV_NNC_DATATYPE_CONVERSION_FORWARD
#define CMD_DATATYPE_CONVERSION_FORWARD() ccv_nnc_cmd(CCV_NNC_DATATYPE_CONVERSION_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_DATATYPE_CONVERSION_BACKWARD
#define CMD_DATATYPE_CONVERSION_BACKWARD() ccv_nnc_cmd(CCV_NNC_DATATYPE_CONVERSION_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SMOOTH_L1_FORWARD
#define CMD_SMOOTH_L1_FORWARD(_b) ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.smooth_l1={.beta=_b}}), 0)
// CCV_NNC_SMOOTH_L1_BACKWARD
#define CMD_SMOOTH_L1_BACKWARD(_b) ccv_nnc_cmd(CCV_NNC_SMOOTH_L1_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.smooth_l1={.beta=_b}}), 0)
// CCV_NNC_MSE_FORWARD
#define CMD_MSE_FORWARD(_reduce_op) ccv_nnc_cmd(CCV_NNC_MSE_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.mse={.reduce_op=_reduce_op}}), 0)
// CCV_NNC_MSE_BACKWARD
#define CMD_MSE_BACKWARD(_reduce_op) ccv_nnc_cmd(CCV_NNC_MSE_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.mse={.reduce_op=_reduce_op}}), 0)
// CCV_NNC_BINARY_CROSSENTROPY_FORWARD
#define CMD_BINARY_CROSSENTROPY_FORWARD_X_0() ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=1}}), 0)
#define CMD_BINARY_CROSSENTROPY_FORWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_BINARY_CROSSENTROPY_FORWARD")
#define CMD_BINARY_CROSSENTROPY_FORWARD_X_1(_pos_weight) ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=_pos_weight}}), 0)
#define CMD_BINARY_CROSSENTROPY_FORWARD_X_SEL(_0, _1, _FX, ...) _FX
#define CMD_BINARY_CROSSENTROPY_FORWARD(...) CMD_BINARY_CROSSENTROPY_FORWARD_X_SEL(CMD_BINARY_CROSSENTROPY_FORWARD_X_F, ##__VA_ARGS__, CMD_BINARY_CROSSENTROPY_FORWARD_X_1, CMD_BINARY_CROSSENTROPY_FORWARD_X_0)(__VA_ARGS__)
// CCV_NNC_BINARY_CROSSENTROPY_BACKWARD
#define CMD_BINARY_CROSSENTROPY_BACKWARD_X_0() ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=1}}), 0)
#define CMD_BINARY_CROSSENTROPY_BACKWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_BINARY_CROSSENTROPY_BACKWARD")
#define CMD_BINARY_CROSSENTROPY_BACKWARD_X_1(_pos_weight) ccv_nnc_cmd(CCV_NNC_BINARY_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=_pos_weight}}), 0)
#define CMD_BINARY_CROSSENTROPY_BACKWARD_X_SEL(_0, _1, _FX, ...) _FX
#define CMD_BINARY_CROSSENTROPY_BACKWARD(...) CMD_BINARY_CROSSENTROPY_BACKWARD_X_SEL(CMD_BINARY_CROSSENTROPY_BACKWARD_X_F, ##__VA_ARGS__, CMD_BINARY_CROSSENTROPY_BACKWARD_X_1, CMD_BINARY_CROSSENTROPY_BACKWARD_X_0)(__VA_ARGS__)
// CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_0() ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_CATEGORICAL_CROSSENTROPY_FORWARD")
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_CATEGORICAL_CROSSENTROPY_FORWARD(...) CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_SEL(CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F, ##__VA_ARGS__, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_2, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_F, CMD_CATEGORICAL_CROSSENTROPY_FORWARD_X_0)(__VA_ARGS__)
// CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_0() ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_CATEGORICAL_CROSSENTROPY_BACKWARD")
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(...) CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_SEL(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F, ##__VA_ARGS__, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_2, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_F, CMD_CATEGORICAL_CROSSENTROPY_BACKWARD_X_0)(__VA_ARGS__)
// CCV_NNC_REDUCE_SUM_FORWARD
#define CMD_REDUCE_SUM_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_SUM_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_SUM_BACKWARD
#define CMD_REDUCE_SUM_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_SUM_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MEAN_FORWARD
#define CMD_REDUCE_MEAN_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MEAN_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MEAN_BACKWARD
#define CMD_REDUCE_MEAN_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MEAN_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MAX_FORWARD
#define CMD_REDUCE_MAX_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MAX_BACKWARD
#define CMD_REDUCE_MAX_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MAX_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MIN_FORWARD
#define CMD_REDUCE_MIN_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MIN_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_MIN_BACKWARD
#define CMD_REDUCE_MIN_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_MIN_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_NORM2_FORWARD
#define CMD_REDUCE_NORM2_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_NORM2_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_NORM2_BACKWARD
#define CMD_REDUCE_NORM2_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_NORM2_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_ARGMAX_FORWARD
#define CMD_ARGMAX_FORWARD(...) ccv_nnc_cmd(CCV_NNC_ARGMAX_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_ARGMAX_BACKWARD
#define CMD_ARGMAX_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_ARGMAX_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_ARGMIN_FORWARD
#define CMD_ARGMIN_FORWARD(...) ccv_nnc_cmd(CCV_NNC_ARGMIN_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_ARGMIN_BACKWARD
#define CMD_ARGMIN_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_ARGMIN_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
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
// CCV_NNC_EWSQRT_FORWARD
#define CMD_EWSQRT_FORWARD() ccv_nnc_cmd(CCV_NNC_EWSQRT_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_EWSQRT_BACKWARD
#define CMD_EWSQRT_BACKWARD() ccv_nnc_cmd(CCV_NNC_EWSQRT_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_CLAMP_FORWARD
#define CMD_CLAMP_FORWARD(_min, _max) ccv_nnc_cmd(CCV_NNC_CLAMP_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.clamp={.min=_min,.max=_max}}, 0)
// CCV_NNC_CLAMP_BACKWARD
#define CMD_CLAMP_BACKWARD(_min, _max) ccv_nnc_cmd(CCV_NNC_CLAMP_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.clamp={.min=_min,.max=_max}}, 0)
// CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_0() ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_SOFTMAX_CROSSENTROPY_FORWARD")
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_SOFTMAX_CROSSENTROPY_FORWARD(...) CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_SEL(CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_F, ##__VA_ARGS__, CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_2, CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_F, CMD_SOFTMAX_CROSSENTROPY_FORWARD_X_0)(__VA_ARGS__)
// CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_0() ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=0,.trim1=1}}), 0)
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_SOFTMAX_CROSSENTROPY_BACKWARD")
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_2(_trim0, _trim1) ccv_nnc_cmd(CCV_NNC_SOFTMAX_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.label_smoothing={.trim0=_trim0,.trim1=_trim1}}), 0)
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_SOFTMAX_CROSSENTROPY_BACKWARD(...) CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_SEL(CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_F, ##__VA_ARGS__, CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_2, CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_F, CMD_SOFTMAX_CROSSENTROPY_BACKWARD_X_0)(__VA_ARGS__)
// CCV_NNC_REDUCE_ISNAN_FORWARD
#define CMD_REDUCE_ISNAN_FORWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_ISNAN_FORWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_REDUCE_ISNAN_BACKWARD
#define CMD_REDUCE_ISNAN_BACKWARD(...) ccv_nnc_cmd(CCV_NNC_REDUCE_ISNAN_BACKWARD, 0, CMD_REDUCE(__VA_ARGS__), 0)
// CCV_NNC_COMM_ALLREDUCE_FORWARD
#define CMD_COMM_ALLREDUCE_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_ALLREDUCE_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMM_ALLREDUCE_BACKWARD
#define CMD_COMM_ALLREDUCE_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_ALLREDUCE_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMM_BROADCAST_FORWARD
#define CMD_COMM_BROADCAST_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_BROADCAST_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMM_ALLREDUCE_BACKWARD
#define CMD_COMM_BROADCAST_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_BROADCAST_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMM_REDUCE_FORWARD
#define CMD_COMM_REDUCE_FORWARD() ccv_nnc_cmd(CCV_NNC_COMM_REDUCE_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMM_REDUCE_BACKWARD
#define CMD_COMM_REDUCE_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMM_REDUCE_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_BATCH_NORM_FORWARD
#define CMD_BATCH_NORM_FORWARD(_epsilon, _is_test, _momentum, ...) ccv_nnc_cmd(CCV_NNC_BATCH_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.bnorm={.epsilon=_epsilon,.is_test=_is_test,.momentum=_momentum,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
// CCV_NNC_BATCH_NORM_BACKWARD
#define CMD_BATCH_NORM_BACKWARD(_epsilon, _is_test, _momentum, ...) ccv_nnc_cmd(CCV_NNC_BATCH_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.bnorm={.epsilon=_epsilon,.is_test=_is_test,.momentum=_momentum,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
// CCV_NNC_LAYER_NORM_FORWARD
#define CMD_LAYER_NORM_FORWARD(_epsilon, ...) ccv_nnc_cmd(CCV_NNC_LAYER_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=_epsilon,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
// CCV_NNC_LAYER_NORM_BACKWARD
#define CMD_LAYER_NORM_BACKWARD(_epsilon, ...) ccv_nnc_cmd(CCV_NNC_LAYER_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.lnorm={.epsilon=_epsilon,.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}}), 0)
// CCV_NNC_GROUP_NORM_FORWARD
#define CMD_GROUP_NORM_FORWARD(_group_axis, _groups, _epsilon, ...) ccv_nnc_cmd(CCV_NNC_GROUP_NORM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gnorm={.group_axis=_group_axis,.groups=_groups,.epsilon=_epsilon,.reduce_count=LIST_COUNT(__VA_ARGS__),.reduce_axis={__VA_ARGS__}}}), 0)
// CCV_NNC_GROUP_NORM_BACKWARD
#define CMD_GROUP_NORM_BACKWARD(_group_axis, _groups, _epsilon, ...) ccv_nnc_cmd(CCV_NNC_GROUP_NORM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.gnorm={.group_axis=_group_axis,.groups=_groups,.epsilon=_epsilon,.reduce_count=LIST_COUNT(__VA_ARGS__),.reduce_axis={__VA_ARGS__}}}), 0)
// CCV_NNC_TANH_FORWARD
#define CMD_TANH_FORWARD() ccv_nnc_cmd(CCV_NNC_TANH_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_TANH_BACKWARD
#define CMD_TANH_BACKWARD() ccv_nnc_cmd(CCV_NNC_TANH_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SGD_FORWARD
#define CMD_SGD_FORWARD(_nesterov, _rate, _scale, _decay, _momentum, _dampening) ccv_nnc_cmd(CCV_NNC_SGD_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.sgd={.nesterov=_nesterov,.rate=_rate,.scale=_scale,.decay=_decay,.momentum=_momentum,.dampening=_dampening}}), 0)
// CCV_NNC_DROPOUT_FORWARD
#define CMD_DROPOUT_FORWARD_X_F(...) ("This should not be used, you should have either 1 parameter or 2 parameters for CMD_DROPOUT_FORWARD")
#define CMD_DROPOUT_FORWARD_X_1(_p) ccv_nnc_cmd(CCV_NNC_DROPOUT_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p,.entirety=0}}), 0)
#define CMD_DROPOUT_FORWARD_X_2(_p, _entirety) ccv_nnc_cmd(CCV_NNC_DROPOUT_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p,.entirety=_entirety}}), 0)
#define CMD_DROPOUT_FORWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_DROPOUT_FORWARD(...) CMD_DROPOUT_FORWARD_X_SEL(CMD_DROPOUT_FORWARD_X_F, ##__VA_ARGS__, CMD_DROPOUT_FORWARD_X_2, CMD_DROPOUT_FORWARD_X_1, CMD_DROPOUT_FORWARD_X_F)(__VA_ARGS__)
// CCV_NNC_DROPOUT_BACKWARD
#define CMD_DROPOUT_BACKWARD_X_F(...) ("This should not be used, you should have either 1 parameter or 2 parameters for CMD_DROPOUT_FORWARD")
#define CMD_DROPOUT_BACKWARD_X_1(_p) ccv_nnc_cmd(CCV_NNC_DROPOUT_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p,.entirety=0}}), 0)
#define CMD_DROPOUT_BACKWARD_X_2(_p, _entirety) ccv_nnc_cmd(CCV_NNC_DROPOUT_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.dropout={.p=_p,.entirety=_entirety}}), 0)
#define CMD_DROPOUT_BACKWARD_X_SEL(_0, _1, _2, _FX, ...) _FX
#define CMD_DROPOUT_BACKWARD(...) CMD_DROPOUT_BACKWARD_X_SEL(CMD_DROPOUT_BACKWARD_X_F, ##__VA_ARGS__, CMD_DROPOUT_BACKWARD_X_2, CMD_DROPOUT_BACKWARD_X_1, CMD_DROPOUT_BACKWARD_X_F)(__VA_ARGS__)
// CCV_NNC_COMPRESSION_LSSC_FORWARD
#define CMD_COMPRESSION_LSSC_FORWARD() ccv_nnc_cmd(CCV_NNC_COMPRESSION_LSSC_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_COMPRESSION_LSSC_BACKWARD
#define CMD_COMPRESSION_LSSC_BACKWARD() ccv_nnc_cmd(CCV_NNC_COMPRESSION_LSSC_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SOFTMAX_FORWARD
#define CMD_SOFTMAX_FORWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SOFTMAX_BACKWARD
#define CMD_SOFTMAX_BACKWARD() ccv_nnc_cmd(CCV_NNC_SOFTMAX_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_MIN_FORWARD
#define CMD_MIN_FORWARD() ccv_nnc_cmd(CCV_NNC_MIN_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
// CCV_NNC_MIN_BACKWARD
#define CMD_MIN_BACKWARD() ccv_nnc_cmd(CCV_NNC_MIN_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
// CCV_NNC_MAX_FORWARD
#define CMD_MAX_FORWARD() ccv_nnc_cmd(CCV_NNC_MAX_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
// CCV_NNC_MAX_BACKWARD
#define CMD_MAX_BACKWARD() ccv_nnc_cmd(CCV_NNC_MAX_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}}, 0)
// CCV_NNC_RELU_FORWARD
#define CMD_RELU_FORWARD() ccv_nnc_cmd(CCV_NNC_RELU_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_RELU_BACKWARD
#define CMD_RELU_BACKWARD() ccv_nnc_cmd(CCV_NNC_RELU_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SIGMOID_FORWARD
#define CMD_SIGMOID_FORWARD() ccv_nnc_cmd(CCV_NNC_SIGMOID_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SIGMOID_BACKWARD
#define CMD_SIGMOID_BACKWARD() ccv_nnc_cmd(CCV_NNC_SIGMOID_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_INDEX_SELECT_FORWARD
#define CMD_INDEX_SELECT_FORWARD() ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_INDEX_SELECT_BACKWARD
#define CMD_INDEX_SELECT_BACKWARD() ccv_nnc_cmd(CCV_NNC_INDEX_SELECT_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD
#define CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_0() ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=1}}), 0)
#define CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 1 parameters for CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD")
#define CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_1(_pos_weight) ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=_pos_weight}}), 0)
#define CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_SEL(_0, _1, _FX, ...) _FX
#define CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD(...) CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_SEL(CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_F, ##__VA_ARGS__, CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_1, CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD_X_0)(__VA_ARGS__)
// CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD
#define CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_0() ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=1}}), 0)
#define CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_F(...) ("This should not be used, you should have either 0 parameter or 2 parameters for CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD")
#define CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_1(_pos_weight) ccv_nnc_cmd(CCV_NNC_SIGMOID_BINARY_CROSSENTROPY_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.binary_crossentropy={.pos_weight=_pos_weight}}), 0)
#define CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_SEL(_0, _1, _FX, ...) _FX
#define CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD(...) CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_SEL(CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_F, ##__VA_ARGS__, CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_1, CMD_SIGMOID_BINARY_CROSSENTROPY_BACKWARD_X_0)(__VA_ARGS__)
// CCV_NNC_NMS_FORWARD
#define CMD_NMS_FORWARD(_iou_threshold) ccv_nnc_cmd(CCV_NNC_NMS_FORWARD, 0, ((ccv_nnc_cmd_param_t){.nms={.iou_threshold=_iou_threshold}}), 0)
// CCV_NNC_NMS_BACKWARD
#define CMD_NMS_BACKWARD(_iou_threshold) ccv_nnc_cmd(CCV_NNC_NMS_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.nms={.iou_threshold=_iou_threshold}}), 0)
// CCV_NNC_RANDOM_UNIFORM_FORWARD
#define CMD_RANDOM_UNIFORM_FORWARD(_lb, _ub) ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_lb, _ub}}}, 0)
// CCV_NNC_RANDOM_UNIFORM_BACKWARD
#define CMD_RANDOM_UNIFORM_BACKWARD(_lb, _ub) ccv_nnc_cmd(CCV_NNC_RANDOM_UNIFORM_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_lb, _ub}}}, 0)
// CCV_NNC_RANDOM_NORMAL_FORWARD
#define CMD_RANDOM_NORMAL_FORWARD(_std, _mean) ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_FORWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_std, _mean}}}, 0)
// CCV_NNC_RANDOM_NORMAL_BACKWARD
#define CMD_RANDOM_NORMAL_BACKWARD(_std, _mean) ccv_nnc_cmd(CCV_NNC_RANDOM_NORMAL_BACKWARD, 0, (ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={_std, _mean}}}, 0)
// CCV_NNC_ADAM_FORWARD
#define CMD_ADAM_FORWARD(_step, _rate, _beta1, _beta2, _decay, _epsilon) ccv_nnc_cmd(CCV_NNC_ADAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.adam={.step=_step,.rate=_rate,.scale=1,.beta1=_beta1,.beta2=_beta2,.decay=_decay,.epsilon=_epsilon}}), 0)
// CCV_NNC_ADAMW_FORWARD
#define CMD_ADAMW_FORWARD(_step, _rate, _beta1, _beta2, _decay, _epsilon) ccv_nnc_cmd(CCV_NNC_ADAMW_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.adam={.step=_step,.rate=_rate,.scale=1,.beta1=_beta1,.beta2=_beta2,.decay=_decay,.epsilon=_epsilon}}), 0)
// CCV_NNC_MAX_POOL_FORWARD
#define CMD_MAX_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_MAX_POOL_BACKWARD
#define CMD_MAX_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_MAX_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_AVERAGE_POOL_FORWARD
#define CMD_AVERAGE_POOL_FORWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_AVERAGE_POOL_BACKWARD
#define CMD_AVERAGE_POOL_BACKWARD(rows, cols) ccv_nnc_cmd(CCV_NNC_AVERAGE_POOL_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={rows, cols,1}}}), 0)
// CCV_NNC_CONVOLUTION_FORWARD
#define CMD_CONVOLUTION_FORWARD(_groups, _count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count,.groups=_groups}}), 0)
// CCV_NNC_CONVOLUTION_BACKWARD
#define CMD_CONVOLUTION_BACKWARD(_groups, _count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count,.groups=_groups}}), 0)
// CCV_NNC_LSTM_FORWARD
#define CMD_LSTM_FORWARD(_hidden_size, _proj_size, _num_layers, _bias, _batch_first, _bidirectional, _dropout, _is_test) ccv_nnc_cmd(CCV_NNC_LSTM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.rnn={.hidden_size=_hidden_size,.proj_size=_proj_size,.num_layers=_num_layers,.bias=_bias,.batch_first=_batch_first,.bidirectional=_bidirectional,.dropout=_dropout,.is_test=_is_test}}), 0)
// CCV_NNC_LSTM_BACKWARD
#define CMD_LSTM_BACKWARD(_hidden_size, _proj_size, _num_layers, _bias, _batch_first, _bidirectional, _dropout, _is_test) ccv_nnc_cmd(CCV_NNC_LSTM_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.rnn={.hidden_size=_hidden_size,.proj_size=_proj_size,.num_layers=_num_layers,.bias=_bias,.batch_first=_batch_first,.bidirectional=_bidirectional,.dropout=_dropout,.is_test=_is_test}}), 0)
// CCV_NNC_SWISH_FORWARD
#define CMD_SWISH_FORWARD() ccv_nnc_cmd(CCV_NNC_SWISH_FORWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_SWISH_BACKWARD
#define CMD_SWISH_BACKWARD() ccv_nnc_cmd(CCV_NNC_SWISH_BACKWARD, 0, ccv_nnc_cmd_auto, 0)
// CCV_NNC_HISTOGRAM_FORWARD
#define CMD_HISTOGRAM_EVEN(_bins, _min, _max) ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_EVEN,.bins=_bins,.min=_min,.max=_max}}), 0)
// CCV_NNC_HISTOGRAM_FORWARD
#define CMD_HISTOGRAM_LOG_X_0() ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_LOGARITHMIC,.min=1e-12,.max=1e20,.rate=1.1}}), 0)
#define CMD_HISTOGRAM_LOG_X_F(...) ("This should not be used, you should have either 0 parameter or 3 parameters for CMD_HISTOGRAM_LOG")
#define CMD_HISTOGRAM_LOG_X_2(_min, _max, _rate) ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_LOGARITHMIC,.min=_min,.max=_max,.rate=_rate}}), 0)
#define CMD_HISTOGRAM_LOG_X_SEL(_0, _1, _2, _3, _FX, ...) _FX
#define CMD_HISTOGRAM_LOG(...) CMD_HISTOGRAM_LOG_X_SEL(CMD_HISTOGRAM_LOG_X_F, ##__VA_ARGS__, CMD_HISTOGRAM_LOG_X_3, CMD_HISTOGRAM_LOG_X_F, CMD_HISTOGRAM_LOG_X_F, CMD_HISTOGRAM_LOG_X_0)(__VA_ARGS__)
// CCV_NNC_HISTOGRAM_FORWARD
#define CMD_HISTOGRAM_BINS() ccv_nnc_cmd(CCV_NNC_HISTOGRAM_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.histogram={.type=CCV_NNC_HISTOGRAM_BINS}}), 0)

/** @} */
