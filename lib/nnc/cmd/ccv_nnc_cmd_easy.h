// CCV_NNC_CONVOLUTION_FORWARD
#define CMD_CONVOLUTION_FORWARD(_count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_FORWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count}}), 0)
// CCV_NNC_CONVOLUTION_BACKWARD
#define CMD_CONVOLUTION_BACKWARD(_count, ...) ccv_nnc_cmd(CCV_NNC_CONVOLUTION_BACKWARD, 0, ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}},.convolution={.count=_count}}), 0)

