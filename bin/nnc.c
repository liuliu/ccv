#include <ccv.h>
#include <nnc/ccv_nnc.h>

int main(int argc, char** argv)
{
	ccv_nnc_net_param_t net_params = {
		.size = {
			.dim = {
				7, 7
			}
		},
		.convolutional = {
			.count = 128,
			.channels = 3
		}
	};
	ccv_nnc_net_t* net = ccv_nnc_net_new(0, CCV_NNC_TYPE_CONVOLUTIONAL, net_params, 0);
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			225, 225
		},
		.channels = 3
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			111, 111
		},
		.channels = 128
	};
	ccv_nnc_net_hint_t hint = ccv_nnc_net_hint_guess(net, a_params, b_params);
	printf("dim %d %d, front %d, %d, back %d, %d\n", hint.stride.dim[0], hint.stride.dim[1], hint.border.front[0], hint.border.front[1], hint.border.back[0], hint.border.back[1]);
	return 0;
}
