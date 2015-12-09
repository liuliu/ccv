#include <ccv.h>
#include <nnc/ccv_nnc.h>

int main(int argc, char** argv)
{
	ccv_nnc_init();
	ccv_nnc_net_param_t net_params = {
		.size = {
			.dim = {
				3, 7, 7
			}
		},
		.convolutional = {
			.count = 128,
		}
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 225, 225
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			128, 111, 111
		},
	};
	ccv_nnc_net_hint_t hint = ccv_nnc_net_hint_guess(net_params, &a_params, 1, &b_params, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_net_t* net = ccv_nnc_net_new(0, CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, net_params, 0);
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 7, 7, 128,
		},
	};
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			128
		},
	};
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* inputs[] = {
		a,
		w,
		bias,
	};
	ccv_nnc_tensor_t* outputs[] = {
		b
	};
	ccv_nnc_net_exec(net, hint, inputs, 3, outputs, 1);
	printf("dim %d %d, front %d, %d, back %d, %d\n", hint.stride.dim[0], hint.stride.dim[1], hint.border.front[0], hint.border.front[1], hint.border.back[0], hint.border.back[1]);
	return 0;
}
