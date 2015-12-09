#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "nnc/ccv_nnc.h"
#include "3rdparty/dsfmt/dSFMT.h"

TEST_CASE("convolutional network of 11x11 on 225x225 with uniform weights")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 225, 225,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4, 55, 55,
		},
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 11, 11, 4,
		},
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4,
		},
	};
	ccv_nnc_net_param_t net_params = {
		.size = {
			.dim = {
				3, 11, 11,
			},
		},
		.convolutional = {
			.count = 4,
		},
	};
	ccv_nnc_net_hint_t hint = ccv_nnc_net_hint_guess(net_params, &a_params, 1, &b_params, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_net_t* net = ccv_nnc_net_new(0, CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, net_params, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* inlets[] = {
		a,
		w,
		bias,
	};
	ccv_nnc_tensor_t* outlets[] = {
		b,
	};
	// configure the inlets.
	int i;
	for (i = 0; i < 11 * 11 * 3 * 4; i++)
		w->data.f32[i] = 1;
	for (i = 0; i < 225 * 225 * 3; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_net_exec(net, hint, inlets, 3, outlets, 1);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 55, CCV_32F | 4, 0, 0);
	int x, y;
	for (y = 0; y < 55; y++)
		for (x = 0; x < 55; x++)
			for (i = 0; i < 4; i++)
			c->data.f32[(y * 55 + x) * 4 + i] = ((x == 0 && y == 0) || (x == 0 && y == 54) || (x == 54 && y == 0) || (x == 54 && y == 54)) ? 300 : ((x == 0 || y == 0 || x == 54 || y == 54) ? 330 : 363);
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly a matrix fill 363, with 300 on the corner and 330 on the border");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_net_free(net);
}

#include "case_main.h"
