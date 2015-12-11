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

TEST_CASE("convolutional network of 5x5 on 27x27 with uniform weights")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			1, 27, 27,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4, 27, 27,
		},
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			1, 5, 5, 4,
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
				1, 5, 5,
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
	for (i = 0; i < 5 * 5 * 4; i++)
		w->data.f32[i] = 1;
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_net_exec(net, hint, inlets, 3, outlets, 1);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | 4, 0, 0);
	int x, y;
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			for (i = 0; i < 4; i++)
			{
				if ((x == 0 && y == 0) || (x == 0 && y == 26) || (x == 26 && y == 0) || (x == 26 && y == 26))
					c->data.f32[(y * 27 + x) * 4 + i] = 9;
				else if ((x == 0 && y == 1) || (x == 0 && y == 25) || (x == 1 && y == 0) || (x == 1 && y == 26) || (x == 25 && y == 0) || (x == 25 && y == 26) || (x == 26 && y == 1) || (x == 26 && y == 25))
					c->data.f32[(y * 27 + x) * 4 + i] = 12;
				else if (x == 0 || y == 0 || x == 26 || y == 26)
					c->data.f32[(y * 27 + x) * 4 + i] = 15;
				else if ((x == 1 && y == 1) || (x == 1 && y == 25) || (x == 25 && y == 1) || (x == 25 && y == 25))
					c->data.f32[(y * 27 + x) * 4 + i] = 16;
				else if (x == 1 || y == 1 || x == 25 || y == 25)
					c->data.f32[(y * 27 + x) * 4 + i] = 20;
				else
					c->data.f32[(y * 27 + x) * 4 + i] = 25;
			}
	REQUIRE_MATRIX_EQ(b, c, "27x27 matrix should be exactly a matrix fill 25, with 9, 16 on the corner and 12, 15, 20 on the border");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_net_free(net);
}

TEST_CASE("convolutional network of 11x11 on 225x225 with non-uniform weights")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			1, 225, 225,
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
			1, 11, 11, 4,
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
				1, 11, 11,
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
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 11 * 11; i++)
			w->data.f32[x * 11 * 11 + i] = i + 1;
	for (i = 0; i < 225 * 225; i++)
		a->data.f32[i] = i + 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_net_exec(net, hint, inlets, 3, outlets, 1);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 55, CCV_32F | 4, 0, 0);
	float sum = 0;
	// first column
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 2) * (y * 225 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[i] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 225 + (x + 3) + 1);
	for (x = 1; x < 54; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[x * 4 + i] = sum + (x - 1) * 4 * (11 * 11 + 12) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 225 + (x + 215) + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[54 * 4 + i] = sum;
	// last column
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 2) * ((y + 215) * 225 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[55 * 54 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 225 + (x + 3) + 1);
	for (x = 1; x < 54; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(55 * 54 + x) * 4 + i] = sum + (x - 1) * 4 * (10 * 11 + 1) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 225 + (x + 215) + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[(55 * 54 + 54) * 4 + i] = sum;
	float border[] = {
		0, 0
	};
	for (y = 0; y < 11; y++)
		for (x = 0; x < 10; x++)
			border[0] += (y * 11 + x + 2) * ((y + 3) * 225 + x + 1);
	for (y = 0; y < 11; y++)
		for (x = 0; x < 10; x++)
			border[1] += (y * 11 + x + 1) * ((y + 3) * 225 + (x + 215) + 1);
	sum = 0;
	for (y = 0; y < 11; y++)
		for (x = 0; x < 11; x++)
			sum += (y * 11 + x + 1) * ((y + 3) * 225 + (x + 3) + 1);
	for (y = 1; y < 54; y++)
	{
		for (i = 0; i < 4; i++)
			c->data.f32[y * 55 * 4 + i] = border[0];
		for (x = 1; x < 54; x++)
			for (i = 0; i < 4; i++)
				c->data.f32[(y * 55 + x) * 4 + i] = sum + (x - 1) * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		for (i = 0; i < 4; i++)
			c->data.f32[(y * 55 + 54) * 4 + i] = border[1];
		sum += 225 * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		border[0] += 225 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (10 * 11 + 1 + 1) * 11 / 2);
		border[1] += 225 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (11 * 11 + 11) * 11 / 2);
	}
	// regularize the output so it is within the tolerance
	for (i = 0; i < 55 * 55 * 4; i++)
		c->data.f32[i] = c->data.f32[i] * 1e-7, b->data.f32[i] = b->data.f32[i] * 1e-7;
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_net_free(net);
}

TEST_CASE("convolutional network of 5x5 on 27x27 with non-uniform weights")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			1, 27, 27,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4, 27, 27,
		},
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			1, 5, 5, 4,
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
				1, 5, 5,
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
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 5 * 5; i++)
			w->data.f32[x * 5 * 5 + i] = i + 1;
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = i + 1;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_net_exec(net, hint, inlets, 3, outlets, 1);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | 4, 0, 0);
	// the first column
	float sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 2) * 5 + x + 3) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 2) * 5 + x + 2) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[4 + i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 5; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[x * 4 + i] = sum + (x - 2) * 36 * 15 / 2;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[25 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[26 * 4 + i] = sum;
	// the second column
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 3) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[27 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 2) * (y * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[28 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 5; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(27 + x) * 4 + i] = sum + (x - 2) * 31 * 20 / 2;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[52 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[53 * 4 + i] = sum;
	sum = 0;
	// the last 2nd column
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 3) * ((y + 23) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[27 * 25 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 2) * ((y + 23) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 25 + 1) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(27 * 25 + x) * 4 + i] = sum + (x - 2) * 21 * 20 / 2;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 25 + 25) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 25 + 26) * 4 + i] = sum;
	// the last column
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 3) * ((y + 24) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[27 * 26 * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 2) * ((y + 24) * 27 + x + 1);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 1) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 1);
	for (x = 2; x < 25; x++)
		for (i = 0; i < 4; i++)
			c->data.f32[(27 * 26 + x) * 4 + i] = sum + (x - 2) * 16 * 15 / 2;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 24);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 25) * 4 + i] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 25);
	for (i = 0; i < 4; i++)
		c->data.f32[(27 * 26 + 26) * 4 + i] = sum;
	float border[] = {
		0, 0, 0, 0
	};
	for (y = 0; y < 5; y++)
		for (x = 0; x < 3; x++)
			border[0] += (y * 5 + x + 3) * (y * 27 + x + 1);
	for (y = 0; y < 5; y++)
		for (x = 0; x < 4; x++)
			border[1] += (y * 5 + x + 2) * (y * 27 + x + 1);
	for (y = 0; y < 5; y++)
		for (x = 0; x < 4; x++)
			border[2] += (y * 5 + x + 1) * (y * 27 + x + 24);
	for (y = 0; y < 5; y++)
		for (x = 0; x < 3; x++)
			border[3] += (y * 5 + x + 1) * (y * 27 + x + 25);
	sum = 0;
	for (y = 0; y < 5; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * (y * 27 + x + 1);
	for (y = 2; y < 25; y++)
	{
		for (i = 0; i < 4; i++)
		{
			c->data.f32[y * 27 * 4 + i] = border[0] + (y - 2) * 27 * (3 + 4 + 5 + 8 + 9 + 10 + 13 + 14 + 15 + 18 + 19 + 20 + 23 + 24 + 25);
			c->data.f32[(y * 27 + 1) * 4 + i] = border[1] + (y - 2) * 27 * (2 + 3 + 4 + 5 + 7 + 8 + 9 + 10 + 12 + 13 + 14 + 15 + 17 + 18 + 19 + 20 + 22 + 23 + 24 + 25);
			for (x = 2; x < 25; x++)
				c->data.f32[(y * 27 + x) * 4 + i] = sum + ((y - 2) * 27 + x - 2) * 26 * 25 / 2;
			c->data.f32[(y * 27 + 25) * 4 + i] = border[2] + (y - 2) * 27 * (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9 + 11 + 12 + 13 + 14 + 16 + 17 + 18 + 19 + 21 + 22 + 23 + 24);
			c->data.f32[(y * 27 + 26) * 4 + i] = border[3] + (y - 2) * 27 * (1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13 + 16 + 17 + 18 + 21 + 22 + 23);
		}
	}
	REQUIRE_MATRIX_EQ(b, c, "27x27 matrix should be exactly the same");
	ccv_matrix_free(c);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_net_free(net);
}

#include "case_main.h"
