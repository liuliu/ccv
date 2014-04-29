#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "3rdparty/dsfmt/dSFMT.h"

TEST_CASE("convolutional network of 11x11 on 225x225 with uniform weights")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 4,
				.strides = 4,
				.border = 1,
				.rows = 11,
				.cols = 11,
				.channels = 3,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(225, 225), &params, 1);
	int i, x, y;
	for (i = 0; i < 11 * 11 * 3 * 4; i++)
		convnet->layers[0].w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(225, 225, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 225 * 225 * 3; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	REQUIRE(b->rows == 55 && b->cols == 55, "11x11 convolves on 225x255 with strides 4 should produce 55x55 matrix");
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 55, CCV_32F | 4, 0, 0);
	for (y = 0; y < 55; y++)
		for (x = 0; x < 55; x++)
			for (i = 0; i < 4; i++)
			c->data.f32[(y * 55 + x) * 4 + i] = ((x == 0 && y == 0) || (x == 0 && y == 54) || (x == 54 && y == 0) || (x == 54 && y == 54)) ? 300 : ((x == 0 || y == 0 || x == 54 || y == 54) ? 330 : 363);
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly a matrix fill 363, with 300 on the corner and 330 on the border");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 5x5 on 27x27 with uniform weights")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 4,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 1,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(27, 27), &params, 1);
	int i, x, y;
	for (i = 0; i < 5 * 5 * 4; i++)
		convnet->layers->w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	REQUIRE(b->rows == 27 && b->cols == 27, "5x5 convolves on 27x27 with border 2 should produce 27x27 matrix");
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | 4, 0, 0);
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
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 11x11 on 225x225 with non-uniform weights")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 4,
				.strides = 4,
				.border = 1,
				.rows = 11,
				.cols = 11,
				.channels = 1,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(225, 225), &params, 1);
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 11 * 11; i++)
			convnet->layers[0].w[x * 11 * 11 + i] = i + 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(225, 225, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 225 * 225; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	REQUIRE(b->rows == 55 && b->cols == 55, "11x11 convolves on 225x255 with strides 4 should produce 55x55 matrix");
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
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 5x5 on 27x27 with non-uniform weights")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 4,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 1,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(27, 27), &params, 1);
	int i, x, y;
	for (x = 0; x < 4; x++)
		for (i = 0; i < 5 * 5; i++)
			convnet->layers->w[x * 5 * 5 + i] = i + 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	REQUIRE(b->rows == 27 && b->cols == 27, "5x5 convolves on 27x27 with border 2 should produce 27x27 matrix");
	ccv_matrix_free(a);
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
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 5x5x4 on 27x27x8 partitioned by 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 4,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.count = 8,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 4,
				.partition = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(27, 27), &params, 1);
	int i, k;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = i;
	for (i = 0; i < convnet->layers->net.convolutional.count; i++)
		convnet->layers->bias[i] = i + 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | 4, 0, 0);
	for (i = 0; i < 27 * 27 * 4; i++)
		a->data.f32[i] = 20 - i;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_convnet_layer_param_t partitioned_params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 2,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 4,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 2,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* partitioned_convnet = ccv_convnet_new(0, ccv_size(27, 27), &partitioned_params, 1);
	memcpy(partitioned_convnet->layers->w, convnet->layers->w, sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(partitioned_convnet->layers->bias, convnet->layers->bias, sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	ccv_dense_matrix_t* aa = ccv_dense_matrix_new(27, 27, CCV_32F | 2, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 2; k++)
			aa->data.f32[i * 2 + k] = a->data.f32[i * 4 + k];
	ccv_dense_matrix_t* bb = ccv_dense_matrix_new(27, 27, CCV_32F | 8, 0, 0);
	ccv_dense_matrix_t* cc = 0;
	ccv_convnet_encode(partitioned_convnet, &aa, &cc, 1);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 4; k++)
			bb->data.f32[i * 8 + k] = cc->data.f32[i * 4 + k];
	memcpy(partitioned_convnet->layers->w, convnet->layers->w + (convnet->layers->wnum / 2), sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(partitioned_convnet->layers->bias, convnet->layers->bias + (convnet->layers->net.convolutional.count / 2), sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 2; k++)
			aa->data.f32[i * 2 + k] = a->data.f32[i * 4 + 2 + k];
	ccv_convnet_encode(partitioned_convnet, &aa, &cc, 1);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 4; k++)
			bb->data.f32[i * 8 + 4 + k] = cc->data.f32[i * 4 + k];
	REQUIRE_MATRIX_EQ(b, bb, "27x27x8 matrix computed from convnet with partition and partitioned convnet should be exactly the same");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(aa);
	ccv_matrix_free(bb);
	ccv_matrix_free(cc);
	ccv_convnet_free(convnet);
	ccv_convnet_free(partitioned_convnet);
}

TEST_CASE("full connect network from 13x13x128 to 2048")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 128,
				.partition = 1,
			},
			.node = {
				.count = 13 * 13 * 128,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 2048,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(13, 13), &params, 1);
	int i;
	for (i = 0; i < 13 * 13 * 128 * 2048; i++)
		convnet->layers->w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(13, 13, CCV_32F | 128, 0, 0);
	for (i = 0; i < 13 * 13 * 128; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	REQUIRE(b->rows == 2048 && b->cols == 1, "full connect network output should be 2048 neurons");
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(2048, 1, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 2048; i++)
		c->data.f32[i] = 13 * 13 * 128;
	REQUIRE_MATRIX_EQ(b, c, "full connect network output should be exactly 13 * 13 * 128");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("maximum pool network of 55x55 with window of 3x3 and stride of 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(55, 55), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 113 + y * 110 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("maximum pool network of 57x57 with window of 3x3 and stride of 3")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 57,
				.cols = 57,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 3,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(57, 57), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(57, 57, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(19, 19, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 19; y++)
		for (x = 0; x < 19; x++)
			c->data.f32[y * 19 + x] = 117 + y * 171 + x * 3;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("maximum pool network of 54x54 with window of 2x2 and stride of 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 54,
				.cols = 54,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 2,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(54, 54), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(54, 54, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 56 + y * 108 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "max pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("average pool network of 55x55 with window of 3x3 and stride of 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(55, 55), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 57 + y * 110 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("average pool network of 57x57 with window of 3x3 and stride of 3")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 57,
				.cols = 57,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 3,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(57, 57), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(57, 57, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(19, 19, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 19; y++)
		for (x = 0; x < 19; x++)
			c->data.f32[y * 19 + x] = 59 + y * 171 + x * 3;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("average pool network of 54x54 with window of 2x2 and stride of 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 54,
				.cols = 54,
				.channels = 1,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 2,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(54, 54), &params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(54, 54, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			c->data.f32[y * 27 + x] = 28.5 + y * 108 + x * 2;
	REQUIRE_MATRIX_EQ(b, c, "average pool network output should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("local response normalization with partitioned by 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 10,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 3,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(27, 27), &params, 1);
	int i, k;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | 10, 0, 0);
	for (i = 0; i < 27 * 27 * 10; i++)
		a->data.f32[i] = i;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_convnet_layer_param_t partitioned_params = {
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 5,
				.partition = 1,
			},
		},
		.output = {
			.rnorm = {
				.size = 3,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	};
	ccv_convnet_t* partitioned_convnet = ccv_convnet_new(0, ccv_size(27, 27), &partitioned_params, 1);
	ccv_dense_matrix_t* aa = ccv_dense_matrix_new(27, 27, CCV_32F | 5, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 5; k++)
			aa->data.f32[i * 5 + k] = a->data.f32[i * 10 + k];
	ccv_dense_matrix_t* bb = ccv_dense_matrix_new(27, 27, CCV_32F | 10, 0, 0);
	ccv_dense_matrix_t* cc = 0;
	ccv_convnet_encode(partitioned_convnet, &aa, &cc, 1);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 5; k++)
			bb->data.f32[i * 10 + k] = cc->data.f32[i * 5 + k];
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 5; k++)
			aa->data.f32[i * 5 + k] = a->data.f32[i * 10 + 5 + k];
	ccv_convnet_encode(partitioned_convnet, &aa, &cc, 1);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 5; k++)
			bb->data.f32[i * 10 + 5 + k] = cc->data.f32[i * 5 + k];
	REQUIRE_MATRIX_EQ(b, bb, "27x27x10 matrix computed from convnet with partition and partitioned convnet should be exactly the same");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(aa);
	ccv_matrix_free(bb);
	ccv_matrix_free(cc);
	ccv_convnet_free(convnet);
	ccv_convnet_free(partitioned_convnet);
}

// we probably won't cover all static functions in this test, disable annoying warnings
#pragma GCC diagnostic ignored "-Wunused-function"
// so that we can test static functions, note that CASE_TESTS is defined in case.h, which will disable all extern functions
#include "ccv_convnet.c"

TEST_CASE("full connect network backward propagate")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.sigma = 0.0001,
		.input = {
			.matrix = {
				.rows = 3,
				.cols = 3,
				.channels = 64,
				.partition = 1,
			},
			.node = {
				.count = 3 * 3 * 64,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 10,
			},
		},
	};
	ccv_convnet_t *convnet = ccv_convnet_new(0, ccv_size(3, 3), &params, 1);
	int i, j;
	for (i = 0; i < 3 * 3 * 64 * 10; i++)
		convnet->layers[0].w[i] = 2;
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(3, 3, CCV_32F | 64, 0, 0);
	for (i = 0; i < 3 * 3 * 64; i++)
		x->data.f32[i] = 1;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 10 && y->cols == 1 && CCV_GET_CHANNEL(y->type) == 1, "y should be a 10-dimensional vector");
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(10, 1, CCV_32F | CCV_C1, 0, 0);
	loss->data.f32[0] = 18;
	for (i = 1; i < 10; i++)
		loss->data.f32[i] = -1;
	ccv_dense_matrix_t* b = 0;
	_ccv_convnet_full_connect_backward_propagate(convnet->layers, loss, y, x, &b, update_params->layers);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	ccv_matrix_free(loss);
	ccv_dense_matrix_t* db = ccv_dense_matrix_new(3, 3, CCV_32F | 64, 0, 0);
	for (i = 0; i < 3 * 3 * 64; i++)
		db->data.f32[i] = 18;
	REQUIRE_MATRIX_EQ(b, db, "propagated error doesn't match the expected value");
	ccv_matrix_free(db);
	ccv_matrix_free(b);
	float* dw = (float*)ccmalloc(sizeof(float) * 10 * 3 * 3 * 64);
	for (j = 0; j < 3 * 3 * 64; j++)
		dw[j] = 18;
	for (i = 1; i < 10; i++)
		for (j = 0; j < 3 * 3 * 64; j++)
			dw[i * 3 * 3 * 64 + j] = -1;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw, update_params->layers[0].w, 10 * 3 * 3 * 64, 1e-4, "weight gradient doesn't match the expected value");
	ccfree(dw);
	float* dbias = (float*)ccmalloc(sizeof(float) * 10);
	dbias[0] = 18;
	for (i = 1; i < 10; i++)
		dbias[i] = -1;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias, update_params->layers[0].bias, 10, 1e-4, "bias gradient doesn't match the expected value");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network backward propagate")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.0001,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.rows = 5,
				.cols = 5,
				.channels = 3,
				.border = 2,
				.strides = 1,
				.count = 32,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), &params, 1);
	int i, j, k;
	for (i = 0; i < 5 * 5 * 3 * 32; i++)
		convnet->layers[0].w[i] = 2;
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 31 * 31 * 3; i++)
		x->data.f32[i] = 1;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 31 && y->cols == 31 && CCV_GET_CHANNEL(y->type) == 32, "convnet should return a 31x31x32 matrix");
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(y->rows, y->cols, CCV_32F | CCV_GET_CHANNEL(y->type), 0, 0);
	for (i = 0; i < 31 * 31 * 32; i++)
		loss->data.f32[i] = 1;
	ccv_dense_matrix_t* d = 0;
	_ccv_convnet_convolutional_backward_propagate(convnet->layers, loss, y, x, &d, update_params->layers);
	ccv_matrix_free(loss);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	ccv_dense_matrix_t* dd = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 31; i++)
		for (j = 0; j < 31; j++)
			dd->data.f32[(i * 31 + j) * 3] =
			dd->data.f32[(i * 31 + j) * 3 + 1] =
			dd->data.f32[(i * 31 + j) * 3 + 2] = 32 * 2 * (5 + ccv_min(i - 2, 0) + ccv_min(28 - i, 0)) * (5 + ccv_min(j - 2, 0) + ccv_min(28 - j, 0));
	REQUIRE_MATRIX_EQ(d, dd, "propagated error doesn't match the expected value");
	ccv_matrix_free(d);
	ccv_matrix_free(dd);
	float* dw = (float*)ccmalloc(sizeof(float) * 5 * 5 * 3 * 32);
	for (k = 0; k < 32; k++)
		for (i = 0; i < 5; i++)
			for (j = 0; j < 5; j++)
				dw[k * 5 * 5 * 3 + (i * 5 + j) * 3] =
				dw[k * 5 * 5 * 3 + (i * 5 + j) * 3 + 1] =
				dw[k * 5 * 5 * 3 + (i * 5 + j) * 3 + 2] = (31 - abs(i - 2)) * (31 - abs(j - 2));
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw, update_params->layers[0].w, 5 * 5 * 3 * 32, 1e-4, "weight gradient doesn't match the expected value");
	ccfree(dw);
	float* dbias = (float*)ccmalloc(sizeof(float) * 32);
	for (i = 0; i < 32; i++)
		dbias[i] = 31 * 31;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias, update_params->layers[0].bias, 32, 1e-4, "bias gradient doesn't match the expected value");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network backward propagate with partitioned by 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.0001,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 4,
				.partition = 2,
			},
		},
		.output = {
			.convolutional = {
				.rows = 5,
				.cols = 5,
				.channels = 4,
				.border = 2,
				.strides = 1,
				.count = 8,
				.partition = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), &params, 1);
	int i, k;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = i * 1e-2;
	for (i = 0; i < convnet->layers->net.convolutional.count; i++)
		convnet->layers->bias[i] = i;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | 4, 0, 0);
	for (i = 0; i < 31 * 31 * 4; i++)
		a->data.f32[i] = 2000 - i;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(b->rows, b->cols, CCV_32F | CCV_GET_CHANNEL(b->type), 0, 0);
	for (i = 0; i < 31 * 31 * 8; i++)
		loss->data.f32[i] = 1;
	ccv_dense_matrix_t* d = 0;
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	_ccv_convnet_convolutional_backward_propagate(convnet->layers, loss, b, a, &d, update_params->layers);
	ccv_matrix_free(loss);
	ccv_convnet_layer_param_t partitioned_params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.0001,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 2,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.rows = 5,
				.cols = 5,
				.channels = 2,
				.border = 2,
				.strides = 1,
				.count = 4,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* partitioned_convnet = ccv_convnet_new(0, ccv_size(31, 31), &partitioned_params, 1);
	ccv_dense_matrix_t* aa = ccv_dense_matrix_new(31, 31, CCV_32F | 2, 0, 0);
	// first partition
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 2; k++)
			aa->data.f32[i * 2 + k] = a->data.f32[i * 4 + k];
	memcpy(partitioned_convnet->layers->w, convnet->layers->w, sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(partitioned_convnet->layers->bias, convnet->layers->bias, sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	ccv_dense_matrix_t* bb = 0;
	ccv_convnet_encode(partitioned_convnet, &aa, &bb, 1);
	ccv_dense_matrix_t* bbb = ccv_dense_matrix_new(31, 31, CCV_32F | 8, 0, 0);
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 4; k++)
			bbb->data.f32[i * 8 + k] = bb->data.f32[i * 4 + k];
	loss = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_GET_CHANNEL(bb->type), 0, 0);
	for (i = 0; i < 31 * 31 * 4; i++)
		loss->data.f32[i] = 1;
	ccv_dense_matrix_t* dd = 0;
	ccv_convnet_t* partitioned_update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(partitioned_update_params);
	_ccv_convnet_convolutional_backward_propagate(partitioned_convnet->layers, loss, bb, aa, &dd, partitioned_update_params->layers);
	ccv_dense_matrix_t* ddd = ccv_dense_matrix_new(31, 31, CCV_32F | 4, 0, 0);
	float* ww = (float*)ccmalloc(sizeof(float) * (convnet->layers->wnum + convnet->layers->net.convolutional.count));
	float* bbias = ww + convnet->layers->wnum;
	memcpy(ww, partitioned_update_params->layers->w, sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(bbias, partitioned_update_params->layers->bias, sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 2; k++)
			ddd->data.f32[i * 4 + k] = dd->data.f32[i * 2 + k];
	// second partition
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 2; k++)
			aa->data.f32[i * 2 + k] = a->data.f32[i * 4 + 2 + k];
	memcpy(partitioned_convnet->layers->w, convnet->layers->w + (convnet->layers->wnum / 2), sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(partitioned_convnet->layers->bias, convnet->layers->bias + (convnet->layers->net.convolutional.count / 2), sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	ccv_convnet_compact(partitioned_convnet); // because it is reused, we need to clear intermediate data
	ccv_convnet_encode(partitioned_convnet, &aa, &bb, 1);
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 4; k++)
			bbb->data.f32[i * 8 + 4 + k] = bb->data.f32[i * 4 + k];
	REQUIRE_MATRIX_EQ(b, bbb, "forward pass doesn't match the expected value");
	_ccv_convnet_update_zero(partitioned_update_params);
	_ccv_convnet_convolutional_backward_propagate(partitioned_convnet->layers, loss, bb, aa, &dd, partitioned_update_params->layers);
	memcpy(ww + (convnet->layers->wnum / 2), partitioned_update_params->layers->w, sizeof(float) * (convnet->layers->wnum / 2));
	memcpy(bbias + (convnet->layers->net.convolutional.count / 2), partitioned_update_params->layers->bias, sizeof(float) * (convnet->layers->net.convolutional.count / 2));
	for (i = 0; i < 31 * 31; i++)
		for (k = 0; k < 2; k++)
			ddd->data.f32[i * 4 + 2 + k] = dd->data.f32[i * 2 + k];
	REQUIRE_MATRIX_EQ(d, ddd, "propagated error doesn't match the expected value");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ww, update_params->layers[0].w, convnet->layers->wnum, 1e-4, "weight gradient doesn't match the expected value");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bbias, update_params->layers[0].bias, convnet->layers->net.convolutional.count, 1e-4, "bias gradient doesn't match the expected value");
	ccfree(ww);
	ccv_matrix_free(loss);
	ccv_matrix_free(ddd);
	ccv_matrix_free(dd);
	ccv_matrix_free(bbb);
	ccv_matrix_free(bb);
	ccv_matrix_free(aa);
	ccv_matrix_free(d);
	ccv_matrix_free(b);
	ccv_matrix_free(a);
	ccv_convnet_free(convnet);
	ccv_convnet_free(update_params);
	ccv_convnet_free(partitioned_convnet);
	ccv_convnet_free(partitioned_update_params);
}

TEST_CASE("local response normalization backward propagate with partitioned by 2")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 6,
				.partition = 2,
			},
		},
		.output = {
			.rnorm = {
				.size = 3,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(27, 27), &params, 1);
	int i, k;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | 6, 0, 0);
	for (i = 0; i < 27 * 27 * 6; i++)
		a->data.f32[i] = i;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, &a, &b, 1);
	ccv_dense_matrix_t* d = 0;
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(27, 27, CCV_32F | 6, 0, 0);
	for (i = 0; i < 27 * 27 * 6; i++)
		loss->data.f32[i] = 1;
	_ccv_convnet_rnorm_backward_propagate(convnet->layers, loss, b, a, convnet->denoms[0], &d);
	ccv_convnet_layer_param_t partitioned_params = {
		.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.rnorm = {
				.size = 3,
				.kappa = 2,
				.alpha = 1e-4,
				.beta = 0.75,
			},
		},
	};
	ccv_convnet_t* partitioned_convnet = ccv_convnet_new(0, ccv_size(27, 27), &partitioned_params, 1);
	ccv_dense_matrix_t* aa = ccv_dense_matrix_new(27, 27, CCV_32F | 3, 0, 0);
	// first partition
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 3; k++)
			aa->data.f32[i * 3 + k] = a->data.f32[i * 6 + k];
	ccv_dense_matrix_t* bb = 0;
	ccv_convnet_encode(partitioned_convnet, &aa, &bb, 1);
	ccv_matrix_free(loss);
	loss = ccv_dense_matrix_new(27, 27, CCV_32F | 3, 0, 0);
	for (i = 0; i < 27 * 27 * 3; i++)
		loss->data.f32[i] = 1;
	ccv_dense_matrix_t* dd = 0;
	_ccv_convnet_rnorm_backward_propagate(partitioned_convnet->layers, loss, bb, aa, partitioned_convnet->denoms[0], &dd);
	ccv_dense_matrix_t* ddd = ccv_dense_matrix_new(27, 27, CCV_32F | 6, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 3; k++)
			ddd->data.f32[i * 6 + k] = dd->data.f32[i * 3 + k];
	// second partition
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 3; k++)
			aa->data.f32[i * 3 + k] = a->data.f32[i * 6 + 3 + k];
	ccv_convnet_encode(partitioned_convnet, &aa, &bb, 1);
	_ccv_convnet_rnorm_backward_propagate(partitioned_convnet->layers, loss, bb, aa, partitioned_convnet->denoms[0], &dd);
	for (i = 0; i < 27 * 27; i++)
		for (k = 0; k < 3; k++)
			ddd->data.f32[i * 6 + 3 + k] = dd->data.f32[i * 3 + k];
	REQUIRE_MATRIX_EQ(d, ddd, "27x27x6 error local response normalization backward propagated from convnet with partition and partitioned convnet should be exactly the same");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(d);
	ccv_matrix_free(aa);
	ccv_matrix_free(bb);
	ccv_matrix_free(dd);
	ccv_matrix_free(ddd);
	ccv_matrix_free(loss);
	ccv_convnet_free(convnet);
	ccv_convnet_free(partitioned_convnet);
}

// five-stencil constants
static float fs[4] = { 1, -8, 8, -1 };
static float fsh[4] = { -2, -1, 1, 2 };

static float dsfmt_genrand_gaussian(dsfmt_t* dsfmt, float sigma)
{
	double rand1 = dsfmt_genrand_open_close(dsfmt);
	rand1 = -2 * log(rand1);
	double rand2 = dsfmt_genrand_open_close(dsfmt) * CCV_PI * 2;
	return (float)(sqrt(sigma * rand1) * cos(rand2));
}

TEST_CASE("numerical gradient versus analytical gradient for full connect network")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 3,
				.cols = 3,
				.channels = 8,
				.partition = 1,
			},
			.node = {
				.count = 3 * 3 * 8,
			},
		},
		.output = {
			.full_connect = {
				.relu = 0,
				.count = 10,
			},
		},
	};
	ccv_convnet_t *convnet = ccv_convnet_new(0, ccv_size(3, 3), &params, 1);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i, j, k;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = dsfmt_genrand_gaussian(&dsfmt, 0.01);
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(3, 3, CCV_32F | 8, 0, 0);
	for (i = 0; i < 3 * 3 * 8; i++)
		x->data.f32[i] = i;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 10 && y->cols == 1 && CCV_GET_CHANNEL(y->type) == 1, "y should be a 10-dimensional vector");
	_ccv_convnet_compute_softmax(y, &y, 0);
	ccv_dense_matrix_t* dloss = ccv_dense_matrix_new(10, 1, CCV_32F | CCV_C1, 0, 0);;
	for (i = 0; i < 10; i++)
		dloss->data.f32[i] = y->data.f32[i] - (i == 2);
	float* dw = (float*)ccmalloc(sizeof(float) * 3 * 3 * 8 * 10);
	static const float eps = 0.0001;
	for (i = 0; i < 10; i++)
		for (j = 0; j < 3 * 3 * 8; j++)
		{
			dw[j + i * 3 * 3 * 8] = 0;
			for (k = 0; k < 4; k++)
			{
				float w = convnet->layers->w[j + i * 3 * 3 * 8];
				convnet->layers->w[j + i * 3 * 3 * 8] += fsh[k] * eps;
				ccv_dense_matrix_t* z = 0;
				ccv_convnet_encode(convnet, &x, &z, 1);
				_ccv_convnet_compute_softmax(z, &z, 0);
				dw[j + i * 3 * 3 * 8] += -logf(z->data.f32[2]) * fs[k];
				ccv_matrix_free(z);
				convnet->layers->w[j + i * 3 * 3 * 8] = w;
			}
			dw[j + i * 3 * 3 * 8] *= 1.0 / (12 * eps);
		}
	float* dbias = (float*)ccmalloc(sizeof(float) * 10);
	for (i = 0; i < 10; i++)
	{
		dbias[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float bias = convnet->layers->bias[i];
			convnet->layers->bias[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dbias[i] += -logf(z->data.f32[2]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->bias[i] = bias;
		}
		dbias[i] *= 1.0 / (12 * eps);
	}
	ccv_dense_matrix_t* b = 0;
	_ccv_convnet_full_connect_backward_propagate(convnet->layers, dloss, y, x, &b, update_params->layers);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	ccv_matrix_free(dloss);
	ccv_matrix_free(b);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dw, update_params->layers[0].w, 3 * 3 * 8 * 10, 30, 2e-1, "weight gradient from analytical method doesn't match the one from numerical method");
	ccfree(dw);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dbias, update_params->layers[0].bias, 10, 30, 2e-1, "bias gradient from analytical method doesn't match the one from numerical method");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("numerical gradient versus analytical gradient for convolutional network")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.0001,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 3,
				.partition = 1,
			},
		},
		.output = {
			.convolutional = {
				.rows = 5,
				.cols = 5,
				.channels = 3,
				.border = 2,
				.strides = 1,
				.count = 4,
				.partition = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), &params, 1);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i, k;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = dsfmt_genrand_gaussian(&dsfmt, 0.0001);
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 31 * 31 * 3; i++)
		x->data.f32[i] = i;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 31 && y->cols == 31 && CCV_GET_CHANNEL(y->type) == 4, "convnet should return a 31x31x4 matrix");
	ccv_dense_matrix_t* softmax = 0;
	_ccv_convnet_compute_softmax(y, &softmax, 0);
	ccv_dense_matrix_t* dloss = ccv_dense_matrix_new(y->rows, y->cols, CCV_32F | CCV_GET_CHANNEL(y->type), 0, 0);
	for (i = 0; i < 31 * 31 * 4; i++)
		dloss->data.f32[i] = softmax->data.f32[i] - (i == 24);
	static const float eps = 0.000005;
	float* dw = (float*)ccmalloc(sizeof(float) * 5 * 5 * 3 * 4); 
	for (i = 0; i < 5 * 5 * 3 * 4; i++)
	{
		dw[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float w = convnet->layers->w[i];
			convnet->layers->w[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dw[i] += -logf(z->data.f32[24]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->w[i] = w;
		}
		dw[i] *= 1.0 / (12 * eps);
	}
	float* dbias = (float*)ccmalloc(sizeof(float) * 4);
	for (i = 0; i < 4; i++)
	{
		dbias[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float bias = convnet->layers->bias[i];
			convnet->layers->bias[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dbias[i] += -logf(z->data.f32[24]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->bias[i] = bias;
		}
		dbias[i] *= 1.0 / (12 * eps);
	}
	ccv_dense_matrix_t* d = 0;
	_ccv_convnet_convolutional_backward_propagate(convnet->layers, dloss, y, x, &d, update_params->layers);
	ccv_matrix_free(softmax);
	ccv_matrix_free(dloss);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	ccv_matrix_free(d);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dw, update_params->layers[0].w, 5 * 5 * 3 * 4, 30, 2e-1, "weight gradient from analytical method doesn't match the one from numerical method");
	ccfree(dw);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dbias, update_params->layers[0].bias, 4, 30, 2e-1, "bias gradient from analytical method doesn't match the one from numerical method");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("numerical gradient versus analytical gradient for full connect network over convolutional network")
{
	ccv_convnet_layer_param_t params[] = {
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.001,
			.input = {
				.matrix = {
					.rows = 5,
					.cols = 5,
					.channels = 2,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.rows = 3,
					.cols = 3,
					.channels = 2,
					.border = 1,
					.strides = 1,
					.count = 4,
					.partition = 1,
				},
			},
		},
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 5,
					.cols = 5,
					.channels = 4,
					.partition = 1,
				},
				.node = {
					.count = 5 * 5 * 4,
				},
			},
			.output = {
				.full_connect = {
					.relu = 0,
					.count = 10,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(5, 5), params, 2);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i, k;
	for (i = 0; i < convnet->layers[0].wnum; i++)
		convnet->layers[0].w[i] = dsfmt_genrand_gaussian(&dsfmt, 0.001);
	for (i = 0; i < convnet->layers[1].wnum; i++)
		convnet->layers[1].w[i] = dsfmt_genrand_gaussian(&dsfmt, 0.01);
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(5, 5, CCV_32F | CCV_C2, 0, 0);
	for (i = 0; i < 5 * 5 * 2; i++)
		x->data.f32[i] = 0.2;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 10 && y->cols == 1 && CCV_GET_CHANNEL(y->type) == 1, "y should be a 10-dimensional vector");
	_ccv_convnet_compute_softmax(y, &y, 0);
	ccv_dense_matrix_t* dloss = ccv_dense_matrix_new(10, 1, CCV_32F | CCV_C1, 0, 0);;
	for (i = 0; i < 10; i++)
		dloss->data.f32[i] = y->data.f32[i] - (i == 2);
	_ccv_convnet_propagate_loss(convnet, x, dloss, update_params);
	ccv_matrix_free(dloss);
	static const float eps = 0.0001;
	float* dw = (float*)ccmalloc(sizeof(float) * 3 * 3 * 2 * 4); 
	for (i = 0; i < 3 * 3 * 2 * 4; i++)
	{
		dw[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float w = convnet->layers->w[i];
			convnet->layers->w[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dw[i] += -logf(z->data.f32[2]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->w[i] = w;
		}
		dw[i] *= 1.0 / (12 * eps);
	}
	float* dbias = (float*)ccmalloc(sizeof(float) * 4);
	for (i = 0; i < 4; i++)
	{
		dbias[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float bias = convnet->layers->bias[i];
			convnet->layers->bias[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dbias[i] += -logf(z->data.f32[2]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->bias[i] = bias;
		}
		dbias[i] *= 1.0 / (12 * eps);
	}
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dw, update_params->layers[0].w, 3 * 3 * 2 * 4, 30, 2e-1, "weight gradient from analytical method doesn't match the one from numerical method");
	ccfree(dw);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dbias, update_params->layers[0].bias, 4, 30, 2e-1, "bias gradient from analytical method doesn't match the one from numerical method");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("numerical gradient versus analytical gradient for local response normalization over convolutional network")
{
	ccv_convnet_layer_param_t params[] = {
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.001,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 2,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 2,
					.border = 2,
					.strides = 1,
					.count = 4,
					.partition = 1,
				},
			},
		},
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 4,
					.partition = 1,
				},
			},
			.output = {
				.rnorm = {
					.size = 3,
					.kappa = 2,
					.alpha = 0.00005,
					.beta = 0.75,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), params, 2);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i, k;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = dsfmt_genrand_gaussian(&dsfmt, 0.001);
	ccv_convnet_t* update_params = _ccv_convnet_update_new(convnet);
	_ccv_convnet_update_zero(update_params);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C2, 0, 0);
	for (i = 0; i < 31 * 31 * 2; i++)
		x->data.f32[i] = i;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	REQUIRE(y->rows == 31 && y->cols == 31 && CCV_GET_CHANNEL(y->type) == 4, "convnet should return a 31x31x4 matrix");
	ccv_dense_matrix_t* softmax = 0;
	_ccv_convnet_compute_softmax(y, &softmax, 0);
	ccv_dense_matrix_t* dloss = ccv_dense_matrix_new(y->rows, y->cols, CCV_32F | CCV_GET_CHANNEL(y->type), 0, 0);
	for (i = 0; i < 31 * 31 * 4; i++)
		dloss->data.f32[i] = softmax->data.f32[i] - (i == 24);
	ccv_dense_matrix_t* d = 0;
	_ccv_convnet_rnorm_backward_propagate(convnet->layers + 1, dloss, y, convnet->acts[0], convnet->denoms[1], update_params->acts);
	_ccv_convnet_convolutional_backward_propagate(convnet->layers, update_params->acts[0], convnet->acts[0], x, &d, update_params->layers);
	static const float eps = 0.000001;
	float* dw = (float*)ccmalloc(sizeof(float) * 5 * 5 * 2 * 4); 
	for (i = 0; i < 5 * 5 * 2 * 4; i++)
	{
		dw[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float w = convnet->layers->w[i];
			convnet->layers->w[i] += fsh[k] * eps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dw[i] += -logf(z->data.f32[24]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->w[i] = w;
		}
		dw[i] *= 1.0 / (12 * eps);
	}
	float* dbias = (float*)ccmalloc(sizeof(float) * 4);
	static const float beps = 0.0001;
	for (i = 0; i < 4; i++)
	{
		dbias[i] = 0;
		for (k = 0; k < 4; k++)
		{
			float bias = convnet->layers->bias[i];
			convnet->layers->bias[i] += fsh[k] * beps;
			ccv_dense_matrix_t* z = 0;
			ccv_convnet_compact(convnet);
			ccv_convnet_encode(convnet, &x, &z, 1);
			_ccv_convnet_compute_softmax(z, &z, 0);
			dbias[i] += -logf(z->data.f32[24]) * fs[k];
			ccv_matrix_free(z);
			convnet->layers->bias[i] = bias;
		}
		dbias[i] *= 1.0 / (12 * beps);
	}
	ccv_matrix_free(softmax);
	ccv_matrix_free(dloss);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
	ccv_matrix_free(d);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dw, update_params->layers[0].w, 5 * 5 * 2 * 4, 30, 2e-1, "weight gradient from analytical method doesn't match the one from numerical method");
	ccfree(dw);
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dbias, update_params->layers[0].bias, 4, 30, 2e-1, "bias gradient from analytical method doesn't match the one from numerical method");
	ccfree(dbias);
	ccv_convnet_free(update_params);
	ccv_convnet_free(convnet);
}

TEST_CASE("max pool network backward propagate")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 2,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), &params, 1);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C2, 0, 0);
	int i, j, k;
	for (i = 0; i < 31 * 31 * 2; i++)
		x->data.f32[i] = i;
	ccv_dense_matrix_t* y = 0;
	ccv_convnet_encode(convnet, &x, &y, 1);
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(15, 15, CCV_32F | CCV_C2, 0, 0);
	for (i = 0; i < 15 * 15 * 2; i++)
		loss->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	_ccv_convnet_max_pool_backward_propagate(convnet->layers, loss, y, x, &b);
	ccv_matrix_free(loss);
	ccv_matrix_free(x);
	ccv_matrix_free(y);
	ccv_dense_matrix_t* db = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C2, 0, 0);
	ccv_zero(db);
	for (i = 0; i < 15; i++)
		for (j = 0; j < 15; j++)
			for (k = 0; k < 2; k++)
				db->data.f32[(j * 2 + 2 + (i * 2 + 2) * 31) * 2 + k] = (i * 15 + j) * 2 + 1 + k;
	REQUIRE_MATRIX_EQ(b, db, "propagated error doesn't match the expected value");
	ccv_matrix_free(db);
	ccv_matrix_free(b);
	ccv_convnet_free(convnet);
}

TEST_CASE("average pool network backward propagate")
{
	ccv_convnet_layer_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.input = {
			.matrix = {
				.rows = 31,
				.cols = 31,
				.channels = 2,
				.partition = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
				.border = 0,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(0, ccv_size(31, 31), &params, 1);
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C2, 0, 0);
	int i, j, k;
	for (i = 0; i < 31 * 31 * 2; i++)
		x->data.f32[i] = i;
	ccv_dense_matrix_t* loss = ccv_dense_matrix_new(15, 15, CCV_32F | CCV_C2, 0, 0);
	for (i = 0; i < 15 * 15 * 2; i++)
		loss->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	_ccv_convnet_average_pool_backward_propagate(convnet->layers, loss, x, &b);
	ccv_matrix_free(x);
	ccv_matrix_free(loss);
	ccv_dense_matrix_t* db = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C2, 0, 0);
	float inv_size = 1.0 / (3 * 3);
	for (i = 0; i < 31; i++)
		for (j = 0; j < 31; j++)
			for (k = 0; k < 2; k++)
			{
				int x, y;
				db->data.f32[(i * 31 + j) * 2 + k] = 0;
				for (y = (i - 1) / 2; y <= i / 2; y++)
					for (x = (j - 1) / 2; x <= j / 2; x++)
						if (x >= 0 && x < 15 && y >= 0 && y < 15)
							db->data.f32[(i * 31 + j) * 2 + k] += ((y * 15 + x) * 2 + k + 1) * inv_size;
			}
	REQUIRE_MATRIX_EQ(b, db, "propagated error doesn't match the expected value");
	ccv_matrix_free(db);
	ccv_matrix_free(b);
	ccv_convnet_free(convnet);
}

#include "case_main.h"
