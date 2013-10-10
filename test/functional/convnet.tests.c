#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("convolutional network of 11x11 on 225x225 with uniform weights")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 3,
			},
		},
		.output = {
			.convolutional = {
				.count = 1,
				.strides = 4,
				.border = 1,
				.rows = 11,
				.cols = 11,
				.channels = 3,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	int i, x, y;
	for (i = 0; i < 11 * 11 * 3; i++)
		convnet->layers[0].w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(225, 225, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 225 * 225 * 3; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
	ccv_matrix_free(a);
	REQUIRE(b->rows == 55 && b->cols == 55, "11x11 convolves on 225x255 with strides 4 should produce 55x55 matrix");
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 55; y++)
		for (x = 0; x < 55; x++)
			c->data.f32[y * 55 + x] = ((x == 0 && y == 0) || (x == 0 && y == 54) || (x == 54 && y == 0) || (x == 54 && y == 54)) ? 300 : ((x == 0 || y == 0 || x == 54 || y == 54) ? 330 : 363);
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly a matrix fill 363, with 300 on the corner and 330 on the border");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 5x5 on 27x27 with uniform weights")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 1,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	int i, x, y;
	for (i = 0; i < 5 * 5; i++)
		convnet->layers->w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
	REQUIRE(b->rows == 27 && b->cols == 27, "5x5 convolves on 27x27 with border 2 should produce 27x27 matrix");
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (y = 0; y < 27; y++)
		for (x = 0; x < 27; x++)
			if ((x == 0 && y == 0) || (x == 0 && y == 26) || (x == 26 && y == 0) || (x == 26 && y == 26))
				c->data.f32[y * 27 + x] = 9;
			else if ((x == 0 && y == 1) || (x == 0 && y == 25) || (x == 1 && y == 0) || (x == 1 && y == 26) || (x == 25 && y == 0) || (x == 25 && y == 26) || (x == 26 && y == 1) || (x == 26 && y == 25))
				c->data.f32[y * 27 + x] = 12;
			else if (x == 0 || y == 0 || x == 26 || y == 26)
				c->data.f32[y * 27 + x] = 15;
			else if ((x == 1 && y == 1) || (x == 1 && y == 25) || (x == 25 && y == 1) || (x == 25 && y == 25))
				c->data.f32[y * 27 + x] = 16;
			else if (x == 1 || y == 1 || x == 25 || y == 25)
				c->data.f32[y * 27 + x] = 20;
			else
				c->data.f32[y * 27 + x] = 25;
	REQUIRE_MATRIX_EQ(b, c, "27x27 matrix should be exactly a matrix fill 25, with 9, 16 on the corner and 12, 15, 20 on the border");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 11x11 on 225x225 with non-uniform weights")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 225,
				.cols = 225,
				.channels = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 1,
				.strides = 4,
				.border = 1,
				.rows = 11,
				.cols = 11,
				.channels = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	int i, x, y;
	for (i = 0; i < 11 * 11; i++)
		convnet->layers[0].w[i] = i + 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(225, 225, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 225 * 225; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
	ccv_matrix_free(a);
	REQUIRE(b->rows == 55 && b->cols == 55, "11x11 convolves on 225x255 with strides 4 should produce 55x55 matrix");
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	float sum = 0;
	// first column
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 2) * (y * 225 + x + 1);
	c->data.f32[0] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 225 + (x + 3) + 1);
	for (x = 1; x < 54; x++)
		c->data.f32[x] = sum + (x - 1) * 4 * (11 * 11 + 12) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += ((y + 1) * 11 + x + 1) * (y * 225 + (x + 215) + 1);
	c->data.f32[54] = sum;
	// last column
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 2) * ((y + 215) * 225 + x + 1);
	c->data.f32[55 * 54] = sum;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 11; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 225 + (x + 3) + 1);
	for (x = 1; x < 54; x++)
		c->data.f32[55 * 54 + x] = sum + (x - 1) * 4 * (10 * 11 + 1) * 11 * 10 / 2;
	sum = 0;
	for (y = 0; y < 10; y++)
		for (x = 0; x < 10; x++)
			sum += (y * 11 + x + 1) * ((y + 215) * 225 + (x + 215) + 1);
	c->data.f32[55 * 54 + 54] = sum;
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
		c->data.f32[y * 55] = border[0];
		for (x = 1; x < 54; x++)
			c->data.f32[y * 55 + x] = sum + (x - 1) * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		c->data.f32[y * 55 + 54] = border[1];
		sum += 225 * 4 * (11 * 11 + 1) * 11 * 11 / 2;
		border[0] += 225 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (10 * 11 + 1 + 1) * 11 / 2);
		border[1] += 225 * 4 * ((11 * 11 + 1) * 11 * 11 / 2 - (11 * 11 + 11) * 11 / 2);
	}
	// regularize the output so it is within the tolerance
	for (i = 0; i < 55 * 55; i++)
		c->data.f32[i] = c->data.f32[i] * 1e-7, b->data.f32[i] = b->data.f32[i] * 1e-7;
	REQUIRE_MATRIX_EQ(b, c, "55x55 matrix should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("convolutional network of 5x5 on 27x27 with non-uniform weights")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 27,
				.cols = 27,
				.channels = 1,
			},
		},
		.output = {
			.convolutional = {
				.count = 1,
				.strides = 1,
				.border = 2,
				.rows = 5,
				.cols = 5,
				.channels = 1,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	int i, x, y;
	for (i = 0; i < 5 * 5; i++)
		convnet->layers->w[i] = i + 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 27 * 27; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
	REQUIRE(b->rows == 27 && b->cols == 27, "5x5 convolves on 27x27 with border 2 should produce 27x27 matrix");
	ccv_matrix_free(a);
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(27, 27, CCV_32F | CCV_C1, 0, 0);
	// the first column
	float sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 2) * 5 + x + 3) * (y * 27 + x + 1);
	c->data.f32[0] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 2) * 5 + x + 2) * (y * 27 + x + 1);
	c->data.f32[1] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 5; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 1);
	for (x = 2; x < 25; x++)
		c->data.f32[x] = sum + (x - 2) * 36 * 15 / 2;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 24);
	c->data.f32[25] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 2) * 5 + x + 1) * (y * 27 + x + 25);
	c->data.f32[26] = sum;
	// the second column
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 3) * (y * 27 + x + 1);
	c->data.f32[27] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 2) * (y * 27 + x + 1);
	c->data.f32[28] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 5; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 1);
	for (x = 2; x < 25; x++)
		c->data.f32[27 + x] = sum + (x - 2) * 31 * 20 / 2;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 24);
	c->data.f32[52] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += ((y + 1) * 5 + x + 1) * (y * 27 + x + 25);
	c->data.f32[53] = sum;
	sum = 0;
	// the last 2nd column
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 3) * ((y + 23) * 27 + x + 1);
	c->data.f32[27 * 25] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 2) * ((y + 23) * 27 + x + 1);
	c->data.f32[27 * 25 + 1] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 1);
	for (x = 2; x < 25; x++)
		c->data.f32[27 * 25 + x] = sum + (x - 2) * 21 * 20 / 2;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 24);
	c->data.f32[27 * 25 + 25] = sum;
	sum = 0;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 1) * ((y + 23) * 27 + x + 25);
	c->data.f32[27 * 25 + 26] = sum;
	// the last column
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 3) * ((y + 24) * 27 + x + 1);
	c->data.f32[27 * 26] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 2) * ((y + 24) * 27 + x + 1);
	c->data.f32[27 * 26 + 1] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 5; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 1);
	for (x = 2; x < 25; x++)
		c->data.f32[27 * 26 + x] = sum + (x - 2) * 16 * 15 / 2;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 4; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 24);
	c->data.f32[27 * 26 + 25] = sum;
	sum = 0;
	for (y = 0; y < 3; y++)
		for (x = 0; x < 3; x++)
			sum += (y * 5 + x + 1) * ((y + 24) * 27 + x + 25);
	c->data.f32[27 * 26 + 26] = sum;
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
		c->data.f32[y * 27] = border[0] + (y - 2) * 27 * (3 + 4 + 5 + 8 + 9 + 10 + 13 + 14 + 15 + 18 + 19 + 20 + 23 + 24 + 25);
		c->data.f32[y * 27 + 1] = border[1] + (y - 2) * 27 * (2 + 3 + 4 + 5 + 7 + 8 + 9 + 10 + 12 + 13 + 14 + 15 + 17 + 18 + 19 + 20 + 22 + 23 + 24 + 25);
		for (x = 2; x < 25; x++)
			c->data.f32[y * 27 + x] = sum + ((y - 2) * 27 + x - 2) * 26 * 25 / 2;
		c->data.f32[y * 27 + 25] = border[2] + (y - 2) * 27 * (1 + 2 + 3 + 4 + 6 + 7 + 8 + 9 + 11 + 12 + 13 + 14 + 16 + 17 + 18 + 19 + 21 + 22 + 23 + 24);
		c->data.f32[y * 27 + 26] = border[3] + (y - 2) * 27 * (1 + 2 + 3 + 6 + 7 + 8 + 11 + 12 + 13 + 16 + 17 + 18 + 21 + 22 + 23);
	}
	REQUIRE_MATRIX_EQ(b, c, "27x27 matrix should be exactly the same");
	ccv_matrix_free(b);
	ccv_matrix_free(c);
	ccv_convnet_free(convnet);
}

TEST_CASE("full connect network from 13x13x128 to 2048")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_FULL_CONNECT,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 13,
				.cols = 13,
				.channels = 128,
			},
			.node = {
				.count = 13 * 13 * 128,
			},
		},
		.output = {
			.full_connect = {
				.count = 2048,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	int i;
	for (i = 0; i < 13 * 13 * 128 * 2048; i++)
		convnet->layers->w[i] = 1;
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(13, 13, CCV_32F | 128, 0, 0);
	for (i = 0; i < 13 * 13 * 128; i++)
		a->data.f32[i] = 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 57,
				.cols = 57,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 3,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(57, 57, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_MAX_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 54,
				.cols = 54,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 2,
				.strides = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(54, 54, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 55,
				.cols = 55,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(55, 55, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 55 * 55; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 57,
				.cols = 57,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 3,
				.strides = 3,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(57, 57, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 57 * 57; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_AVERAGE_POOL,
		.bias = 0,
		.sigma = 0.01,
		.input = {
			.matrix = {
				.rows = 54,
				.cols = 54,
				.channels = 1,
			},
		},
		.output = {
			.pool = {
				.size = 2,
				.strides = 2,
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(&params, 1);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(54, 54, CCV_32F | CCV_C1, 0, 0);
	int i, x, y;
	for (i = 0; i < 54 * 54; i++)
		a->data.f32[i] = i + 1;
	ccv_dense_matrix_t* b = 0;
	ccv_convnet_encode(convnet, a, &b, 0);
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

#include "case_main.h"
