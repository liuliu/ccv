#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("convolutional network of 11x11 on 225x225")
{
	ccv_convnet_param_t params = {
		.type = CCV_CONVNET_CONVOLUTIONAL,
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

#include "case_main.h"
