#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("zero out a tensor")
{
	ccv_nnc_tensor_param_t params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = {
			10, 20, 30, 4, 5, 6,
		},
	};
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, params, 0);
	int i;
	for (i = 0; i < 10 * 20 * 30 * 4 * 5 * 6; i++)
		tensor->data.f32[i] = 1;
	ccv_nnc_tensor_zero(tensor);
	for (i = 0; i < 10 * 20 * 30 * 4 * 5 * 6; i++)
		REQUIRE_EQ(0, tensor->data.f32[i], "should be zero'ed at %d", i);
	ccv_nnc_tensor_free(tensor);
}

TEST_CASE("zero out a tensor view")
{
	ccv_nnc_tensor_param_t params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = {
			10, 20, 30, 4, 5, 6,
		},
	};
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_new(0, params, 0);
	int c;
	for (c = 0; c < 10 * 20 * 30 * 4 * 5 * 6; c++)
		a_tensor->data.f32[c] = 1;
	int ofs[CCV_NNC_MAX_DIM_ALLOC] = {
		1, 2, 5, 1, 1, 1,
	};
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {
		8, 12, 15, 2, 3, 4,
	};
	ccv_nnc_tensor_view_t a_tensor_view = ccv_nnc_tensor_view(a_tensor, dim, ofs, a_tensor->info.dim);
	ccv_nnc_tensor_zero(&a_tensor_view);
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_new(0, params, 0);
	for (c = 0; c < 10 * 20 * 30 * 4 * 5 * 6; c++)
		b_tensor->data.f32[c] = 1;
	ccv_nnc_tensor_view_t b_tensor_view = ccv_nnc_tensor_view(b_tensor, dim, ofs, b_tensor->info.dim);
	int i[6];
	float* tvp[6];
	tvp[5] = b_tensor_view.data.f32;
	for (i[5] = 0; i[5] < b_tensor_view.info.dim[0]; i[5]++)
	{
		tvp[4] = tvp[5];
		for (i[4] = 0; i[4] < b_tensor_view.info.dim[1]; i[4]++)
		{
			tvp[3] = tvp[4];
			for (i[3] = 0; i[3] < b_tensor_view.info.dim[2]; i[3]++)
			{
				tvp[2] = tvp[3];
				for (i[2] = 0; i[2] < b_tensor_view.info.dim[3]; i[2]++)
				{
					tvp[1] = tvp[2];
					for (i[1] = 0; i[1] < b_tensor_view.info.dim[4]; i[1]++)
					{
						tvp[0] = tvp[1];
						for (i[0] = 0; i[0] < b_tensor_view.info.dim[5]; i[0]++)
						{
							tvp[0][i[0]] = 0;
						}
						tvp[1] += b_tensor_view.inc[5];
					}
					tvp[2] += b_tensor_view.inc[4] * b_tensor_view.inc[5];
				}
				tvp[3] += b_tensor_view.inc[3] * b_tensor_view.inc[4] * b_tensor_view.inc[5];
			}
			tvp[4] += b_tensor_view.inc[2] * b_tensor_view.inc[3] * b_tensor_view.inc[4] * b_tensor_view.inc[5];
		}
		tvp[5] += b_tensor_view.inc[1] * b_tensor_view.inc[2] * b_tensor_view.inc[3] * b_tensor_view.inc[4] * b_tensor_view.inc[5];
	}
	REQUIRE_TENSOR_EQ(a_tensor, b_tensor, "zero'ed tensor view should be equal");
	ccv_nnc_tensor_free(a_tensor);
	ccv_nnc_tensor_free(b_tensor);
}

TEST_CASE("hint tensor")
{
	ccv_nnc_tensor_param_t a = ONE_CPU_TENSOR(234, 128, 3);
	ccv_nnc_hint_t hint = {
		.border = {
			.begin = {1, 1},
			.end = {1, 2}
		},
		.stride = {
			.dim = {8, 7}
		}
	};
	ccv_nnc_cmd_t cmd = CMD_CONVOLUTION_FORWARD(1, 128, 4, 5, 3);
	ccv_nnc_tensor_param_t b;
	ccv_nnc_tensor_param_t w = ONE_CPU_TENSOR(128, 4, 5, 3);
	ccv_nnc_tensor_param_t bias = ONE_CPU_TENSOR(128);
	ccv_nnc_hint_tensor_auto(cmd, TENSOR_PARAM_LIST(a, w, bias), hint, &b, 1);
	REQUIRE_EQ(b.dim[0], 30, "height should be 30");
	REQUIRE_EQ(b.dim[1], 19, "width should be 19");
	REQUIRE_EQ(b.dim[2], 128, "channel should be the convolution filter count");
}

#include "case_main.h"
