#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "nnc/ccv_nnc.h"

TEST_CASE("convolutional network of 3x5 on 21x31 for error backward propagation")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			32, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t h_params = a_params;
	ccv_nnc_tensor_param_t g_params = b_params;
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			3, 3, 5, 32,
		},
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			32,
		},
	};
	ccv_nnc_cmd_param_t cmd_params = {
		.size = {
			.dim = {
				3, 3, 5,
			},
		},
		.convolutional = {
			.count = 32,
		},
	};
	ccv_nnc_hint_t hint = ccv_nnc_hint_guess(cmd_params, &a_params, 1, &b_params, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_cmd_t forw_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, 0, cmd_params, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, bias_params, 0);
	int i, j, k;
	for (i = 0; i < 3 * 5 * 3 * 32; i++)
		w->data.f32[i] = 2;
	for (i = 0; i < 21 * 31 * 3; i++)
		a->data.f32[i] = 1;
	for (i = 0; i < 32; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_tensor_t* forw_inlets[] = {
		a,
		w,
		bias,
	};
	ccv_nnc_tensor_t* forw_outlets[] = {
		b,
	};
	ccv_nnc_cmd_exec(forw_cmd, hint, 0, forw_inlets, 3, forw_outlets, 1);
	ccv_nnc_cmd_t back_cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD, 0, cmd_params, 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, g_params, 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, h_params, 0);
	for (i = 0; i < 21 * 31 * 32; i++)
		g->data.f32[i] = 1;
	ccv_nnc_tensor_t* back_inlets[] = {
		g,
		a,
		w,
	};
	ccv_nnc_tensor_t* back_outlets[] = {
		gw,
		gbias,
		h,
	};
	ccv_nnc_cmd_exec(back_cmd, hint, 0, back_inlets, 3, back_outlets, 3);
	ccv_dense_matrix_t* dd = ccv_dense_matrix_new(31, 21, CCV_32F | CCV_C3, 0, 0);
	for (i = 0; i < 31; i++)
		for (j = 0; j < 21; j++)
			dd->data.f32[(i * 21 + j) * 3] =
			dd->data.f32[(i * 21 + j) * 3 + 1] =
			dd->data.f32[(i * 21 + j) * 3 + 2] = 32 * 2 * (5 + ccv_min(i - 2, 0) + ccv_min(28 - i, 0)) * (3 + ccv_min(j - 1, 0) + ccv_min(19 - j, 0));
	REQUIRE_MATRIX_EQ(h, dd, "propagated error doesn't match the expected value");
	ccv_matrix_free(dd);
	float* dw = (float*)ccmalloc(sizeof(float) * 5 * 3 * 3 * 32);
	for (k = 0; k < 32; k++)
		for (i = 0; i < 5; i++)
			for (j = 0; j < 3; j++)
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3] =
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3 + 1] =
				dw[k * 5 * 3 * 3 + (i * 3 + j) * 3 + 2] = (31 - abs(i - 2)) * (21 - abs(j - 1));
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dw, gw->data.f32, 5 * 3 * 3 * 32, 1e-4, "weight gradient doesn't match the expected value");
	ccfree(dw);
	float* dbias = (float*)ccmalloc(sizeof(float) * 32);
	for (i = 0; i < 32; i++)
		dbias[i] = 21 * 31;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbias, gbias->data.f32, 32, 1e-4, "bias gradient doesn't match the expected value");
	ccfree(dbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

#include "case_main.h"
