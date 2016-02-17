#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "nnc/ccv_nnc.h"
#include "3rdparty/dsfmt/dSFMT.h"

// five-stencil constants
static double fs[4] = { 1, -8, 8, -1 };
static double fsh[4] = { -2, -1, 1, 2 };

TEST_CASE("numerical gradient versus analytical gradient for convolutional network")
{
	ccv_nnc_init();
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			2, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4, 21, 31,
		},
	};
	ccv_nnc_tensor_param_t h_params = a_params;
	ccv_nnc_tensor_param_t g_params = b_params;
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			2, 3, 5, 4,
		},
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.dim = {
			4,
		},
	};
	ccv_nnc_net_command_param_t command_params = {
		.size = {
			.dim = {
				2, 3, 5,
			},
		},
		.convolutional = {
			.count = 4,
		},
	};
	ccv_nnc_net_hint_t hint = ccv_nnc_net_hint_guess(command_params, &a_params, 1, &b_params, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_net_command_t forw_command = ccv_nnc_net_command(CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD, command_params, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, bias_params, 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i, j;
	for (i = 0; i < 2 * 3 * 5 * 4; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 1.41421356237 / sqrtf(21 * 31 * 2 + 21 * 31 * 4);
	float denom = (21 * 31 * 2 - 1) * 21 * 31 * 2;
	for (i = 0; i < 21 * 31 * 2; i++)
		a->data.f32[i] = (float)(i - 21 * 31) / denom;
	for (i = 0; i < 4; i++)
		bias->data.f32[i] = 0;
	ccv_nnc_tensor_t* forw_inlets[] = {
		a,
		w,
		bias,
	};
	ccv_nnc_tensor_t* forw_outlets[] = {
		b,
	};
	ccv_nnc_net_command_exec(forw_command, hint, 0, forw_inlets, 3, forw_outlets, 1);
	ccv_nnc_net_command_t softmax_command = ccv_nnc_net_command(CCV_NNC_COMPUTE_SOFTMAX_FORWARD, command_params, 0);
	ccv_nnc_tensor_t* m = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* max_inlets[] = {
		b,
	};
	ccv_nnc_tensor_t* max_outlets[] = {
		m,
	};
	ccv_nnc_net_command_exec(softmax_command, hint, 0, max_inlets, 1, max_outlets, 1);
	ccv_nnc_net_command_t back_command = ccv_nnc_net_command(CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD, command_params, 0);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, g_params, 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, h_params, 0);
	for (i = 0; i < 21 * 31 * 4; i++)
		g->data.f32[i] = m->data.f32[i] - (i == 24);
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
	ccv_nnc_net_command_exec(back_command, hint, 0, back_inlets, 3, back_outlets, 3);
	// Now doing numeric gradient computation
	static const double eps = 0.001;
	float* dw = (float*)ccmalloc(sizeof(float) * 2 * 3 * 5 * 4); 
	for (i = 0; i < 2 * 3 * 5 * 4; i++)
	{
		double vw = 0;
		for (j = 0; j < 4; j++)
		{
			float old_w = w->data.f32[i];
			w->data.f32[i] += fsh[j] * eps;
			ccv_nnc_net_command_exec(forw_command, hint, 0, forw_inlets, 3, forw_outlets, 1);
			ccv_nnc_net_command_exec(softmax_command, hint, 0, max_inlets, 1, max_outlets, 1);
			vw += -log(m->data.f32[24]) * fs[j];
			w->data.f32[i] = old_w;
		}
		dw[i] = vw / (12 * eps);
	}
	float* dbias = (float*)ccmalloc(sizeof(float) * 4);
	for (i = 0; i < 4; i++)
	{
		dbias[i] = 0;
		for (j = 0; j < 4; j++)
		{
			float old_bias = bias->data.f32[i];
			bias->data.f32[i] += fsh[j] * eps;
			ccv_nnc_net_command_exec(forw_command, hint, 0, forw_inlets, 3, forw_outlets, 1);
			ccv_nnc_net_command_exec(softmax_command, hint, 0, max_inlets, 1, max_outlets, 1);
			dbias[i] += -logf(m->data.f32[24]) * fs[j];
			bias->data.f32[i] = old_bias;
		}
		dbias[i] *= 1.0 / (12 * eps);
	}
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dw, gw->data.f32, 2 * 3 * 5 * 4, 30, 2e-1, "weight gradient from analytical method doesn't match the one from numerical method");
	REQUIRE_ARRAY_EQ_WITHIN_ANGLE_AND_MAGNITUDE(float, dbias, gbias->data.f32, 4, 30, 2e-1, "bias gradient from analytical method doesn't match the one from numerical method");
	ccfree(dw);
	ccfree(dbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

#include "case_main.h"
