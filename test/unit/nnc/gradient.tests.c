#include "case.h"
#include "ccv_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

// five-stencil constants
static double fs[4] = { 1, -8, 8, -1 };
static double fsh[4] = { -2, -1, 1, 2 };

TEST_CASE("numerical gradient versus analytical gradient for convolutional network")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(31, 21, 4), 0);
	ccv_nnc_cmd_t forw_cmd = CMD_CONVOLUTION_FORWARD(1, 4, 5, 3, 2);
	ccv_nnc_hint_t hint = ccv_nnc_hint_auto(forw_cmd.info, a->info, b->info);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 5, 3, 2), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4), 0);
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
	ccv_nnc_cmd_exec(forw_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* ba = ccv_nnc_tensor_new(b->data.f32, ONE_CPU_TENSOR(31 * 21 * 4), 0);
	ccv_nnc_tensor_t* m = ccv_nnc_tensor_new(0, ba->info, 0);
	ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), hint, 0, TENSOR_LIST(ba), TENSOR_LIST(m), 0);
	ccv_nnc_cmd_t back_cmd = CMD_CONVOLUTION_BACKWARD(1, 4, 5, 3, 2);
	ccv_nnc_tensor_t* gw = ccv_nnc_tensor_new(0, w->info, 0);
	ccv_nnc_tensor_t* gbias = ccv_nnc_tensor_new(0, bias->info, 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, b->info, 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, a->info, 0);
	for (i = 0; i < 21 * 31 * 4; i++)
		g->data.f32[i] = m->data.f32[i] - (i == 24);
	ccv_nnc_cmd_exec(back_cmd, hint, 0, TENSOR_LIST(g, a, w), TENSOR_LIST(h, gw, gbias), 0);
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
			ccv_nnc_cmd_exec(forw_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
			ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), hint, 0, TENSOR_LIST(ba), TENSOR_LIST(m), 0);
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
			ccv_nnc_cmd_exec(forw_cmd, hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
			ccv_nnc_cmd_exec(CMD_SOFTMAX_FORWARD(), hint, 0, TENSOR_LIST(ba), TENSOR_LIST(m), 0);
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
	ccv_nnc_tensor_free(ba);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(gbias);
}

#include "case_main.h"
