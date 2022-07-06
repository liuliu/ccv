#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("mse loss forward")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i, j;
	for (i = 0; i < 100; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 100; i++)
		b->data.f32[i] = 0;
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* tc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	for (i = 0; i < 10; i++)
	{
		tc->data.f32[i] = 0;
		for (j = 0; j < 10; j++)
			tc->data.f32[i] += a->data.f32[j + i * 10] * a->data.f32[j + i * 10];
		tc->data.f32[i] *= 1.0 / 10.0;
	}
	REQUIRE_TENSOR_EQ(tc, c, "CPU computed output should be the same as simply computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(tc);
}

TEST_CASE("mse loss backward")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 100; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 100; i++)
		b->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_MSE_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_MSE_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(da, db), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	ccv_nnc_tensor_t* tdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 10), 0);
	for (i = 0; i < 100; i++)
		tda->data.f32[i] = 2 * a->data.f32[i] / 10;
	for (i = 0; i < 100; i++)
		tdb->data.f32[i] = -2 * a->data.f32[i] / 10;
	REQUIRE_TENSOR_EQ(tda, da, "CPU computed output should be the same as simply computed ones");
	REQUIRE_TENSOR_EQ(tdb, db, "CPU computed output should be the same as simply computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdb);
}

#include "case_main.h"
