#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("logarithmic histogram v.s. bins histogram")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3000, 2000), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 774 * 2), 0);
	double v = 1e-12;
	int i;
	for (i = 0; v < 1e20; i++, v *= 1.1)
	{
		h->data.f32[i + 774] = v;
		h->data.f32[774 - i - 1] = -v;
	}
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 3000 * 2000; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2000 - 1000;
	a->data.f32[0] = NAN;
	a->data.f32[1] = -FLT_MAX;
	a->data.f32[2] = FLT_MAX;
	a->data.f32[3] = 0;
	a->data.f32[4] = 1e-20;
	a->data.f32[5] = -1e-20;
	a->data.f32[6] = 1e20 - 1e18;
	a->data.f32[7] = -1e20 + 1e18;
	ccv_nnc_tensor_t* const bl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 775 * 2), 0);
	ccv_nnc_tensor_t* const bb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 775 * 2), 0);
	ccv_nnc_tensor_t* const sl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_tensor_t* const sb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_LOG(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(bl, sl), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_BINS(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, h), TENSOR_LIST(bb, sb), 0);
	for (i = 0; i < 775 * 2; i++)
	{
		const int sum = bb->data.i32[i] + bl->data.i32[i];
		const float tol = sum > 10 ? fabsf((float)(bl->data.i32[i] - bb->data.i32[i])) / (float)sum : abs(bl->data.i32[i] - bb->data.i32[i]);
		REQUIRE(tol < 0.1, "should be smaller than tolerance for bl->data.i32[%d](%d), bb->data.i32[%d](%d)", i, bl->data.i32[i], i, bb->data.i32[i]);
	}
	REQUIRE_EQ(sl->data.f32[0], -FLT_MAX, "should be equal to the minimal");
	REQUIRE_EQ(sl->data.f32[1], FLT_MAX, "should be equal to the maximal");
	REQUIRE(isnan(sl->data.f32[2]), "sum is nan");
	REQUIRE(isnan(sl->data.f32[3]), "sum of squares is nan");
	REQUIRE_EQ(bb->data.i32[0], 1, "should have 1 smaller than minimal value");
	REQUIRE_EQ(bb->data.i32[774 * 2], 1, "should have 1 larger than maximal value");
	REQUIRE_EQ(bb->data.i32[775 * 2 - 1], 1, "should have 1 nan");
	REQUIRE_EQ(bb->data.i32[774 * 2 - 1], 1, "should have 1 in the range (1e20 / 1.1, 1e20)");
	REQUIRE_EQ(bb->data.i32[1], 1, "should have 1 in the range (-1e20, -1e20 / 1.1)");
	REQUIRE_EQ(bb->data.i32[774], 3, "should have 3 in the range (-1e-12, 1e-12)");
	REQUIRE_EQ(bl->data.i32[0], 1, "should have 1 smaller than minimal value");
	REQUIRE_EQ(bl->data.i32[774 * 2], 1, "should have 1 larger than maximal value");
	REQUIRE_EQ(bl->data.i32[775 * 2 - 1], 1, "should have 1 nan");
	REQUIRE_EQ(bl->data.i32[774 * 2 - 1], 1, "should have 1 in the range (1e20 / 1.1, 1e20)");
	REQUIRE_EQ(bl->data.i32[1], 1, "should have 1 in the range (-1e20, -1e20 / 1.1)");
	REQUIRE_EQ(bl->data.i32[774], 3, "should have 3 in the range (-1e-12, 1e-12)");
	REQUIRE_TENSOR_EQ(sl, sb, "computed stats should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bl);
	ccv_nnc_tensor_free(bb);
	ccv_nnc_tensor_free(sl);
	ccv_nnc_tensor_free(sb);
}

TEST_CASE("even histogram v.s. bins histogram")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3000, 2000), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 201), 0);
	int i;
	for (i = 0; i < 201; i++)
	{
		h->data.f32[i] = (i - 100) * 10;
	}
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 3000 * 2000; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2000 - 1000;
	a->data.f32[0] = NAN;
	a->data.f32[1] = -FLT_MAX;
	a->data.f32[2] = FLT_MAX;
	a->data.f32[3] = 0;
	a->data.f32[4] = 1e-20;
	a->data.f32[5] = -1e-20;
	a->data.f32[6] = 1e20 - 1e18;
	a->data.f32[7] = -1e20 + 1e18;
	ccv_nnc_tensor_t* const bl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 203), 0);
	ccv_nnc_tensor_t* const bb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 203), 0);
	ccv_nnc_tensor_t* const sl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_tensor_t* const sb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_EVEN(200, -1000, 1000), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(bl, sl), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_BINS(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, h), TENSOR_LIST(bb, sb), 0);
	for (i = 0; i < 203; i++)
	{
		const int diff = abs(bb->data.i32[i] - bl->data.i32[i]);
		REQUIRE(diff < 5, "differences should be smaller than 5 for bl->data.i32[%d](%d), bb->data.i32[%d](%d)", i, bl->data.i32[i], i, bb->data.i32[i]);
	}
	REQUIRE_EQ(sl->data.f32[0], -FLT_MAX, "should be equal to the minimal");
	REQUIRE_EQ(sl->data.f32[1], FLT_MAX, "should be equal to the maximal");
	REQUIRE(isnan(sl->data.f32[2]), "sum is nan");
	REQUIRE(isnan(sl->data.f32[3]), "sum of squares is nan");
	REQUIRE_EQ(bb->data.i32[0], 2, "should have 2 smaller than minimal value");
	REQUIRE_EQ(bb->data.i32[201], 2, "should have 2 larger than maximal value");
	REQUIRE_EQ(bb->data.i32[202], 1, "should have 1 nan");
	REQUIRE_EQ(bl->data.i32[0], 2, "should have 2 smaller than minimal value");
	REQUIRE_EQ(bl->data.i32[201], 2, "should have 2 larger than maximal value");
	REQUIRE_EQ(bl->data.i32[202], 1, "should have 1 nan");
	REQUIRE_TENSOR_EQ(sl, sb, "computed stats should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bl);
	ccv_nnc_tensor_free(bb);
	ccv_nnc_tensor_free(sl);
	ccv_nnc_tensor_free(sb);
}

TEST_CASE("logarithmic histogram v.s. bins histogram on tensor view")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 30, 100, 20, 100), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 774 * 2), 0);
	double v = 1e-12;
	int i;
	for (i = 0; v < 1e20; i++, v *= 1.1)
	{
		h->data.f32[i + 774] = v;
		h->data.f32[774 - i - 1] = -v;
	}
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 3000 * 2000; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2000 - 1000;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NCHW(32F, 30, 20, 20, 40), DIM_ALLOC(0, 10, 0, 20), DIM_ALLOC(30, 100, 20, 100));
	// All these skipped.
	a->data.f32[0] = NAN;
	a->data.f32[1] = -FLT_MAX;
	a->data.f32[2] = FLT_MAX;
	a->data.f32[3] = 0;
	a->data.f32[4] = 1e-20;
	a->data.f32[5] = -1e-20;
	a->data.f32[6] = 1e20 - 1e18;
	a->data.f32[7] = -1e20 + 1e18;
	ccv_nnc_tensor_t* const bl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 775 * 2), 0);
	ccv_nnc_tensor_t* const bb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 775 * 2), 0);
	ccv_nnc_tensor_t* const sl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_tensor_t* const sb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_LOG(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(bl, sl), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_BINS(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, h), TENSOR_LIST(bb, sb), 0);
	for (i = 0; i < 775 * 2; i++)
	{
		const int sum = bb->data.i32[i] + bl->data.i32[i];
		const float tol = sum > 10 ? fabsf((float)(bl->data.i32[i] - bb->data.i32[i])) / (float)sum : abs(bl->data.i32[i] - bb->data.i32[i]);
		REQUIRE(tol < 0.1, "should be smaller than tolerance for bl->data.i32[%d](%d), bb->data.i32[%d](%d)", i, bl->data.i32[i], i, bb->data.i32[i]);
	}
	REQUIRE(sl->data.f32[0] < -990, "minimal should be close to the edge");
	REQUIRE(sl->data.f32[1] > 990, "maximal should be close to the edge");
	REQUIRE(sl->data.f32[2] / (float)ccv_nnc_tensor_count(av->info) < 1, "sum should be close to 0");
	REQUIRE_EQ_WITH_TOLERANCE(sl->data.f32[3] / (float)ccv_nnc_tensor_count(av->info), 2000 * 2000 / 12, 1e3, "sum of squares should be close to its expected value in uniform distribution");
	REQUIRE_EQ(bb->data.i32[0], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[774 * 2], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[775 * 2 - 1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[774 * 2 - 1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[774], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[0], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[774 * 2], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[775 * 2 - 1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[774 * 2 - 1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[1], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[774], 0, "should've skipped all manual numbers");
	REQUIRE_TENSOR_EQ(sl, sb, "computed stats should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bl);
	ccv_nnc_tensor_free(bb);
	ccv_nnc_tensor_free(sl);
	ccv_nnc_tensor_free(sb);
}

TEST_CASE("even histogram v.s. bins histogram on tensor view")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 30, 100, 20, 100), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 201), 0);
	int i;
	for (i = 0; i < 201; i++)
	{
		h->data.f32[i] = (i - 100) * 10;
	}
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 3000 * 2000; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2000 - 1000;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NCHW(32F, 30, 20, 20, 40), DIM_ALLOC(0, 10, 0, 20), DIM_ALLOC(30, 100, 20, 100));
	a->data.f32[0] = NAN;
	a->data.f32[1] = -FLT_MAX;
	a->data.f32[2] = FLT_MAX;
	a->data.f32[3] = 0;
	a->data.f32[4] = 1e-20;
	a->data.f32[5] = -1e-20;
	a->data.f32[6] = 1e20 - 1e18;
	a->data.f32[7] = -1e20 + 1e18;
	ccv_nnc_tensor_t* const bl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 203), 0);
	ccv_nnc_tensor_t* const bb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 203), 0);
	ccv_nnc_tensor_t* const sl = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_tensor_t* const sb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 4), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_EVEN(200, -1000, 1000), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(bl, sl), 0);
	ccv_nnc_cmd_exec(CMD_HISTOGRAM_BINS(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, h), TENSOR_LIST(bb, sb), 0);
	for (i = 0; i < 203; i++)
	{
		const int diff = abs(bb->data.i32[i] - bl->data.i32[i]);
		REQUIRE(diff < 5, "differences should be smaller than 5 for bl->data.i32[%d](%d), bb->data.i32[%d](%d)", i, bl->data.i32[i], i, bb->data.i32[i]);
	}
	REQUIRE(sl->data.f32[0] < -990, "minimal should be close to the edge");
	REQUIRE(sl->data.f32[1] > 990, "maximal should be close to the edge");
	REQUIRE(sl->data.f32[2] / (float)ccv_nnc_tensor_count(av->info) < 1, "sum should be close to 0");
	REQUIRE_EQ_WITH_TOLERANCE(sl->data.f32[3] / (float)ccv_nnc_tensor_count(av->info), 2000 * 2000 / 12, 1e3, "sum of squares should be close to its expected value in uniform distribution");
	REQUIRE_EQ(bb->data.i32[0], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[201], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bb->data.i32[202], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[0], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[201], 0, "should've skipped all manual numbers");
	REQUIRE_EQ(bl->data.i32[202], 0, "should've skipped all manual numbers");
	REQUIRE_TENSOR_EQ(sl, sb, "computed stats should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bl);
	ccv_nnc_tensor_free(bb);
	ccv_nnc_tensor_free(sl);
	ccv_nnc_tensor_free(sb);
}

#include "case_main.h"
