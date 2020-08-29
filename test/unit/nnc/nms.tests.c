#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("compare non-maximal suppression forward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	int i;
	for (i = 0; i < 10; i++)
	{
		a->data.f32[i * 5] = i;
		a->data.f32[i * 5 + 1] = i;
		a->data.f32[i * 5 + 2] = 0;
		a->data.f32[i * 5 + 3] = 1;
		a->data.f32[i * 5 + 4] = 1;
	}
	ccv_nnc_cmd_exec(CMD_NMS_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, c), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 5), 0);
	for (i = 0; i < 10; i++)
	{
		bt->data.f32[i * 5] = 9 - i;
		bt->data.f32[i * 5 + 1] = 9 - i;
		bt->data.f32[i * 5 + 2] = 0;
		bt->data.f32[i * 5 + 3] = 1;
		bt->data.f32[i * 5 + 4] = 1;
	}
	REQUIRE_TENSOR_EQ(b, bt, "should be equal");
	int ct[10] = {};
	for (i = 0; i < 10; i++)
		ct[i] = 9 - i;
	REQUIRE_ARRAY_EQ(int, c->data.i32, ct, 10, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("compare non-maximal suppression forward with tensor views")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 8), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 6), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10), 0);
	int i;
	for (i = 0; i < 10; i++)
	{
		a->data.f32[i * 8] = i;
		a->data.f32[i * 8 + 1] = i;
		a->data.f32[i * 8 + 2] = 0;
		a->data.f32[i * 8 + 3] = 1;
		a->data.f32[i * 8 + 4] = 1;
	}
	memset(b->data.f32, 0, sizeof(float) * 10 * 6);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 10, 5), ccv_nnc_no_ofs, DIM_ALLOC(10, 8));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 10, 5), ccv_nnc_no_ofs, DIM_ALLOC(10, 6));
	ccv_nnc_cmd_exec(CMD_NMS_FORWARD(0.5), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST((ccv_nnc_tensor_t*)bv, c), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 6), 0);
	for (i = 0; i < 10; i++)
	{
		bt->data.f32[i * 6] = 9 - i;
		bt->data.f32[i * 6 + 1] = 9 - i;
		bt->data.f32[i * 6 + 2] = 0;
		bt->data.f32[i * 6 + 3] = 1;
		bt->data.f32[i * 6 + 4] = 1;
		bt->data.f32[i * 6 + 5] = 0;
	}
	REQUIRE_TENSOR_EQ(b, bt, "should be equal");
	int ct[10] = {};
	for (i = 0; i < 10; i++)
		ct[i] = 9 - i;
	REQUIRE_ARRAY_EQ(int, c->data.i32, ct, 10, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(bv);
}

#include "case_main.h"
