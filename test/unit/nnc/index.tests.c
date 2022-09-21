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

TEST_CASE("index select a tensor")
{
	float ap[] = {
		1, 2,
		2, 3,
		3, 4,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(b), 0);
	float btp[] = {
		2, 3,
		2, 3,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("index select a 1d tensor")
{
	float ap[] = {
		1, 2, 3, 4, 5
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices), TENSOR_LIST(b), 0);
	float btp[] = {
		4, 3, 5
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("index select a tensor view")
{
	float ap[] = {
		1, 2, 3, 4,
		2, 3, 4, 5,
		3, 4, 5, 6,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	memset(b->data.f32, 0, 2 * 4 * sizeof(float));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, indices), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	float btp[] = {
		0, 3, 4, 0,
		0, 3, 4, 0,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
}

TEST_CASE("backward index select a tensor")
{
	float bp[] = {
		1, 2,
		2, 3,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, 0, indices), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0,
		3, 5,
		0, 0,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("backward index select a 1d tensor")
{
	float bp[] = {
		4, 3, 5,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5), 0);
	int ip[] = {3, 2, 4};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, 0, indices), TENSOR_LIST(a), 0);
	float atp[] = {
		0, 0, 3, 4, 5
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 5), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("backward index select a tensor view")
{
	float bp[] = {
		0, 3, 4, 0,
		0, 1, 5, 0,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	int i;
	for (i = 0; i < 3 * 4; i++)
		a->data.f32[i] = i;
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, CPU_TENSOR_NHWC(32F, 3, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	int ip[] = {1, 1};
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(ip, CPU_TENSOR_NHWC(32S, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, CPU_TENSOR_NHWC(32F, 2, 2), DIM_ALLOC(0, 1), DIM_ALLOC(4, 1));
	ccv_nnc_cmd_exec(CMD_INDEX_SELECT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)bv, 0, indices), TENSOR_LIST((ccv_nnc_tensor_t*)av), 0);
	float atp[] = {
		0, 0, 0, 3,
		4, 4, 9, 7,
		8, 0, 0, 11,
	};
	ccv_nnc_tensor_t const at = ccv_nnc_tensor(atp, CPU_TENSOR_NHWC(32F, 3, 4), 0);
	REQUIRE_TENSOR_EQ(a, &at, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(bv);
}

#include "case_main.h"
