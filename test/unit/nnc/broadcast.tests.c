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

TEST_CASE("broadcasting semantics for add [[1, 2, 3], [4, 5, 6]] + [7, 8, 9]")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	b->data.f32[0] = 7;
	b->data.f32[1] = 8;
	b->data.f32[2] = 9;
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	float ctp[] = {
		8, 10, 12,
		11, 13, 15
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, ONE_CPU_TENSOR(2, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("broadcasting semantics for add [[1], [2], [3], [4]] + [5, 6]")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	float ctp[] = {
		6, 7,
		7, 8,
		8, 9,
		9, 10
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, ONE_CPU_TENSOR(4, 2), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("broadcasting semantics for mul [[1, 2, 3], [4, 5, 6]] * [7, 8, 9]")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	b->data.f32[0] = 7;
	b->data.f32[1] = 8;
	b->data.f32[2] = 9;
	ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	float ctp[] = {
		7, 16, 27,
		28, 40, 54
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, ONE_CPU_TENSOR(2, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("broadcasting semantics for mul [[1], [2], [3], [4]] * [5, 6]")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(4, 2), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	b->data.f32[0] = 5;
	b->data.f32[1] = 6;
	ccv_nnc_cmd_exec(CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	float ctp[] = {
		5, 6,
		10, 12,
		15, 18,
		20, 24
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, ONE_CPU_TENSOR(4, 2), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

#include "case_main.h"
