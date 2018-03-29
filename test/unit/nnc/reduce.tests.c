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

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 1")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 1), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		6,
		15
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, ONE_CPU_TENSOR(2, 1), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("reduce sum for [[1, 2, 3], [4, 5, 6]] on axis 0")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(3), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		5, 7, 9
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, ONE_CPU_TENSOR(3), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
