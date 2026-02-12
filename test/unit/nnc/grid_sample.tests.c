#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "case.h"
#include "ccv_case.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("grid_sample forward with align corners")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, 1), 0);
	int i;
	for (i = 0; i < 4; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 2, 2), 0);
	grid->data.f32[0] = -1;
	grid->data.f32[1] = -1;
	grid->data.f32[2] = 1;
	grid->data.f32[3] = 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 2, 1), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	float bt[2] = {
		1, 4
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 2, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("grid_sample forward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, 1), 0);
	int i;
	for (i = 0; i < 4; i++)
		a->data.f32[i] = i + 1;
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 2, 2), 0);
	grid->data.f32[0] = -1;
	grid->data.f32[1] = -1;
	grid->data.f32[2] = 1;
	grid->data.f32[3] = 1;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 2, 1), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	float bt[2] = {
		0.25, 1
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 2, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
