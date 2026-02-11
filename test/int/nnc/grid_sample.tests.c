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

TEST_CASE("grid sample forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 1, 2, 2), 0);
	ccv_nnc_tensor_t* const grid = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, 2, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 1, 1, 2), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 1, 2, 2), 0);
	int i;
	for (i = 0; i < 4; i++)
		ha->data.f32[i] = i + 1;
	ccv_nnc_tensor_t* const hgrid = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, 2, 2), 0);
	hgrid->data.f32[0] = -1;
	hgrid->data.f32[1] = -1;
	hgrid->data.f32[2] = 1;
	hgrid->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgrid), TENSOR_LIST(grid), 0);
	ccv_nnc_cmd_exec(CMD_GRID_SAMPLE_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a, grid), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 1, 1, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	float bt[2] = {
		1, 4
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt, 2, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(grid);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hgrid);
	ccv_nnc_tensor_free(hb);
}

#include "case_main.h"
