#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("compare rotate half with mpsgraph fallback")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	float ap[] = {
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 8), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 8), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	float bp[] = {
		5, 6, 7, 8, 1, 2, 3, 4,
		13, 14, 15, 16, 9, 10, 11, 12,
		21, 22, 23, 24, 17, 18, 19, 20,
		29, 30, 31, 32, 25, 26, 27, 28,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(bp, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	REQUIRE_TENSOR_EQ(hb, &bt, "mps should rotate the last dim by half");
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("compare rotate half backward with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
	};
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 8), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 8), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg), TENSOR_LIST(g), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(h), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h), TENSOR_LIST(hh), 0);
	float hp[] = {
		5, 6, 7, 8, 1, 2, 3, 4,
		13, 14, 15, 16, 9, 10, 11, 12,
	};
	ccv_nnc_tensor_t const ht = ccv_nnc_tensor(hp, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	REQUIRE_TENSOR_EQ(hh, &ht, "mps should rotate gradient by half");
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(hh);
}

#include "case_main.h"
