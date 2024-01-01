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

TEST_CASE("gemm no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm no transpose with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 + 1, 1 * 8 + 2 * 11 - 1, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 + 1, 3 * 8 + 4 * 11 - 1, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 + 1, 5 * 8 + 6 * 11 - 1, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 + 1, 7 * 8 + 8 * 11 - 1, 7 * 9 + 8 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm no transpose batch 2, no batch b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 7 + 3 * 10, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12,
		4 * 7 + 5 * 10, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12,
		6 * 7 + 7 * 10, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12,
		8 * 7 + 9 * 10, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm no transpose batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
		8, 9, 10,
		11, 12, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 8 + 3 * 11, 2 * 9 + 3 * 12, 2 * 10 + 3 * 13,
		4 * 8 + 5 * 11, 4 * 9 + 5 * 12, 4 * 10 + 5 * 13,
		6 * 8 + 7 * 11, 6 * 9 + 7 * 12, 6 * 10 + 7 * 13,
		8 * 8 + 9 * 11, 8 * 9 + 9 * 12, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a batch 2, no batch b, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
		2, 4, 6, 8,
		3, 5, 7, 9,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		-1, 0, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 7 + 3 * 10 - 1, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12 + 1,
		4 * 7 + 5 * 10 - 1, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12 + 1,
		6 * 7 + 7 * 10 - 1, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12 + 1,
		8 * 7 + 9 * 10 - 1, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm transpose a batch 2, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
		2, 4, 6, 8,
		3, 5, 7, 9,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
		8, 9, 10,
		11, 12, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 8 + 3 * 11, 2 * 9 + 3 * 12, 2 * 10 + 3 * 13,
		4 * 8 + 5 * 11, 4 * 9 + 5 * 12, 4 * 10 + 5 * 13,
		6 * 8 + 7 * 11, 6 * 9 + 7 * 12, 6 * 10 + 7 * 13,
		8 * 8 + 9 * 11, 8 * 9 + 9 * 12, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose b batch 2, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
		80, 110,
		90, 120,
		10, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	float dp[] = {
		-1, 0, 1,
		2, 3, -4,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 80 + 3 * 110 + 2, 2 * 90 + 3 * 120 + 3, 2 * 10 + 3 * 13 - 4,
		4 * 80 + 5 * 110 + 2, 4 * 90 + 5 * 120 + 3, 4 * 10 + 5 * 13 - 4,
		6 * 80 + 7 * 110 + 2, 6 * 90 + 7 * 120 + 3, 6 * 10 + 7 * 13 - 4,
		8 * 80 + 9 * 110 + 2, 8 * 90 + 9 * 120 + 3, 8 * 10 + 9 * 13 - 4,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm transpose b batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
		80, 110,
		90, 120,
		10, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 80 + 3 * 110, 2 * 90 + 3 * 120, 2 * 10 + 3 * 13,
		4 * 80 + 5 * 110, 4 * 90 + 5 * 120, 4 * 10 + 5 * 13,
		6 * 80 + 7 * 110, 6 * 90 + 7 * 120, 6 * 10 + 7 * 13,
		8 * 80 + 9 * 110, 8 * 90 + 9 * 120, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("mps forward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_TENSOR_EQ(tb1, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps forward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 5e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps forward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_TENSOR_EQ(tb1, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps forward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps handle permute")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);
	int i;
	for (i = 0; i < 2 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 2 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(hb, hbt), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw), TENSOR_LIST(dat, tdw), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w, dbias), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw, hdbias), TENSOR_LIST(dat, tdw, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("ewdiv forward with reciprocal")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewdiv forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(ct), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_TENSOR_EQ(ct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("exp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewlog forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewsqrt forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("compare set with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 11, 10, 9, 8), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 11, 10, 9, 8), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 11, 10, 9, 8), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	REQUIRE_TENSOR_EQ(ha, ga, "format transform result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("scaled dot product attention with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 3
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] =         {  32,  32,   3, 2, 1 };
		int R_candidates[num_trials] =         { 128, 128,  61, 6, 2 };
		int C_candidates[num_trials] =         { 128, 128,  49, 2, 1 };
		int Hq_candidates[num_trials] =        {   8,  32,  13, 3, 1 };
		int Hk_candidates[num_trials] =        {   8,   8,  13, 3, 1 };
		int D_candidates[num_trials] =         {  64, 128, 191, 4, 8 };
		int is_causal_candidates[num_trials] = {   0,   1,   0, 1, 0 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		if (is_causal)
		{
			ccv_nnc_tensor_t* const causal_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
			ccv_nnc_tensor_t* const gpu_causal_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, R, C), 0);
			for (int i = 0; i < R; i++)
				for (int j = 0; j < C; j++)
					causal_mask->data.f32[i * C + j] = 0;
			for (int i = 0; i < R - 1; i++)
				for (int j = i - R + C + 1; j < C; j++)
					causal_mask->data.f32[i * C + j] = -FLT_MAX;
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(causal_mask), TENSOR_LIST(gpu_causal_mask), 0);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_causal_mask), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_tensor_free(gpu_causal_mask);
			ccv_nnc_tensor_free(causal_mask);
		} else {
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		}

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		REQUIRE_TENSOR_EQ(copy_of_gpu_o_tensor, o_tensor, "scaled dot product attention result should be the same");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention + unify head with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512), "bias");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(q, k, v, NO_TENSOR_SYMBOL, w, bias), TENSOR_SYMBOL_LIST(r, NO_TENSOR_SYMBOL, c), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, q);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, k);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, v);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bias);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512 * 512; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(g_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(gq, gk, gv, NO_TENSOR_SYMBOL, gw, gbias), TENSOR_SYMBOL_LIST(gr, NO_TENSOR_SYMBOL, gc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(g_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* g_graph = 0;
	ccv_nnc_tensor_arena_t* g_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* g_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(g_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(g_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(g_symbolic_graph), &g_graph, &g_tensor_arena, &g_graph_exec_arena);
	ccv_nnc_tensor_t* const gq_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gq);
	ccv_nnc_tensor_t* const gk_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gk);
	ccv_nnc_tensor_t* const gv_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gv);
	ccv_nnc_tensor_t* const gw_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gw);
	ccv_nnc_tensor_t* const gbias_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gbias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gr_tensor), TENSOR_LIST(hr), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, r_tensor->data.f32, hr->data.f32, 32 * 128 * 512, 1e-4, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(hr);
}

TEST_CASE("scaled dot product attention gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
#define num_long_trials 2
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   3, 2, 1 };
		int R_candidates[num_trials] = { 128,  61, 6, 2 };
		int C_candidates[num_trials] = { 128,  49, 2, 1 };
		int H_candidates[num_trials] = {   8,  13, 3, 1 };
		int D_candidates[num_trials] = {  64, 191, 4, 8 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int H = H_candidates[trial];
		int D = D_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);

		for (int i = 0; i < B * R * H * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * H * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * H * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		for (int i = 0; i < B * R * H * D; ++i) {
			o_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, o_tensor), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_o_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_TENSOR_EQ(copy_of_gpu_dv_tensor, dv_tensor, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * H * D, 1e-6, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * H * D, 1e-6, "scaled dot product attention result should be the same");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("backward gemm with no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);

	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};

	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);

	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};

	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);

	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);

	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_BACKWARD();
	cmd.backend = CCV_NNC_BACKEND_MPS;
	cmd.algorithm = 1; // This is cblas.

	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(h, db, dbias), 0);

	ccv_nnc_tensor_t* const ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const cdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h, db, dbias), TENSOR_LIST(ch, cdb, cdbias), 0);

	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);

	REQUIRE_TENSOR_EQ(cdbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);

	REQUIRE_TENSOR_EQ(ch, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(cdb, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
}

TEST_CASE("backward gemm with transpose a")
{
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b")
{
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}


TEST_CASE("backward gemm large data set")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, h), TENSOR_LIST(tb, tdw, tdbias, th), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdw, hdw, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdbias, hdbias, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(th, hh, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hg), TENSOR_LIST(a, w, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, 0), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, 0), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, h), TENSOR_LIST(tb, tdw, th), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdw, hdw, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(th, hh, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("backward gemm no h")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(0, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(0, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, 0), TENSOR_LIST(tb, tdw, tdbias, 0), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdw, hdw, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdbias, hdbias, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backward gemm no dw")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, 0, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, 0, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, 0, dbias, h), TENSOR_LIST(tb, 0, tdbias, th), 0);
	REQUIRE_TENSOR_EQ(tb, hb, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(tdbias, hdbias, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_TENSOR_EQ(th, hh, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backwar gemm with no transpose batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 10 * 24 + 20 * 25 + 30 * 26,
		40 * 21 + 50 * 22 + 60 * 23, 40 * 24 + 50 * 25 + 60 * 26,
		70 * 21 + 80 * 22 + 90 * 23, 70 * 24 + 80 * 25 + 90 * 26,
		100 * 21 + 110 * 22 + 120 * 23, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
		212, 222, 232,
		242, 252, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose batch 2, batched b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
		212, 222, 232,
		242, 252, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("backward gemm with transpose b batch 2, batched b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("backward gemm with transpose a and b batch 2, batch b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2), TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 40 * 212 + 50 * 222 + 60 * 232, 70 * 212 + 80 * 222 + 90 * 232, 100 * 212 + 110 * 222 + 120 * 232,
		10 * 242 + 20 * 252 + 30 * 262, 40 * 242 + 50 * 252 + 60 * 262, 70 * 242 + 80 * 252 + 90 * 262, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

#include "case_main.h"
