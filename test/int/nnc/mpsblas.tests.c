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

TEST_CASE("gemm transpose a batch 2")
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
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

#include "case_main.h"
