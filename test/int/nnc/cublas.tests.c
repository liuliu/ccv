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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("backward gemm with no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
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

TEST_CASE("backward gemm with transpose a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("gemm no transpose batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("backward gemm with no transpose batch 2, same b")
{
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

TEST_CASE("cublas forward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas forward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas forward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas forward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas backward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas backward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);

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
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_tensor_t* hg2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(ha2, hw2, hbias2, hg2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2, hg2), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, h), TENSOR_LIST(tb, tdw, tdbias, th), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb, tdw, tdbias, th), TENSOR_LIST(tb1, tdw1, tdbias1, th1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 10 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw1->data.f32, hdw->data.f32, 64 * 128, 1e-2, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias1->data.f32, hdbias->data.f32, 64, 5e-3, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th1->data.f32, hh->data.f32, 10 * 128, 1e-3, "GPU computed output should be the same as CPU computed ones");
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
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
	ccv_nnc_tensor_free(hg2);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(tdw1);
	ccv_nnc_tensor_free(tdbias1);
	ccv_nnc_tensor_free(th1);
}

TEST_CASE("cublas backward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas backward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);

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
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hg2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hg), TENSOR_LIST(ha2, hw2, hg2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hg2), TENSOR_LIST(a, w, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, 0), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, 0), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* th1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, h), TENSOR_LIST(tb, tdw, th), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb, tdw, th), TENSOR_LIST(tb1, tdw1, th1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 10 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw1->data.f32, hdw->data.f32, 64 * 128, 1e-2, "GPU computed output should be the same as CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th1->data.f32, hh->data.f32, 10 * 128, 1e-3, "GPU computed output should be the same as CPU computed ones");
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
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hg2);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(tdw1);
	ccv_nnc_tensor_free(th1);
}

TEST_CASE("ewdiv forward with reciprocal")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
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

TEST_CASE("ewdiv backward with output 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("ewdiv backward with output 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* db = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b, c), TENSOR_LIST(0, db), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb, hc), TENSOR_LIST(0, dbt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(db), TENSOR_LIST(hdb), 0);
	REQUIRE_TENSOR_EQ(dbt, hdb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdb);
	ccv_nnc_tensor_free(dbt);
}

TEST_CASE("clamp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
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

TEST_CASE("clamp backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("clamp forward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
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

TEST_CASE("clamp backward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(NAN, 5), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("clamp forward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
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

TEST_CASE("clamp backward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_BACKWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

#include "case_main.h"
