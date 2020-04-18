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

TEST_CASE("cross entropy loss forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hb->data.f32[i] = (i + 1) * 9;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* tc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(tc), 0);
	REQUIRE_TENSOR_EQ(tc, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(tc);
}

TEST_CASE("cross entropy loss forward with label smoothing")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hb->data.f32[i] = (i + 1) * 9;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* tc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(tc), 0);
	REQUIRE_TENSOR_EQ(tc, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(tc);
}

TEST_CASE("cross entropy loss backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hb->data.f32[i] = (i + 1) * 9;
	for (i = 0; i < 10; i++)
		hg->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hd), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(d), 0);
	ccv_nnc_tensor_t* td = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(td), 0);
	REQUIRE_TENSOR_EQ(td, hd, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(td);
}

TEST_CASE("cross entropy loss backward with label smoothing")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i = 0;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		hb->data.f32[i] = (i + 1) * 9;
	for (i = 0; i < 10; i++)
		hg->data.f32[i] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb, hg), TENSOR_LIST(a, b, g), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hd), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_FORWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_CATEGORICAL_CROSSENTROPY_BACKWARD(0.1, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(d), 0);
	ccv_nnc_tensor_t* td = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d), TENSOR_LIST(td), 0);
	REQUIRE_TENSOR_EQ(td, hd, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(td);
}

TEST_CASE("random uniform distribution")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 100000), "x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RANDOM_UNIFORM_FORWARD(-8, 4), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(x), "random uniform");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	int i;
	int h[4 + 8] = {};
	for (i = 0; i < 100000; i++)
	{
		REQUIRE(xt->data.f32[i] > -8 - 1e-5, "it must be bigger than lower bound");
		REQUIRE(xt->data.f32[i] < 4 + 1e-5, "and smaller than upper bound");
		int b = (int)roundf(xt->data.f32[i] - 0.5) + 8;
		b = ccv_max(ccv_min(b, 11), 0);
		++h[b];
	}
	const int count = (int)roundf(100000. / (4 + 8));
	for (i = 0; i < 12; i++)
		{ REQUIRE(h[i] >= count - 1000 && h[i] <= count + 1000, "uniform distribution"); }
	ccv_nnc_tensor_free(xt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("random uniform distribution in half precision")
{
	ccv_nnc_symbolic_graph_t* symbolic_graph = ccv_nnc_symbolic_graph_new();
	const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 100000), "x");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RANDOM_UNIFORM_FORWARD(-8, 4), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(x), "random uniform");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16t = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 100000), 0);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 100000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16t), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16t), TENSOR_LIST(xt), 0);
	int i;
	int h[4 + 8] = {};
	for (i = 0; i < 100000; i++)
	{
		REQUIRE(xt->data.f32[i] > -8 - 1e-5, "it must be bigger than lower bound");
		REQUIRE(xt->data.f32[i] < 4 + 1e-5, "and smaller than upper bound");
		int b = (int)roundf(xt->data.f32[i] - 0.5) + 8;
		b = ccv_max(ccv_min(b, 11), 0);
		++h[b];
	}
	const int count = (int)roundf(100000. / (4 + 8));
	for (i = 0; i < 12; i++)
		{ REQUIRE(h[i] >= count - 1000 && h[i] <= count + 1000, "uniform distribution"); }
	ccv_nnc_tensor_free(xt);
	ccv_nnc_tensor_free(x16t);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("data conversion from float to half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(short, (short*)hb->data.f16, (short*)bt->data.f16, 128, 1, "Result should be exactly equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("masked fill forward with integer")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MASKED_FILL_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 6 * 5 * 4; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5, 4), 0);
	for (i = 0; i < 5 * 4; i++)
		hb->data.i32[i] = (i % 2 == 1) ? 0 : 1;
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_FORWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 5, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_FORWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hd), 0);
	REQUIRE_TENSOR_EQ(hc, hd, "cpu and gpu result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
}

TEST_CASE("masked fill forward with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MASKED_FILL_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 6 * 5 * 4; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	for (i = 0; i < 5 * 4; i++)
		hb->data.f32[i] = (i % 2 == 1) ? 0 : 1;
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_FORWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_FORWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hd), 0);
	REQUIRE_TENSOR_EQ(hc, hd, "cpu and gpu result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
}

TEST_CASE("masked fill backward with integer")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MASKED_FILL_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 6 * 5 * 4; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5, 4), 0);
	for (i = 0; i < 5 * 4; i++)
		hb->data.i32[i] = (i % 2 == 1) ? 0 : 1;
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_BACKWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, 0, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 5, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_BACKWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(a, 0, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hd), 0);
	REQUIRE_TENSOR_EQ(hc, hd, "cpu and gpu result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
}

TEST_CASE("masked fill backward with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MASKED_FILL_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	int i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	for (i = 0; i < 6 * 5 * 4; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	for (i = 0; i < 5 * 4; i++)
		hb->data.f32[i] = (i % 2 == 1) ? 0 : 1;
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_BACKWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, 0, hb), TENSOR_LIST(hc), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_MASKED_FILL_BACKWARD(0, -1e8), ccv_nnc_no_hint, 0, TENSOR_LIST(a, 0, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hd), 0);
	REQUIRE_TENSOR_EQ(hc, hd, "cpu and gpu result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
}

TEST_CASE("swish in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_TENSOR_EQ(ty, y_tensor, "sigmoid from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(), TENSOR_SYMBOL_LIST(a), TENSOR_SYMBOL_LIST(b), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty->data.f32, y_tensor->data.f32, 20 * 10, 1e-3, "sigmoid from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(ty);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_TENSOR_EQ(ty_tensor, y_tensor, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_TENSOR_EQ(tdx_tensor, dx_tensor, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("swish gradient in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "x");
	ccv_nnc_tensor_symbol_t y = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 10, 100), "y");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SWISH_FORWARD(), TENSOR_SYMBOL_LIST(x), TENSOR_SYMBOL_LIST(y), "sigmoid");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, TENSOR_SYMBOL_LIST(y), TENSOR_SYMBOL_LIST(x), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph));
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_symbol_t dy = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, y);
	ccv_nnc_tensor_symbol_t dx = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10 * 100; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const dy_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	for (i = 0; i < 10 * 100; i++)
		dy_tensor->data.f32[i] = 0;
	for (i = 0; i < 10; i++)
		dy_tensor->data.f32[i * 100 + i] = 1;
	ccv_nnc_tensor_t* const dy16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dyt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor), TENSOR_LIST(dy16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy16_tensor), TENSOR_LIST(dyt), 0);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, TENSOR_BIND_MAP(KV(dy, dyt)), TENSOR_SYMBOL_LIST(y), SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const xt = ccv_nnc_tensor_from_symbol(tensor_arena, x);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(xt), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const dx16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const dx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dxt = ccv_nnc_tensor_from_symbol(tensor_arena, dx);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 100), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const yt = ccv_nnc_tensor_from_symbol(tensor_arena, y);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dxt), TENSOR_LIST(dx16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dx16_tensor), TENSOR_LIST(dx_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(yt), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(y_tensor), 0);
	ccv_nnc_tensor_t* const ty_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(ty_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ty_tensor->data.f32, y_tensor->data.f32, 10 * 100, 1e-3, "forward pass should match");
	ccv_nnc_tensor_t* const tdx_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 100), 0);
	ccv_nnc_cmd_exec(CMD_SWISH_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dy_tensor, x_tensor, 0), TENSOR_LIST(tdx_tensor), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdx_tensor->data.f32, dx_tensor->data.f32, 10 * 100, 1e-3, "backward pass should match");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(dx_tensor);
	ccv_nnc_tensor_free(dx16_tensor);
	ccv_nnc_tensor_free(dy_tensor);
	ccv_nnc_tensor_free(dy16_tensor);
	ccv_nnc_tensor_free(ty_tensor);
	ccv_nnc_tensor_free(tdx_tensor);
	ccv_nnc_tensor_free(dyt);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("SGD in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gm), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
}

TEST_CASE("SGD in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const m16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const n16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(g16, a16, m16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a16, m16), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gbt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gnt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt16, gnt16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, m16), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt16, gnt16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(m16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(n16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gbt16);
	ccv_nnc_tensor_free(gnt16);
}

TEST_CASE("SGD in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(g16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a, m), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(0, 0.9, 0.5, 0.999, 0.9, 0.9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
}

TEST_CASE("Nesterov SGD in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gm), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
}

TEST_CASE("Nesterov SGD in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const m16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const n16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(g16, a16, m16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a16, m16), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gbt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gnt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt16, gnt16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, m16), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt16, gnt16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(m16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(n16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gbt16);
	ccv_nnc_tensor_free(gnt16);
}

TEST_CASE("Nesterov SGD in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m), TENSOR_LIST(b, n), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(g16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a, m), TENSOR_LIST(gg, ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m), TENSOR_LIST(ga, gm), 0);
	ccv_nnc_cmd_exec(CMD_SGD_FORWARD(1, 0.9, 0.5, 0.999, 0.9, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm), TENSOR_LIST(gb, gn), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn), TENSOR_LIST(gbt, gnt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
}

TEST_CASE("rmsprop in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSPROP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

TEST_CASE("rmsprop in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSPROP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const m16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const v16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const n16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const u16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(g16, a16, m16, v16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a16, m16, v16), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gbt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gnt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gut16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, m16, v16), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(m16);
	ccv_nnc_tensor_free(v16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(n16);
	ccv_nnc_tensor_free(u16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
	ccv_nnc_tensor_free(gbt16);
	ccv_nnc_tensor_free(gnt16);
	ccv_nnc_tensor_free(gut16);
}

TEST_CASE("rmsprop in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_RMSPROP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(g16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_RMSPROP_FORWARD(0.001, 0.0001, 0.9, 0.9, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

TEST_CASE("adam in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADAM_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_TENSOR_EQ(gbt, b, "cpu result should match");
	REQUIRE_TENSOR_EQ(gnt, n, "cpu result should match");
	REQUIRE_TENSOR_EQ(gut, u, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

TEST_CASE("adam in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADAM_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const m16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const v16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const n16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const u16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(g16, a16, m16, v16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a16, m16, v16), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gbt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gnt16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_tensor_t* const gut16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16, m16, v16), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt16, gnt16, gut16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbt16, gnt16, gut16), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(m16);
	ccv_nnc_tensor_free(v16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(n16);
	ccv_nnc_tensor_free(u16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
	ccv_nnc_tensor_free(gbt16);
	ccv_nnc_tensor_free(gnt16);
	ccv_nnc_tensor_free(gut16);
}

TEST_CASE("adam in mixed precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ADAM_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const m = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const n = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const u = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const g16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10; i++)
		g->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		m->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10; i++)
		v->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, m, v), TENSOR_LIST(b, n, u), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gm = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(g16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g16, a, m, v), TENSOR_LIST(gg, ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_tensor_t* const gbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gnt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_tensor_t* const gut = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gm, gv), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gn = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_tensor_t* const gu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, m, v), TENSOR_LIST(ga, gm, gv), 0);
	ccv_nnc_cmd_exec(CMD_ADAM_FORWARD(1, 0.002, 0.9, 0.98, 0, 1e-9), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gm, gv), TENSOR_LIST(gb, gn, gu), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb, gn, gu), TENSOR_LIST(gbt, gnt, gut), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gbt->data.f32, b->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gnt->data.f32, n->data.f32, 10, 1e-3, "cpu result should match");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gut->data.f32, u->data.f32, 10, 1e-3, "cpu result should match");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(m);
	ccv_nnc_tensor_free(v);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(n);
	ccv_nnc_tensor_free(u);
	ccv_nnc_tensor_free(g16);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gm);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gn);
	ccv_nnc_tensor_free(gu);
	ccv_nnc_tensor_free(gbt);
	ccv_nnc_tensor_free(gnt);
	ccv_nnc_tensor_free(gut);
}

TEST_CASE("upsample NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BILINEAR_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BILINEAR_FORWARD(2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.forward.bin", "the forward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

TEST_CASE("upsample NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BILINEAR_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, image->rows, image->cols), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, image->rows * 2, image->cols * 2), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BILINEAR_FORWARD(2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(at), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(bt), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.forward.bin", "the forward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

TEST_CASE("downsample NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BILINEAR_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BILINEAR_BACKWARD(2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.backward.bin", "the backward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

TEST_CASE("downsample NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BILINEAR_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, image->rows, image->cols), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, image->rows / 2, image->cols / 2), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BILINEAR_BACKWARD(2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(at), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(bt), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.backward.bin", "the backward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

#include "case_main.h"
