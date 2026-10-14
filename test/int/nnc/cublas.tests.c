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

TEST_CASE("gemm no transpose with bias and palettize weights")
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
	ccv_nnc_tensor_t* const pb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(CPU_TENSOR_NHWC(32F, 2, 3), 4, 128), 0);
	(void)ccv_nnc_palettize(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 4, 128, pb->data.u8, ccv_nnc_tensor_data_size_without_padding(pb->info));
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(GPU_TENSOR_NHWC(000, 32F, 2, 3), 4, 128), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, pb, d), TENSOR_LIST(ga, gb, gd), 0);
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
	ccv_nnc_tensor_free(pb);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm no transpose with bias and row-wise int8 weights")
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
	ccv_nnc_tensor_t* const qb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 2, 3)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 3, 0, 0, qb->data.u8, ccv_nnc_tensor_data_size_without_padding(qb->info));
	ccv_nnc_tensor_t* const dq_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_dequantize_8i_rowwise(qb->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, 3, dq_b->data.u8, 6);
	float dp[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD();
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, dq_b, d), TENSOR_LIST(ct), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gqb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, 2, 3)), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, qb, d), TENSOR_LIST(ga, gqb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ct->data.f32, c->data.f32, 12, 1e-5, "result should match CPU with dequantized row-wise weights");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(qb);
	ccv_nnc_tensor_free(dq_b);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ct);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gqb);
	ccv_nnc_tensor_free(gd);
	ccv_nnc_tensor_free(gc);
}

static void _cublas_scaled_gemm_to_float(const int datatype, const void* const data, const size_t count, float* const values)
{
	if (datatype == CCV_16F)
		ccv_half_precision_to_float((const uint16_t*)data, values, count);
	else if (datatype == CCV_16BF)
		ccv_bfloat_to_float((const uint16_t*)data, values, count);
	else
		memcpy(values, data, sizeof(float) * count);
}

static void _cublas_scaled_gemm_from_float(const int datatype, const float* const values, const size_t count, void* const data)
{
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, count);
	else
		memcpy(data, values, sizeof(float) * count);
}

// Rounds values to the nearest datatype value (ties to even), as the GPU conversions do. The ccv conversions
// truncate, thus pick the closer of the truncated value and its neighbor away from zero.
static void _cublas_scaled_gemm_round(const int datatype, float* const values, const size_t count)
{
	if (datatype != CCV_16F && datatype != CCV_16BF)
		return;
	size_t i;
	for (i = 0; i < count; i++)
	{
		uint16_t h[2];
		_cublas_scaled_gemm_from_float(datatype, values + i, 1, h);
		h[1] = h[0] + 1;
		float f[2];
		_cublas_scaled_gemm_to_float(datatype, h, 2, f);
		const float d0 = fabsf(values[i] - f[0]);
		const float d1 = fabsf(f[1] - values[i]);
		values[i] = (d1 < d0 || (d1 == d0 && (h[1] & 1) == 0)) ? f[1] : f[0];
	}
}

// Multiplies every 256 consecutive values by R256 = H4^tensor4 / 16, where H4 is +1 except -1 on its anti-diagonal,
// then rounds them to the datatype like the GPU rotation does.
static void _cublas_scaled_gemm_hadamard_256(const int datatype, float* const values, const size_t count)
{
	size_t i;
	int j, stride;
	double v[256];
	for (i = 0; i < count; i += 256)
	{
		for (j = 0; j < 256; j++)
			v[j] = values[i + j];
		for (stride = 1; stride < 256; stride *= 4)
			for (j = 0; j < 256; j++)
				if ((j / stride) % 4 == 0)
				{
					const double a = v[j], b = v[j + stride], c = v[j + stride * 2], d = v[j + stride * 3];
					const double sum = a + b + c + d;
					v[j] = sum - 2 * d;
					v[j + stride] = sum - 2 * c;
					v[j + stride * 2] = sum - 2 * b;
					v[j + stride * 3] = sum - 2 * a;
				}
		for (j = 0; j < 256; j++)
			values[i + j] = (float)(v[j] / 16);
	}
	_cublas_scaled_gemm_round(datatype, values, count);
}

// Runs a [batch, m, k] x w[n, k]^T (+ bias[n]) on the GPU with row-wise int8 weights, and returns the largest
// difference to a double precision CPU reference, relative to max(1, |reference|, |actual|). On the row-wise int8
// path with more than 4 rows in total, the GPU quantizes the activations row-wise to int8 (scale = max |a| / 127),
// which the reference repeats; otherwise it multiplies the activations as they are. A non-zero format packs the
// weights as that row-wise-x codec instead; with CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 the payload holds the codes of
// W R, which the GPU multiplies with the rotated activations A R, or rotates back. fast_path_out reports whether the row-wise int8
// path ran, rather than the dequantizing fallback.
static double _cublas_scaled_gemm_max_rel_diff(const int datatype, const int format, const int use_bias, const int batch, const int m, const int n, const int k, int* const fast_path_out)
{
	const int rows = batch * m;
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, batch, m, k);
	if (batch == 1)
		a_params = (ccv_nnc_tensor_param_t)CPU_TENSOR_NHWC(32F, m, k);
	a_params.datatype = datatype;
	ccv_nnc_tensor_param_t w_params = CPU_TENSOR_NHWC(32F, n, k);
	w_params.datatype = datatype;
	ccv_nnc_tensor_param_t bias_params = CPU_TENSOR_NHWC(32F, n);
	bias_params.datatype = datatype;
	ccv_nnc_tensor_param_t b_params = a_params;
	b_params.dim[ccv_nnc_tensor_nd(a_params.dim) - 1] = n;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, w_params, 0);
	const int hadamard_256 = !!(format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, format ? ccv_nnc_tensor_8i_rowwise_x(w_params, format) : ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	float* const a_f = (float*)ccmalloc(sizeof(float) * rows * k);
	float* const w_f = (float*)ccmalloc(sizeof(float) * n * k);
	float* const bias_f = (float*)ccmalloc(sizeof(float) * n);
	float* const b_f = (float*)ccmalloc(sizeof(float) * rows * n);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i, j, l;
	// Integer activations keep the rotation exact in float, thus the GPU and the reference round it the same way.
	for (i = 0; i < rows * k; i++)
		a_f[i] = hadamard_256 ? (float)(floor(dsfmt_genrand_open_close(&dsfmt) * 129) - 64) : (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	// Vary the activation row magnitudes so that the row-wise scales differ.
	for (i = 0; i < rows; i++)
		for (j = 0; j < k; j++)
			a_f[i * k + j] *= 0.25 + (i % 7);
	for (i = 0; i < n * k; i++)
		w_f[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 0.1;
	for (i = 0; i < n; i++)
		bias_f[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	_cublas_scaled_gemm_from_float(datatype, a_f, rows * k, ha->data.u8);
	_cublas_scaled_gemm_from_float(datatype, w_f, n * k, hw->data.u8);
	if (use_bias)
		_cublas_scaled_gemm_from_float(datatype, bias_f, n, hbias->data.u8);
	const size_t qsize = format ?
		ccv_nnc_quantize_8i_rowwise_x(hw->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n * k, k, format, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info)) :
		ccv_nnc_quantize_8i_rowwise(hw->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n * k, k, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	// The reference reads back what the GPU sees: rounded activations and bias, and the int8 weights times
	// their row scales in float, which only the dequantizing fallback rounds into the datatype again.
	_cublas_scaled_gemm_to_float(datatype, ha->data.u8, rows * k, a_f);
	if (use_bias)
		_cublas_scaled_gemm_to_float(datatype, hbias->data.u8, n, bias_f);
	// A short payload fails the comparison below.
	const int qsize_ok = qsize == ccv_nnc_tensor_data_size_without_padding(hwq->info);
	ccv_nnc_tensor_param_t ga_params = a_params;
	ga_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gw_params = w_params;
	gw_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gbias_params = bias_params;
	gbias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gb_params = b_params;
	gb_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const gwq = ccv_nnc_tensor_new(0, format ? ccv_nnc_tensor_8i_rowwise_x(gw_params, format) : ccv_nnc_tensor_8i_rowwise(gw_params), 0);
	ccv_nnc_tensor_t* const gbias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
#ifdef HAVE_CUDA
	// The reference follows the path the GPU takes. Without the row-wise int8 kernels (no SM80 support, or an older
	// GPU), the weights are dequantized into the datatype and the activations multiplied as they are.
	extern int ccv_nnc_compat_8i_rowwise_gemm_supported(const int datatype, const void* const a, const void* const w, const void* const bias, const void* const b, const int m, const int n, const int k, size_t* const workspace_size);
	size_t workspace_size;
	const int fast_path = ccv_nnc_compat_8i_rowwise_gemm_supported(datatype, ga->data.u8, gwq->data.u8, gbias ? gbias->data.u8 : 0, gb->data.u8, rows, n, k, &workspace_size);
#else
	const int fast_path = 0;
#endif
	*fast_path_out = fast_path;
	// Both payloads end with one scale per row.
	const size_t scale_offset = ccv_nnc_tensor_data_size_without_padding(hwq->info) - (size_t)n * CCV_GET_DATA_TYPE_SIZE(datatype);
	float* const w_scale = (float*)ccmalloc(sizeof(float) * n);
	_cublas_scaled_gemm_to_float(datatype, hwq->data.u8 + scale_offset, n, w_scale);
	if (format)
	{
		// The packed bits do not depend on the datatype, thus decoding them with FP32 scales of 1 yields the codes.
		// Without H256 in the format, the decode keeps the stored codes of W R.
		unsigned char* const codes = (unsigned char*)ccmalloc(scale_offset + sizeof(float) * n);
		memcpy(codes, hwq->data.u8, scale_offset);
		for (i = 0; i < n; i++)
			((float*)(codes + scale_offset))[i] = 1;
		ccv_nnc_dequantize_8i_rowwise_x(codes, CCV_32F, CCV_TENSOR_CPU_MEMORY, scale_offset + sizeof(float) * n, k, format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK, w_f, (size_t)n * k);
		ccfree(codes);
		for (i = 0; i < n; i++)
			for (j = 0; j < k; j++)
				w_f[i * k + j] *= w_scale[i];
	} else {
		const int8_t* const w_q = (const int8_t*)hwq->data.u8;
		for (i = 0; i < n; i++)
			for (j = 0; j < k; j++)
				w_f[i * k + j] = w_q[i * k + j] * w_scale[i];
	}
	ccfree(w_scale);
	if (hadamard_256)
	{
		// A W^T = (A R) Wq^T. The row-wise int8 path rotates the activations. The dequantizing fallback does so
		// only for the IQ codecs with few activation rows, and otherwise undoes the rotation in the weights.
		const int codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
		const int rotate_a = fast_path || (codec >= CCV_NNC_QX_8I_ROWWISE_IQ2_S && codec <= CCV_NNC_QX_8I_ROWWISE_IQ2_XXS && (size_t)rows * 8 <= n);
		if (rotate_a)
			_cublas_scaled_gemm_hadamard_256(datatype, a_f, (size_t)rows * k);
		else
			_cublas_scaled_gemm_hadamard_256(datatype, w_f, (size_t)n * k);
	}
	if (!fast_path)
		_cublas_scaled_gemm_round(datatype, w_f, n * k);
	if (fast_path && rows > 4)
		for (i = 0; i < rows; i++)
		{
			float max_abs = 0;
			for (j = 0; j < k; j++)
				max_abs = ccv_max(max_abs, fabsf(a_f[i * k + j]));
			const float scale = max_abs > 0 ? max_abs / 127.f : 1.f / 127.f;
			const float inv_scale = max_abs > 0 ? 127.f / max_abs : 127.f;
			for (j = 0; j < k; j++)
				a_f[i * k + j] = ccv_clamp(rintf(a_f[i * k + j] * inv_scale), -127, 127) * scale;
		}
	if (use_bias)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hbias), TENSOR_LIST(ga, gwq, gbias), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gwq, gbias), TENSOR_LIST(gb), 0);
	} else {
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq), TENSOR_LIST(ga, gwq), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gwq), TENSOR_LIST(gb), 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
	_cublas_scaled_gemm_to_float(datatype, hb->data.u8, rows * n, b_f);
	double max_rel = qsize_ok ? 0 : HUGE_VAL;
	for (i = 0; i < rows; i++)
		for (j = 0; j < n; j++)
		{
			double expected = use_bias ? bias_f[j] : 0;
			for (l = 0; l < k; l++)
				expected += (double)a_f[i * k + l] * w_f[j * k + l];
			const double actual = b_f[i * n + j];
			const double rel = fabs(actual - expected) / ccv_max(1.0, ccv_max(fabs(actual), fabs(expected)));
			max_rel = ccv_max(max_rel, rel);
		}
	ccfree(a_f);
	ccfree(w_f);
	ccfree(bias_f);
	ccfree(b_f);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gwq);
	if (gbias)
		ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gb);
	return max_rel;
}

TEST_CASE("cublas forward gemm with row-wise int8 weights matches CPU across precisions and shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int datatypes[] = { CCV_32F, CCV_16F, CCV_16BF };
	// The reference repeats the activation quantization exactly, thus the output precision bounds the error.
	const double tolerances[] = { 1e-4, 1e-3, 5e-3 };
	// The dequantizing fallback runs cuBLAS, which may reduce split-k partial sums in the output datatype.
	const double fallback_tolerances[] = { 1e-4, 2e-2, 5e-2 };
	// Up to 4 rows multiply the activations as they are, more rows multiply int8 activation codes.
	const int shapes[][3] = {
		{ 1, 384, 256 },
		{ 3, 384, 256 },
		{ 4, 1024, 2048 },
		{ 5, 384, 256 },
		{ 37, 1024, 2048 },
		{ 257, 384, 256 },
		{ 512, 8192, 1024 },
		{ 1100, 512, 768 },
	};
	int d, s, use_bias;
	for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
		for (s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
			for (use_bias = 0; use_bias < 2; use_bias++)
			{
				int fast_path;
				const double max_rel = _cublas_scaled_gemm_max_rel_diff(datatypes[d], 0, use_bias, 1, shapes[s][0], shapes[s][1], shapes[s][2], &fast_path);
				const double tolerance = fast_path ? tolerances[d] : fallback_tolerances[d];
				if (max_rel > tolerance)
					printf("datatype %x, bias %d, m %d, n %d, k %d, fast path %d: max relative difference %g\n", datatypes[d], use_bias, shapes[s][0], shapes[s][1], shapes[s][2], fast_path, max_rel);
				REQUIRE(max_rel <= tolerance, "row-wise int8 GEMM should match the CPU reference");
			}
}

TEST_CASE("cublas forward batched gemm with row-wise int8 weights matches CPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	int fast_path;
	// The weights are shared across the batch, thus 2 x 2 rows still multiply the activations as they are.
	double max_rel = _cublas_scaled_gemm_max_rel_diff(CCV_16F, 0, 1, 2, 2, 384, 256, &fast_path);
	REQUIRE(max_rel <= (fast_path ? 1e-3 : 2e-2), "batched row-wise int8 GEMV should match the CPU reference");
	max_rel = _cublas_scaled_gemm_max_rel_diff(CCV_16F, 0, 1, 3, 37, 384, 256, &fast_path);
	REQUIRE(max_rel <= (fast_path ? 1e-3 : 2e-2), "batched row-wise int8 GEMM should match the CPU reference");
	max_rel = _cublas_scaled_gemm_max_rel_diff(CCV_32F, 0, 0, 4, 65, 512, 512, &fast_path);
	REQUIRE(max_rel <= 1e-4, "batched row-wise int8 GEMM should match the CPU reference");
}

TEST_CASE("cublas forward gemm with row-wise-x int8 weights matches CPU across codecs, H256 and precisions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	const int datatypes[] = { CCV_32F, CCV_16F, CCV_16BF };
	const double tolerances[] = { 1e-4, 1e-3, 5e-3 };
	const double fallback_tolerances[] = { 1e-4, 2e-2, 5e-2 };
	// 3 rows multiply the activations as they are, 37 and 257 rows multiply int8 activation codes. Some CPU
	// quantizers are slow, thus every codec runs each row count once and rotates the datatypes across them.
	const int ms[] = { 3, 37, 257 };
	int f, h, i;
	for (f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
		for (h = 0; h < 2; h++)
			for (i = 0; i < sizeof(ms) / sizeof(ms[0]); i++)
			{
				const int format = formats[f] | (h ? CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 : 0);
				const int d = (f + h + i) % 3;
				const int use_bias = (f + i) % 2;
				int fast_path;
				const double max_rel = _cublas_scaled_gemm_max_rel_diff(datatypes[d], format, use_bias, 1, ms[i], 256, 256, &fast_path);
				const double tolerance = fast_path ? tolerances[d] : fallback_tolerances[d];
				if (max_rel > tolerance)
					printf("format %x, datatype %x, bias %d, m %d, fast path %d: max relative difference %g\n", format, datatypes[d], use_bias, ms[i], fast_path, max_rel);
				REQUIRE(max_rel <= tolerance, "row-wise-x int8 GEMM should match the CPU reference");
			}
}

TEST_CASE("gemm no transpose with packed row-wise int8-x weights in all formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int m = 17;
	const int k = 65;
	const int n = 128;
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, k, n), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < m * k; i++)
		a->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	for (i = 0; i < k * n; i++)
		b->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, m, k), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ga), 0);
	int f;
	for (f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
	{
		const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, k, n), formats[f]);
		ccv_nnc_tensor_t* const qb = ccv_nnc_tensor_new(0, q_params, 0);
		const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, (size_t)k * n, n, formats[f], 0, 0, qb->data.u8, ccv_nnc_tensor_data_size_without_padding(qb->info));
		REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(qb->info), "packed row-wise int8-x size should match");
		ccv_nnc_tensor_t* const dq_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, k, n), 0);
		ccv_nnc_dequantize_8i_rowwise_x(qb->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, n, formats[f], dq_b->data.u8, (size_t)k * n);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
		ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD();
		cmd.backend = CCV_NNC_BACKEND_CPU_REF;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, dq_b), TENSOR_LIST(expected), 0);
		ccv_nnc_tensor_param_t gpu_q_params = q_params;
		gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_t* const gqb = ccv_nnc_tensor_new(0, gpu_q_params, 0);
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, m, n), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(qb), TENSOR_LIST(gqb), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqb), TENSOR_LIST(gout), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, m * n, 1e-3, "result should match CPU with packed row-wise int8-x weights");
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gqb);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(dq_b);
		ccv_nnc_tensor_free(qb);
	}
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(a);
}

static void _ccv_nnc_h256_fill(const int datatype, void* const data, const size_t count, dsfmt_t* const dsfmt)
{
	size_t i;
	float* const f32 = (float*)ccmalloc(sizeof(float) * count);
	for (i = 0; i < count; i++)
		f32[i] = (float)(dsfmt_genrand_open_close(dsfmt) * 2 - 1);
	if (datatype == CCV_32F)
		memcpy(data, f32, sizeof(float) * count);
	else if (datatype == CCV_16F)
		ccv_float_to_half_precision(f32, (uint16_t*)data, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(f32, (uint16_t*)data, count);
	else {
		for (i = 0; i < count; i++)
			((double*)data)[i] = f32[i];
	}
	ccfree(f32);
}

static double _ccv_nnc_h256_value(const int datatype, const void* const data, const size_t i)
{
	if (datatype == CCV_32F)
		return ((const float*)data)[i];
	if (datatype == CCV_64F)
		return ((const double*)data)[i];
	float f;
	if (datatype == CCV_16F)
		ccv_half_precision_to_float((uint16_t*)data + i, &f, 1);
	else
		ccv_bfloat_to_float((uint16_t*)data + i, &f, 1);
	return f;
}

TEST_CASE("H256 row-wise int8-x dequantization matches CPU across codecs and precisions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF, CCV_64F};
	const double tolerances[] = {1e-6, 1e-3, 1e-2, 1e-12};
	// Two block-aligned shapes and one wide row to cover a partial last CTA.
	const int rows[] = {13, 29, 1};
	const int cols[] = {256, 768, 65792};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 256);
	int d, codec, shape;
	for (d = 0; d < 4; d++)
		for (codec = CCV_NNC_QX_8I_ROWWISE_Q4_K; codec <= CCV_NNC_QX_8I_ROWWISE_Q6_K; codec++)
			for (shape = 0; shape < 3; shape++)
			{
				if (shape == 2 && (d != 0 || codec != CCV_NNC_QX_8I_ROWWISE_Q6_K))
					continue;
				const int format = codec | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
				const size_t count = (size_t)rows[shape] * cols[shape];
				ccv_nnc_tensor_param_t host_params = CPU_TENSOR_NHWC(32F, rows[shape], cols[shape]);
				host_params.datatype = datatypes[d];
				ccv_nnc_tensor_param_t gpu_params = host_params;
				gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
				ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, host_params, 0);
				ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
				ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_params, 0);
				ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, gpu_params, 0);
				ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_params, 0);
				_ccv_nnc_h256_fill(datatypes[d], source->data.u8, count, &dsfmt);
				const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols[shape], format, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
				REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "H256 quantization size should match");
				ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, qsize, cols[shape], format, expected->data.u8, count);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
				ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], CCV_TENSOR_GPU_MEMORY, qsize, cols[shape], format, gout->data.u8, count);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
				double max_abs = 0;
				size_t i;
				for (i = 0; i < count; i++)
					max_abs = ccv_max(max_abs, fabs(_ccv_nnc_h256_value(datatypes[d], expected->data.u8, i) - _ccv_nnc_h256_value(datatypes[d], actual->data.u8, i)));
				REQUIRE(max_abs <= tolerances[d], "H256 logical dequantization should match CPU: codec=%d datatype=%d shape=%d max_abs=%g", codec, datatypes[d], shape, max_abs);
				ccv_nnc_tensor_free(actual);
				ccv_nnc_tensor_free(gout);
				ccv_nnc_tensor_free(expected);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(q);
				ccv_nnc_tensor_free(source);
			}
}

TEST_CASE("gemm with H256 row-wise int8-x weights matches CPU on logical weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int k = 768;
	// For IQ codecs, 8M <= N rotates the activations; otherwise the decoded weights are rotated. N is odd, thus the
	// row-wise int8 GEMM (which stores 16 bytes at a time) declines and these dequantizing routes run.
	const int ms[] = {8, 160};
	const int ns[] = {97, 65};
	const int datatypes[] = {CCV_32F, CCV_16F};
	const int codecs[] = {
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 257);
	int d, shape, f;
	for (d = 0; d < 2; d++)
		for (shape = 0; shape < 2; shape++)
			for (f = 0; f < sizeof(codecs) / sizeof(codecs[0]); f++)
			{
				const int datatype = datatypes[d];
				const int m = ms[shape];
				const int n = ns[shape];
				const int format = codecs[f] | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
				ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, m, k);
				ccv_nnc_tensor_param_t w_params = CPU_TENSOR_NHWC(32F, n, k);
				ccv_nnc_tensor_param_t bias_params = CPU_TENSOR_NHWC(32F, n);
				ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, m, n);
				a_params.datatype = w_params.datatype = bias_params.datatype = b_params.datatype = datatype;
				ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
				ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, w_params, 0);
				ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, bias_params, 0);
				ccv_nnc_tensor_t* const qw = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(w_params, format), 0);
				_ccv_nnc_h256_fill(datatype, a->data.u8, m * k, &dsfmt);
				_ccv_nnc_h256_fill(datatype, w->data.u8, n * k, &dsfmt);
				_ccv_nnc_h256_fill(datatype, bias->data.u8, n, &dsfmt);
				const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(w->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)n * k, k, format, 0, 0, qw->data.u8, ccv_nnc_tensor_data_size_without_padding(qw->info));
				REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(qw->info), "H256 quantization size should match");
				// Reference on the logical (de-rotated) weights in 32F.
				ccv_nnc_dequantize_8i_rowwise_x(qw->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, k, format, w->data.u8, (size_t)n * k);
				ccv_nnc_tensor_t* const fa = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
				ccv_nnc_tensor_t* const fw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
				ccv_nnc_tensor_t* const fbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n), 0);
				ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
				int i;
				for (i = 0; i < m * k; i++)
					fa->data.f32[i] = _ccv_nnc_h256_value(datatype, a->data.u8, i);
				for (i = 0; i < n * k; i++)
					fw->data.f32[i] = _ccv_nnc_h256_value(datatype, w->data.u8, i);
				for (i = 0; i < n; i++)
					fbias->data.f32[i] = _ccv_nnc_h256_value(datatype, bias->data.u8, i);
				ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
				cmd.backend = CCV_NNC_BACKEND_CPU_REF;
				ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(fa, fw, fbias), TENSOR_LIST(expected), 0);
				ccv_nnc_tensor_param_t ga_params = a_params, gbias_params = bias_params, gb_params = b_params;
				ga_params.type = gbias_params.type = gb_params.type = CCV_TENSOR_GPU_MEMORY | 000;
				ccv_nnc_tensor_param_t gqw_params = qw->info;
				gqw_params.type = CCV_TENSOR_GPU_MEMORY | 000;
				ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
				ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, gbias_params, 0);
				ccv_nnc_tensor_t* const gqw = ccv_nnc_tensor_new(0, gqw_params, 0);
				ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
				ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
#ifdef HAVE_CUDA
				extern int ccv_nnc_compat_8i_rowwise_gemm_supported(const int datatype, const void* const a, const void* const w, const void* const bias, const void* const b, const int m, const int n, const int k, size_t* const workspace_size);
				size_t workspace_size;
				REQUIRE(!ccv_nnc_compat_8i_rowwise_gemm_supported(datatype, ga->data.u8, 0, gbias->data.u8, gb->data.u8, m, n, k, &workspace_size), "the row-wise int8 GEMM should decline this shape");
#endif
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, bias, qw), TENSOR_LIST(ga, gbias, gqw), 0);
				ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqw, gbias), TENSOR_LIST(gb), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
				double max_rel = 0;
				for (i = 0; i < m * n; i++)
				{
					const double x = expected->data.f32[i];
					const double y = _ccv_nnc_h256_value(datatype, b->data.u8, i);
					max_rel = ccv_max(max_rel, fabs(x - y) / ccv_max(ccv_max(fabs(x), fabs(y)), 1));
				}
				// FP16 tolerance covers the cuBLAS half GEMM itself (both routes land around 1e-2).
				REQUIRE(max_rel <= (datatype == CCV_32F ? 1e-4 : 2e-2), "result should match CPU on logical weights: datatype=%d m=%d n=%d codec=%d max_rel=%g", datatype, m, n, codecs[f], max_rel);
				ccv_nnc_tensor_free(b);
				ccv_nnc_tensor_free(gb);
				ccv_nnc_tensor_free(gqw);
				ccv_nnc_tensor_free(gbias);
				ccv_nnc_tensor_free(ga);
				ccv_nnc_tensor_free(expected);
				ccv_nnc_tensor_free(fbias);
				ccv_nnc_tensor_free(fw);
				ccv_nnc_tensor_free(fa);
				ccv_nnc_tensor_free(qw);
				ccv_nnc_tensor_free(bias);
				ccv_nnc_tensor_free(w);
				ccv_nnc_tensor_free(a);
			}
}

TEST_CASE("gemm with very tall H256 row-wise int8-x weights rotates activations in separate launches")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// 65536 weight rows exceed the fused launch's grid, so decode and activation
	// rotation run as two launches. With K = 256, an IQ2_XXS row is 64 payload bytes,
	// so tile one quantized 256 x 256 block instead of quantizing every row.
	const int k = 256;
	const int n = 65536;
	const int block_rows = 256;
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 258);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, block_rows, k), 0);
	_ccv_nnc_h256_fill(CCV_32F, w->data.u8, block_rows * k, &dsfmt);
	ccv_nnc_tensor_t* const qs = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, block_rows, k), format), 0);
	const size_t qs_size = ccv_nnc_quantize_8i_rowwise_x(w->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, block_rows * k, k, format, 0, 0, qs->data.u8, ccv_nnc_tensor_data_size_without_padding(qs->info));
	REQUIRE_EQ(qs_size, ccv_nnc_tensor_data_size_without_padding(qs->info), "H256 quantization size should match");
	const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, n, k), format);
	ccv_nnc_tensor_t* const qw = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t row_bytes = 64;
	const size_t qs_scale_offset = ((size_t)block_rows * row_bytes + 127) & ~(size_t)127;
	const size_t qw_scale_offset = ((size_t)n * row_bytes + 127) & ~(size_t)127;
	REQUIRE_EQ(qw_scale_offset + sizeof(float) * n, ccv_nnc_tensor_data_size_without_padding(q_params), "IQ2_XXS rows should be 64 bytes");
	int i;
	for (i = 0; i < n; i++)
	{
		memcpy(qw->data.u8 + i * row_bytes, qs->data.u8 + (i % block_rows) * row_bytes, row_bytes);
		((float*)(qw->data.u8 + qw_scale_offset))[i] = ((float*)(qs->data.u8 + qs_scale_offset))[i % block_rows];
	}
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, k), 0);
	_ccv_nnc_h256_fill(CCV_32F, a->data.u8, k, &dsfmt);
	ccv_nnc_tensor_t* const dq_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_dequantize_8i_rowwise_x(qw->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, ccv_nnc_tensor_data_size_without_padding(q_params), k, format, dq_w->data.u8, (size_t)n * k);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, dq_w), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_param_t gq_params = q_params;
	gq_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, k), 0);
	ccv_nnc_tensor_t* const gqw = ccv_nnc_tensor_new(0, gq_params, 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, n), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, n), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, qw), TENSOR_LIST(ga, gqw), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqw), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, b->data.f32, n, 1e-3, "result should match CPU on logical weights");
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gqw);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(dq_w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(qw);
	ccv_nnc_tensor_free(qs);
	ccv_nnc_tensor_free(w);
}

TEST_CASE("gemm with H256 row-wise int8-x weights handles an unaligned activation view")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// A contiguous 2 x 256 view starting one element into its buffer still rotates the
	// activations for IQ2_XXS (8M <= N), through the element-wise load path.
	const int m = 2;
	const int k = 256;
	const int n = 16;
	const int datatypes[] = {CCV_32F, CCV_16F};
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 259);
	int d;
	for (d = 0; d < 2; d++)
	{
		const int datatype = datatypes[d];
		ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, m, k);
		ccv_nnc_tensor_param_t w_params = CPU_TENSOR_NHWC(32F, n, k);
		ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, m, n);
		a_params.datatype = w_params.datatype = b_params.datatype = datatype;
		ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
		ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, w_params, 0);
		_ccv_nnc_h256_fill(datatype, a->data.u8, m * k, &dsfmt);
		_ccv_nnc_h256_fill(datatype, w->data.u8, n * k, &dsfmt);
		const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(w_params, format);
		ccv_nnc_tensor_t* const qw = ccv_nnc_tensor_new(0, q_params, 0);
		const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(w->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n * k, k, format, 0, 0, qw->data.u8, ccv_nnc_tensor_data_size_without_padding(q_params));
		REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q_params), "H256 quantization size should match");
		ccv_nnc_dequantize_8i_rowwise_x(qw->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, k, format, w->data.u8, n * k);
		// Reference on the logical (de-rotated) weights in 32F.
		ccv_nnc_tensor_t* const fa = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
		ccv_nnc_tensor_t* const fw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
		int i;
		for (i = 0; i < m * k; i++)
			fa->data.f32[i] = _ccv_nnc_h256_value(datatype, a->data.u8, i);
		for (i = 0; i < n * k; i++)
			fw->data.f32[i] = _ccv_nnc_h256_value(datatype, w->data.u8, i);
		ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
		cmd.backend = CCV_NNC_BACKEND_CPU_REF;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(fa, fw), TENSOR_LIST(expected), 0);
		ccv_nnc_tensor_param_t ga_buffer_params = CPU_TENSOR_NHWC(32F, m * k + 8);
		ccv_nnc_tensor_param_t ga_params = a_params, gb_params = b_params, gq_params = q_params;
		ga_buffer_params.datatype = datatype;
		ga_buffer_params.type = ga_params.type = gb_params.type = gq_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_t* const ga_buffer = ccv_nnc_tensor_new(0, ga_buffer_params, 0);
		ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(ga_buffer->data.u8 + CCV_GET_DATA_TYPE_SIZE(datatype), ga_params, 0);
		ccv_nnc_tensor_t* const gqw = ccv_nnc_tensor_new(0, gq_params, 0);
		ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
		ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, qw), TENSOR_LIST(ga, gqw), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqw), TENSOR_LIST(gb), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
		double max_rel = 0;
		for (i = 0; i < m * n; i++)
		{
			const double x = expected->data.f32[i];
			const double y = _ccv_nnc_h256_value(datatype, b->data.u8, i);
			max_rel = ccv_max(max_rel, fabs(x - y) / ccv_max(ccv_max(fabs(x), fabs(y)), 1));
		}
		REQUIRE(max_rel <= (datatype == CCV_32F ? 1e-4 : 2e-2), "result should match CPU on logical weights: datatype=%d max_rel=%g", datatype, max_rel);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(gb);
		ccv_nnc_tensor_free(gqw);
		ccv_nnc_tensor_free(ga);
		ccv_nnc_tensor_free(ga_buffer);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(fw);
		ccv_nnc_tensor_free(fa);
		ccv_nnc_tensor_free(qw);
		ccv_nnc_tensor_free(w);
		ccv_nnc_tensor_free(a);
	}
}

TEST_CASE("gemm with H256 row-wise IQ int8-x weights matches CPU on every GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// With 8M > N the IQ weights are decoded with R undone by the persistent kernel,
	// whose grid is sized from the current device's SM count. Run on each GPU in turn
	// from the same thread. N is odd, thus the row-wise int8 GEMM (which stores 16 bytes
	// at a time) declines and this dequantizing route runs.
	const int m = 64;
	const int n = 63;
	const int k = 512;
	const int format = CCV_NNC_QX_8I_ROWWISE_IQ2_XXS | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 260);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	_ccv_nnc_h256_fill(CCV_32F, a->data.u8, m * k, &dsfmt);
	_ccv_nnc_h256_fill(CCV_32F, w->data.u8, n * k, &dsfmt);
	const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, n, k), format);
	ccv_nnc_tensor_t* const qw = ccv_nnc_tensor_new(0, q_params, 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(w->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, n * k, k, format, 0, 0, qw->data.u8, ccv_nnc_tensor_data_size_without_padding(q_params));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q_params), "H256 quantization size should match");
	ccv_nnc_dequantize_8i_rowwise_x(qw->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, k, format, w->data.u8, n * k);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(expected), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	int device;
	for (device = 0; device < device_count; device++)
	{
		ccv_nnc_tensor_param_t ga_params = CPU_TENSOR_NHWC(32F, m, k);
		ccv_nnc_tensor_param_t gb_params = CPU_TENSOR_NHWC(32F, m, n);
		ccv_nnc_tensor_param_t gq_params = q_params;
		ga_params.type = gb_params.type = gq_params.type = CCV_TENSOR_GPU_MEMORY;
		CCV_TENSOR_SET_DEVICE_ID(ga_params.type, device);
		CCV_TENSOR_SET_DEVICE_ID(gb_params.type, device);
		CCV_TENSOR_SET_DEVICE_ID(gq_params.type, device);
		ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
		ccv_nnc_tensor_t* const gqw = ccv_nnc_tensor_new(0, gq_params, 0);
		ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
#ifdef HAVE_CUDA
		extern int ccv_nnc_compat_8i_rowwise_gemm_supported(const int datatype, const void* const a, const void* const w, const void* const bias, const void* const b, const int m, const int n, const int k, size_t* const workspace_size);
		size_t workspace_size;
		REQUIRE(!ccv_nnc_compat_8i_rowwise_gemm_supported(CCV_32F, ga->data.u8, 0, 0, gb->data.u8, m, n, k, &workspace_size), "the row-wise int8 GEMM should decline this shape");
#endif
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, qw), TENSOR_LIST(ga, gqw), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gqw), TENSOR_LIST(gb), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, b->data.f32, m * n, 1e-3, "result should match CPU on logical weights on device %d", device);
		ccv_nnc_tensor_free(gb);
		ccv_nnc_tensor_free(gqw);
		ccv_nnc_tensor_free(ga);
	}
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(qw);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(a);
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

TEST_CASE("backward gemm with no transpose and palettize weights")
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
	ccv_nnc_tensor_t* const pb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(CPU_TENSOR_NHWC(32F, 2, 3), 4, 128), 0);
	(void)ccv_nnc_palettize(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 4, 128, pb->data.u8, ccv_nnc_tensor_data_size_without_padding(pb->info));
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_palettize(GPU_TENSOR_NHWC(000, 32F, 2, 3), 4, 128), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, pb), TENSOR_LIST(gg, ga, gb), 0);
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
	ccv_nnc_tensor_free(pb);
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

TEST_CASE("backward gemm with no transpose and row-wise int8 weights")
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
	ccv_nnc_tensor_t* const qb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, 2, 3)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(b->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, 6, 3, 0, 0, qb->data.u8, ccv_nnc_tensor_data_size_without_padding(qb->info));
	ccv_nnc_tensor_t* const dq_b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_dequantize_8i_rowwise(qb->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, 3, dq_b->data.u8, 6);
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const dbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbiast = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_BACKWARD();
	cmd.backend = CCV_NNC_BACKEND_CPU_REF;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, dq_b), TENSOR_LIST(ht, dbt, dbiast), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gqb = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, 2, 3)), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, qb), TENSOR_LIST(gg, ga, gqb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gqb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, ht->data.f32, h->data.f32, 8, 1e-5, "h should match CPU with dequantized row-wise weights");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbt->data.f32, db->data.f32, 6, 1e-5, "db should match CPU with dequantized row-wise weights");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dbiast->data.f32, dbias->data.f32, 3, 1e-5, "dbias should match CPU with dequantized row-wise weights");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(qb);
	ccv_nnc_tensor_free(dq_b);
	ccv_nnc_tensor_free(ht);
	ccv_nnc_tensor_free(dbt);
	ccv_nnc_tensor_free(dbiast);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gqb);
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("cublas forward gemm in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 64), 0);

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
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("cublas forward gemv in half precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
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

TEST_CASE("cublas forward gemv in bfloat precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("cublas forward gemv in half precision no bias, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
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

TEST_CASE("cublas forward gemv in half precision no bias, variant 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 128, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 1), 0);

	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 128, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(hw, ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(w, a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
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
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias1->data.f32, hdbias->data.f32, 64, 1e-2, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("cublas handle permute")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("generalized batched gemm with batch (2, 4) compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("generalized batched gemm with batch (2, 4) and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("generalized batched gemm with batch (2, 4) with bias compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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

TEST_CASE("generalized batched gemm with batch (2, 4) with bias and broadcast compare cublas")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
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

static ccv_nnc_tensor_param_t _ccv_nnc_ew_cpu_nchw_datatype(const int datatype)
{
	ccv_nnc_tensor_param_t params = CPU_TENSOR_NCHW(32F, 10, 100);
	params.datatype = datatype;
	return params;
}

static ccv_nnc_tensor_param_t _ccv_nnc_ew_gpu_nchw_datatype(const int datatype)
{
	ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, 32F, 10, 100);
	params.datatype = datatype;
	return params;
}

static void _ccv_nnc_ewexp_gpu_ref_datatype(const int datatype, float* const forward_max_diff, float* const backward_max_diff)
{
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hg32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const har = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hgr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hb32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hbr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hda = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hda32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const dat32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const datr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 6 - 3;
	for (i = 0; i < 1000; i++)
		hg32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha32, hg32), TENSOR_LIST(ha, hg), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(har, hgr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(har), TENSOR_LIST(hb32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb32), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(hbr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(hb32), 0);
	*forward_max_diff = 0;
	for (i = 0; i < 1000; i++)
	{
		const float max_val = ccv_max(ccv_max(fabsf(hbr->data.f32[i]), fabsf(hb32->data.f32[i])), 1);
		*forward_max_diff = ccv_max(*forward_max_diff, fabsf(hbr->data.f32[i] - hb32->data.f32[i]) / max_val);
	}
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgr, 0, hbr), TENSOR_LIST(dat32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dat32), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(datr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hda), TENSOR_LIST(hda32), 0);
	*backward_max_diff = 0;
	for (i = 0; i < 1000; i++)
	{
		const float max_val = ccv_max(ccv_max(fabsf(datr->data.f32[i]), fabsf(hda32->data.f32[i])), 1);
		*backward_max_diff = ccv_max(*backward_max_diff, fabsf(datr->data.f32[i] - hda32->data.f32[i]) / max_val);
	}
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(hg32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(har);
	ccv_nnc_tensor_free(hgr);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hb32);
	ccv_nnc_tensor_free(hbr);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hda32);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dat32);
	ccv_nnc_tensor_free(datr);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
}

static void _ccv_nnc_ewsoftplus_gpu_ref_datatype(const int datatype, float* const forward_max_diff, float* const backward_max_diff)
{
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hg32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const har = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hgr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hb32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hbr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const hda = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const hda32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const dat = ccv_nnc_tensor_new(0, _ccv_nnc_ew_cpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const dat32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const datr = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	ccv_nnc_tensor_t* const da = ccv_nnc_tensor_new(0, _ccv_nnc_ew_gpu_nchw_datatype(datatype), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 16 - 8;
	for (i = 0; i < 1000; i++)
		hg32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha32, hg32), TENSOR_LIST(ha, hg), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(har, hgr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(har), TENSOR_LIST(hb32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb32), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(hbr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(hb32), 0);
	*forward_max_diff = 0;
	for (i = 0; i < 1000; i++)
	{
		const float max_val = ccv_max(ccv_max(fabsf(hbr->data.f32[i]), fabsf(hb32->data.f32[i])), 1);
		*forward_max_diff = ccv_max(*forward_max_diff, fabsf(hbr->data.f32[i] - hb32->data.f32[i]) / max_val);
	}
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hgr, har), TENSOR_LIST(dat32), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dat32), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(datr), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hda), TENSOR_LIST(hda32), 0);
	*backward_max_diff = 0;
	for (i = 0; i < 1000; i++)
	{
		const float max_val = ccv_max(ccv_max(fabsf(datr->data.f32[i]), fabsf(hda32->data.f32[i])), 1);
		*backward_max_diff = ccv_max(*backward_max_diff, fabsf(datr->data.f32[i] - hda32->data.f32[i]) / max_val);
	}
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(hg32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(har);
	ccv_nnc_tensor_free(hgr);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hb32);
	ccv_nnc_tensor_free(hbr);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hda32);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dat32);
	ccv_nnc_tensor_free(datr);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
}

TEST_CASE("exp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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

TEST_CASE("ewexp backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWEXP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
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

TEST_CASE("ewexp half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWEXP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float forward_max_diff, backward_max_diff;
	_ccv_nnc_ewexp_gpu_ref_datatype(CCV_16F, &forward_max_diff, &backward_max_diff);
	REQUIRE(forward_max_diff <= 5e-3, "GPU computed output should be close to CPU computed ones after 16F rounding");
	REQUIRE(backward_max_diff <= 5e-3, "GPU computed gradient should be close to CPU computed ones after 16F rounding");
}

TEST_CASE("ewexp bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWEXP_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float forward_max_diff, backward_max_diff;
	_ccv_nnc_ewexp_gpu_ref_datatype(CCV_16BF, &forward_max_diff, &backward_max_diff);
	REQUIRE(forward_max_diff <= 2e-2, "GPU computed output should be close to CPU computed ones after 16BF rounding");
	REQUIRE(backward_max_diff <= 2e-2, "GPU computed gradient should be close to CPU computed ones after 16BF rounding");
}

TEST_CASE("ewsoftplus forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 40 - 20;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 1000, 1e-5, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewsoftplus backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 40 - 20;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dat->data.f32, hda->data.f32, 1000, 1e-5, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("ewsoftplus half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float forward_max_diff, backward_max_diff;
	_ccv_nnc_ewsoftplus_gpu_ref_datatype(CCV_16F, &forward_max_diff, &backward_max_diff);
	REQUIRE(forward_max_diff <= 5e-3, "GPU computed output should be close to CPU computed ones after 16F rounding");
	REQUIRE(backward_max_diff <= 5e-3, "GPU computed gradient should be close to CPU computed ones after 16F rounding");
}

TEST_CASE("ewsoftplus bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float forward_max_diff, backward_max_diff;
	_ccv_nnc_ewsoftplus_gpu_ref_datatype(CCV_16BF, &forward_max_diff, &backward_max_diff);
	REQUIRE(forward_max_diff <= 2e-2, "GPU computed output should be close to CPU computed ones after 16BF rounding");
	REQUIRE(backward_max_diff <= 2e-2, "GPU computed gradient should be close to CPU computed ones after 16BF rounding");
}

TEST_CASE("ewpow forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWPOW_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 + 0.1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ct), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_TENSOR_EQ(ct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("ewpow backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWPOW_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWPOW_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 + 0.1;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_BACKWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_BACKWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_TENSOR_EQ(dat, hda, "GPU computed dA should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(dat);
}

TEST_CASE("ewsin forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSIN_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewsin backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSIN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSIN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dat->data.f32, hda->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("ewcos forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWCOS_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewcos backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWCOS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWCOS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da), TENSOR_LIST(hda), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, dat->data.f32, hda->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
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

TEST_CASE("ewlog forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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

TEST_CASE("ewlog backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWLOG_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
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

TEST_CASE("ewsqrt forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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

TEST_CASE("ewsqrt backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, b), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, 0, hb), TENSOR_LIST(dat), 0);
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

TEST_CASE("ewabs forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWABS_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewabs backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWABS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWABS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10;
	for (i = 0; i < 1000; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a), TENSOR_LIST(da), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha), TENSOR_LIST(dat), 0);
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
	int i;
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
	int i;
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
	int i;
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

TEST_CASE("scaled dot product attention with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 40, 160, 224, 224, 128 };
		int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, B * R * Hq * D, 3e-3, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention with flash_attn varlen")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int B = 4;
	const int Hq = 4;
	const int Hk = 2;
	const int D = 64;
	const int q_offsets[2][5] = {
		{ 0, 5, 8, 15, 17 },
		{ 0, 4, 10, 13, 20 },
	};
	const int kv_offsets[2][5] = {
		{ 0, 6, 13, 16, 22 },
		{ 0, 4, 11, 15, 21 },
	};
	const int max_seqlen_q[2] = { 7, 7 };
	const int max_seqlen_kv[2] = { 7, 7 };
	for (int trial = 0; trial < 2; ++trial)
	{
		const int is_causal = trial;
		const int total_q = q_offsets[trial][B];
		const int total_k = kv_offsets[trial][B];
		const float scale = 1.0 / sqrt((float)D);
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		for (int i = 0; i < total_q * Hq * D; ++i)
			q_tensor->data.f32[i] = (float)((i * 17) % 97 - 48) / 256;
		for (int i = 0; i < total_k * Hk * D; ++i)
		{
			k_tensor->data.f32[i] = (float)((i * 13) % 89 - 44) / 256;
			v_tensor->data.f32[i] = (float)((i * 19) % 101 - 50) / 128;
		}
		ccv_nnc_tensor_t* const o_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		for (int b = 0; b < B; ++b)
		{
			const int q_start = q_offsets[trial][b];
			const int k_start = kv_offsets[trial][b];
			const int R = q_offsets[trial][b + 1] - q_start;
			const int C = kv_offsets[trial][b + 1] - k_start;
			ccv_nnc_tensor_t* const q_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const o_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			memcpy(q_seq->data.f32, q_tensor->data.f32 + q_start * Hq * D, sizeof(float) * R * Hq * D);
			memcpy(k_seq->data.f32, k_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			memcpy(v_seq->data.f32, v_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_seq, k_seq, v_seq), TENSOR_LIST(o_seq), 0);
			memcpy(o_tensor_ref->data.f32 + q_start * Hq * D, o_seq->data.f32, sizeof(float) * R * Hq * D);
			ccv_nnc_tensor_free(q_seq);
			ccv_nnc_tensor_free(k_seq);
			ccv_nnc_tensor_free(v_seq);
			ccv_nnc_tensor_free(o_seq);
		}
		ccv_nnc_tensor_t* const h_q_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		ccv_nnc_tensor_t* const h_kv_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		for (int i = 0; i < B + 1; ++i)
		{
			h_q_seq_offsets->data.i32[i] = q_offsets[trial][i];
			h_kv_seq_offsets->data.i32[i] = kv_offsets[trial][i];
		}
		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.is_varlen = 1;
		cmd.info.scaled_dot_product_attention.max_seqlen_q = max_seqlen_q[trial];
		cmd.info.scaled_dot_product_attention.max_seqlen_kv = max_seqlen_kv[trial];
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL, h_q_seq_offsets, h_kv_seq_offsets), TENSOR_LIST(o_tensor, NULL), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, o_tensor_ref->data.f32, total_q * Hq * D, 1e-5, "varlen CPU computed output should match per-sequence CPU output when causal=%d", is_causal);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_q_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_kv_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, h_q_seq_offsets, h_kv_seq_offsets), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_q_seq_offsets, gpu_kv_seq_offsets), 0);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL, gpu_q_seq_offsets, gpu_kv_seq_offsets), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, total_q * Hq * D, 5e-3, "varlen GPU computed output should match packed CPU output when causal=%d", is_causal);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(o_tensor_ref);
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(h_q_seq_offsets);
		ccv_nnc_tensor_free(h_kv_seq_offsets);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_q_seq_offsets);
		ccv_nnc_tensor_free(gpu_kv_seq_offsets);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
	}
}

TEST_CASE("scaled dot product attention with flash_attn in bfloat")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 4
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1 };
		int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5 };
		int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5 };
		int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32 };
		int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8 };
		int D_candidates[num_trials] = {  64, 40, 160, 224, 224, 128 };
		int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL), TENSOR_LIST(o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, NULL), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, B * R * Hq * D, 1e-2, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention + unify head with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / sqrtf(512);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const w_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 512, 512), 0);
	ccv_nnc_tensor_t* const bias_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), 0);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16F, 32, 128, 512), "r");
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
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, c);
	ccv_nnc_tensor_t* const gc_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gc);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const ho_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc_tensor, gr_tensor), TENSOR_LIST(ho_f16, hr_f16), 0);
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ho_f16, hr_f16), TENSOR_LIST(ho, hr), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, ho->data.f32, 32 * 128 * 8 * 64, 3e-3, "graph computed result should match scaled dot product attention op result");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, r_tensor->data.f32, hr->data.f32, 32 * 128 * 512, 3e-2, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(ho_f16);
	ccv_nnc_tensor_free(hr_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(w_tensor_f16);
	ccv_nnc_tensor_free(bias_tensor_f16);
}

TEST_CASE("scaled dot product attention + unify head with flash_attn in bfloat")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
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
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / sqrtf(512);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const w_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 512, 512), 0);
	ccv_nnc_tensor_t* const bias_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), 0);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 16BF, 32, 128, 512), "r");
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
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, w_tensor_f16, bias_tensor_f16), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, c);
	ccv_nnc_tensor_t* const gc_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gc);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const ho_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc_tensor, gr_tensor), TENSOR_LIST(ho_f16, hr_f16), 0);
	ccv_nnc_tensor_t* const ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), 0);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ho_f16, hr_f16), TENSOR_LIST(ho, hr), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, ho->data.f32, 32 * 128 * 8 * 64, 1e-2, "graph computed result should match scaled dot product attention op result");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, r_tensor->data.f32, hr->data.f32, 32 * 128 * 512, 1e-1, "graph computed result should match scaled dot product attention op result");
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(ho_f16);
	ccv_nnc_tensor_free(hr_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(w_tensor_f16);
	ccv_nnc_tensor_free(bias_tensor_f16);
}

TEST_CASE("scaled dot product attention gradient with flash_attn")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
#define num_long_trials 8
#define num_short_trials 4
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		const int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1, 32,   12, 16, 1, 2, 1 };
		const int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5, 160,  256, 128, 77, 77, 5 };
		const int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5, 128,  128, 128, 128, 128, 5 };
		const int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int Hk_candidates[num_trials] = {   8,  8, 8, 8, 2, 8, 8,  8, 8, 8, 2, 8 };
		const int D_candidates[num_trials] = {  64, 40, 160, 192, 256, 128, 64, 40, 160, 192, 256, 128 };
		const int is_causal_candidates[num_trials] = {  1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1 };
		const int deterministic_candidates[num_trials] = {  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

		const int B = B_candidates[trial];
		const int R = R_candidates[trial];
		const int C = C_candidates[trial];
		const int Hq = Hq_candidates[trial];
		const int Hk = Hk_candidates[trial];
		const int D = D_candidates[trial];
		const int is_causal = is_causal_candidates[trial];
		const int deterministic = deterministic_candidates[trial];
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		for (int i = 0; i < B * R * Hq * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.deterministic = deterministic;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * Hq * D, 1e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * Hk * D, 3e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dv_tensor->data.f32, dv_tensor->data.f32, B * C * Hk * D, 6e-3, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(do_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

static ccv_nnc_tensor_param_t _ccv_nnc_cmul_mixed_params(const int memory, const int datatype)
{
	return (ccv_nnc_tensor_param_t){
		.type = memory,
		.format = CCV_TENSOR_FORMAT_NCHW,
		.datatype = datatype,
		.dim = { 2, 3, 10, 0 },
	};
}

static int _ccv_nnc_cmul_round_to_datatype(const ccv_nnc_tensor_t* const source, ccv_nnc_tensor_t* const typed, ccv_nnc_tensor_t* const rounded, const int count)
{
	int i;
	switch (typed->info.datatype)
	{
		case CCV_32F:
			for (i = 0; i < count; i++)
			{
				typed->data.f32[i] = source->data.f32[i];
				rounded->data.f32[i] = source->data.f32[i];
			}
			return CCV_NNC_EXEC_SUCCESS;
		case CCV_16F: {
			int status = ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)source), TENSOR_LIST(typed), 0);
			if (status != CCV_NNC_EXEC_SUCCESS)
				return status;
			return ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(typed), TENSOR_LIST(rounded), 0);
		}
		case CCV_16BF:
			ccv_float_to_bfloat(source->data.f32, (uint16_t*)typed->data.f16, count);
			ccv_bfloat_to_float((uint16_t*)typed->data.f16, rounded->data.f32, count);
			return CCV_NNC_EXEC_SUCCESS;
	}
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_cmul_convert_to_float(const ccv_nnc_tensor_t* const typed, ccv_nnc_tensor_t* const rounded, const int count)
{
	int i;
	switch (typed->info.datatype)
	{
		case CCV_32F:
			for (i = 0; i < count; i++)
				rounded->data.f32[i] = typed->data.f32[i];
			return CCV_NNC_EXEC_SUCCESS;
		case CCV_16F:
			return ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)typed), TENSOR_LIST(rounded), 0);
		case CCV_16BF:
			ccv_bfloat_to_float((uint16_t*)typed->data.f16, rounded->data.f32, count);
			return CCV_NNC_EXEC_SUCCESS;
	}
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_cmul_mixed_precision_case(const int backend, const int a_datatype, const int b_datatype, const int c_datatype, float* const max_diff)
{
	const int count = 2 * 3 * 10;
	ccv_nnc_tensor_param_t ha32_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, CCV_32F);
	ccv_nnc_tensor_param_t hb32_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, CCV_32F);
	ccv_nnc_tensor_param_t hc32_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, CCV_32F);
	ccv_nnc_tensor_param_t ha_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, a_datatype);
	ccv_nnc_tensor_param_t hb_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, b_datatype);
	ccv_nnc_tensor_param_t hc_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, c_datatype);
	ccv_nnc_tensor_param_t ga_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, a_datatype);
	ccv_nnc_tensor_param_t gb_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, b_datatype);
	ccv_nnc_tensor_param_t gc_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, c_datatype);
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, ha32_params, 0);
	ccv_nnc_tensor_t* const hb32 = ccv_nnc_tensor_new(0, hb32_params, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ha_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, hb_params, 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, hc_params, 0);
	ccv_nnc_tensor_t* const ha_rounded = ccv_nnc_tensor_new(0, ha32_params, 0);
	ccv_nnc_tensor_t* const hb_rounded = ccv_nnc_tensor_new(0, hb32_params, 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, hc32_params, 0);
	ccv_nnc_tensor_t* const expected_rounded = ccv_nnc_tensor_new(0, hc32_params, 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, hc32_params, 0);
	ccv_nnc_tensor_t* const expected_typed = ccv_nnc_tensor_new(0, hc_params, 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, gc_params, 0);
	int i;
	for (i = 0; i < count; i++)
	{
		ha32->data.f32[i] = (float)((i % 13) - 6) * 0.07f;
		hb32->data.f32[i] = (float)(((i * 5 + 3) % 17) - 8) * 0.05f;
	}
	int status = _ccv_nnc_cmul_round_to_datatype(ha32, ha, ha_rounded, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_round_to_datatype(hb32, hb, hb_rounded, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(ga, gb), 0);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		ccv_nnc_cmd_t cmd = CMD_CMUL_FORWARD();
		cmd.backend = backend;
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	}
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(hc), 0);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_convert_to_float(hc, actual, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		for (i = 0; i < count; i += 2)
		{
			const float a0 = ha_rounded->data.f32[i];
			const float a1 = ha_rounded->data.f32[i + 1];
			const float b0 = hb_rounded->data.f32[i];
			const float b1 = hb_rounded->data.f32[i + 1];
			expected->data.f32[i] = a0 * b0 - a1 * b1;
			expected->data.f32[i + 1] = a0 * b1 + a1 * b0;
		}
		status = _ccv_nnc_cmul_round_to_datatype(expected, expected_typed, expected_rounded, count);
	}
	*max_diff = 0;
	if (status == CCV_NNC_EXEC_SUCCESS)
		for (i = 0; i < count; i++)
			*max_diff = ccv_max(*max_diff, fabsf(actual->data.f32[i] - expected_rounded->data.f32[i]));
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(hb32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ha_rounded);
	ccv_nnc_tensor_free(hb_rounded);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(expected_rounded);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(expected_typed);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	return status;
}

static int _ccv_nnc_cmul_mixed_precision_backward_case(const int backend, const int g_datatype, const int a_datatype, const int b_datatype, const int c_datatype, const int d_datatype, const int conjugate, float* const max_diff_c, float* const max_diff_d)
{
	const int count = 2 * 3 * 10;
	ccv_nnc_tensor_param_t h32_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, CCV_32F);
	ccv_nnc_tensor_param_t hg_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, g_datatype);
	ccv_nnc_tensor_param_t ha_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, a_datatype);
	ccv_nnc_tensor_param_t hb_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, b_datatype);
	ccv_nnc_tensor_param_t hc_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, c_datatype);
	ccv_nnc_tensor_param_t hd_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_CPU_MEMORY, d_datatype);
	ccv_nnc_tensor_param_t gg_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, g_datatype);
	ccv_nnc_tensor_param_t ga_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, a_datatype);
	ccv_nnc_tensor_param_t gb_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, b_datatype);
	ccv_nnc_tensor_param_t gc_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, c_datatype);
	ccv_nnc_tensor_param_t gd_params = _ccv_nnc_cmul_mixed_params(CCV_TENSOR_GPU_MEMORY | 000, d_datatype);
	ccv_nnc_tensor_t* const hg32 = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const hb32 = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, hg_params, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ha_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, hb_params, 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, hc_params, 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, hd_params, 0);
	ccv_nnc_tensor_t* const hg_rounded = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const ha_rounded = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const hb_rounded = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const expected_c = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const expected_d = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const expected_c_rounded = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const expected_d_rounded = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const actual_c = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const actual_d = ccv_nnc_tensor_new(0, h32_params, 0);
	ccv_nnc_tensor_t* const expected_c_typed = ccv_nnc_tensor_new(0, hc_params, 0);
	ccv_nnc_tensor_t* const expected_d_typed = ccv_nnc_tensor_new(0, hd_params, 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, gg_params, 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const gc = ccv_nnc_tensor_new(0, gc_params, 0);
	ccv_nnc_tensor_t* const gd = ccv_nnc_tensor_new(0, gd_params, 0);
	int i;
	for (i = 0; i < count; i++)
	{
		hg32->data.f32[i] = (float)(((i * 7 + 1) % 19) - 9) * 0.04f;
		ha32->data.f32[i] = (float)((i % 13) - 6) * 0.07f;
		hb32->data.f32[i] = (float)(((i * 5 + 3) % 17) - 8) * 0.05f;
	}
	int status = _ccv_nnc_cmul_round_to_datatype(hg32, hg, hg_rounded, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_round_to_datatype(ha32, ha, ha_rounded, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_round_to_datatype(hb32, hb, hb_rounded, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(gg, ga, gb), 0);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		ccv_nnc_cmd_t cmd = CMD_CMUL_BACKWARD();
		cmd.backend = backend;
		cmd.info.cmul.conjugate = conjugate;
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gc, gd), 0);
	}
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc, gd), TENSOR_LIST(hc, hd), 0);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_convert_to_float(hc, actual_c, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_convert_to_float(hd, actual_d, count);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		for (i = 0; i < count; i += 2)
		{
			const float g0 = hg_rounded->data.f32[i];
			const float g1 = hg_rounded->data.f32[i + 1];
			const float a0 = ha_rounded->data.f32[i];
			const float a1 = ha_rounded->data.f32[i + 1];
			const float b0 = hb_rounded->data.f32[i];
			const float b1 = hb_rounded->data.f32[i + 1];
			expected_c->data.f32[i] = conjugate ? g0 * b0 - g1 * b1 : g0 * b0 + g1 * b1;
			expected_c->data.f32[i + 1] = conjugate ? g0 * b1 + g1 * b0 : -g0 * b1 + g1 * b0;
			expected_d->data.f32[i] = g0 * a0 + g1 * a1;
			expected_d->data.f32[i + 1] = conjugate ? g0 * a1 - g1 * a0 : -g0 * a1 + g1 * a0;
		}
		status = _ccv_nnc_cmul_round_to_datatype(expected_c, expected_c_typed, expected_c_rounded, count);
	}
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = _ccv_nnc_cmul_round_to_datatype(expected_d, expected_d_typed, expected_d_rounded, count);
	*max_diff_c = 0;
	*max_diff_d = 0;
	if (status == CCV_NNC_EXEC_SUCCESS)
		for (i = 0; i < count; i++)
		{
			*max_diff_c = ccv_max(*max_diff_c, fabsf(actual_c->data.f32[i] - expected_c_rounded->data.f32[i]));
			*max_diff_d = ccv_max(*max_diff_d, fabsf(actual_d->data.f32[i] - expected_d_rounded->data.f32[i]));
		}
	ccv_nnc_tensor_free(hg32);
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(hb32);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(hg_rounded);
	ccv_nnc_tensor_free(ha_rounded);
	ccv_nnc_tensor_free(hb_rounded);
	ccv_nnc_tensor_free(expected_c);
	ccv_nnc_tensor_free(expected_d);
	ccv_nnc_tensor_free(expected_c_rounded);
	ccv_nnc_tensor_free(expected_d_rounded);
	ccv_nnc_tensor_free(actual_c);
	ccv_nnc_tensor_free(actual_d);
	ccv_nnc_tensor_free(expected_c_typed);
	ccv_nnc_tensor_free(expected_d_typed);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
	return status;
}

TEST_CASE("cmul mixed half and bfloat precision with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_mfa_0 = 1e30f;
	float max_diff_mfa_1 = 1e30f;
	float max_diff_mfa_2 = 1e30f;
	const int status_mfa_0 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, &max_diff_mfa_0);
	const int status_mfa_1 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16BF, CCV_16F, CCV_16BF, &max_diff_mfa_1);
	const int status_mfa_2 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_32F, &max_diff_mfa_2);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_graph_0 = 1e30f;
	float max_diff_graph_1 = 1e30f;
	float max_diff_graph_2 = 1e30f;
	const int status_graph_0 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, &max_diff_graph_0);
	const int status_graph_1 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16BF, CCV_16F, CCV_16BF, &max_diff_graph_1);
	const int status_graph_2 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_32F, &max_diff_graph_2);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_mfa_0, "mixed half / bfloat cmul through mfa should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_mfa_1, "mixed bfloat / half cmul through mfa should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_mfa_2, "mixed half / bfloat cmul to float through mfa should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_graph_0, "mixed half / bfloat cmul through graph fallback should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_graph_1, "mixed bfloat / half cmul through graph fallback should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_graph_2, "mixed half / bfloat cmul to float through graph fallback should run");
	REQUIRE(max_diff_mfa_0 <= 3e-2, "mixed half / bfloat cmul through mfa should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_mfa_1 <= 3e-2, "mixed bfloat / half cmul through mfa should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_mfa_2 <= 3e-2, "mixed half / bfloat cmul to float through mfa should match fp32 reference");
	REQUIRE(max_diff_graph_0 <= 3e-2, "mixed half / bfloat cmul through graph fallback should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_graph_1 <= 3e-2, "mixed bfloat / half cmul through graph fallback should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_graph_2 <= 3e-2, "mixed half / bfloat cmul to float through graph fallback should match fp32 reference");
}

TEST_CASE("cmul gradient mixed half and bfloat precision with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_mfa_c = 1e30f;
	float max_diff_mfa_d = 1e30f;
	const int status_mfa = _ccv_nnc_cmul_mixed_precision_backward_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, CCV_16BF, CCV_16F, 0, &max_diff_mfa_c, &max_diff_mfa_d);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_graph_c = 1e30f;
	float max_diff_graph_d = 1e30f;
	const int status_graph = _ccv_nnc_cmul_mixed_precision_backward_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, CCV_16BF, CCV_16F, 0, &max_diff_graph_c, &max_diff_graph_d);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_mfa, "mixed half / bfloat cmul gradient through mfa should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_graph, "mixed half / bfloat cmul gradient through graph fallback should run");
	REQUIRE(max_diff_mfa_c <= 3e-2, "mixed half / bfloat cmul gradient first output through mfa should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_mfa_d <= 3e-2, "mixed half / bfloat cmul gradient second output through mfa should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_graph_c <= 3e-2, "mixed half / bfloat cmul gradient first output through graph fallback should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_graph_d <= 3e-2, "mixed half / bfloat cmul gradient second output through graph fallback should match fp32 reference rounded to output dtype");
}

TEST_CASE("conjugating cmul gradient mixed half and bfloat precision with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_mfa_c = 1e30f;
	float max_diff_mfa_d = 1e30f;
	const int status_mfa = _ccv_nnc_cmul_mixed_precision_backward_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, CCV_16BF, CCV_16F, 1, &max_diff_mfa_c, &max_diff_mfa_d);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	float max_diff_graph_c = 1e30f;
	float max_diff_graph_d = 1e30f;
	const int status_graph = _ccv_nnc_cmul_mixed_precision_backward_case(CCV_NNC_BACKEND_MPS, CCV_16F, CCV_16BF, CCV_16F, CCV_16BF, CCV_16F, 1, &max_diff_graph_c, &max_diff_graph_d);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_mfa, "conjugating mixed half / bfloat cmul gradient through mfa should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_graph, "conjugating mixed half / bfloat cmul gradient through graph fallback should run");
	REQUIRE(max_diff_mfa_c <= 3e-2, "conjugating mixed half / bfloat cmul first gradient through mfa should match fp32 reference");
	REQUIRE(max_diff_mfa_d <= 3e-2, "conjugating mixed half / bfloat cmul second gradient through mfa should match fp32 reference");
	REQUIRE(max_diff_graph_c <= 3e-2, "conjugating mixed half / bfloat cmul first gradient through graph fallback should match fp32 reference");
	REQUIRE(max_diff_graph_d <= 3e-2, "conjugating mixed half / bfloat cmul second gradient through graph fallback should match fp32 reference");
}

TEST_CASE("cmul mixed half and bfloat precision with cuda")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float max_diff_0 = 1e30f;
	float max_diff_1 = 1e30f;
	float max_diff_2 = 1e30f;
	const int status_0 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_GPU_REF, CCV_16F, CCV_16BF, CCV_16F, &max_diff_0);
	const int status_1 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_GPU_REF, CCV_16BF, CCV_16F, CCV_16BF, &max_diff_1);
	const int status_2 = _ccv_nnc_cmul_mixed_precision_case(CCV_NNC_BACKEND_GPU_REF, CCV_16F, CCV_16BF, CCV_32F, &max_diff_2);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_0, "mixed half / bfloat cmul through cuda should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_1, "mixed bfloat / half cmul through cuda should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status_2, "mixed half / bfloat cmul to float through cuda should run");
	REQUIRE(max_diff_0 <= 3e-2, "mixed half / bfloat cmul through cuda should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_1 <= 3e-2, "mixed bfloat / half cmul through cuda should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_2 <= 3e-2, "mixed half / bfloat cmul to float through cuda should match fp32 reference");
}

TEST_CASE("cmul gradient mixed half and bfloat precision with cuda")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	float max_diff_c = 1e30f;
	float max_diff_d = 1e30f;
	const int status = _ccv_nnc_cmul_mixed_precision_backward_case(CCV_NNC_BACKEND_GPU_REF, CCV_16F, CCV_16BF, CCV_16F, CCV_16BF, CCV_16F, 0, &max_diff_c, &max_diff_d);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, status, "mixed half / bfloat cmul gradient through cuda should run");
	REQUIRE(max_diff_c <= 3e-2, "mixed half / bfloat cmul gradient first output through cuda should match fp32 reference rounded to output dtype");
	REQUIRE(max_diff_d <= 3e-2, "mixed half / bfloat cmul gradient second output through cuda should match fp32 reference rounded to output dtype");
}

TEST_CASE("cmul in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z16_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tz->data.f32, z_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z16_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in float, broadcast semantics")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 8, 128), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 1, 128), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 5, 8, 128), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 1, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1 * 5 * 8 * 128; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 5 * 1 * 128; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 5, 8, 128), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in float, broadcast semantics with longer than 65535 sequence")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 8, 16), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 1, 16), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 70000, 8, 16), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 1, 16), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1 * 70000 * 8 * 16; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 70000 * 1 * 16; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 70000, 8, 16), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul in float, broadcast semantics with longer than 65535 sequence and more than 1 batch size")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 2, 40000, 8, 16), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 1, 40000, 1, 16), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 2, 40000, 8, 16), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_FORWARD(), TENSOR_SYMBOL_LIST(a, b), TENSOR_SYMBOL_LIST(c), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 40000, 1, 16), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 40000 * 8 * 16; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 1 * 40000 * 1 * 16; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c_tensor), TENSOR_LIST(z_tensor), 0);
	ccv_nnc_tensor_t* const tz = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 40000, 8, 16), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor), TENSOR_LIST(tz), 0);
	REQUIRE_TENSOR_EQ(tz, z_tensor, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(tz);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul gradient in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "d");
	ccv_nnc_tensor_symbol_t e = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 20, 10), "e");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_BACKWARD(), TENSOR_SYMBOL_LIST(a, b, c), TENSOR_SYMBOL_LIST(d, e), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		z_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z_tensor), TENSOR_LIST(c_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const od_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d);
	ccv_nnc_tensor_t* const oe_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const e_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, e);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d_tensor, e_tensor), TENSOR_LIST(od_tensor, oe_tensor), 0);
	ccv_nnc_tensor_t* const td = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const te = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor, z_tensor), TENSOR_LIST(td, te), 0);
	REQUIRE_TENSOR_EQ(td, od_tensor, "cmul gradient from cudnn should match from CPU");
	REQUIRE_TENSOR_EQ(te, oe_tensor, "cmul gradient from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(od_tensor);
	ccv_nnc_tensor_free(oe_tensor);
	ccv_nnc_tensor_free(td);
	ccv_nnc_tensor_free(te);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("cmul gradient in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t a = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "a");
	ccv_nnc_tensor_symbol_t b = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "b");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t d = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_tensor_symbol_t e = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 16F, 20, 10), "c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CMUL_BACKWARD(), TENSOR_SYMBOL_LIST(a, b, c), TENSOR_SYMBOL_LIST(d, e), "cmul");
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), &graph, &tensor_arena, &graph_exec_arena);
	GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const z_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20 * 10; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		y_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 20 * 10; i++)
		z_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, a);
	ccv_nnc_tensor_t* const x16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor), TENSOR_LIST(x16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x16_tensor), TENSOR_LIST(a_tensor), 0);
	ccv_nnc_tensor_t* const b_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, b);
	ccv_nnc_tensor_t* const y16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y_tensor), TENSOR_LIST(y16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y16_tensor), TENSOR_LIST(b_tensor), 0);
	ccv_nnc_tensor_t* const c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, c);
	ccv_nnc_tensor_t* const z16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z_tensor), TENSOR_LIST(z16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(z16_tensor), TENSOR_LIST(c_tensor), 0);
	ccv_nnc_graph_run(graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const od16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const od_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const d_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, d);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(d_tensor), TENSOR_LIST(od16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(od16_tensor), TENSOR_LIST(od_tensor), 0);
	ccv_nnc_tensor_t* const oe16_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 20, 10), 0);
	ccv_nnc_tensor_t* const oe_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const e_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, e);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(e_tensor), TENSOR_LIST(oe16_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(oe16_tensor), TENSOR_LIST(oe_tensor), 0);
	ccv_nnc_tensor_t* const td = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_tensor_t* const te = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 20, 10), 0);
	ccv_nnc_cmd_exec(CMD_CMUL_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(x_tensor, y_tensor, z_tensor), TENSOR_LIST(td, te), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, td->data.f32, od_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, te->data.f32, oe_tensor->data.f32, 20 * 10, 2e-3, "gelu from cudnn should match from CPU");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(x16_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(y16_tensor);
	ccv_nnc_tensor_free(z_tensor);
	ccv_nnc_tensor_free(z16_tensor);
	ccv_nnc_tensor_free(od_tensor);
	ccv_nnc_tensor_free(od16_tensor);
	ccv_nnc_tensor_free(td);
	ccv_nnc_tensor_free(oe_tensor);
	ccv_nnc_tensor_free(oe16_tensor);
	ccv_nnc_tensor_free(te);
	ccv_nnc_graph_free(graph);
	ccv_nnc_tensor_arena_free(tensor_arena);
	ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	ccv_nnc_symbolic_graph_free(symbolic_graph);
}

TEST_CASE("segmented gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 2;
	hindices->data.i32[2] = 1;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm with packed row-wise int8-x weights in all formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int rows = 17;
	const int k = 65;
	const int n = 16;
	const int segments = 3;
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, k), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	indices->data.i32[0] = 2;
	indices->data.i32[1] = 0;
	indices->data.i32[2] = 1;
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	counts->data.i32[0] = 5;
	counts->data.i32[1] = 7;
	counts->data.i32[2] = 5;
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, segments, n, k), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, segments, n), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < rows * k; i++)
		a->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	for (i = 0; i < segments * n * k; i++)
		w->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	for (i = 0; i < segments * n; i++)
		bias->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 2 - 1);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, k), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const gcounts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, segments, n), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, bias), TENSOR_LIST(ga, gindices, gcounts, gbias), 0);
	int f;
	for (f = 0; f < sizeof(formats) / sizeof(formats[0]); f++)
	{
		const ccv_nnc_tensor_param_t q_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, segments, n, k), formats[f]);
		ccv_nnc_tensor_t* const qw = ccv_nnc_tensor_new(0, q_params, 0);
		const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(w->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, (size_t)segments * n * k, k, formats[f], 0, 0, qw->data.u8, ccv_nnc_tensor_data_size_without_padding(qw->info));
		REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(qw->info), "packed row-wise int8-x size should match");
		ccv_nnc_tensor_t* const dq_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, segments, n, k), 0);
		ccv_nnc_dequantize_8i_rowwise_x(qw->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, k, formats[f], dq_w->data.u8, (size_t)segments * n * k);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
		ccv_nnc_tensor_t* const expected_with_bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
		ccv_nnc_cmd_t cpu_cmd = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
		cpu_cmd.backend = CCV_NNC_BACKEND_CPU_REF;
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, dq_w), TENSOR_LIST(expected), 0);
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, dq_w, bias), TENSOR_LIST(expected_with_bias), 0);
		ccv_nnc_tensor_param_t gpu_q_params = q_params;
		gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		ccv_nnc_tensor_t* const gqw = ccv_nnc_tensor_new(0, gpu_q_params, 0);
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
		ccv_nnc_tensor_t* const gout_with_bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, n), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
		ccv_nnc_tensor_t* const actual_with_bias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, n), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(qw), TENSOR_LIST(gqw), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
		gpu_cmd.backend = CCV_NNC_BACKEND_GPU_CUBLAS;
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices, gcounts, gqw), TENSOR_LIST(gout), 0);
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices, gcounts, gqw, gbias), TENSOR_LIST(gout_with_bias), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout, gout_with_bias), TENSOR_LIST(actual, actual_with_bias), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, rows * n, 1e-3, "segmented gemm with packed row-wise int8-x weights should match CPU");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_with_bias->data.f32, actual_with_bias->data.f32, rows * n, 1e-3, "segmented gemm with bias and packed row-wise int8-x weights should match CPU");
		ccv_nnc_tensor_free(actual_with_bias);
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(gout_with_bias);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gqw);
		ccv_nnc_tensor_free(expected_with_bias);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(dq_w);
		ccv_nnc_tensor_free(qw);
	}
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
}

TEST_CASE("segmented gemm with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 3 * 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(a, indices, counts, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 2;
	hindices->data.i32[2] = 1;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 128), 0);
	ccv_nnc_tensor_t* hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(ha16, hw16), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hindices, hcounts, hw16), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb16), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 80 * 64, 1e-3, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hw16);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm with bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 25;
	hcounts->data.i32[2] = 35;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 3 * 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 80, 128), 0);
	ccv_nnc_tensor_t* hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hbias16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 3, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(ha16, hw16, hbias16), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 3, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hindices, hcounts, hw16, hbias16), TENSOR_LIST(a, indices, counts, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb16), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 80 * 64, 1e-3, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hw16);
	ccv_nnc_tensor_free(hbias16);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm, reuse")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 128), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 30;
	hcounts->data.i32[2] = 30;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 64), 0);
	int i;
	for (i = 0; i < 3 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 80 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 128), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("segmented gemm, large k")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) || ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 5120), 0);
	ccv_nnc_tensor_t* hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 0;
	hindices->data.i32[1] = 1;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 20;
	hcounts->data.i32[1] = 30;
	hcounts->data.i32[2] = 30;
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2560, 5120), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 2560), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 80, 2560), 0);
	int i;
	for (i = 0; i < 3 * 2560 * 5120; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 5120;
	for (i = 0; i < 80 * 5120; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 5120), 0);
	ccv_nnc_tensor_t* indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2560, 5120), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 80, 2560), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should match from CPU");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

#include "case_main.h"
