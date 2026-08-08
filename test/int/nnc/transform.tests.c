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

TEST_CASE("set forward and backward skip empty tensors on CPU")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 0, 128), 0);
	ccv_nnc_cmd_t set = CMD_SET_FORWARD(1);
	set.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(set, ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0), "CPU set forward should skip an empty tensor");
	set = CMD_SET_BACKWARD(1);
	set.backend = CCV_NNC_BACKEND_CPU_REF;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(set, ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0), "CPU set backward should skip an empty tensor");
	ccv_nnc_tensor_free(a);
}

TEST_CASE("set forward and backward skip empty tensors on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SET_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 0, 128), 0);
	ccv_nnc_cmd_t set = CMD_SET_FORWARD(1);
	set.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(set, ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0), "MPS set forward should skip an empty tensor");
	set = CMD_SET_BACKWARD(1);
	set.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(set, ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0), "MPS set backward should skip an empty tensor");
	ccv_nnc_tensor_free(a);
}

TEST_CASE("data conversion from float to half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_MPS));
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

TEST_CASE("mps data conversion skips empty tensors")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const empty_a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 0, 128), 0);
	ccv_nnc_tensor_t* const empty_b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	int i;
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = (float)i / 128;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_cmd_t move = CMD_DATA_TRANSFER_FORWARD();
	move.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t cast = CMD_DATATYPE_CONVERSION_FORWARD();
	cast.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cast, ccv_nnc_no_hint, 0, TENSOR_LIST(empty_a, a), TENSOR_LIST(empty_b, b), 0), "empty datatype conversion should be skipped");
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_cmd_exec(move, ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(expected), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(short, (short*)expected->data.f16, (short*)hb->data.f16, 128, 1, "non-empty conversion should still run");
	ccv_nnc_tensor_free(empty_a);
	ccv_nnc_tensor_free(empty_b);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(expected);
}

TEST_CASE("mps transpose skips empty tensors")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_TRANSPOSE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 4, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128, 4), 0);
	ccv_nnc_cmd_t transpose = CMD_TRANSPOSE_FORWARD(1, 2);
	transpose.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(transpose, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "empty transpose should be skipped");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("MPS DeepSeek 4 compressor operations skip empty tensors")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128), 0);
	ccv_nnc_tensor_t* const reduced = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 1), 0);
	ccv_nnc_tensor_t* const saved_inv_std = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 1), 0);
	ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 128, 128), 0);
	ccv_nnc_cmd_t op = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a, weight), TENSOR_LIST(c), 0), "empty compressor GEMM should be skipped");
	op = CMD_ADD_FORWARD(1, 1);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0), "empty compressor add should be skipped");
	op = CMD_SOFTMAX_FORWARD();
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0), "empty compressor softmax should be skipped");
	op = CMD_MUL_FORWARD(1);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0), "empty compressor multiply should be skipped");
	op = CMD_REDUCE_SUM_FORWARD(1);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(reduced), 0), "empty compressor reduction should be skipped");
	op = CMD_RMSNORM_FORWARD(1e-6, 0, 1);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c, saved_inv_std), 0), "empty compressor RMSNorm should be skipped");
	op = CMD_CMUL_FORWARD();
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0), "empty compressor complex multiply should be skipped");
	op = CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(1);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0), "empty indexer Walsh-Hadamard transform should be skipped");
	op = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0), "empty compressed-state conform data format should be skipped");
	op = CMD_CONFORM_DATA_FORMAT_BACKWARD(CCV_NNC_FP8_E4M3, 64);
	op.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(op, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0), "empty compressed-state conform data format gradient should be skipped");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(reduced);
	ccv_nnc_tensor_free(saved_inv_std);
	ccv_nnc_tensor_free(weight);
}

TEST_CASE("mps format transform skips empty tensors across formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_CHWN(000, 16F, 0, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 0, 128), 0);
	ccv_nnc_cmd_t transform = CMD_FORMAT_TRANSFORM_FORWARD();
	transform.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(
		CCV_NNC_EXEC_SUCCESS,
		ccv_nnc_cmd_exec(transform, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0),
		"empty cross-format transform should be skipped");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("data conversion from double to half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(64F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha->data.f64[i] = (double)dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64F, 1, 128), 0);
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

TEST_CASE("data conversion from double to float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(64F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha->data.f64[i] = (double)dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64F, 1, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 128, 1e-5, "Result should be exactly equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("data conversion from float to double")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(64F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(64F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, hb->data.f64, bt->data.f64, 128, 1e-5, "Result should be exactly equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("data conversion from double to half precision and to float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATATYPE_CONVERSION_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(64F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha->data.f64[i] = (double)dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64F, 1, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(hc), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(ct), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hc->data.f32, ct->data.f32, 128, 1, "Result should be exactly equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
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

TEST_CASE("format transform from for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 4, 5), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 3, 4), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, 4, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 3, 4), 0);
	int i, j, k;
	for (i = 0; i < 3 * 4 * 5; i++)
		ha->data.f32[i] = i;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NCHW(000, 32F, 2, 3, 4), DIM_ALLOC(1, 0, 1), DIM_ALLOC(4 * 5, 5, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 3, 4), 0);
	for (i = 0; i < 3; i++)
		for (j = 0; j < 4; j++)
			for (k = 0; k < 5; k++)
				if (i > 0 && k > 0 && j < 3)
					bt->data.f32[(i - 1) * 3 * 4 + j * 4 + k - 1] = i * 4 * 5 + j * 5 + k;
	REQUIRE_TENSOR_EQ(hb, bt, "cpu and gpu result should be the same");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("format transform to fill for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 3, 2), 0);
	ccv_nnc_tensor_t* const ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 3, 2), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 3, 4), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 3, 4), 0);
	ccv_nnc_tensor_t* const a0 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 3, 2), 0);
	ccv_nnc_tensor_t* const a1 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 3, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 3, 4), 0);
	int i;
	for (i = 0; i < 3 * 2; i++)
		ha0->data.f32[i] = i;
	for (i = 0; i < 3 * 2; i++)
		ha1->data.f32[i] = i - 5;
	for (i = 0; i < 3 * 4; i++)
		hc->data.f32[i] = i + 2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha0, ha1, hc), TENSOR_LIST(a0, a1, c), 0);
	ccv_nnc_tensor_view_t* const bv0 = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, 1, 3, 4), DIM_ALLOC(0, 0, 0), DIM_ALLOC(3 * 4, 4, 1));
	ccv_nnc_tensor_view_t* const bv10 = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, 1, 3, 2), DIM_ALLOC(1, 0, 0), DIM_ALLOC(3 * 4, 4, 1));
	ccv_nnc_tensor_view_t* const bv11 = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, 1, 3, 2), DIM_ALLOC(1, 0, 2), DIM_ALLOC(3 * 4, 4, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST((ccv_nnc_tensor_t*)bv0), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a0), TENSOR_LIST((ccv_nnc_tensor_t*)bv10), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a1), TENSOR_LIST((ccv_nnc_tensor_t*)bv11), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 3, 4), 0);
	for (i = 0; i < 3 * 4; i++)
		bt->data.f32[i] = i + 2;
	for (i = 0; i < 3 * 2; i++)
		bt->data.f32[(i / 2) * 4 + (i % 2) + 3 * 4] = i;
	for (i = 0; i < 3 * 2; i++)
		bt->data.f32[(i / 2) * 4 + (i % 2) + 2 + 3 * 4] = i - 5;
	REQUIRE_TENSOR_EQ(hb, bt, "cpu and gpu result should be the same");
	ccv_nnc_tensor_free(ha0);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_view_free(bv0);
	ccv_nnc_tensor_view_free(bv10);
	ccv_nnc_tensor_view_free(bv11);
	ccv_nnc_tensor_free(a0);
	ccv_nnc_tensor_free(a1);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("format transform to new strides for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 10, 9, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 3, 10, 9), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1, 3, 10, 9), 0);
	ccv_nnc_tensor_view_t* const btv = ccv_nnc_tensor_view_new(bt, CPU_TENSOR_NCHW(32F, 1, 10, 9, 3), DIM_ALLOC(), DIM_ALLOC(3 * 10 * 9, 9, 1, 10 * 9));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 10, 9, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1, 3, 10, 9), 0);
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, 1, 10, 9, 3), DIM_ALLOC(), DIM_ALLOC(3 * 10 * 9, 9, 1, 10 * 9));
	int i;
	for (i = 0; i < 10 * 9 * 3; i++)
		ha->data.f32[i] = i;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST((ccv_nnc_tensor_t*)btv), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "cpu and gpu result should be the same");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_view_free(btv);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 32f vectorized for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 13;
	const int cols = 28;
	const int source_row_stride = 40;
	const int source_row_offset = 2;
	const int source_col_offset = 4;
	const int output_row_offset = 1;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + output_row_offset, cols), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + output_row_offset, cols), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, rows + output_row_offset, cols), 0);
	int i;
	for (i = 0; i < (rows + source_row_offset + 1) * source_row_stride; i++)
		ha->data.f32[i] = (float)((i * 17) % 257 - 128) / 17;
	for (i = 0; i < (rows + output_row_offset) * cols; i++)
		hb->data.f32[i] = bt->data.f32[i] = -1000 - i;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_view_t* const hav = ccv_nnc_tensor_view_new(ha, CPU_TENSOR_NCHW(32F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NCHW(000, 32F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const btv = ccv_nnc_tensor_view_new(bt, CPU_TENSOR_NCHW(32F, rows, cols), DIM_ALLOC(output_row_offset, 0), DIM_ALLOC(cols, 1));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, rows, cols), DIM_ALLOC(output_row_offset, 0), DIM_ALLOC(cols, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)hav), TENSOR_LIST((ccv_nnc_tensor_t*)btv), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "strided copy should match cpu reference");
	ccv_nnc_tensor_view_free(hav);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(btv);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 16f scalar for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 11;
	const int cols = 23;
	const int source_row_stride = 35;
	const int source_row_offset = 1;
	const int source_col_offset = 1;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, rows, cols), 0);
	int i;
	for (i = 0; i < (rows + source_row_offset + 1) * source_row_stride; i++)
	{
		const float v = (float)((i * 13) % 193 - 96) / 19;
		ccv_float_to_half_precision(&v, (uint16_t*)ha->data.f16 + i, 1);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_view_t* const hav = ccv_nnc_tensor_view_new(ha, CPU_TENSOR_NCHW(16F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NCHW(000, 16F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)hav), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)bt->data.f16, (uint16_t*)hb->data.f16, rows * cols, "strided copy should match cpu reference");
	ccv_nnc_tensor_view_free(hav);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 16bf vectorized for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 9;
	const int cols = 32;
	const int source_row_stride = 48;
	const int source_row_offset = 1;
	const int source_col_offset = 4;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows, cols), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows, cols), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, rows, cols), 0);
	int i;
	for (i = 0; i < (rows + source_row_offset + 1) * source_row_stride; i++)
	{
		const float v = (float)((i * 29) % 251 - 125) / 23;
		ccv_float_to_bfloat(&v, (uint16_t*)ha->data.f16 + i, 1);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_view_t* const hav = ccv_nnc_tensor_view_new(ha, CPU_TENSOR_NCHW(16BF, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NCHW(000, 16BF, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)hav), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)bt->data.f16, (uint16_t*)hb->data.f16, rows * cols, "strided copy should match cpu reference");
	ccv_nnc_tensor_view_free(hav);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 32f destination stride vectorized for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 13;
	const int cols = 28;
	const int source_row_stride = 40;
	const int source_row_offset = 2;
	const int source_col_offset = 4;
	const int destination_row_stride = 44;
	const int destination_row_offset = 1;
	const int destination_col_offset = 8;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, rows + source_row_offset + 1, source_row_stride), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, rows + destination_row_offset + 1, destination_row_stride), 0);
	int i;
	for (i = 0; i < (rows + source_row_offset + 1) * source_row_stride; i++)
		ha->data.f32[i] = (float)((i * 31) % 509 - 254) / 31;
	for (i = 0; i < (rows + destination_row_offset + 1) * destination_row_stride; i++)
		hb->data.f32[i] = bt->data.f32[i] = -3000 - i;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_view_t* const hav = ccv_nnc_tensor_view_new(ha, CPU_TENSOR_NCHW(32F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NCHW(000, 32F, rows, cols), DIM_ALLOC(source_row_offset, source_col_offset), DIM_ALLOC(source_row_stride, 1));
	ccv_nnc_tensor_view_t* const btv = ccv_nnc_tensor_view_new(bt, CPU_TENSOR_NCHW(32F, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 32F, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)hav), TENSOR_LIST((ccv_nnc_tensor_t*)btv), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "strided destination copy should match cpu reference");
	ccv_nnc_tensor_view_free(hav);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(btv);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 16f destination stride scalar for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 11;
	const int cols = 23;
	const int destination_row_stride = 35;
	const int destination_row_offset = 1;
	const int destination_col_offset = 1;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows, cols), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, rows, cols), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, rows + destination_row_offset + 1, destination_row_stride), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
	{
		const float v = (float)((i * 37) % 263 - 131) / 29;
		ccv_float_to_half_precision(&v, (uint16_t*)ha->data.f16 + i, 1);
	}
	for (i = 0; i < (rows + destination_row_offset + 1) * destination_row_stride; i++)
	{
		const float v = (float)(-5000 - i);
		ccv_float_to_half_precision(&v, (uint16_t*)hb->data.f16 + i, 1);
		ccv_float_to_half_precision(&v, (uint16_t*)bt->data.f16 + i, 1);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_view_t* const btv = ccv_nnc_tensor_view_new(bt, CPU_TENSOR_NCHW(16F, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 16F, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST((ccv_nnc_tensor_t*)btv), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)bt->data.f16, (uint16_t*)hb->data.f16, (rows + destination_row_offset + 1) * destination_row_stride, "strided destination copy should match cpu reference");
	ccv_nnc_tensor_view_free(btv);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("format transform strided copy 16bf destination stride vectorized for mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 9;
	const int cols = 32;
	const int destination_row_stride = 48;
	const int destination_row_offset = 1;
	const int destination_col_offset = 4;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows, cols), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16BF, rows + destination_row_offset + 1, destination_row_stride), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, rows, cols), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16BF, rows + destination_row_offset + 1, destination_row_stride), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
	{
		const float v = (float)((i * 41) % 307 - 153) / 31;
		ccv_float_to_bfloat(&v, (uint16_t*)ha->data.f16 + i, 1);
	}
	for (i = 0; i < (rows + destination_row_offset + 1) * destination_row_stride; i++)
	{
		const float v = (float)(-7000 - i);
		ccv_float_to_bfloat(&v, (uint16_t*)hb->data.f16 + i, 1);
		ccv_float_to_bfloat(&v, (uint16_t*)bt->data.f16 + i, 1);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_view_t* const btv = ccv_nnc_tensor_view_new(bt, CPU_TENSOR_NCHW(16BF, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, GPU_TENSOR_NCHW(000, 16BF, rows, cols), DIM_ALLOC(destination_row_offset, destination_col_offset), DIM_ALLOC(destination_row_stride, 1));
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST((ccv_nnc_tensor_t*)btv), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST((ccv_nnc_tensor_t*)bv), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)bt->data.f16, (uint16_t*)hb->data.f16, (rows + destination_row_offset + 1) * destination_row_stride, "strided destination copy should match cpu reference");
	ccv_nnc_tensor_view_free(btv);
	ccv_nnc_tensor_view_free(bv);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
