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

TEST_CASE("reduce sum forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce sum forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce sum backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_SUM_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hb->data.f32[0] = 1;
	hb->data.f32[1] = 2;
	hb->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(ha), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_SUM_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	REQUIRE_TENSOR_EQ(ha, at, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(at);
}

TEST_CASE("reduce mean forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce mean forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce mean backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MEAN_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hb->data.f32[0] = 1;
	hb->data.f32[1] = 2;
	hb->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(ha), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MEAN_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	REQUIRE_TENSOR_EQ(ha, at, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(at);
}

TEST_CASE("reduce logsumexp forward and backward on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 37;
	const int columns = 257;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = 40 * dsfmt_genrand_open_close(&dsfmt) - 20;
	for (i = 0; i < rows; i++)
		hg->data.f32[i] = 2 * dsfmt_genrand_open_close(&dsfmt) - 1;
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hh), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, 1), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hg), TENSOR_LIST(a, g), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_BACKWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, h), TENSOR_LIST(bt, ht), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, rows, 1e-5, "MPS logsumexp should match CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hh->data.f32, ht->data.f32, rows * columns, 1e-5, "MPS logsumexp backward should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(ht);
}

TEST_CASE("reduce logsumexp over multiple axes in NCHW on MPS")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_LOGSUMEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 17, 19), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 1, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < 3 * 17 * 19; i++)
		ha->data.f32[i] = 40 * dsfmt_genrand_open_close(&dsfmt) - 20;
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, 17, 19), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_LOGSUMEXP_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 3, 1e-5, "multi-axis MPS logsumexp should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce max forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MAX_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MAX_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce min forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_MIN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_MIN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmin with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMIN_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMIN_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ARGMIN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMIN_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with bfloat")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hrounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	int i;
	for (i = 0; i < 10 * 3 * 5 * 3; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_bfloat(ha32->data.f32, (uint16_t*)ha->data.f16, 10 * 3 * 5 * 3);
	ccv_bfloat_to_float((uint16_t*)ha->data.f16, hrounded->data.f32, 10 * 3 * 5 * 3);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(hrounded), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 3, 5, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10, 1, 5, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hrounded);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("argmax with bfloat over large axis")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int length = 2048;
	const int winner = 1500;
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, length), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, length), 0);
	ccv_nnc_tensor_t* const hrounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, length), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 1), 0);
	int i;
	for (i = 0; i < length; i++)
		ha32->data.f32[i] = -2;
	ha32->data.f32[0] = 9.875;
	ha32->data.f32[599] = 19.25;
	ha32->data.f32[1044] = 19.5;
	ha32->data.f32[winner] = 19.625;
	ccv_float_to_bfloat(ha32->data.f32, (uint16_t*)ha->data.f16, length);
	ccv_bfloat_to_float((uint16_t*)ha->data.f16, hrounded->data.f32, length);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(hrounded), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, length), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	REQUIRE_EQ(winner, bt->data.i32[0], "large-axis argmax should find winner");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hrounded);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps argmax over multiple MFA partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 5;
	const int columns = 8193;
	const int winners[] = { 0, 4095, 4096, 8191, 8192 };
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
			ha->data.f32[i * columns + j] = -2;
		ha->data.f32[i * columns + winners[i]] = 10;
	}
	ccv_nnc_cmd_exec(CMD_ARGMAX_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t cmd = CMD_ARGMAX_FORWARD(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "partitioned MFA argmax should match CPU");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps gumbel argmax one-pass MFA is reproducible")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 64;
	const int columns = 257;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i;
	for (i = 0; i < rows * columns; i++)
		ha->data.f32[i] = 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const ha_strided = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns + 1), 0);
	ccv_nnc_tensor_t* const a_strided = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns + 1), 0);
	ccv_nnc_tensor_view_t* const av = ccv_nnc_tensor_view_new(a_strided, GPU_TENSOR_NHWC(000, 32F, rows, columns), ccv_nnc_no_ofs, DIM_ALLOC(columns + 1, 1));
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_tensor_t* const e = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	for (i = 0; i < rows * (columns + 1); i++)
		ha_strided->data.f32[i] = 0;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_strided), TENSOR_LIST(a_strided), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(d), stream_context);
	ccv_nnc_stream_context_set_seed(stream_context, 177);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av), TENSOR_LIST(e), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const hd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_tensor_t* const he = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, c, d, e), TENSOR_LIST(hb, hc, hd, he), 0);
	REQUIRE_TENSOR_EQ(hb, hc, "the same stream seed should reproduce MFA gumbel argmax");
	REQUIRE_TENSOR_EQ(hd, he, "the same stream seed should reproduce MPSGraph gumbel argmax");
	int different = 0;
	for (i = 0; i < rows; i++)
	{
		REQUIRE(hb->data.i32[i] >= 0 && hb->data.i32[i] < columns, "MFA gumbel argmax should return a valid index");
		different |= hb->data.i32[i] != hb->data.i32[0];
	}
	REQUIRE(different, "equal logits should sample more than one category");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha_strided);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(a_strided);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(e);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hd);
	ccv_nnc_tensor_free(he);
}

TEST_CASE("mps gumbel argmax over multiple MFA partitions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 4;
	const int columns = 8193;
	const int winners[] = { 17, 4095, 4096, 8192 };
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, columns), 0);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
			ha->data.f32[i * columns + j] = -100;
		ha->data.f32[i * columns + winners[i]] = 100;
	}
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, columns), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_set_seed(stream_context, 31);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	for (i = 0; i < rows; i++)
		REQUIRE_EQ(winners[i], hb->data.i32[i], "partitioned MFA gumbel argmax should find the dominant logit");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps gumbel argmax falls back for a non-final axis")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS));
	const int batches = 2;
	const int categories = 3;
	const int width = 5;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, batches, categories, width), 0);
	int i, j, k;
	for (i = 0; i < batches; i++)
		for (j = 0; j < categories; j++)
			for (k = 0; k < width; k++)
				ha->data.f32[(i * categories + j) * width + k] = j == 1 ? 100 : -100;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, batches, categories, width), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, batches, 1, width), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_set_seed(stream_context, 47);
	ccv_nnc_cmd_t cmd = CMD_GUMBEL_ARGMAX(1);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	assert(cmd.backend >= 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, batches, 1, width), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	for (i = 0; i < batches * width; i++)
		REQUIRE_EQ(1, hb->data.i32[i], "MPSGraph gumbel argmax should find the dominant logit");
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("reduce norm2 forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce norm2 forward noop")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce norm2 backward")
{
	GUARD_ELSE_RETURN((ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS)) &&
		(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_BACKWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_NORM2_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	hg->data.f32[0] = 1;
	hg->data.f32[1] = 2;
	hg->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hb), TENSOR_LIST(hh), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg), TENSOR_LIST(g), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_NORM2_BACKWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(h), 0);
	ccv_nnc_tensor_t* const ht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h), TENSOR_LIST(ht), 0);
	REQUIRE_TENSOR_EQ(hh, ht, "result should be equal");
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ht);
}

TEST_CASE("reduce isnan float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	ha->data.f32[0] = NAN;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("reduce isnan in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) || ccv_nnc_cmd_ok(CCV_NNC_REDUCE_ISNAN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ha->data.f32[0] = NAN;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ha16), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_REDUCE_ISNAN_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

#include "case_main.h"
