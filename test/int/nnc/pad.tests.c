#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("implement pad zero 1d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 9), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 9), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 9), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 2d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 3d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 4), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 3, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 4d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4, 3, 2), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ha->data.f32[6] = 7;
	ha->data.f32[7] = 8;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad replicate 1d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 9), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 9), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 9), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad replicate 2d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad replicate 3d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 3), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 4), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 3, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad replicate 4d")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 1), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4, 3, 2), 0);
	ha->data.f32[0] = 1;
	ha->data.f32[1] = 2;
	ha->data.f32[2] = 3;
	ha->data.f32[3] = 4;
	ha->data.f32[4] = 5;
	ha->data.f32[5] = 6;
	ha->data.f32[6] = 7;
	ha->data.f32[7] = 8;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 1), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 1d gradient")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 9), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 9; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 9), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 6), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2), (1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 6), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 2d gradient")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 20; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 3d gradient")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 3, 4), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 3), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 3 * 3 * 4; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 3, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (1, 1, 0), (1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("implement pad zero 4d gradient")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4, 3, 2), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 1), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i <  5 * 4 * 3 * 2; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 5, 4, 3, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 2, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_PAD_BACKWARD(CCV_NNC_PAD_ZERO, (2, 1, 1, 0), (1, 1, 0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "result should be equal");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(bt);
}

#include "case_main.h"
