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

TEST_CASE("sort a 1d tensor")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10000), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	ccv_nnc_tensor_t* const indicest = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt, indicest), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(hb, hindices), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should be equal");
	REQUIRE_TENSOR_EQ(hindices, indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(indicest);
}

TEST_CASE("sort a 1d tensor, descending")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) + (i / 1000);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10000), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10000), 0);
	ccv_nnc_tensor_t* const indicest = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt, indicest), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(hb, hindices), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should be equal");
	REQUIRE_TENSOR_EQ(hindices, indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(indicest);
}

TEST_CASE("sort a 1d tensor, int")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10000; i++)
		ha->data.i32[i] = (int)dsfmt_genrand_uint32(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indicest = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt, indicest), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(hb, hindices), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should be equal");
	REQUIRE_TENSOR_EQ(hindices, indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(indicest);
}

TEST_CASE("sort a 1d tensor, descending, int")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 10000; i++)
		ha->data.i32[i] = (int)dsfmt_genrand_uint32(&dsfmt);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, indices), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_tensor_t* const indicest = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 10000), 0);
	ccv_nnc_cmd_exec(CMD_SORT_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt, indicest), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, indices), TENSOR_LIST(hb, hindices), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "should be equal");
	REQUIRE_TENSOR_EQ(hindices, indicest, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(indicest);
}

#include "case_main.h"
