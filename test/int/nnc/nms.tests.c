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

TEST_CASE("compare nms forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_NMS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_NMS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1000, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1000, 5), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32S, 1000), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1000, 5), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1000, 5), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 1000), 0);
	int i;
	for (i = 0; i < 1000; i++)
	{
		ha->data.f32[i * 5] = i;
		ha->data.f32[i * 5 + 1] = i;
		ha->data.f32[i * 5 + 2] = 0;
		ha->data.f32[i * 5 + 3] = 2;
		ha->data.f32[i * 5 + 4] = 1;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_NMS_FORWARD(0.3), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b, c), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, c), TENSOR_LIST(hb, hc), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1000, 5), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 1000), 0);
	ccv_nnc_cmd_exec(CMD_NMS_FORWARD(0.3), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt, hct), 0);
	REQUIRE_TENSOR_EQ(hbt, hb, "should be equal");
	REQUIRE_ARRAY_EQ(int, hc->data.i32, hct->data.i32, 1000, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hbt);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("compare nms backward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_NMS_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_NMS_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 100, 5), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 100, 5), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32S, 100), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 100, 5), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 100, 5), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, 100), 0);
	int i;
	for (i = 0; i < 100; i++)
	{
		ha->data.f32[i * 5] = i;
		ha->data.f32[i * 5 + 1] = i;
		ha->data.f32[i * 5 + 2] = 0;
		ha->data.f32[i * 5 + 3] = 2;
		ha->data.f32[i * 5 + 4] = 1;
	}
	for (i = 0; i < 10; i++)
		hc->data.i32[i] = 10 - i;
	for (i = 10; i < 100; i++)
		hc->data.i32[i] = -1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hc), TENSOR_LIST(a, c), 0);
	ccv_nnc_cmd_exec(CMD_NMS_BACKWARD(0.3), ccv_nnc_no_hint, 0, TENSOR_LIST(a, 0, 0, 0, c), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 100, 5), 0);
	for (i = 0; i < 100 * 5; i++)
		bt->data.f32[i] = 0;
	for (i = 1; i < 11; i++)
	{
		const int j = 10 - i;
		bt->data.f32[i * 5] = j;
		bt->data.f32[i * 5 + 1] = j;
		bt->data.f32[i * 5 + 2] = 0;
		bt->data.f32[i * 5 + 3] = 2;
		bt->data.f32[i * 5 + 4] = 1;
	}
	REQUIRE_TENSOR_EQ(hb, bt, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(bt);
}

#include "case_main.h"
