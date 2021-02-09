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

TEST_CASE("roi align forward with NCHW")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hcf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 32 * 32 * 128; i++)
		hat->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hat), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc), TENSOR_LIST(hcf), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hat, hb), TENSOR_LIST(hct), 0);
	REQUIRE_TENSOR_EQ(hct, hcf, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hcf);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align forward with NHWC")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 32 * 32 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hct), 0);
	REQUIRE_TENSOR_EQ(hct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align forward with NCHW, batch of 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hcf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 32 * 32 * 128; i++)
		hat->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hat), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc), TENSOR_LIST(hcf), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hat, hb), TENSOR_LIST(hct), 0);
	REQUIRE_TENSOR_EQ(hct, hcf, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(hcf);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align forward with NHWC, batch of 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 32 * 32 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	hb->data.f32[4] = 0 / 32;
	hb->data.f32[5] = 0 / 32;
	hb->data.f32[6] = 1;
	hb->data.f32[7] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(hct), 0);
	REQUIRE_TENSOR_EQ(hct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align backward with NCHW")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const haf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 4 * 4 * 128; i++)
		hct->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hct), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, hb), TENSOR_LIST(c, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(haf), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hct, 0, hb), TENSOR_LIST(hat), 0);
	REQUIRE_TENSOR_EQ(hat, haf, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(haf);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align backward with NHWC")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 32, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 4 * 4 * 128; i++)
		hc->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, hb), TENSOR_LIST(c, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, 0, hb), TENSOR_LIST(hat), 0);
	REQUIRE_TENSOR_EQ(hat, ha, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
}

TEST_CASE("roi align backward with NCHW, batch of 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 128, 32, 32), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 128, 4, 4), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const haf = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 4 * 4 * 128; i++)
		hct->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hct), TENSOR_LIST(hc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, hb), TENSOR_LIST(c, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(haf), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hct, 0, hb), TENSOR_LIST(hat), 0);
	REQUIRE_TENSOR_EQ(hat, haf, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
	ccv_nnc_tensor_free(haf);
	ccv_nnc_tensor_free(hct);
}

TEST_CASE("roi align backward with NHWC, batch of 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_ROI_ALIGN_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 4, 128), 0);
	ccv_nnc_tensor_t* const hat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 32, 32, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 2 * 4 * 4 * 128; i++)
		hc->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	hb->data.f32[0] = 0 / 32;
	hb->data.f32[1] = 0 / 32;
	hb->data.f32[2] = 1;
	hb->data.f32[3] = 1;
	hb->data.f32[4] = 0 / 32;
	hb->data.f32[5] = 0 / 32;
	hb->data.f32[6] = 1;
	hb->data.f32[7] = 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, hb), TENSOR_LIST(c, b), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(hc, 0, hb), TENSOR_LIST(hat), 0);
	REQUIRE_TENSOR_EQ(hat, ha, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(hat);
}

#include "case_main.h"
