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

#include "case_main.h"
