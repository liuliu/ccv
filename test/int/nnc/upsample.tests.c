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

TEST_CASE("upsample bilinear NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.forward.bin", "the forward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

TEST_CASE("upsample bilinear NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
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
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(at), TENSOR_LIST(bt), 0);
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

TEST_CASE("downsample bilinear NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows, image->cols, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)hb, "../../unit/nnc/data/upsample.backward.bin", "the backward of upsample should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_matrix_free(image);
}

TEST_CASE("downsample bilinear NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
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
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(at), TENSOR_LIST(bt), 0);
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

TEST_CASE("downsample bilinear NHWC in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* b16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 14 * 14 * 5; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, hbt->data.f32, 7 * 7 * 5, 1e-2, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("downsample bilinear NCHW in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* b16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 15 * 14 * 6; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, hbt->data.f32, 15 * 7 * 3, 1e-2, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("upsample nearest NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 15, 15, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 30, 30, 5), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 15, 15, 5), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 30, 30, 5), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 30, 30, 5), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 15 * 15 * 5; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("upsample nearest NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 15, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 30, 10), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 15, 5), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 30, 10), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 30, 10), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 15 * 15 * 5; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("downsample nearest NHWC in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 14 * 14 * 5; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("downsample nearest NCHW in float")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 15 * 14 * 6; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("downsample nearest NHWC in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* b16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 14, 14, 5), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 7, 5), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 14 * 14 * 5; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, hbt->data.f32, 7 * 7 * 5, 1e-2, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("downsample nearest NCHW in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* a16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* b16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 14, 6), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 15, 7, 3), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 15 * 14 * 6; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(a16), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, hbt->data.f32, 15 * 7 * 3, 1e-2, "CPU and GPU results should match.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(hbt);
}

#include "case_main.h"
