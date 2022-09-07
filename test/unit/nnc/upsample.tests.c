#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "case.h"
#include "ccv_case.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("upsample chessbox")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = (ccv_nnc_tensor_t*)fimage;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)b, "data/upsample.forward.bin", "the forward of upsample should be equal");
	ccv_nnc_tensor_free(b);
	ccv_matrix_free(image);
	ccv_matrix_free(fimage);
}

TEST_CASE("downsample chessbox")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = (ccv_nnc_tensor_t*)fimage;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)b, "data/upsample.backward.bin", "the backward of upsample should be equal");
	ccv_nnc_tensor_free(b);
	ccv_matrix_free(image);
	ccv_matrix_free(fimage);
}

TEST_CASE("upsample chessbox in NCHW")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows, image->cols), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows * 2, image->cols * 2), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_BILINEAR, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows * 2, image->cols * 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_free(b);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)bt, "data/upsample.forward.bin", "the forward of upsample should be equal");
	ccv_matrix_free(image);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("downsample chessbox in NCHW")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/chessbox.png", &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	ccv_dense_matrix_t* fimage = 0;
	ccv_shift(image, (ccv_matrix_t**)&fimage, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows, image->cols), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)fimage), TENSOR_LIST(a), 0);
	ccv_matrix_free(fimage);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows / 2, image->cols / 2), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_BILINEAR,  2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, image->rows / 2, image->cols / 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_free(b);
	REQUIRE_MATRIX_FILE_EQ((ccv_matrix_t*)bt, "data/upsample.backward.bin", "the backward of upsample should be equal");
	ccv_matrix_free(image);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("upsample nearest")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2), 0);
	int i;
	for (i = 0; i < 8; i++)
		a->data.f32[i] = i;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 2), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float bt[32] = {
		0, 1, 0, 1, 2, 3, 2, 3,
		0, 1, 0, 1, 2, 3, 2, 3,
		4, 5, 4, 5, 6, 7, 6, 7,
		4, 5, 4, 5, 6, 7, 6, 7,
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 32, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("downsample nearest")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 2), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 2), 0);
	int i;
	for (i = 0; i < 32; i++)
		a->data.f32[i] = i;
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float bt[8] = {
		0 + 2 + 8 + 10, 1 + 3 + 9 + 11, 4 + 6 + 12 + 14, 5 + 7 + 13 + 15,
		16 + 18 + 24 + 26, 17 + 19 + 25 + 27, 20 + 22 + 28 + 30, 21 + 23 + 29 + 31,

	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 8, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("upsample nearest in NCHW")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 2, 2), 0);
	int i;
	for (i = 0; i < 8; i++)
		a->data.f32[i] = i;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 4, 4), 0);
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_FORWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float bt[32] = {
		0, 0, 1, 1,
		0, 0, 1, 1,
		2, 2, 3, 3,
		2, 2, 3, 3,
		4, 4, 5, 5,
		4, 4, 5, 5,
		6, 6, 7, 7,
		6, 6, 7, 7,
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 32, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("downsample nearest in NCHW")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 4, 4), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2, 2, 2), 0);
	int i;
	for (i = 0; i < 32; i++)
		a->data.f32[i] = i;
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BACKWARD(CCV_NNC_UPSAMPLE_NEAREST, 2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float bt[8] = {
		0 + 1 + 4 + 5, 2 + 3 + 6 + 7,
		8 + 9 + 12 + 13, 10 + 11 + 14 + 15,
		16 + 17 + 20 + 21, 18 + 19 + 22 + 23,
		24 + 25 + 28 + 29, 26 + 27 + 30 + 31,
	};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, b->data.f32, bt, 8, 1e-5, "should match ground truth");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
