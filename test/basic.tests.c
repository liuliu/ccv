#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("sobel operation")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_sobel(image, &x, 0, 0, 1);
	ccv_dense_matrix_t* y = 0;
	ccv_sobel(image, &y, 0, 1, 0);
	ccv_matrix_free(image);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.sobel.x.bin", "should be sobel of partial derivative on x");
	REQUIRE_MATRIX_FILE_EQ(y, "data/chessbox.sobel.y.bin", "should be sobel of partial derivative on y");
	ccv_matrix_free(x);
	ccv_matrix_free(y);
	ccv_garbage_collect();
}

TEST_CASE("resample operation of CCV_INTER_AREA")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_resample(image, &x, 0, image->rows / 5, image->cols / 5, CCV_INTER_AREA);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.resample.bin", "should be a image of color dot");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
}

TEST_CASE("sample down operation with source offset (10, 10)")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_sample_down(image, &x, 0, 10, 10);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.sample_down.bin", "should be down sampled (/2) image with offset from source (10, 10)");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
}

TEST_CASE("sample up operation with source offset (10, 10)")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_sample_up(image, &x, 0, 10, 10);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.sample_up.bin", "should be down sampled (/2) image with offset from source (10, 10)");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
}

TEST_CASE("blur operation with sigma 10")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/nature.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_blur(image, &x, 0, sqrt(10));
	REQUIRE_MATRIX_FILE_EQ(x, "data/nature.blur.bin", "should be image applied with Gaussian filter with sigma sqrt(10)");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
}

TEST_CASE("flip operation")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_flip(image, &x, 0, CCV_FLIP_X);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.flip_x.bin", "flipped x-axis (around y-axis)");
	ccv_dense_matrix_t* y = 0;
	ccv_flip(image, &y, 0, CCV_FLIP_Y);
	REQUIRE_MATRIX_FILE_EQ(y, "data/chessbox.flip_y.bin", "flipped y-axis (around x-axis)");
	ccv_dense_matrix_t* xy = 0;
	ccv_flip(image, &xy, 0, CCV_FLIP_X | CCV_FLIP_Y);
	REQUIRE_MATRIX_FILE_EQ(xy, "data/chessbox.flip_xy.bin", "flipped xy-axis (rotated 180)");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_matrix_free(y);
	ccv_matrix_free(xy);
	ccv_garbage_collect();
}

#include "case_main.h"
