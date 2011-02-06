#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("sobel operation")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/chessbox.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_sobel(image, &x, 0, 0, 1);
	REQUIRE_MATRIX_FILE_EQ(x, "data/chessbox.sobel.x.bin", "should be sobel of partial derivative on x");
	ccv_dense_matrix_t* y = 0;
	ccv_sobel(image, &y, 0, 1, 0);
	REQUIRE_MATRIX_FILE_EQ(y, "data/chessbox.sobel.y.bin", "should be sobel of partial derivative on y");
	ccv_dense_matrix_t* x3 = 0;
	ccv_sobel(image, &x3, 0, 0, 3);
	REQUIRE_MATRIX_FILE_EQ(x3, "data/chessbox.sobel.x.3.bin", "should be sobel of partial derivative on x within 3x3 window");
	ccv_dense_matrix_t* y3 = 0;
	ccv_sobel(image, &y3, 0, 3, 0);
	REQUIRE_MATRIX_FILE_EQ(y3, "data/chessbox.sobel.y.3.bin", "should be sobel of partial derivative on y within 3x3 window");
	ccv_dense_matrix_t* x5 = 0;
	ccv_sobel(image, &x5, 0, 0, 5);
	REQUIRE_MATRIX_FILE_EQ(x5, "data/chessbox.sobel.x.5.bin", "should be sobel of partial derivative on x within 5x5 window");
	ccv_dense_matrix_t* y5 = 0;
	ccv_sobel(image, &y5, 0, 5, 0);
	REQUIRE_MATRIX_FILE_EQ(y5, "data/chessbox.sobel.y.5.bin", "should be sobel of partial derivative on y within 5x5 window");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_matrix_free(y);
	ccv_matrix_free(x3);
	ccv_matrix_free(y3);
	ccv_matrix_free(x5);
	ccv_matrix_free(y5);
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

TEST_CASE("canny edge detector")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/blackbox.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	ccv_canny(image, &x, 0, 3, 36, 36 * 3);
	REQUIRE_MATRIX_FILE_EQ(x, "data/blackbox.canny.bin", "Canny edge detector on artificial image");
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
}

#include "case_main.h"
