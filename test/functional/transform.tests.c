#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("matrix decimal slice")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/chessbox.png", &image, CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_decimal_slice(image, &b, 0, 33.5, 41.5, 111, 91);
	REQUIRE_MATRIX_FILE_EQ(b, "data/chessbox.decimal.slice.bin", "should have data/chessbox.png sliced at (33.5, 41.5) with 111 x 91");
	ccv_matrix_free(image);
	ccv_matrix_free(b);
}

TEST_CASE("matrix perspective transform")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/chessbox.png", &image, CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_perspective_transform(image, &b, 0, cosf(CCV_PI / 6), 0, 0, 0, 1, 0, -sinf(CCV_PI / 6), 0, cosf(CCV_PI / 6));
	REQUIRE_MATRIX_FILE_EQ(b, "data/chessbox.perspective.transform.bin", "should have data/chessbox.png rotated along y-axis for 30");
	ccv_matrix_free(image);
	ccv_matrix_free(b);
}

#include "case_main.h"
