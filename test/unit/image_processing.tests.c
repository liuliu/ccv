#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

TEST_CASE("image saturation")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/nature.png", &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_saturation(image, &b, 0, 0.5);
	REQUIRE_MATRIX_FILE_EQ(b, "data/nature.saturation.0.5.bin", "should be desaturated image");
	ccv_matrix_free(b);
	b = 0;
	ccv_saturation(image, &b, 0, 1.5);
	REQUIRE_MATRIX_FILE_EQ(b, "data/nature.saturation.1.5.bin", "should be oversaturated image");
	ccv_matrix_free(b);
	ccv_matrix_free(image);
}

TEST_CASE("image contrast")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/nature.png", &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* b = 0;
	ccv_contrast(image, &b, 0, 0.5);
	REQUIRE_MATRIX_FILE_EQ(b, "data/nature.contrast.0.5.bin", "should be decontrasted image");
	ccv_matrix_free(b);
	b = 0;
	ccv_contrast(image, &b, 0, 1.5);
	REQUIRE_MATRIX_FILE_EQ(b, "data/nature.contrast.1.5.bin", "should be overcontrasted image");
	ccv_matrix_free(b);
	ccv_matrix_free(image);
}

#include "case_main.h"
