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
	ccv_nnc_cmd_exec(CMD_UPSAMPLE_BILINEAR_FORWARD(2, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_free(b);
	ccv_matrix_free(image);
	ccv_matrix_free(fimage);
}

#include "case_main.h"
