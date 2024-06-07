#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

#ifdef HAVE_LIBPNG
TEST_CASE("LSSC for natural image")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/nature.png", &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* i32 = 0;
	ccv_shift(image, (ccv_matrix_t**)&i32, CCV_32F, 0, 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows, image->cols), 0);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)i32), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NCHW(16F, 3, image->rows, image->cols);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_param_t b_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &a_params, 1, ccv_nnc_no_hint, &b_params, 1);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const c16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3, image->rows, image->cols), 0);
	ccv_float_to_half_precision(a->data.f32, (uint16_t*)a16->data.f16, 3 * image->rows * image->cols);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(c16), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, c->data.f32, 3 * image->rows * image->cols);
	ccv_nnc_cmd_exec(CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST((ccv_nnc_tensor_t*)i32), 0);
	ccv_dense_matrix_t* restore = 0;
	ccv_shift(i32, (ccv_matrix_t**)&restore, CCV_8U, 0, 0);
	REQUIRE_MATRIX_FILE_EQ(restore, "data/nature.lssc.bin", "the natural image should be equal");
	ccv_matrix_free(i32);
	ccv_matrix_free(restore);
	ccv_matrix_free(image);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(c16);
}
#endif

TEST_CASE("LSSC should give exact result for 2-value data")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 5, 10, 11), 0);
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NCHW(16F, 5, 10, 11);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_param_t b_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &a_params, 1, ccv_nnc_no_hint, &b_params, 1);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const c16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 5, 10, 11), 0);
	dsfmt_t dsfmt;
	int i, j, k, x, y;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 5 * 10 * 11; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_half_precision(a->data.f32, (uint16_t*)a16->data.f16, 5 * 10 * 11);
	// Now, for each 4x4 region, pick 2 values and assign to the rest of that region.
	for (k = 0; k < 5; k++)
	{
		for (i = 0; i < 10; i += 4)
			for (j = 0; j < 11; j += 4)
			{
				ccv_float16_t v[2] = { a16->data.f16[k * 11 * 10 + i * 11 + j], a16->data.f16[k * 11 * 10 + i * 11 + j + 1] };
				for (y = 0; y < ccv_min(i + 4, 10) - i; y++)
					for (x = 0; x < ccv_min(j + 4, 11) - j; x++)
						a16->data.f16[k * 11 * 10 + (i + y) * 11 + j + x] = v[dsfmt_genrand_uint32(&dsfmt) % 2];
			}
	}
	ccv_half_precision_to_float((uint16_t*)a16->data.f16, a->data.f32, 5 * 10 * 11);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(c16), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, c->data.f32, 5 * 10 * 11);
	REQUIRE_TENSOR_EQ(a, c, "decompressed tensor should be equal to original");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(c16);
}

#include "case_main.h"
