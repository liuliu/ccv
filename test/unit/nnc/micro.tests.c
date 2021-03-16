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

TEST_CASE("represent convolution with micro ops")
{
	ccv_nnc_micro_io_t x = ccv_nnc_micro_input(4);
	ccv_nnc_micro_io_t xx = ccv_nnc_micro_reindex((const char*[]){
		"d0",
		"d1 - $kh + 1",
		"d2 - $kw + 1",
		"$kh",
		"$kw",
		"d3",
		"$kc"
	}, 7, x, (const char*[]){
		"i0",
		"i1 + i3",
		"i2 + i4",
		"i5"
	}, 4, x);
	ccv_nnc_micro_io_t w = ccv_nnc_micro_input(4);
	ccv_nnc_micro_io_t ww = ccv_nnc_micro_reindex((const char*[]){
		"d0",
		"d1 - $kh + 1",
		"d2 - $kw + 1",
		"$kh",
		"$kw",
		"d3",
		"$kc"
	}, 7, x, (const char*[]){
		"i6",
		"i3",
		"i4",
		"i5"
	}, 4, w);
	ccv_nnc_micro_io_t yy = ccv_nnc_micro_binary(CCV_NNC_MICRO_BINARY_OP_MUL, xx, ww);
	ccv_nnc_micro_io_t y5 = ccv_nnc_micro_reduce(CCV_NNC_MICRO_REDUCE_OP_SUM, 5, yy);
	ccv_nnc_micro_io_t y4 = ccv_nnc_micro_reduce(CCV_NNC_MICRO_REDUCE_OP_SUM, 4, y5);
	ccv_nnc_micro_io_t y = ccv_nnc_micro_reduce(CCV_NNC_MICRO_REDUCE_OP_SUM, 3, y4);
	ccv_nnc_micro_combine_t* combine = ccv_nnc_micro_combine_new((ccv_nnc_micro_io_t[]){
		x,
		w
	}, 2, (const char*[]){
		"$kh",
		"$kw",
		"$kc"
	}, 3, &y, 1);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 4, 4, 5), 0);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 3, 5), 0);
	ccv_nnc_tensor_t* const y_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, 2), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < 4 * 4 * 5; i++)
		x_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 3 * 3 * 5; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_micro_combine_interpret(combine, CCV_NNC_CUSTOM_FORWARD, TENSOR_LIST(x_tensor, w_tensor),
		(const ccv_nnc_micro_scalar_t[]){
			{
				.type = CCV_32S,
				.i32 = 3,
			},
			{
				.type = CCV_32S,
				.i32 = 3,
			},
			{
				.type = CCV_32S,
				.i32 = 2,
			}
		}, 3, TENSOR_LIST(y_tensor));
	ccv_nnc_tensor_t* const gt_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2, 2), 0);
	ccv_nnc_cmd_exec(CMD_CONVOLUTION_FORWARD(1, 2, 3, 3, 5), HINT((1, 1)), 0, TENSOR_LIST(x_tensor, w_tensor), TENSOR_LIST(gt_tensor), 0);
	REQUIRE_TENSOR_EQ(y_tensor, gt_tensor, "micro op composed convolution should match the existing convolution");
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(w_tensor);
	ccv_nnc_tensor_free(y_tensor);
	ccv_nnc_tensor_free(gt_tensor);
	ccv_nnc_micro_combine_free(combine);
}

#include "case_main.h"
