#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("represent convolution with micro ops")
{
	ccv_nnc_micro_io_t x = ccv_nnc_micro_input(4);
	ccv_nnc_micro_io_t y = ccv_nnc_micro_reindex((const char*[]){
		"d0",
		"d1 - $kh + 1",
		"d2 - $kw + 1",
		"$kh",
		"$kw",
		"d3",
		"$kc"
	}, 6, (const char*[]){
		"i0",
		"i1 + i3",
		"i2 + i4",
		"i5"
	}, 4, x);
	ccv_nnc_micro_combine_t* combine = ccv_nnc_micro_combine_new(&x, 1, (const char*[]){
		"$kh",
		"$kw",
		"$kc"
	}, 2, &y, 1);
	ccv_nnc_micro_combine_free(combine);
}

#include "case_main.h"
