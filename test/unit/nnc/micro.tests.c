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
	ccv_nnc_micro_io_t x = ccv_nnc_micro_input(3);
	const char* shape = "(d0 + d1) * d2 + 4 - 10";
	const char* reindex = "-(i0 + d1)  * 11";
	ccv_nnc_micro_io_t y = ccv_nnc_micro_reindex(&shape, &reindex, 1, x);
	ccv_nnc_micro_combine_t* combine = ccv_nnc_micro_combine_new(&x, 1, 0, 0, &y, 1);
	ccv_nnc_micro_combine_free(combine);
}

#include "case_main.h"
