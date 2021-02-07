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

TEST_CASE("min of two tensors")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	int i;
	for (i = 0; i < 2 * 2 * 3; i++)
		a->data.f32[i] = i;
	for (i = 0; i < 2 * 2 * 3; i++)
		b->data.f32[i] = i - 1;
	ccv_nnc_cmd_exec(CMD_MIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	REQUIRE_TENSOR_EQ(b, c, "c should be the minimal of the two");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("min of two tensor views")
{
}

TEST_CASE("min of two tensors backward")
{
}

TEST_CASE("min of two tensor views backward")
{
}

#include "case_main.h"
