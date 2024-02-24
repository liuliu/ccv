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

TEST_CASE("implement pad zero")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_ZERO, (2, 1), (1, 0)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 1, 2, 3,
		0, 4, 5, 6,
		0, 0, 0, 0,
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 5, 4), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("implement pad replicate")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 5), 0);
	a->data.f32[0] = 1;
	a->data.f32[1] = 2;
	a->data.f32[2] = 3;
	a->data.f32[3] = 4;
	a->data.f32[4] = 5;
	a->data.f32[5] = 6;
	ccv_nnc_cmd_exec(CMD_PAD_FORWARD(CCV_NNC_PAD_REPLICATE, (2, 1), (1, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	float btp[] = {
		1, 1, 2, 3, 3,
		1, 1, 2, 3, 3,
		1, 1, 2, 3, 3,
		4, 4, 5, 6, 6,
		4, 4, 5, 6, 6,
	};
	ccv_nnc_tensor_t bt = ccv_nnc_tensor(btp, CPU_TENSOR_NHWC(32F, 5, 5), 0);
	REQUIRE_TENSOR_EQ(b, &bt, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

#include "case_main.h"
