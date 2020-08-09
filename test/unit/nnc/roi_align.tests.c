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

TEST_CASE("compare ROI align forward")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 24, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	int i, j;
	for (i = 0; i < 12 * 24; i++)
		for (j = 0; j < 3; j++)
			a->data.f32[i * 3 + j] = i;
	b->data.f32[0] = 0 / 24; // x
	b->data.f32[1] = 0 / 12; // y
	b->data.f32[2] = 1; // w
	b->data.f32[3] = 1; // h
	// This should be look like no bi-linear filtering at all.
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	for (i = 0; i < 4; i++)
	{
		for (j = 0; j < 4; j++)
		{
			int x, y;
			float v = 0;
			for (y = 0; y < 3; y++)
				for (x = 0; x < 6; x++)
					v += a->data.f32[(i * 3 + y) * 24 * 3 + (j * 6 + x) * 3];
			ct->data.f32[(i * 4 + j) * 3] =
				ct->data.f32[(i * 4 + j) * 3 + 1] =
				ct->data.f32[(i * 4 + j) * 3 + 2] = v / (3 * 6);
		}
	}
	REQUIRE_TENSOR_EQ(c, ct, "should have no loss of accuracy");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("compare ROI align forward with average pool")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 24, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	int i, j;
	for (i = 0; i < 12 * 24; i++)
		for (j = 0; j < 3; j++)
			a->data.f32[i * 3 + j] = i;
	b->data.f32[0] = 0 / 24; // x
	b->data.f32[1] = 0 / 12; // y
	b->data.f32[2] = 1; // w
	b->data.f32[3] = 1; // h
	// This should be look like no bi-linear filtering at all.
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_FORWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_tensor_t* const ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_FORWARD(3, 6), HINT((3, 6)), 0, TENSOR_LIST(a), TENSOR_LIST(ct), 0);
	REQUIRE_TENSOR_EQ(c, ct, "should have no loss of accuracy");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("compare ROI align backward with average pool")
{
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 24, 3), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 4, 3), 0);
	int i, j;
	for (i = 0; i < 4 * 4; i++)
		for (j = 0; j < 3; j++)
			c->data.f32[i * 3 + j] = i;
	b->data.f32[0] = 0 / 24; // x
	b->data.f32[1] = 0 / 12; // y
	b->data.f32[2] = 1; // w
	b->data.f32[3] = 1; // h
	// This should be look like no bi-linear filtering at all.
	ccv_nnc_cmd_exec(CMD_ROI_ALIGN_BACKWARD(4, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(c, 0, b), TENSOR_LIST(a), 0);
	ccv_nnc_tensor_t* const at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 12, 24, 3), 0);
	ccv_nnc_cmd_exec(CMD_AVERAGE_POOL_BACKWARD(3, 6), HINT((3, 6)), 0, TENSOR_LIST(c), TENSOR_LIST(at), 0);
	REQUIRE_TENSOR_EQ(a, at, "should have no loss of accuracy");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(at);
}

#include "case_main.h"
