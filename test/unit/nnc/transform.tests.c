#include "case.h"
#include "ccv_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_CASE("data transfer between different tensor views")
{
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(128, 56, 56), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(64, 32, 24), 0);
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_DATA_TRANSFER, 0, ccv_nnc_default_cmd_params, 0);
	int i;
	for (i = 0; i < 128 * 56 * 56; i++)
		a->data.f32[i] = i;
	int a_ofs[] = {2, 3, 4};
	int b_ofs[] = {0, 0, 0};
	int dim[] = {4, 3, 2}; // 6 values, manageable.
	ccv_nnc_tensor_view_t a_view = ccv_nnc_tensor_view(a, a_ofs, dim);
	ccv_nnc_tensor_view_t b_view = ccv_nnc_tensor_view(b, b_ofs, dim);
	memset(b->data.f32, 0, sizeof(float) * 64 * 32 * 24);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_default_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&a_view), TENSOR_LIST((ccv_nnc_tensor_t*)&b_view), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(64, 32, 24), 0);
	memset(c->data.f32, 0, sizeof(float) * 64 * 32 * 24);
	c->data.f32[0] = 128 * 56 * 4 + 128 * 3 + 2;
	c->data.f32[1] = 128 * 56 * 4 + 128 * 3 + 3;
	c->data.f32[2] = 128 * 56 * 4 + 128 * 3 + 4;
	c->data.f32[3] = 128 * 56 * 4 + 128 * 3 + 5;
	c->data.f32[64] = 128 * 56 * 4 + 128 * (3 + 1) + 2;
	c->data.f32[65] = 128 * 56 * 4 + 128 * (3 + 1) + 3;
	c->data.f32[66] = 128 * 56 * 4 + 128 * (3 + 1) + 4;
	c->data.f32[67] = 128 * 56 * 4 + 128 * (3 + 1) + 5;
	c->data.f32[128] = 128 * 56 * 4 + 128 * (3 + 2) + 2;
	c->data.f32[129] = 128 * 56 * 4 + 128 * (3 + 2) + 3;
	c->data.f32[130] = 128 * 56 * 4 + 128 * (3 + 2) + 4;
	c->data.f32[131] = 128 * 56 * 4 + 128 * (3 + 2) + 5;
	c->data.f32[64 * 32] = 128 * 56 * (4 + 1) + 128 * 3 + 2;
	c->data.f32[64 * 32 + 1] = 128 * 56 * (4 + 1) + 128 * 3 + 3;
	c->data.f32[64 * 32 + 2] = 128 * 56 * (4 + 1) + 128 * 3 + 4;
	c->data.f32[64 * 32 + 3] = 128 * 56 * (4 + 1) + 128 * 3 + 5;
	c->data.f32[64 * 32 + 64] = 128 * 56 * (4 + 1) + 128 * (3 + 1) + 2;
	c->data.f32[64 * 32 + 65] = 128 * 56 * (4 + 1) + 128 * (3 + 1) + 3;
	c->data.f32[64 * 32 + 66] = 128 * 56 * (4 + 1) + 128 * (3 + 1) + 4;
	c->data.f32[64 * 32 + 67] = 128 * 56 * (4 + 1) + 128 * (3 + 1) + 5;
	c->data.f32[64 * 32 + 128] = 128 * 56 * (4 + 1) + 128 * (3 + 2) + 2;
	c->data.f32[64 * 32 + 129] = 128 * 56 * (4 + 1) + 128 * (3 + 2) + 3;
	c->data.f32[64 * 32 + 130] = 128 * 56 * (4 + 1) + 128 * (3 + 2) + 4;
	c->data.f32[64 * 32 + 131] = 128 * 56 * (4 + 1) + 128 * (3 + 2) + 5;
	REQUIRE_MATRIX_EQ(b, c, "64x32x24 tensor should be exactly the same.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("format transform between NHWC to NCHW")
{
	ccv_nnc_init();
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(2, 3, 4), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(3, 4, 2), 0);
	ccv_nnc_cmd_t cmd = ccv_nnc_cmd(CCV_NNC_COMPUTE_FORMAT_TRANSFORM, 0, ccv_nnc_default_cmd_params, 0);
	int i;
	for (i = 0; i < 2 * 3 * 4; i++)
		a->data.f32[i] = i;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_default_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(3, 4, 2), 0);
	c->data.f32[0] = 0;
	c->data.f32[1] = 2;
	c->data.f32[2] = 4;
	c->data.f32[3] = 6;
	c->data.f32[4] = 8;
	c->data.f32[5] = 10;
	c->data.f32[6] = 12;
	c->data.f32[7] = 14;
	c->data.f32[8] = 16;
	c->data.f32[9] = 18;
	c->data.f32[10] = 20;
	c->data.f32[11] = 22;
	c->data.f32[12] = 1;
	c->data.f32[13] = 3;
	c->data.f32[14] = 5;
	c->data.f32[15] = 7;
	c->data.f32[16] = 9;
	c->data.f32[17] = 11;
	c->data.f32[18] = 13;
	c->data.f32[19] = 15;
	c->data.f32[20] = 17;
	c->data.f32[21] = 19;
	c->data.f32[22] = 21;
	c->data.f32[23] = 23;
	REQUIRE_MATRIX_EQ(b, c, "3x4x2 tensor should be exactly the same.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

#include "case_main.h"
