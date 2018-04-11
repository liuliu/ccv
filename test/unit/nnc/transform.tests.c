#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("data transfer between different tensor views")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(56, 56, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(24, 32, 64), 0);
	ccv_nnc_cmd_t cmd = CMD_DATA_TRANSFER_FORWARD();
	int i;
	for (i = 0; i < 128 * 56 * 56; i++)
		a->data.f32[i] = i;
	// 6 values, manageable.
	ccv_nnc_tensor_view_t a_view = ccv_nnc_tensor_view(a, DIM_ALLOC(2, 3, 4), DIM_ALLOC(4, 3, 2), a->info.dim);
	ccv_nnc_tensor_view_t b_view = ccv_nnc_tensor_view(b, DIM_ALLOC(2, 3, 4), DIM_ALLOC(0, 0, 0), b->info.dim);
	memset(b->data.f32, 0, sizeof(float) * 64 * 32 * 24);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&a_view), TENSOR_LIST((ccv_nnc_tensor_t*)&b_view), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, ONE_CPU_TENSOR(24, 32, 64), 0);
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
	REQUIRE_TENSOR_EQ(b, c, "64x32x24 tensor should be exactly the same.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
}

TEST_CASE("format transform between NHWC and NCHW tensors")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(4, 3, 2), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(2, 4, 3), 0);
	ccv_nnc_cmd_t cmd = CMD_FORMAT_TRANSFORM_FORWARD();
	int i;
	for (i = 0; i < 2 * 3 * 4; i++)
		a->data.f32[i] = i;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(2, 4, 3), 0);
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
	REQUIRE_TENSOR_EQ(b, c, "3x4x2 tensor should be exactly the same.");
	ccv_nnc_tensor_t* d = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(4, 3, 2), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(d), 0);
	REQUIRE_TENSOR_EQ(d, a, "2x3x4 tensor should be exactly the same.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
}

TEST_CASE("format transform between NHWC and NCHW tensor views")
{
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(7, 6, 5), 0);
	ccv_nnc_tensor_view_t a_view = ccv_nnc_tensor_view(a, DIM_ALLOC(4, 3, 2), DIM_ALLOC(3, 2, 1), a->info.dim);
	memset(a->data.f32, 0, sizeof(float) * 5 * 6 * 7);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(8, 10, 8), 0);
	memset(b->data.f32, 0, sizeof(float) * 8 * 10 * 8);
	ccv_nnc_tensor_view_t b_view = ccv_nnc_tensor_view(b, DIM_ALLOC(2, 4, 3), DIM_ALLOC(0, 0, 0), b->info.dim);
	ccv_nnc_cmd_t cmd = CMD_FORMAT_TRANSFORM_FORWARD();
	int i, j, k;
	for (i = 0; i < 4; i++)
		for (j = 0; j < 3; j++)
			for (k = 0; k < 2; k++)
				a->data.f32[(i + 3) * 5 * 6 + (j + 2) * 5 + (k + 1)] = k + j * 2 + i * 3 * 2;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&a_view), TENSOR_LIST((ccv_nnc_tensor_t*)&b_view), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(8, 10, 8), 0);
	memset(c->data.f32, 0, sizeof(float) * 8 * 10 * 8);
	c->data.f32[0] = 0;
	c->data.f32[1] = 2;
	c->data.f32[2] = 4;
	c->data.f32[8] = 6;
	c->data.f32[9] = 8;
	c->data.f32[10] = 10;
	c->data.f32[16] = 12;
	c->data.f32[17] = 14;
	c->data.f32[18] = 16;
	c->data.f32[24] = 18;
	c->data.f32[25] = 20;
	c->data.f32[26] = 22;
	c->data.f32[80] = 1;
	c->data.f32[81] = 3;
	c->data.f32[82] = 5;
	c->data.f32[88] = 7;
	c->data.f32[89] = 9;
	c->data.f32[90] = 11;
	c->data.f32[96] = 13;
	c->data.f32[97] = 15;
	c->data.f32[98] = 17;
	c->data.f32[104] = 19;
	c->data.f32[105] = 21;
	c->data.f32[106] = 23;
	REQUIRE_TENSOR_EQ(b, c, "3x4x2 tensor view should be exactly the same.");
	ccv_nnc_tensor_view_t c_view = ccv_nnc_tensor_view(c, DIM_ALLOC(2, 4, 3), DIM_ALLOC(0, 0, 0), c->info.dim);
	ccv_nnc_tensor_t* d = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(7, 6, 5), 0);
	memset(d->data.f32, 0, sizeof(float) * 5 * 6 * 7);
	ccv_nnc_tensor_view_t d_view = ccv_nnc_tensor_view(d, DIM_ALLOC(4, 3, 2), DIM_ALLOC(3, 2, 1), d->info.dim);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)&c_view), TENSOR_LIST((ccv_nnc_tensor_t*)&d_view), 0);
	REQUIRE_TENSOR_EQ(d, a, "2x3x4 tensor should be exactly the same.");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
}

#include "case_main.h"
