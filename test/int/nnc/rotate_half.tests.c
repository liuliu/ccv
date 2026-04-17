#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <3rdparty/dsfmt/dSFMT.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _rotate_half_u16(const uint16_t* const a, uint16_t* const b, const int row_count, const int dim)
{
	const int half = dim / 2;
	int i, j;
	for (i = 0; i < row_count; i++)
	{
		const uint16_t* const ap = a + i * dim;
		uint16_t* const bp = b + i * dim;
		for (j = 0; j < half; j++)
		{
			bp[j] = ap[j + half];
			bp[j + half] = ap[j];
		}
	}
}

TEST_CASE("compare rotate half with gpu ref")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int row_count = 7 * 5 * 9;
	const int dim = 128;
	const int count = row_count * dim;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 5, 9, 128), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 5, 9, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 7, 5, 9, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 5, 9, 128), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 7, 5, 9, 128), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < count; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t rotate_half = CMD_ROTATE_HALF_FORWARD();
	rotate_half.backend = CCV_NNC_BACKEND_GPU_REF;
	ccv_nnc_cmd_exec(rotate_half, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(hbt), 0);
	REQUIRE_TENSOR_EQ(hb, hbt, "gpu ref should match CPU reference implementation");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare rotate half backward in half precision with gpu ref")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	const int row_count = 13 * 17;
	const int dim = 64;
	const int count = row_count * dim;
	ccv_nnc_tensor_t* const hg32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 13, 17, 64), 0);
	ccv_nnc_tensor_t* const hht = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 13, 17, 64), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2);
	int i;
	for (i = 0; i < count; i++)
		hg32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_half_precision(hg32->data.f32, (uint16_t*)hg->data.f16, count);
	_rotate_half_u16((uint16_t*)hg->data.f16, (uint16_t*)hht->data.f16, row_count, dim);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg), TENSOR_LIST(g), 0);
	ccv_nnc_cmd_t rotate_half = CMD_ROTATE_HALF_BACKWARD();
	rotate_half.backend = CCV_NNC_BACKEND_GPU_REF;
	ccv_nnc_cmd_exec(rotate_half, ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(h), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h), TENSOR_LIST(hh), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)hht->data.f16, (uint16_t*)hh->data.f16, count, "gpu ref should rotate half-precision gradient by half");
	ccv_nnc_tensor_free(hg32);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(hht);
}

TEST_CASE("compare rotate half in bfloat precision with gpu ref")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int row_count = 11 * 19;
	const int dim = 96;
	const int count = row_count * dim;
	ccv_nnc_tensor_t* const ha32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 11, 19, 96), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 11, 19, 96), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 11, 19, 96), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 11, 19, 96), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 11, 19, 96), 0);
	ccv_nnc_tensor_t* const hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 11, 19, 96), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3);
	int i;
	for (i = 0; i < count; i++)
		ha32->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_bfloat(ha32->data.f32, (uint16_t*)ha->data.f16, count);
	_rotate_half_u16((uint16_t*)ha->data.f16, (uint16_t*)hbt->data.f16, row_count, dim);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_t rotate_half = CMD_ROTATE_HALF_FORWARD();
	rotate_half.backend = CCV_NNC_BACKEND_GPU_REF;
	ccv_nnc_cmd_exec(rotate_half, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ(uint16_t, (uint16_t*)hbt->data.f16, (uint16_t*)hb->data.f16, count, "gpu ref should rotate bfloat-precision tensor by half");
	ccv_nnc_tensor_free(ha32);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("compare rotate half with mpsgraph fallback")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	float ap[] = {
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 8), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 8), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	float bp[] = {
		5, 6, 7, 8, 1, 2, 3, 4,
		13, 14, 15, 16, 9, 10, 11, 12,
		21, 22, 23, 24, 17, 18, 19, 20,
		29, 30, 31, 32, 25, 26, 27, 28,
	};
	ccv_nnc_tensor_t const bt = ccv_nnc_tensor(bp, CPU_TENSOR_NHWC(32F, 4, 8), 0);
	REQUIRE_TENSOR_EQ(hb, &bt, "mps should rotate the last dim by half");
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("compare rotate half backward with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
	};
	ccv_nnc_tensor_t* const hg = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 8), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 8), 0);
	ccv_nnc_tensor_t* const hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hg), TENSOR_LIST(g), 0);
	ccv_nnc_cmd_exec(CMD_ROTATE_HALF_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g), TENSOR_LIST(h), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h), TENSOR_LIST(hh), 0);
	float hp[] = {
		5, 6, 7, 8, 1, 2, 3, 4,
		13, 14, 15, 16, 9, 10, 11, 12,
	};
	ccv_nnc_tensor_t const ht = ccv_nnc_tensor(hp, CPU_TENSOR_NHWC(32F, 2, 8), 0);
	REQUIRE_TENSOR_EQ(hh, &ht, "mps should rotate gradient by half");
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(hh);
}

#include "case_main.h"
