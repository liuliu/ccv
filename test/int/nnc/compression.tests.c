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

TEST_CASE("LSSC should give exact result from GPU")
{
	const int N = 128;
	const int C = 64;
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMPRESSION_LSSC_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_COMPRESSION_LSSC_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, 113, 114), 0);
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NCHW(16F, N, C, 113, 114);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_param_t b_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &a_params, 1, ccv_nnc_no_hint, &b_params, 1);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const c16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, 113, 114), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < N * C * 113 * 114; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_half_precision(a->data.f32, (uint16_t*)a16->data.f16, N * C * 113 * 114);
	ccv_half_precision_to_float((uint16_t*)a16->data.f16, a->data.f32, N * C * 113 * 114);
	int x, y, ix, iy;
	for (i = 0; i < N * C; i++)
		for (y = 0; y < 113; y += 4)
			for (x = 0; x < 114; x += 4)
			{
				float* const ap = a->data.f32 + x + y * 114 + i * 113 * 114;
				float v[4] = { ap[0], ap[1], ap[0] * 2 / 3 + ap[1] / 3, ap[0] / 3 + ap[1] * 2 / 3 };
				for (iy = 0; iy < ccv_min(y + 4, 113) - y; iy++)
					for (ix = 0; ix < ccv_min(x + 4, 114) - x; ix++)
						ap[iy * 114 + ix] = v[dsfmt_genrand_uint32(&dsfmt) % 4];
				ap[0] = v[0];
				ap[1] = v[1]; // Make sure we still have max min.
			}
	ccv_float_to_half_precision(a->data.f32, (uint16_t*)a16->data.f16, N * C * 113 * 114);
	ccv_half_precision_to_float((uint16_t*)a16->data.f16, a->data.f32, N * C * 113 * 114);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(c16), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, c->data.f32, N * C * 113 * 114);
	// Compare against GPU computation
	ccv_nnc_tensor_param_t ag_params = GPU_TENSOR_NCHW(000, 16F, N, C, 113, 114);
	ccv_nnc_tensor_t* const a16g = ccv_nnc_tensor_new(0, ag_params, 0);
	ccv_nnc_tensor_param_t bg_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &ag_params, 1, ccv_nnc_no_hint, &bg_params, 1);
	ccv_nnc_tensor_t* const b16g = ccv_nnc_tensor_new(0, bg_params, 0);
	ccv_nnc_tensor_t* const c16g = ccv_nnc_tensor_new(0, ag_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(a16g), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16g), TENSOR_LIST(b16g), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16g), TENSOR_LIST(c16g), 0);
	memset(c16->data.f16, 0, sizeof(ccv_float16_t) * N * C * 113 * 114);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c16g), TENSOR_LIST(c16), 0);
	ccv_nnc_tensor_t* const cgc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, N, C, 113, 114), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, cgc->data.f32, N * C * 113 * 114);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, c->data.f32, cgc->data.f32, N * C * 113 * 114, 1e-3, "GPU and CPU computed result should match");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(cgc);
	ccv_nnc_tensor_free(a16);
	ccv_nnc_tensor_free(b16);
	ccv_nnc_tensor_free(c16);
	ccv_nnc_tensor_free(a16g);
	ccv_nnc_tensor_free(b16g);
	ccv_nnc_tensor_free(c16g);
}

#include "case_main.h"
