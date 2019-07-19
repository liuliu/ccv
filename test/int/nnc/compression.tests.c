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
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMPRESSION_LSSC_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_COMPRESSION_LSSC_BACKWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 64, 113, 114), 0);
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NCHW(16F, 64, 113, 114);
	ccv_nnc_tensor_t* const a16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_param_t b_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &a_params, 1, ccv_nnc_no_hint, &b_params, 1);
	ccv_nnc_tensor_t* const b16 = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const c16 = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 64, 113, 114), 0);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 64 * 113 * 114; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_float_to_half_precision(a->data.f32, (uint16_t*)a16->data.f16, 64 * 113 * 114);
	ccv_half_precision_to_float((uint16_t*)a16->data.f16, a->data.f32, 64 * 113 * 114);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(b16), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16), TENSOR_LIST(c16), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, c->data.f32, 64 * 113 * 114);
	// Compare against GPU computation
	ccv_nnc_tensor_param_t ag_params = GPU_TENSOR_NCHW(000, 16F, 64, 113, 114);
	ccv_nnc_tensor_t* const a16g = ccv_nnc_tensor_new(0, ag_params, 0);
	ccv_nnc_tensor_param_t bg_params;
	ccv_nnc_hint_tensor_auto(CMD_COMPRESSION_LSSC_FORWARD(), &ag_params, 1, ccv_nnc_no_hint, &bg_params, 1);
	ccv_nnc_tensor_t* const b16g = ccv_nnc_tensor_new(0, bg_params, 0);
	ccv_nnc_tensor_t* const c16g = ccv_nnc_tensor_new(0, ag_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16), TENSOR_LIST(a16g), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a16g), TENSOR_LIST(b16g), 0);
	ccv_nnc_cmd_exec(CMD_COMPRESSION_LSSC_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b16g), TENSOR_LIST(c16g), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c16g), TENSOR_LIST(c16), 0);
	ccv_nnc_tensor_t* const cgc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 64, 113, 114), 0);
	ccv_half_precision_to_float((uint16_t*)c16->data.f16, cgc->data.f32, 64 * 113 * 114);
	REQUIRE_ARRAY_EQ(float, c->data.f32, cgc->data.f32, 1e-5, "GPU and CPU computed result should match");
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
