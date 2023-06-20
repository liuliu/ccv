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

TEST_CASE("datatype conversion model can reference to the last parameter for the type")
{
	GUARD_ELSE_RETURN((ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF)) ||
		(ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS)));
	const ccv_nnc_tensor_param_t a1_params = GPU_TENSOR_NHWC(000, 32F, 2);
	ccv_nnc_tensor_t* const ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ha1->data.f32[0] = 0.24;
	ha1->data.f32[1] = -1.4;
	const ccv_nnc_tensor_param_t a2_params = GPU_TENSOR_NHWC(000, 32F, 2);
	ccv_nnc_tensor_t* const ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	ha2->data.f32[0] = -3.23;
	ha2->data.f32[1] = 2.44;
	const ccv_nnc_tensor_param_t a3_params = GPU_TENSOR_NHWC(000, 16F, 2);
	ccv_nnc_tensor_t* const ha3 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2), 0);
	ccv_float_to_half_precision(ha2->data.f32, (uint16_t*)ha3->data.f16, 2);
	ccv_nnc_tensor_t* const a1 = ccv_nnc_tensor_new(0, a1_params, 0);
	ccv_nnc_tensor_t* const a2 = ccv_nnc_tensor_new(0, a2_params, 0);
	ccv_nnc_tensor_t* const a3 = ccv_nnc_tensor_new(0, a3_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, ha2, ha3), TENSOR_LIST(a1, a2, a3), 0);
	const ccv_cnnp_model_io_t input1 = ccv_cnnp_input();
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_datatype_conversion(CCV_16F, 0, 0), MODEL_IO_LIST(input1));
	const ccv_cnnp_model_io_t input3 = ccv_cnnp_input();
	output = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output, input3));
	const ccv_cnnp_model_io_t input2 = ccv_cnnp_input();
	output = ccv_cnnp_model_apply(ccv_cnnp_datatype_conversion(0, 1, 0), MODEL_IO_LIST(output, input2));
	output = ccv_cnnp_model_apply(ccv_cnnp_sum(0), MODEL_IO_LIST(output, input2));
	ccv_cnnp_model_t* const final = ccv_cnnp_model_new(MODEL_IO_LIST(input1, input2, input3), MODEL_IO_LIST(output), 1, 0);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a1_params, a2_params, a3_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, a1_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2), 0);
	hb->data.f32[0] = 1;
	hb->data.f32[1] = 2;
	ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a1, a2, a3), TENSOR_LIST(b), 0, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hb->data.f32[0], 0.24 + (-3.23) * 2, 1e-2, "should match");
	REQUIRE_EQ_WITH_TOLERANCE(hb->data.f32[1], -1.4 + 2.44 * 2, 1e-2, "should match");
	ccv_nnc_tensor_free(a1);
	ccv_nnc_tensor_free(a2);
	ccv_nnc_tensor_free(a3);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(ha3);
	ccv_nnc_tensor_free(hb);
	ccv_cnnp_model_free(final);
}

#include "case_main.h"
