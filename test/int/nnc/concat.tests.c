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

TEST_CASE("concatenate several tensors together")
{
	GUARD_ELSE_RETURN((ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)) ||
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_cnnp_model_t* const concat = ccv_cnnp_concat(0, "concat");
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_t* const full = ccv_cnnp_sequential_new(MODEL_LIST(concat, dense), 1, "full");
	ccv_nnc_tensor_param_t a_params = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_nnc_tensor_param_t b_params = GPU_TENSOR_NCHW(000, 32F, 2);
	ccv_cnnp_model_compile(full, TENSOR_PARAM_LIST(a_params, b_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(full, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ha->data.f32[0] = -0.5;
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2), 0);
	hb->data.f32[0] = 0.3;
	hb->data.f32[1] = 2;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_cnnp_model_evaluate(full, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	ccv_cnnp_model_parameters_map(full, ccv_cnnp_model_parameters(full, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_evaluate(full, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hc->data.f32[0], -0.5 + 0.3 + 2, 1e-5, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_cnnp_model_free(full);
}

TEST_CASE("concatenate several tensors together and make sure they are simplified away")
{
	GUARD_ELSE_RETURN((ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS) &&
		ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)) ||
		(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_FORMAT_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS)));
	ccv_cnnp_model_t* const x_dense = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_t* const y_dense = ccv_cnnp_dense(2, 1, 0, 1, "linear");
	ccv_cnnp_model_t* const concat = ccv_cnnp_concat(0, "concat");
	ccv_cnnp_model_t* const dense = ccv_cnnp_dense(1, 1, 0, 1, "linear");
	ccv_cnnp_model_io_t const x = ccv_cnnp_input();
	ccv_cnnp_model_io_t const y = ccv_cnnp_input();
	ccv_cnnp_model_io_t xz = ccv_cnnp_model_apply(x_dense, MODEL_IO_LIST(x));
	ccv_cnnp_model_io_t yz = ccv_cnnp_model_apply(y_dense, MODEL_IO_LIST(y));
	ccv_cnnp_model_io_t z = ccv_cnnp_model_apply(concat, MODEL_IO_LIST(xz, yz));
	z = ccv_cnnp_model_apply(dense, MODEL_IO_LIST(z));
	ccv_cnnp_model_t* const full = ccv_cnnp_model_new(MODEL_IO_LIST(x, y), MODEL_IO_LIST(z), 1, "full");
	ccv_nnc_tensor_param_t a_params = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_nnc_tensor_param_t b_params = GPU_TENSOR_NCHW(000, 32F, 2);
	ccv_cnnp_model_compile(full, TENSOR_PARAM_LIST(a_params, b_params), CMD_NOOP(), CMD_NOOP());
	CNNP_MODEL_GEN(full, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ha->data.f32[0] = -0.5;
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2), 0);
	hb->data.f32[0] = 0.3;
	hb->data.f32[1] = 2;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 1), 0);
	ccv_nnc_tensor_t* const hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_cnnp_model_evaluate(full, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	ccv_cnnp_model_parameters_map(full, ccv_cnnp_model_parameters(x_dense, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(0.5), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_parameters_map(full, ccv_cnnp_model_parameters(y_dense, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(-0.5), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_parameters_map(full, ccv_cnnp_model_parameters(dense, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	ccv_cnnp_model_evaluate(full, (ccv_cnnp_evaluate_param_t){}, TENSOR_LIST(a, b), TENSOR_LIST(c), 0, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hc->data.f32[0], -0.5 * 0.5 + (0.3 + 2) * -0.5 * 2, 1e-5, "should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_cnnp_model_free(full);
}

#include "case_main.h"
