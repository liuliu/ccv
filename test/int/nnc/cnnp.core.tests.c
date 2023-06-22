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

ccv_cnnp_model_t* _math_2_x_10()
{
	ccv_cnnp_model_t* mul = ccv_cnnp_dense(1, 1, 1, "mul");
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t left_out = ccv_cnnp_model_apply(mul, MODEL_IO_LIST(input));
	ccv_cnnp_model_io_t fit = ccv_cnnp_input();
	// Because we don't have L2 loss function available yet, manually create L2 loss.
	ccv_cnnp_model_io_t diff = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_ADD_FORWARD(1, -1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(left_out, fit));
	ccv_cnnp_model_io_t sqr = ccv_cnnp_model_apply(
		ccv_cnnp_cmd_exec(CMD_MUL_FORWARD(1), ccv_nnc_no_hint, 0,
			MODEL_CMD_EXEC_IO_MAP(KV(CCV_CNNP_IO), KV(CCV_CNNP_IO)),
			MODEL_CMD_EXEC_IO_LIST(CCV_CNNP_IO), 1, 0),
		MODEL_IO_LIST(diff, diff));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input, fit), MODEL_IO_LIST(sqr), 1, 0);
}

TEST_CASE("train a simple math 2 * x = 10, x = 5 and copy parameter to a new model entirely")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count >= 1);
	ccv_cnnp_model_t* const final = _math_2_x_10();
	const ccv_nnc_tensor_param_t a = GPU_TENSOR_NCHW(000, 32F, 1);
	const ccv_nnc_tensor_param_t f = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final, device_count);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor[device_count];
	ccv_nnc_tensor_t* f_tensor[device_count];
	ccv_nnc_tensor_t* o_tensor[device_count];
	ccv_nnc_tensor_t* ingrad[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t ai = a;
		CCV_TENSOR_SET_DEVICE_ID(ai.type, i);
		a_tensor[i] = ccv_nnc_tensor_new(0, ai, 0);
		ccv_nnc_tensor_param_t fi = f;
		CCV_TENSOR_SET_DEVICE_ID(fi.type, i);
		f_tensor[i] = ccv_nnc_tensor_new(0, fi, 0);
		ccv_nnc_tensor_param_t oi = o;
		CCV_TENSOR_SET_DEVICE_ID(oi.type, i);
		o_tensor[i] = ccv_nnc_tensor_new(0, oi, 0);
		ingrad[i] = ccv_nnc_tensor_new(0, oi, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(a_tensor[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(f_tensor[i]), 0);
	}
	ccv_nnc_tensor_t* inputs[device_count * 2];
	for (i = 0; i < 10; i++)
	{
		int j;
		for (j = 0; j < device_count; j++)
		{
			inputs[j * 2] = a_tensor[j];
			inputs[j * 2 + 1] = f_tensor[j];
		}
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	ccv_nnc_tensor_t* ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[0]), TENSOR_LIST(ho), 0);
	const float o_final = ho->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_10();
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final2, device_count);
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], 100, 1e-5, "should match the output when x is 0");
	}
	ccv_cnnp_model_t* const final3 = ccv_cnnp_model_copy(final, 1);
	ccv_cnnp_model_set_data_parallel(final3, device_count);
	ccv_cnnp_model_set_parameters(final3, ccv_cnnp_model_parameters(final3, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final3, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(a_tensor[i]);
		ccv_nnc_tensor_free(f_tensor[i]);
		ccv_nnc_tensor_free(o_tensor[i]);
		ccv_nnc_tensor_free(ingrad[i]);
	}
	ccv_nnc_tensor_free(ho);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
	ccv_cnnp_model_free(final3);
}

TEST_CASE("train a simple math 2 * x = 10, x = 5 and copy parameter to a new model entirely with a stream context")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	GUARD_ELSE_RETURN(device_count >= 1);
	ccv_cnnp_model_t* const final = _math_2_x_10();
	const ccv_nnc_tensor_param_t a = GPU_TENSOR_NCHW(000, 32F, 1);
	const ccv_nnc_tensor_param_t f = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final, device_count);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor[device_count];
	ccv_nnc_tensor_t* f_tensor[device_count];
	ccv_nnc_tensor_t* o_tensor[device_count];
	ccv_nnc_tensor_t* ingrad[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t ai = a;
		CCV_TENSOR_SET_DEVICE_ID(ai.type, i);
		a_tensor[i] = ccv_nnc_tensor_new(0, ai, 0);
		ccv_nnc_tensor_param_t fi = f;
		CCV_TENSOR_SET_DEVICE_ID(fi.type, i);
		f_tensor[i] = ccv_nnc_tensor_new(0, fi, 0);
		ccv_nnc_tensor_param_t oi = o;
		CCV_TENSOR_SET_DEVICE_ID(oi.type, i);
		o_tensor[i] = ccv_nnc_tensor_new(0, oi, 0);
		ingrad[i] = ccv_nnc_tensor_new(0, oi, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(a_tensor[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(f_tensor[i]), 0);
	}
	ccv_nnc_tensor_t* inputs[device_count * 2];
	for (i = 0; i < 10; i++)
	{
		int j;
		for (j = 0; j < device_count; j++)
		{
			inputs[j * 2] = a_tensor[j];
			inputs[j * 2 + 1] = f_tensor[j];
		}
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	ccv_nnc_tensor_t* ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[0]), TENSOR_LIST(ho), 0);
	const float o_final = ho->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_10();
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final2, device_count);
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, 0, 0, 0, 0, stream_context);
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], 100, 1e-5, "should match the output when x is 0");
	}
	ccv_cnnp_model_t* const final3 = ccv_cnnp_model_copy(final, 1);
	ccv_cnnp_model_set_data_parallel(final3, device_count);
	ccv_cnnp_model_set_parameters(final3, ccv_cnnp_model_parameters(final3, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final3, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(a_tensor[i]);
		ccv_nnc_tensor_free(f_tensor[i]);
		ccv_nnc_tensor_free(o_tensor[i]);
		ccv_nnc_tensor_free(ingrad[i]);
	}
	ccv_nnc_tensor_free(ho);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
	ccv_nnc_stream_context_free(stream_context);
	ccv_cnnp_model_free(final3);
}

TEST_CASE("train a simple math 2 * x = 10, x = 5 and merge parameters with a model")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_model_t* const final = _math_2_x_10();
	const ccv_nnc_tensor_param_t a = GPU_TENSOR_NCHW(000, 32F, 1);
	const ccv_nnc_tensor_param_t f = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final, device_count);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor[device_count];
	ccv_nnc_tensor_t* f_tensor[device_count];
	ccv_nnc_tensor_t* o_tensor[device_count];
	ccv_nnc_tensor_t* ingrad[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t ai = a;
		CCV_TENSOR_SET_DEVICE_ID(ai.type, i);
		a_tensor[i] = ccv_nnc_tensor_new(0, ai, 0);
		ccv_nnc_tensor_param_t fi = f;
		CCV_TENSOR_SET_DEVICE_ID(fi.type, i);
		f_tensor[i] = ccv_nnc_tensor_new(0, fi, 0);
		ccv_nnc_tensor_param_t oi = o;
		CCV_TENSOR_SET_DEVICE_ID(oi.type, i);
		o_tensor[i] = ccv_nnc_tensor_new(0, oi, 0);
		ingrad[i] = ccv_nnc_tensor_new(0, oi, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(a_tensor[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(f_tensor[i]), 0);
	}
	ccv_nnc_tensor_t* inputs[device_count * 2];
	for (i = 0; i < 10; i++)
	{
		int j;
		for (j = 0; j < device_count; j++)
		{
			inputs[j * 2] = a_tensor[j];
			inputs[j * 2 + 1] = f_tensor[j];
		}
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	ccv_nnc_tensor_t* ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[0]), TENSOR_LIST(ho), 0);
	const float o_final = ho->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_10();
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final2, device_count);
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], 64, 1e-5, "should match the output when x is 1");
	}
	ccv_cnnp_model_parameters_zip_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_ADD_FORWARD(0.6, 0.4), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0, final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_cnnp_model_parameter_copy(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), x_tensor);
	const float x_final = x_tensor->data.f32[0] * 0.4 + 1 * 0.6;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], (x_final * 2 - 10) * (x_final * 2 - 10), 1e-5, "should match the previous output");
	}
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(a_tensor[i]);
		ccv_nnc_tensor_free(f_tensor[i]);
		ccv_nnc_tensor_free(o_tensor[i]);
		ccv_nnc_tensor_free(ingrad[i]);
	}
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(x_tensor);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
}

TEST_CASE("train a simple math 2 * x = 10, x = 5 and merge parameters with a model with a stream context")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_EWSUM_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN) &&
		ccv_nnc_cmd_ok(CCV_NNC_SGD_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_model_t* const final = _math_2_x_10();
	const ccv_nnc_tensor_param_t a = GPU_TENSOR_NCHW(000, 32F, 1);
	const ccv_nnc_tensor_param_t f = GPU_TENSOR_NCHW(000, 32F, 1);
	ccv_cnnp_model_compile(final, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final, device_count);
	CNNP_MODEL_GEN(final, CCV_NNC_LONG_DOT_GRAPH);
	ccv_nnc_tensor_param_t o = {};
	ccv_cnnp_model_tensor_auto(final, &o, 1);
	ccv_nnc_tensor_t* a_tensor[device_count];
	ccv_nnc_tensor_t* f_tensor[device_count];
	ccv_nnc_tensor_t* o_tensor[device_count];
	ccv_nnc_tensor_t* ingrad[device_count];
	int i;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t ai = a;
		CCV_TENSOR_SET_DEVICE_ID(ai.type, i);
		a_tensor[i] = ccv_nnc_tensor_new(0, ai, 0);
		ccv_nnc_tensor_param_t fi = f;
		CCV_TENSOR_SET_DEVICE_ID(fi.type, i);
		f_tensor[i] = ccv_nnc_tensor_new(0, fi, 0);
		ccv_nnc_tensor_param_t oi = o;
		CCV_TENSOR_SET_DEVICE_ID(oi.type, i);
		o_tensor[i] = ccv_nnc_tensor_new(0, oi, 0);
		ingrad[i] = ccv_nnc_tensor_new(0, oi, 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(ingrad[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(a_tensor[i]), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, 0, 0, TENSOR_LIST(f_tensor[i]), 0);
	}
	ccv_nnc_tensor_t* inputs[device_count * 2];
	for (i = 0; i < 10; i++)
	{
		int j;
		for (j = 0; j < device_count; j++)
		{
			inputs[j * 2] = a_tensor[j];
			inputs[j * 2 + 1] = f_tensor[j];
		}
		ccv_cnnp_model_evaluate(final, (ccv_cnnp_evaluate_param_t){
			.requires_grad = 1,
		}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
		ccv_cnnp_model_backward(final, TENSOR_LIST(), TENSOR_LIST(), 0, 0);
		ccv_cnnp_model_apply_gradients(final, 0);
	}
	ccv_nnc_tensor_t* ho = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[0]), TENSOR_LIST(ho), 0);
	const float o_final = ho->data.f32[0];
	ccv_cnnp_model_t* const final2 = _math_2_x_10();
	ccv_cnnp_model_compile(final2, TENSOR_PARAM_LIST(a, f), CMD_SGD_FORWARD(0, 0.1, 1.0 / device_count, 0.1, 0, 0), CMD_NOOP());
	ccv_cnnp_model_set_data_parallel(final2, device_count);
	ccv_cnnp_model_set_parameters(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], o_final, 1e-5, "should match the previous output");
	}
	ccv_cnnp_model_parameters_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_SET_FORWARD(1), ccv_nnc_no_hint, 0, 0, 0, 0, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, 0);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], 64, 1e-5, "should match the output when x is 1");
	}
	ccv_nnc_stream_context_t* stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_model_parameters_zip_map(final2, ccv_cnnp_model_parameters(final2, ALL_PARAMETERS, ALL_PARAMETERS), CMD_ADD_FORWARD(0.6, 0.4), ccv_nnc_no_hint, 0, 0, 0, 0, 0, stream_context, final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS));
	for (i = 0; i < device_count; i++)
	{
		inputs[i * 2] = a_tensor[i];
		inputs[i * 2 + 1] = f_tensor[i];
	}
	ccv_cnnp_model_evaluate(final2, (ccv_cnnp_evaluate_param_t){}, inputs, device_count * 2, o_tensor, device_count, 0, stream_context);
	ccv_nnc_stream_context_wait(stream_context);
	ccv_nnc_tensor_t* x_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 1), 0);
	ccv_cnnp_model_parameter_copy(final, ccv_cnnp_model_parameters(final, ALL_PARAMETERS, ALL_PARAMETERS), x_tensor);
	const float x_final = x_tensor->data.f32[0] * 0.4 + 1 * 0.6;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(o_tensor[i]), TENSOR_LIST(ho), 0);
		REQUIRE_EQ_WITH_TOLERANCE(ho->data.f32[0], (x_final * 2 - 10) * (x_final * 2 - 10), 1e-5, "should match the previous output");
	}
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(a_tensor[i]);
		ccv_nnc_tensor_free(f_tensor[i]);
		ccv_nnc_tensor_free(o_tensor[i]);
		ccv_nnc_tensor_free(ingrad[i]);
	}
	ccv_nnc_tensor_free(ho);
	ccv_nnc_tensor_free(x_tensor);
	ccv_cnnp_model_free(final);
	ccv_cnnp_model_free(final2);
	ccv_nnc_stream_context_free(stream_context);
}

#include "case_main.h"
