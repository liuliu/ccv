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

TEST_CASE("run dynamic graph on multiple streams")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, 0);
	ccv_nnc_tensor_t* const hy1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y1, 0)), TENSOR_LIST(hy1), 0);
	ccv_nnc_tensor_t* const hy0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y0, 0)), TENSOR_LIST(hy0), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hy0->data.f32[0], 2.2, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hy1->data.f32[0], -1.1, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(hy1);
	ccv_nnc_tensor_free(hy0);
}

TEST_CASE("async run dynamic graph on multiple streams, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, stream);
	ccv_nnc_tensor_t* const hy0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y0, stream)), TENSOR_LIST(hy0), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_t* const hy1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y1, 0)), TENSOR_LIST(hy1), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hy0->data.f32[0], 2.2, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hy1->data.f32[0], -1.1, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(hy1);
	ccv_nnc_tensor_free(hy0);
}

TEST_CASE("async run dynamic graph on multiple streams, variant 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_t* const hy1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y1, 0)), TENSOR_LIST(hy1), 0);
	ccv_nnc_tensor_t* const hy0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, y0, 0)), TENSOR_LIST(hy0), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hy0->data.f32[0], 2.2, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hy1->data.f32[0], -1.1, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(hy1);
	ccv_nnc_tensor_free(hy0);
}

TEST_CASE("run dynamic graph backward & apply gradients on multiple devices")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, 0);
	ccv_nnc_tensor_variable_t const dx0 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t const dx1 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(dx0, dx1), 0);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(dx0, dx1), 0);
	ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_ADD_FORWARD(1, 1), TENSOR_VARIABLE_LIST(dx0, dx1), TENSOR_VARIABLE_LIST(x0, x1), 0, 2, 0);
	ccv_nnc_tensor_t* const hx0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x0, 0)), TENSOR_LIST(hx0), 0);
	ccv_nnc_tensor_t* const hx1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x1, 0)), TENSOR_LIST(hx1), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hx0->data.f32[0], 2 + 1.1 * 4, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hx1->data.f32[0], -1 + 1.1 * 4, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(hx0);
	ccv_nnc_tensor_free(hx1);
}

TEST_CASE("async run dynamic graph backward & apply gradients on multiple devices")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, stream);
	ccv_nnc_tensor_variable_t const dx0 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t const dx1 = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(dx0, dx1), stream);
	ccv_nnc_dynamic_graph_backward(graph, TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(dx0, dx1), stream);
	ccv_nnc_dynamic_graph_apply_gradients(graph, CMD_ADD_FORWARD(1, 1), TENSOR_VARIABLE_LIST(dx0, dx1), TENSOR_VARIABLE_LIST(x0, x1), 0, 2, stream);
	ccv_nnc_tensor_t* const hx0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x0, 0)), TENSOR_LIST(hx0), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_t* const hx1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x1, 0)), TENSOR_LIST(hx1), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hx0->data.f32[0], 2 + 1.1 * 4, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hx1->data.f32[0], -1 + 1.1 * 4, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(hx0);
	ccv_nnc_tensor_free(hx1);
}

TEST_CASE("run dynamic graph minimize on multiple devices")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, 0);
	ccv_nnc_dynamic_graph_minimize(graph, CMD_ADD_FORWARD(1, 1), TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), 0, 2, 0);
	ccv_nnc_tensor_t* const hx0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x0, 0)), TENSOR_LIST(hx0), 0);
	ccv_nnc_tensor_t* const hx1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x1, 0)), TENSOR_LIST(hx1), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hx0->data.f32[0], 2 + 1.1 * 2, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hx1->data.f32[0], -1 + 1.1 * 2, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_tensor_free(hx0);
	ccv_nnc_tensor_free(hx1);
}

TEST_CASE("async run dynamic graph minimize on multiple devices")
{
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2 &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t const x0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const x1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(2), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x0), 0, 0);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SET_FORWARD(-1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(), TENSOR_VARIABLE_LIST(x1), 0, 0);
	ccv_nnc_tensor_variable_t const y0 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 1));
	ccv_nnc_tensor_variable_t const y1 = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(001, 32F, 1));
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SCALAR_MUL_FORWARD(1.1), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(x0, x1), TENSOR_VARIABLE_LIST(y0, y1), 2, stream);
	ccv_nnc_dynamic_graph_minimize(graph, CMD_ADD_FORWARD(1, 1), TENSOR_VARIABLE_LIST(y0, y1), 0, TENSOR_VARIABLE_LIST(x0, x1), 0, 2, stream);
	ccv_nnc_tensor_t* const hx0 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x0, 0)), TENSOR_LIST(hx0), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_tensor_t* const hx1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_variable(graph, x1, 0)), TENSOR_LIST(hx1), 0);
	REQUIRE_EQ_WITH_TOLERANCE(hx0->data.f32[0], 2 + 1.1 * 2, 1e-5, "should be equal");
	REQUIRE_EQ_WITH_TOLERANCE(hx1->data.f32[0], -1 + 1.1 * 2, 1e-5, "should be equal");
	ccv_nnc_dynamic_graph_free(graph);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(hx0);
	ccv_nnc_tensor_free(hx1);
}

TEST_CASE("dynamic graph memory reuse logic")
{
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_nnc_tensor_variable_t a = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 128));
	ccv_nnc_tensor_t* a_tensor = ccv_nnc_tensor_from_variable(graph, a);
	const intptr_t a_ptr = (intptr_t)a_tensor->data.u8;
	ccv_nnc_tensor_variable_free(graph, a);
	ccv_nnc_tensor_variable_t b = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 64));
	ccv_nnc_tensor_t* b_tensor = ccv_nnc_tensor_from_variable(graph, b);
	const intptr_t b_ptr = (intptr_t)b_tensor->data.u8;
	REQUIRE(a_ptr == b_ptr, "allocate to the same region, even though it is smaller");
	ccv_nnc_tensor_variable_t c = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 128));
	ccv_nnc_tensor_t* c_tensor = ccv_nnc_tensor_from_variable(graph, c);
	const intptr_t c_ptr = (intptr_t)c_tensor->data.u8;
	ccv_nnc_tensor_variable_free(graph, b);
	ccv_nnc_tensor_variable_free(graph, c);
	ccv_nnc_tensor_variable_t d = ccv_nnc_tensor_variable_new(graph, GPU_TENSOR_NHWC(000, 32F, 128));
	ccv_nnc_tensor_t* d_tensor = ccv_nnc_tensor_from_variable(graph, d);
	const intptr_t d_ptr = (intptr_t)d_tensor->data.u8;
	REQUIRE(c_ptr == d_ptr, "c freed last, it is the first to be reused");
	ccv_nnc_dynamic_graph_free(graph);
}

#include "case_main.h"
