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

TEST_CASE("schedule symbolic graph to data parallel with broadcast and reduce")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2);
	ccv_nnc_tensor_t* updated[4];
	ccv_nnc_tensor_t* cpu_inputs[2];
	ccv_nnc_tensor_t* cpu_fits[2];
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t y4a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y4, ccv_nnc_no_ofs, DIM_ALLOC(16, 8), GPU_TENSOR_NCHW(000, 32F, 16, 8), 0);
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4a, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 2, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, gradients, 4, CCV_NNC_PARALLEL_REDUCE_OP_SUM, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_graph_static_schedule(graph, CCV_STREAM_CONTEXT_GPU);
		GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
		cpu_inputs[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		cpu_inputs[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 0);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[0]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[1]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		cpu_fits[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		cpu_fits[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		for (i = 0; i < 16; i++)
			cpu_fits[0]->data.f32[i] = cpu_fits[1]->data.f32[i] = (int)(dsfmt_genrand_open_close(&dsfmt) * 7.4); // Between 0 to 7.
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[0], cpu_inputs[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, x, 1))), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fits[0], cpu_fits[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, label, 1))), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		for (i = 0; i < 8 * 3 * 5 * 5; i++)
			w1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 8 * 8 * 5 * 5; i++)
			w3_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, ccv_nnc_graph_default_stream(graph), 0, TRAVERSE_FULL);
		ccv_nnc_stream_context_wait(ccv_nnc_graph_default_stream(graph));
		updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	}
	// Now, doing exactly the same, but with no parallel.
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_tensor_t* cpu_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 3, 32, 32), 0);
		memcpy(cpu_input->data.f32, cpu_inputs[0]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		memcpy(cpu_input->data.f32 + 16 * 3 * 32 * 32, cpu_inputs[1]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		ccv_nnc_tensor_t* cpu_fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32), 0);
		memcpy(cpu_fit->data.f32, cpu_fits[0]->data.f32, sizeof(float) * 16);
		memcpy(cpu_fit->data.f32 + 16, cpu_fits[1]->data.f32, sizeof(float) * 16);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_input), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fit), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label)), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, 0, 0, TRAVERSE_FULL);
		ccv_nnc_tensor_t* np_updated[4];
		np_updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		np_updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		np_updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		np_updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), np_updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
		REQUIRE_TENSOR_EQ(np_updated[0], updated[0], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[1], updated[1], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[2], updated[2], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[3], updated[3], "updated params should be equal");
		ccv_nnc_tensor_free(cpu_input);
		ccv_nnc_tensor_free(cpu_fit);
		ccv_nnc_tensor_free(np_updated[0]);
		ccv_nnc_tensor_free(np_updated[1]);
		ccv_nnc_tensor_free(np_updated[2]);
		ccv_nnc_tensor_free(np_updated[3]);
	}
	ccv_nnc_tensor_free(updated[0]);
	ccv_nnc_tensor_free(updated[1]);
	ccv_nnc_tensor_free(updated[2]);
	ccv_nnc_tensor_free(updated[3]);
	ccv_nnc_tensor_free(cpu_inputs[0]);
	ccv_nnc_tensor_free(cpu_inputs[1]);
	ccv_nnc_tensor_free(cpu_fits[0]);
	ccv_nnc_tensor_free(cpu_fits[1]);
	ccv_nnc_tensor_free(w1_tensor);
	ccv_nnc_tensor_free(w3_tensor);
}

TEST_CASE("schedule symbolic graph to data parallel with allreduce")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN));
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_COMM_ALLREDUCE_FORWARD, CCV_NNC_BACKEND_GPU_NCCL));
	GUARD_ELSE_RETURN(ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) >= 2);
	ccv_nnc_tensor_t* updated[4];
	ccv_nnc_tensor_t* cpu_inputs[2];
	ccv_nnc_tensor_t* cpu_fits[2];
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t y4a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, y4, ccv_nnc_no_ofs, DIM_ALLOC(16, 8), GPU_TENSOR_NCHW(000, 32F, 16, 8), 0);
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 16), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4a, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_symbolic_graph_data_parallel(symbolic_graph, 2, TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), gradients, 4, 0, 0, CCV_NNC_PARALLEL_REDUCE_OP_SUM, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), updated_execs, 4);
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		SYMBOLIC_GRAPH_GEN(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_graph_static_schedule(graph, CCV_STREAM_CONTEXT_GPU);
		GRAPH_GEN(graph, CCV_NNC_LONG_DOT_GRAPH);
		cpu_inputs[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		cpu_inputs[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16, 3, 32, 32), 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 0);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[0]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 16 * 3 * 32 * 32; i++)
			cpu_inputs[1]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		cpu_fits[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		cpu_fits[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 16), 0);
		for (i = 0; i < 16; i++)
			cpu_fits[0]->data.f32[i] = cpu_fits[1]->data.f32[i] = (int)(dsfmt_genrand_open_close(&dsfmt) * 7.4); // Between 0 to 7.
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_inputs[0], cpu_inputs[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, x, 1))), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fits[0], cpu_fits[1]), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label), ccv_nnc_tensor_from_symbol(tensor_arena, ccv_nnc_tensor_symbol_copy(symbolic_graph, label, 1))), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		for (i = 0; i < 8 * 3 * 5 * 5; i++)
			w1_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		for (i = 0; i < 8 * 8 * 5 * 5; i++)
			w3_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, ccv_nnc_graph_default_stream(graph), 0, TRAVERSE_FULL);
		ccv_nnc_stream_context_wait(ccv_nnc_graph_default_stream(graph));
		updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
	}
	// Now, doing exactly the same, but with no parallel.
	{
		ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
		const ccv_nnc_tensor_symbol_t x = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 3, 32, 32), 0);
		const ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 3, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 32, 32), 0);
		const ccv_nnc_graph_exec_symbol_t conv1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(x, w1, bias1), TENSOR_SYMBOL_LIST(y1), "conv1");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv1, HINT((1, 1), (2, 2)));
		const ccv_nnc_tensor_symbol_t y2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 16, 16), 0);
		const ccv_nnc_graph_exec_symbol_t avg2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(2, 2), TENSOR_SYMBOL_LIST(y1), TENSOR_SYMBOL_LIST(y2), "avg2");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, avg2, HINT((2, 2)));
		const ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8, 8, 5, 5), 0);
		const ccv_nnc_tensor_symbol_t bias3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 8), 0);
		const ccv_nnc_tensor_symbol_t y3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 8, 8), 0);
		const ccv_nnc_graph_exec_symbol_t conv3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(1, 8, 5, 5), TENSOR_SYMBOL_LIST(y2, w3, bias3), TENSOR_SYMBOL_LIST(y3), "conv3");
		ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, conv3, HINT((2, 2), (2, 2)));
		const ccv_nnc_tensor_symbol_t y4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8, 1, 1), 0);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(8, 8), TENSOR_SYMBOL_LIST(y3), TENSOR_SYMBOL_LIST(y4), "avg4");
		const ccv_nnc_tensor_symbol_t label = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "label");
		const ccv_nnc_tensor_symbol_t y5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32, 8), "y5");
		const ccv_nnc_tensor_symbol_t loss = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32F, 32), "loss");
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_CROSSENTROPY_FORWARD(), TENSOR_SYMBOL_LIST(y4, label), TENSOR_SYMBOL_LIST(loss, y5), "softmax crossentropy");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_tensor_symbol_t updated_params[4];
		ccv_nnc_tensor_symbol_t gradients[4];
		const int saved_aux_size = ccv_nnc_minimizer_saved_aux_size(CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9));
		ccv_nnc_tensor_symbol_map_t saved_aux[saved_aux_size * 4];
		ccv_nnc_graph_exec_symbol_t updated_execs[4];
		ccv_nnc_symbolic_graph_minimize(symbolic_graph, CMD_SGD_FORWARD(0.001, 0.99, 0.9, 0.9), TENSOR_SYMBOL_LIST(loss), TENSOR_SYMBOL_LIST(w1, bias1, w3, bias3), 0, 0, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), gradients, updated_params, saved_aux, updated_execs);
		const ccv_nnc_tensor_symbol_t dloss = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, loss);
		ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(1), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(dloss), "set 1");
		int i;
		for (i = 0; i < saved_aux_size * 4; i++)
			ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SET_FORWARD(0), TENSOR_SYMBOL_LIST(), TENSOR_SYMBOL_LIST(saved_aux[i].source), "set 0");
		ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_symbolic_graph_compile(symbolic_graph,
			0, 0,
			updated_params, 4,
			SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
			&graph, &tensor_arena, &graph_exec_arena);
		ccv_nnc_tensor_t* cpu_input = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32, 3, 32, 32), 0);
		memcpy(cpu_input->data.f32, cpu_inputs[0]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		memcpy(cpu_input->data.f32 + 16 * 3 * 32 * 32, cpu_inputs[1]->data.f32, sizeof(float) * 16 * 3 * 32 * 32);
		ccv_nnc_tensor_t* cpu_fit = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 32), 0);
		memcpy(cpu_fit->data.f32, cpu_fits[0]->data.f32, sizeof(float) * 16);
		memcpy(cpu_fit->data.f32 + 16, cpu_fits[1]->data.f32, sizeof(float) * 16);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_input), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, x)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(cpu_fit), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, label)), 0);
		ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, bias1), ccv_nnc_tensor_from_symbol(tensor_arena, bias3)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, w3_tensor), TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, w1), ccv_nnc_tensor_from_symbol(tensor_arena, w3)), 0);
		ccv_nnc_graph_run(graph, 0, 0, 0, TRAVERSE_FULL);
		ccv_nnc_tensor_t* np_updated[4];
		np_updated[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 3, 5, 5), 0);
		np_updated[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		np_updated[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8, 8, 5, 5), 0);
		np_updated[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[0]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[1]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[2]), ccv_nnc_tensor_from_symbol(tensor_arena, updated_params[3])), np_updated, 4, 0);
		ccv_nnc_symbolic_graph_free(symbolic_graph);
		ccv_nnc_graph_free(graph);
		ccv_nnc_tensor_arena_free(tensor_arena);
		ccv_nnc_graph_exec_arena_free(graph_exec_arena);
		REQUIRE_TENSOR_EQ(np_updated[0], updated[0], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[1], updated[1], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[2], updated[2], "updated params should be equal");
		REQUIRE_TENSOR_EQ(np_updated[3], updated[3], "updated params should be equal");
		ccv_nnc_tensor_free(cpu_input);
		ccv_nnc_tensor_free(cpu_fit);
		ccv_nnc_tensor_free(np_updated[0]);
		ccv_nnc_tensor_free(np_updated[1]);
		ccv_nnc_tensor_free(np_updated[2]);
		ccv_nnc_tensor_free(np_updated[3]);
	}
	ccv_nnc_tensor_free(updated[0]);
	ccv_nnc_tensor_free(updated[1]);
	ccv_nnc_tensor_free(updated[2]);
	ccv_nnc_tensor_free(updated[3]);
	ccv_nnc_tensor_free(cpu_inputs[0]);
	ccv_nnc_tensor_free(cpu_inputs[1]);
	ccv_nnc_tensor_free(cpu_fits[0]);
	ccv_nnc_tensor_free(cpu_fits[1]);
	ccv_nnc_tensor_free(w1_tensor);
	ccv_nnc_tensor_free(w3_tensor);
}

#include "case_main.h"
