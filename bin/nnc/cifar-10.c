#include <ctype.h>
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>

static void glorot_random_device(ccv_nnc_tensor_t* const b, float glorot)
{
	ccv_nnc_tensor_param_t params = b->info;
	params.type = CCV_TENSOR_CPU_MEMORY;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_RANDOM_UNIFORM_FORWARD(-glorot, glorot), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_free(a);
}

static void train_cifar_10(ccv_array_t* const training_set, const float mean[3], ccv_array_t* const test_set)
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t input = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NCHW(128, 3, 31, 31), "input");
	ccv_nnc_tensor_symbol_t x0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 3, 31, 31), "x0");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(input), TENSOR_SYMBOL_LIST(x0), 0);
	ccv_nnc_tensor_symbol_t w0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32, 3, 5, 5), "w0");
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32), "b0");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 31, 31), "x1");
	ccv_nnc_graph_exec_symbol_t ly1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 3), TENSOR_SYMBOL_LIST(x0, w0, b0), TENSOR_SYMBOL_LIST(x1), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly1, HINT((1, 1), (2, 2)));
	 ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 31, 31), "x2");
	 ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x1), TENSOR_SYMBOL_LIST(x2), 0);
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 15, 15), "x3");
	ccv_nnc_graph_exec_symbol_t ly2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x2), TENSOR_SYMBOL_LIST(x3), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly2, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32, 32, 5, 5), "w1");
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 32), "b1");
	ccv_nnc_tensor_symbol_t x4 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 15, 15), "x4");
	ccv_nnc_graph_exec_symbol_t ly3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 32), TENSOR_SYMBOL_LIST(x3, w1, b1), TENSOR_SYMBOL_LIST(x4), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly3, HINT((1, 1,), (2, 2)));
	ccv_nnc_tensor_symbol_t x5 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 15, 15), "x5");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x4), TENSOR_SYMBOL_LIST(x5), 0);
	ccv_nnc_tensor_symbol_t x6 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 32, 7, 7), "x6");
	ccv_nnc_graph_exec_symbol_t ly4 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x5), TENSOR_SYMBOL_LIST(x6), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly4, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 64, 32, 5, 5), "w2");
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 64), "b2");
	ccv_nnc_tensor_symbol_t x7 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 64, 7, 7), "x7");
	ccv_nnc_graph_exec_symbol_t ly5 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(64, 5, 5, 32), TENSOR_SYMBOL_LIST(x6, w2, b2), TENSOR_SYMBOL_LIST(x7), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly5, HINT((1, 1), (2, 2)));
	ccv_nnc_tensor_symbol_t x8 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 64, 7, 7), "x8");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x7), TENSOR_SYMBOL_LIST(x8), 0);
	ccv_nnc_tensor_symbol_t x9 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 64, 3, 3), "x9");
	ccv_nnc_graph_exec_symbol_t ly6 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x8), TENSOR_SYMBOL_LIST(x9), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly6, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t x9a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x9, ccv_nnc_no_ofs, DIM_ALLOC(128, 3 * 3 * 64), GPU_TENSOR_NCHW(000, 128, 3 * 3 * 64), "x9a");
	ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 10, 3 * 3 * 64), "w3");
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 10), "b3");
	ccv_nnc_tensor_symbol_t x10 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 10), "x10");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(10), TENSOR_SYMBOL_LIST(x9a, w3, b3), TENSOR_SYMBOL_LIST(x10), 0);
	ccv_nnc_tensor_symbol_t x11 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 10), "x11");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x10), TENSOR_SYMBOL_LIST(x11), 0);
	ccv_nnc_tensor_symbol_t x12 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NCHW(000, 128, 10), "x12");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(x11), TENSOR_SYMBOL_LIST(x12), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), TENSOR_SYMBOL_LIST(x12), TENSOR_SYMBOL_LIST(w0, b0, w1, b1, w2, b2, w3, b3));
	ccv_nnc_tensor_symbol_t dw0 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w0);
	ccv_nnc_tensor_symbol_t db0 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b0);
	ccv_nnc_tensor_symbol_t dw1 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w1);
	ccv_nnc_tensor_symbol_t db1 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b1);
	ccv_nnc_tensor_symbol_t dw2 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w2);
	ccv_nnc_tensor_symbol_t db2 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b2);
	ccv_nnc_tensor_symbol_t dw3 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w3);
	ccv_nnc_tensor_symbol_t db3 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b3);
	ccv_nnc_tensor_symbol_t dx12 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, x12);
	ccv_nnc_tensor_symbol_t dx12c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NCHW(128, 10), "dx12c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(dx12c), TENSOR_SYMBOL_LIST(dx12), 0);
	ccv_nnc_tensor_symbol_t x12c = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NCHW(128, 10), "x12c");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(x12), TENSOR_SYMBOL_LIST(x12c), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_t* w0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32, 3, 5, 5), 0);
	glorot_random_device(w0_tensor, sqrtf(2) / sqrtf(5 * 5 * 3 + 32));
	ccv_nnc_tensor_t* b0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b0_tensor), 0);
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32, 32, 5, 5), 0);
	glorot_random_device(w1_tensor, sqrtf(2) / sqrtf(5 * 5 * 32 + 32));
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b1_tensor), 0);
	ccv_nnc_tensor_t* w2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64, 32, 5, 5), 0);
	glorot_random_device(w2_tensor, sqrtf(2) / sqrtf(5 * 5 * 32 + 64));
	ccv_nnc_tensor_t* b2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b2_tensor), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 10, 3 * 3 * 64), 0);
	glorot_random_device(w3_tensor, sqrtf(2) / sqrtf(3 * 3 * 64 + 10));
	ccv_nnc_tensor_t* b3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 10), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b3_tensor), 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph,
		TENSOR_BIND_MAP(KV(w0, w0_tensor), KV(b0, b0_tensor), KV(w1, w1_tensor), KV(b1, b1_tensor), KV(w2, w2_tensor), KV(b2, b2_tensor), KV(w3, w3_tensor), KV(b3, b3_tensor)),
		TENSOR_SYMBOL_LIST(dw0, db0, dw1, db1, dw2, db2, dw3, db3, x12c),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	FILE* w = fopen("cifar-10.dot", "w+");
	ccv_nnc_symbolic_graph_dot(symbolic_graph, CCV_NNC_LONG_DOT_GRAPH, w);
	fclose(w);
	w = fopen("cifar-10-c.dot", "w+");
	ccv_nnc_graph_dot(graph, CCV_NNC_LONG_DOT_GRAPH, w);
	fclose(w);
	ccv_nnc_tensor_t* tw0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32, 3, 5, 5), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tw0_tensor), 0);
	ccv_nnc_tensor_t* tb0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tb0_tensor), 0);
	ccv_nnc_tensor_t* tw1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32, 32, 5, 5), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tw1_tensor), 0);
	ccv_nnc_tensor_t* tb1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tb1_tensor), 0);
	ccv_nnc_tensor_t* tw2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64, 32, 5, 5), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tw2_tensor), 0);
	ccv_nnc_tensor_t* tb2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tb2_tensor), 0);
	ccv_nnc_tensor_t* tw3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 10, 3 * 3 * 64), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tw3_tensor), 0);
	ccv_nnc_tensor_t* tb3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 10), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(tb3_tensor), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* w0c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32, 3, 5, 5), 0);
	ccv_nnc_tensor_t* w1c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32, 32, 5, 5), 0);
	ccv_nnc_tensor_t* w2c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64, 32, 5, 5), 0);
	ccv_nnc_tensor_t* w3c = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(10, 3 * 3 * 64), 0);
	int i, j, k;
	int c[128];
	double correct_ratio = 0;
	ccv_nnc_graph_autotune(graph, 1024 * 1024 * 1024, 0, TRAVERSE_FULL);
	for (i = 0; i < 100000; i++)
	{
		// Load data.
		ccv_nnc_tensor_t* input_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, input);
		ccv_nnc_tensor_t* dx12c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dx12c);
		ccv_nnc_tensor_zero(dx12c_tensor);
		for (j = 0; j < 128; j++)
		{
			k = (int)(dsfmt_genrand_close_open(&dsfmt) * training_set->rnum);
			assert(k < training_set->rnum);
			ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(training_set, k);
			float* const ip = input_tensor->data.f32 + j * 31 * 31 * 3;
			float* const cp = categorized->matrix->data.f32;
			int fi, fj, fk;
			for (fi = 0; fi < 31; fi++)
				for (fj = 0; fj < 31; fj++)
					for (fk = 0; fk < 3; fk++)
						ip[fi * 31 + fj + fk * 31 * 31] = cp[fi * 31 * 3 + fj * 3 + fk] - mean[fk];
			assert(categorized->c >= 0 && categorized->c < 10);
			dx12c_tensor->data.f32[j * 10 + categorized->c] = 1;
			c[j] = categorized->c;
		}
		ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
		ccv_nnc_tensor_t* x12c_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, x12c);
		int correct = 0;
		for (j = 0; j < 128; j++)
		{
			float max = -FLT_MAX;
			int t = -1;
			for (k = 0; k < 10; k++)
				if (x12c_tensor->data.f32[j * 10 + k] > max)
					max = x12c_tensor->data.f32[j * 10 + k], t = k;
			if (c[j] == t)
				++correct;
		}
		correct_ratio = correct_ratio * 0.9 + correct * 0.1 / 128.;
		ccv_nnc_tensor_t* dw0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dw0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tw0_tensor, dw0_tensor), TENSOR_LIST(tw0_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w0_tensor, tw0_tensor), TENSOR_LIST(w0_tensor), 0);
		ccv_nnc_tensor_t* db0_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, db0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tb0_tensor, db0_tensor), TENSOR_LIST(tb0_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(b0_tensor, tb0_tensor), TENSOR_LIST(b0_tensor), 0);
		ccv_nnc_tensor_t* dw1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dw1);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tw1_tensor, dw1_tensor), TENSOR_LIST(tw1_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w1_tensor, tw1_tensor), TENSOR_LIST(w1_tensor), 0);
		ccv_nnc_tensor_t* db1_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, db1);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tb1_tensor, db1_tensor), TENSOR_LIST(tb1_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(b1_tensor, tb1_tensor), TENSOR_LIST(b1_tensor), 0);
		ccv_nnc_tensor_t* dw2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dw2);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tw2_tensor, dw2_tensor), TENSOR_LIST(tw2_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w2_tensor, tw2_tensor), TENSOR_LIST(w2_tensor), 0);
		ccv_nnc_tensor_t* db2_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, db2);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tb2_tensor, db2_tensor), TENSOR_LIST(tb2_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(b2_tensor, tb2_tensor), TENSOR_LIST(b2_tensor), 0);
		ccv_nnc_tensor_t* dw3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, dw3);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tw3_tensor, dw3_tensor), TENSOR_LIST(tw3_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w3_tensor, tw3_tensor), TENSOR_LIST(w3_tensor), 0);
		ccv_nnc_tensor_t* db3_tensor = ccv_nnc_tensor_from_symbol(tensor_arena, db3);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(0.9, 0.00005), ccv_nnc_no_hint, 0, TENSOR_LIST(tb3_tensor, db3_tensor), TENSOR_LIST(tb3_tensor), 0);
		ccv_nnc_cmd_exec(CMD_ADD_FORWARD(1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(b3_tensor, tb3_tensor), TENSOR_LIST(b3_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dw0_tensor), TENSOR_LIST(w0c), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dw1_tensor), TENSOR_LIST(w1c), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dw2_tensor), TENSOR_LIST(w2c), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dw3_tensor), TENSOR_LIST(w3c), 0);
		float mean0 = 0;
		float std0 = 0;
		for (j = 0; j < 32 * 3 * 5 * 5; j++)
			mean0 += w0c->data.f32[j];
		mean0 = mean0 / (32 * 3 * 5 * 5);
		for (j = 0; j < 32 * 3 * 5 * 5; j++)
			std0 += (w0c->data.f32[j] - mean0) * (w0c->data.f32[j] - mean0);
		std0 = std0 / (32 * 3 * 5 * 5);
		float mean1 = 0;
		float std1 = 0;
		for (j = 0; j < 32 * 32 * 5 * 5; j++)
			mean1 += w1c->data.f32[j];
		mean1 = mean1 / (32 * 32 * 5 * 5);
		for (j = 0; j < 32 * 32 * 5 * 5; j++)
			std1 += (w1c->data.f32[j] - mean1) * (w1c->data.f32[j] - mean1);
		std1 = std1 / (32 * 32 * 5 * 5);
		float mean2 = 0;
		float std2 = 0;
		for (j = 0; j < 64 * 32 * 5 * 5; j++)
			mean2 += w2c->data.f32[j];
		mean2 = mean2 / (64 * 32 * 5 * 5);
		for (j = 0; j < 64 * 32 * 5 * 5; j++)
			std2 += (w2c->data.f32[j] - mean2) * (w2c->data.f32[j] - mean2);
		std2 = std2 / (64 * 32 * 5 * 5);
		float std3 = 0;
		float mean3 = 0;
		for (j = 0; j < 10 * 3 * 3 * 64; j++)
			mean3 += w3c->data.f32[j];
		mean3 = mean3 / (10 * 3 * 3 * 64);
		for (j = 0; j < 10 * 3 * 3 * 64; j++)
			std3 += (w3c->data.f32[j] - mean3) * (w3c->data.f32[j] - mean3);
		std3 = std3 / (10 * 3 * 3 * 64);
		if (i % 11 == 0)
			FLUSH(CCV_CLI_INFO, "Batch %d, Correct %f, mean (%f, %f, %f, %f), std (%f, %f, %f, %f)", i + 1, correct_ratio, mean0, mean1, mean2, mean3, std0, std1, std2, std3);
	}
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	assert(argc == 5);
	int num1 = atoi(argv[2]);
	int num2 = atoi(argv[4]);
	FILE* r1 = fopen(argv[1], "rb");
	FILE* r2 = fopen(argv[3], "rb");
	if (r1 && r2)
	{
		int i, j, k;
		unsigned char bytes[32 * 32 + 1];
		double mean[3] = {};
		ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), num1, 0);
		for (k = 0; k < num1; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r1);
			double per_mean[3] = {};
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[0] += (a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[1] += (a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] * 2. / 255.);
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					per_mean[2] += (a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] * 2. / 255.);
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(categorizeds, &categorized);
			mean[0] += per_mean[0] / (31 * 31);
			mean[1] += per_mean[1] / (31 * 31);
			mean[2] += per_mean[2] / (31 * 31);
		}
		ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), num2, 0);
		for (k = 0; k < num2; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r2);
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] * 2. / 255.;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] * 2. / 255.;
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(tests, &categorized);
		}
		float meanf[3];
		meanf[0] = mean[0] / num1;
		meanf[1] = mean[1] / num1;
		meanf[2] = mean[2] / num1;
		train_cifar_10(categorizeds, meanf, tests);
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	return 0;
}
