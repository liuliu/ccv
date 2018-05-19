#include <ctype.h>
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

static void glorot_random_device(ccv_nnc_tensor_t* const b, float glorot)
{
	ccv_nnc_tensor_param_t params = b->info;
	params.type = CCV_TENSOR_CPU_MEMORY;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_RANDOM_UNIFORM_FORWARD(-glorot, glorot), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_free(b);
}

static void train_cifar_10(ccv_array_t* const training_set, ccv_array_t* const test_set)
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t input = ccv_nnc_tensor_symbol_new(symbolic_graph, CPU_TENSOR_NHWC(128, 31, 31, 3), "input");
	ccv_nnc_tensor_symbol_t x0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 128, 31, 31, 3), "x0");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(input), TENSOR_SYMBOL_LIST(x0), 0);
	ccv_nnc_tensor_symbol_t w0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32, 5, 5, 3), "w0");
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32), "b0");
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x1");
	ccv_nnc_graph_exec_symbol_t ly1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 3), TENSOR_SYMBOL_LIST(x0, w0, b0), TENSOR_SYMBOL_LIST(x1), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly1, HINT((1, 1), (2, 2)));
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x2");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x1), TENSOR_SYMBOL_LIST(x2), 0);
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x3");
	ccv_nnc_graph_exec_symbol_t ly2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x2), TENSOR_SYMBOL_LIST(x3), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly2, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32, 5, 5, 32), "w1");
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 32), "b1");
	ccv_nnc_tensor_symbol_t x4 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x4");
	ccv_nnc_graph_exec_symbol_t ly3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 32), TENSOR_SYMBOL_LIST(x3, w1, b1), TENSOR_SYMBOL_LIST(x4), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly3, HINT((1, 1,), (2, 2)));
	ccv_nnc_tensor_symbol_t x5 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x5");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x4), TENSOR_SYMBOL_LIST(x5), 0);
	ccv_nnc_tensor_symbol_t x6 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x6");
	ccv_nnc_graph_exec_symbol_t ly4 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x5), TENSOR_SYMBOL_LIST(x6), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly4, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 64, 5, 5, 32), "w2");
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 64), "b2");
	ccv_nnc_tensor_symbol_t x7 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x7");
	ccv_nnc_graph_exec_symbol_t ly5 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(64, 5, 5, 32), TENSOR_SYMBOL_LIST(x6, w2, b2), TENSOR_SYMBOL_LIST(x7), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly5, HINT((1, 1), (2, 2)));
	ccv_nnc_tensor_symbol_t x8 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x8");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x7), TENSOR_SYMBOL_LIST(x8), 0);
	ccv_nnc_tensor_symbol_t x9 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, "x9");
	ccv_nnc_graph_exec_symbol_t ly6 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x8), TENSOR_SYMBOL_LIST(x9), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly6, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t x9a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x9, ccv_nnc_no_ofs, DIM_ALLOC(128, 3 * 3 * 64), GPU_TENSOR_NHWC(000, 128, 3 * 3 * 64), "x9a");
	ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 10, 3 * 3 * 64), "w3");
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 10), "b3");
	ccv_nnc_tensor_symbol_t x10 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 10), "x10");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(10), TENSOR_SYMBOL_LIST(x9a, w3, b3), TENSOR_SYMBOL_LIST(x10), 0);
	ccv_nnc_tensor_symbol_t x11 = ccv_nnc_tensor_symbol_new(symbolic_graph, GPU_TENSOR_NHWC(000, 10), "x11");
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_SOFTMAX_FORWARD(), TENSOR_SYMBOL_LIST(x10), TENSOR_SYMBOL_LIST(x11), 0);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_symbolic_graph_backward(symbolic_graph, SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph), TENSOR_SYMBOL_LIST(x11), TENSOR_SYMBOL_LIST(w0, b0, w1, b1, w2, b2, w3, b3));
	ccv_nnc_tensor_symbol_t dw0 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w0);
	ccv_nnc_tensor_symbol_t db0 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b0);
	ccv_nnc_tensor_symbol_t dw1 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w1);
	ccv_nnc_tensor_symbol_t db1 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b1);
	ccv_nnc_tensor_symbol_t dw2 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w2);
	ccv_nnc_tensor_symbol_t db2 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b2);
	ccv_nnc_tensor_symbol_t dw3 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, w3);
	ccv_nnc_tensor_symbol_t db3 = ccv_nnc_tensor_symbol_for_backward(symbolic_graph, b3);
	ccv_nnc_graph_exec_symbol_autogen(symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* graph = 0;
	ccv_nnc_tensor_arena_t* tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena = 0;
	ccv_nnc_tensor_t* w0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32, 5, 5, 3), 0);
	glorot_random_device(w0_tensor, sqrtf(2) / sqrtf(5 * 5 * 3 + 32));
	ccv_nnc_tensor_t* b0_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b0_tensor), 0);
	ccv_nnc_tensor_t* w1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32, 5, 5, 32), 0);
	glorot_random_device(w1_tensor, sqrtf(2) / sqrtf(5 * 5 * 32 + 32));
	ccv_nnc_tensor_t* b1_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b1_tensor), 0);
	ccv_nnc_tensor_t* w2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64, 5, 5, 32), 0);
	glorot_random_device(w2_tensor, sqrtf(2) / sqrtf(5 * 5 * 32 + 64));
	ccv_nnc_tensor_t* b2_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 64), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b2_tensor), 0);
	ccv_nnc_tensor_t* w3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 10, 3 * 3 * 64), 0);
	glorot_random_device(w3_tensor, sqrtf(2) / sqrtf(3 * 3 * 64 + 10));
	ccv_nnc_tensor_t* b3_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 10), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(b3_tensor), 0);
	ccv_nnc_symbolic_graph_compile(symbolic_graph,
		TENSOR_BIND_MAP(KV(w0, w0_tensor), KV(b0, b0_tensor), KV(w1, w1_tensor), KV(b1, b1_tensor), KV(w2, w2_tensor), KV(b2, b2_tensor), KV(w3, w3_tensor), KV(b3, b3_tensor)),
		TENSOR_SYMBOL_LIST(dw0, db0, dw1, db1, dw2, db2, dw3, db3),
		SYMBOLIC_GRAPH_SOURCES(symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(symbolic_graph),
		&graph, &tensor_arena, &graph_exec_arena);
	int i;
	for (i = 0; i < 100; i++)
	{
		ccv_nnc_graph_run(graph, 0, 0, TRAVERSE_FULL);
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
		ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), num1, 0);
		for (k = 0; k < num1; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r1);
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1];
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32];
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32];
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(categorizeds, &categorized);
		}
		ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), num2, 0);
		for (k = 0; k < num2; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r2);
			int c = bytes[0];
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(32, 32, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3] = bytes[j + i * 32 + 1];
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 1] = bytes[j + i * 32];
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 32; i++)
				for (j = 0; j < 32; j++)
					a->data.f32[(j + i * 32) * 3 + 2] = bytes[j + i * 32];
			ccv_categorized_t categorized = ccv_categorized(c, a, 0);
			ccv_array_push(tests, &categorized);
		}
		train_cifar_10(categorizeds, tests);
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	return 0;
}
