#include <ctype.h>
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>

void cifar_10(void)
{
	ccv_nnc_symbolic_graph_t* const symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t input = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_CPU_TENSOR(128, 31, 31, 3), 0);
	ccv_nnc_tensor_symbol_t x0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 128, 31, 31, 3), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_DATA_TRANSFER_FORWARD(), TENSOR_SYMBOL_LIST(input), TENSOR_SYMBOL_LIST(x0), 0);
	ccv_nnc_tensor_symbol_t w0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 32, 5, 5, 3), 0);
	ccv_nnc_tensor_symbol_t b0 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 32), 0);
	ccv_nnc_tensor_symbol_t x1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly1 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 3), TENSOR_SYMBOL_LIST(x0, w0, b0), TENSOR_SYMBOL_LIST(x1), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly1, HINT((1, 1), (2, 2)));
	ccv_nnc_tensor_symbol_t x2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x1), TENSOR_SYMBOL_LIST(x2), 0);
	ccv_nnc_tensor_symbol_t x3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly2 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_MAX_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x2), TENSOR_SYMBOL_LIST(x3), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly2, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 32, 5, 5, 32), 0);
	ccv_nnc_tensor_symbol_t b1 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 32), 0);
	ccv_nnc_tensor_symbol_t x4 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly3 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(32, 5, 5, 32), TENSOR_SYMBOL_LIST(x3, w1, b1), TENSOR_SYMBOL_LIST(x4), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly3, HINT((1, 1,), (2, 2)));
	ccv_nnc_tensor_symbol_t x5 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x4), TENSOR_SYMBOL_LIST(x5), 0);
	ccv_nnc_tensor_symbol_t x6 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly4 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x5), TENSOR_SYMBOL_LIST(x6), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly4, HINT((2, 2), (0, 0)));
	ccv_nnc_tensor_symbol_t w2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 64, 5, 5, 32), 0);
	ccv_nnc_tensor_symbol_t b2 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 64), 0);
	ccv_nnc_tensor_symbol_t x7 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly5 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_CONVOLUTION_FORWARD(64, 5, 5, 32), TENSOR_SYMBOL_LIST(x6, w2, b2), TENSOR_SYMBOL_LIST(x7), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly5, HINT((1, 1), (2, 2)));
	ccv_nnc_tensor_symbol_t x8 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_RELU_FORWARD(), TENSOR_SYMBOL_LIST(x7), TENSOR_SYMBOL_LIST(x8), 0);
	ccv_nnc_tensor_symbol_t x9 = ccv_nnc_tensor_symbol_new(symbolic_graph, ccv_nnc_tensor_auto, 0);
	ccv_nnc_graph_exec_symbol_t ly6 = ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_AVERAGE_POOL_FORWARD(3, 3), TENSOR_SYMBOL_LIST(x8), TENSOR_SYMBOL_LIST(x9), 0);
	ccv_nnc_graph_exec_symbol_set_hint(symbolic_graph, ly6, HINT((2, 2), (1, 1)));
	ccv_nnc_tensor_symbol_t x9a = ccv_nnc_tensor_symbol_alias_new(symbolic_graph, x9, ccv_nnc_no_ofs, DIM_ALLOC(128, 3 * 3 * 64), ONE_GPU_TENSOR(000, 128, 3 * 3 * 64), 0);
	ccv_nnc_tensor_symbol_t w3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 10, 3 * 3 * 64), 0);
	ccv_nnc_tensor_symbol_t b3 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 10), 0);
	ccv_nnc_tensor_symbol_t x10 = ccv_nnc_tensor_symbol_new(symbolic_graph, ONE_GPU_TENSOR(000, 10), 0);
	ccv_nnc_graph_exec_symbol_new(symbolic_graph, CMD_GEMM_FORWARD(10), TENSOR_SYMBOL_LIST(x9a, w3, b3), TENSOR_SYMBOL_LIST(x10), 0);
}

int main(int argc, char** argv)
{
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
		/*
		layer_params[0].w.decay = 0.005;
		layer_params[0].w.learn_rate = 0.001;
		layer_params[0].w.momentum = 0.9;
		layer_params[0].bias.decay = 0;
		layer_params[0].bias.learn_rate = 0.001;
		layer_params[0].bias.momentum = 0.9;

		layer_params[3].w.decay = 0.005;
		layer_params[3].w.learn_rate = 0.001;
		layer_params[3].w.momentum = 0.9;
		layer_params[3].bias.decay = 0;
		layer_params[3].bias.learn_rate = 0.001;
		layer_params[3].bias.momentum = 0.9;

		layer_params[6].w.decay = 0.005;
		layer_params[6].w.learn_rate = 0.001;
		layer_params[6].w.momentum = 0.9;
		layer_params[6].bias.decay = 0;
		layer_params[6].bias.learn_rate = 0.001;
		layer_params[6].bias.momentum = 0.9;

		layer_params[8].w.decay = 0.01;
		layer_params[8].w.learn_rate = 0.001;
		layer_params[8].w.momentum = 0.9;
		layer_params[8].bias.decay = 0;
		layer_params[8].bias.learn_rate = 0.001;
		layer_params[8].bias.momentum = 0.9;

		ccv_convnet_train_param_t params = {
			.max_epoch = 999,
			.mini_batch = 128,
			.iterations = 500,
			.symmetric = 1,
			.color_gain = 0,
			.device_count = 1,
			.layer_params = layer_params,
		};
		ccv_convnet_supervised_train(convnet, categorizeds, tests, "cifar-10.sqlite3", params);
		*/
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	return 0;
}
