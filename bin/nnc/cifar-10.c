#include "ccv.h"
#include <ctype.h>

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
		ccv_convnet_layer_train_param_t layer_params[9];
		memset(layer_params, 0, sizeof(layer_params));
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
