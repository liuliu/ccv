#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_enable_default_cache();
	ccv_convnet_param_t params[] = {
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.dropout_rate = 0,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 3,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 3,
					.border = 2,
					.strides = 1,
					.count = 32,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 31,
					.cols = 31,
					.channels = 32,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.dropout_rate = 0,
			.input = {
				.matrix = {
					.rows = 15,
					.cols = 15,
					.channels = 32,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 32,
					.border = 2,
					.strides = 1,
					.count = 32,
				},
			},
		},
		{
			.type = CCV_CONVNET_AVERAGE_POOL,
			.input = {
				.matrix = {
					.rows = 15,
					.cols = 15,
					.channels = 32,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.dropout_rate = 0,
			.input = {
				.matrix = {
					.rows = 7,
					.cols = 7,
					.channels = 32,
				},
			},
			.output = {
				.convolutional = {
					.rows = 5,
					.cols = 5,
					.channels = 32,
					.border = 2,
					.strides = 1,
					.count = 64,
				},
			},
		},
		{
			.type = CCV_CONVNET_AVERAGE_POOL,
			.input = {
				.matrix = {
					.rows = 7,
					.cols = 7,
					.channels = 64,
				},
			},
			.output = {
				.pool = {
					.size = 3,
					.strides = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 0,
			.sigma = 0.01,
			.dropout_rate = 0,
			.input = {
				.matrix = {
					.rows = 3,
					.cols = 3,
					.channels = 64,
				},
				.node = {
					.count = 3 * 3 * 64,
				},
			},
			.output = {
				.full_connect = {
					.count = 10,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(params, 7);
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
			ccv_categorized_t categorized;
			categorized.c = bytes[0]; // the class
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] / 255.0 * 2 - 1;
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] / 255.0 * 2 - 1;
			fread(bytes, 32 * 32, 1, r1);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] / 255.0 * 2 - 1;
			categorized.matrix = a;
			ccv_array_push(categorizeds, &categorized);
		}
		ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), num2, 0);
		for (k = 0; k < num2; k++)
		{
			fread(bytes, 32 * 32 + 1, 1, r2);
			ccv_categorized_t categorized;
			categorized.c = bytes[0]; // the class
			ccv_dense_matrix_t* a = ccv_dense_matrix_new(31, 31, CCV_32F | CCV_C3, 0, 0);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3] = bytes[j + i * 32 + 1] / 255.0 * 2 - 1;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 1] = bytes[j + i * 32] / 255.0 * 2 - 1;
			fread(bytes, 32 * 32, 1, r2);
			for (i = 0; i < 31; i++)
				for (j = 0; j < 31; j++)
					a->data.f32[(j + i * 31) * 3 + 2] = bytes[j + i * 32] / 255.0 * 2 - 1;
			categorized.matrix = a;
			ccv_array_push(tests, &categorized);
		}
		ccv_convnet_train_param_t params = {
			.max_epoch = 100,
			.mini_batch = 128,
			.decay = 0.005,
			.learn_rate = 0.00005,
			.momentum = 0.9,
		};
		ccv_convnet_supervised_train(convnet, categorizeds, tests, params);
	}
	if (r1)
		fclose(r1);
	if (r2)
		fclose(r2);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
