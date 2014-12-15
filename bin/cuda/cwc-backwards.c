#include "ccv.h"
#include <ctype.h>

void cwc_backwards_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params);

int main(int argc, char** argv)
{
	assert(argc >= 2);
	FILE *r = fopen(argv[1], "r");
	char* file = (char*)malloc(1024);
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	size_t len = 1024;
	ssize_t read;
	while ((read = getline(&file, &len, r)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		ccv_file_info_t input;
		input.filename = (char*)ccmalloc(1024);
		strncpy(input.filename, file, 1024);
		ccv_categorized_t categorized = ccv_categorized(0, 0, &input);
		ccv_array_push(categorizeds, &categorized);
	}
	fclose(r);
	free(file);
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 128,
		.device_count = 2,
		.peer_access = 0,
		.input = {
			.min_dim = 257,
			.max_dim = 257,
		},
	};
	/* MattNet parameters */
	ccv_convnet_layer_param_t params[13] = {
		// first layer (convolutional => max pool => rnorm)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 225,
					.cols = 225,
					.channels = 3,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 96,
					.strides = 2,
					.border = 1,
					.rows = 7,
					.cols = 7,
					.channels = 3,
					.partition = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 111,
					.cols = 111,
					.channels = 96,
					.partition = 2,
				},
			},
			.output = {
				.rnorm = {
					.size = 5,
					.kappa = 2,
					.alpha = 1e-4,
					.beta = 0.75,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 111,
					.cols = 111,
					.channels = 96,
					.partition = 2,
				},
			},
			.output = {
				.pool = {
					.strides = 2,
					.size = 3,
					.border = 0,
				},
			},
		},
		// second layer (convolutional => max pool => rnorm)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 55,
					.cols = 55,
					.channels = 96,
					.partition = 2,
				},
			},
			.output = {
				.convolutional = {
					.count = 256,
					.strides = 2,
					.border = 1,
					.rows = 5,
					.cols = 5,
					.channels = 96,
					.partition = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 27,
					.cols = 27,
					.channels = 256,
					.partition = 2,
				},
			},
			.output = {
				.rnorm = {
					.size = 5,
					.kappa = 2,
					.alpha = 1e-4,
					.beta = 0.75,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 27,
					.cols = 27,
					.channels = 256,
					.partition = 2,
				},
			},
			.output = {
				.pool = {
					.strides = 2,
					.size = 3,
					.border = 0,
				},
			},
		},
		// third layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 256,
					.partition = 1,
				},
			},
			.output = {
				.convolutional = {
					.count = 384,
					.strides = 1,
					.border = 1,
					.rows = 3,
					.cols = 3,
					.channels = 256,
					.partition = 2,
				},
			},
		},
		// fourth layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 384,
					.partition = 2,
				},
			},
			.output = {
				.convolutional = {
					.count = 384,
					.strides = 1,
					.border = 1,
					.rows = 3,
					.cols = 3,
					.channels = 384,
					.partition = 2,
				},
			},
		},
		// fifth layer (convolutional => max pool)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 1,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 384,
					.partition = 2,
				},
			},
			.output = {
				.convolutional = {
					.count = 256,
					.strides = 1,
					.border = 1,
					.rows = 3,
					.cols = 3,
					.channels = 384,
					.partition = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 256,
					.partition = 2,
				},
			},
			.output = {
				.pool = {
					.strides = 2,
					.size = 3,
					.border = 0,
				},
			},
		},
		// sixth layer (full connect)
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 1,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 6,
					.cols = 6,
					.channels = 256,
					.partition = 1,
				},
				.node = {
					.count = 6 * 6 * 256,
				},
			},
			.output = {
				.full_connect = {
					.relu = 1,
					.count = 4096,
				},
			},
		},
		// seventh layer (full connect)
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 1,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 4096,
					.cols = 1,
					.channels = 1,
					.partition = 1,
				},
				.node = {
					.count = 4096,
				},
			},
			.output = {
				.full_connect = {
					.relu = 1,
					.count = 4096,
				},
			},
		},
		// eighth layer (full connect)
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 0,
			.glorot = sqrtf(2),
			.input = {
				.matrix = {
					.rows = 4096,
					.cols = 1,
					.channels = 1,
					.partition = 1,
				},
				.node = {
					.count = 4096,
				},
			},
			.output = {
				.full_connect = {
					.relu = 0,
					.count = 1000,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(1, ccv_size(257, 257), params, sizeof(params) / sizeof(ccv_convnet_layer_param_t));
	ccv_convnet_verify(convnet, 1000);
	ccv_convnet_layer_train_param_t layer_params[16];
	memset(layer_params, 0, sizeof(layer_params));
	train_params.layer_params = layer_params;
	cwc_backwards_runtime(convnet, categorizeds, train_params);
	return 0;
}
