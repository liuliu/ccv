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
	assert(argc == 2);
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
	ccv_convnet_layer_param_t params[] = {
		// first layer (convolutional => max pool => rnorm)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 225,
					.cols = 225,
					.channels = 3,
				},
			},
			.output = {
				.convolutional = {
					.count = 96,
					.strides = 4,
					.border = 1,
					.rows = 11,
					.cols = 11,
					.channels = 3,
				},
			},
		},
		{
			.type = CCV_CONVNET_MAX_POOL,
			.input = {
				.matrix = {
					.rows = 55,
					.cols = 55,
					.channels = 96,
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
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 27,
					.cols = 27,
					.channels = 96,
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
		// second layer (convolutional => max pool => rnorm)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.input = {
				.matrix = {
					.rows = 27,
					.cols = 27,
					.channels = 96,
				},
			},
			.output = {
				.convolutional = {
					.count = 256,
					.strides = 1,
					.border = 2,
					.rows = 5,
					.cols = 5,
					.channels = 96,
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
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 256,
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
		// third layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 256,
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
				},
			},
		},
		// fourth layer (convolutional)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 384,
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
				},
			},
		},
		// fifth layer (convolutional => max pool)
		{
			.type = CCV_CONVNET_CONVOLUTIONAL,
			.input = {
				.matrix = {
					.rows = 13,
					.cols = 13,
					.channels = 384,
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
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 6,
					.cols = 6,
					.channels = 256,
				},
				.node = {
					.count = 6 * 6 * 256,
				},
			},
			.output = {
				.full_connect = {
					.count = 4096,
				},
			},
		},
		// seventh layer (full connect)
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 4096,
					.cols = 1,
					.channels = 1,
				},
				.node = {
					.count = 4096,
				},
			},
			.output = {
				.full_connect = {
					.count = 4096,
				},
			},
		},
		// eighth layer (full connect)
		{
			.type = CCV_CONVNET_FULL_CONNECT,
			.bias = 0,
			.sigma = 0.01,
			.input = {
				.matrix = {
					.rows = 4096,
					.cols = 1,
					.channels = 1,
				},
				.node = {
					.count = 4096,
				},
			},
			.output = {
				.full_connect = {
					.count = 1000,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(1, params, sizeof(params) / sizeof(ccv_convnet_layer_param_t));
	ccv_convnet_verify(convnet, 1000);
	ccv_convnet_layer_train_param_t layer_params[13];
	memset(layer_params, 0, sizeof(layer_params));
	int i;
	for (i = 0; i < 13; i++)
	{
		layer_params[i].w.decay = 0.005;
		layer_params[i].w.learn_rate = 0.0005;
		layer_params[i].w.momentum = 0.9;
		layer_params[i].bias.decay = 0;
		layer_params[i].bias.learn_rate = 0.001;
		layer_params[i].bias.momentum = 0.9;
	}
	layer_params[11].dor = 0.5;
	layer_params[12].dor = 0.5;
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 256,
		.layer_params = layer_params,
	};
	ccv_convnet_supervised_train(convnet, categorizeds, categorizeds, train_params);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
