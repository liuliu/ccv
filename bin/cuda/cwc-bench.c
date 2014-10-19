#include "ccv.h"
#include <ctype.h>

void cwc_bench_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params);

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
	/* AlexNet 12 (ImageNet 2012 winner)
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
					.strides = 4,
					.border = 1,
					.rows = 11,
					.cols = 11,
					.channels = 3,
					.partition = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 55,
					.cols = 55,
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
					.rows = 55,
					.cols = 55,
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
					.rows = 27,
					.cols = 27,
					.channels = 96,
					.partition = 2,
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
	*/
	/* AlexNet 14 (One Weird Trick)
	 * Note that Alex claimed that this is a one tower model,
	 * but if this is a true one tower model, it should has
	 * 11 * 11 * 64 * 3 + 5 * 5 * 64 * 192 + 3 * 3 * 192 * 384 + 3 * 3 * 384 * 384 + 3 * 3 * 384 * 256 + 6 * 6 * 256 * 4096 + 4096 * 4096 + 4096 * 1000 = 61827776 parameters
	 * However, AlexNet 12 (ImageNet 2012 winner, the two towers model) has
	 * 11 * 11 * 96 * 3 + 5 * 5 * 96 * 256 / 2 + 3 * 3 * 256 * 384 + 3 * 3 * 384 * 384 / 2 + 3 * 3 * 384 * 256 / 2 + 6 * 6 * 256 * 4096 + 4096 * 4096 + 4096 * 1000 = 60954656 parameters
	 * That works out to be (61827776 - 60954656) / 60954656 = 1.4% more parameters
	 * The (One Weird Trick claimed to have only 0.2% more parameters, that works out to be around 61076565 parameters
	 * Thus, the following model, with
	 * 11 * 11 * 64 * 3 + 5 * 5 * 64 * 192 / 2 + 3 * 3 * 192 * 384 + 3 * 3 * 384 * 384 / 2 + 3 * 3 * 384 * 256 + 6 * 6 * 256 * 4096 + 4096 * 4096 + 4096 * 1000 = 61010624 parameters
	 * seems to be the closest (and the libccv's implementation works out to be roughly 500ms per 128 examples, about 100+ hours for 90 epochs, about the same performance as (One Weird Trick)'s one GPU case
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
					.count = 64,
					.strides = 4,
					.border = 1,
					.rows = 11,
					.cols = 11,
					.channels = 3,
					.partition = 2,
				},
			},
		},
		{
			.type = CCV_CONVNET_LOCAL_RESPONSE_NORM,
			.input = {
				.matrix = {
					.rows = 55,
					.cols = 55,
					.channels = 64,
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
					.rows = 55,
					.cols = 55,
					.channels = 64,
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
					.rows = 27,
					.cols = 27,
					.channels = 64,
					.partition = 2,
				},
			},
			.output = {
				.convolutional = {
					.count = 192,
					.strides = 1,
					.border = 2,
					.rows = 5,
					.cols = 5,
					.channels = 64,
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
					.channels = 192,
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
					.channels = 192,
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
					.channels = 192,
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
					.channels = 192,
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
					.partition = 1,
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
					.partition = 1,
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
					.partition = 1,
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
	*/
	ccv_convnet_t* convnet = ccv_convnet_new(1, ccv_size(225, 225), params, sizeof(params) / sizeof(ccv_convnet_layer_param_t));
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
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 128,
		.device_count = 1,
		.layer_params = layer_params,
	};
	for (i = 0; i < 128; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		ccv_dense_matrix_t* b = 0;
		if (image->rows > 225 && image->cols > 225)
			ccv_resample(image, &b, 0, ccv_max(225, (int)(image->rows * 225.0 / image->cols + 0.5)), ccv_max(225, (int)(image->cols * 225.0 / image->rows + 0.5)), CCV_INTER_AREA);
		else if (image->rows < 225 || image->cols < 225)
			ccv_resample(image, &b, 0, ccv_max(225, (int)(image->rows * 225.0 / image->cols + 0.5)), ccv_max(225, (int)(image->cols * 225.0 / image->rows + 0.5)), CCV_INTER_CUBIC);
		else
			b = image;
		if (b != image)
			ccv_matrix_free(image);
		ccv_dense_matrix_t* c = 0;
		ccv_slice(b, (ccv_matrix_t**)&c, CCV_32F, 0, 0, 225, 225);
		ccv_matrix_free(b);
		categorized->type = CCV_CATEGORIZED_DENSE_MATRIX;
		categorized->matrix = c;
	}
	cwc_bench_runtime(convnet, categorizeds, train_params);
	ccv_disable_cache();
	return 0;
}
