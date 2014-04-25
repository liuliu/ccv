#include "ccv.h"
#include "ccv_internal.h"
#include <ctype.h>
#include <getopt.h>

void exit_with_help(void)
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    image-net [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --train-list : text file contains a list of image files in format:\n"
	"                      class-label <file name>\\newline\n"
	"    --test-list : text file contains a list of image files in format:\n"
	"                      class-label <file name>\\newline\n"
	"    --working-dir : the directory to save progress and produce result model\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	"    --max-epoch : how many epoch are needed for stochastic gradient descent (an epoch corresponds to go through the full train list) [DEFAULT TO 100]\n"
	"    --iterations : how many iterations are needed for stochastic gradient descent (an iteration corresponds to go through a mini batch) [DEFAULT TO 5000]\n\n"
	);
	exit(0);
}

int main(int argc, char** argv)
{
	static struct option image_net_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"train-list", 1, 0, 0},
		{"test-list", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{"max-epoch", 1, 0, 0},
		{"iterations", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* train_list = 0;
	char* test_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 128,
		.iterations = 20000,
		.symmetric = 1,
		.color_gain = 0.001,
	};
	int i, c;
	while (getopt_long_only(argc, argv, "", image_net_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				exit_with_help();
			case 1:
				train_list = optarg;
				break;
			case 2:
				test_list = optarg;
				break;
			case 3:
				working_dir = optarg;
				break;
			case 4:
				base_dir = optarg;
				break;
			case 5:
				train_params.max_epoch = atoi(optarg);
				break;
			case 6:
				train_params.iterations = atoi(optarg);
				break;
		}
	}
	if (!train_list || !test_list || !working_dir)
		exit_with_help();
	ccv_enable_default_cache();
	FILE *r0 = fopen(train_list, "r");
	assert(r0 && "train-list doesn't exists");
	FILE* r1 = fopen(test_list, "r");
	assert(r1 && "test-list doesn't exists");
	char* file = (char*)malloc(1024);
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	while (fscanf(r0, "%d %s", &c, file) != EOF)
	{
		char* filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
		}
		strncpy(filename + dirlen, file, 1024 - dirlen);
		ccv_file_info_t file_info = {
			.filename = filename,
		};
		// imageNet's category class starts from 1, thus, minus 1 to get 0-index
		ccv_categorized_t categorized = ccv_categorized(c - 1, 0, &file_info);
		ccv_array_push(categorizeds, &categorized);
	}
	fclose(r0);
	ccv_array_t* tests = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	while (fscanf(r1, "%d %s", &c, file) != EOF)
	{
		char* filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
		}
		strncpy(filename + dirlen, file, 1024 - dirlen);
		ccv_file_info_t file_info = {
			.filename = filename,
		};
		// imageNet's category class starts from 1, thus, minus 1 to get 0-index
		ccv_categorized_t categorized = ccv_categorized(c - 1, 0, &file_info);
		ccv_array_push(tests, &categorized);
	}
	fclose(r1);
	free(file);
	ccv_convnet_layer_param_t params[13] = {
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
			.sigma = 0.01,
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
	ccv_convnet_layer_train_param_t layer_params[13];
	memset(layer_params, 0, sizeof(layer_params));
	for (i = 0; i < 13; i++)
	{
		layer_params[i].w.decay = 0.0005;
		layer_params[i].w.learn_rate = 0.01;
		layer_params[i].w.momentum = 0.9;
		layer_params[i].bias.decay = 0;
		layer_params[i].bias.learn_rate = 0.01;
		layer_params[i].bias.momentum = 0.9;
	}
	layer_params[10].dor = 0.5;
	layer_params[11].dor = 0.5;
	train_params.layer_params = layer_params;
	ccv_convnet_supervised_train(convnet, categorizeds, tests, working_dir, train_params);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
