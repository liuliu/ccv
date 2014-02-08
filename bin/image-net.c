#include "ccv.h"
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
	exit(-1);
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
		.mini_batch = 256,
		.iterations = 5000,
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
		ccv_categorized_t categorized = ccv_categorized(c, 0, &file_info);
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
		ccv_categorized_t categorized = ccv_categorized(c, 0, &file_info);
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
			.bias = 1,
			.sigma = 0.01,
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
			.bias = 0,
			.sigma = 0.01,
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
			.bias = 1,
			.sigma = 0.01,
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
			.bias = 1,
			.sigma = 0.01,
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
			.bias = 1,
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
			.bias = 1,
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
	for (i = 0; i < 13; i++)
	{
		layer_params[i].w.decay = 0.0005;
		layer_params[i].w.learn_rate = 0.00000001;
		layer_params[i].w.momentum = 0.9;
		layer_params[i].bias.decay = 0;
		layer_params[i].bias.learn_rate = 0.00000001;
		layer_params[i].bias.momentum = 0.9;
	}
	layer_params[10].dor = 0.5;
	layer_params[11].dor = 0.5;
	train_params.layer_params = layer_params;
	train_params.size = ccv_size(251, 251);
	/*
	ccv_size_t size = ccv_size(251, 251);
	ccv_dense_matrix_t* mean = ccv_dense_matrix_new(251, 251, CCV_64F | CCV_C3, 0, 0);
	for (i = 203228; i < categorizeds->rnum; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		if (image)
		{
			ccv_dense_matrix_t* norm = 0;
			if (image->rows > size.height && image->cols > size.width)
				ccv_resample(image, &norm, 0, ccv_max(size.height, (int)(image->rows * (float)size.height / image->cols + 0.5)), ccv_max(size.width, (int)(image->cols * (float)size.width / image->rows + 0.5)), CCV_INTER_AREA);
			else if (image->rows < size.height || image->cols < size.width)
				ccv_resample(image, &norm, 0, ccv_max(size.height, (int)(image->rows * (float)size.height / image->cols + 0.5)), ccv_max(size.width, (int)(image->cols * (float)size.width / image->rows + 0.5)), CCV_INTER_CUBIC);
			else
				norm = image;
			if (norm != image)
				ccv_matrix_free(image);
			char filename[1024];
			snprintf(filename, 1024, "%s.resize.png", categorized->file.filename);
			ccv_write(norm, filename, 0, CCV_IO_PNG_FILE, 0);
			ccv_dense_matrix_t* patch = 0;
			int x = (norm->cols - size.width) / 2;
			int y = (norm->rows - size.height) / 2;
			ccv_slice(norm, (ccv_matrix_t**)&patch, CCV_64F, y, x, size.width, size.height);
			ccv_matrix_free(norm);
			int j = 0;
			for (j = 0; j < patch->rows * patch->cols * 3; j++)
				mean->data.f64[j] += patch->data.f64[j];
			ccv_matrix_free(patch);
			printf("done %s, %d / %d\n", filename, i + 1, categorizeds->rnum);
		}
	}
	for (i = 0; i < size.width * size.height * 3; i++)
		mean->data.f64[i] /= categorizeds->rnum;
	ccv_write(mean, "mean.bin", 0, CCV_IO_BINARY_FILE, 0);
	ccv_matrix_free(mean);
	for (i = 0; i < tests->rnum; i++)
	{
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(tests, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
		if (image)
		{
			ccv_dense_matrix_t* norm = 0;
			if (image->rows > size.height && image->cols > size.width)
				ccv_resample(image, &norm, 0, ccv_max(size.height, (int)(image->rows * (float)size.height / image->cols + 0.5)), ccv_max(size.width, (int)(image->cols * (float)size.width / image->rows + 0.5)), CCV_INTER_AREA);
			else if (image->rows < size.height || image->cols < size.width)
				ccv_resample(image, &norm, 0, ccv_max(size.height, (int)(image->rows * (float)size.height / image->cols + 0.5)), ccv_max(size.width, (int)(image->cols * (float)size.width / image->rows + 0.5)), CCV_INTER_CUBIC);
			else
				norm = image;
			if (norm != image)
				ccv_matrix_free(image);
			char filename[1024];
			snprintf(filename, 1024, "%s.resize.png", categorized->file.filename);
			ccv_write(norm, filename, 0, CCV_IO_PNG_FILE, 0);
			ccv_matrix_free(norm);
			printf("done %s, %d / %d\n", filename, i + 1, tests->rnum);
		}
	}
	*/
	ccv_convnet_supervised_train(convnet, categorizeds, tests, working_dir, train_params);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
