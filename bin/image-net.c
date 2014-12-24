#include "ccv.h"
#include "ccv_internal.h"
#include <ctype.h>
#include <getopt.h>
#include "models.inl"

static void exit_with_help(void)
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
		.sgd_frequency = 1, // do sgd every sgd_frequency batches (mini_batch * device_count * sgd_frequency)
		.iterations = 50000,
		.device_count = 4,
		.peer_access = 1,
		.symmetric = 1,
		.image_manipulation = 0.2,
		.color_gain = 0.001,
		.input = {
			.min_dim = 257,
			.max_dim = 257,
		},
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
// #define model_params vgg_d_params
#define model_params matt_params
	int depth = sizeof(model_params) / sizeof(ccv_convnet_layer_param_t);
	ccv_convnet_t* convnet = ccv_convnet_new(1, ccv_size(257, 257), model_params, depth);
	ccv_convnet_verify(convnet, 1000);
	ccv_convnet_layer_train_param_t layer_params[depth];
	memset(layer_params, 0, sizeof(layer_params));
	for (i = 0; i < depth; i++)
	{
		layer_params[i].w.decay = 0.0005;
		layer_params[i].w.learn_rate = 0.01;
		layer_params[i].w.momentum = 0.9;
		layer_params[i].bias.decay = 0;
		layer_params[i].bias.learn_rate = 0.01;
		layer_params[i].bias.momentum = 0.9;
	}
	// set the two full connect layers to last with dropout rate at 0.5
	for (i = depth - 3; i < depth - 1; i++)
		layer_params[i].dor = 0.5;
	train_params.layer_params = layer_params;
	ccv_set_cli_output_levels(ccv_cli_output_level_and_above(CCV_CLI_INFO));
	ccv_convnet_supervised_train(convnet, categorizeds, tests, working_dir, train_params);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
