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
		ccv_categorized_t categorized;
		categorized.file.filename = (char*)ccmalloc(1024);
		strncpy(categorized.file.filename, file, 1024);
		ccv_array_push(categorizeds, &categorized);
	}
	fclose(r);
	free(file);
	ccv_convnet_param_t params[] = {
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
					.size = 5,
					.border = 1,
				},
			},
		},
	};
	ccv_convnet_t* convnet = ccv_convnet_new(params, 2);
	ccv_convnet_train_param_t train_params = {
		.max_epoch = 100,
		.mini_batch = 128,
		.decay = 0.005,
		.learn_rate = 0.00008,
		.momentum = 0.9,
	};
	int i;
	for (i = 0; i < convnet->layers->wnum; i++)
		convnet->layers->w[i] = 1;
	for (i = 0; i < convnet->layers->net.convolutional.count; i++)
		convnet->layers->bias[i] = 1;
	ccv_convnet_supervised_train(convnet, categorizeds, 0, train_params);
	ccv_convnet_free(convnet);
	ccv_disable_cache();
	return 0;
}
