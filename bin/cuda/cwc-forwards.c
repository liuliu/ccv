#include "ccv.h"
#include <ctype.h>

void cwc_forwards_runtime(ccv_convnet_t* convnet, ccv_array_t* categorizeds, ccv_convnet_train_param_t params);

int main(int argc, char** argv)
{
	assert(argc >= 3);
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
		.device_count = 4,
		.peer_access = 0,
		.input = {
			.min_dim = 257,
			.max_dim = 257,
		},
	};
	ccv_convnet_layer_train_param_t layer_params[16];
	memset(layer_params, 0, sizeof(layer_params));
	train_params.layer_params = layer_params;
	ccv_convnet_t *convnet = ccv_convnet_read(1, argv[2]);
	cwc_forwards_runtime(convnet, categorizeds, train_params);
	return 0;
}
