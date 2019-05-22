#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static ccv_cnnp_model_t* _building_block_new(const int filters, const int expansion, const int strides, const int projection_shortcut)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t shortcut = input;
	if (projection_shortcut)
	{
		ccv_cnnp_model_t* const conv0 = ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((strides, strides), (0, 0)),
		});
		shortcut = ccv_cnnp_model_apply(conv0, MODEL_IO_LIST(input));
	}
	ccv_cnnp_model_t* const conv1 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((strides, strides), (0, 0)),
	});
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(conv1, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const conv2 = ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.activation = CCV_CNNP_ACTIVATION_RELU,
		.hint = HINT((1, 1), (1, 1)),
	});
	output = ccv_cnnp_model_apply(conv2, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const conv3 = ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.norm = CCV_CNNP_BATCH_NORM,
		.hint = HINT((1, 1), (0, 0)),
	});
	output = ccv_cnnp_model_apply(conv3, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const add = ccv_cnnp_add();
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output, shortcut));
	ccv_cnnp_model_t* const identity = ccv_cnnp_identity((ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_RELU,
	});
	output = ccv_cnnp_model_apply(identity, MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

static ccv_cnnp_model_t* _block_layer_new(const int filters, const int expansion, const int strides, const int blocks)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* first_block = _building_block_new(filters, expansion, strides, 1);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(first_block, MODEL_IO_LIST(input));
	int i;
	for (i = 1; i < blocks; i++)
	{
		ccv_cnnp_model_t* block = _building_block_new(filters, expansion, 1, 0);
		output = ccv_cnnp_model_apply(block, MODEL_IO_LIST(output));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

ccv_cnnp_model_t* _imagenet_resnet101_v1d(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((2, 2), (1, 1)),
		}),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (1, 1)),
		}),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.norm = CCV_CNNP_BATCH_NORM,
			.activation = CCV_CNNP_ACTIVATION_RELU,
			.hint = HINT((1, 1), (1, 1)),
		}),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		})
	));
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_block_layer_new(64, 4, 1, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(128, 4, 2, 4), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(256, 4, 2, 23), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_block_layer_new(512, 4, 2, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_flatten(), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_dense(1000, (ccv_cnnp_param_t){
		.activation = CCV_CNNP_ACTIVATION_SOFTMAX,
	}), MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output));
}

static void train_imagenet(const int batch_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data)
{
	ccv_cnnp_model_t* const imagenet = _imagenet_resnet101_v1d();
	ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, 16F, batch_size, 3, 224, 224);
	float learn_rate = 0.001;
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	ccv_cnnp_model_compile(imagenet, &input, 1, CMD_SGD_FORWARD(learn_rate, 0.99, 0.9, 0.9), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	ccv_cnnp_model_set_workspace_size(imagenet, 1llu * 1024 * 1024 * 1024);
	const int read_image_idx = ccv_cnnp_dataframe_read_image(train_data, 0, offsetof(ccv_categorized_t, file) + offsetof(ccv_file_info_t, filename));
	ccv_cnnp_random_jitter_t random_jitter = {
		.brightness = 0.4,
		.contrast = 0.4,
		.saturation = 0.4,
		.lighting = 0.1,
		.symmetric = 1,
		.resize = {
			.min = 180,
			.max = 280,
		},
		.normalize = {
			.mean = {
				123.68, 116.779, 103.939
			},
			.std = {
				58.393, 57.12, 57.375
			},
		},
		.aspect_ratio = 0,
		.size = {
			.cols = 224,
			.rows = 224,
		},
	};
	const int image_jitter_idx = ccv_cnnp_dataframe_image_random_jitter(train_data, read_image_idx, CCV_32F, random_jitter);
	ccv_nnc_tensor_param_t fp16_params = CPU_TENSOR_NHWC(16F, 224, 224, 3);
	const int image_jitter_in_idx = ccv_cnnp_dataframe_make_tuple(train_data, COLUMN_ID_LIST(image_jitter_idx));
	const int image_jitter_out_fp16_idx = ccv_cnnp_dataframe_cmd_exec(train_data, image_jitter_in_idx, CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 0, 1, &fp16_params, 1, 0);
	const int image_jitter_fp16_idx = ccv_cnnp_dataframe_extract_tuple(train_data, image_jitter_out_fp16_idx, 0);
	const int one_hot_idx = ccv_cnnp_dataframe_one_hot(train_data, 0, offsetof(ccv_categorized_t, c), 1000, 1, 0, CCV_16F, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_cnnp_dataframe_t* const batch_train_data = ccv_cnnp_dataframe_batching_new(train_data, COLUMN_ID_LIST(image_jitter_fp16_idx, one_hot_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int t, i, j, k;
	int train_device_columns[device_count * 2 + 1];
	for (i = 0; i < device_count; i++)
	{
		int stream_type = CCV_STREAM_CONTEXT_GPU;
		CCV_STREAM_SET_DEVICE_ID(stream_type, i);
		train_device_columns[i] = ccv_cnnp_dataframe_copy_to_gpu(batch_train_data, 0, i * 2, 2, i);
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, 16F, batch_size, 1000);
		CCV_TENSOR_SET_DEVICE_ID(params.type, i);
		train_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux(batch_train_data, params);
	}
	train_device_columns[device_count * 2] = 0;
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch_train_data, train_device_columns, device_count * 2 + 1);
	ccv_nnc_stream_context_t* stream_contexts[2];
	stream_contexts[0] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	stream_contexts[1] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int p = 0, q = 1;
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + batch_size * device_count - 1) / (batch_size * device_count);
	ccv_cnnp_model_set_data_parallel(imagenet, device_count);
	// ccv_cnnp_model_checkpoint(imagenet, "imagenet.checkpoint", 0);
	unsigned int current_time = get_current_time();
	ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[p]);
	ccv_nnc_tensor_t* cpu_outputs[device_count];
	for (i = 0; i < device_count; i++)
		cpu_outputs[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, batch_size, 1000), 0);
	ccv_nnc_tensor_t* oh_fp32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1000), 0);
	ccv_nnc_tensor_t* out_fp32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1000), 0);
	ccv_nnc_tensor_t** input_fits[device_count * 2 + 1];
	ccv_nnc_tensor_t* input_fit_inputs[device_count];
	ccv_nnc_tensor_t* input_fit_fits[device_count];
	ccv_nnc_tensor_t* outputs[device_count];
	uint64_t correct = 0;
	uint64_t all = 0;
	int epoch = 0;
	for (t = 0; epoch < 100; t++)
	{
		ccv_cnnp_dataframe_iter_next(iter, (void**)input_fits, device_count * 2 + 1, stream_contexts[p]);
		ccv_nnc_stream_context_wait(stream_contexts[q]); // Need to wait the other context to finish, we use the same tensor_arena.
		for (i = 0; i < device_count; i++)
		{
			input_fit_inputs[i] = input_fits[i][0];
			input_fit_fits[i] = input_fits[i][1];
			outputs[i] = (ccv_nnc_tensor_t*)input_fits[device_count + i];
		}
		ccv_cnnp_model_fit(imagenet, input_fit_inputs, device_count, input_fit_fits, device_count, outputs, device_count, stream_contexts[p]);
		// Prefetch the next round.
		ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[q]);
		ccv_nnc_stream_context_wait(stream_contexts[p]);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs, device_count, cpu_outputs, device_count, 0);
		if (i == 0)
		{
			FILE* w1 = fopen("imagenet.dot", "w+");
			FILE* w2 = fopen("imagenet-computation.dot", "w+");
			FILE* w[] = {w1, w2};
			ccv_cnnp_model_dot(imagenet, CCV_NNC_LONG_DOT_GRAPH, w, 2);
			fclose(w1);
			fclose(w2);
		}
		all = 0;
		correct = 0;
		for (i = 0; i < device_count; i++)
		{
			ccv_nnc_tensor_t* fit = input_fits[device_count * 2][i * 2 + 1];
			ccv_nnc_tensor_t* cpu_fit = cpu_outputs[i];
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(fit, cpu_fit), TENSOR_LIST(oh_fp32, out_fp32), 0);
			for (j = 0; j < batch_size; j++)
			{
				float max = -1;
				int max_idx = -1;
				for (k = 0; k < 1000; k++)
				{
					assert(!isnan(out_fp32->data.f32[j * 1000 + k]));
					if (out_fp32->data.f32[j * 1000 + k] > max)
					{
						max = out_fp32->data.f32[j * 1000 + k];
						max_idx = k;
					}
				}
				assert(max_idx >= 0);
				int right = -1;
				for (k = 0; k < 1000; k++)
					if (oh_fp32->data.f32[j * 1000 + k] > 0.5)
					{
						right = k;
						break;
					}
				if (right == max_idx)
					++correct;
			}
		}
		all += device_count * batch_size;
		unsigned int elapsed_time = get_current_time() - current_time;
		PRINT(CCV_CLI_INFO, "%.3lf GiB (%.3f seconds), %lf%%\n", (unsigned long)ccv_cnnp_model_memory_size(imagenet) / 1024 / 1024.0 / 1024, (float)elapsed_time / 1000, (double)correct / all * 100);
		if ((i + 1) % epoch_end == 0)
		{
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
		int n;
		CCV_SWAP(p, q, n);
	}
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batch_train_data);
	for (i = 0; i < device_count; i++)
		ccv_nnc_tensor_free(cpu_outputs[i]);
	ccv_nnc_tensor_free(oh_fp32);
	ccv_nnc_tensor_free(out_fp32);
}

static ccv_cnnp_dataframe_t* _dataframe_from_disk_new(const char* const list, const char* const base_dir)
{
	FILE *r = fopen(list, "r");
	assert(r && "list doesn't exists");
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_categorized_t), 64, 0);
	int c;
	char* file = (char*)malloc(1024);
	while (fscanf(r, "%d %s", &c, file) != EOF)
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
	free(file);
	fclose(r);
	ccv_cnnp_dataframe_t* data = ccv_cnnp_dataframe_from_array_new(categorizeds);
	return data;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	static struct option imagenet_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"train-list", 1, 0, 0},
		{"test-list", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	int c;
	char* train_list = 0;
	char* test_list = 0;
	char* base_dir = 0;
	while (getopt_long_only(argc, argv, "", imagenet_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				exit(0);
			case 1:
				train_list = optarg;
				break;
			case 2:
				test_list = optarg;
				break;
			case 3:
				base_dir = optarg;
				break;
		}
	}
	ccv_cnnp_dataframe_t* const train_data = _dataframe_from_disk_new(train_list, base_dir);
	ccv_cnnp_dataframe_t* const test_data = _dataframe_from_disk_new(test_list, base_dir);
	train_imagenet(96, train_data, test_data);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	return 0;
}
