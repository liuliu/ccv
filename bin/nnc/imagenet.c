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

static ccv_cnnp_model_t* _resnet_block_new(const int filters, const int expansion, const int strides, const int projection_shortcut)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t shortcut = input;
	if (projection_shortcut)
	{
		ccv_cnnp_model_t* const avgdown = ccv_cnnp_average_pool(DIM_ALLOC(strides, strides), (ccv_cnnp_param_t){
			.hint = HINT((strides, strides), (0, 0))
		}, 0);
		shortcut = ccv_cnnp_model_apply(avgdown, MODEL_IO_LIST(input));
		ccv_cnnp_model_t* const conv0 = ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (0, 0)),
		}, 0);
		shortcut = ccv_cnnp_model_apply(conv0, MODEL_IO_LIST(shortcut));
	}
	ccv_cnnp_model_t* const conv1 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0)
	), 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(conv1, MODEL_IO_LIST(input));
	ccv_cnnp_model_t* const conv2 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((strides, strides), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0)
	), 0);
	output = ccv_cnnp_model_apply(conv2, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const conv3 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, filters * expansion, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0)
	), 0);
	output = ccv_cnnp_model_apply(conv3, MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const add = ccv_cnnp_add(0);
	output = ccv_cnnp_model_apply(add, MODEL_IO_LIST(output, shortcut));
	ccv_cnnp_model_t* const relu = ccv_cnnp_relu(0);
	output = ccv_cnnp_model_apply(relu, MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 0);
}

static ccv_cnnp_model_t* _resnet_block_layer_new(const int filters, const int expansion, const int strides, const int blocks)
{
	ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* first_block = _resnet_block_new(filters, expansion, strides, 1);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(first_block, MODEL_IO_LIST(input));
	int i;
	for (i = 1; i < blocks; i++)
	{
		ccv_cnnp_model_t* block = _resnet_block_new(filters, expansion, 1, 0);
		output = ccv_cnnp_model_apply(block, MODEL_IO_LIST(output));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 0);
}

ccv_cnnp_model_t* _imagenet_resnet50_v1d(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0)
	), 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(64, 4, 1, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(128, 4, 2, 4), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(256, 4, 2, 6), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(512, 4, 2, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_dense(1000, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 0);
}

ccv_cnnp_model_t* _imagenet_resnet101_v1d(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* init_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0)
	), 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(64, 4, 1, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(128, 4, 2, 4), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(256, 4, 2, 23), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(512, 4, 2, 3), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_dense(1000, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 0);
}

ccv_cnnp_model_t* _imagenet_vgg13(void)
{
	ccv_cnnp_model_t* vgg13 = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 64, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_convolution(1, 128, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 128, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_convolution(1, 256, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 256, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_convolution(1, 512, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 512, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_max_pool(DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_convolution(1, 512, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_convolution(1, 512, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (1, 1)),
		}, 0),
		ccv_cnnp_relu(0),
		ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0),
		ccv_cnnp_flatten(0),
		ccv_cnnp_dense(1000, (ccv_cnnp_param_t){}, 0),
		ccv_cnnp_softmax(0)
	), 0);
	return vgg13;
}

static ccv_cnnp_model_t* _mconv_block_new(const int kernel_size, const int strides, const int expand_ratio, const int input_filters, const int output_filters, const float se_ratio, const float dropout)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_io_t x = input;
	int expand_filters = input_filters;
	if (expand_ratio > 1)
	{
		expand_filters = input_filters * expand_ratio;
		ccv_cnnp_model_t* const expand_conv = ccv_cnnp_sequential_new(MODEL_LIST(
			ccv_cnnp_convolution(1, expand_filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
				.no_bias = 1,
				.hint = HINT((1, 1), (0, 0)),
			}, 0),
			ccv_cnnp_batch_norm(0.9, 1e-4, 0),
			ccv_cnnp_swish(0)
		), 0);
		x = ccv_cnnp_model_apply(expand_conv, MODEL_IO_LIST(x));
	}
	const int paddings = (kernel_size - 1) / 2;
	ccv_cnnp_model_t* const depthwise_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(expand_filters, expand_filters, DIM_ALLOC(kernel_size, kernel_size), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((strides, strides), (paddings, paddings)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_swish(0)
	), 0);
	x = ccv_cnnp_model_apply(depthwise_conv, MODEL_IO_LIST(x));
	const int se_filters = ccv_max(1, (int)(input_filters * se_ratio + 0.5));
	ccv_cnnp_model_t* const se_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0),
		ccv_cnnp_convolution(1, se_filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_swish(0),
		ccv_cnnp_convolution(1, expand_filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_sigmoid(0)
	), 0);
	const ccv_cnnp_model_io_t se = ccv_cnnp_model_apply(se_conv, MODEL_IO_LIST(x));
	x = ccv_cnnp_model_apply(ccv_cnnp_mul(0), MODEL_IO_LIST(x, se));
	ccv_cnnp_model_t* const proj_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, output_filters, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0)
	), 0);
	x = ccv_cnnp_model_apply(proj_conv, MODEL_IO_LIST(x));
	if (input_filters == output_filters && strides == 1)
		x = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(x, input));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(x), 0);
}

static ccv_cnnp_model_t* _mconv_block_layer_new(const int num_repeats, const int kernel_size, const int strides, const int expand_ratio, const int input_filters, const int output_filters, const float se_ratio, const float dropout)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	ccv_cnnp_model_t* const first_block = _mconv_block_new(kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, dropout);
	ccv_cnnp_model_io_t x = ccv_cnnp_model_apply(first_block, MODEL_IO_LIST(input));
	if (num_repeats > 1)
	{
		int i;
		for (i = 1; i < num_repeats; i++)
			x = ccv_cnnp_model_apply(_mconv_block_new(kernel_size, 1, expand_ratio, output_filters, output_filters, se_ratio, dropout), MODEL_IO_LIST(x));
	}
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(x), 0);
}

ccv_cnnp_model_t* _efficientnet_b0(void)
{
	const ccv_cnnp_model_io_t input = ccv_cnnp_input();
	const float dropout = 0.2;
	ccv_cnnp_model_t* const init_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 32, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((2, 2), (1, 1)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_swish(0)
	), 0);
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(init_conv, MODEL_IO_LIST(input));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(1, 3, 1, 1, 32, 16, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(2, 3, 2, 6, 16, 24, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(2, 5, 2, 6, 24, 40, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(3, 3, 2, 6, 40, 80, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(3, 5, 1, 6, 80, 112, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(4, 5, 2, 6, 112, 192, 0.25, dropout), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(_mconv_block_layer_new(1, 3, 1, 6, 192, 320, 0.25, dropout), MODEL_IO_LIST(output));
	ccv_cnnp_model_t* const head_conv = ccv_cnnp_sequential_new(MODEL_LIST(
		ccv_cnnp_convolution(1, 1280, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (0, 0)),
		}, 0),
		ccv_cnnp_batch_norm(0.9, 1e-4, 0),
		ccv_cnnp_swish(0)
	), 0);
	output = ccv_cnnp_model_apply(head_conv, MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(0, 0), (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_flatten(0), MODEL_IO_LIST(output));
	if (dropout > 0)
		output = ccv_cnnp_model_apply(ccv_cnnp_dropout(dropout, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_dense(1000, (ccv_cnnp_param_t){}, 0), MODEL_IO_LIST(output));
	output = ccv_cnnp_model_apply(ccv_cnnp_softmax(0), MODEL_IO_LIST(output));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), MODEL_IO_LIST(output), 0);
}

static double _accuracy_from_gpu(ccv_nnc_tensor_t* const* const gpu_outputs, ccv_nnc_tensor_t* const* const cpu_outputs, ccv_nnc_tensor_t* const* const fits, ccv_nnc_tensor_t* const fit_fp32, ccv_nnc_tensor_t* const output_fp32, const int batch_size, const int classes, const int device_count)
{
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, gpu_outputs, device_count, cpu_outputs, device_count, 0);
	int correct = 0;
	int i, j, k;
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_t* fit = fits[i * 2 + 1];
		ccv_nnc_tensor_t* output = cpu_outputs[i];
		if (fit->info.datatype != CCV_32F || output->info.datatype != CCV_32F)
		{
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(fit, output), TENSOR_LIST(fit_fp32, output_fp32), 0);
			fit = fit_fp32;
			output = output_fp32;
		}
		for (j = 0; j < batch_size; j++)
		{
			float max = -FLT_MAX;
			int max_idx = -1;
			for (k = 0; k < classes; k++)
			{
				assert(!isnan(output->data.f32[j * classes + k]));
				if (output->data.f32[j * classes + k] > max)
				{
					max = output->data.f32[j * classes + k];
					max_idx = k;
				}
			}
			assert(max_idx >= 0);
			int right = -1;
			for (k = 0; k < classes; k++)
				if (fit->data.f32[j * classes + k] > 0.5)
				{
					right = k;
					break;
				}
			if (right == max_idx)
				++correct;
		}
	}
	return (double)correct / (device_count * batch_size);
}

float _resnet_learn_rate(const int epoch, const int t, const int epoch_end)
{
	const int warmup_epoch = 5;
	float learn_rate = 0.0001;
	if (epoch < warmup_epoch)
	{
		learn_rate = ccv_max(0.0001, t / (epoch_end * 5));
	} else if (epoch < 40) {
		learn_rate = 1 - (1 - 0.5) * (epoch - 5) / 35;
	} else if (epoch < 60) {
		learn_rate = 0.5 - (0.5 - 0.25) * (epoch - 40) / 20;
	} else if (epoch < 100) {
		learn_rate = 0.25 - (0.25 - 0.025) * (epoch - 60) / 40;
	} else {
		learn_rate = 0.025 - (0.025 - 0.00025) * (epoch - 100) / 20;
	}
	learn_rate = ccv_max(learn_rate, 0.00004);
	return 0.4 * learn_rate;
}

ccv_nnc_cmd_t _resnet_optimizer(const float learn_rate, const int batch_size, const float wd)
{
	return CMD_SGD_FORWARD(1, learn_rate, 1. / batch_size, wd, 0.9, 0);
}

float _efficientnet_learn_rate(const int epoch, const int t, const int epoch_end)
{
	const float scaled_lr = 0.016;
	const int warmup_epoch = 5;
	if (epoch < warmup_epoch)
		return ccv_max(0.00001, scaled_lr * t / (epoch_end * 5));
	else
		return scaled_lr * powf(0.97, (t - warmup_epoch * epoch_end) / (epoch_end * 2.4));
}

ccv_nnc_cmd_t _efficientnet_optimizer(const float learn_rate, const int batch_size, const float wd)
{
	return CMD_RMSPROP_FORWARD(learn_rate * batch_size / 256., wd, 0.9, 0.9, sqrt(1e-3));
}

#define CCV_TRAIN_DT CCV_32F
#define _net_learn_rate _efficientnet_learn_rate
#define _net_optimizer _efficientnet_optimizer
#define _net_model _efficientnet_b0

static void train_imagenet(const int batch_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const test_data, ccv_array_t* const test_set)
{
	// Prepare model.
	ccv_cnnp_model_t* const imagenet = _net_model();
	ccv_nnc_tensor_param_t input = GPU_TENSOR_NCHW(000, TRAIN_DT, batch_size, 3, 224, 224);
	const int device_count = ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU);
	const float wd = 0.00001; // 0.0001;
	ccv_cnnp_model_compile(imagenet, &input, 1, _net_optimizer(0.0001, batch_size * device_count, wd), CMD_CATEGORICAL_CROSSENTROPY_FORWARD());
	FILE *w = fopen("imagenet.dot", "w+");
	ccv_cnnp_model_dot(imagenet, CCV_NNC_LONG_DOT_GRAPH, &w, 1);
	fclose(w);
	ccv_cnnp_model_set_workspace_size(imagenet, 1llu * 1024 * 1024 * 1024);
	// ccv_cnnp_model_set_memory_compression(imagenet, 1);
	// Prepare training data.
	const int read_image_idx = ccv_cnnp_dataframe_read_image(train_data, 0, offsetof(ccv_categorized_t, file) + offsetof(ccv_file_info_t, filename));
	ccv_cnnp_random_jitter_t random_jitter = {
		.brightness = 0.4,
		.contrast = 0.4,
		.saturation = 0.4,
		.lighting = 0.1,
		.symmetric = 1,
		.resize = {
			.min = 180,
			.max = 480,
		},
		.normalize = {
			.mean = {
				123.68, 116.779, 103.939
			},
			.std = {
				58.393, 57.12, 57.375
			},
		},
		.aspect_ratio = 0.5,
		.size = {
			.cols = 224,
			.rows = 224,
		},
	};
	const int image_jitter_idx = ccv_cnnp_dataframe_image_random_jitter(train_data, read_image_idx, CCV_32F, random_jitter);
	ccv_nnc_tensor_param_t fp16_params = CPU_TENSOR_NHWC(TRAIN_DT, 224, 224, 3);
	const int image_jitter_in_idx = ccv_cnnp_dataframe_make_tuple(train_data, COLUMN_ID_LIST(image_jitter_idx));
	const int image_jitter_out_fp16_idx = ccv_cnnp_dataframe_cmd_exec(train_data, image_jitter_in_idx, CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 0, 1, &fp16_params, 1, 0);
	const int image_jitter_fp16_idx = ccv_cnnp_dataframe_extract_tuple(train_data, image_jitter_out_fp16_idx, 0);
	const float eta = 0.1;
	const int one_hot_idx = ccv_cnnp_dataframe_one_hot(train_data, 0, offsetof(ccv_categorized_t, c), 1000, 1 - eta + eta / 1000, eta / 1000, CCV_TRAIN_DT, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_shuffle(train_data);
	ccv_cnnp_dataframe_t* const batch_train_data = ccv_cnnp_dataframe_batching_new(train_data, COLUMN_ID_LIST(image_jitter_fp16_idx, one_hot_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int t, i, j;
	int train_device_columns[device_count * 2 + 1];
	for (i = 0; i < device_count; i++)
	{
		train_device_columns[i] = ccv_cnnp_dataframe_copy_to_gpu(batch_train_data, 0, i * 2, 2, i);
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, TRAIN_DT, batch_size, 1000);
		CCV_TENSOR_SET_DEVICE_ID(params.type, i);
		train_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux(batch_train_data, params);
	}
	train_device_columns[device_count * 2] = 0;
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch_train_data, train_device_columns, device_count * 2 + 1);
	// Prepare test data.
	const int read_test_image_idx = ccv_cnnp_dataframe_read_image(test_data, 0, offsetof(ccv_categorized_t, file) + offsetof(ccv_file_info_t, filename));
	ccv_cnnp_random_jitter_t no_jitter = {
		.resize = {
			.min = 256,
			.max = 256,
		},
		.center_crop = 1,
		.normalize = {
			.mean = {
				123.68, 116.779, 103.939
			},
			.std = {
				58.393, 57.12, 57.375
			},
		},
		.size = {
			.cols = 224,
			.rows = 224,
		},
	};
	const int test_image_idx = ccv_cnnp_dataframe_image_random_jitter(test_data, read_test_image_idx, CCV_32F, no_jitter);
	const int test_image_in_idx = ccv_cnnp_dataframe_make_tuple(test_data, COLUMN_ID_LIST(test_image_idx));
	const int test_image_out_fp16_idx = ccv_cnnp_dataframe_cmd_exec(test_data, test_image_in_idx, CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, 0, 1, &fp16_params, 1, 0);
	const int test_image_fp16_idx = ccv_cnnp_dataframe_extract_tuple(test_data, test_image_out_fp16_idx, 0);
	ccv_cnnp_dataframe_t* const batch_test_data = ccv_cnnp_dataframe_batching_new(test_data, COLUMN_ID_LIST(test_image_fp16_idx), batch_size, device_count, CCV_TENSOR_FORMAT_NCHW);
	int test_device_columns[device_count * 2];
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_param_t params = GPU_TENSOR_NCHW(000, TRAIN_DT, batch_size, 1000);
		CCV_TENSOR_SET_DEVICE_ID(params.type, i);
		test_device_columns[i] = ccv_cnnp_dataframe_copy_to_gpu(batch_test_data, 0, i, 1, i);
		test_device_columns[device_count + i] = ccv_cnnp_dataframe_add_aux(batch_test_data, params);
	}
	ccv_cnnp_dataframe_iter_t* const test_iter = ccv_cnnp_dataframe_iter_new(batch_test_data, test_device_columns, device_count * 2);

	// Prepare training context.
	ccv_nnc_stream_context_t* stream_contexts[2];
	stream_contexts[0] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	stream_contexts[1] = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	int p = 0, q = 1;
	const int epoch_end = (ccv_cnnp_dataframe_row_count(train_data) + batch_size * device_count - 1) / (batch_size * device_count);
	ccv_cnnp_model_set_data_parallel(imagenet, device_count);
	ccv_cnnp_model_checkpoint(imagenet, "imagenet.checkpoint", 0);
	unsigned int current_time = get_current_time();
	unsigned int batch_start_time = current_time;
	ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[p]);
	ccv_nnc_tensor_t* cpu_outputs[device_count];
	ccv_nnc_tensor_t* outputs_fp32[device_count];
	for (i = 0; i < device_count; i++)
	{
		cpu_outputs[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(TRAIN_DT, batch_size, 1000), 0);
		outputs_fp32[i] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1000), 0);
	}
	ccv_nnc_tensor_t* fit_fp32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, batch_size, 1000), 0);
	ccv_nnc_tensor_t** input_fits[device_count * 2 + 1];
	ccv_nnc_tensor_t* input_fit_inputs[device_count];
	ccv_nnc_tensor_t* input_fit_fits[device_count];
	ccv_nnc_tensor_t* outputs[device_count];
	int epoch = 0;
	double overall_accuracy = 0;
	// Start 100 epoch of training.
	for (t = epoch * epoch_end; epoch < 350; t++)
	{
		const float learn_rate = _net_learn_rate(epoch, t, epoch_end);
		ccv_cnnp_model_set_minimizer(imagenet, _net_optimizer(learn_rate, batch_size * device_count, wd), 0, 0);
		ccv_cnnp_model_set_minimizer(imagenet, _net_optimizer(learn_rate, batch_size * device_count, 0), MODEL_IO_LIST(ccv_cnnp_model_parameters(imagenet, CCV_CNNP_PARAMETER_SELECT_BIAS, ALL_PARAMETERS)));
		ccv_cnnp_dataframe_iter_next(iter, (void**)input_fits, device_count * 2 + 1, stream_contexts[p]);
		ccv_nnc_stream_context_wait(stream_contexts[q]); // Need to wait the other context to finish, we use the same tensor_arena.
		// Re-layout data for model fitting.
		for (i = 0; i < device_count; i++)
		{
			input_fit_inputs[i] = input_fits[i][0];
			input_fit_fits[i] = input_fits[i][1];
			outputs[i] = (ccv_nnc_tensor_t*)input_fits[device_count + i];
		}
		ccv_cnnp_model_fit(imagenet, input_fit_inputs, device_count, input_fit_fits, device_count, outputs, device_count, 0, stream_contexts[p]);
		// Prefetch the next round.
		ccv_cnnp_dataframe_iter_prefetch(iter, 1, stream_contexts[q]);
		if (t % 50 == 49)
		{
			// Only sample the accuracy. Although truth to be told, this doesn't impact performance because moving small amount of data on / off GPU is relatively easy.
			ccv_nnc_stream_context_wait(stream_contexts[p]);
			double accuracy = _accuracy_from_gpu(outputs, cpu_outputs, input_fits[device_count * 2], fit_fp32, outputs_fp32[0], batch_size, 1000, device_count);
			overall_accuracy = overall_accuracy * 0.5 + accuracy * 0.5;
			unsigned int elapsed_time = get_current_time() - batch_start_time;
			printf("Epoch %d (%d) %.3lf GiB (%.3f samples per sec), accuracy = %lf%%, lr = %f\n", epoch, t, (unsigned long)ccv_cnnp_model_memory_size(imagenet) / 1024 / 1024.0 / 1024, 50 * device_count * batch_size / ((float)elapsed_time / 1000), overall_accuracy * 100, learn_rate);
			batch_start_time = get_current_time();
		}
		if ((t + 1) % epoch_end == 0)
		{
			ccv_nnc_stream_context_wait(stream_contexts[p]);
			ccv_nnc_stream_context_wait(stream_contexts[q]);
			char checkpoint[256];
			snprintf(checkpoint, 256, "imagenet.checkpoint.%d", epoch);
			ccv_cnnp_model_checkpoint(imagenet, checkpoint, 0);
			int correct = 0;
			p = 0, q = 1;
			for (i = 0; i < ccv_cnnp_dataframe_row_count(test_data); i += batch_size * device_count)
			{
				ccv_cnnp_dataframe_iter_next(test_iter, (void**)input_fits, device_count * 2, 0);
				for (j = 0; j < device_count; j++)
				{
					input_fit_inputs[j] = input_fits[j][0];
					outputs[j] = (ccv_nnc_tensor_t*)input_fits[device_count + j];
				}
				ccv_cnnp_model_evaluate(imagenet, (ccv_cnnp_evaluate_param_t){
					.is_test = 1
				}, input_fit_inputs, device_count, outputs, device_count, 0, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs, device_count, cpu_outputs, device_count, 0);
				ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, cpu_outputs, device_count, outputs_fp32, device_count, 0);
				for (j = 0; j < ccv_min(ccv_cnnp_dataframe_row_count(test_data) - i, batch_size * device_count); j++)
				{
					ccv_categorized_t* const categorized = (ccv_categorized_t*)ccv_array_get(test_set, i + j);
					const int d = j / batch_size;
					const int b = j % batch_size;
					float max = -FLT_MAX;
					int c = -1, k;
					for (k = 0; k < 1000; k++)
						if (outputs_fp32[d]->data.f32[b * 1000 + k] > max)
							max = outputs_fp32[d]->data.f32[b * 1000 + k], c = k;
					if (categorized->c == c)
						++correct;
				}
			}
			unsigned int elapsed_time = get_current_time() - current_time;
			current_time = get_current_time();
			printf("Epoch %d (%d), test accuracy %lf%%, time %.3f\n", epoch, t, (double)correct * 100 / ccv_cnnp_dataframe_row_count(test_data), (float)elapsed_time / 1000);
			ccv_cnnp_dataframe_iter_set_cursor(test_iter, 0);
			++epoch;
			ccv_cnnp_dataframe_shuffle(train_data);
			ccv_cnnp_dataframe_iter_set_cursor(iter, 0);
		}
		int n;
		CCV_SWAP(p, q, n);
	}
	ccv_cnnp_dataframe_iter_free(test_iter);
	ccv_cnnp_dataframe_free(batch_test_data);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batch_train_data);
	for (i = 0; i < device_count; i++)
	{
		ccv_nnc_tensor_free(cpu_outputs[i]);
		ccv_nnc_tensor_free(outputs_fp32[i]);
	}
	ccv_nnc_tensor_free(fit_fp32);
	ccv_cnnp_model_free(imagenet);
}

static ccv_array_t* _array_from_disk_new(const char* const list, const char* const base_dir)
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
	return categorizeds;
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
	ccv_array_t* const train_set = _array_from_disk_new(train_list, base_dir);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	ccv_array_t* const test_set = _array_from_disk_new(test_list, base_dir);
	ccv_cnnp_dataframe_t* const test_data = ccv_cnnp_dataframe_from_array_new(test_set);
	train_imagenet(64, train_data, test_data, test_set);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	int i;
	for (i = 0; i < train_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(train_set, i))->file.filename);
	ccv_array_free(train_set);
	for (i = 0; i < test_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(test_set, i))->file.filename);
	ccv_array_free(test_set);
	return 0;
}
