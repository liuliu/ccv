#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>
#include <stddef.h>

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

static void _fpn(const int d, const ccv_cnnp_model_io_t* const c, const int c_size, ccv_cnnp_model_io_t* const p)
{
	int i;
	ccv_cnnp_model_io_t output = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.hint = HINT((1, 1), (0, 0)),
	}, 0), MODEL_IO_LIST(c[c_size - 1]));
	p[c_size - 1] = output;
	for (i = c_size - 2; i >= 0; i--)
	{
		const ccv_cnnp_model_io_t lateral = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
			.hint = HINT((1, 1), (0, 0)),
		}, 0), MODEL_IO_LIST(c[i]));
		const ccv_cnnp_model_io_t up = ccv_cnnp_model_apply(ccv_cnnp_upsample(2, 2, 0), MODEL_IO_LIST(output));
		const ccv_cnnp_model_io_t sum = ccv_cnnp_model_apply(ccv_cnnp_add(0), MODEL_IO_LIST(lateral, up));
		output = ccv_cnnp_model_apply(ccv_cnnp_convolution(1, d, DIM_ALLOC(3, 3), (ccv_cnnp_param_t){
			.no_bias = 1,
			.hint = HINT((1, 1), (1, 1)),
		}, 0), MODEL_IO_LIST(sum));
		p[i] = output;
	}
}

ccv_cnnp_model_t* _imagenet_resnet50_v1d_fpn(void)
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
	const ccv_cnnp_model_io_t c2 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(128, 4, 2, 4), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c3 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(256, 4, 2, 6), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c4 = output;
	output = ccv_cnnp_model_apply(_resnet_block_layer_new(512, 4, 2, 3), MODEL_IO_LIST(output));
	const ccv_cnnp_model_io_t c5 = output;
	const ccv_cnnp_model_io_t c[] = { c2, c3, c4, c5 };
	ccv_cnnp_model_io_t p[5];
	_fpn(256, c, 4, p);
	p[4] = ccv_cnnp_model_apply(ccv_cnnp_average_pool(DIM_ALLOC(2, 2), (ccv_cnnp_param_t){
		.hint = HINT((2, 2), (0, 0)),
	}, 0), MODEL_IO_LIST(p[3]));
	// 3 aspect ratios (1:2, 1:1, 2:1). Each has 4 + 2 (x, y, w, h, object, non-object), total 18.
	ccv_cnnp_model_t* const rpn_proposals = ccv_cnnp_convolution(1, 18, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
		.hint = HINT((1, 1), (0, 0)),
	}, "rpn");
	ccv_cnnp_model_io_t proposals[5];
	int i;
	for (i = 0; i < 5; i++)
		proposals[i] = ccv_cnnp_model_apply(rpn_proposals, MODEL_IO_LIST(p[i]));
	return ccv_cnnp_model_new(MODEL_IO_LIST(input), proposals, 5, 0);
}

typedef struct {
	int c;
	ccv_decimal_rect_t rect;
} ccv_nnc_bbox_t;

typedef struct {
	const char* filename;
	ccv_array_t* bboxes;
} ccv_nnc_annotation_t;

static void _rpn_data_batching(void* const* const input_data, const int input_size, void** const output_data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
}

static void _rpn_data_deinit(void* const self, void* const context)
{
}

static void train_coco(const int batch_size, ccv_cnnp_dataframe_t* const train_data, ccv_cnnp_dataframe_t* const val_data)
{
	ccv_cnnp_model_t* rpn = _imagenet_resnet50_v1d_fpn();
	ccv_cnnp_model_set_workspace_size(rpn, 1llu * 1024 * 1024 * 1024);
	const int read_image_idx = ccv_cnnp_dataframe_read_image(train_data, 0, offsetof(ccv_nnc_annotation_t, filename));
	ccv_cnnp_random_jitter_t random_jitter = {
		.brightness = 0.4,
		.contrast = 0.4,
		.saturation = 0.4,
		.lighting = 0.1,
		.symmetric = 0, // If it is flipped, I cannot distinguish.
		.resize = {
			.min = 600,
			.max = 800,
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
		.size = {}, // 0 means no cropping at this point.
	};
	const int image_jitter_idx = ccv_cnnp_dataframe_image_random_jitter(train_data, read_image_idx, CCV_32F, random_jitter);
	const int tuple_idx = ccv_cnnp_dataframe_make_tuple(train_data, COLUMN_ID_LIST(0, image_jitter_idx));
	ccv_cnnp_dataframe_t* const batch_data = ccv_cnnp_dataframe_reduce_new(train_data, _rpn_data_batching, _rpn_data_deinit, tuple_idx, 4, rpn, 0);
	ccv_cnnp_dataframe_free(batch_data);
	/*
	ccv_cnnp_model_compile(rpn, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_param_t output_params[5];
	ccv_cnnp_model_tensor_auto(rpn, output_params, 5);
	*/
}

static ccv_array_t* _array_from_disk_new(const char* const list, const char* const base_dir)
{
	FILE *r = fopen(list, "r");
	assert(r && "list doesn't exists");
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* categorizeds = ccv_array_new(sizeof(ccv_nnc_annotation_t), 64, 0);
	int c;
	ccv_nnc_annotation_t* last_annotation = 0;
	char* file = (char*)malloc(1024);
	float x, y, width, height;
	while (fscanf(r, "%d %s %f %f %f %f", &c, file, &x, &y, &width, &height) != EOF)
	{
		char* filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
		}
		strncpy(filename + dirlen, file, 1024 - dirlen);
		// Coco's category class starts from 1, thus, minus 1 to get 0-index
		ccv_nnc_bbox_t bbox = {
			.c = c - 1,
			.rect = ccv_decimal_rect(x, y, width, height),
		};
		if (!last_annotation->filename || memcmp(last_annotation->filename, filename, strnlen(filename, 1024)) != 0)
		{
			ccv_nnc_annotation_t annotation = {
				.filename = filename,
				.bboxes = ccv_array_new(sizeof(ccv_nnc_bbox_t), 1, 0),
			};
			ccv_array_push(annotation.bboxes, &bbox);
			ccv_array_push(categorizeds, &annotation);
			last_annotation = (ccv_nnc_annotation_t*)ccv_array_get(categorizeds, categorizeds->rnum - 1);
		} else {
			ccfree(filename);
			ccv_array_push(last_annotation->bboxes, &bbox);
		}
	}
	free(file);
	fclose(r);
	return categorizeds;
}

int main(int argc, char** argv)
{
	ccv_nnc_init();
	static struct option coco_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"train-list", 1, 0, 0},
		{"val-list", 1, 0, 0},
		/* optional parameters */
		{"train-dir", 1, 0, 0},
		{"val-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	int c;
	char* train_list = 0;
	char* val_list = 0;
	char* train_dir = 0;
	char* val_dir = 0;
	while (getopt_long_only(argc, argv, "", coco_options, &c) != -1)
	{
		switch (c)
		{
			case 0:
				exit(0);
			case 1:
				train_list = optarg;
				break;
			case 2:
				val_list = optarg;
				break;
			case 3:
				train_dir = optarg;
				break;
			case 4:
				val_dir = optarg;
				break;
		}
	}
	ccv_array_t* const train_set = _array_from_disk_new(train_list, train_dir);
	ccv_cnnp_dataframe_t* const train_data = ccv_cnnp_dataframe_from_array_new(train_set);
	ccv_array_t* const val_set = _array_from_disk_new(val_list, val_dir);
	ccv_cnnp_dataframe_t* const test_data = ccv_cnnp_dataframe_from_array_new(val_set);
	train_coco(128, train_data, test_data);
	ccv_cnnp_dataframe_free(train_data);
	ccv_cnnp_dataframe_free(test_data);
	int i;
	for (i = 0; i < train_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(train_set, i))->file.filename);
	ccv_array_free(train_set);
	for (i = 0; i < val_set->rnum; i++)
		ccfree(((ccv_categorized_t*)ccv_array_get(val_set, i))->file.filename);
	ccv_array_free(val_set);
	return 0;
}
