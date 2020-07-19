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
		shortcut = input;
		if (strides > 1)
		{
			ccv_cnnp_model_t* const avgdown = ccv_cnnp_average_pool(DIM_ALLOC(strides, strides), (ccv_cnnp_param_t){
				.hint = HINT((strides, strides), (0, 0))
			}, 0);
			shortcut = ccv_cnnp_model_apply(avgdown, MODEL_IO_LIST(input));
		}
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
	// 3 aspect ratios (1:2, 1:1, 2:1). Each has 4 + 1 (x, y, w, h, objectness), total 15.
	ccv_cnnp_model_t* const rpn_proposals = ccv_cnnp_convolution(1, 15, DIM_ALLOC(1, 1), (ccv_cnnp_param_t){
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

typedef struct {
	int batch_count;
	int select_count;
	ccv_cnnp_model_t* rpn;
} ccv_nnc_rpn_data_batching_t;

typedef struct {
	struct {
		float iou;
		float* val;
		int x;
		int y;
		int anchor_width;
		int anchor_height;
	} gt;
	ccv_decimal_rect_t rect;
} ccv_nnc_rpn_rect_t;

static void _rpn_gt(const int width, const int height, const int scale, const int offset_x, const int offset_y, const int anchor_width, const int anchor_height, ccv_nnc_rpn_rect_t* const rects, const int rect_size, float* const cp, const int cp_step)
{
	float* cp0 = cp;
	int k, x, y;
	for (y = 0; y < height; y++)
	{
		const int ry = y * scale - offset_y;
		for (x = 0; x < width; x++)
		{
			const int rx = x * scale - offset_x;
			float best_iou = 0;
			int bbox_idx = -1;
			for (k = 0; k < rect_size; k++)
			{
				const float iarea = ccv_max(0, ccv_min(rx + anchor_width, rects[k].rect.x + rects[k].rect.width) - ccv_max(rx, rects[k].rect.x)) * ccv_max(0, ccv_min(ry + anchor_height, rects[k].rect.y + rects[k].rect.height) - ccv_max(ry, rects[k].rect.y));
				const float iou = iarea / (rects[k].rect.width * rects[k].rect.height + anchor_width * anchor_height - iarea);
				if (iou > best_iou)
				{
					bbox_idx = k;
					best_iou = iou;
				}
				if (iou > rects[k].gt.iou)
				{
					rects[k].gt.iou = iou;
					rects[k].gt.val = cp0;
					rects[k].gt.x = x * scale;
					rects[k].gt.y = y * scale;
					rects[k].gt.anchor_width = anchor_width;
					rects[k].gt.anchor_height = anchor_height;
				}
			}
			if (best_iou >= 0.7)
			{
				cp0[0] = 1;
				cp0[1] = (rects[bbox_idx].rect.x + rects[bbox_idx].rect.width * 0.5 - x * scale) / anchor_width;
				cp0[2] = (rects[bbox_idx].rect.y + rects[bbox_idx].rect.height * 0.5 - y * scale) / anchor_height;
				cp0[3] = log(rects[bbox_idx].rect.width / anchor_width);
				cp0[4] = log(rects[bbox_idx].rect.height / anchor_height);
			} else if (best_iou <= 0.3) {
				cp0[0] = 0;
				cp0[1] = cp0[2] = cp0[3] = cp0[4] = 0;
			} else {
				cp0[0] = -1; // Ignore.
				cp0[1] = cp0[2] = cp0[3] = cp0[4] = 0;
			}
			cp0 += cp_step;
		}
	}
}

static void _rpn_rect_missing_gt(ccv_nnc_rpn_rect_t* const rects, const int rect_size)
{
	int i;
	for (i = 0; i < rect_size; i++)
		if (rects[i].gt.val && rects[i].gt.val[0] != 1) // The best matching one hasn't assigned yet.
		{
			float* const cp = rects[i].gt.val;
			cp[0] = 1;
			cp[1] = (rects[i].rect.x + rects[i].rect.width * 0.5 - rects[i].gt.x) / rects[i].gt.anchor_width;
			cp[2] = (rects[i].rect.y + rects[i].rect.height * 0.5 - rects[i].gt.y) / rects[i].gt.anchor_height;
			cp[3] = log(rects[i].rect.width / rects[i].gt.anchor_width);
			cp[4] = log(rects[i].rect.height / rects[i].gt.anchor_height);
		}
}

// Batching to NCHW format.
static void _rpn_data_batching(void* const* const input_data, const int input_size, void** const output_data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	int max_rows = 0;
	int max_cols = 0;
	int max_bbox_size = 0;
	for (i = 0; i < input_size; i++)
	{
		void* const* const tuple = input_data[i];
		ccv_dense_matrix_t* const jitter_image = tuple[2];
		max_rows = ccv_max(max_rows, jitter_image->rows);
		max_cols = ccv_max(max_cols, jitter_image->cols);
		ccv_nnc_annotation_t* const annotation = tuple[0];
		ccv_array_t* const bboxes = annotation->bboxes;
		max_bbox_size = ccv_max(max_bbox_size, bboxes->rnum);
	}
	ccv_nnc_rpn_data_batching_t* const rpn_data = (ccv_nnc_rpn_data_batching_t*)context;
	const ccv_nnc_tensor_param_t input_params = CPU_TENSOR_NCHW(32F, rpn_data->batch_count, 3, max_rows, max_cols);
	ccv_cnnp_model_compile(rpn_data->rpn, TENSOR_PARAM_LIST(input_params), CMD_NOOP(), CMD_NOOP());
	static const int fpn_size = 5;
	ccv_nnc_tensor_param_t gt_params[fpn_size];
	ccv_cnnp_model_tensor_auto(rpn_data->rpn, gt_params, fpn_size);
	int total_proposals = 0;
	for (i = 0; i < fpn_size; i++)
		total_proposals += gt_params[i].dim[2] * gt_params[i].dim[3];
	// 3 means: 1:1, 1:2, 2:1 aspect ratios.
	ccv_nnc_tensor_t* input;
	ccv_nnc_tensor_t* gt;
	ccv_nnc_tensor_t* select;
	if (!output_data[0])
	{
		ccv_nnc_tensor_t** tensors = output_data[0] = ccmalloc(sizeof(ccv_nnc_tensor_t*) * 3);
		input = tensors[0] = ccv_nnc_tensor_new(0, input_params, 0);
		gt = tensors[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, rpn_data->batch_count * total_proposals * 3, 5), 0);
		select = tensors[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32S, rpn_data->batch_count * rpn_data->select_count), 0);
	} else {
		ccv_nnc_tensor_t** tensors = output_data[0];
		input = tensors[0] = ccv_nnc_tensor_resize(tensors[0], input_params);
		gt = tensors[1] = ccv_nnc_tensor_resize(tensors[1], CPU_TENSOR_NCHW(32F, rpn_data->batch_count * total_proposals * 3, 5));
		select = tensors[2] = ccv_nnc_tensor_resize(tensors[2], CPU_TENSOR_NCHW(32S, rpn_data->batch_count * rpn_data->select_count));
	}
	int j, x, y;
	ccv_nnc_rpn_rect_t* const rects = (ccv_nnc_rpn_rect_t*)ccmalloc(sizeof(ccv_nnc_rpn_rect_t) * max_bbox_size);
	for (i = 0; i < rpn_data->batch_count; i++)
	{
		void* const* const tuple = input_data[i % input_size];
		ccv_dense_matrix_t* const image = tuple[1];
		ccv_dense_matrix_t* const jitter_image = tuple[2];
		assert(jitter_image->cols <= max_cols);
		assert(jitter_image->rows <= max_rows);
		const int pad_x = (max_cols - jitter_image->cols + 1) / 2;
		const int pad_y = (max_rows - jitter_image->rows + 1) / 2;
		const int plane_size = max_cols * max_rows;
		float* bp = input->data.f32 + i * plane_size * 3;
		memset(bp, 0, sizeof(float) * plane_size * 3);
		bp += max_cols * pad_y + pad_x;
		const float* ap = jitter_image->data.f32;
		for (y = 0; y < jitter_image->rows; y++)
		{
			for (x = 0; x < jitter_image->cols; x++)
			{
				bp[x] = ap[x * 3];
				bp[x + plane_size] = ap[x * 3 + 1];
				bp[x + plane_size * 2] = ap[x * 3 + 2];
			}
			ap += jitter_image->cols * 3;
			bp += max_cols;
		}
		ccv_nnc_annotation_t* const annotation = tuple[0];
		ccv_array_t* const bboxes = annotation->bboxes;
		const float width_scale = (float)jitter_image->cols / image->cols;
		const float height_scale = (float)jitter_image->rows / image->rows;
		// Generate ground truth.
		float* cp = gt->data.f32 + i * 3 * 5 * total_proposals;
		memset(rects, 0, sizeof(ccv_nnc_rpn_rect_t) * bboxes->rnum);
		for (j = 0; j < bboxes->rnum; j++)
		{
			const ccv_nnc_bbox_t* const bbox = (ccv_nnc_bbox_t*)ccv_array_get(bboxes, j);
			rects[j].rect = ccv_decimal_rect(bbox->rect.x * width_scale + pad_x, bbox->rect.y * height_scale + pad_y, bbox->rect.width * width_scale, bbox->rect.height * height_scale);
		}
		// Input scaled down twice to get to the first layer. Since we always padding beginning
		// for convolution, we don't need to have extra shift for x-y axis. Simply scale up is
		// sufficient.
		// Because the size is 32x32, 64x64, 128x128, 256x256, 512x512, it is all even number.
		// The anchor is in the middle, assuming 0 index, it is (15, 15), (31, 31) etc.
		int scale = 4;
		int box_size = 8;
		for (j = 0; j < fpn_size; j++)
		{
			// 1:1
			const int anchor_size = box_size * scale;
			const int offset = (anchor_size - 1) / 2;
			_rpn_gt(gt_params[j].dim[3], gt_params[j].dim[2], scale, offset, offset, anchor_size, anchor_size, rects, bboxes->rnum, cp, 3 * 5);
			// 1:2
			const int anchor_size_1 = (float)(sqrt(box_size * scale * box_size * scale / 2.0) + 0.5);
			const int anchor_size_2 = anchor_size_1 * 2;
			const int offset_1 = (anchor_size_1 - 1) / 2;
			const int offset_2 = anchor_size_1 - 1;
			_rpn_gt(gt_params[j].dim[3], gt_params[j].dim[2], scale, offset_1, offset_2, anchor_size_1, anchor_size_2, rects, bboxes->rnum, cp + 5, 3 * 5);
			// 2:1
			_rpn_gt(gt_params[j].dim[3], gt_params[j].dim[2], scale, offset_2, offset_1, anchor_size_2, anchor_size_1, rects, bboxes->rnum, cp + 2 * 5, 3 * 5);
			scale *= 2;
			cp += gt_params[j].dim[2] * gt_params[j].dim[3] * 3 * 5;
		}
		_rpn_rect_missing_gt(rects, bboxes->rnum);
		cp = gt->data.f32 + i * 3 * 5 * total_proposals;
		const int half_select_count = rpn_data->select_count / 2;
		int k = 0;
		int* const sp = select->data.i32 + i * rpn_data->select_count;
		// First, select half of the positives.
		for (j = 0; k < half_select_count && j < total_proposals * 3; j++)
			if (cp[j * 5] == 1)
				sp[k++] = j + i * total_proposals * 3;
		// Fill the rest with negatives.
		for (j = 0; k < rpn_data->select_count && j < total_proposals * 3; j++)
			if (cp[j * 5] == 0)
				sp[k++] = j + i * total_proposals * 3;
		assert(k == rpn_data->select_count);
	}
	ccfree(rects);
}

static void _rpn_data_deinit(void* const self, void* const context)
{
	ccv_nnc_tensor_t** const data = (ccv_nnc_tensor_t**)self;
	ccv_nnc_tensor_free(data[0]);
	ccv_nnc_tensor_free(data[1]);
	ccv_nnc_tensor_free(data[2]);
	ccfree(data);
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
			.roundup = 64,
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
	const int tuple_idx = ccv_cnnp_dataframe_make_tuple(train_data, COLUMN_ID_LIST(0, read_image_idx, image_jitter_idx));
	ccv_nnc_rpn_data_batching_t rpn_data = {
		.batch_count = 1,
		.select_count = 256,
		.rpn = ccv_cnnp_model_copy(rpn)
	};
	ccv_cnnp_dataframe_t* const batch_data = ccv_cnnp_dataframe_reduce_new(train_data, _rpn_data_batching, _rpn_data_deinit, tuple_idx, 4, &rpn_data, 0);
	const int train_image_column = ccv_cnnp_dataframe_copy_to_gpu(batch_data, 0, 0, 1, 0);
	const int train_gt_column = ccv_cnnp_dataframe_copy_to_gpu(batch_data, 0, 1, 1, 0);
	const int train_select_column = ccv_cnnp_dataframe_copy_to_gpu(batch_data, 0, 2, 1, 0);
	ccv_nnc_dynamic_graph_t* const graph = ccv_nnc_dynamic_graph_new();
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch_data, COLUMN_ID_LIST(train_image_column, train_gt_column, train_select_column));
	ccv_nnc_tensor_t** data[3] = {};
	ccv_cnnp_dataframe_iter_next(iter, (void **)data, 3, 0);
	ccv_nnc_tensor_variable_t const input = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_t* const train_image = data[0][0];
	ccv_nnc_tensor_t* const train_gt = data[1][0];
	ccv_nnc_tensor_t* const train_select = data[2][0];
	ccv_nnc_tensor_variable_set(graph, input, train_image);
	ccv_nnc_tensor_variable_t outputs[5];
	int i;
	for (i = 0; i < 5; i++)
		outputs[i] = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_evaluate(graph, rpn, 0, TENSOR_VARIABLE_LIST(input), outputs, 5, 0, 0);
	ccv_nnc_tensor_variable_t remap_out = ccv_nnc_tensor_variable_new(graph, train_gt->info);
	int off = 0;
	CCV_CLI_SET_OUTPUT_LEVEL_AND_ABOVE(CCV_CLI_VERBOSE);
	for (i = 0; i < 5; i++)
	{
		ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_from_variable(graph, outputs[i]);
		ccv_nnc_tensor_variable_t remap_alias = ccv_nnc_tensor_variable_alias_new(graph, remap_out, DIM_ALLOC(0, 0, off, 0), DIM_ALLOC(1, tensor->info.dim[2], tensor->info.dim[3], tensor->info.dim[1]), GPU_TENSOR_NHWC(000, 32F, 1, tensor->info.dim[2], tensor->info.dim[3], tensor->info.dim[1]));
		off += tensor->info.dim[2] * tensor->info.dim[3];
		ccv_nnc_dynamic_graph_exec(graph, CMD_FORMAT_TRANSFORM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(outputs[i]), TENSOR_VARIABLE_LIST(remap_alias), 0, 0);
		ccv_nnc_tensor_variable_free(graph, remap_alias);
	}
	ccv_nnc_tensor_variable_t const select = ccv_nnc_tensor_constant_new(graph);
	ccv_nnc_tensor_variable_set(graph, select, train_select);
	ccv_nnc_tensor_variable_t const select_out = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(remap_out,  select), TENSOR_VARIABLE_LIST(select_out), 0, 0);
	ccv_nnc_tensor_variable_t const gt = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_set(graph, gt, train_gt);
	ccv_nnc_tensor_variable_t const select_gt = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(gt,  select), TENSOR_VARIABLE_LIST(select_gt), 0, 0);
	ccv_nnc_tensor_variable_t const cls_out = ccv_nnc_tensor_variable_alias_new(graph, select_out, DIM_ALLOC(), DIM_ALLOC(rpn_data.select_count, 5), GPU_TENSOR_NHWC(000, 32F, rpn_data.select_count, 1));
	ccv_nnc_tensor_variable_t const cls_gt = ccv_nnc_tensor_variable_alias_new(graph, select_gt, DIM_ALLOC(), DIM_ALLOC(rpn_data.select_count, 5), GPU_TENSOR_NHWC(000, 32F, rpn_data.select_count, 1));
	ccv_nnc_tensor_variable_t const cls_loss = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_tensor_variable_t const sigmoid = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SIGMOID_BINARY_CROSSENTROPY_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(cls_out, cls_gt), TENSOR_VARIABLE_LIST(cls_loss, sigmoid), 0, 0);
	ccv_nnc_tensor_variable_t const anchor_out = ccv_nnc_tensor_variable_alias_new(graph, select_out, DIM_ALLOC(0, 1), DIM_ALLOC(rpn_data.select_count, 5), GPU_TENSOR_NHWC(000, 32F, rpn_data.select_count, 4));
	ccv_nnc_tensor_variable_t const anchor_gt = ccv_nnc_tensor_variable_alias_new(graph, select_gt, DIM_ALLOC(0, 1), DIM_ALLOC(rpn_data.select_count, 5), GPU_TENSOR_NHWC(000, 32F, rpn_data.select_count, 4));
	ccv_nnc_tensor_variable_t const l1_loss = ccv_nnc_tensor_variable_new(graph);
	ccv_nnc_dynamic_graph_exec(graph, CMD_SMOOTH_L1_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_VARIABLE_LIST(anchor_out, anchor_gt), TENSOR_VARIABLE_LIST(l1_loss), 0, 0);
	ccv_nnc_tensor_variable_free(graph, l1_loss);
	ccv_nnc_tensor_variable_free(graph, anchor_gt);
	ccv_nnc_tensor_variable_free(graph, anchor_out);
	ccv_nnc_tensor_variable_free(graph, sigmoid);
	ccv_nnc_tensor_variable_free(graph, cls_loss);
	ccv_nnc_tensor_variable_free(graph, cls_gt);
	ccv_nnc_tensor_variable_free(graph, cls_out);
	ccv_nnc_tensor_variable_free(graph, select_gt);
	ccv_nnc_tensor_variable_free(graph, select_out);
	ccv_nnc_tensor_variable_free(graph, select);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batch_data);
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
		if (!last_annotation || memcmp(last_annotation->filename, filename, strnlen(filename, 1024)) != 0)
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
