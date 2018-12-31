#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_dataframe.h"

#include <3rdparty/sfmt/SFMT.h>

#pragma mark - Create Dataframe from Array

static void _ccv_cnnp_array_enum(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	int i;
	ccv_array_t* const array = (ccv_array_t*)context;
	for (i = 0; i < row_size; i++)
		data[i] = ccv_array_get(array, row_idxs[i]);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_from_array_new(ccv_array_t* const array)
{
	const ccv_cnnp_column_data_t array_column_data = {
		.data_enum = _ccv_cnnp_array_enum,
		.context = array
	};
	return ccv_cnnp_dataframe_new(&array_column_data, 1, array->rnum);
}

typedef struct {
	ccv_cnnp_dataframe_tuple_t tuple;
	int tensor_offset;
	int device_id;
} ccv_cnnp_copy_to_gpu_context_t;

#pragma mark - Copy Tensors from CPU to GPU

static void _ccv_cnnp_tensor_list_deinit(void* const data, void* const context)
{
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu = (ccv_cnnp_copy_to_gpu_context_t*)context;
	ccv_nnc_tensor_t** const tensor_list = (ccv_nnc_tensor_t**)data;
	int i;
	for (i = 0; i < copy_to_gpu->tuple.size; i++)
		if (tensor_list[i])
			ccv_nnc_tensor_free(tensor_list[i]);
	ccfree(tensor_list);
}

static void _ccv_cnnp_copy_to_gpu(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu_context = (ccv_cnnp_copy_to_gpu_context_t*)context;
	int i, j;
	for (i = 0; i < batch_size; i++)
	{
		ccv_nnc_tensor_t** inputs = (ccv_nnc_tensor_t**)column_data[0][i] + copy_to_gpu_context->tensor_offset;
		ccv_nnc_tensor_t** outputs = (ccv_nnc_tensor_t**)data[i];
		if (!outputs)
		{
			outputs = (ccv_nnc_tensor_t**)(data[i] = ccmalloc(sizeof(ccv_nnc_tensor_t*) * copy_to_gpu_context->tuple.size));
			for (j = 0; j < copy_to_gpu_context->tuple.size; j++)
			{
				ccv_nnc_tensor_param_t params = inputs[j]->info;
				params.type &= ~CCV_TENSOR_CPU_MEMORY;
				params.type |= CCV_TENSOR_GPU_MEMORY; // Change to GPU memory.
				CCV_TENSOR_SET_DEVICE_ID(params.type, copy_to_gpu_context->device_id);
				outputs[j] = ccv_nnc_tensor_new(0, params, 0);
			}
		}
		for (j = 0; j < copy_to_gpu_context->tuple.size; j++)
			ccv_nnc_tensor_pin_memory(inputs[j]);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, copy_to_gpu_context->tuple.size, outputs, copy_to_gpu_context->tuple.size, stream_context);
	}
}

int ccv_cnnp_dataframe_copy_to_gpu(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int tensor_offset, const int tensor_size, int device_id)
{
	assert(tensor_size > 0);
	int stream_type = CCV_STREAM_CONTEXT_GPU;
	CCV_STREAM_SET_DEVICE_ID(stream_type, device_id);
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu_context = (ccv_cnnp_copy_to_gpu_context_t*)ccmalloc(sizeof(ccv_cnnp_copy_to_gpu_context_t));
	copy_to_gpu_context->tuple.size = tensor_size;
	copy_to_gpu_context->tensor_offset = tensor_offset;
	copy_to_gpu_context->device_id = device_id;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_copy_to_gpu, stream_type, _ccv_cnnp_tensor_list_deinit, COLUMN_ID_LIST(column_idx), copy_to_gpu_context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

#pragma mark - Make Auxiliary Tensor as a new Column

static void _ccv_cnnp_tensor_deinit(void* const data, void* const context)
{
	ccv_nnc_tensor_free((ccv_nnc_tensor_t*)data);
}

static void _ccv_cnnp_tensor_new(const int column_idx, const int* const row_idxs, const int row_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_param_t params = *(ccv_nnc_tensor_param_t*)context;
	int i;
	for (i = 0; i < row_size; i++)
		if (!data[i])
			data[i] = ccv_nnc_tensor_new(0, params, 0);
}

int ccv_cnnp_dataframe_add_aux(ccv_cnnp_dataframe_t* const dataframe, const ccv_nnc_tensor_param_t params)
{
	int stream_type = CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY ? 0 : CCV_STREAM_CONTEXT_GPU;
	if (stream_type == CCV_STREAM_CONTEXT_GPU)
		CCV_STREAM_SET_DEVICE_ID(stream_type, CCV_TENSOR_GET_DEVICE_ID(params.type));
	ccv_nnc_tensor_param_t* const context = (ccv_nnc_tensor_param_t*)ccmalloc(sizeof(ccv_nnc_tensor_param_t));
	context[0] = params;
	return ccv_cnnp_dataframe_add(dataframe, _ccv_cnnp_tensor_new, stream_type, _ccv_cnnp_tensor_deinit, context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

#pragma mark - Load Tensor from File Path

static void _ccv_cnnp_image_deinit(void* const data, void* const context)
{
	ccv_matrix_free(data);
}

static void _ccv_cnnp_read_image(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	parallel_for(i, batch_size) {
		if (data[i])
			ccv_matrix_free(data[i]);
		off_t structof = (off_t)context;
		char* const filename = *(char**)((char*)column_data[0][i] + structof);
		data[i] = 0;
		ccv_read(filename, (ccv_dense_matrix_t**)&data[i], CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	} parallel_endfor
}

int ccv_cnnp_dataframe_read_image(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const off_t structof)
{
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_read_image, 0, _ccv_cnnp_image_deinit, COLUMN_ID_LIST(column_idx), (void*)(uintptr_t)structof, 0);
}

#pragma mark - Apply Random Jitter to Image

typedef struct {
	sfmt_t sfmt;
	int datatype;
	ccv_cnnp_random_jitter_t random_jitter;
} ccv_cnnp_random_jitter_context_t;

static void _ccv_cnnp_image_lighting(ccv_dense_matrix_t* image, const float alpha_r, const float alpha_g, const float alpha_b)
{
	assert(CCV_GET_DATA_TYPE(image->type) == CCV_32F);
	assert(CCV_GET_CHANNEL(image->type) == CCV_C3);
	// These eigenvector values can be computed out of imageNet dataset (see ccv_convnet for how that is done). Here I just copied
	// from mxnet: https://github.com/apache/incubator-mxnet/blob/master/src/operator/image/image_random-inl.h#L632
	const float pca_r = alpha_r * (55.46 * -0.5675) + alpha_g * (4.794 * 0.7192) + alpha_b * (1.148 * 0.4009);
	const float pca_g = alpha_r * (55.46 * -0.5808) + alpha_g * (4.794 * -0.0045) + alpha_b * (1.148 * -0.8140);
	const float pca_b = alpha_r * (55.46 * -0.5836) + alpha_g * (4.794 * -0.6948) + alpha_b * (1.148 * 0.4203);
	int i;
	const int size = image->rows * image->cols;
	float* const ptr = image->data.f32;
	for (i = 0; i < size; i++)
	{
		ptr[i * 3] = ccv_clamp(ptr[i * 3] + pca_r, 0, 255);
		ptr[i * 3 + 1] = ccv_clamp(ptr[i * 3 + 1] + pca_g, 0, 255);
		ptr[i * 3 + 2] = ccv_clamp(ptr[i * 3 + 2] + pca_b, 0, 255);
	}
}

static void _ccv_cnnp_image_manip(ccv_dense_matrix_t* image, const ccv_cnnp_random_jitter_t random_jitter, sfmt_t* const sfmt)
{
	assert(sfmt && CCV_GET_CHANNEL(image->type) == CCV_C3);
	int idx[4] = {0, 1, 2, 3};
	sfmt_genrand_shuffle(sfmt, idx, 4, sizeof(int));
	int i;
	for (i = 0; i < 4; i++)
		// change the applying order
		switch (idx[i])
		{
			case 0:
				if (random_jitter.brightness == 0)
					break;
				// introduce some brightness changes to the original image
				ccv_scale(image, (ccv_matrix_t**)&image, 0, sfmt_genrand_real1(sfmt) * random_jitter.brightness * 2 + (1 - random_jitter.brightness));
				break;
			case 1:
				// introduce some saturation changes to the original image
				if (random_jitter.saturation == 0)
					break;
				ccv_saturation(image, &image, 0, sfmt_genrand_real1(sfmt) * random_jitter.saturation * 2 + (1 - random_jitter.saturation));
				break;
			case 2:
				// introduce some contrast changes to the original image
				if (random_jitter.contrast == 0)
					break;
				ccv_contrast(image, &image, 0, sfmt_genrand_real1(sfmt) * random_jitter.contrast * 2 + (1 - random_jitter.contrast));
				break;
			case 3:
				if (random_jitter.lighting == 0)
					break;
				_ccv_cnnp_image_lighting(image, sfmt_genrand_real1(sfmt) * random_jitter.lighting, sfmt_genrand_real1(sfmt) * random_jitter.lighting, sfmt_genrand_real1(sfmt) * random_jitter.lighting);
				break;
		}
}

static void _ccv_cnnp_normalize(ccv_dense_matrix_t* const image, const float mean[3], const float inv_std[3])
{
	int i;
	const int count = image->rows * image->cols;
	float* ap = image->data.f32;
	for (i = 0; i < count; i++)
	{
		ap[i * 3] = (ap[i * 3] - mean[0]) * inv_std[0];
		ap[i * 3 + 1] = (ap[i * 3 + 1] - mean[1]) * inv_std[1];
		ap[i * 3 + 2] = (ap[i * 3 + 2] - mean[2]) * inv_std[2];
	}
}

static void _ccv_cnnp_random_jitter(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	sfmt_t* const sfmt = (sfmt_t*)alloca(sizeof(sfmt_t) * batch_size);
	ccv_cnnp_random_jitter_context_t* const ctx = (ccv_cnnp_random_jitter_context_t*)context;
	int i;
	for (i = 0; i < batch_size; i++)
		sfmt_init_gen_rand(&sfmt[i], sfmt_genrand_uint32(&ctx->sfmt));
	const ccv_cnnp_random_jitter_t random_jitter = ctx->random_jitter;
	assert(random_jitter.resize.min > 0);
	assert(random_jitter.resize.max >= random_jitter.resize.min);
	parallel_for(i, batch_size) {
		if (data[i])
			ccv_matrix_free(data[i]);
		ccv_dense_matrix_t* const input = (ccv_dense_matrix_t*)column_data[0][i];
		const int resize = ccv_clamp((int)(sfmt_genrand_real1(&sfmt[i]) * (random_jitter.resize.max - random_jitter.resize.min) + 0.5) + random_jitter.resize.min, random_jitter.resize.min, random_jitter.resize.max);
		int resize_rows = ccv_max(resize, (int)(input->rows * (float)resize / input->cols + 0.5));
		int resize_cols = ccv_max(resize, (int)(input->cols * (float)resize / input->rows + 0.5));
		if (random_jitter.aspect_ratio > 0)
		{
			const float aspect_ratio = sqrtf((random_jitter.aspect_ratio + (1 - 1 / (1 + random_jitter.aspect_ratio))) * sfmt_genrand_real1(&sfmt[i]) + 1 / (1 + random_jitter.aspect_ratio));
			resize_rows = (int)(resize_rows * aspect_ratio + 0.5);
			resize_cols = (int)(resize_cols / aspect_ratio + 0.5);
		}
		ccv_dense_matrix_t* resized = 0;
		// First, resize.
		if (input->rows > resize && input->cols > resize)
			ccv_resample(input, &resized, CCV_32F, resize_rows, resize_cols, CCV_INTER_AREA);
		else if (input->rows != resize_rows || input->cols != resize_cols)
			ccv_resample(input, &resized, CCV_32F, resize_rows, resize_cols, CCV_INTER_CUBIC);
		else
			ccv_shift(input, (ccv_matrix_t**)&resized, CCV_32F, 0, 0); // converting to 32f
		if (random_jitter.symmetric && (sfmt_genrand_uint32(&sfmt[i]) & 1) == 0)
			ccv_flip(resized, &resized, 0, CCV_FLIP_X);
		_ccv_cnnp_image_manip(resized, random_jitter, &sfmt[i]);
		// Apply normalization before slice. Slice will introduce 0 padding, which won't be correct before normalization.
		if (random_jitter.normalize.mean[0] != 0 || random_jitter.normalize.std[0] != 1 ||
			random_jitter.normalize.mean[1] != 0 || random_jitter.normalize.std[1] != 1 ||
			random_jitter.normalize.mean[2] != 0 || random_jitter.normalize.std[2] != 1)
			_ccv_cnnp_normalize(resized, random_jitter.normalize.mean, random_jitter.normalize.std);
		// Then slice down.
		ccv_dense_matrix_t* patch = 0;
		if (random_jitter.size.cols > 0 && random_jitter.size.rows > 0 &&
			((resized->cols != random_jitter.size.cols || resized->rows != random_jitter.size.rows) ||
			 (random_jitter.offset.x != 0 || random_jitter.offset.y != 0)))
		{
			int x = ccv_clamp((int)(sfmt_genrand_real1(&sfmt[i]) * (resized->cols - random_jitter.size.cols + 1)),
						ccv_min(0, resized->cols - random_jitter.size.cols),
						ccv_max(0, resized->cols - random_jitter.size.cols));
			int y = ccv_clamp((int)(sfmt_genrand_real1(&sfmt[i]) * (resized->rows - random_jitter.size.rows + 1)),
					ccv_min(0, resized->rows - random_jitter.size.rows),
					ccv_max(0, resized->rows - random_jitter.size.rows));
			if (random_jitter.offset.x != 0)
				x += sfmt_genrand_real1(&sfmt[i]) * random_jitter.offset.x * 2 - random_jitter.offset.x;
			if (random_jitter.offset.y != 0)
				y += sfmt_genrand_real1(&sfmt[i]) * random_jitter.offset.y * 2 - random_jitter.offset.y;
			ccv_slice(resized, (ccv_matrix_t**)&patch, CCV_32F, y, x, random_jitter.size.rows, random_jitter.size.cols);
			ccv_matrix_free(resized);
		} else
			patch = resized;
		data[i] = patch;
	} parallel_endfor
}

int ccv_cnnp_dataframe_image_random_jitter(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int datatype, const ccv_cnnp_random_jitter_t random_jitter)
{
	assert(datatype == CCV_32F);
	ccv_cnnp_random_jitter_context_t* const random_jitter_context = (ccv_cnnp_random_jitter_context_t*)ccmalloc(sizeof(ccv_cnnp_random_jitter_context_t));
	if (random_jitter.seed)
		sfmt_init_gen_rand(&random_jitter_context->sfmt, (uint32_t)random_jitter.seed);
	else
		sfmt_init_gen_rand(&random_jitter_context->sfmt, (uint32_t)(uintptr_t)dataframe);
	random_jitter_context->datatype = datatype;
	random_jitter_context->random_jitter = random_jitter;
	int i;
	// The std in the random jitter should be inv_std.
	for (i = 0; i < 3; i++)
		random_jitter_context->random_jitter.normalize.std[i] = random_jitter_context->random_jitter.normalize.std[i] ? 1. / random_jitter_context->random_jitter.normalize.std[i] : 1;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_random_jitter, 0, _ccv_cnnp_image_deinit, COLUMN_ID_LIST(column_idx), random_jitter_context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

typedef struct {
	int range;
	int datatype;
	int format;
	float onval;
	float offval;
	off_t structof;
} ccv_cnnp_one_hot_context_t;

static void _ccv_cnnp_one_hot(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_one_hot_context_t* const one_hot = (ccv_cnnp_one_hot_context_t*)context;
	ccv_nnc_tensor_param_t params = {
		.datatype = one_hot->datatype,
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = one_hot->format,
		.dim = {
			one_hot->range,
		},
	};
	parallel_for(i, batch_size) {
		int j;
		const int label = *(int*)((char*)column_data[0][i] + one_hot->structof);
		if (!data[i])
			data[i] = ccv_nnc_tensor_new(0, params, 0);
		ccv_nnc_tensor_t* const tensor = (ccv_nnc_tensor_t*)data[i];
		assert(label >= 0 && label < one_hot->range);
		for (j = 0; j < one_hot->range; j++)
			tensor->data.f32[j] = (j == label) ? one_hot->onval : one_hot->offval;
	} parallel_endfor
}

int ccv_cnnp_dataframe_one_hot(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const off_t structof, const int range, const float onval, const float offval, const int datatype, const int format)
{
	assert(datatype == CCV_32F);
	ccv_cnnp_one_hot_context_t* const one_hot = (ccv_cnnp_one_hot_context_t*)ccmalloc(sizeof(ccv_cnnp_one_hot_context_t));
	one_hot->range = range;
	one_hot->datatype = datatype;
	one_hot->format = format;
	one_hot->onval = onval;
	one_hot->offval = offval;
	one_hot->structof = structof;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_one_hot, 0, _ccv_cnnp_tensor_deinit, COLUMN_ID_LIST(column_idx), one_hot, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}

typedef struct {
	ccv_cnnp_dataframe_tuple_t tuple;
	int format;
	int batch_count;
	int group_count;
} ccv_cnnp_batch_context_t;

static void _ccv_cnnp_batching_new(void** const input_data, const int input_size, void** const output_data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_cnnp_batch_context_t* const batch = (ccv_cnnp_batch_context_t*)context;
	const int output_tuple_size = batch->tuple.size;
	const int batch_count = batch->batch_count;
	const int group_count = batch->group_count;
	const int input_tuple_size = output_tuple_size / group_count;
	int i, j, k;
	assert(input_size > 0);
	if (!output_data[0])
	{
		ccv_nnc_tensor_t** const inputs = (ccv_nnc_tensor_t**)input_data[0];
		ccv_nnc_tensor_t** const tensors = (ccv_nnc_tensor_t**)(output_data[0] = ccmalloc(sizeof(ccv_nnc_tensor_t*) * output_tuple_size));
		for (i = 0; i < group_count; i++)
			for (j = 0; j < input_tuple_size; j++)
			{
				ccv_nnc_tensor_param_t params = inputs[j]->info;
				assert(params.datatype == CCV_32F); // Only support 32 bit float yet.
				assert(params.format == CCV_TENSOR_FORMAT_NHWC || params.format == CCV_TENSOR_FORMAT_NCHW);
				params.format = batch->format;
				// Special-case for dim count is 3 and 1, in these two cases, the N is not provided.
				if (batch->format == inputs[j]->info.format)
				{
					const int nd = ccv_nnc_tensor_nd(params.dim);
					if (nd == 3 || nd == 1)
					{
						memset(params.dim, 0, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
						memcpy(params.dim + 1, inputs[j]->info.dim, sizeof(int) * nd);
					}
				} else {
					const int nd = ccv_nnc_tensor_nd(params.dim);
					if (nd == 1)
					{
						memset(params.dim, 0, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
						memcpy(params.dim + 1, inputs[j]->info.dim, sizeof(int) * nd);
					} else if (nd >= 3) {
						memset(params.dim, 0, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
						const int hw = ccv_nnc_tensor_hw(inputs[j]->info, nd);
						if (batch->format == CCV_TENSOR_FORMAT_NCHW)
						{
							params.dim[1] = ccv_nnc_tensor_get_c(inputs[j]->info);
							for (k = 0; k < CCV_NNC_MAX_DIM; k++)
								params.dim[k + 2] = inputs[j]->info.dim[k + hw];
						} else {
							params.dim[CCV_NNC_MAX_DIM + 1] = ccv_nnc_tensor_get_c(inputs[j]->info);
							for (k = 0; k < CCV_NNC_MAX_DIM; k++)
								params.dim[k + 1] = inputs[j]->info.dim[k + hw];
						}
					}
				}
				params.dim[0] = batch_count; // Set the batch count now.
				tensors[i * input_tuple_size + j] = ccv_nnc_tensor_new(0, params, 0);
			}
	}
	for (i = 0; i < group_count; i++)
		for (j = 0; j < input_tuple_size; j++)
		{
			ccv_nnc_tensor_t* const output = ((ccv_nnc_tensor_t**)output_data[0])[i * input_tuple_size + j];
			parallel_for(k, batch_count) {
				ccv_nnc_tensor_t* const input = ((ccv_nnc_tensor_t**)input_data[(k + i * batch_count) % input_size])[j];
				const size_t tensor_count = ccv_nnc_tensor_count(input->info);
				float* const ap = input->data.f32;
				float* const bp = output->data.f32 + k * tensor_count;
				if (input->info.format == output->info.format)
					memcpy(bp, ap, sizeof(float) * tensor_count);
				else {
					// Do a simple format conversion.
					const int c = ccv_nnc_tensor_get_c(input->info);
					const size_t hw_count = tensor_count / c;
					size_t x;
					int y;
					if (input->info.format == CCV_TENSOR_FORMAT_NHWC && output->info.format == CCV_TENSOR_FORMAT_NCHW)
						for (x = 0; x < hw_count; x++)
							for (y = 0; y < c; y++)
								bp[y * hw_count + x] = ap[x * c + y];
					else if (input->info.format == CCV_TENSOR_FORMAT_NCHW && output->info.format == CCV_TENSOR_FORMAT_NHWC)
						for (x = 0; x < hw_count; x++)
							for (y = 0; y < c; y++)
								bp[x * c + y] = ap[y * hw_count + x];
				}
			} parallel_endfor
		}
}

static void _ccv_cnnp_batching_deinit(void* const self, void* const context)
{
	ccv_cnnp_batch_context_t* const batch = (ccv_cnnp_batch_context_t*)context;
	ccv_nnc_tensor_t** const tensors = (ccv_nnc_tensor_t**)self;
	const int size = batch->tuple.size;
	int i;
	for (i = 0; i < size; i++)
		ccv_nnc_tensor_free(tensors[i]);
	ccfree(tensors);
}

ccv_cnnp_dataframe_t* ccv_cnnp_dataframe_batching_new(ccv_cnnp_dataframe_t* const dataframe, const int* const column_idxs, const int column_idx_size, const int batch_count, const int group_count, const int format)
{
	assert(format == CCV_TENSOR_FORMAT_NCHW || format == CCV_TENSOR_FORMAT_NHWC);
	assert(column_idx_size >= 1);
	assert(batch_count > 0);
	assert(group_count > 0);
	const int derived = ccv_cnnp_dataframe_make_tuple(dataframe, column_idxs, column_idx_size);
	ccv_cnnp_batch_context_t* const batch = (ccv_cnnp_batch_context_t*)ccmalloc(sizeof(ccv_cnnp_batch_context_t));
	batch->tuple.size = column_idx_size * group_count;
	batch->format = format;
	batch->batch_count = batch_count;
	batch->group_count = group_count;
	return ccv_cnnp_dataframe_reduce_new(dataframe, _ccv_cnnp_batching_new, _ccv_cnnp_batching_deinit, derived, batch_count * group_count, batch, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}
