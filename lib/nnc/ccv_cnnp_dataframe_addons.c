#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"

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
	int tensor_offset;
	int tensor_size;
	int device_id;
} ccv_cnnp_copy_to_gpu_context_t;

#pragma mark - Copy Tensors from CPU to GPU

static void _ccv_cnnp_tensor_list_deinit(void* const data, void* const context)
{
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu = (ccv_cnnp_copy_to_gpu_context_t*)context;
	ccv_nnc_tensor_t** const tensor_list = (ccv_nnc_tensor_t**)data;
	int i;
	for (i = 0; i < copy_to_gpu->tensor_size; i++)
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
			outputs = (ccv_nnc_tensor_t**)(data[i] = ccmalloc(sizeof(ccv_nnc_tensor_t*) * copy_to_gpu_context->tensor_size));
			for (j = 0; j < copy_to_gpu_context->tensor_size; j++)
			{
				ccv_nnc_tensor_param_t params = inputs[j]->info;
				params.type &= ~CCV_TENSOR_CPU_MEMORY;
				params.type |= CCV_TENSOR_GPU_MEMORY; // Change to GPU memory.
				CCV_TENSOR_SET_DEVICE_ID(params.type, copy_to_gpu_context->device_id);
				outputs[j] = ccv_nnc_tensor_new(0, params, 0);
			}
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, inputs, copy_to_gpu_context->tensor_size, outputs, copy_to_gpu_context->tensor_size, stream_context);
	}
}

int ccv_cnnp_dataframe_copy_to_gpu(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int tensor_offset, const int tensor_size, int device_id)
{
	assert(tensor_size > 0);
	int stream_type = CCV_STREAM_CONTEXT_GPU;
	CCV_STREAM_SET_DEVICE_ID(stream_type, device_id);
	ccv_cnnp_copy_to_gpu_context_t* const copy_to_gpu_context = (ccv_cnnp_copy_to_gpu_context_t*)ccmalloc(sizeof(ccv_cnnp_copy_to_gpu_context_t));
	copy_to_gpu_context->tensor_offset = tensor_offset;
	copy_to_gpu_context->tensor_size = tensor_size;
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
	int stream_type = CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_CPU_MEMORY ? CCV_STREAM_CONTEXT_CPU : CCV_STREAM_CONTEXT_GPU;
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
		off_t offset = (off_t)context;
		char* const filename = *(char**)((char*)column_data[0][i] + offset);
		data[i] = 0;
		ccv_read(filename, (ccv_dense_matrix_t**)&data[i], CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	} parallel_endfor
}

int ccv_cnnp_dataframe_read_image(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const off_t filename_offset)
{
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_read_image, 0, _ccv_cnnp_image_deinit, COLUMN_ID_LIST(column_idx), (void*)(uintptr_t)filename_offset, 0);
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

static void _ccv_cnnp_random_jitter(void*** const column_data, const int column_size, const int batch_size, void** const data, void* const context, ccv_nnc_stream_context_t* const stream_context)
{
	sfmt_t sfmt[batch_size];
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
		ccv_dense_matrix_t* resized = 0;
		// First, resize.
		if (input->rows > resize && input->cols > resize)
			ccv_resample(input, &resized, CCV_32F, ccv_max(resize, (int)(input->rows * (float)resize / input->cols + 0.5)), ccv_max(resize, (int)(input->cols * (float)resize / input->rows + 0.5)), CCV_INTER_AREA);
		else if (input->rows < resize || input->cols < resize)
			ccv_resample(input, &resized, CCV_32F, ccv_max(resize, (int)(input->rows * (float)resize / input->cols + 0.5)), ccv_max(resize, (int)(input->cols * (float)resize / input->rows + 0.5)), CCV_INTER_CUBIC);
		else
			ccv_shift(input, (ccv_matrix_t**)resized, CCV_32F, 0, 0); // converting to 32f
		// Then slice down.
		ccv_dense_matrix_t* patch = 0;
		if (random_jitter.size.cols > 0 && random_jitter.size.rows > 0 &&
			(resized->cols != random_jitter.size.cols || resized->rows != random_jitter.size.rows))
		{
			assert(resized->cols >= random_jitter.size.cols);
			assert(resized->rows >= random_jitter.size.rows);
			int x = ccv_clamp((int)(sfmt_genrand_real1(&sfmt[i]) * (resized->cols - random_jitter.size.cols + 1)), 0, resized->cols - random_jitter.size.cols);
			int y = ccv_clamp((int)(sfmt_genrand_real1(&sfmt[i]) * (resized->rows - random_jitter.size.rows + 1)), 0, resized->rows - random_jitter.size.rows);
			ccv_slice(resized, (ccv_matrix_t**)&patch, CCV_32F, y, x, random_jitter.size.rows, random_jitter.size.cols);
			ccv_matrix_free(resized);
		} else
			patch = resized;
		if (random_jitter.symmetric && (sfmt_genrand_uint32(&sfmt[i]) & 1) == 0)
			ccv_flip(patch, &patch, 0, CCV_FLIP_X);
		_ccv_cnnp_image_manip(patch, random_jitter, &sfmt[i]);
		data[i] = patch;
	} parallel_endfor
}

int ccv_cnnp_dataframe_image_random_jitter(ccv_cnnp_dataframe_t* const dataframe, const int column_idx, const int datatype, const ccv_cnnp_random_jitter_t random_jitter)
{
	assert(datatype == CCV_32F);
	ccv_cnnp_random_jitter_context_t* const random_jitter_context = (ccv_cnnp_random_jitter_context_t*)ccmalloc(sizeof(ccv_cnnp_random_jitter_context_t));
	sfmt_init_gen_rand(&random_jitter_context->sfmt, (uint32_t)(uintptr_t)dataframe);
	random_jitter_context->datatype = datatype;
	random_jitter_context->random_jitter = random_jitter;
	return ccv_cnnp_dataframe_map(dataframe, _ccv_cnnp_random_jitter, 0, _ccv_cnnp_image_deinit, COLUMN_ID_LIST(column_idx), random_jitter_context, (ccv_cnnp_column_data_context_deinit_f)ccfree);
}
