#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("derive one-hot tensor from label")
{
	int int_array[8] = {
		2, 3, 4, 5, 6, 7, 8, 9
	};
	ccv_array_t* const array = ccv_array_new(sizeof(int), 8, 0);
	ccv_array_resize(array, 8);
	memcpy(ccv_array_get(array, 0), int_array, sizeof(int) * 8);
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_from_array_new(array);
	const int oh = ccv_cnnp_dataframe_one_hot(dataframe, 0, 0, 10, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	assert(oh > 0);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(dataframe, COLUMN_ID_LIST(oh));
	ccv_nnc_tensor_t* const one_hot = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(10), 0);
	void* data;
	int i = 0, j;
	while (0 == ccv_cnnp_dataframe_iter_next(iter, &data, 1, 0))
	{
		for (j = 0; j < 10; j++)
			one_hot->data.f32[j] = 0;
		one_hot->data.f32[int_array[i]] = 1;
		REQUIRE_TENSOR_EQ(one_hot, (ccv_nnc_tensor_t*)data, "The one-hot tensor should be the same");
		++i;
	}
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(dataframe);
	ccv_nnc_tensor_free(one_hot);
	ccv_array_free(array);
}

TEST_CASE("batching tensors")
{
	int int_array[8] = {
		2, 3, 4, 5, 6, 7, 8, 9
	};
	ccv_array_t* const array = ccv_array_new(sizeof(int), 8, 0);
	ccv_array_resize(array, 8);
	memcpy(ccv_array_get(array, 0), int_array, sizeof(int) * 8);
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_from_array_new(array);
	const int oh = ccv_cnnp_dataframe_one_hot(dataframe, 0, 0, 10, 1, 0, CCV_32F, CCV_TENSOR_FORMAT_NCHW);
	assert(oh > 0);
	ccv_cnnp_dataframe_t* const batch = ccv_cnnp_dataframe_batching_new(dataframe, COLUMN_ID_LIST(oh), 3, 2, CCV_TENSOR_FORMAT_NCHW);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(batch, COLUMN_ID_LIST(0));
	ccv_nnc_tensor_t* const one_hot = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(3, 10), 0);
	void* data;
	int i = 0, j;
	ccv_cnnp_dataframe_iter_next(iter, &data, 1, 0);
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 10; j++)
			one_hot->data.f32[i * 10 + j] = 0;
		one_hot->data.f32[i * 10 + int_array[i]] = 1;
	}
	REQUIRE_TENSOR_EQ(one_hot, ((ccv_nnc_tensor_t**)data)[0], "The one-hot tensor should be the same");
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 10; j++)
			one_hot->data.f32[i * 10 + j] = 0;
		one_hot->data.f32[i * 10 + int_array[i + 3]] = 1;
	}
	REQUIRE_TENSOR_EQ(one_hot, ((ccv_nnc_tensor_t**)data)[1], "The one-hot tensor should be the same");
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(batch);
	ccv_cnnp_dataframe_free(dataframe);
	ccv_nnc_tensor_free(one_hot);
	ccv_array_free(array);
}

TEST_CASE("read image and add random jitter")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../../samples/nature.png", &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_array_t* const array = ccv_array_new(sizeof(ccv_dense_matrix_t), 1, 0);
	ccv_array_push(array, image);
	ccv_cnnp_dataframe_t* const dataframe = ccv_cnnp_dataframe_from_array_new(array);
	const ccv_cnnp_random_jitter_t random_jitter = {
		.resize = {
			.min = 200,
			.max = 200,
		},
		.size = {
			.rows = 224,
			.cols = 224,
		},
		.normalize = {
			.mean = {
				123.68, 116.779, 103.939
			},
			.std = {
				58.393, 57.12, 57.375
			},
		},
		.offset = {
			.x = 10,
			.y = 10,
		},
		.aspect_ratio = 0.33,
		.contrast = 0.4,
		.saturation = 0.4,
		.brightness = 0.4,
		.lighting = 0.1,
		.seed = 1,
	};
	const int im = ccv_cnnp_dataframe_image_random_jitter(dataframe, 0, CCV_32F, random_jitter);
	ccv_cnnp_dataframe_iter_t* const iter = ccv_cnnp_dataframe_iter_new(dataframe, COLUMN_ID_LIST(im));
	ccv_dense_matrix_t* data;
	ccv_cnnp_dataframe_iter_next(iter, (void**)&data, 1, 0);
	REQUIRE_MATRIX_FILE_EQ(data, "data/nature.random-jitter.bin", "should be the same random jitter image.");
	ccv_matrix_free(image);
	ccv_array_free(array);
	ccv_cnnp_dataframe_iter_free(iter);
	ccv_cnnp_dataframe_free(dataframe);
}

#include "case_main.h"
