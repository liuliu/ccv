#include "cwc.h"
#include "../ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#include "cwc_helper.h"
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

#ifdef HAVE_GSL
static void _cwc_convnet_random_image_manipulation(gsl_rng* rng, ccv_dense_matrix_t* image, float image_manipulation)
{
	assert(rng && CCV_GET_CHANNEL(image->type) == CCV_C3 && image_manipulation > 0 && image_manipulation <= 1);
	int ord[3] = {0, 1, 2};
	gsl_ran_shuffle(rng, ord, 3, sizeof(int));
	int i;
	for (i = 0; i < 3; i++)
		// change the applying order
		switch (ord[i])
		{
			case 0:
				// introduce some brightness changes to the original image
				ccv_scale(image, (ccv_matrix_t**)&image, 0, gsl_rng_uniform_pos(rng) * image_manipulation * 2 + (1 - image_manipulation));
				break;
			case 1:
				// introduce some saturation changes to the original image
				ccv_saturation(image, &image, 0, gsl_rng_uniform_pos(rng) * image_manipulation * 2 + (1 - image_manipulation));
				break;
			case 2:
				// introduce some contrast changes to the original image
				ccv_contrast(image, &image, 0, gsl_rng_uniform_pos(rng) * image_manipulation * 2 + (1 - image_manipulation));
				break;
		}
}

void cwc_convnet_batch_formation(gsl_rng* rng, ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_dense_matrix_t* eigenvectors, ccv_dense_matrix_t* eigenvalues, float image_manipulation, float color_gain, int* idx, ccv_size_t dim, int min_dim, int max_dim, int rows, int cols, int channels, int category_count, int symmetric, int batch, int offset, int size, float* b, int* c)
{
	assert(size > 0 && size <= batch);
	assert(min_dim >= rows && min_dim >= cols);
	assert(max_dim >= min_dim);
	float* channel_gains = (float*)alloca(sizeof(float) * channels * size);
	memset(channel_gains, 0, sizeof(float) * channels * size);
	int i;
	gsl_rng** rngs = (gsl_rng**)alloca(sizeof(gsl_rng*) * size);
	memset(rngs, 0, sizeof(gsl_rng*) * size);
	if (rng)
		for (i = 0; i < size; i++)
		{
			rngs[i] = gsl_rng_alloc(gsl_rng_default);
			gsl_rng_set(rngs[i], gsl_rng_get(rng));
		}
	parallel_for(i, size) {
		int j, k;
		assert(offset + i < categorizeds->rnum);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, idx ? idx[offset + i] : offset + i);
		assert(categorized->c < category_count && categorized->c >= 0); // now only accept classes listed
		if (c)
			c[i] = categorized->c;
		ccv_dense_matrix_t* image = 0;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				image = 0;
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				break;
		}
		if (image)
		{
			// first resize to between min_dim and max_dim
			ccv_dense_matrix_t* input = 0;
			ccv_size_t resize = dim;
			if (rngs[i]) // randomize the resized dimensions
			{
				int d = gsl_rng_uniform_int(rngs[i], max_dim - min_dim + 1) + min_dim;
				resize = ccv_size(d, d);
			}
			// if neither side is the same as expected, we have to resize first
			if (resize.height != image->rows && resize.width != image->cols)
				ccv_convnet_input_formation(resize, image, &input);
			else
				input = image;
			if (rngs[i] && image_manipulation > 0)
				_cwc_convnet_random_image_manipulation(rngs[i], input, image_manipulation);
			ccv_dense_matrix_t* patch = 0;
			if (input->cols != cols || input->rows != rows)
			{
				int x = rngs[i] ? gsl_rng_uniform_int(rngs[i], input->cols - cols + 1) : (input->cols - cols + 1) / 2;
				int y = rngs[i] ? gsl_rng_uniform_int(rngs[i], input->rows - rows + 1) : (input->rows - rows + 1) / 2;
				ccv_slice(input, (ccv_matrix_t**)&patch, CCV_32F, y, x, rows, cols);
			} else
				ccv_shift(input, (ccv_matrix_t**)&patch, CCV_32F, 0, 0); // converting to 32f
			if (input != image) // only unload if we created new input
				ccv_matrix_free(input);
			// we loaded image in, deallocate it now
			if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
				ccv_matrix_free(image);
			// random horizontal reflection
			if (symmetric && rngs[i] && gsl_rng_uniform_int(rngs[i], 2) == 0)
				ccv_flip(patch, &patch, 0, CCV_FLIP_X);
			int x = rngs[i] ? gsl_rng_uniform_int(rngs[i], mean_activity->cols - cols + 1) : (mean_activity->cols - cols + 1) / 2;
			int y = rngs[i] ? gsl_rng_uniform_int(rngs[i], mean_activity->rows - rows + 1) : (mean_activity->rows - rows + 1) / 2;
			ccv_dense_matrix_t mean_patch = ccv_reshape(mean_activity, y, x, rows, cols);
			ccv_subtract(patch, &mean_patch, (ccv_matrix_t**)&patch, 0);
			assert(channels == CCV_GET_CHANNEL(patch->type));
			if (color_gain > 0 && rngs[i] && eigenvectors && eigenvalues)
			{
				assert(channels == 3); // only support RGB color gain
				memset(channel_gains + channels * i, 0, sizeof(float) * channels);
				for (j = 0; j < channels; j++)
				{
					float alpha = gsl_ran_gaussian(rngs[i], color_gain) * eigenvalues->data.f64[j];
					for (k = 0; k < channels; k++)
						channel_gains[k + i * channels] += eigenvectors->data.f64[j * channels + k] * alpha;
				}
			}
			for (j = 0; j < channels; j++)
				for (k = 0; k < rows * cols; k++)
					b[(j * rows * cols + k) * batch + i] = patch->data.f32[k * channels + j] + channel_gains[j + i * channels];
			ccv_matrix_free(patch);
		} else
			PRINT(CCV_CLI_ERROR, "cannot load %s.\n", categorized->file.filename);
	} parallel_endfor
	if (rng)
		for (i = 0; i < size; i++)
			gsl_rng_free(rngs[i]);
}
#endif

void cwc_convnet_mean_formation(ccv_array_t* categorizeds, ccv_size_t dim, int channels, int symmetric, ccv_dense_matrix_t** b)
{
	int i, count = 0;
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(dim.height, dim.width, channels | CCV_64F, 0, 0);
	ccv_zero(c);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, dim.height, dim.width, channels | CCV_32F, channels | CCV_32F, 0);
	for (i = 0; i < categorizeds->rnum; i++)
	{
		if (i % 23 == 0 || i == categorizeds->rnum - 1)
			FLUSH(CCV_CLI_INFO, " - compute mean activity %d / %d", i + 1, categorizeds->rnum);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, i);
		ccv_dense_matrix_t* image = 0;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				break;
		}
		if (!image)
		{
			PRINT(CCV_CLI_ERROR, "cannot load %s.\n", categorized->file.filename);
			continue;
		}
		ccv_dense_matrix_t* patch = 0;
		if (image->cols != dim.width || image->rows != dim.height)
		{
			int x = (image->cols - dim.width + 1) / 2;
			int y = (image->rows - dim.height + 1) / 2;
			assert(x == 0 || y == 0);
			ccv_slice(image, (ccv_matrix_t**)&patch, CCV_32F, y, x, dim.height, dim.width);
		} else
			ccv_shift(image, (ccv_matrix_t**)&patch, CCV_32F, 0, 0); // converting to 32f
		if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
			ccv_matrix_free(image);
		ccv_add(patch, c, (ccv_matrix_t**)&c, CCV_64F);
		++count;
		ccv_matrix_free(patch);
	}
	if (symmetric)
	{
		int j, k;
		double p = 0.5 / count;
		double* cptr = c->data.f64;
		float* dbptr = db->data.f32;
		for (i = 0; i < db->rows; i++)
		{
			for (j = 0; j < db->cols; j++)
				for (k = 0; k < channels; k++)
					dbptr[j * channels + k] = p * (cptr[j * channels + k] + cptr[(c->cols - j - 1) * channels + k]);
			dbptr += db->cols * channels;
			cptr += c->cols * channels;
		}
	} else {
		double p = 1.0 / count;
		for (i = 0; i < dim.height * dim.width * channels; i++)
			db->data.f32[i] = p * c->data.f64[i];
	}
	ccv_matrix_free(c);
	PRINT(CCV_CLI_INFO, "\n");
}

void cwc_convnet_channel_eigen(ccv_array_t* categorizeds, ccv_dense_matrix_t* mean_activity, ccv_size_t dim, int channels, ccv_dense_matrix_t** eigenvectors, ccv_dense_matrix_t** eigenvalues)
{
	assert(channels == 3); // this function cannot handle anything other than 3x3 covariance matrix
	double* mean_value = (double*)alloca(sizeof(double) * channels);
	memset(mean_value, 0, sizeof(double) * channels);
	assert(CCV_GET_CHANNEL(mean_activity->type) == channels);
	assert(mean_activity->rows == dim.height);
	assert(mean_activity->cols == dim.width);
	int i, j, k, c, count = 0;
	for (i = 0; i < dim.height * dim.width; i++)
		for (k = 0; k < channels; k++)
			mean_value[k] += mean_activity->data.f32[i * channels + k];
	for (i = 0; i < channels; i++)
		mean_value[i] = mean_value[i] / (dim.height * dim.width);
	double* covariance = (double*)alloca(sizeof(double) * channels * channels);
	memset(covariance, 0, sizeof(double) * channels * channels);
	for (c = 0; c < categorizeds->rnum; c++)
	{
		if (c % 23 == 0 || c == categorizeds->rnum - 1)
			FLUSH(CCV_CLI_INFO, " - compute covariance matrix for data augmentation (color gain) %d / %d", c + 1, categorizeds->rnum);
		ccv_categorized_t* categorized = (ccv_categorized_t*)ccv_array_get(categorizeds, c);
		ccv_dense_matrix_t* image = 0;
		switch (categorized->type)
		{
			case CCV_CATEGORIZED_DENSE_MATRIX:
				image = categorized->matrix;
				break;
			case CCV_CATEGORIZED_FILE:
				ccv_read(categorized->file.filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				break;
		}
		if (!image)
		{
			PRINT(CCV_CLI_ERROR, "cannot load %s.\n", categorized->file.filename);
			continue;
		}
		ccv_dense_matrix_t* patch = 0;
		if (image->cols != dim.width || image->rows != dim.height)
		{
			int x = (image->cols - dim.width + 1) / 2;
			int y = (image->rows - dim.height + 1) / 2;
			assert(x == 0 || y == 0);
			ccv_slice(image, (ccv_matrix_t**)&patch, CCV_32F, y, x, dim.height, dim.width);
		} else
			ccv_shift(image, (ccv_matrix_t**)&patch, CCV_32F, 0, 0); // converting to 32f
		if (categorized->type != CCV_CATEGORIZED_DENSE_MATRIX)
			ccv_matrix_free(image);
		for (i = 0; i < dim.width * dim.height; i++)
			for (j = 0; j < channels; j++)
				for (k = j; k < channels; k++)
					covariance[j * channels + k] += (patch->data.f32[i * channels + j] - mean_value[j]) * (patch->data.f32[i * channels + k] - mean_value[k]);
		++count;
		ccv_matrix_free(patch);
	}
	for (i = 0; i < channels; i++)
		for (j = 0; j < i; j++)
			covariance[i * channels + j] = covariance[j * channels + i];
	double p = 1.0 / ((double)count * dim.height * dim.width);
	for (i = 0; i < channels; i++)
		for (j = 0; j < channels; j++)
			covariance[i * channels + j] *= p; // scale down
	ccv_dense_matrix_t covm = ccv_dense_matrix(3, 3, CCV_64F | CCV_C1, covariance, 0);
	ccv_eigen(&covm, eigenvectors, eigenvalues, CCV_64F, 1e-8);
	PRINT(CCV_CLI_INFO, "\n");
}
