#include "../src/ccv.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_qrng.h>

double gabor_kernel(double x, double y, void* data)
{
	int* so = (int*)data;
	double kv = 3.141592654 * exp(-0.34657359 * (so[0] + 2));
	double qu = so[1] * 3.141592654 / so[2];
	double cos_qu = cos(qu);
	double sin_qu = sin(qu);
	double kv_kv_si = kv * kv * 0.050660592;
	double exp_kv_x_y = exp(-kv * kv * (x * x + y * y) * 0.025330296);
	double kv_qu_x_y = kv * (cos_qu * x + sin_qu * y);
	return exp_kv_x_y * (cos(kv_qu_x_y) - 0.000051723) * kv_kv_si;
}

typedef struct {
	int label;
	float desc[48];
} rich_float_pixel_t;

#define KMEANS_GABOR_SIZE (30)
#define KMEANS_TEXTON_SIZE (12)

int colors[] =  { 0, 0, 255, 0, 128, 255, 0, 255, 255, 0, 255, 0, 255, 128, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 128, 0, 255, 128, 128, 255, 255, 128, 128, 128, 255, 128 };

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* im = NULL;
	ccv_unserialize(argv[1], &im, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* imf = ccv_dense_matrix_new(im->rows, im->cols, CCV_32F | CCV_C1, NULL, NULL);
	int i, j;
	for (i = 0; i < imf->rows; i++)
		for (j = 0; j < imf->cols; j++)
			imf->data.fl[i * imf->cols + j] = (im->data.ptr[i * im->step + j * 3] * 29 + im->data.ptr[i * im->step + j * 3 + 1] * 61 + im->data.ptr[i * im->step + j * 3 + 2] * 10) / 100;
	ccv_dense_matrix_t* kernel = ccv_dense_matrix_new(48, 48, CCV_32F | CCV_C1, NULL, NULL);
	rich_float_pixel_t* gim = (rich_float_pixel_t*)malloc(im->rows * im->cols * sizeof(rich_float_pixel_t));
	ccv_dense_matrix_t* out = NULL;
	int k = 0;
	for (i = 0; i < 8; i++)
		for (j = 0; j < 6; j++)
		{
			int so[3] = { i, j, 6 };
			ccv_filter_kernel(kernel, gabor_kernel, so);
			ccv_filter(imf, kernel, &out);
			int x;
			for (x = 0; x < imf->rows * imf->cols; x++)
			{
				gim[x].desc[k] = out->data.fl[x];
			}
			k++;
		}
	rich_float_pixel_t cs[KMEANS_GABOR_SIZE];

	gsl_rng_env_setup();

	const gsl_rng_type* T = gsl_rng_default;
	gsl_rng* r = gsl_rng_alloc(T);
	gsl_rng_set(r, 0);
	
	for (i = 0; i < KMEANS_GABOR_SIZE; i++)
	{
		k = (int)(gsl_rng_uniform(r) * imf->rows * imf->cols);
		cs[i].label = k;
		memcpy(cs[i].desc, gim[k].desc, 48 * sizeof(float));
	}
	int epoch = 0;
	/*
	for (;; epoch++)
	{
		int f = 0;
		double total_dist = 0;
		for (i = 0; i < imf->rows * imf->cols; i++)
		{
			double dist = 1e16; int dist_i = -1;
			for (j = 0; j < KMEANS_GABOR_SIZE; j++)
			{
				double tmp = 0;
				for (k = 0; k < 48; k++)
					tmp += (cs[j].desc[k] - gim[i].desc[k]) * (cs[j].desc[k] - gim[i].desc[k]);
				if (dist_i == -1 || tmp < dist)
				{
					dist_i = j;
					dist = tmp;
				}
			}
			total_dist += dist;
			if (gim[i].label != dist_i)
			{
				gim[i].label = dist_i;
				f = 1;
			}
		}
		printf("epoch %d, %f\n", epoch, total_dist);
		if (!f)
			break;
		for (i = 0; i < KMEANS_GABOR_SIZE; i++)
		{
			cs[i].label = 0;
			for (j = 0; j < 48; j++)
				cs[i].desc[j] = 0;
		}
		for (i = 0; i < imf->rows * imf->cols; i++)
		{
			k = gim[i].label;
			for (j = 0; j < 48; j++)
				cs[k].desc[j] += gim[i].desc[j];
			cs[k].label++;
		}
		for (i = 0; i < KMEANS_GABOR_SIZE; i++)
			for (j = 0; j < 48; j++)
				cs[i].desc[j] = cs[i].desc[j] / (float)cs[i].label;
	}
	*/
	FILE* fi = fopen("i.tex", "r");
	for (i = 0; i < imf->rows * imf->cols; i++)
		fscanf(fi, "%d", &gim[i].label);
	fclose(fi);
	/*
	FILE* fi = fopen("i.tex", "w+");
	for (i = 0; i < imf->rows * imf->cols; i++)
		fprintf(fi, "%d\n", gim[i].label);
	fclose(fi);
	*/
	int tex_rows = imf->rows - 10;
	int tex_cols = imf->cols - 10;
	rich_float_pixel_t* tex = (rich_float_pixel_t*)malloc(tex_rows * tex_cols * sizeof(rich_float_pixel_t));
	int x, y;
	for (i = 0; i < tex_rows; i++)
		for (j = 0; j < tex_cols; j++)
		{
			for (k = 0; k < KMEANS_GABOR_SIZE; k++)
				tex[i * tex_cols + j].desc[k] = 0;
			for (y = i; y <= i + 10; y++)
				for (x = j; x <= j + 10; x++)
					tex[i * tex_cols + j].desc[gim[y * imf->cols + x].label]++;
		}

	rich_float_pixel_t tcs[KMEANS_TEXTON_SIZE];

	for (i = 0; i < KMEANS_TEXTON_SIZE; i++)
	{
		k = (int)(gsl_rng_uniform(r) * tex_rows * tex_cols);
		tcs[i].label = k;
		memcpy(tcs[i].desc, tex[k].desc, KMEANS_GABOR_SIZE * sizeof(float));
	}
	epoch = 0;
	for (;; epoch++)
	{
		int f = 0;
		double total_dist = 0;
		for (i = 0; i < tex_rows * tex_cols; i++)
		{
			double dist = 1e16; int dist_i = -1;
			for (j = 0; j < KMEANS_TEXTON_SIZE; j++)
			{
				double tmp = 0;
				for (k = 0; k < KMEANS_GABOR_SIZE; k++)
					tmp += (tcs[j].desc[k] - tex[i].desc[k]) * (tcs[j].desc[k] - tex[i].desc[k]);
				if (dist_i == -1 || tmp < dist)
				{
					dist_i = j;
					dist = tmp;
				}
			}
			total_dist += dist;
			if (tex[i].label != dist_i)
			{
				tex[i].label = dist_i;
				f = 1;
			}
		}
		printf("epoch %d, %f\n", epoch, total_dist);
		if (!f)
			break;
		for (i = 0; i < KMEANS_TEXTON_SIZE; i++)
		{
			tcs[i].label = 0;
			for (j = 0; j < KMEANS_GABOR_SIZE; j++)
				tcs[i].desc[j] = 0;
		}
		for (i = 0; i < tex_rows * tex_cols; i++)
		{
			k = tex[i].label;
			for (j = 0; j < KMEANS_GABOR_SIZE; j++)
				tcs[k].desc[j] += tex[i].desc[j];
			tcs[k].label++;
		}
		for (i = 0; i < KMEANS_TEXTON_SIZE; i++)
			for (j = 0; j < KMEANS_GABOR_SIZE; j++)
				tcs[i].desc[j] = tcs[i].desc[j] / (float)tcs[i].label;
	}
	ccv_dense_matrix_t* cl = ccv_dense_matrix_new(tex_rows, tex_cols, CCV_8U | CCV_C3, NULL, NULL);
	k = 0;
	for (i = 0; i < tex_rows; i++)
		for (j = 0; j < tex_cols; j++)
		{
			cl->data.ptr[i * cl->step + j * 3] = colors[(tex[k].label % 8) * 3];
			cl->data.ptr[i * cl->step + j * 3 + 1] =  colors[(tex[k].label % 8) * 3 + 1];
			cl->data.ptr[i * cl->step + j * 3 + 2] =  colors[(tex[k].label % 8) * 3 + 2];
			k++;
		}
	/*
	ccv_dense_matrix_t* cl = ccv_dense_matrix_new(imf->rows, imf->cols, CCV_8U | CCV_C3, NULL, NULL);
	k = 0;
	for (i = 0; i < imf->rows; i++)
		for (j = 0; j < imf->cols; j++)
		{
			cl->data.ptr[i * cl->step + j * 3] = colors[(gim[k].label % 8) * 3];
			cl->data.ptr[i * cl->step + j * 3 + 1] =  colors[(gim[k].label % 8) * 3 + 1];
			cl->data.ptr[i * cl->step + j * 3 + 2] =  colors[(gim[k].label % 8) * 3 + 2];
			k++;
		}
	*/
	int len, quality = 95;
	// ccv_serialize(cl, argv[2], &len, CCV_SERIAL_JPEG_FILE, &quality);
	ccv_serialize(cl, argv[2], &len, CCV_SERIAL_PNG_FILE, &quality);

	gsl_rng_free(r);
	return 0;
}
