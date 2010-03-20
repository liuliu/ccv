#include "ccv.h"
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_qrng.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

typedef struct {
	int label;
	float desc[200];
} rich_float_pixel_t;

#define DAISY_GRID_SIZE (200)
#define KMEANS_DAISY_SIZE (8)

int colors[] =  { 0, 0, 255, 0, 128, 255, 0, 255, 255, 0, 255, 0, 255, 128, 0, 255, 255, 0, 255, 0, 0, 255, 0, 255, 128, 0, 255, 128, 128, 255, 255, 128, 128, 128, 255, 128 };

int main(int argc, char** argv)
{
	int len, quality = 95;
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, NULL, NULL);
	int i, j, k, t;
	for (i = 0; i < image->rows; i++)
		for (j = 0; j < image->cols; j++)
			a->data.ptr[i * a->step + j] = (image->data.ptr[i * image->step + j * 3] * 29 + image->data.ptr[i * image->step + j * 3 + 1] * 61 + image->data.ptr[i * image->step + j * 3 + 2] * 10) / 100;
	ccv_dense_matrix_t* x = NULL;
	ccv_daisy_param_t param;
	param.radius = 15;
	param.rad_q_no = 3;
	param.th_q_no = 8;
	param.hist_th_q_no = 8;
	param.normalize_threshold = 0.154;
	param.normalize_method = CCV_DAISY_NORMAL_PARTIAL;
	unsigned int elapsed_time = get_current_time();
	ccv_daisy(a, &x, param);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	int* labels = (int*)malloc(a->rows * a->cols * sizeof(int));

	rich_float_pixel_t cs[KMEANS_DAISY_SIZE];

	gsl_rng_env_setup();

	gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(r, 0);
	
	for (i = 0; i < KMEANS_DAISY_SIZE; i++)
	{
		k = (int)(gsl_rng_uniform(r) * a->rows * a->cols);
		cs[i].label = k;
		memcpy(cs[i].desc, x->data.fl + k * DAISY_GRID_SIZE, DAISY_GRID_SIZE * sizeof(float));
	}
	int epoch = 0;

	for (;; epoch++)
	{
		int f = 0;
		double total_dist = 0;
		for (i = 0; i < a->rows * a->cols; i++)
		{
			double dist = 1e16; int dist_i = -1;
			for (j = 0; j < KMEANS_DAISY_SIZE; j++)
			{
				double tmp = 0;
				for (k = 0; k < DAISY_GRID_SIZE; k++)
					tmp += (cs[j].desc[k] - x->data.fl[i * DAISY_GRID_SIZE + k]) * (cs[j].desc[k] - x->data.fl[i * DAISY_GRID_SIZE + k]);
				if (dist_i == -1 || tmp < dist)
				{
					dist_i = j;
					dist = tmp;
				}
			}
			total_dist += dist;
			if (labels[i] != dist_i)
			{
				labels[i] = dist_i;
				f = 1;
			}
		}
		printf("epoch %d, %f\n", epoch, total_dist);
		if (!f)
			break;
		for (i = 0; i < KMEANS_DAISY_SIZE; i++)
		{
			cs[i].label = 0;
			for (j = 0; j < DAISY_GRID_SIZE; j++)
				cs[i].desc[j] = 0;
		}
		for (i = 0; i < a->rows * a->cols; i++)
		{
			k = labels[i];
			for (j = 0; j < DAISY_GRID_SIZE; j++)
				cs[k].desc[j] += x->data.fl[i * DAISY_GRID_SIZE + j];
			cs[k].label++;
		}
		for (i = 0; i < KMEANS_DAISY_SIZE; i++)
			for (j = 0; j < DAISY_GRID_SIZE; j++)
				cs[i].desc[j] = cs[i].desc[j] / (float)cs[i].label;
	}
	ccv_dense_matrix_t* cl = ccv_dense_matrix_new(a->rows, a->cols, CCV_8U | CCV_C3, NULL, NULL);
	k = 0;
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			cl->data.ptr[i * cl->step + j * 3] = colors[(labels[k] % 8) * 3];
			cl->data.ptr[i * cl->step + j * 3 + 1] =  colors[(labels[k] % 8) * 3 + 1];
			cl->data.ptr[i * cl->step + j * 3 + 2] =  colors[(labels[k] % 8) * 3 + 2];
			k++;
		}
	ccv_serialize(cl, argv[2], &len, CCV_SERIAL_PNG_FILE, &quality);
	gsl_rng_free(r);
	free(labels);
	ccv_matrix_free(image);
	ccv_matrix_free(a);
	ccv_matrix_free(x);
	ccv_garbage_collect();
	return 0;
}

