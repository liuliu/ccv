#include "ccv.h"
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, 0, 0);
	int i, j, k, t;
	for (i = 0; i < image->rows; i++)
		for (j = 0; j < image->cols; j++)
			a->data.ptr[i * a->step + j] = (image->data.ptr[i * image->step + j * 3] * 29 + image->data.ptr[i * image->step + j * 3 + 1] * 61 + image->data.ptr[i * image->step + j * 3 + 2] * 10) / 100;
	ccv_dense_matrix_t* x = 0;
	ccv_daisy_param_t param;
	param.radius = 15;
	param.rad_q_no = 3;
	param.th_q_no = 8;
	param.hist_th_q_no = 8;
	param.normalize_threshold = 0.154;
	param.normalize_method = CCV_DAISY_NORMAL_PARTIAL;
	unsigned int elapsed_time = get_current_time();
	ccv_daisy(a, &x, 0, param);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	j = atoi(argv[2]);
	i = atoi(argv[3]);
	float* x_ptr = x->data.fl + i * x->cols + j * 200;
	for (k = 0; k < 25; k++)
	{
		for (t = 0; t < 8; t++)
			printf("%f ", x_ptr[k * 8 + t]);
		printf("\n");
	}
	ccv_matrix_free(image);
	ccv_matrix_free(a);
	ccv_matrix_free(x);
	ccv_garbage_collect();
	return 0;
}
