#include "ccv.h"
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	int i, j;
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = NULL;
	unsigned int elapsed_time = get_current_time();
	ccv_sobel(image, &x, 0, 1);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(x->rows, x->cols, CCV_8U | CCV_C1, NULL, NULL);
	for (i = 0; i < x->rows; i++)
		for (j = 0; j < x->cols; j++)
			imx->data.ptr[i * imx->step + j] = ccv_clamp(x->data.i[i * x->cols + j] / 4, 0, 255);
	int len;
	ccv_serialize(imx, argv[2], &len, CCV_SERIAL_JPEG_FILE, NULL);
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_matrix_free(imx);
	ccv_garbage_collect();
	return 0;
}

