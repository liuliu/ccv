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
	int i, j;
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	unsigned int elapsed_time = get_current_time();
	ccv_canny(image, &x, 0, 5, 75, 75 * 3);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	for (i = 0; i < x->rows; i++)
		for (j = 0; j < x->cols; j++)
			x->data.ptr[i * x->step + j] *= 255;
	ccv_serialize(x, argv[2], 0, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
	return 0;
}

