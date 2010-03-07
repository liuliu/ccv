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
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = NULL;
	unsigned int elapsed_time = get_current_time();
	ccv_sample_down(image, &x);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	int len;
	ccv_serialize(x, argv[2], &len, CCV_SERIAL_JPEG_FILE, NULL);
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_garbage_collect();
	return 0;
}
