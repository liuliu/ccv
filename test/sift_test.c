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
	unsigned int elapsed_time = get_current_time();
	ccv_sift_param_t param;
	param.noctaves = 3;
	param.nlevels = 5;
	param.sigma = 1.2;
	ccv_sift(image, param);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

