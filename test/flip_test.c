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
	unsigned int elapsed_time = get_current_time();
	ccv_flip(image, NULL, CCV_FLIP_Y);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	int len, quality = 95;
	ccv_serialize(image, argv[2], &len, CCV_SERIAL_JPEG_FILE, &quality);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}
