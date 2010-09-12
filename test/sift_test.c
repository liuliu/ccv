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
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	unsigned int elapsed_time = get_current_time();
	ccv_sift_param_t param;
	param.noctaves = 3;
	param.nlevels = 6;
	param.edge_threshold = 10;
	param.peak_threshold = 0;
	ccv_sift(image, param);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	image = 0;
	ccv_unserialize("keypoint.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	memset(image->data.ptr, 0, image->step * image->rows);
	FILE* frame = fopen("box.frame", "r");
	float x, y, s0, s1;
	while (fscanf(frame, "%f %f %f %f", &x, &y, &s0, &s1) != EOF)
	{
		image->data.ptr[(int)(y + 0.5) * image->step + (int)(x + 0.5)] = 255;
	}
	fclose(frame);
	int len;
	ccv_serialize(image, "mixkp.png", &len, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

