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
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	unsigned int elapsed_time = get_current_time();
	ccv_swt_param_t params = { .size = 5, .low_thresh = 100, .high_thresh = 100 * 3, .max_height = 300, .min_height = 10, .aspect_ratio = 10, .variance_ratio = 1, .thickness_ratio = 2, .height_ratio = 2, .intensity_thresh = 15, .distance_ratio = 3, .intersect_ratio = 2, .letter_thresh = 3, .breakdown = 0, .breakdown_ratio = 1 };
	ccv_array_t* words = ccv_swt_detect_words(image, params);
	elapsed_time = get_current_time() - elapsed_time;
	int i;
	/*
	int i, j;
	ccv_dense_matrix_t* x = 0;
	params.direct = 1;
	ccv_swt(image, &x, 0, params);
	ccv_dense_matrix_t* imx = 0;
	ccv_shift(x, &imx, CCV_8U | CCV_C1, 0, 0);
	for (i = 0; i < x->rows; i++)
		for (j = 0; j < x->cols; j++)
			if (imx->data.ptr[j + i * imx->step] != 0)
				imx->data.ptr[j + i * imx->step] = ccv_clamp(255 - imx->data.ptr[j + i * imx->step] * 10, 0, 255);
	ccv_serialize(imx, argv[2], 0, CCV_SERIAL_PNG_FILE, 0);
	*/
	for (i = 0; i < words->rnum; i++)
	{
		ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
		printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
	}
	printf("total : %d in time %dms\n", words->rnum, elapsed_time);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

