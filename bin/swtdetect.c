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
	ccv_swt_param_t params = { .size = 5, .low_thresh = 75, .high_thresh = 75 * 3 };
	ccv_array_t* words = ccv_swt_detect_words(image, params);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, 0, 0);
	ccv_zero(imx);
	int i, j;
	for (i = 0; i < words->rnum; i++)
	{
		ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
		for (j = rect->x; j < rect->x + rect->width; j++)
			imx->data.ptr[j + rect->y * imx->step] = imx->data.ptr[j + (rect->y + rect->height - 1) * imx->step] = 255;
		for (j = rect->y; j < rect->y + rect->height; j++)
			imx->data.ptr[rect->x + j * imx->step] = imx->data.ptr[rect->x + rect->width - 1 + j * imx->step] = 255;
	}
	ccv_serialize(imx, argv[2], 0, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(image);
	ccv_matrix_free(imx);
	ccv_garbage_collect();
	return 0;
}

