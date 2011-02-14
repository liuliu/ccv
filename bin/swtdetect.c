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
	ccv_swt_param_t params = { .size = 5, .low_thresh = 50, .high_thresh = 50 * 3 };
	ccv_array_t* words = ccv_swt_detect_words(image, params);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	int i, j;
	/*
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
		for (j = rect->x; j < rect->x + rect->width; j++)
			image->data.ptr[j + rect->y * image->step] = image->data.ptr[j + (rect->y + rect->height - 1) * image->step] = 255;
		for (j = rect->y; j < rect->y + rect->height; j++)
			image->data.ptr[rect->x + j * image->step] = image->data.ptr[rect->x + rect->width - 1 + j * image->step] = 255;
	}
	ccv_serialize(image, argv[2], 0, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

