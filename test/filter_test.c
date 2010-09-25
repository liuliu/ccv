#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(image->rows, image->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(5, 5, CCV_32F | CCV_C1, 0, 0);
	int i, j;
	for (i = 0; i < image->rows; i++)
		for (j = 0; j < image->cols; j++)
			a->data.fl[i * a->cols + j] = (image->data.ptr[i * image->step + j * 3] * 29 + image->data.ptr[i * image->step + j * 3 + 1] * 61 + image->data.ptr[i * image->step + j * 3 + 2] * 10) / 100;
	double tb = 0;
	for (i = 0; i < b->rows; i++)
		for (j = 0; j < b->cols; j++)
			tb += b->data.fl[i * b->cols + j] = exp(-((i - b->rows / 2) * (i - b->rows / 2) + (j - b->cols / 2) * (j - b->cols / 2)) / 10);
	for (i = 0; i < b->rows; i++)
		for (j = 0; j < b->cols; j++)
			b->data.fl[i * b->cols + j] /= tb;
	ccv_dense_matrix_t* x = 0;
	ccv_filter(a, b, (ccv_matrix_t**)&x, 0);
	
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(x->rows, x->cols, CCV_8U | CCV_C1, 0, 0);
	for (i = 0; i < x->rows; i++)
		for (j = 0; j < x->cols; j++)
			imx->data.ptr[i * imx->step + j] = ccv_clamp((int)x->data.fl[i * x->cols + j], 0, 255);
	
	int len;
	ccv_serialize(imx, argv[2], &len, CCV_SERIAL_JPEG_FILE, 0);
	ccv_matrix_free(image);
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(x);
	ccv_matrix_free(imx);
	ccv_garbage_collect();
	return 0;
}
