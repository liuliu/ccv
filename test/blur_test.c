#include "ccv.h"
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static double __ccv_gaussian_kernel(double x, double y, void* data)
{
	double sigma = *(double*)data;
	return exp(-(x * x + y * y) / (2 * sigma * sigma));
}

static void __ccv_gaussian_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int filter_size, double sigma)
{
	ccv_dense_matrix_t* kernel = ccv_dense_matrix_new(filter_size, filter_size, CCV_32F | CCV_C1, 0, 0);
	ccv_filter_kernel(kernel, __ccv_gaussian_kernel, &sigma);
	float total = ccv_sum(kernel);
	int i, j;
	for (i = 0; i < kernel->rows; i++)
		for (j = 0; j < kernel->cols; j++)
			kernel->data.fl[i * kernel->cols + j] = kernel->data.fl[i * kernel->cols + j] / total;
	ccv_filter(a, kernel, (ccv_matrix_t**)b);
	ccv_matrix_free(kernel);
}

static int __ccv_filter_size(double sigma)
{
	int fsz = (int)(5 * sigma);
	// kernel size must be odd
	if(fsz % 2 == 0)
		fsz++;
	// kernel size cannot be smaller than 3
	if(fsz < 3)
		fsz = 3;
   return fsz;
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* x = 0;
	unsigned int elapsed_time = get_current_time();
	double sigma = 10;
	ccv_blur(image, &x, sigma);
	printf("ccv_blur elpased time : %d\n", get_current_time() - elapsed_time);
	elapsed_time = get_current_time();
	ccv_dense_matrix_t* y = 0;
	__ccv_gaussian_blur(image, &y, __ccv_filter_size(sigma), sigma);
	printf("ccv_filter elpased time : %d\n", get_current_time() - elapsed_time);
	int len;
	ccv_serialize(x, argv[2], &len, CCV_SERIAL_JPEG_FILE, 0);
	ccv_serialize(y, argv[3], &len, CCV_SERIAL_JPEG_FILE, 0);
	ccv_matrix_free(image);
	ccv_matrix_free(x);
	ccv_matrix_free(y);
	ccv_garbage_collect();
	return 0;
}

