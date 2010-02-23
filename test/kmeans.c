#include "../src/ccv.h"

double gabor_kernel(double x, double y, void* data)
{
	int* so = (int*)data;
	double kv = 3.141592654 * exp(-0.34657359 * (so[0] + 2));
	double qu = orientation * 3.141592654 / so[1];
	double cos_qu = cos(qu);
	double sin_qu = sin(qu);
	double kv_kv_si = kv * kv * 0.050660592;
	double exp_kv_x_y = exp(-kv * kv * (x * x + y * y) * 0.025330296);
	double kv_qu_x_y = kv * (cos_qu * x + sin_qu * y);
	return exp_kv_x_y * (cos(kv_qu_x_y) - 0.000051723) * kv_kv_si;
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* im = NULL;
	ccv_unserialize(argv[1], &im, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* kernel = ccv_dense_matrix_new(48, 48, CCV_32F | CCV_C1, NULL, NULL);
	int i, j;
	for (i = 0; i < 8; i++)
		for (j = 0; j < 6; j++)
		{
			int so[2] = { i, j };
			ccv_filter_kernel(kernel, gabor_kernel, so);
			ccv_dense_matrix* out;
			ccv_filter(im, kernel, &out);
		}
	return 0;
}
