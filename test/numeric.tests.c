#include "ccv.h"
#include "case.h"
#include "ccv_case.h"

int rosenbrock(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void* data)
{
	int* steps = (int*)data;
	(*steps)++;
	int i;
	double rf = 0;
	double* x_vec = x->data.db;
	for (i = 0; i < 1; i++)
		rf += 100 * (x_vec[i + 1] - x_vec[i] * x_vec[i]) * (x_vec[i + 1] - x_vec[i] * x_vec[i]) + (1 - x_vec[i]) * (1 - x_vec[i]);
	*f = rf;
	double* df_vec = df->data.db;
	ccv_zero(df);
	df_vec[0] = df_vec[1] = 0;
	for (i = 0; i < 1; i++)
		df_vec[i] = -400 * x_vec[i] * (x_vec[i+1] - x_vec[i] * x_vec[i]) - 2 * (1 - x_vec[i]);
	for (i = 1; i < 2; i++)
		df_vec[i] += 200 * (x_vec[i] - x_vec[i - 1] * x_vec[i - 1]);
	return 0;
}

TEST_CASE("minimize rosenbrock")
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(1, 2, CCV_64F | CCV_C1, 0, 0);
	ccv_zero(x);
	int steps = 0;
	ccv_minimize_param_t params;
	params.interp = 0.1;
	params.extrap = 3.0;
	params.max_iter = 20;
	params.ratio = 10.0;
	params.sig = 0.1;
	params.rho = 0.05;
	ccv_minimize(x, 25, 1.0, rosenbrock, params, &steps);
	double dx[2] = { 1, 1 };
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, x->data.db, dx, 2, 1e-6, "the global minimal should be at (1.0, 1.0)");
	ccv_matrix_free(x);
}

double gaussian(double x, double y, void* data)
{
	return exp(-(x * x + y * y) / 20) / sqrt(CCV_PI * 20);
}

TEST_CASE("FFTW-based filter on Gaussian kernel")
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize("../samples/nature.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* gray = 0;ccv_dense_matrix_new(image->rows, image->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_shift(image, (ccv_matrix_t**)&gray, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* kernel = ccv_dense_matrix_new(10, 10, CCV_32F | CCV_C1, 0, 0);
	ccv_filter_kernel(kernel, gaussian, 0);
	ccv_normalize(kernel, (ccv_matrix_t**)&kernel, 0, CCV_L1_NORM);
	ccv_dense_matrix_t* x = 0;
	ccv_filter(gray, kernel, (ccv_matrix_t**)&x, 0);
	REQUIRE_MATRIX_FILE_EQ(x, "data/nature.filter.bin", "should apply Gaussian filter with FFTW on nature.png with sigma = sqrt(10)");
	ccv_matrix_free(image);
	ccv_matrix_free(gray);
	ccv_matrix_free(kernel);
	ccv_matrix_free(x);
}

#include "case_main.h"
