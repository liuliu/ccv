#include "ccv.h"

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

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(1, 2, CCV_64F | CCV_C1, NULL, NULL);
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
	printf("step(s) : %d\n", steps);
	int i = 0;
	for (i = 0; i < 2; i++)
		printf("%f\n", x->data.db[i]);
	return 0;
}
