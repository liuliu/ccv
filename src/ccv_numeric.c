#include "ccv.h"
#include <complex.h>
#include <fftw3.h>

void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b)
{
}

void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** x)
{
}

void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** x)
{
}

void ccv_minimize(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** x)
{
}

static void __ccv_filter_fftw(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* x)
{
	int rows = a->rows, cols = a->cols, cols_2c = 2 * (cols / 2 + 1);
	double* fftw_a = (double*)fftw_malloc(rows * cols_2c * sizeof(double));
	double* fftw_b = (double*)fftw_malloc(rows * cols_2c * sizeof(double));
	memset(fftw_b, 0, rows * cols_2c * sizeof(double));
	double* fftw_x = (double*)fftw_malloc(rows * cols * sizeof(double));
	fftw_complex* fftw_ac = (fftw_complex*)fftw_a;
	fftw_complex* fftw_bc = (fftw_complex*)fftw_b;
	fftw_complex* fftw_xc = (fftw_complex*)fftw_malloc(rows * (cols / 2 + 1) * sizeof(fftw_complex));
	fftw_plan p, pinv;
	double scale = 1.0 / (rows * cols);
	p = fftw_plan_dft_r2c_2d(rows, cols, NULL, NULL, FFTW_ESTIMATE);
	pinv = fftw_plan_dft_c2r_2d(rows, cols, fftw_xc, fftw_x, FFTW_ESTIMATE);

	/* aliases for accessing complex transform outputs: */
	int i, j;
	float* flap = a->data.fl;
	double* fftw_ap = fftw_a;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			*fftw_ap = *flap;
			flap++;
			fftw_ap++;
		}
		fftw_ap += cols & 1;
	}
	/* discrete kernel is always meant to be (0,0) centered, but in most case, it is (0,0) toplefted.
	 * to compensate that, fourier function will assume it is periodic function, which, will result
	 * the following interleaving */
	int rows_bc = rows - b->rows / 2;
	int cols_bc = cols - b->cols / 2;
	for (i = 0; i < b->rows; i++)
		for (j = 0; j < b->cols; j++)
			fftw_b[((i + rows_bc) % rows) * cols_2c + (j + cols_bc) % cols_2c] = b->data.fl[i * b->cols + j];

	fftw_execute_dft_r2c(p, fftw_a, fftw_ac);
	fftw_execute_dft_r2c(p, fftw_b, fftw_bc);
	
	fftw_complex* fftw_acp = fftw_ac;
	fftw_complex* fftw_bcp = fftw_bc;
	fftw_complex* fftw_xcp = fftw_xc;
	for (i = 0; i < rows * (cols / 2 + 1); i++)
	{
		*fftw_xcp = (fftw_acp[0] * fftw_bcp[0]) * scale;
		fftw_acp++;
		fftw_bcp++;
		fftw_xcp++;
	}

	fftw_free(fftw_a);
	fftw_free(fftw_b);

	fftw_execute(pinv);
	fftw_destroy_plan(p);
	fftw_destroy_plan(pinv);
	fftw_free(fftw_xc);
	for (i = 0; i < rows * cols; i++)
		x->data.fl[i] = fftw_x[i];
	fftw_free(fftw_x);
}

void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** x)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dx;
	if (*x == NULL)
		*x = dx = ccv_dense_matrix_new(da->rows, da->cols, da->type, NULL, NULL);
	else
		dx = ccv_get_dense_matrix(*x);
	/* 100 is the constant to indicate the high cost of FFT (even with O(nlog(n)) */
	if (db->rows * db->cols < log(da->rows * da->cols) * 100)
	{
		printf("not implemented\n");
	} else {
		__ccv_filter_fftw(da, db, dx);
	}
}
