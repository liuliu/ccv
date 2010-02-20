#include "ccv.h"
#include <rfftw.h>

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

void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** x)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dx;
	if (*x == NULL)
		*x = dx = ccv_dense_matrix_new(da->rows, da->cols, da->type);
	else
		dx = ccv_get_dense_matrix(*x);
	fftw_real* fftw_a = fftw_malloc(da->rows * 2 * (da->cols / 2 + 1) * sizeof(fftw_real));
	fftw_real* fftw_b = fftw_malloc(db->rows * 2 * (db->cols / 2 + 1) * sizeof(fftw_real));
	fftw_real* fftw_x = fftw_malloc(dx->rows * dx->cols * sizeof(fftw_real));
	fftw_complex* fftw_ac = (fftw_complex*)fftw_a;
	fftw_complex* fftw_bc = (fftw_complex*)fftw_b;
	fftw_complex* fftw_xc = fftw_malloc(dx->rows * (dx->cols / 2 + 1) * sizeof(fftw_complex));
	rfftwnd_plan p, pinv;
	fftw_real scale = 1.0 / (da->rows * da->cols);
	p = rfftw2d_create_plan(M, N, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE | FFTW_IN_PLACE);
	pinv = rfftw2d_create_plan(M, N, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);

	/* aliases for accessing complex transform outputs: */
	int i, j;
	float* flap = da->data.fl;
	fftw_real* fftw_ap = fftw_a;
	for (i = 0; i < da->rows; i++)
	{
		for (j = 0; j < da->cols; j++)
		{
			*fftw_ap = *flap;
			flap++;
			fftw_ap++;
		}
		fftw_ap += da->cols & 1;
	}
	float* flbp = db->data.fl;
	fftw_real* fftw_bp = fftw_b;
	for (i = 0; i < db->rows; i++)
	{
		for (j = 0; j < db->cols; j++)
		{
			*fftw_bp = *flbp;
			flbp++;
			fftw_bp++;
		}
		fftw_bp += db->cols & 1;
	}

	rfftwnd_one_real_to_complex(p, fftw_a, NULL);
	rfftwnd_one_real_to_complex(p, fftw_b, NULL);

	fftw_complex* fftw_xcp = fftw_xc;
	for (i = 0; i < dx->rows; i++)
		for (j = 0; j < dx->cols / 2 + 1; j++)
		{
			int ij = i * (dx->cols / 2 + 1) + j;
			fftw_xcp->re = (fftw_ac[ij].re * fftw_bc[ij].re - fftw_ac[ij].im * fftw_bc[ij].im) * scale;
			fftw_xcp->im = (fftw_ac[ij].re * fftw_bc[ij].im + fftw_ac[ij].im * fftw_bc[ij].re) * scale;
		}

	rfftwnd_one_complex_to_real(pinv, fftw_xc, fftw_x);
	rfftwnd_destroy_plan(p);
	rfftwnd_destroy_plan(pinv);
	fftw_free(fftw_a);
	fftw_free(fftw_b);
	fftw_free(fftw_xc);
	for (i = 0; i < dx->rows * dx->cols; i++)
		dx->data.fl[i] = fftw_x[i];
	fftw_free(fftw_x);
}
