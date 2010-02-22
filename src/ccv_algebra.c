#include "ccv.h"
#include <cblas.h>

double ccv_trace(ccv_matrix_t* mat)
{
}

double ccv_norm(ccv_matrix_t* mat, int type)
{
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* c, int transpose, ccv_matrix_t** d)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
}
