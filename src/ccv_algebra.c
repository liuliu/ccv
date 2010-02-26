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

	ccv_dense_matrix_t* dd;

	int sig[5];
	char identifier[20];
	memset(identifier, 0, 20);
	sprintf(identifier, "ccv_gemm(%d)", transpose);
	ccv_matrix_generate_signature(identifier, 16, sig, da->sig, db->sig, NULL);
	if (*d == NULL)
	{
		*d = dd = ccv_dense_matrix_new(da->rows, db->cols, da->type, NULL, sig);
		if (dd->type & CCV_GARBAGE)
		{
			dd->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		dd = ccv_get_dense_matrix(*d);
		memcpy(dd->sig, sig, 20);
	}

	assert(db->type == dd->type && db->rows == dd->rows && db->cols == dd->cols);

	cblas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, da->data.db, db->data.db, 0.0, dd->data.db);
}
