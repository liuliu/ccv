#include "ccv.h"
#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

double ccv_trace(ccv_matrix_t* mat)
{
	return 0;
}

double ccv_norm(ccv_matrix_t* mat, int type)
{
	return 0;
}

double ccv_sum(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	double sum = 0;
	unsigned char* m_ptr = dmt->data.ptr;
	int i, j;
#define for_block(dummy, __for_get) \
	for (i = 0; i < dmt->rows; i++) \
	{ \
		for (j = 0; j < dmt->cols; j++) \
			sum += __for_get(m_ptr, j); \
		m_ptr += dmt->step; \
	}
	ccv_matrix_getter(dmt->type, for_block);
#undef for_block
	return sum;
}

void ccv_zero(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	memset(dmt->data.ptr, 0, dmt->step * dmt->rows);
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dc = (c == 0) ? 0 : ccv_get_dense_matrix(c);

	assert(da->type == db->type && ((transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols) == ((transpose & CCV_B_TRANSPOSE) ? db->cols : db->rows));

	if (dc != 0)
		assert(dc->type == da->type && ((transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows) == dc->rows && ((transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols) == dc->cols);

	char identifier[20];
	memset(identifier, 0, 20);
	sprintf(identifier, "ccv_gemm(%d)", transpose);
	uint64_t sig = (dc == 0) ? ((da->sig == 0 || db->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, 0)) : ((da->sig == 0 || db->sig == 0 || dc->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, dc->sig, 0));

	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, (transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows, (transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols, da->type, da->type, sig);

	if (dd != dc)
		memcpy(dd->data.ptr, dc->data.ptr, dc->step * dc->rows);

#ifdef HAVE_CBLAS
	switch (CCV_GET_DATA_TYPE(dd->type))
	{
		case CCV_32F:
			cblas_sgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, da->cols, alpha, da->data.fl, da->cols, db->data.fl, db->cols, beta, dd->data.fl, dd->cols);
			break;
		case CCV_64F:
			cblas_dgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, (transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols, alpha, da->data.db, da->cols, db->data.db, db->cols, beta, dd->data.db, dd->cols);
			break;
	}
#endif
}
