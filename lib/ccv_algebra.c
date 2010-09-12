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
			sum += __for_get(m_ptr, j, 0); \
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

void ccv_substract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	uint64_t sig = ccv_matrix_generate_signature("ccv_substract", 13, da->sig, db->sig, 0);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), no_8u_type | CCV_GET_CHANNEL(da->type), sig);
	ccv_cache_return(dc, );
	int i, j;
	unsigned char* aptr = da->data.ptr;
	unsigned char* bptr = db->data.ptr;
	unsigned char* cptr = dc->data.ptr;
#define for_block(__for_get, __for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
		{ \
			__for_set(cptr, j, __for_get(aptr, j, 0) - __for_get(bptr, j, 0), 0); \
		} \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dc = (c == 0) ? 0 : ccv_get_dense_matrix(c);

	assert(CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL_NUM(da->type) == 1 && CCV_GET_CHANNEL_NUM(db->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols) == ((transpose & CCV_B_TRANSPOSE) ? db->cols : db->rows));

	if (dc != 0)
		assert(CCV_GET_DATA_TYPE(dc->type) == CCV_GET_DATA_TYPE(da->type) && CCV_GET_CHANNEL_NUM(dc->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows) == dc->rows && ((transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols) == dc->cols);

	char identifier[20];
	memset(identifier, 0, 20);
	snprintf(identifier, 20, "ccv_gemm(%d)", transpose);
	uint64_t sig = (dc == 0) ? ((da->sig == 0 || db->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, 0)) : ((da->sig == 0 || db->sig == 0 || dc->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, dc->sig, 0));

	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, (transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows, (transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols, da->type, da->type, sig);
	ccv_cache_return(dd, );

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
