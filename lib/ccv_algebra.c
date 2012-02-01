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

double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int l_type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	assert(CCV_GET_CHANNEL(da->type) == CCV_C1);
	char identifier[20];
	memset(identifier, 0, 20);
	snprintf(identifier, 20, "ccv_normalize(%d)", l_type);
	uint64_t sig = (da->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, 0);
	btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_C1 : CCV_GET_DATA_TYPE(btype) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_C1, btype, sig);
	ccv_cache_return(db, 0);
	double sum = 0, inv;
	int i, j;
	unsigned char* a_ptr = da->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	switch (l_type)
	{
		case CCV_L1_NORM:
#define for_block(_for_set, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					sum += _for_get(a_ptr, j, 0); \
				a_ptr += da->step; \
			} \
			inv = 1.0 / sum; \
			a_ptr = da->data.ptr; \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr, j, 0) * inv, 0); \
				a_ptr += da->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
			break;
		case CCV_L2_NORM:
#define for_block(_for_set, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					sum += _for_get(a_ptr, j, 0) * _for_get(a_ptr, j, 0); \
				a_ptr += da->step; \
			} \
			sum = sqrt(sum); \
			inv = 1.0 / sum; \
			a_ptr = da->data.ptr; \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					_for_set(b_ptr, j, _for_get(a_ptr, j, 0) * inv, 0); \
				a_ptr += da->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter(db->type, ccv_matrix_getter, da->type, for_block);
#undef for_block
			break;
	}
	return sum;
}

double ccv_sum(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	double sum = 0;
	unsigned char* m_ptr = dmt->data.ptr;
	int i, j;
#define for_block(_, _for_get) \
	for (i = 0; i < dmt->rows; i++) \
	{ \
		for (j = 0; j < dmt->cols; j++) \
			sum += _for_get(m_ptr, j, 0); \
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

void ccv_substract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	uint64_t sig = ccv_matrix_generate_signature("ccv_substract", 13, da->sig, db->sig, 0);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_cache_return(dc, );
	int i, j;
	unsigned char* aptr = da->data.ptr;
	unsigned char* bptr = db->data.ptr;
	unsigned char* cptr = dc->data.ptr;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols; j++) \
		{ \
			_for_set(cptr, j, _for_get(aptr, j, 0) - _for_get(bptr, j, 0), 0); \
		} \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dc = (c == 0) ? 0 : ccv_get_dense_matrix(c);

	assert(CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == 1 && CCV_GET_CHANNEL(db->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols) == ((transpose & CCV_B_TRANSPOSE) ? db->cols : db->rows));

	if (dc != 0)
		assert(CCV_GET_DATA_TYPE(dc->type) == CCV_GET_DATA_TYPE(da->type) && CCV_GET_CHANNEL(dc->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows) == dc->rows && ((transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols) == dc->cols);

	char identifier[20];
	memset(identifier, 0, 20);
	snprintf(identifier, 20, "ccv_gemm(%d)", transpose);
	uint64_t sig = (dc == 0) ? ((da->sig == 0 || db->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, 0)) : ((da->sig == 0 || db->sig == 0 || dc->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 20, da->sig, db->sig, dc->sig, 0));
	type = CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, (transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows, (transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols, type, type, sig);
	ccv_cache_return(dd, );

	if (dd != dc && dc != 0)
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
