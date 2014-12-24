#include "ccv.h"
#include "ccv_internal.h"
#if HAVE_ACCELERATE_FRAMEWORK
#include <Accelerate/Accelerate.h>
#elif HAVE_CBLAS
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

double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int flag)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	assert(CCV_GET_CHANNEL(da->type) == CCV_C1);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(20, "ccv_normalize(%d)", flag), da->sig, CCV_EOF_SIGN);
	btype = (btype == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_C1 : CCV_GET_DATA_TYPE(btype) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_C1, btype, sig);
	assert(db);
	ccv_object_return_if_cached(db->tag.f64, db);
	double sum = 0, inv;
	int i, j;
	unsigned char* a_ptr = da->data.u8;
	unsigned char* b_ptr = db->data.u8;
	switch (flag)
	{
		case CCV_L1_NORM:
#define for_block(_for_set, _for_get) \
			for (i = 0; i < da->rows; i++) \
			{ \
				for (j = 0; j < da->cols; j++) \
					sum += fabs((double)_for_get(a_ptr, j, 0)); \
				a_ptr += da->step; \
			} \
			inv = 1.0 / sum; \
			a_ptr = da->data.u8; \
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
			a_ptr = da->data.u8; \
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
	return db->tag.f64 = sum;
}

void ccv_sat(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int padding_pattern)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(20, "ccv_sat(%d)", padding_pattern), a->sig, CCV_EOF_SIGN);
	int safe_type = (a->type & CCV_8U) ? ((a->rows * a->cols >= 0x808080) ? CCV_64S : CCV_32S) : ((a->type & CCV_32S) ? CCV_64S : a->type);
	type = (type == 0) ? CCV_GET_DATA_TYPE(safe_type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	int ch = CCV_GET_CHANNEL(a->type);
	int i, j;
	unsigned char* a_ptr = a->data.u8;
	ccv_dense_matrix_t* db;
	unsigned char* b_ptr;
	switch (padding_pattern)
	{
		case CCV_NO_PADDING:
			db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
			ccv_object_return_if_cached(, db);
			b_ptr = db->data.u8;
#define for_block(_for_set_b, _for_get_b, _for_get) \
			for (j = 0; j < ch; j++) \
				_for_set_b(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			for (j = ch; j < a->cols * ch; j++) \
				_for_set_b(b_ptr, j, _for_get_b(b_ptr, j - ch, 0) + _for_get(a_ptr, j, 0), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
			for (i = 1; i < a->rows; i++) \
			{ \
				for (j = 0; j < ch; j++) \
					_for_set_b(b_ptr, j, _for_get_b(b_ptr - db->step, j, 0) + _for_get(a_ptr, j, 0), 0); \
				for (j = ch; j < a->cols * ch; j++) \
					_for_set_b(b_ptr, j, _for_get_b(b_ptr, j - ch, 0) - _for_get_b(b_ptr - db->step, j - ch, 0) + _for_get_b(b_ptr - db->step, j, 0) + _for_get(a_ptr, j, 0), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter_getter(db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
			break;
		case CCV_PADDING_ZERO:
			db = *b = ccv_dense_matrix_renew(*b, a->rows + 1, a->cols + 1, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
			ccv_object_return_if_cached(, db);
			b_ptr = db->data.u8;
#define for_block(_for_set_b, _for_get_b, _for_get) \
			for (j = 0; j < db->cols * ch; j++) \
				_for_set_b(b_ptr, j, 0, 0); \
			b_ptr += db->step; \
			for (i = 0; i < a->rows; i++) \
			{ \
				for (j = 0; j < ch; j++) \
					_for_set_b(b_ptr, j, 0, 0); \
				for (j = ch; j < db->cols * ch; j++) \
					_for_set_b(b_ptr, j, _for_get_b(b_ptr, j - ch, 0) - _for_get_b(b_ptr - db->step, j - ch, 0) + _for_get_b(b_ptr - db->step, j, 0) + _for_get(a_ptr, j - ch, 0), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			}
			ccv_matrix_setter_getter(db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
			break;
	}
}

double ccv_sum(ccv_matrix_t* mat, int flag)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	double sum = 0;
	unsigned char* m_ptr = dmt->data.u8;
	int i, j, ch = CCV_GET_CHANNEL(dmt->type);
#define for_block(_, _for_get) \
	switch (flag) \
	{ \
		case CCV_UNSIGNED: \
			for (i = 0; i < dmt->rows; i++) \
			{ \
				for (j = 0; j < dmt->cols * ch; j++) \
					sum += fabs((double)(_for_get(m_ptr, j, 0))); \
				m_ptr += dmt->step; \
			} \
			break; \
		case CCV_SIGNED: \
		default: \
			for (i = 0; i < dmt->rows; i++) \
			{ \
				for (j = 0; j < dmt->cols * ch; j++) \
					sum += _for_get(m_ptr, j, 0); \
				m_ptr += dmt->step; \
			} \
	}
	ccv_matrix_getter(dmt->type, for_block);
#undef for_block
	return sum;
}

double ccv_variance(ccv_matrix_t* mat)
{
	ccv_dense_matrix_t* dmt = ccv_get_dense_matrix(mat);
	double mean = 0, variance = 0;
	unsigned char* m_ptr = dmt->data.u8;
	int i, j, ch = CCV_GET_CHANNEL(dmt->type);
#define for_block(_, _for_get) \
	for (i = 0; i < dmt->rows; i++) \
	{ \
		for (j = 0; j < dmt->cols * ch; j++) \
		{ \
			mean += _for_get(m_ptr, j, 0); \
			variance += _for_get(m_ptr, j, 0) * _for_get(m_ptr, j, 0); \
		} \
		m_ptr += dmt->step; \
	}
	ccv_matrix_getter(dmt->type, for_block);
#undef for_block
	mean = mean / (dmt->rows * dmt->cols * ch);
	variance = variance / (dmt->rows * dmt->cols * ch);
	return variance - mean * mean;
}

void ccv_multiply(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	ccv_declare_derived_signature(sig, da->sig != 0 && db->sig != 0, ccv_sign_with_literal("ccv_multiply"), da->sig, db->sig, CCV_EOF_SIGN);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_object_return_if_cached(, dc);
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
	unsigned char* cptr = dc->data.u8;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols * ch; j++) \
			_for_set(cptr, j, _for_get(aptr, j, 0) * _for_get(bptr, j, 0), 0); \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_add(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	ccv_declare_derived_signature(sig, da->sig != 0 && db->sig != 0, ccv_sign_with_literal("ccv_add"), da->sig, db->sig, CCV_EOF_SIGN);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_object_return_if_cached(, dc);
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
	unsigned char* cptr = dc->data.u8;
#define for_block(_for_get_a, _for_get_b, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols * ch; j++) \
			_for_set(cptr, j, _for_get_a(aptr, j, 0) + _for_get_b(bptr, j, 0), 0); \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter_a(da->type, ccv_matrix_getter_b, db->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_subtract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	assert(da->rows == db->rows && da->cols == db->cols && CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == CCV_GET_CHANNEL(db->type));
	ccv_declare_derived_signature(sig, da->sig != 0 && db->sig != 0, ccv_sign_with_literal("ccv_subtract"), da->sig, db->sig, CCV_EOF_SIGN);
	int no_8u_type = (da->type & CCV_8U) ? CCV_32S : da->type;
	type = (type == 0) ? CCV_GET_DATA_TYPE(no_8u_type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dc = *c = ccv_dense_matrix_renew(*c, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_object_return_if_cached(, dc);
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
	unsigned char* cptr = dc->data.u8;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < da->rows; i++) \
	{ \
		for (j = 0; j < da->cols * ch; j++) \
			_for_set(cptr, j, _for_get(aptr, j, 0) - _for_get(bptr, j, 0), 0); \
		aptr += da->step; \
		bptr += db->step; \
		cptr += dc->step; \
	}
	ccv_matrix_getter(da->type, ccv_matrix_setter, dc->type, for_block);
#undef for_block
}

void ccv_scale(ccv_matrix_t* a, ccv_matrix_t** b, int type, double ds)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_declare_derived_signature(sig, da->sig != 0, ccv_sign_with_format(20, "ccv_scale(%la)", ds), da->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, da->rows, da->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(da->type), type, sig);
	ccv_object_return_if_cached(, db);
	int i, j, ch = CCV_GET_CHANNEL(da->type);
	unsigned char* aptr = da->data.u8;
	unsigned char* bptr = db->data.u8;
	if (CCV_GET_DATA_TYPE(da->type) == CCV_8U && CCV_GET_DATA_TYPE(db->type) == CCV_8U) // specialize for 8U type
	{
		unsigned char us[256];
		for (i = 0; i < 256; i++)
			us[i] = ccv_clamp(ds * i, 0, 255);
		for (i = 0; i < da->rows; i++)
		{
			for (j = 0; j < da->cols * ch; j++)
				bptr[j] = us[aptr[j]];
			aptr += da->step;
			bptr += db->step;
		}
	} else {
#define for_block(_for_get, _for_set) \
		for (i = 0; i < da->rows; i++) \
		{ \
			for (j = 0; j < da->cols * ch; j++) \
				_for_set(bptr, j, ds * _for_get(aptr, j, 0), 0); \
			aptr += da->step; \
			bptr += db->step; \
		}
		ccv_matrix_getter(da->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
	}
}

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d, int type)
{
	ccv_dense_matrix_t* da = ccv_get_dense_matrix(a);
	ccv_dense_matrix_t* db = ccv_get_dense_matrix(b);
	ccv_dense_matrix_t* dc = (c == 0) ? 0 : ccv_get_dense_matrix(c);

	assert(CCV_GET_DATA_TYPE(da->type) == CCV_GET_DATA_TYPE(db->type) && CCV_GET_CHANNEL(da->type) == 1 && CCV_GET_CHANNEL(db->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols) == ((transpose & CCV_B_TRANSPOSE) ? db->cols : db->rows));

	if (dc != 0)
		assert(CCV_GET_DATA_TYPE(dc->type) == CCV_GET_DATA_TYPE(da->type) && CCV_GET_CHANNEL(dc->type) == 1 && ((transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows) == dc->rows && ((transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols) == dc->cols);

	ccv_declare_derived_signature_case(sig, ccv_sign_with_format(20, "ccv_gemm(%d)", transpose), ccv_sign_if(dc == 0 && da->sig != 0 && db->sig != 0, da->sig, db->sig, CCV_EOF_SIGN), ccv_sign_if(dc != 0 && da->sig != 0 && db->sig != 0 && dc->sig != 0, da->sig, db->sig, dc->sig, CCV_EOF_SIGN));
	type = CCV_GET_DATA_TYPE(da->type) | CCV_GET_CHANNEL(da->type);
	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, (transpose & CCV_A_TRANSPOSE) ? da->cols : da->rows, (transpose & CCV_B_TRANSPOSE) ? db->rows : db->cols, type, type, sig);
	ccv_object_return_if_cached(, dd);

	if (dd != dc && dc != 0)
		memcpy(dd->data.u8, dc->data.u8, dc->step * dc->rows);
	else if (dc == 0) // clean up dd if dc is not provided
		memset(dd->data.u8, 0, dd->step * dd->rows);

#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	switch (CCV_GET_DATA_TYPE(dd->type))
	{
		case CCV_32F:
			cblas_sgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, (transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols, alpha, da->data.f32, da->cols, db->data.f32, db->cols, beta, dd->data.f32, dd->cols);
			break;
		case CCV_64F:
			cblas_dgemm(CblasRowMajor, (transpose & CCV_A_TRANSPOSE) ? CblasTrans : CblasNoTrans, (transpose & CCV_B_TRANSPOSE) ? CblasTrans : CblasNoTrans, dd->rows, dd->cols, (transpose & CCV_A_TRANSPOSE) ? da->rows : da->cols, alpha, da->data.f64, da->cols, db->data.f64, db->cols, beta, dd->data.f64, dd->cols);
			break;
	}
#else
	assert(0 && "You need a BLAS compatible library for this function, e.g. libatlas.");
#endif
}
