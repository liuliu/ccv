#include "ccv.h"
#include "ccv_internal.h"

static void _ccv_rgb_to_yuv(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	unsigned char* a_ptr = a->data.u8;
	unsigned char* b_ptr = b->data.u8;
	int i, j;
#define for_block(_for_get, _for_set_b, _for_get_b) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
		{ \
			_for_set_b(b_ptr, j * 3, (_for_get(a_ptr, j * 3, 0) * 1225 + _for_get(a_ptr, j * 3 + 1, 0) * 2404 + _for_get(a_ptr, j * 3 + 2, 0) * 467) / 4096, 0); \
			_for_set_b(b_ptr, j * 3 + 1, (_for_get(a_ptr, j * 3 + 2, 0) - _for_get_b(b_ptr, j * 3, 0)) * 2015 / 4096 + 128, 0); \
			_for_set_b(b_ptr, j * 3 + 2, (_for_get(a_ptr, j * 3, 0) - _for_get_b(b_ptr, j * 3, 0)) * 3592 / 4096 + 128, 0); \
		} \
		a_ptr += a->step; \
		b_ptr += b->step; \
	}
	ccv_matrix_getter(a->type, ccv_matrix_setter_getter, b->type, for_block);
#undef for_block
}

void ccv_color_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int flag)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_color_transform(%d)", flag), a->sig, CCV_EOF_SIGN);
	assert(flag == CCV_RGB_TO_YUV);
	switch (flag)
	{
		case CCV_RGB_TO_YUV:
			assert(CCV_GET_CHANNEL(a->type) == CCV_C3);
			type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_C3 : CCV_GET_DATA_TYPE(type) | CCV_C3;
			break;
	}
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_C3, type, sig);
	ccv_object_return_if_cached(, db);
	switch (flag)
	{
		case CCV_RGB_TO_YUV:
			_ccv_rgb_to_yuv(a, db);
			break;
	}
}

void ccv_saturation(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double ds)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C3); // only works in RGB space
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_saturation(%la)", ds), a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_object_return_if_cached(, db);
	int i, j;
	unsigned char* aptr = a->data.u8;
	unsigned char* bptr = db->data.u8;
#define for_block(_for_get, _for_set) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
		{ \
			double gs = _for_get(aptr, j * 3, 0) * 0.299 + _for_get(aptr, j * 3 + 1, 0) * 0.587 + _for_get(aptr, j * 3 + 2, 0) * 0.114; \
			_for_set(bptr, j * 3, (_for_get(aptr, j * 3, 0) - gs) * ds + gs, 0); \
			_for_set(bptr, j * 3 + 1, (_for_get(aptr, j * 3 + 1, 0) - gs) * ds + gs, 0); \
			_for_set(bptr, j * 3 + 2, (_for_get(aptr, j * 3 + 2, 0) - gs) * ds + gs, 0); \
		} \
		aptr += a->step; \
		bptr += db->step; \
	}
	ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
}

void ccv_contrast(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double ds)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_contrast(%la)", ds), a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_object_return_if_cached(, db);
	int i, j, k, ch = CCV_GET_CHANNEL(a->type);
	double* ms = (double*)alloca(sizeof(double) * ch);
	memset(ms, 0, sizeof(double) * ch);
	unsigned char* aptr = a->data.u8;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			for (k = 0; k < ch; k++) \
				ms[k] += _for_get(aptr, j * ch + k, 0); \
		aptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	for (i = 0; i < ch; i++)
		ms[i] = ms[i] / (a->rows * a->cols);
	aptr = a->data.u8;
	unsigned char* bptr = db->data.u8;
	if (CCV_GET_DATA_TYPE(a->type) == CCV_8U && CCV_GET_DATA_TYPE(db->type) == CCV_8U) // specialize for 8U type
	{
		unsigned char* us = (unsigned char*)alloca(sizeof(unsigned char) * ch * 256);
		for (i = 0; i < 256; i++)
			for (j = 0; j < ch; j++)
				us[i * ch + j] = ccv_clamp((i - ms[j]) * ds + ms[j], 0, 255);
		for (i = 0; i < a->rows; i++)
		{
			for (j = 0; j < a->cols; j++)
				for (k = 0; k < ch; k++)
					bptr[j * ch + k] = us[(aptr[j * ch + k]) * ch + k];
			aptr += a->step;
			bptr += db->step;
		}
	} else {
#define for_block(_for_get, _for_set) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
				for (k = 0; k < ch; k++) \
					_for_set(bptr, j * ch + k, (_for_get(aptr, j * ch + k, 0) - ms[k]) * ds + ms[k], 0); \
			aptr += a->step; \
			bptr += db->step; \
		}
		ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
	}
}
