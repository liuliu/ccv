#include "ccv.h"
#include "ccv_internal.h"

void ccv_decimal_slice(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float y, float x, int rows, int cols)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_decimal_slice(%a,%a,%d,%d)", y, x, rows, cols), a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_object_return_if_cached(, db);
	int i, j, ch = CCV_GET_CHANNEL(a->type);
	int ix = (int)x, iy = (int)y;
	float xd = x - ix, yd = y - iy;
	float w00 = (1 - xd) * (1 - yd);
	float w01 = xd * (1 - yd);
	float w10 = (1 - xd) * yd;
	float w11 = xd * yd;
	int dx = 0, dy = 0;
	int rows_1 = 0, cols_1 = 0;
	// it is going to be hard to deal with border efficiently, since this is used for tld, will ignore border for now
	if (!(iy >= 0 && iy + rows < a->rows && ix >= 0 && ix + cols < a->cols))
	{
		ccv_zero(db);
		if (iy < 0) { rows += iy; dy = -iy; iy = 0; }
		if (iy + rows >= a->rows)
		{
			rows = a->rows - iy - 1;
			if (iy + rows > a->rows)
				rows_1 = 1; // we need to do our best to padding the last row
		}
		if (ix < 0) { cols += ix; dx = -ix; ix = 0; }
		if (x + cols >= a->cols)
		{
			cols = a->cols - ix - 1;
			if (x + cols > a->cols)
				cols_1 = 1; // we need to do our best to padding the last col
		}
	}
	unsigned char* a_ptr = (unsigned char*)ccv_get_dense_matrix_cell(a, iy, ix, 0);
	unsigned char* b_ptr = (unsigned char*)ccv_get_dense_matrix_cell(db, dy, dx, 0);
#define for_block(_for_set, _for_get) \
	for (i = 0; i < rows; i++) \
	{ \
		for (j = 0; j < cols * ch; j++) \
		{ \
			_for_set(b_ptr, j, (_for_get(a_ptr, j, 0) * G00 + _for_get(a_ptr, j + ch, 0) * G01 + _for_get(a_ptr + a->step, j, 0) * G10 + _for_get(a_ptr + a->step, j + ch, 0) * G11) / GALL, 0); \
		} \
		if (cols_1) \
			_for_set(b_ptr, j, (_for_get(a_ptr, j, 0) * (G00 + G01) + _for_get(a_ptr + a->step, j, 0) * G10 + _for_get(a_ptr + a->step, j + ch, 0) * G11) / GALL, 0); \
		a_ptr += a->step; \
		b_ptr += db->step; \
	} \
	if (rows_1) \
	{ \
		for (j = 0; j < cols * ch; j++) \
		{ \
			_for_set(b_ptr, j, (_for_get(a_ptr, j, 0) * (G00 + G10) + _for_get(a_ptr, j + ch, 0) * (G01 + G11)) / GALL, 0); \
		} \
		if (cols_1) \
			_for_set(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
	}
	/* unswitch in the manual way so that we can use integer interpolation */
	if ((a->type & CCV_8U) || (a->type & CCV_32S) || (a->type & CCV_64S))
	{
		const int W_BITS14 = 14;
		int iw00 = (int)(w00 * (1 << W_BITS14) + 0.5);
		int iw01 = (int)(w01 * (1 << W_BITS14) + 0.5);
		int iw10 = (int)(w10 * (1 << W_BITS14) + 0.5);
		int iw11 = (1 << W_BITS14) - iw00 - iw01 - iw10;
#define G00 (iw00)
#define G01 (iw01)
#define G10 (iw10)
#define G11 (iw11)
#define GCOM (1 << (W_BITS14 - 1))
#define GALL (1 << (W_BITS14))
		ccv_matrix_setter(db->type, ccv_matrix_getter_integer_only, a->type, for_block);
#undef G00
#undef G01
#undef G10
#undef G11
#undef GCOM
#undef GALL
	} else {
#define G00 (w00)
#define G01 (w01)
#define G10 (w10)
#define G11 (w11)
#define GCOM (0)
#define GALL (1)
		ccv_matrix_setter(db->type, ccv_matrix_getter_float_only, a->type, for_block);
#undef G00
#undef G01
#undef G10
#undef G11
#undef GCOM
#undef GALL
	}
#undef for_block
}

ccv_decimal_point_t ccv_perspective_transform_apply(ccv_decimal_point_t point, ccv_size_t size, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
{
	m00 *= 1.0 / ccv_max(size.width, size.height);
	m01 *= 1.0 / ccv_max(size.width, size.height);
	m02 *= 1.0 / ccv_max(size.width, size.height);
	m10 *= 1.0 / ccv_max(size.width, size.height);
	m11 *= 1.0 / ccv_max(size.width, size.height);
	m12 *= 1.0 / ccv_max(size.width, size.height);
	m20 *= 1.0 / (ccv_max(size.width, size.height) * ccv_max(size.width, size.height));
	m21 *= 1.0 / (ccv_max(size.width, size.height) * ccv_max(size.width, size.height));
	m22 *= 1.0 / ccv_max(size.width, size.height);
	point.x -= size.width * 0.5;
	point.y -= size.height * 0.5;
	float wz = 1.0 / (point.x * m20 + point.y * m21 + m22);
	float wx = size.width * 0.5 + (point.x * m00 + point.y * m01 + m02) * wz;
	float wy = size.height * 0.5 + (point.x * m10 + point.y * m11 + m12) * wz;
	return ccv_decimal_point(wx, wy);
}

// this method is a merely baseline implementation and has no optimization effort ever put into it, if at all
void ccv_perspective_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_perspective_transform(%a,%a,%a,%a,%a,%a,%a,%a,%a)", m00, m01, m02, m10, m11, m12, m20, m21, m22), a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_object_return_if_cached(, db);
	// with default of bilinear interpolation
	int i, j, k, ch = CCV_GET_CHANNEL(a->type);
	unsigned char* a_ptr = a->data.u8;
	unsigned char* b_ptr = db->data.u8;
	// assume field of view is 60, modify the matrix value to reflect that
	// (basically, apply x / ccv_max(a->rows, a->cols), y / ccv_max(a->rows, a->cols) before hand
	m00 *= 1.0 / ccv_max(a->rows, a->cols);
	m01 *= 1.0 / ccv_max(a->rows, a->cols);
	m02 *= 1.0 / ccv_max(a->rows, a->cols);
	m10 *= 1.0 / ccv_max(a->rows, a->cols);
	m11 *= 1.0 / ccv_max(a->rows, a->cols);
	m12 *= 1.0 / ccv_max(a->rows, a->cols);
	m20 *= 1.0 / (ccv_max(a->rows, a->cols) * ccv_max(a->rows, a->cols));
	m21 *= 1.0 / (ccv_max(a->rows, a->cols) * ccv_max(a->rows, a->cols));
	m22 *= 1.0 / ccv_max(a->rows, a->cols);
#define for_block(_for_set, _for_get) \
	for (i = 0; i < db->rows; i++) \
	{ \
		float cy = i - db->rows * 0.5; \
		float crx = cy * m01 + m02; \
		float cry = cy * m11 + m12; \
		float crz = cy * m21 + m22; \
		for (j = 0; j < db->cols; j++) \
		{ \
			float cx = j - db->cols * 0.5; \
			float wz = 1.0 / (cx * m20 + crz); \
			float wx = a->cols * 0.5 + (cx * m00 + crx) * wz; \
			float wy = a->rows * 0.5 + (cx * m10 + cry) * wz; \
			int iwx = (int)wx; \
			int iwy = (int)wy; \
			wx = wx - iwx; \
			wy = wy - iwy; \
			int iwx1 = ccv_min(iwx + 1, a->cols - 1); \
			int iwy1 = ccv_min(iwy + 1, a->rows - 1); \
			if (iwx >= 0 && iwx < a->cols && iwy >= 0 && iwy < a->rows) \
				for (k = 0; k < ch; k++) \
					_for_set(b_ptr, j * ch + k, _for_get(a_ptr + iwy * a->step, iwx * ch + k, 0) * (1 - wx) * (1 - wy) + \
												_for_get(a_ptr + iwy * a->step, iwx1 * ch + k, 0) * wx * (1 - wy) + \
												_for_get(a_ptr + iwy1 * a->step, iwx * ch + k, 0) * (1 - wx) * wy + \
												_for_get(a_ptr + iwy1 * a->step, iwx1 * ch + k, 0) * wx * wy, 0); \
			else \
				for (k = 0; k < ch; k++) \
					_for_set(b_ptr, j * ch + k, 0, 0); \
		} \
		b_ptr += db->step; \
	}
	ccv_matrix_setter(db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
}
