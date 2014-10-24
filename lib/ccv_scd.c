#include "ccv.h"
#include "ccv_internal.h"

void ccv_scd(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	int ch = CCV_GET_CHANNEL(a->type);
	assert(ch == 1 || ch == 3);
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_literal("ccv_scd"), a->sig, CCV_EOF_SIGN);
	// diagonal u v, and x, y, therefore 8 channels
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_32F | 8, CCV_32F | 8, sig);
	ccv_object_return_if_cached(, db);
	ccv_dense_matrix_t* dx = 0;
	ccv_sobel(a, &dx, 0, 1, 0);
	ccv_dense_matrix_t* dy = 0;
	ccv_sobel(a, &dy, 0, 0, 1);
	ccv_dense_matrix_t* du = 0;
	ccv_sobel(a, &du, 0, 1, 1);
	ccv_dense_matrix_t* dv = 0;
	ccv_sobel(a, &dv, 0, -1, 1);
	assert(CCV_GET_DATA_TYPE(dx->type) == CCV_GET_DATA_TYPE(dy->type));
	assert(CCV_GET_DATA_TYPE(dy->type) == CCV_GET_DATA_TYPE(du->type));
	assert(CCV_GET_DATA_TYPE(du->type) == CCV_GET_DATA_TYPE(dv->type));
	assert(CCV_GET_CHANNEL(dx->type) == CCV_GET_CHANNEL(dy->type));
	assert(CCV_GET_CHANNEL(dy->type) == CCV_GET_CHANNEL(du->type));
	assert(CCV_GET_CHANNEL(du->type) == CCV_GET_CHANNEL(dv->type));
	// this is a naive unoptimized implementation yet
	int i, j, k;
	unsigned char* dx_ptr = dx->data.u8;
	unsigned char* dy_ptr = dy->data.u8;
	unsigned char* du_ptr = du->data.u8;
	unsigned char* dv_ptr = dv->data.u8;
	float* dbp = db->data.f32;
	if (ch == 1)
	{
#define for_block(_, _for_get) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
			{ \
				float fdx = _for_get(dx_ptr, j, 0), fdy = _for_get(dy_ptr, j, 0); \
				float fdu = _for_get(du_ptr, j, 0), fdv = _for_get(dv_ptr, j, 0); \
				float adx = fabsf(fdx), ady = fabsf(fdy); \
				float adu = fabsf(fdu), adv = fabsf(fdv); \
				dbp[0] = adx - fdx, dbp[1] = adx + fdx; \
				dbp[2] = ady - fdy, dbp[3] = ady + fdy; \
				dbp[4] = adu - fdu, dbp[5] = adu + fdu; \
				dbp[6] = adv - fdv, dbp[7] = adv + fdv; \
				dbp += 8; \
			} \
			dx_ptr += dx->step; \
			dy_ptr += dy->step; \
			du_ptr += du->step; \
			dv_ptr += dv->step; \
		}
		ccv_matrix_getter(dx->type, for_block);
#undef for_block
	} else {
#define for_block(_, _for_get) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
			{ \
				float fdx = _for_get(dx_ptr, j * ch, 0), fdy = _for_get(dy_ptr, j * ch, 0); \
				float fdu = _for_get(du_ptr, j * ch, 0), fdv = _for_get(dv_ptr, j * ch, 0); \
				float adx = fabsf(fdx), ady = fabsf(fdy); \
				float adu = fabsf(fdu), adv = fabsf(fdv); \
				/* select the strongest ones from all the channels */ \
				for (k = 1; k < ch; k++) \
				{ \
					if (fabsf((float)_for_get(dx_ptr, j * ch + k, 0)) > adx) \
					{ \
						fdx = _for_get(dx_ptr, j * ch + k, 0); \
						adx = fabsf(fdx); \
					} \
					if (fabsf((float)_for_get(dy_ptr, j * ch + k, 0)) > ady) \
					{ \
						fdy = _for_get(dy_ptr, j * ch + k, 0); \
						ady = fabsf(fdy); \
					} \
					if (fabsf((float)_for_get(du_ptr, j * ch + k, 0)) > adu) \
					{ \
						fdu = _for_get(du_ptr, j * ch + k, 0); \
						adu = fabsf(fdu); \
					} \
					if (fabsf((float)_for_get(dv_ptr, j * ch + k, 0)) > adv) \
					{ \
						fdv = _for_get(dv_ptr, j * ch + k, 0); \
						adv = fabsf(fdv); \
					} \
				} \
				dbp[0] = adx - fdx, dbp[1] = adx + fdx; \
				dbp[2] = ady - fdy, dbp[3] = ady + fdy; \
				dbp[4] = adu - fdu, dbp[5] = adu + fdu; \
				dbp[6] = adv - fdv, dbp[7] = adv + fdv; \
				dbp += 8; \
			} \
			dx_ptr += dx->step; \
			dy_ptr += dy->step; \
			du_ptr += du->step; \
			dv_ptr += dv->step; \
		}
		ccv_matrix_getter(dx->type, for_block);
#undef for_block
	}
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
	ccv_matrix_free(du);
	ccv_matrix_free(dv);
}

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_new(ccv_array_t* categorizeds, ccv_array_t* hard_mine, const char* filename, ccv_scd_train_param_t params)
{
	return 0;
}

void ccv_scd_classifier_cascade_write(ccv_scd_classifier_cascade_t* cascade, const char* filename)
{
}

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_read(const char* filename)
{
	return 0;
}

void ccv_scd_classifier_cascade_free(ccv_scd_classifier_cascade_t* cascade)
{
}

ccv_array_t* ccv_scd_detect_objects(ccv_dense_matrix_t* a, ccv_scd_classifier_cascade_t** cascades, int count, ccv_scd_param_t params)
{
	return 0;
}
