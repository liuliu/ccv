#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

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

#ifdef HAVE_GSL
static ccv_dense_matrix_t* _ccv_scd_slice_with_distortion(gsl_rng* rng, ccv_dense_matrix_t* image, ccv_decimal_pose_t pose, ccv_size_t size, ccv_margin_t margin, float deform_angle, float deform_scale, float deform_shift)
{
	float rotate_x = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.pitch;
	float rotate_y = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.yaw;
	float rotate_z = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.roll;
	float scale = gsl_rng_uniform(rng);
	// to make the scale evenly distributed, for example, when deforming of 1/2 ~ 2, we want it to distribute around 1, rather than any average of 1/2 ~ 2
	scale = (1 + deform_scale * scale) / (1 + deform_scale * (1 - scale));
	float scale_ratio = sqrtf((float)(size.width * size.height) / (pose.a * pose.b * 4));
	float m00 = cosf(rotate_z) * scale;
	float m01 = cosf(rotate_y) * sinf(rotate_z) * scale;
	float m02 = (deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) / scale_ratio + pose.x + (margin.right - margin.left) / scale_ratio - image->cols * 0.5;
	float m10 = (sinf(rotate_y) * cosf(rotate_z) - cosf(rotate_x) * sinf(rotate_z)) * scale;
	float m11 = (sinf(rotate_y) * sinf(rotate_z) + cosf(rotate_x) * cosf(rotate_z)) * scale;
	float m12 = (deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) / scale_ratio + pose.y + (margin.bottom - margin.top) / scale_ratio - image->rows * 0.5;
	float m20 = (sinf(rotate_y) * cosf(rotate_z) + sinf(rotate_x) * sinf(rotate_z)) * scale;
	float m21 = (sinf(rotate_y) * sinf(rotate_z) - sinf(rotate_x) * cosf(rotate_z)) * scale;
	float m22 = cosf(rotate_x) * cosf(rotate_y);
	ccv_dense_matrix_t* b = 0;
	ccv_perspective_transform(image, &b, 0, m00, m01, m02, m10, m11, m12, m20, m21, m22);
	ccv_dense_matrix_t* resize = 0;
	// have 1px border around the grayscale image because we need these to compute correct gradient feature
	ccv_size_t scale_size = {
		.width = (int)((size.width + margin.left + margin.right + 2) / scale_ratio + 0.5),
		.height = (int)((size.height + margin.top + margin.bottom + 2) / scale_ratio + 0.5),
	};
	assert(scale_size.width > 0 && scale_size.height > 0);
	ccv_slice(b, (ccv_matrix_t**)&resize, 0, (int)(b->rows * 0.5 - (size.height + margin.top + margin.bottom + 2) / scale_ratio * 0.5 + 0.5), (int)(b->cols * 0.5 - (size.width + margin.left + margin.right + 2) / scale_ratio * 0.5 + 0.5), scale_size.height, scale_size.width);
	ccv_matrix_free(b);
	b = 0;
	if (scale_ratio > 1)
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom + 2, size.width + margin.left + margin.right + 2, CCV_INTER_CUBIC);
	else
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom + 2, size.width + margin.left + margin.right + 2, CCV_INTER_AREA);
	ccv_matrix_free(resize);
	return b;
}

static ccv_array_t* _ccv_scd_collect_negatives(gsl_rng* rng, ccv_size_t size, ccv_array_t* hard_mine, int total, float deform_angle, float deform_scale, float deform_shift, int grayscale)
{
	ccv_array_t* negatives = ccv_array_new(ccv_compute_dense_matrix_size(size.height + 2, size.width + 2, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), total, 0);
	int i, j, k;
	for (i = 0; i < total;)
	{
		FLUSH(CCV_CLI_INFO, " - collect negatives %d%% (%d / %d)", (i + 1) * 100 / total, i + 1, total);
		double ratio = (double)(total - i) / hard_mine->rnum;
		for (j = 0; j < hard_mine->rnum && i < total; j++)
		{
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(hard_mine, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
			if (image == 0)
			{
				PRINT(CCV_CLI_ERROR, "\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
				continue;
			}
			double max_scale_ratio = ccv_min((double)image->rows / size.height, (double)image->cols / size.width);
			if (max_scale_ratio <= 0.5) // too small to be interesting
				continue;
			for (k = 0; k < ratio; k++)
				if (k < (int)ratio || gsl_rng_uniform(rng) <= ratio - (int)ratio)
				{
					FLUSH(CCV_CLI_INFO, " - collect negatives %d%% (%d / %d)", (i + 1) * 100 / total, i + 1, total);
					ccv_decimal_pose_t pose;
					double scale_ratio = gsl_rng_uniform(rng) * (max_scale_ratio - 0.5) + 0.5;
					pose.a = size.width * 0.5 * scale_ratio;
					pose.b = size.height * 0.5 * scale_ratio;
					pose.x = gsl_rng_uniform_int(rng, ccv_max((int)(image->cols - pose.a * 2 + 1.5), 1)) + pose.a;
					pose.y = gsl_rng_uniform_int(rng, ccv_max((int)(image->rows - pose.b * 2 + 1.5), 1)) + pose.b;
					pose.roll = pose.pitch = pose.yaw = 0;
					ccv_dense_matrix_t* sliced = _ccv_scd_slice_with_distortion(rng, image, pose, size, ccv_margin(0, 0, 0, 0), deform_angle, deform_scale, deform_shift);
					sliced->sig = 0;
					// this leveraged the fact that because I know the ccv_dense_matrix_t is continuous in memory
					ccv_array_push(negatives, sliced);
					ccv_matrix_free(sliced);
					++i;
					if (i >= total)
						break;
				}
			ccv_matrix_free(image);
		}
	}
	PRINT(CCV_CLI_INFO, "\n");
	return negatives;
}

static ccv_array_t*_ccv_scd_collect_positives(ccv_size_t size, ccv_array_t* posfiles, int grayscale)
{
	ccv_array_t* positives = ccv_array_new(ccv_compute_dense_matrix_size(size.height, size.width, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), posfiles->rnum, 0);
	int i;
	for (i = 0; i < posfiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, i);
		ccv_dense_matrix_t* a = 0;
		ccv_read(file_info->filename, &a, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
		a->sig = 0;
		ccv_array_push(positives, a);
		ccv_matrix_free(a);
	}
	return positives;
}
#endif

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_new(ccv_array_t* posfiles, ccv_array_t* hard_mine, int negative_count, const char* filename, ccv_scd_train_param_t params)
{
#ifdef HAVE_GSL
	assert(posfiles->rnum > 0);
	assert(hard_mine->rnum > 0);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	ccv_array_t* positives = _ccv_scd_collect_positives(params.size, posfiles, params.grayscale);
	ccv_array_t* negatives = _ccv_scd_collect_negatives(rng, params.size, hard_mine, negative_count, params.deform.angle, params.deform.scale, params.deform.shift, params.grayscale);
	int t;
	for (t = 0; t < params.boosting; t++)
	{
	}
	return 0;
#else
	assert(0 && "ccv_scd_classifier_cascade_new requires GSL library support");
	return 0;
#endif
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
	ccv_dense_matrix_t* scd = 0;
	ccv_scd(a, &scd, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(scd, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(scd);
	ccv_matrix_free(sat);
	return 0;
}
