#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifdef HAVE_LIBLINEAR
#include <linear.h>
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

#if defined(HAVE_GSL) && defined(HAVE_LIBLINEAR)
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
				if (k < (int)ratio || gsl_rng_uniform(rng) <= ccv_max(0.1, ratio - (int)ratio))
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
		FLUSH(CCV_CLI_INFO, " - collect positives %d%% (%d / %d)", (i + 1) * 100 / posfiles->rnum, i + 1, posfiles->rnum);
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, i);
		ccv_dense_matrix_t* a = 0;
		ccv_read(file_info->filename, &a, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
		a->sig = 0;
		ccv_array_push(positives, a);
		ccv_matrix_free(a);
	}
	PRINT(CCV_CLI_INFO, "\n");
	return positives;
}

static ccv_array_t* _ccv_scd_features(ccv_size_t base, int range_through, int step_through, ccv_size_t size)
{
	ccv_array_t* features = ccv_array_new(sizeof(ccv_scd_feature_t), 64, 0);
	int x, y, w, h;
	for (w = base.width; w <= size.width; w += range_through)
		if (w % 4 == 0) // only allow 4:1
		{
			h = w / 4;
			for (x = 0; x <= size.width - w; x += step_through)
				for (y = 0; y <= size.height - h; y += step_through)
				{
					// 4x1 feature
					ccv_scd_feature_t feature;
					feature.sx[0] = x;
					feature.dx[0] = x + (w / 4);
					feature.sx[1] = x + (w / 4);
					feature.dx[1] = x + 2 * (w / 4);
					feature.sx[2] = x + 2 * (w / 4);
					feature.dx[2] = x + 3 * (w / 4);
					feature.sx[3] = x + 3 * (w / 4);
					feature.dx[3] = x + w;
					feature.sy[0] = feature.sy[1] = feature.sy[2] = feature.sy[3] = y;
					feature.dy[0] = feature.dy[1] = feature.dy[2] = feature.dy[3] = y + h;
					ccv_array_push(features, &feature);
				}
		}
	for (h = base.height; h <= size.height; h += range_through)
		if (h % 4 == 0) // only allow 1:4
		{
			w = h / 4;
			for (x = 0; x <= size.width - w; x += step_through)
				for (y = 0; y <= size.height - h; y += step_through)
				{
					// 1x4 feature
					ccv_scd_feature_t feature;
					feature.sx[0] = feature.sx[1] = feature.sx[2] = feature.sx[3] = x;
					feature.dx[0] = feature.dx[1] = feature.dx[2] = feature.dx[3] = x + w;
					feature.sy[0] = y;
					feature.dy[0] = y + (h / 4);
					feature.sy[1] = y + (h / 4);
					feature.dy[1] = y + 2 * (h / 4);
					feature.sy[2] = y + 2 * (h / 4);
					feature.dy[2] = y + 3 * (h / 4);
					feature.sy[3] = y + 3 * (h / 4);
					feature.dy[3] = y + h;
					ccv_array_push(features, &feature);
				}
		}
	for (w = base.width; w <= size.width; w += range_through)
		for (h = base.height; h <= size.height; h += range_through)
			for (x = 0; x <= size.width - w; x += step_through)
				for (y = 0; y <= size.height - h; y += step_through)
					if (w % 2 == 0 && h % 2 == 0 &&
						(w == h || w == h * 2 || w * 2 == h || w * 2 == h * 3 || w * 3 == h * 2)) // allow 1:1, 1:2, 2:1, 2:3, 3:2
					{
						// 2x2 feature
						ccv_scd_feature_t feature;
						feature.sx[0] = feature.sx[1] = x;
						feature.dx[0] = feature.dx[1] = x + (w / 2);
						feature.sy[0] = feature.sy[2] = y;
						feature.dy[0] = feature.dy[2] = y + (h / 2);
						feature.sx[2] = feature.sx[3] = x + (w / 2);
						feature.dx[2] = feature.dx[3] = x + w;
						feature.sy[1] = feature.sy[3] = y + (h / 2);
						feature.dy[1] = feature.dy[3] = y + h;
						ccv_array_push(features, &feature);
					}
	return features;
}

typedef struct {
	double value;
	int index;
} ccv_scd_value_index_t;

#define more_than(s1, s2, aux) ((s1).value >= (s2).value)
static CCV_IMPLEMENT_QSORT(_ccv_scd_value_index_qsort, ccv_scd_value_index_t, more_than)
#undef more_than

static void _ccv_scd_run_feature_sat(ccv_dense_matrix_t* sat, ccv_scd_feature_t* feature, float surf[32])
{
	int i, j;
	// extract feature
	for (i = 0; i < 4; i++)
	{
		float* d = sat->data.f32 + (sat->cols * feature->sy[i] + feature->sx[i]) * 8;
		float* du = sat->data.f32 + (sat->cols * feature->dy[i] + feature->sx[i]) * 8;
		float* dv = sat->data.f32 + (sat->cols * feature->sy[i] + feature->dx[i]) * 8;
		float* duv = sat->data.f32 + (sat->cols * feature->dy[i] + feature->dx[i]) * 8;
		for (j = 0; j < 8; j++)
			surf[i * 8 + j] = duv[j] - du[j] + d[j] - dv[j];
	}
	// L2Hys normalization
	float v = 0;
	for (i = 0; i < 32; i++)
		v += surf[i] * surf[i];
	v = 1.0 / (sqrtf(v) + 1e-6);
	static float theta = 2.0 / 5.65685424949; // sqrtf(32)
	float u = 0;
	for (i = 0; i < 32; i++)
	{
		surf[i] = surf[i] * v;
		surf[i] = ccv_clamp(surf[i], -theta, theta);
		u += surf[i] * surf[i];
	}
	u = 1.0 / (sqrtf(u) + 1e-6);
	for (i = 0; i < 32; i++)
		surf[i] = surf[i] * u;
}

static void _ccv_scd_run_feature(ccv_dense_matrix_t* a, ccv_scd_feature_t* feature, float surf[32])
{
	ccv_dense_matrix_t* b = 0;
	ccv_scd(a, &b, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(b);
	_ccv_scd_run_feature_sat(sat, feature, surf);
	ccv_matrix_free(sat);
}

static void _ccv_scd_liblinear_null(const char* str) { /* do nothing */ }

static void _ccv_scd_feature_supervised_train(gsl_rng* rng, ccv_array_t* features, ccv_array_t* positives, double* pw, ccv_array_t* negatives, double* nw, int active_set, int wide_set, double C)
{
	int i, j, k;
	ccv_scd_value_index_t* pwidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * positives->rnum);
	for (i = 0; i < positives->rnum; i++)
		pwidx[i].index = i, pwidx[i].value = pw[i];
	_ccv_scd_value_index_qsort(pwidx, positives->rnum, 0);
	int adjusted_positive_set = positives->rnum;
	for (i = wide_set - 1; i < positives->rnum; i++)
		if (fabs(pwidx[i].value - pwidx[i + 1].value) > FLT_MIN)
		{
			adjusted_positive_set = i + 1;
			break;
		}
	ccv_scd_value_index_t* nwidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * negatives->rnum);
	for (i = 0; i < negatives->rnum; i++)
		nwidx[i].index = i, nwidx[i].value = nw[i];
	_ccv_scd_value_index_qsort(nwidx, negatives->rnum, 0);
	int adjusted_negative_set = negatives->rnum;
	for (i = wide_set - 1; i < negatives->rnum; i++)
		if (fabs(nwidx[i].value - nwidx[i + 1].value) > FLT_MIN)
		{
			adjusted_negative_set = i + 1;
			break;
		}
	for (i = 0; i < features->rnum; i++)
	{
		struct problem prob;
		prob.l = active_set * 2;
		prob.n = 32 + 1;
		prob.bias = 1.0;
		prob.y = malloc(sizeof(prob.y[0]) * active_set * 2);
		prob.x = (struct feature_node**)malloc(sizeof(struct feature_node*) * active_set * 2);
		ccv_scd_feature_t* feature = (ccv_scd_feature_t*)ccv_array_get(features, i);
		gsl_ran_shuffle(rng, pwidx, adjusted_positive_set, sizeof(ccv_scd_value_index_t));
		float surf[32];
		struct feature_node* surf_feature;
		for (j = 0; j < active_set; j++)
		{
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(positives, pwidx[j].index);
			a->data.u8 = (unsigned char*)(a + 1);
			_ccv_scd_run_feature(a, feature, surf);
		 	surf_feature = (struct feature_node*)malloc(sizeof(struct feature_node) * (32 + 2));
			for (k = 0; k < 32; k++)
				surf_feature[k].index = k + 1, surf_feature[k].value = surf[k];
			surf_feature[32].index = 33;
			surf_feature[32].value = prob.bias;
			surf_feature[33].index = -1;
			prob.x[j] = surf_feature;
			prob.y[j] = 1;
		}
		gsl_ran_shuffle(rng, nwidx, adjusted_negative_set, sizeof(ccv_scd_value_index_t));
		for (j = 0; j < active_set; j++)
		{
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(negatives, nwidx[j].index);
			a->data.u8 = (unsigned char*)(a + 1);
			_ccv_scd_run_feature(a, feature, surf);
		 	surf_feature = (struct feature_node*)malloc(sizeof(struct feature_node) * (32 + 2));
			for (k = 0; k < 32; k++)
				surf_feature[k].index = k + 1, surf_feature[k].value = surf[k];
			surf_feature[32].index = 33;
			surf_feature[32].value = prob.bias;
			surf_feature[33].index = -1;
			prob.x[j + active_set] = surf_feature;
			prob.y[j + active_set] = -1;
		}
		struct parameter linear_parameters = { .solver_type = L1R_LR,
											   .eps = 1e-2,
											   .C = C,
											   .nr_weight = 0,
											   .weight_label = 0,
											   .weight = 0 };
		const char* err = check_parameter(&prob, &linear_parameters);
		if (err)
		{
			PRINT(CCV_CLI_ERROR, " - ERROR: cannot pass check parameter: %s\n", err);
			exit(-1);
		}
		set_print_string_function(_ccv_scd_liblinear_null);
		struct model* linear = train(&prob, &linear_parameters);
		for (j = 0; j < 32; j++)
			feature->w[j] = linear->w[j];
		feature->bias = linear->w[32];
		free_and_destroy_model(&linear);
		free(prob.y);
		for (j = 0; j < prob.l; j++)
			free(prob.x[j]);
		free(prob.x);
	}
	ccfree(pwidx);
	ccfree(nwidx);
}

static ccv_scd_feature_t _ccv_scd_best_feature(ccv_array_t* features, ccv_array_t* positives, double* pw, ccv_array_t* negatives, double* nw)
{
	ccv_scd_feature_t best_feature;
	int i, j, k;
	float surf[32];
	double* fw = (double*)cccalloc(features->rnum, sizeof(double));
	for (i = 0; i < positives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(positives, i);
		a->data.u8 = (unsigned char*)(a + 1);
		ccv_dense_matrix_t* b = 0;
		ccv_scd(a, &b, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(b);
		for (j = 0; j < features->rnum; j++)
		{
			ccv_scd_feature_t* feature = (ccv_scd_feature_t*)ccv_array_get(features, j);
			_ccv_scd_run_feature_sat(sat, feature, surf);
			float v = feature->bias;
			for (k = 0; k < 32; k++)
				v += surf[k] * feature->w[k];
			if (v > 0)
				fw[j] += pw[i];
		}
		ccv_matrix_free(sat);
	}
	for (i = 0; i < negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(negatives, i);
		a->data.u8 = (unsigned char*)(a + 1);
		ccv_dense_matrix_t* b = 0;
		ccv_scd(a, &b, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(b);
		for (j = 0; j < features->rnum; j++)
		{
			ccv_scd_feature_t* feature = (ccv_scd_feature_t*)ccv_array_get(features, j);
			_ccv_scd_run_feature_sat(sat, feature, surf);
			float v = feature->bias;
			for (k = 0; k < 32; k++)
				v += surf[k] * feature->w[k];
			if (v < 0)
				fw[j] += nw[i];
		}
		ccv_matrix_free(sat);
	}
	j = 0;
	double max_w = fw[0];
	for (i = 1; i < features->rnum; i++)
		if (fw[i] > max_w)
		{
			max_w = fw[i];
			j = i;
		}
	memcpy(&best_feature, ccv_array_get(features, j), sizeof(ccv_scd_feature_t));
	ccfree(fw);
	return best_feature;
}

static double _ccv_scd_compute_auc(double* s, int posnum, int negnum)
{
	ccv_scd_value_index_t* sidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * (posnum + negnum));
	int i;
	for (i = 0; i < posnum + negnum; i++)
		sidx[i].value = s[i], sidx[i].index = i;
	_ccv_scd_value_index_qsort(sidx, posnum + negnum, 0);
	int fp = 0, tp = 0, fp_prev = 0, tp_prev = 0;
	double a = 0;
	double f_prev = -DBL_MAX;
	for (i = 0; i < posnum + negnum; i++)
	{
		if (sidx[i].value != f_prev)
		{
			a += (double)(fp - fp_prev) * (tp + tp_prev) * 0.5;
			f_prev = sidx[i].value;
			fp_prev = fp;
			tp_prev = tp;
		}
		if (sidx[i].index < posnum)
			++tp;
		else
			++fp;
	}
	a += (double)(negnum - fp_prev) * (posnum + tp_prev) * 0.5;
	return a / ((double)posnum * negnum);
}
#endif

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_new(ccv_array_t* posfiles, ccv_array_t* hard_mine, int negative_count, const char* filename, ccv_scd_train_param_t params)
{
#if defined(HAVE_GSL) && defined(HAVE_LIBLINEAR)
	assert(posfiles->rnum > 0);
	assert(hard_mine->rnum > 0);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	ccv_array_t* features = _ccv_scd_features(params.feature.base, params.feature.range_through, params.feature.step_through, params.size);
	PRINT(CCV_CLI_INFO, " - %d features\n", features->rnum);
	ccv_array_t* positives = _ccv_scd_collect_positives(params.size, posfiles, params.grayscale);
	ccv_array_t* negatives = _ccv_scd_collect_negatives(rng, params.size, hard_mine, negative_count, params.deform.angle, params.deform.scale, params.deform.shift, params.grayscale);
	int t, k, i, j;
	double* pw = (double*)ccmalloc(sizeof(double) * positives->rnum);
	double* nw = (double*)ccmalloc(sizeof(double) * negatives->rnum);
	int* h = (int*)ccmalloc(sizeof(int) * (positives->rnum + negatives->rnum));
	double* s = (double*)ccmalloc(sizeof(double) * (positives->rnum + negatives->rnum));
	ccv_scd_classifier_cascade_t* cascade = (ccv_scd_classifier_cascade_t*)ccmalloc(sizeof(ccv_scd_classifier_cascade_t));
	cascade->margin = ccv_margin(0, 0, 0, 0);
	cascade->size = params.size;
	cascade->count = 0;
	float surf[32];
	for (t = 0; t < params.boosting; t++)
	{
		for (i = 0; i < positives->rnum; i++)
			pw[i] = 0.5 / positives->rnum;
		for (i = 0; i < negatives->rnum; i++)
			nw[i] = 0.5 / negatives->rnum;
		memset(s, 0, sizeof(double) * (positives->rnum + negatives->rnum));
		for (k = 0; ; k++)
		{
			_ccv_scd_feature_supervised_train(rng, features, positives, pw, negatives, nw, params.feature.active_set, params.feature.wide_set, params.C);
			ccv_scd_feature_t best_feature = _ccv_scd_best_feature(features, positives, pw, negatives, nw);
			double alpha = 0;
			for (i = 0; i < positives->rnum + negatives->rnum; i++)
			{
				ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(i < positives->rnum ? ccv_array_get(positives, i) : ccv_array_get(negatives, i - positives->rnum));
				a->data.u8 = (unsigned char*)(a + 1);
				_ccv_scd_run_feature(a, &best_feature, surf);
				float v = best_feature.bias;
				for (j = 0; j < 32; j++)
					v += best_feature.w[j] * surf[j];
				h[i] = v > 0 ? 1 : -1;
				int y = i < positives->rnum ? 1 : -1;
				double w = i < positives->rnum ? pw[i] : nw[i - positives->rnum];
				// Discrete Adaboost
				alpha += (y != h[i]) ? w : 0;
			}
			assert(alpha < 0.5);
			printf("error: %lf\n", alpha);
			alpha = 0.5 * log((1 - alpha) / alpha);
			printf("alpha: %lf\n", alpha);
			// compute the total score so far
			for (i = 0; i < positives->rnum + negatives->rnum; i++)
				s[i] += h[i] * alpha;
			// compute AUC
			double auc = _ccv_scd_compute_auc(s, positives->rnum, negatives->rnum);
			printf("auc: %lf\n", auc);
			// re-weight
			for (i = 0; i < positives->rnum; i++)
				pw[i] *= exp(-alpha * h[i]);
			for (i = 0; i < negatives->rnum; i++)
				nw[i] *= exp(alpha * h[i + positives->rnum]);
			// re-normalize
			double w = 0;
			for (i = 0; i < positives->rnum; i++)
				w += pw[i];
			w = 0.5 / w;
			for (i = 0; i < positives->rnum; i++)
				pw[i] *= w;
			w = 0;
			for (i = 0; i < negatives->rnum; i++)
				w += nw[i];
			w = 0.5 / w;
			for (i = 0; i < negatives->rnum; i++)
				nw[i] *= w;
		}
		break;
	}
	ccfree(s);
	ccfree(h);
	ccfree(pw);
	ccfree(nw);
	return 0;
#else
	assert(0 && "ccv_scd_classifier_cascade_new requires GSL library and liblinear support");
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
