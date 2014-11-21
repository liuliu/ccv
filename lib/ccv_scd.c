#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#ifdef HAVE_LIBLINEAR
#include <linear.h>
#endif
#include "3rdparty/sqlite3/sqlite3.h"

const ccv_scd_param_t ccv_scd_default_params = {
	.interval = 5,
	.min_neighbors = 2,
	.flags = 0,
	.step_through = 4,
	.size = {
		.width = 40,
		.height = 40,
	},
};

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

static inline void _ccv_scd_run_feature_at(float* at, int cols, ccv_scd_feature_t* feature, float surf[32])
{
	int i, j;
	// extract feature
	for (i = 0; i < 4; i++)
	{
		float* d = at + (cols * feature->sy[i] + feature->sx[i]) * 8;
		float* du = at + (cols * feature->dy[i] + feature->sx[i]) * 8;
		float* dv = at + (cols * feature->sy[i] + feature->dx[i]) * 8;
		float* duv = at + (cols * feature->dy[i] + feature->dx[i]) * 8;
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
	ccv_size_t scale_size = {
		.width = (int)((size.width + margin.left + margin.right) / scale_ratio + 0.5),
		.height = (int)((size.height + margin.top + margin.bottom) / scale_ratio + 0.5),
	};
	assert(scale_size.width > 0 && scale_size.height > 0);
	ccv_slice(b, (ccv_matrix_t**)&resize, 0, (int)(b->rows * 0.5 - (size.height + margin.top + margin.bottom) / scale_ratio * 0.5 + 0.5), (int)(b->cols * 0.5 - (size.width + margin.left + margin.right) / scale_ratio * 0.5 + 0.5), scale_size.height, scale_size.width);
	ccv_matrix_free(b);
	b = 0;
	if (scale_ratio > 1)
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom, size.width + margin.left + margin.right, CCV_INTER_CUBIC);
	else
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom, size.width + margin.left + margin.right, CCV_INTER_AREA);
	ccv_matrix_free(resize);
	return b;
}

static ccv_array_t* _ccv_scd_collect_negatives(gsl_rng* rng, ccv_size_t size, ccv_array_t* hard_mine, int total, float deform_angle, float deform_scale, float deform_shift, int grayscale)
{
	ccv_array_t* negatives = ccv_array_new(ccv_compute_dense_matrix_size(size.height, size.width, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), total, 0);
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
	ccv_make_array_immutable(negatives);
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
	ccv_make_array_immutable(positives);
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

static void _ccv_scd_run_feature(ccv_dense_matrix_t* a, ccv_scd_feature_t* feature, float surf[32])
{
	ccv_dense_matrix_t* b = 0;
	ccv_scd(a, &b, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(b);
	_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, surf);
	ccv_matrix_free(sat);
}

static void _ccv_scd_liblinear_null(const char* str) { /* do nothing */ }

static void _ccv_scd_feature_supervised_train(gsl_rng* rng, ccv_array_t* features, ccv_array_t* positives, double* pw, ccv_array_t* negatives, double* nw, int active_set, int wide_set, double C)
{
	int i;
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
	parallel_for(i, features->rnum) {
		int j, k;
		if ((i + 1) % 31 == 1 || (i + 1) == features->rnum)
			FLUSH(CCV_CLI_INFO, " - supervised train feature %d / %d with logistic regression", (int)(i + 1), features->rnum);
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
	} parallel_endfor
	ccfree(pwidx);
	ccfree(nwidx);
}

static double _ccv_scd_auc(double* s, int posnum, int negnum)
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
	ccfree(sidx);
	a += (double)(negnum - fp_prev) * (posnum + tp_prev) * 0.5;
	return a / ((double)posnum * negnum);
}

static ccv_scd_feature_t _ccv_scd_best_feature_with_auc(double* s, ccv_array_t* features, ccv_array_t* positives, ccv_array_t* negatives)
{
	ccv_scd_feature_t best_feature;
	int i;
	double* sn = (double*)cccalloc(features->rnum * (positives->rnum + negatives->rnum), sizeof(double));
	assert(positives->rnum + negatives->rnum > 0);
	parallel_for(i, positives->rnum + negatives->rnum) {
		int j, k;
		if ((i + 1) % 1111 == 1 || (i + 1) == positives->rnum + negatives->rnum)
			FLUSH(CCV_CLI_INFO, " - go through %d / %d (%.1f%%) for auc", (int)(i + 1), positives->rnum + negatives->rnum, (float)(i + 1) * 100 / (positives->rnum + negatives->rnum));
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(i < positives->rnum ? ccv_array_get(positives, i) : ccv_array_get(negatives, i - positives->rnum));
		a->data.u8 = (unsigned char*)(a + 1);
		ccv_dense_matrix_t* b = 0;
		ccv_scd(a, &b, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(b);
		float surf[32];
		for (j = 0; j < features->rnum; j++)
		{
			ccv_scd_feature_t* feature = (ccv_scd_feature_t*)ccv_array_get(features, j);
			_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, surf);
			float v = feature->bias;
			for (k = 0; k < 32; k++)
				v += surf[k] * feature->w[k];
			v = expf(v);
			sn[i + j * (positives->rnum + negatives->rnum)] = s[i] + (v - 1) / (v + 1); // probability
		}
		ccv_matrix_free(sat);
	} parallel_endfor
	double max_auc = _ccv_scd_auc(sn, positives->rnum, negatives->rnum);
	int j = 0;
	for (i = 1; i < features->rnum; i++)
	{
		double auc = _ccv_scd_auc(sn + i * (positives->rnum + negatives->rnum), positives->rnum, negatives->rnum);
		if (auc > max_auc)
		{
			max_auc = auc;
			j = i;
		}
	}
	memcpy(&best_feature, ccv_array_get(features, j), sizeof(ccv_scd_feature_t));
	ccfree(sn);
	return best_feature;
}

static float _ccv_scd_threshold_at_hit_rate(double* s, int posnum, int negnum, float hit_rate, float* tp_out, float* fp_out)
{
	ccv_scd_value_index_t* psidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * posnum);
	int i;
	for (i = 0; i < posnum; i++)
		psidx[i].value = s[i], psidx[i].index = i;
	_ccv_scd_value_index_qsort(psidx, posnum, 0);
	float threshold = psidx[(int)((posnum - 0.5) * hit_rate - 0.5)].value - 1e-6;
	ccfree(psidx);
	int tp = 0;
	for (i = 0; i < posnum; i++)
		if (s[i] > threshold)
			++tp;
	int fp = 0;
	for (i = 0; i < negnum; i++)
		if (s[i + posnum] > threshold)
			++fp;
	if (tp_out)
		*tp_out = (float)tp / posnum;
	if (fp_out)
		*fp_out = (float)fp / negnum;
	return threshold;
}

static int _ccv_scd_classifier_cascade_pass(ccv_scd_classifier_cascade_t* cascade, ccv_dense_matrix_t* a)
{
	float surf[32];
	ccv_dense_matrix_t* b = 0;
	ccv_scd(a, &b, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(b);
	int pass = 1;
	int i, j, k;
	for (i = 0; i < cascade->count; i++)
	{
		ccv_scd_classifier_t* classifier = cascade->classifiers + i;
		float v = 0;
		for (j = 0; j < classifier->count; j++)
		{
			ccv_scd_feature_t* feature = classifier->features + j;
			_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, surf);
			float u = feature->bias;
			for (k = 0; k < 32; k++)
				u += surf[k] * feature->w[k];
			u = expf(u);
			v += (u - 1) / (u + 1);
		}
		if (v <= classifier->threshold)
		{
			pass = 0;
			break;
		}
	}
	ccv_matrix_free(sat);
	return pass;
}

static ccv_array_t* _ccv_scd_hard_mining(gsl_rng* rng, ccv_scd_classifier_cascade_t* cascade, ccv_array_t* hard_mine, ccv_array_t* negatives, int negative_count, int grayscale)
{
	ccv_array_t* hard_negatives = ccv_array_new(ccv_compute_dense_matrix_size(cascade->size.height, cascade->size.width, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), negative_count, 0);
	int i, j, t;
	for (i = 0; i < negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(negatives, i);
		a->data.u8 = (unsigned char*)(a + 1);
		if (_ccv_scd_classifier_cascade_pass(cascade, a))
			ccv_array_push(hard_negatives, a);
	}
	int n_per_mine = ccv_max((negative_count - hard_negatives->rnum) / hard_mine->rnum, 10);
	// the hard mining comes in following fashion:
	// 1). original, with n_per_mine set;
	// 2). horizontal flip, with n_per_mine set;
	// 3). vertical flip, with n_per_mine set;
	// 4). 180 rotation, with n_per_mine set;
	// 5~8). repeat above, but with no n_per_mine set;
	// after above, if we still cannot collect enough, so be it.
	for (t = 0; t < 8 /* exhausted all variations */ && hard_negatives->rnum < negative_count; t++)
	{
		for (i = 0; i < hard_mine->rnum; i++)
		{
			FLUSH(CCV_CLI_INFO, " - hard mine negatives %d%% with %d-th permutation", 100 * hard_negatives->rnum / negative_count, t + 1);
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(hard_mine, i);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
			if (image == 0)
			{
				PRINT(CCV_CLI_ERROR, "\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
				continue;
			}
			if (t % 2 != 0)
				ccv_flip(image, 0, 0, CCV_FLIP_X);
			if (t % 4 >= 2)
				ccv_flip(image, 0, 0, CCV_FLIP_Y);
			if (t >= 4)
				n_per_mine = negative_count; // no hard limit on n_per_mine anymore for the last pass
			ccv_scd_param_t params = {
				.interval = 3,
				.min_neighbors = 0,
				.flags = 0,
				.step_through = 4,
				.size = ccv_size(40, 40),
			};
			ccv_array_t* objects = ccv_scd_detect_objects(image, &cascade, 1, params);
			if (objects->rnum > 0)
			{
				gsl_ran_shuffle(rng, objects->data, objects->rnum, objects->rsize);
				for (j = 0; j < ccv_min(objects->rnum, n_per_mine); j++)
				{
					ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(objects, j);
					if (rect->x < 0 || rect->y < 0 || rect->x + rect->width > image->cols || rect->y + rect->height > image->rows)
						continue;
					ccv_dense_matrix_t* sliced = 0;
					ccv_slice(image, (ccv_matrix_t**)&sliced, 0, rect->y, rect->x, rect->height, rect->width);
					ccv_dense_matrix_t* resized = 0;
					assert(sliced->rows >= cascade->size.height && sliced->cols >= cascade->size.width);
					if (sliced->rows > cascade->size.height || sliced->cols > cascade->size.width)
					{
						ccv_resample(sliced, &resized, 0, cascade->size.height, cascade->size.width, CCV_INTER_CUBIC);
						ccv_matrix_free(sliced);
					} else {
						resized = sliced;
					}
					if (_ccv_scd_classifier_cascade_pass(cascade, resized))
						ccv_array_push(hard_negatives, resized);
					ccv_matrix_free(resized);
					if (hard_negatives->rnum >= negative_count)
						break;
				}
			}
			ccv_matrix_free(image);
			if (hard_negatives->rnum >= negative_count)
				break;
		}
	}
	FLUSH(CCV_CLI_INFO, " - hard mine negatives : %d\n", hard_negatives->rnum);
	ccv_make_array_immutable(hard_negatives);
	return hard_negatives;
}

typedef struct {
	ccv_function_state_reserve_field;
	int t, k;
	int positive_count;
	uint64_t array_signature;
	ccv_array_t* negatives;
	double* s;
	double* pw;
	double* nw;
	double auc_prev;
	double accu_true_positive_rate;
	double accu_false_positive_rate;
	ccv_scd_classifier_cascade_t* cascade;
	ccv_scd_train_param_t params;
} ccv_scd_classifier_cascade_new_function_state_t;

static void _ccv_scd_classifier_cascade_new_function_state_read(const char* filename, ccv_scd_classifier_cascade_new_function_state_t* z)
{
	ccv_scd_classifier_cascade_t* cascade = ccv_scd_classifier_cascade_read(filename);
	if (!cascade)
		return;
	if (z->cascade)
		ccv_scd_classifier_cascade_free(z->cascade);
	z->cascade = cascade;
	assert(z->cascade->size.width == z->params.size.width);
	assert(z->cascade->size.height == z->params.size.height);
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char negative_data_qs[] =
			"SELECT data, rnum, rsize FROM negative_data WHERE id=0;";
		sqlite3_stmt* negative_data_stmt = 0;
		if (SQLITE_OK == sqlite3_prepare_v2(db, negative_data_qs, sizeof(negative_data_qs), &negative_data_stmt, 0))
		{
			if (sqlite3_step(negative_data_stmt) == SQLITE_ROW)
			{
				int rsize = ccv_compute_dense_matrix_size(z->cascade->size.height, z->cascade->size.width, CCV_8U | (z->params.grayscale ? CCV_C1 : CCV_C3));
				int rnum = sqlite3_column_int(negative_data_stmt, 1);
				assert(sqlite3_column_int(negative_data_stmt, 2) == rsize);
				size_t size = sqlite3_column_bytes(negative_data_stmt, 0);
				assert(size == (size_t)rsize * rnum);
				if (z->negatives)
					ccv_array_clear(z->negatives);
				else
					z->negatives = ccv_array_new(rsize, rnum, 0);
				int i;
				const uint8_t* data = (const uint8_t*)sqlite3_column_blob(negative_data_stmt, 0);
				for (i = 0; i < rnum; i++)
					ccv_array_push(z->negatives, data + (off_t)i * rsize);
				ccv_make_array_immutable(z->negatives);
				z->array_signature = z->negatives->sig;
			}
			sqlite3_finalize(negative_data_stmt);
		}
		const char function_state_qs[] =
			"SELECT t, k, positive_count, auc_prev, " // 4
			"accu_true_positive_rate, accu_false_positive_rate, " // 6
			"line_no, s, pw, nw FROM function_state WHERE fsid = 0;"; // 10
		sqlite3_stmt* function_state_stmt = 0;
		if (SQLITE_OK == sqlite3_prepare_v2(db, function_state_qs, sizeof(function_state_qs), &function_state_stmt, 0))
		{
			if (sqlite3_step(function_state_stmt) == SQLITE_ROW)
			{
				z->t = sqlite3_column_int(function_state_stmt, 0);
				z->k = sqlite3_column_int(function_state_stmt, 1);
				z->positive_count = sqlite3_column_int(function_state_stmt, 2);
				z->auc_prev = sqlite3_column_double(function_state_stmt, 3);
				z->accu_true_positive_rate = sqlite3_column_double(function_state_stmt, 4);
				z->accu_false_positive_rate = sqlite3_column_double(function_state_stmt, 5);
				z->line_no = sqlite3_column_int(function_state_stmt, 6);
				size_t size = sqlite3_column_bytes(function_state_stmt, 7);
				const void* s = sqlite3_column_blob(function_state_stmt, 7);
				memcpy(z->s, s, size);
				size = sqlite3_column_bytes(function_state_stmt, 8);
				const void* pw = sqlite3_column_blob(function_state_stmt, 8);
				memcpy(z->pw, pw, size);
			}
			sqlite3_finalize(function_state_stmt);
		}
		sqlite3_close(db);
	}
}

static void _ccv_scd_classifier_cascade_new_function_state_write(ccv_scd_classifier_cascade_new_function_state_t* z, const char* filename)
{
	ccv_scd_classifier_cascade_write(z->cascade, filename);
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char function_state_create_table_qs[] =
			"CREATE TABLE IF NOT EXISTS function_state "
			"(fsid INTEGER PRIMARY KEY ASC, t INTEGER, k INTEGER, positive_count INTEGER, auc_prev DOUBLE, accu_true_positive_rate DOUBLE, accu_false_positive_rate DOUBLE, line_no INTEGER, s BLOB, pw BLOB, nw BLOB);"
			"CREATE TABLE IF NOT EXISTS negative_data "
			"(id INTEGER PRIMARY KEY ASC, data BLOB, rnum INTEGER, rsize INTEGER);";
		assert(SQLITE_OK == sqlite3_exec(db, function_state_create_table_qs, 0, 0, 0));
		const char function_state_insert_qs[] =
			"REPLACE INTO function_state "
			"(fsid, t, k, positive_count, auc_prev, accu_true_positive_rate, accu_false_positive_rate, line_no, s, pw, nw) VALUES "
			"(0, $t, $k, $positive_count, $auc_prev, $accu_true_positive_rate, $accu_false_positive_rate, $line_no, $s, $pw, $nw);";
		sqlite3_stmt* function_state_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, function_state_insert_qs, sizeof(function_state_insert_qs), &function_state_insert_stmt, 0));
		sqlite3_bind_int(function_state_insert_stmt, 1, z->t);
		sqlite3_bind_int(function_state_insert_stmt, 2, z->k);
		sqlite3_bind_int(function_state_insert_stmt, 3, z->positive_count);
		sqlite3_bind_double(function_state_insert_stmt, 4, z->auc_prev);
		sqlite3_bind_double(function_state_insert_stmt, 5, z->accu_true_positive_rate);
		sqlite3_bind_double(function_state_insert_stmt, 6, z->accu_false_positive_rate);
		sqlite3_bind_int(function_state_insert_stmt, 7, z->line_no);
		sqlite3_bind_blob(function_state_insert_stmt, 8, z->s, sizeof(double) * (z->positive_count + z->negatives->rnum), SQLITE_STATIC);
		sqlite3_bind_blob(function_state_insert_stmt, 9, z->pw, sizeof(double) * z->positive_count, SQLITE_STATIC);
		sqlite3_bind_blob(function_state_insert_stmt, 10, z->nw, sizeof(double) * z->negatives->rnum, SQLITE_STATIC);
		assert(SQLITE_DONE == sqlite3_step(function_state_insert_stmt));
		sqlite3_finalize(function_state_insert_stmt);
		if (z->array_signature != z->negatives->sig)
		{
			const char negative_data_insert_qs[] =
				"REPLACE INTO negative_data "
				"(id, data, rnum, rsize) VALUES (0, $data, $rnum, $rsize);";
			sqlite3_stmt* negative_data_insert_stmt = 0;
			assert(SQLITE_OK == sqlite3_prepare_v2(db, negative_data_insert_qs, sizeof(negative_data_insert_qs), &negative_data_insert_stmt, 0));
			sqlite3_bind_blob(negative_data_insert_stmt, 1, z->negatives->data, z->negatives->rsize * z->negatives->rnum, SQLITE_STATIC);
			sqlite3_bind_int(negative_data_insert_stmt, 2, z->negatives->rnum);
			sqlite3_bind_int(negative_data_insert_stmt, 3, z->negatives->rsize);
			assert(SQLITE_DONE == sqlite3_step(negative_data_insert_stmt));
			sqlite3_finalize(negative_data_insert_stmt);
			z->array_signature = z->negatives->sig;
		}
		sqlite3_close(db);
	}
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
	PRINT(CCV_CLI_INFO, " - using %d features\n", features->rnum);
	int i, j, p, q;
	ccv_array_t* positives = _ccv_scd_collect_positives(params.size, posfiles, params.grayscale);
	double* h = (double*)ccmalloc(sizeof(double) * (positives->rnum + negative_count));
	ccv_scd_classifier_cascade_new_function_state_t z = {0};
	z.s = (double*)ccmalloc(sizeof(double) * (positives->rnum + negative_count));
	z.pw = (double*)ccmalloc(sizeof(double) * positives->rnum);
	z.nw = (double*)ccmalloc(sizeof(double) * negative_count);
	z.params = params;
	ccv_function_state_begin(_ccv_scd_classifier_cascade_new_function_state_read, z, filename);
	z.positive_count = positives->rnum;
	z.negatives = _ccv_scd_collect_negatives(rng, params.size, hard_mine, negative_count, params.deform.angle, params.deform.scale, params.deform.shift, params.grayscale);
	z.cascade = (ccv_scd_classifier_cascade_t*)ccmalloc(sizeof(ccv_scd_classifier_cascade_t));
	z.cascade->margin = ccv_margin(0, 0, 0, 0);
	z.cascade->size = params.size;
	z.cascade->count = 0;
	z.cascade->classifiers = 0;
	float surf[32];
	z.accu_true_positive_rate = 1;
	z.accu_false_positive_rate = 1;
	ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
	for (z.t = 0; z.t < params.boosting; z.t++)
	{
		for (i = 0; i < positives->rnum; i++)
			z.pw[i] = 0.5 / positives->rnum;
		for (i = 0; i < z.negatives->rnum; i++)
			z.nw[i] = 0.5 / z.negatives->rnum;
		memset(z.s, 0, sizeof(double) * (positives->rnum + z.negatives->rnum));
		z.cascade->classifiers = (ccv_scd_classifier_t*)ccrealloc(z.cascade->classifiers, sizeof(ccv_scd_classifier_t) * (z.t + 1));
		ccv_scd_classifier_t* classifier = z.cascade->classifiers + z.t;
		classifier->threshold = 0;
		classifier->features = 0;
		classifier->count = 0;
		z.auc_prev = 0;
		assert(positives->rnum > 0 && z.negatives->rnum > 0);
		// for the first light stages, we have more restrictive number of features (faster)
		for (z.k = 0; z.k < (z.t < params.stop_criteria.light_stage ? params.stop_criteria.light_feature : params.stop_criteria.maximum_feature); z.k++)
		{
			ccv_scd_classifier_t* classifier = z.cascade->classifiers + z.t;
			classifier->features = (ccv_scd_feature_t*)ccrealloc(classifier->features, sizeof(ccv_scd_feature_t) * (z.k + 1));
			_ccv_scd_feature_supervised_train(rng, features, positives, z.pw, z.negatives, z.nw, params.feature.active_set, params.feature.wide_set, params.C);
			ccv_scd_feature_t best_feature = _ccv_scd_best_feature_with_auc(z.s, features, positives, z.negatives);
			for (i = 0; i < positives->rnum + z.negatives->rnum; i++)
			{
				ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(i < positives->rnum ? ccv_array_get(positives, i) : ccv_array_get(z.negatives, i - positives->rnum));
				a->data.u8 = (unsigned char*)(a + 1);
				_ccv_scd_run_feature(a, &best_feature, surf);
				float v = best_feature.bias;
				for (j = 0; j < 32; j++)
					v += best_feature.w[j] * surf[j];
				v = expf(v);
				h[i] = (v - 1) / (v + 1);
			}
			// compute the total score so far
			for (i = 0; i < positives->rnum + z.negatives->rnum; i++)
				z.s[i] += h[i];
			// compute AUC
			double auc = _ccv_scd_auc(z.s, positives->rnum, z.negatives->rnum);
			FLUSH(CCV_CLI_INFO, " - at %d-th iteration, auc: %lf\n", z.k + 1, auc);
			if (auc - z.auc_prev < params.stop_criteria.auc_crit)
			{
				z.auc_prev = auc;
				break;
			}
			z.auc_prev = auc;
			PRINT(CCV_CLI_INFO, " --- pick feature %s @ (%d, %d, %d, %d)\n", ((best_feature.dy[3] == best_feature.dy[0] ? "4x1" : (best_feature.dx[3] == best_feature.dx[0] ? "1x4" : "2x2"))), best_feature.sx[0], best_feature.sy[0], best_feature.dx[3], best_feature.dy[3]);
			classifier->features[z.k] = best_feature;
			classifier->count = z.k + 1;
			// re-weight, with Gentle AdaBoost
			for (i = 0; i < positives->rnum; i++)
				z.pw[i] *= exp(-h[i]);
			for (i = 0; i < z.negatives->rnum; i++)
				z.nw[i] *= exp(h[i + positives->rnum]);
			// re-normalize
			double w = 0;
			for (i = 0; i < positives->rnum; i++)
				w += z.pw[i];
			w = 0.5 / w;
			for (i = 0; i < positives->rnum; i++)
				z.pw[i] *= w;
			w = 0;
			for (i = 0; i < z.negatives->rnum; i++)
				w += z.nw[i];
			w = 0.5 / w;
			for (i = 0; i < z.negatives->rnum; i++)
				z.nw[i] *= w;
			ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
		}
		// backtrack removal
		while (classifier->count > 1)
		{
			double max_auc = 0;
			p = -1;
			for (i = 0; i < classifier->count; i++)
			{
				ccv_scd_feature_t* feature = classifier->features + i;
				for (j = 0; j < positives->rnum + z.negatives->rnum; j++)
				{
					ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(j < positives->rnum ? ccv_array_get(positives, j) : ccv_array_get(z.negatives, j - positives->rnum));
					a->data.u8 = (unsigned char*)(a + 1);
					_ccv_scd_run_feature(a, feature, surf);
					float v = feature->bias;
					for (q = 0; q < 32; q++)
						v += feature->w[q]* surf[q];
					v = expf(v);
					h[j] = z.s[j] - (v - 1) / (v + 1);
				}
				double auc = _ccv_scd_auc(h, positives->rnum, z.negatives->rnum);
				if (auc > max_auc)
					max_auc = auc, p = i;
			}
			if (max_auc >= z.auc_prev)
			{
				PRINT(CCV_CLI_INFO, " - remove %d-th feature with new auc %lf\n", p, max_auc);
				ccv_scd_feature_t* feature = classifier->features + p;
				for (j = 0; j < positives->rnum + z.negatives->rnum; j++)
				{
					ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(j < positives->rnum ? ccv_array_get(positives, j) : ccv_array_get(z.negatives, j - positives->rnum));
					a->data.u8 = (unsigned char*)(a + 1);
					_ccv_scd_run_feature(a, feature, surf);
					float v = feature->bias;
					for (q = 0; q < 32; q++)
						v += feature->w[q] * surf[q];
					v = expf(v);
					z.s[j] -= (v - 1) / (v + 1);
				}
				z.auc_prev = _ccv_scd_auc(z.s, positives->rnum, z.negatives->rnum);
				--classifier->count;
				if (p < classifier->count)
					memmove(classifier->features + p + 1, classifier->features + p, sizeof(ccv_scd_feature_t) * (classifier->count - p));
			} else
				break;
		}
		float true_positive_rate = 0;
		float false_positive_rate = 0;
		classifier->threshold = _ccv_scd_threshold_at_hit_rate(z.s, positives->rnum, z.negatives->rnum, params.stop_criteria.hit_rate, &true_positive_rate, &false_positive_rate);
		z.accu_true_positive_rate *= true_positive_rate;
		z.accu_false_positive_rate *= false_positive_rate;
		PRINT(CCV_CLI_INFO, " - %d-th stage classifier TP rate : %f, FP rate : %f, ATP rate : %lf, AFP rate : %lg, at threshold : %f\n", z.t + 1, true_positive_rate, false_positive_rate, z.accu_true_positive_rate, z.accu_false_positive_rate, classifier->threshold);
		z.cascade->count = z.t + 1;
		if (z.accu_false_positive_rate < params.stop_criteria.false_positive_rate)
			break;
		if (z.t < params.boosting - 1)
		{
			ccv_array_t* hard_negatives = _ccv_scd_hard_mining(rng, z.cascade, hard_mine, z.negatives, negative_count, params.grayscale);
			ccv_array_free(z.negatives);
			z.negatives = hard_negatives;
		}
		ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
	}
	ccv_array_free(features);
	ccv_array_free(positives);
	ccv_array_free(z.negatives);
	ccfree(h);
	ccfree(z.s);
	ccfree(z.pw);
	ccfree(z.nw);
	gsl_rng_free(rng);
	ccv_function_state_finish();
	return z.cascade;
#else
	assert(0 && "ccv_scd_classifier_cascade_new requires GSL library and liblinear support");
	return 0;
#endif
}

void ccv_scd_classifier_cascade_write(ccv_scd_classifier_cascade_t* cascade, const char* filename)
{
	sqlite3* db = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char create_table_qs[] =
			"CREATE TABLE IF NOT EXISTS cascade_params "
			"(id INTEGER PRIMARY KEY ASC, count INTEGER, "
			"margin_left INTEGER, margin_top INTEGER, margin_right INTEGER, margin_bottom INTEGER, "
			"size_width INTEGER, size_height INTEGER);"
			"CREATE TABLE IF NOT EXISTS classifier_params "
			"(classifier INTEGER PRIMARY KEY ASC, count INTEGER, threshold DOUBLE);"
			"CREATE TABLE IF NOT EXISTS feature_params "
			"(classifier INTEGER, id INTEGER, "
			"sx_0 INTEGER, sy_0 INTEGER, dx_0 INTEGER, dy_0 INTEGER, "
			"sx_1 INTEGER, sy_1 INTEGER, dx_1 INTEGER, dy_1 INTEGER, "
			"sx_2 INTEGER, sy_2 INTEGER, dx_2 INTEGER, dy_2 INTEGER, "
			"sx_3 INTEGER, sy_3 INTEGER, dx_3 INTEGER, dy_3 INTEGER, "
			"bias DOUBLE, w BLOB, UNIQUE (classifier, id));";
		assert(SQLITE_OK == sqlite3_exec(db, create_table_qs, 0, 0, 0));
		const char cascade_params_insert_qs[] = 
			"REPLACE INTO cascade_params "
			"(id, count, "
			"margin_left, margin_top, margin_right, margin_bottom, "
			"size_width, size_height) VALUES "
			"(0, $count, " // 0
			"$margin_left, $margin_top, $margin_bottom, $margin_right, " // 4
			"$size_width, $size_height);"; // 6
		sqlite3_stmt* cascade_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, cascade_params_insert_qs, sizeof(cascade_params_insert_qs), &cascade_params_insert_stmt, 0));
		sqlite3_bind_int(cascade_params_insert_stmt, 1, cascade->count);
		sqlite3_bind_int(cascade_params_insert_stmt, 2, cascade->margin.left);
		sqlite3_bind_int(cascade_params_insert_stmt, 3, cascade->margin.top);
		sqlite3_bind_int(cascade_params_insert_stmt, 4, cascade->margin.right);
		sqlite3_bind_int(cascade_params_insert_stmt, 5, cascade->margin.bottom);
		sqlite3_bind_int(cascade_params_insert_stmt, 6, cascade->size.width);
		sqlite3_bind_int(cascade_params_insert_stmt, 7, cascade->size.height);
		assert(SQLITE_DONE == sqlite3_step(cascade_params_insert_stmt));
		sqlite3_finalize(cascade_params_insert_stmt);
		const char classifier_params_insert_qs[] = 
			"REPLACE INTO classifier_params "
			"(classifier, count, threshold) VALUES "
			"($classifier, $count, $threshold);";
		sqlite3_stmt* classifier_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, classifier_params_insert_qs, sizeof(classifier_params_insert_qs), &classifier_params_insert_stmt, 0));
		const char feature_params_insert_qs[] =
			"REPLACE INTO feature_params "
			"(classifier, id, "
			"sx_0, sy_0, dx_0, dy_0, "
			"sx_1, sy_1, dx_1, dy_1, "
			"sx_2, sy_2, dx_2, dy_2, "
			"sx_3, sy_3, dx_3, dy_3, "
			"bias, w) VALUES "
			"($classifier, $id, " // 1
			"$sx_0, $sy_0, $dx_0, $dy_0, " // 5
			"$sx_1, $sy_1, $dx_1, $dy_1, " // 9
			"$sx_2, $sy_2, $dx_2, $dy_2, " // 13
			"$sx_3, $sy_3, $dx_3, $dy_3, " // 17
			"$bias, $w);"; // 19
		sqlite3_stmt* feature_params_insert_stmt = 0;
		assert(SQLITE_OK == sqlite3_prepare_v2(db, feature_params_insert_qs, sizeof(feature_params_insert_qs), &feature_params_insert_stmt, 0));
		int i, j, k;
		for (i = 0; i < cascade->count; i++)
		{
			ccv_scd_classifier_t* classifier = cascade->classifiers + i;
			sqlite3_bind_int(classifier_params_insert_stmt, 1, i);
			sqlite3_bind_int(classifier_params_insert_stmt, 2, classifier->count);
			sqlite3_bind_double(classifier_params_insert_stmt, 3, classifier->threshold);
			assert(SQLITE_DONE == sqlite3_step(classifier_params_insert_stmt));
			sqlite3_reset(classifier_params_insert_stmt);
			sqlite3_clear_bindings(classifier_params_insert_stmt);
			for (j = 0; j < classifier->count; j++)
			{
				ccv_scd_feature_t* feature = classifier->features + j;
				sqlite3_bind_int(feature_params_insert_stmt, 1, i);
				sqlite3_bind_int(feature_params_insert_stmt, 2, j);
				for (k = 0; k < 4; k++)
				{
					sqlite3_bind_int(feature_params_insert_stmt, 3 + k * 4, feature->sx[k]);
					sqlite3_bind_int(feature_params_insert_stmt, 4 + k * 4, feature->sy[k]);
					sqlite3_bind_int(feature_params_insert_stmt, 5 + k * 4, feature->dx[k]);
					sqlite3_bind_int(feature_params_insert_stmt, 6 + k * 4, feature->dy[k]);
				}
				sqlite3_bind_double(feature_params_insert_stmt, 19, feature->bias);
				sqlite3_bind_blob(feature_params_insert_stmt, 20, feature->w, sizeof(float) * 32, SQLITE_STATIC);
				assert(SQLITE_DONE == sqlite3_step(feature_params_insert_stmt));
				sqlite3_reset(feature_params_insert_stmt);
				sqlite3_clear_bindings(feature_params_insert_stmt);
			}
		}
		sqlite3_finalize(classifier_params_insert_stmt);
		sqlite3_finalize(feature_params_insert_stmt);
		sqlite3_close(db);
	}
}

ccv_scd_classifier_cascade_t* ccv_scd_classifier_cascade_read(const char* filename)
{
	int i;
	sqlite3* db = 0;
	ccv_scd_classifier_cascade_t* cascade = 0;
	if (SQLITE_OK == sqlite3_open(filename, &db))
	{
		const char cascade_params_qs[] =
			"SELECT count, " // 1
			"margin_left, margin_top, margin_right, margin_bottom, " // 5
			"size_width, size_height FROM cascade_params WHERE id = 0;"; // 7
		sqlite3_stmt* cascade_params_stmt = 0;
		if (SQLITE_OK == sqlite3_prepare_v2(db, cascade_params_qs, sizeof(cascade_params_qs), &cascade_params_stmt, 0))
		{
			if (sqlite3_step(cascade_params_stmt) == SQLITE_ROW)
			{
				cascade = (ccv_scd_classifier_cascade_t*)ccmalloc(sizeof(ccv_scd_classifier_cascade_t));
				cascade->count = sqlite3_column_int(cascade_params_stmt, 0);
				cascade->classifiers = (ccv_scd_classifier_t*)cccalloc(cascade->count, sizeof(ccv_scd_classifier_t));
				cascade->margin = ccv_margin(sqlite3_column_int(cascade_params_stmt, 1), sqlite3_column_int(cascade_params_stmt, 2), sqlite3_column_int(cascade_params_stmt, 3), sqlite3_column_int(cascade_params_stmt, 4));
				cascade->size = ccv_size(sqlite3_column_int(cascade_params_stmt, 5), sqlite3_column_int(cascade_params_stmt, 6));
			}
			sqlite3_finalize(cascade_params_stmt);
		}
		if (cascade)
		{
			const char classifier_params_qs[] =
				"SELECT classifier, count, threshold FROM classifier_params ORDER BY classifier ASC;";
			sqlite3_stmt* classifier_params_stmt = 0;
			if (SQLITE_OK == sqlite3_prepare_v2(db, classifier_params_qs, sizeof(classifier_params_qs), &classifier_params_stmt, 0))
			{
				while (sqlite3_step(classifier_params_stmt) == SQLITE_ROW)
					if (sqlite3_column_int(classifier_params_stmt, 0) < cascade->count)
					{
						ccv_scd_classifier_t* classifier = cascade->classifiers + sqlite3_column_int(classifier_params_stmt, 0);
						classifier->count = sqlite3_column_int(classifier_params_stmt, 1);
						classifier->features = (ccv_scd_feature_t*)ccmalloc(sizeof(ccv_scd_feature_t) * classifier->count);
						classifier->threshold = (float)sqlite3_column_double(classifier_params_stmt, 2);
					}
				sqlite3_finalize(classifier_params_stmt);
			}
			const char feature_params_qs[] =
				"SELECT classifier, id, "
				"sx_0, sy_0, dx_0, dy_0, "
				"sx_1, sy_1, dx_1, dy_1, "
				"sx_2, sy_2, dx_2, dy_2, "
				"sx_3, sy_3, dx_3, dy_3, "
				"bias, w FROM feature_params ORDER BY classifier, id ASC;";
			sqlite3_stmt* feature_params_stmt = 0;
			if (SQLITE_OK == sqlite3_prepare_v2(db, feature_params_qs, sizeof(feature_params_qs), &feature_params_stmt, 0))
			{
				while (sqlite3_step(feature_params_stmt) == SQLITE_ROW)
					if (sqlite3_column_int(feature_params_stmt, 0) < cascade->count)
					{
						ccv_scd_classifier_t* classifier = cascade->classifiers + sqlite3_column_int(feature_params_stmt, 0);
						if (sqlite3_column_int(feature_params_stmt, 1) < classifier->count)
						{
							ccv_scd_feature_t* feature = classifier->features + sqlite3_column_int(feature_params_stmt, 1);
							for (i = 0; i < 4; i++)
							{
								feature->sx[i] = sqlite3_column_int(feature_params_stmt, 2 + i * 4);
								feature->sy[i] = sqlite3_column_int(feature_params_stmt, 3 + i * 4);
								feature->dx[i] = sqlite3_column_int(feature_params_stmt, 4 + i * 4);
								feature->dy[i] = sqlite3_column_int(feature_params_stmt, 5 + i * 4);
							}
							feature->bias = (float)sqlite3_column_double(feature_params_stmt, 18);
							int wnum = sqlite3_column_bytes(feature_params_stmt, 19);
							assert(wnum == 32 * sizeof(float));
							const void* w = sqlite3_column_blob(feature_params_stmt, 19);
							memcpy(feature->w, w, sizeof(float) * 32);
						}
					}
				sqlite3_finalize(feature_params_stmt);
			}
		}
		sqlite3_close(db);
	}
	return cascade;
}

void ccv_scd_classifier_cascade_free(ccv_scd_classifier_cascade_t* cascade)
{
	int i;
	for (i = 0; i < cascade->count; i++)
	{
		ccv_scd_classifier_t* classifier = cascade->classifiers + i;
		ccfree(classifier->features);
	}
	ccfree(cascade->classifiers);
	ccfree(cascade);
}

static int _ccv_is_equal_same_class(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	int distance = (int)(ccv_min(r1->rect.width, r1->rect.height) * 0.25 + 0.5);

	return r2->classification.id == r1->classification.id &&
		r2->rect.x <= r1->rect.x + distance &&
		r2->rect.x >= r1->rect.x - distance &&
		r2->rect.y <= r1->rect.y + distance &&
		r2->rect.y >= r1->rect.y - distance &&
		r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		(int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width &&
		r2->rect.height <= (int)(r1->rect.height * 1.5 + 0.5) &&
		(int)(r2->rect.height * 1.5 + 0.5) >= r1->rect.height;
}

ccv_array_t* ccv_scd_detect_objects(ccv_dense_matrix_t* a, ccv_scd_classifier_cascade_t** cascades, int count, ccv_scd_param_t params)
{
	int i, j, k, x, y, p, q, r;
	int scale_upto = 1;
	float up_ratio = 1.0;
	for (i = 0; i < count; i++)
		up_ratio = ccv_max(up_ratio, ccv_max((float)cascades[i]->size.width / params.size.width, (float)cascades[i]->size.height / params.size.height));
	if (up_ratio - 1.0 > 1e-4)
	{
		ccv_dense_matrix_t* resized = 0;
		ccv_resample(a, &resized, 0, (int)(a->rows * up_ratio + 0.5), (int)(a->cols * up_ratio + 0.5), CCV_INTER_CUBIC);
		a = resized;
	}
	for (i = 0; i < count; i++)
		scale_upto = ccv_max(scale_upto, (int)(log(ccv_min((double)a->rows / (cascades[i]->size.height - cascades[i]->margin.top - cascades[i]->margin.bottom), (double)a->cols / (cascades[i]->size.width - cascades[i]->margin.left - cascades[i]->margin.right))) / log(2.) - DBL_MIN) + 1);
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * scale_upto);
	pyr[0] = a;
	for (i = 1; i < scale_upto; i++)
	{
		pyr[i] = 0;
		ccv_sample_down(pyr[i - 1], &pyr[i], 0, 0, 0);
	}
	float surf[32];
	ccv_array_t** seq = (ccv_array_t**)alloca(sizeof(ccv_array_t*) * count);
	for (i = 0; i < count; i++)
		seq[i] = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	for (i = 0; i < scale_upto; i++)
	{
		// run it
		for (j = 0; j < count; j++)
		{
			double scale_ratio = pow(2., 1. / (params.interval + 1));
			double scale = 1;
			ccv_scd_classifier_cascade_t* cascade = cascades[j];
			for (k = 0; k <= params.interval; k++)
			{
				int rows = (int)(pyr[i]->rows / scale + 0.5);
				int cols = (int)(pyr[i]->cols / scale + 0.5);
				if (rows < cascade->size.height || cols < cascade->size.width)
					break;
				ccv_dense_matrix_t* image = k == 0 ? pyr[i] : 0;
				if (k > 0)
					ccv_resample(pyr[i], &image, 0, rows, cols, CCV_INTER_AREA);
				ccv_dense_matrix_t* bordered = 0;
				ccv_border(image, (ccv_matrix_t**)&bordered, 0, cascade->margin);
				if (k > 0)
					ccv_matrix_free(image);
				ccv_dense_matrix_t* scd = 0;
				ccv_scd(bordered, &scd, 0);
				ccv_matrix_free(bordered);
				ccv_dense_matrix_t* sat = 0;
				ccv_sat(scd, &sat, 0, CCV_PADDING_ZERO);
				assert(CCV_GET_CHANNEL(sat->type) == 8);
				ccv_matrix_free(scd);
				float* ptr = sat->data.f32;
				for (y = 0; y < rows; y += params.step_through)
				{
					if (y >= sat->rows - cascade->size.height - 1)
						break;
					for (x = 0; x < cols; x += params.step_through)
					{
						if (x >= sat->cols - cascade->size.width - 1)
							break;
						int pass = 1;
						float sum = 0;
						for (p = 0; p < cascade->count; p++)
						{
							ccv_scd_classifier_t* classifier = cascade->classifiers + p;
							float v = 0;
							for (q = 0; q < classifier->count; q++)
							{
								ccv_scd_feature_t* feature = classifier->features + q;
								_ccv_scd_run_feature_at(ptr + x * 8, sat->cols, feature, surf);
								float u = feature->bias;
								for (r = 0; r < 32; r++)
									u += surf[r] * feature->w[r];
								u = expf(u);
								v += (u - 1) / (u + 1);
							}
							if (v <= classifier->threshold)
							{
								pass = 0;
								break;
							}
							sum = v / classifier->count;
						}
						if (pass)
						{
							ccv_comp_t comp;
							comp.rect = ccv_rect((int)((x + 0.5) * (scale / up_ratio) * (1 << i) - 0.5),
												 (int)((y + 0.5) * (scale / up_ratio) * (1 << i) - 0.5),
												 (cascade->size.width - cascade->margin.left - cascade->margin.right) * (scale / up_ratio) * (1 << i),
												 (cascade->size.height - cascade->margin.top - cascade->margin.bottom) * (scale / up_ratio) * (1 << i));
							comp.neighbors = 1;
							comp.classification.id = j + 1;
							comp.classification.confidence = sum + (cascade->count - 1);
							ccv_array_push(seq[j], &comp);
						}
					}
					ptr += sat->cols * 8 * params.step_through;
				}
				ccv_matrix_free(sat);
				scale *= scale_ratio;
			}
		}
	}

	for (i = 1; i < scale_upto; i++)
		ccv_matrix_free(pyr[i]);
	if (up_ratio - 1.0 > 1e-4)
		ccv_matrix_free(a);

	ccv_array_t* result_seq = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	ccv_array_t* seq2 = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	for (k = 0; k < count; k++)
	{
		/* the following code from OpenCV's haar feature implementation */
		if(params.min_neighbors == 0)
		{
			for (i = 0; i < seq[k]->rnum; i++)
			{
				ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq[k], i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			ccv_array_t* idx_seq = 0;
			ccv_array_clear(seq2);
			// group retrieved rectangles in order to filter out noise
			int ncomp = ccv_array_group(seq[k], &idx_seq, _ccv_is_equal_same_class, 0);
			ccv_comp_t* comps = (ccv_comp_t*)cccalloc(ncomp + 1, sizeof(ccv_comp_t));

			// count number of neighbors
			for (i = 0; i < seq[k]->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq[k], i);
				int idx = *(int*)ccv_array_get(idx_seq, i);

				comps[idx].classification.id = r1.classification.id;
				if (r1.classification.confidence > comps[idx].classification.confidence || comps[idx].neighbors == 0)
				{
					comps[idx].rect = r1.rect;
					comps[idx].classification.confidence = r1.classification.confidence;
				}

				++comps[idx].neighbors;
			}

			// calculate average bounding box
			for (i = 0; i < ncomp; i++)
			{
				int n = comps[i].neighbors;
				if (n >= params.min_neighbors)
					ccv_array_push(seq2, comps + i);
			}

			// filter out large object rectangles contains small object rectangles
			for (i = 0; i < seq2->rnum; i++)
			{
				ccv_comp_t* r2 = (ccv_comp_t*)ccv_array_get(seq2, i);
				int distance = (int)(ccv_min(r2->rect.width, r2->rect.height) * 0.25 + 0.5);
				for (j = 0; j < seq2->rnum; j++)
				{
					ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq2, j);
					if (i != j &&
						abs(r1.classification.id) == r2->classification.id &&
						r1.rect.x >= r2->rect.x - distance &&
						r1.rect.y >= r2->rect.y - distance &&
						r1.rect.x + r1.rect.width <= r2->rect.x + r2->rect.width + distance &&
						r1.rect.y + r1.rect.height <= r2->rect.y + r2->rect.height + distance &&
						// if r1 (the smaller one) is better, mute r2
						(r2->classification.confidence <= r1.classification.confidence && r2->neighbors < r1.neighbors))
					{
						r2->classification.id = -r2->classification.id;
						break;
					}
				}
			}

			// filter out small object rectangles inside large object rectangles
			for (i = 0; i < seq2->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq2, i);
				if (r1.classification.id > 0)
				{
					int flag = 1;

					for (j = 0; j < seq2->rnum; j++)
					{
						ccv_comp_t r2 = *(ccv_comp_t*)ccv_array_get(seq2, j);
						int distance = (int)(ccv_min(r2.rect.width, r2.rect.height) * 0.25 + 0.5);

						if (i != j &&
							abs(r1.classification.id) == abs(r2.classification.id) &&
							r1.rect.x >= r2.rect.x - distance &&
							r1.rect.y >= r2.rect.y - distance &&
							r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
							r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
							// if r2 is better, we mute r1
							(r2.classification.confidence > r1.classification.confidence || r2.neighbors >= r1.neighbors))
						{
							flag = 0;
							break;
						}
					}

					if (flag)
						ccv_array_push(result_seq, &r1);
				}
			}
			ccv_array_free(idx_seq);
			ccfree(comps);
		}
		ccv_array_free(seq[k]);
	}
	ccv_array_free(seq2);

	return result_seq;
}
