#include "ccv.h"
#include "ccv_internal.h"
#if defined(HAVE_SSE2)
#include <xmmintrin.h>
#elif defined(HAVE_NEON)
#include <arm_neon.h>
#endif
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#include "3rdparty/sqlite3/sqlite3.h"

const ccv_scd_param_t ccv_scd_default_params = {
	.interval = 5,
	.min_neighbors = 1,
	.step_through = 4,
	.size = {
		.width = 48,
		.height = 48,
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
	ccv_dense_matrix_t* blur = 0;
	ccv_blur(a, &blur, 0, 0.5); // do a modest blur, which suppresses noise
	ccv_dense_matrix_t* dx = 0;
	ccv_sobel(blur, &dx, 0, 1, 0);
	ccv_dense_matrix_t* dy = 0;
	ccv_sobel(blur, &dy, 0, 0, 1);
	ccv_dense_matrix_t* du = 0;
	ccv_sobel(blur, &du, 0, 1, 1);
	ccv_dense_matrix_t* dv = 0;
	ccv_sobel(blur, &dv, 0, -1, 1);
	ccv_matrix_free(blur);
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
				dbp[0] = fdx, dbp[1] = fdy; \
				dbp[2] = fdu, dbp[3] = fdv; \
				dbp[4] = adx, dbp[5] = ady; \
				dbp[6] = adu, dbp[7] = adv; \
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
				dbp[0] = fdx, dbp[1] = fdy; \
				dbp[2] = fdu, dbp[3] = fdv; \
				dbp[4] = adx, dbp[5] = ady; \
				dbp[6] = adu, dbp[7] = adv; \
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

#if defined(HAVE_SSE2)
static inline void _ccv_scd_run_feature_at_sse2(float* at, int cols, ccv_scd_stump_feature_t* feature, __m128 surf[8])
{
	int i;
	// extract feature
	for (i = 0; i < 4; i++)
	{
		__m128 d0 = _mm_loadu_ps(at + (cols * feature->sy[i] + feature->sx[i]) * 8);
		__m128 d1 = _mm_loadu_ps(at + 4 + (cols * feature->sy[i] + feature->sx[i]) * 8);
		__m128 du0 = _mm_loadu_ps(at + (cols * feature->dy[i] + feature->sx[i]) * 8);
		__m128 du1 = _mm_loadu_ps(at + 4 + (cols * feature->dy[i] + feature->sx[i]) * 8);
		__m128 dv0 = _mm_loadu_ps(at + (cols * feature->sy[i] + feature->dx[i]) * 8);
		__m128 dv1 = _mm_loadu_ps(at + 4 + (cols * feature->sy[i] + feature->dx[i]) * 8);
		__m128 duv0 = _mm_loadu_ps(at + (cols * feature->dy[i] + feature->dx[i]) * 8);
		__m128 duv1 = _mm_loadu_ps(at + 4 + (cols * feature->dy[i] + feature->dx[i]) * 8);
		surf[i * 2] = _mm_sub_ps(_mm_add_ps(duv0, d0), _mm_add_ps(du0, dv0));
		surf[i * 2 + 1] = _mm_sub_ps(_mm_add_ps(duv1, d1), _mm_add_ps(du1, dv1));
	}
	// L2Hys normalization
	__m128 v0 = _mm_add_ps(_mm_mul_ps(surf[0], surf[0]), _mm_mul_ps(surf[1], surf[1]));
	__m128 v1 = _mm_add_ps(_mm_mul_ps(surf[2], surf[2]), _mm_mul_ps(surf[3], surf[3]));
	__m128 v2 = _mm_add_ps(_mm_mul_ps(surf[4], surf[4]), _mm_mul_ps(surf[5], surf[5]));
	__m128 v3 = _mm_add_ps(_mm_mul_ps(surf[6], surf[6]), _mm_mul_ps(surf[7], surf[7]));
	v0 = _mm_add_ps(v0, v1);
	v2 = _mm_add_ps(v2, v3);
	union {
		float f[4];
		__m128 p;
	} vx;
	vx.p = _mm_add_ps(v0, v2);
	v0 = _mm_set1_ps(1.0 / (sqrtf(vx.f[0] + vx.f[1] + vx.f[2] + vx.f[3]) + 1e-6));
	static float thlf = -2.0 / 5.65685424949; // -sqrtf(32)
	static float thuf = 2.0 / 5.65685424949; // sqrtf(32)
	const __m128 thl = _mm_set1_ps(thlf);
	const __m128 thu = _mm_set1_ps(thuf);
	for (i = 0; i < 8; i++)
	{
		surf[i] = _mm_mul_ps(surf[i], v0);
		surf[i] = _mm_min_ps(surf[i], thu);
		surf[i] = _mm_max_ps(surf[i], thl);
	}
	__m128 u0 = _mm_add_ps(_mm_mul_ps(surf[0], surf[0]), _mm_mul_ps(surf[1], surf[1]));
	__m128 u1 = _mm_add_ps(_mm_mul_ps(surf[2], surf[2]), _mm_mul_ps(surf[3], surf[3]));
	__m128 u2 = _mm_add_ps(_mm_mul_ps(surf[4], surf[4]), _mm_mul_ps(surf[5], surf[5]));
	__m128 u3 = _mm_add_ps(_mm_mul_ps(surf[6], surf[6]), _mm_mul_ps(surf[7], surf[7]));
	u0 = _mm_add_ps(u0, u1);
	u2 = _mm_add_ps(u2, u3);
	union {
		float f[4];
		__m128 p;
	} ux;
	ux.p = _mm_add_ps(u0, u2);
	u0 = _mm_set1_ps(1.0 / (sqrtf(ux.f[0] + ux.f[1] + ux.f[2] + ux.f[3]) + 1e-6));
	for (i = 0; i < 8; i++)
		surf[i] = _mm_mul_ps(surf[i], u0);
}
#else
static inline void _ccv_scd_run_feature_at(float* at, int cols, ccv_scd_stump_feature_t* feature, float surf[32])
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
#endif

#ifdef HAVE_GSL
static ccv_array_t* _ccv_scd_collect_negatives(gsl_rng* rng, ccv_size_t size, ccv_array_t* hard_mine, int total, int grayscale)
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
					ccv_rect_t rect;
					double scale_ratio = gsl_rng_uniform(rng) * (max_scale_ratio - 0.5) + 0.5;
					rect.width = ccv_min(image->cols, (int)(size.width * scale_ratio + 0.5));
					rect.height = ccv_min(image->rows, (int)(size.height * scale_ratio + 0.5));
					rect.x = gsl_rng_uniform_int(rng, ccv_max(image->cols - rect.width + 1, 1));
					rect.y = gsl_rng_uniform_int(rng, ccv_max(image->rows - rect.height + 1, 1));
					ccv_dense_matrix_t* sliced = 0;
					ccv_slice(image, (ccv_matrix_t**)&sliced, 0, rect.y, rect.x, rect.height, rect.width);
					ccv_dense_matrix_t* b = 0;
					if (size.width > rect.width)
						ccv_resample(sliced, &b, 0, size.height, size.width, CCV_INTER_CUBIC);
					else
						ccv_resample(sliced, &b, 0, size.height, size.width, CCV_INTER_AREA);
					ccv_matrix_free(sliced);
					b->sig = 0;
					// this leveraged the fact that because I know the ccv_dense_matrix_t is continuous in memory
					ccv_array_push(negatives, b);
					ccv_matrix_free(b);
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

static ccv_array_t* _ccv_scd_stump_features(ccv_size_t base, int range_through, int step_through, ccv_size_t size)
{
	ccv_array_t* features = ccv_array_new(sizeof(ccv_scd_stump_feature_t), 64, 0);
	int x, y, w, h;
	for (w = base.width; w <= size.width; w += range_through)
		if (w % 4 == 0) // only allow 4:1
		{
			h = w / 4;
			for (x = 0; x <= size.width - w; x += step_through)
				for (y = 0; y <= size.height - h; y += step_through)
				{
					// 4x1 feature
					ccv_scd_stump_feature_t feature;
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
					ccv_scd_stump_feature_t feature;
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
						ccv_scd_stump_feature_t feature;
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
static CCV_IMPLEMENT_QSORT(_ccv_scd_value_index_sortby_value, ccv_scd_value_index_t, more_than)
#undef more_than
#define less_than(s1, s2, aux) ((s1).index < (s2).index)
static CCV_IMPLEMENT_QSORT(_ccv_scd_value_index_sortby_index, ccv_scd_value_index_t, less_than)
#undef less_than

static float* _ccv_scd_get_surf_at(float* fv, int feature_no, int example_no, int positive_count, int negative_count)
{
	return fv + ((off_t)example_no + feature_no * (positive_count + negative_count)) * 32;
}

static void _ccv_scd_precompute_feature_vectors(const ccv_array_t* features, const ccv_array_t* positives, const ccv_array_t* negatives, float* fv)
{
	parallel_for(i, positives->rnum) {
		int j;
		if ((i + 1) % 4031 == 1)
			FLUSH(CCV_CLI_INFO, " - precompute feature vectors of example %d / %d over %d features", (int)(i + 1), positives->rnum + negatives->rnum, features->rnum);
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(positives, i);
		a->data.u8 = (unsigned char*)(a + 1);
		ccv_dense_matrix_t* b = 0;
		ccv_scd(a, &b, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(b);
		for (j = 0; j < features->rnum; j++)
		{
			ccv_scd_stump_feature_t* feature = (ccv_scd_stump_feature_t*)ccv_array_get(features, j);
			// save to fv
#if defined(HAVE_SSE2)
			_ccv_scd_run_feature_at_sse2(sat->data.f32, sat->cols, feature, (__m128*)_ccv_scd_get_surf_at(fv, j, i, positives->rnum, negatives->rnum));
#else
			_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, _ccv_scd_get_surf_at(fv, j, i, positives->rnum, negatives->rnum));
#endif
		}
		ccv_matrix_free(sat);
	} parallel_endfor
	parallel_for(i, negatives->rnum) {
		int j;
		if ((i + 1) % 731 == 1 || (i + 1) == negatives->rnum)
			FLUSH(CCV_CLI_INFO, " - precompute feature vectors of example %d / %d over %d features", (int)(i + positives->rnum + 1), positives->rnum + negatives->rnum, features->rnum);
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(negatives, i);
		a->data.u8 = (unsigned char*)(a + 1);
		ccv_dense_matrix_t* b = 0;
		ccv_scd(a, &b, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(b);
		for (j = 0; j < features->rnum; j++)
		{
			ccv_scd_stump_feature_t* feature = (ccv_scd_stump_feature_t*)ccv_array_get(features, j);
			// save to fv
#if defined(HAVE_SSE2)
			_ccv_scd_run_feature_at_sse2(sat->data.f32, sat->cols, feature, (__m128*)_ccv_scd_get_surf_at(fv, j, i + positives->rnum, positives->rnum, negatives->rnum));
#else
			_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, _ccv_scd_get_surf_at(fv, j, i + positives->rnum, positives->rnum, negatives->rnum));
#endif
		}
		ccv_matrix_free(sat);
	} parallel_endfor
}

typedef struct {
	int feature_no;
	double C;
	ccv_scd_value_index_t* pwidx;
	ccv_scd_value_index_t* nwidx;
	int positive_count;
	int negative_count;
	int active_positive_count;
	int active_negative_count;
	float* fv;
} ccv_loss_minimize_context_t;

static int _ccv_scd_stump_feature_gentle_adaboost_loss(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void* data)
{
	ccv_loss_minimize_context_t* context = (ccv_loss_minimize_context_t*)data;
	int i, j;
	float loss = 0;
	float* d = df->data.f32;
	for (i = 0; i < 32; i++)
	{
		loss += context->C * fabs(x->data.f32[i]);
		d[i] = x->data.f32[i] > 0 ? context->C : -context->C;
	}
	d[32] = 0;
	float* surf = _ccv_scd_get_surf_at(context->fv, context->feature_no, 0, context->positive_count, context->negative_count);
	for (i = 0; i < context->active_positive_count; i++)
	{
		float* cur_surf = surf + (off_t)(context->pwidx[i].index) * 32;
		float v = x->data.f32[32];
		for (j = 0; j < 32; j++)
			v += cur_surf[j] * x->data.f32[j];
		v = expf(v);
		float tanh = (v - 1) / (v + 1);
		loss += context->pwidx[i].value * (1.0 - tanh) * (1.0 - tanh);
		float dv = -8.0 * context->pwidx[i].value * v / ((1.0 + v) * (1.0 + v) * (1.0 + v));
		for (j = 0; j < 32; j++)
			d[j] += dv * cur_surf[j];
		d[32] += dv;
	}
	for (i = 0; i < context->active_negative_count; i++)
	{
		float* cur_surf = surf + (off_t)(context->nwidx[i].index + context->positive_count) * 32;
		float v = x->data.f32[32];
		for (j = 0; j < 32; j++)
			v += cur_surf[j] * x->data.f32[j];
		v = expf(v);
		float tanh = (v - 1) / (v + 1);
		loss += context->nwidx[i].value * (-1.0 - tanh) * (-1.0 - tanh);
		float dv = 8.0 * context->nwidx[i].value * v * v / ((1.0 + v) * (1.0 + v) * (1.0 + v));
		for (j = 0; j < 32; j++)
			d[j] += dv * cur_surf[j];
		d[32] += dv;
	}
	f[0] = loss;
	return 0;
}

static int _ccv_scd_weight_trimming(ccv_scd_value_index_t* idx, int count, double weight_trimming)
{
	int active_count = count;
	int i;
	double w = 0;
	for (i = 0; i < count; i++)
	{
		w += idx[i].value;
		if (w >= weight_trimming)
		{
			active_count = i + 1;
			break;
		}
	}
	assert(active_count > 0);
	for (i = active_count; i < count; i++)
		if (idx[i - 1].value == idx[i].value) // for exactly the same weight, we skip
			active_count = i + 1;
		else
			break;
	return active_count;
}

static void _ccv_scd_stump_feature_supervised_train(gsl_rng* rng, ccv_array_t* features, int positive_count, int negative_count, double* pw, double* nw, float* fv, double C, double weight_trimming)
{
	int i;
	ccv_scd_value_index_t* pwidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * positive_count);
	ccv_scd_value_index_t* nwidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * negative_count);
	for (i = 0; i < positive_count; i++)
		pwidx[i].value = pw[i], pwidx[i].index = i;
	for (i = 0; i < negative_count; i++)
		nwidx[i].value = nw[i], nwidx[i].index = i;
	_ccv_scd_value_index_sortby_value(pwidx, positive_count, 0);
	_ccv_scd_value_index_sortby_value(nwidx, negative_count, 0);
	int active_positive_count = _ccv_scd_weight_trimming(pwidx, positive_count, weight_trimming * 0.5); // the sum of positive weights is 0.5
	int active_negative_count = _ccv_scd_weight_trimming(nwidx, negative_count, weight_trimming * 0.5); // the sum of negative weights is 0.5
	_ccv_scd_value_index_sortby_index(pwidx, active_positive_count, 0);
	_ccv_scd_value_index_sortby_index(nwidx, active_negative_count, 0);
	parallel_for(i, features->rnum) {
		if ((i + 1) % 31 == 1 || (i + 1) == features->rnum)
			FLUSH(CCV_CLI_INFO, " - supervised train feature %d / %d with logistic regression, active set {%d, %d}", (int)(i + 1), features->rnum, active_positive_count, active_negative_count);
		ccv_scd_stump_feature_t* feature = (ccv_scd_stump_feature_t*)ccv_array_get(features, i);
		ccv_loss_minimize_context_t context = {
			.feature_no = i,
			.C = C,
			.positive_count = positive_count,
			.negative_count = negative_count,
			.active_positive_count = active_positive_count,
			.active_negative_count = active_negative_count,
			.pwidx = pwidx,
			.nwidx = nwidx,
			.fv = fv,
		};
		ccv_dense_matrix_t* x = ccv_dense_matrix_new(1, 33, CCV_32F | CCV_C1, 0, 0);
		int j;
		for (j = 0; j < 33; j++)
			x->data.f32[j] = gsl_rng_uniform_pos(rng) * 2 - 1.0;
		ccv_minimize(x, 10, 1.0, _ccv_scd_stump_feature_gentle_adaboost_loss, ccv_minimize_default_params, &context);
		for (j = 0; j < 32; j++)
			feature->w[j] = x->data.f32[j];
		feature->bias = x->data.f32[32];
		ccv_matrix_free(x);
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
	_ccv_scd_value_index_sortby_value(sidx, posnum + negnum, 0);
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

static int _ccv_scd_find_match_feature(ccv_scd_stump_feature_t* value, ccv_array_t* features)
{
	int i;
	for (i = 0; i < features->rnum; i++)
	{
		ccv_scd_stump_feature_t* feature = (ccv_scd_stump_feature_t*)ccv_array_get(features, i);
		if (feature->sx[0] == value->sx[0] && feature->sy[0] == value->sy[0] &&
			feature->dx[0] == value->dx[0] && feature->dy[0] == value->dy[0] &&
			feature->sx[1] == value->sx[1] && feature->sy[1] == value->sy[1] &&
			feature->dx[1] == value->dx[1] && feature->dy[1] == value->dy[1] &&
			feature->sx[2] == value->sx[2] && feature->sy[2] == value->sy[2] &&
			feature->dx[2] == value->dx[2] && feature->dy[2] == value->dy[2] &&
			feature->sx[3] == value->sx[3] && feature->sy[3] == value->sy[3] &&
			feature->dx[3] == value->dx[3] && feature->dy[3] == value->dy[3])
			return i;
	}
	return -1;
}

static int _ccv_scd_best_feature_gentle_adaboost(double* s, ccv_array_t* features, double* pw, double* nw, int positive_count, int negative_count, float* fv)
{
	int i;
	double* error_rate = (double*)cccalloc(features->rnum, sizeof(double));
	assert(positive_count + negative_count > 0);
	parallel_for(i, features->rnum) {
		int j, k;
		if ((i + 1) % 331 == 1 || (i + 1) == features->rnum)
			FLUSH(CCV_CLI_INFO, " - go through %d / %d (%.1f%%) for adaboost", (int)(i + 1), features->rnum, (float)(i + 1) * 100 / features->rnum);
		ccv_scd_stump_feature_t* feature = (ccv_scd_stump_feature_t*)ccv_array_get(features, i);
		for (j = 0; j < positive_count; j++)
		{
			float* surf = _ccv_scd_get_surf_at(fv, i, j, positive_count, negative_count);
			float v = feature->bias;
			for (k = 0; k < 32; k++)
				v += surf[k] * feature->w[k];
			v = expf(v);
			v = (v - 1) / (v + 1); // probability
			error_rate[i] += pw[j] * (1 - v) * (1 - v);
		}
		for (j = 0; j < negative_count; j++)
		{
			float* surf = _ccv_scd_get_surf_at(fv, i, j + positive_count, positive_count, negative_count);
			float v = feature->bias;
			for (k = 0; k < 32; k++)
				v += surf[k] * feature->w[k];
			v = expf(v);
			v = (v - 1) / (v + 1); // probability
			error_rate[i] += nw[j] * (-1 - v) * (-1 - v);
		}
	} parallel_endfor
	double min_error_rate = error_rate[0];
	int j = 0;
	for (i = 1; i < features->rnum; i++)
		if (error_rate[i] < min_error_rate)
		{
			min_error_rate = error_rate[i];
			j = i;
		}
	ccfree(error_rate);
	return j;
}

static float _ccv_scd_threshold_at_hit_rate(double* s, int posnum, int negnum, float hit_rate, float* tp_out, float* fp_out)
{
	ccv_scd_value_index_t* psidx = (ccv_scd_value_index_t*)ccmalloc(sizeof(ccv_scd_value_index_t) * posnum);
	int i;
	for (i = 0; i < posnum; i++)
		psidx[i].value = s[i], psidx[i].index = i;
	_ccv_scd_value_index_sortby_value(psidx, posnum, 0);
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
#if defined(HAVE_SSE2)
	__m128 surf[8];
#else
	float surf[32];
#endif
	ccv_dense_matrix_t* b = 0;
	ccv_scd(a, &b, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(b, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(b);
	int pass = 1;
	int i, j;
	for (i = 0; i < cascade->count; i++)
	{
		ccv_scd_stump_classifier_t* classifier = cascade->classifiers + i;
		float v = 0;
		for (j = 0; j < classifier->count; j++)
		{
			ccv_scd_stump_feature_t* feature = classifier->features + j;
#if defined(HAVE_SSE2)
			_ccv_scd_run_feature_at_sse2(sat->data.f32, sat->cols, feature, surf);
			__m128 u0 = _mm_add_ps(_mm_mul_ps(surf[0], _mm_loadu_ps(feature->w)), _mm_mul_ps(surf[1], _mm_loadu_ps(feature->w + 4)));
			__m128 u1 = _mm_add_ps(_mm_mul_ps(surf[2], _mm_loadu_ps(feature->w + 8)), _mm_mul_ps(surf[3], _mm_loadu_ps(feature->w + 12)));
			__m128 u2 = _mm_add_ps(_mm_mul_ps(surf[4], _mm_loadu_ps(feature->w + 16)), _mm_mul_ps(surf[5], _mm_loadu_ps(feature->w + 20)));
			__m128 u3 = _mm_add_ps(_mm_mul_ps(surf[6], _mm_loadu_ps(feature->w + 24)), _mm_mul_ps(surf[7], _mm_loadu_ps(feature->w + 28)));
			u0 = _mm_add_ps(u0, u1);
			u2 = _mm_add_ps(u2, u3);
			union {
				float f[4];
				__m128 p;
			} ux;
			ux.p = _mm_add_ps(u0, u2);
			float u = expf(feature->bias + ux.f[0] + ux.f[1] + ux.f[2] + ux.f[3]);
#else
			_ccv_scd_run_feature_at(sat->data.f32, sat->cols, feature, surf);
			float u = feature->bias;
			int k;
			for (k = 0; k < 32; k++)
				u += surf[k] * feature->w[k];
			u = expf(u);
#endif
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

static ccv_array_t* _ccv_scd_hard_mining(gsl_rng* rng, ccv_scd_classifier_cascade_t* cascade, ccv_array_t* hard_mine, ccv_array_t* negatives, int negative_count, int grayscale, int even_dist)
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
	for (t = (even_dist ? 0 : 4); t < 8 /* exhausted all variations */ && hard_negatives->rnum < negative_count; t++)
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
				.step_through = 4,
				.size = cascade->size,
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
	uint64_t array_signature;
	ccv_array_t* features;
	ccv_array_t* positives;
	ccv_array_t* negatives;
	double* s;
	double* pw;
	double* nw;
	float* fv; // feature vector for examples * feature
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
				int positive_count = sqlite3_column_int(function_state_stmt, 2);
				assert(positive_count == z->positives->rnum);
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
				size = sqlite3_column_bytes(function_state_stmt, 9);
				const void* nw = sqlite3_column_blob(function_state_stmt, 9);
				memcpy(z->nw, nw, size);
			}
			sqlite3_finalize(function_state_stmt);
		}
		_ccv_scd_precompute_feature_vectors(z->features, z->positives, z->negatives, z->fv);
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
		sqlite3_bind_int(function_state_insert_stmt, 3, z->positives->rnum);
		sqlite3_bind_double(function_state_insert_stmt, 4, z->auc_prev);
		sqlite3_bind_double(function_state_insert_stmt, 5, z->accu_true_positive_rate);
		sqlite3_bind_double(function_state_insert_stmt, 6, z->accu_false_positive_rate);
		sqlite3_bind_int(function_state_insert_stmt, 7, z->line_no);
		sqlite3_bind_blob(function_state_insert_stmt, 8, z->s, sizeof(double) * (z->positives->rnum + z->negatives->rnum), SQLITE_STATIC);
		sqlite3_bind_blob(function_state_insert_stmt, 9, z->pw, sizeof(double) * z->positives->rnum, SQLITE_STATIC);
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
#ifdef HAVE_GSL
	assert(posfiles->rnum > 0);
	assert(hard_mine->rnum > 0);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	ccv_scd_classifier_cascade_new_function_state_t z = {0};
	z.features = _ccv_scd_stump_features(params.feature.base, params.feature.range_through, params.feature.step_through, params.size);
	PRINT(CCV_CLI_INFO, " - using %d features\n", z.features->rnum);
	int i, j, p, q;
	z.positives = _ccv_scd_collect_positives(params.size, posfiles, params.grayscale);
	double* h = (double*)ccmalloc(sizeof(double) * (z.positives->rnum + negative_count));
	z.s = (double*)ccmalloc(sizeof(double) * (z.positives->rnum + negative_count));
	assert(z.s);
	z.pw = (double*)ccmalloc(sizeof(double) * z.positives->rnum);
	assert(z.pw);
	z.nw = (double*)ccmalloc(sizeof(double) * negative_count);
	assert(z.nw);
	ccmemalign((void**)&z.fv, 16, sizeof(float) * (z.positives->rnum + negative_count) * z.features->rnum * 32);
	assert(z.fv);
	z.params = params;
	ccv_function_state_begin(_ccv_scd_classifier_cascade_new_function_state_read, z, filename);
	z.negatives = _ccv_scd_collect_negatives(rng, params.size, hard_mine, negative_count, params.grayscale);
	_ccv_scd_precompute_feature_vectors(z.features, z.positives, z.negatives, z.fv);
	z.cascade = (ccv_scd_classifier_cascade_t*)ccmalloc(sizeof(ccv_scd_classifier_cascade_t));
	z.cascade->margin = ccv_margin(0, 0, 0, 0);
	z.cascade->size = params.size;
	z.cascade->count = 0;
	z.cascade->classifiers = 0;
	z.accu_true_positive_rate = 1;
	z.accu_false_positive_rate = 1;
	ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
	for (z.t = 0; z.t < params.boosting; z.t++)
	{
		for (i = 0; i < z.positives->rnum; i++)
			z.pw[i] = 0.5 / z.positives->rnum;
		for (i = 0; i < z.negatives->rnum; i++)
			z.nw[i] = 0.5 / z.negatives->rnum;
		memset(z.s, 0, sizeof(double) * (z.positives->rnum + z.negatives->rnum));
		z.cascade->classifiers = (ccv_scd_stump_classifier_t*)ccrealloc(z.cascade->classifiers, sizeof(ccv_scd_stump_classifier_t) * (z.t + 1));
		z.cascade->count = z.t + 1;
		z.cascade->classifiers[z.t].threshold = 0;
		z.cascade->classifiers[z.t].features = 0;
		z.cascade->classifiers[z.t].count = 0;
		z.auc_prev = 0;
		assert(z.positives->rnum > 0 && z.negatives->rnum > 0);
		// for the first prune stages, we have more restrictive number of features (faster)
		for (z.k = 0; z.k < (z.t < params.stop_criteria.prune_stage ? params.stop_criteria.prune_feature : params.stop_criteria.maximum_feature); z.k++)
		{
			ccv_scd_stump_classifier_t* classifier = z.cascade->classifiers + z.t;
			classifier->features = (ccv_scd_stump_feature_t*)ccrealloc(classifier->features, sizeof(ccv_scd_stump_feature_t) * (z.k + 1));
			_ccv_scd_stump_feature_supervised_train(rng, z.features, z.positives->rnum, z.negatives->rnum, z.pw, z.nw, z.fv, params.C, params.weight_trimming);
			int best_feature_no = _ccv_scd_best_feature_gentle_adaboost(z.s, z.features, z.pw, z.nw, z.positives->rnum, z.negatives->rnum, z.fv);
			ccv_scd_stump_feature_t best_feature = *(ccv_scd_stump_feature_t*)ccv_array_get(z.features, best_feature_no);
			for (i = 0; i < z.positives->rnum + z.negatives->rnum; i++)
			{
				float* surf = _ccv_scd_get_surf_at(z.fv, best_feature_no, i, z.positives->rnum, z.negatives->rnum);
				float v = best_feature.bias;
				for (j = 0; j < 32; j++)
					v += best_feature.w[j] * surf[j];
				v = expf(v);
				h[i] = (v - 1) / (v + 1);
			}
			// compute the total score so far
			for (i = 0; i < z.positives->rnum + z.negatives->rnum; i++)
				z.s[i] += h[i];
			// compute AUC
			double auc = _ccv_scd_auc(z.s, z.positives->rnum, z.negatives->rnum);
			float true_positive_rate = 0;
			float false_positive_rate = 0;
			// compute true positive / false positive rate
			_ccv_scd_threshold_at_hit_rate(z.s, z.positives->rnum, z.negatives->rnum, params.stop_criteria.hit_rate, &true_positive_rate, &false_positive_rate);
			FLUSH(CCV_CLI_INFO, " - at %d-th iteration, auc: %lf, TP rate: %f, FP rate: %f\n", z.k + 1, auc, true_positive_rate, false_positive_rate);
			PRINT(CCV_CLI_INFO, " --- pick feature %s @ (%d, %d, %d, %d)\n", ((best_feature.dy[3] == best_feature.dy[0] ? "4x1" : (best_feature.dx[3] == best_feature.dx[0] ? "1x4" : "2x2"))), best_feature.sx[0], best_feature.sy[0], best_feature.dx[3], best_feature.dy[3]);
			classifier->features[z.k] = best_feature;
			classifier->count = z.k + 1;
			double auc_prev = z.auc_prev;
			z.auc_prev = auc;
			// auc stop to improve, as well as the false positive rate goal reach, at that point, we stop
			if (auc - auc_prev < params.stop_criteria.auc_crit && false_positive_rate < params.stop_criteria.false_positive_rate)
				break;
			// re-weight, with Gentle AdaBoost
			for (i = 0; i < z.positives->rnum; i++)
				z.pw[i] *= exp(-h[i]);
			for (i = 0; i < z.negatives->rnum; i++)
				z.nw[i] *= exp(h[i + z.positives->rnum]);
			// re-normalize
			double w = 0;
			for (i = 0; i < z.positives->rnum; i++)
				w += z.pw[i];
			w = 0.5 / w;
			for (i = 0; i < z.positives->rnum; i++)
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
		while (z.cascade->classifiers[z.t].count > 1)
		{
			double max_auc = 0;
			p = -1;
			for (i = 0; i < z.cascade->classifiers[z.t].count; i++)
			{
				ccv_scd_stump_feature_t* feature = z.cascade->classifiers[z.t].features + i;
				int k = _ccv_scd_find_match_feature(feature, z.features);
				assert(k >= 0);
				for (j = 0; j < z.positives->rnum + z.negatives->rnum; j++)
				{
					float* surf = _ccv_scd_get_surf_at(z.fv, k, j, z.positives->rnum, z.negatives->rnum);
					float v = feature->bias;
					for (q = 0; q < 32; q++)
						v += feature->w[q]* surf[q];
					v = expf(v);
					h[j] = z.s[j] - (v - 1) / (v + 1);
				}
				double auc = _ccv_scd_auc(h, z.positives->rnum, z.negatives->rnum);
				FLUSH(CCV_CLI_INFO, " - attempting without %d-th feature, auc: %lf", i + 1, auc);
				if (auc >= max_auc)
					max_auc = auc, p = i;
			}
			if (max_auc >= z.auc_prev)
			{
				FLUSH(CCV_CLI_INFO, " - remove %d-th feature with new auc %lf\n", p + 1, max_auc);
				ccv_scd_stump_feature_t* feature = z.cascade->classifiers[z.t].features + p;
				int k = _ccv_scd_find_match_feature(feature, z.features);
				assert(k >= 0);
				for (j = 0; j < z.positives->rnum + z.negatives->rnum; j++)
				{
					float* surf = _ccv_scd_get_surf_at(z.fv, k, j, z.positives->rnum, z.negatives->rnum);
					float v = feature->bias;
					for (q = 0; q < 32; q++)
						v += feature->w[q] * surf[q];
					v = expf(v);
					z.s[j] -= (v - 1) / (v + 1);
				}
				z.auc_prev = _ccv_scd_auc(z.s, z.positives->rnum, z.negatives->rnum);
				--z.cascade->classifiers[z.t].count;
				if (p < z.cascade->classifiers[z.t].count)
					memmove(z.cascade->classifiers[z.t].features + p + 1, z.cascade->classifiers[z.t].features + p, sizeof(ccv_scd_stump_feature_t) * (z.cascade->classifiers[z.t].count - p));
			} else
				break;
		}
		float true_positive_rate = 0;
		float false_positive_rate = 0;
		z.cascade->classifiers[z.t].threshold = _ccv_scd_threshold_at_hit_rate(z.s, z.positives->rnum, z.negatives->rnum, params.stop_criteria.hit_rate, &true_positive_rate, &false_positive_rate);
		z.accu_true_positive_rate *= true_positive_rate;
		z.accu_false_positive_rate *= false_positive_rate;
		FLUSH(CCV_CLI_INFO, " - %d-th stage classifier TP rate : %f, FP rate : %f, ATP rate : %lf, AFP rate : %lg, at threshold : %f\n", z.t + 1, true_positive_rate, false_positive_rate, z.accu_true_positive_rate, z.accu_false_positive_rate, z.cascade->classifiers[z.t].threshold);
		if (z.accu_false_positive_rate < params.stop_criteria.accu_false_positive_rate)
			break;
		ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
		if (z.t < params.boosting - 1)
		{
			int pass = 0;
			for (i = 0; i < z.positives->rnum; i++)
			{
				ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(z.positives, i);
				a->data.u8 = (unsigned char*)(a + 1);
				if (_ccv_scd_classifier_cascade_pass(z.cascade, a))
					++pass;
			}
			PRINT(CCV_CLI_INFO, " - %d-th stage classifier TP rate (with pass) : %f\n", z.t + 1, (float)pass / z.positives->rnum);
			ccv_array_t* hard_negatives = _ccv_scd_hard_mining(rng, z.cascade, hard_mine, z.negatives, negative_count, params.grayscale, z.t < params.stop_criteria.prune_stage /* try to balance even distribution among negatives when we are in prune stage */);
			ccv_array_free(z.negatives);
			z.negatives = hard_negatives;
			_ccv_scd_precompute_feature_vectors(z.features, z.positives, z.negatives, z.fv);
		}
		ccv_function_state_resume(_ccv_scd_classifier_cascade_new_function_state_write, z, filename);
	}
	ccv_array_free(z.negatives);
	ccv_function_state_finish();
	ccv_array_free(z.features);
	ccv_array_free(z.positives);
	ccfree(h);
	ccfree(z.s);
	ccfree(z.pw);
	ccfree(z.nw);
	ccfree(z.fv);
	gsl_rng_free(rng);
	return z.cascade;
#else
	assert(0 && "ccv_scd_classifier_cascade_new requires GSL library and support");
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
			ccv_scd_stump_classifier_t* classifier = cascade->classifiers + i;
			sqlite3_bind_int(classifier_params_insert_stmt, 1, i);
			sqlite3_bind_int(classifier_params_insert_stmt, 2, classifier->count);
			sqlite3_bind_double(classifier_params_insert_stmt, 3, classifier->threshold);
			assert(SQLITE_DONE == sqlite3_step(classifier_params_insert_stmt));
			sqlite3_reset(classifier_params_insert_stmt);
			sqlite3_clear_bindings(classifier_params_insert_stmt);
			for (j = 0; j < classifier->count; j++)
			{
				ccv_scd_stump_feature_t* feature = classifier->features + j;
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
				cascade->classifiers = (ccv_scd_stump_classifier_t*)cccalloc(cascade->count, sizeof(ccv_scd_stump_classifier_t));
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
						ccv_scd_stump_classifier_t* classifier = cascade->classifiers + sqlite3_column_int(classifier_params_stmt, 0);
						classifier->count = sqlite3_column_int(classifier_params_stmt, 1);
						classifier->features = (ccv_scd_stump_feature_t*)ccmalloc(sizeof(ccv_scd_stump_feature_t) * classifier->count);
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
						ccv_scd_stump_classifier_t* classifier = cascade->classifiers + sqlite3_column_int(feature_params_stmt, 0);
						if (sqlite3_column_int(feature_params_stmt, 1) < classifier->count)
						{
							ccv_scd_stump_feature_t* feature = classifier->features + sqlite3_column_int(feature_params_stmt, 1);
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
		ccv_scd_stump_classifier_t* classifier = cascade->classifiers + i;
		ccfree(classifier->features);
	}
	ccfree(cascade->classifiers);
	ccfree(cascade);
}

static int _ccv_is_equal_same_class(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;

	if (r2->classification.id != r1->classification.id)
		return 0;

	int i = ccv_max(ccv_min(r2->rect.x + r2->rect.width, r1->rect.x + r1->rect.width) - ccv_max(r2->rect.x, r1->rect.x), 0) * ccv_max(ccv_min(r2->rect.y + r2->rect.height, r1->rect.y + r1->rect.height) - ccv_max(r2->rect.y, r1->rect.y), 0);
	int m = ccv_min(r2->rect.width * r2->rect.height, r1->rect.width * r1->rect.height);

	return i >= 0.3 * m; // IoM > 0.3 like HeadHunter does
}

ccv_array_t* ccv_scd_detect_objects(ccv_dense_matrix_t* a, ccv_scd_classifier_cascade_t** cascades, int count, ccv_scd_param_t params)
{
	int i, j, k, x, y, p, q;
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
#if defined(HAVE_SSE2)
	__m128 surf[8];
#else
	float surf[32];
#endif
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
				ccv_dense_matrix_t* scd = 0;
				if (cascade->margin.left == 0 && cascade->margin.top == 0 && cascade->margin.right == 0 && cascade->margin.bottom == 0)
				{
					ccv_scd(image, &scd, 0);
					if (k > 0)
						ccv_matrix_free(image);
				} else {
					ccv_dense_matrix_t* bordered = 0;
					ccv_border(image, (ccv_matrix_t**)&bordered, 0, cascade->margin);
					if (k > 0)
						ccv_matrix_free(image);
					ccv_scd(bordered, &scd, 0);
					ccv_matrix_free(bordered);
				}
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
							ccv_scd_stump_classifier_t* classifier = cascade->classifiers + p;
							float v = 0;
							for (q = 0; q < classifier->count; q++)
							{
								ccv_scd_stump_feature_t* feature = classifier->features + q;
#if defined(HAVE_SSE2)
								_ccv_scd_run_feature_at_sse2(ptr + x * 8, sat->cols, feature, surf);
								__m128 u0 = _mm_add_ps(_mm_mul_ps(surf[0], _mm_loadu_ps(feature->w)), _mm_mul_ps(surf[1], _mm_loadu_ps(feature->w + 4)));
								__m128 u1 = _mm_add_ps(_mm_mul_ps(surf[2], _mm_loadu_ps(feature->w + 8)), _mm_mul_ps(surf[3], _mm_loadu_ps(feature->w + 12)));
								__m128 u2 = _mm_add_ps(_mm_mul_ps(surf[4], _mm_loadu_ps(feature->w + 16)), _mm_mul_ps(surf[5], _mm_loadu_ps(feature->w + 20)));
								__m128 u3 = _mm_add_ps(_mm_mul_ps(surf[6], _mm_loadu_ps(feature->w + 24)), _mm_mul_ps(surf[7], _mm_loadu_ps(feature->w + 28)));
								u0 = _mm_add_ps(u0, u1);
								u2 = _mm_add_ps(u2, u3);
								union {
									float f[4];
									__m128 p;
								} ux;
								ux.p = _mm_add_ps(u0, u2);
								float u = expf(feature->bias + ux.f[0] + ux.f[1] + ux.f[2] + ux.f[3]);
#else
								_ccv_scd_run_feature_at(ptr + x * 8, sat->cols, feature, surf);
								float u = feature->bias;
								int r;
								for (r = 0; r < 32; r++)
									u += surf[r] * feature->w[r];
								u = expf(u);
#endif
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
	for (k = 0; k < count; k++)
	{
		/* simple non-maximum suppression, we merge when intersected area / min area > 0.3  */
		if(params.min_neighbors == 0)
		{
			for (i = 0; i < seq[k]->rnum; i++)
			{
				ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq[k], i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			ccv_array_t* idx_seq = 0;
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

			// push merged bounding box to result_seq
			for (i = 0; i < ncomp; i++)
			{
				int n = comps[i].neighbors;
				if (n >= params.min_neighbors)
					ccv_array_push(result_seq, comps + i);
			}
			ccv_array_free(idx_seq);
			ccfree(comps);
		}
		ccv_array_free(seq[k]);
	}

	return result_seq;
}
