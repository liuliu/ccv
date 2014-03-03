#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

const ccv_icf_param_t ccv_icf_default_params = {
	.min_neighbors = 2,
	.threshold = 0,
	.step_through = 2,
	.flags = 0,
	.interval = 8,
};

// cube root approximation using bit hack for 32-bit float
// provides a very crude approximation
static inline float cbrt_5_f32(float f)
{
	unsigned int* p = (unsigned int*)(&f);
	*p = *p / 3 + 709921077;
	return f;
}

// iterative cube root approximation using Halley's method (float)
static inline float cbrta_halley_f32(const float a, const float R)
{
	const float a3 = a * a * a;
	const float b = a * (a3 + R + R) / (a3 + a3 + R);
	return b;
}

// Code based on
// http://metamerist.com/cbrt/cbrt.htm
// cube root approximation using 2 iterations of Halley's method (float)
// this is expected to be ~2.5x times faster than std::pow(x, 3)
static inline float fast_cube_root(const float d)
{
	float a = cbrt_5_f32(d);
	a = cbrta_halley_f32(a, d);
	return cbrta_halley_f32(a, d);
}

static inline void _ccv_rgb_to_luv(const float r, const float g, const float b, float* pl, float* pu, float* pv)
{
	const float x = 0.412453f * r + 0.35758f * g + 0.180423f * b;
	const float y = 0.212671f * r + 0.71516f * g + 0.072169f * b;
	const float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;

	const float x_n = 0.312713f, y_n = 0.329016f;
	const float uv_n_divisor = -2.f * x_n + 12.f * y_n + 3.f;
	const float u_n = 4.f * x_n / uv_n_divisor;
	const float v_n = 9.f * y_n / uv_n_divisor;

    const float uv_divisor = ccv_max((x + 15.f * y + 3.f * z), FLT_EPSILON);
	const float u = 4.f * x / uv_divisor;
	const float v = 9.f * y / uv_divisor;

	const float y_cube_root = fast_cube_root(y);

	const float l_value = ccv_max(0.f, ((116.f * y_cube_root) - 16.f));
	const float u_value = 13.f * l_value * (u - u_n);
	const float v_value = 13.f * l_value * (v - v_n);

	// L in [0, 100], U in [-134, 220], V in [-140, 122]
	*pl = l_value * (255.f / 100.f);
	*pu = (u_value + 134.f) * (255.f / (220.f + 134.f));
	*pv = (v_value + 140.f) * (255.f / (122.f + 140.f));
}

// generating the integrate channels features (which combines the grayscale, gradient magnitude, and 6-direction HOG)
void ccv_icf(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	int ch = CCV_GET_CHANNEL(a->type);
	assert(ch == 1 || ch == 3);
	int nchr = (ch == 1) ? 8 : 10;
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_literal("ccv_icf"), a->sig, CCV_EOF_SIGN);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_32F | nchr, CCV_32F | nchr, sig);
	ccv_object_return_if_cached(, db);
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.f32;
	float* mgp = mg->data.f32;
	float* dbp = db->data.f32;
	ccv_zero(db);
	int i, j, k;
	unsigned char* a_ptr = a->data.u8;
	float magnitude_scaling = 1 / sqrtf(2); // regularize it to 0~1
	if (ch == 1)
	{
#define for_block(_, _for_get) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
			{ \
				dbp[0] = _for_get(a_ptr, j, 0); \
				dbp[1] = mgp[j] * magnitude_scaling; \
				float agr = (ccv_clamp(agp[j] <= 180 ? agp[j] : agp[j] - 180, 0, 179.99) / 180.0) * 6; \
				int ag0 = (int)agr; \
				int ag1 = ag0 < 5 ? ag0 + 1 : 0; \
				agr = agr - ag0; \
				dbp[2 + ag0] = dbp[1] * (1 - agr); \
				dbp[2 + ag1] = dbp[1] * agr; \
				dbp += 8; \
			} \
			a_ptr += a->step; \
			agp += a->cols; \
			mgp += a->cols; \
		}
		ccv_matrix_getter(a->type, for_block);
#undef for_block
	} else {
		// color one, luv, gradient magnitude, and 6-direction HOG
#define for_block(_, _for_get) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
			{ \
				_ccv_rgb_to_luv(_for_get(a_ptr, j * ch, 0) / 255.0, \
								_for_get(a_ptr, j * ch + 1, 0) / 255.0, \
								_for_get(a_ptr, j * ch + 2, 0) / 255.0, \
								dbp, dbp + 1, dbp + 2); \
				float agv = agp[j * ch]; \
				float mgv = mgp[j * ch]; \
				for (k = 1; k < ch; k++) \
				{ \
					if (mgp[j * ch + k] > mgv) \
					{ \
						mgv = mgp[j * ch + k]; \
						agv = agp[j * ch + k]; \
					} \
				} \
				dbp[3] = mgv * magnitude_scaling; \
				float agr = (ccv_clamp(agv <= 180 ? agv : agv - 180, 0, 179.99) / 180.0) * 6; \
				int ag0 = (int)agr; \
				int ag1 = ag0 < 5 ? ag0 + 1 : 0; \
				agr = agr - ag0; \
				dbp[4 + ag0] = dbp[3] * (1 - agr); \
				dbp[4 + ag1] = dbp[3] * agr; \
				dbp += 10; \
			} \
			a_ptr += a->step; \
			agp += a->cols * ch; \
			mgp += a->cols * ch; \
		}
		ccv_matrix_getter(a->type, for_block);
#undef for_block
	}
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
}

static inline float _ccv_icf_run_feature(ccv_icf_feature_t* feature, float* ptr, int cols, int ch, int x, int y)
{
	float c = feature->beta;
	int q;
	for (q = 0; q < feature->count; q++)
		c += (ptr[(feature->sat[q * 2 + 1].x + x + 1 + (feature->sat[q * 2 + 1].y + y + 1) * cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2].x + x + (feature->sat[q * 2 + 1].y + y + 1) * cols) * ch + feature->channel[q]] + ptr[(feature->sat[q * 2].x + x + (feature->sat[q * 2].y + y) * cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2 + 1].x + x + 1 + (feature->sat[q * 2].y + y) * cols) * ch + feature->channel[q]]) * feature->alpha[q];
	return c;
}

static inline int _ccv_icf_run_weak_classifier(ccv_icf_decision_tree_t* weak_classifier, float* ptr, int cols, int ch, int x, int y)
{
	float c = _ccv_icf_run_feature(weak_classifier->features, ptr, cols, ch, x, y);
	if (c > 0)
	{
		if (!(weak_classifier->pass & 0x1))
			return 1;
		return _ccv_icf_run_feature(weak_classifier->features + 2, ptr, cols, ch, x, y) > 0;
	} else {
		if (!(weak_classifier->pass & 0x2))
			return 0;
		return _ccv_icf_run_feature(weak_classifier->features + 1, ptr, cols, ch, x, y) > 0;
	}
}

#ifdef HAVE_GSL
static void _ccv_icf_randomize_feature(gsl_rng* rng, ccv_size_t size, int minimum, ccv_icf_feature_t* feature, int grayscale)
{
	feature->count = gsl_rng_uniform_int(rng, CCV_ICF_SAT_MAX) + 1;
	assert(feature->count <= CCV_ICF_SAT_MAX);
	int i;
	feature->beta = 0;
	for (i = 0; i < feature->count; i++)
	{
		int x0, y0, x1, y1;
		do {
			x0 = gsl_rng_uniform_int(rng, size.width);
			x1 = gsl_rng_uniform_int(rng, size.width);
			y0 = gsl_rng_uniform_int(rng, size.height);
			y1 = gsl_rng_uniform_int(rng, size.height);
		} while ((ccv_max(x0, x1) - ccv_min(x0, x1) + 1) * (ccv_max(y0, y1) - ccv_min(y0, y1) + 1) < (minimum + 1) * (minimum + 1) ||
				 (ccv_max(x0, x1) - ccv_min(x0, x1) + 1) < minimum ||
				 (ccv_max(y0, y1) - ccv_min(y0, y1) + 1) < minimum);
		feature->sat[i * 2].x = ccv_min(x0, x1);
		feature->sat[i * 2].y = ccv_min(y0, y1);
		feature->sat[i * 2 + 1].x = ccv_max(x0, x1);
		feature->sat[i * 2 + 1].y = ccv_max(y0, y1);
		feature->channel[i] = gsl_rng_uniform_int(rng, grayscale ? 8 : 10); // 8-channels for grayscale, and 10-channels for rgb
		assert(feature->channel[i] >= 0 && feature->channel[i] < (grayscale ? 8 : 10));
		feature->alpha[i] = gsl_rng_uniform(rng) / (float)((feature->sat[i * 2 + 1].x - feature->sat[i * 2].x + 1) * (feature->sat[i * 2 + 1].y - feature->sat[i * 2].y + 1));
	}
}

static void _ccv_icf_check_params(ccv_icf_new_param_t params)
{
	assert(params.size.width > 0 && params.size.height > 0);
	assert(params.deform_shift >= 0);
	assert(params.deform_angle >= 0);
	assert(params.deform_scale >= 0 && params.deform_scale < 1);
	assert(params.feature_size > 0);
	assert(params.acceptance > 0 && params.acceptance < 1.0);
}

static ccv_dense_matrix_t* _ccv_icf_capture_feature(gsl_rng* rng, ccv_dense_matrix_t* image, ccv_decimal_pose_t pose, ccv_size_t size, ccv_margin_t margin, float deform_angle, float deform_scale, float deform_shift)
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

typedef struct {
	uint8_t correct:1;
	double weight;
	float rate;
} ccv_icf_example_state_t;

typedef struct {
	uint8_t classifier:1;
	uint8_t positives:1;
	uint8_t negatives:1;
	uint8_t features:1;
	uint8_t example_state:1;
	uint8_t precomputed:1;
} ccv_icf_classifier_cascade_persistence_state_t;

typedef struct {
	uint32_t index;
	float value;
} ccv_icf_value_index_t;

typedef struct {
	ccv_function_state_reserve_field;
	int i;
	int bootstrap;
	ccv_icf_new_param_t params;
	ccv_icf_classifier_cascade_t* classifier;
	ccv_array_t* positives;
	ccv_array_t* negatives;
	ccv_icf_feature_t* features;
	ccv_size_t size;
	ccv_margin_t margin;
	ccv_icf_example_state_t* example_state;
	uint8_t* precomputed;
	ccv_icf_classifier_cascade_persistence_state_t x;
} ccv_icf_classifier_cascade_state_t;

static void _ccv_icf_write_classifier_cascade_state(ccv_icf_classifier_cascade_state_t* state, const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/state", directory);
	FILE* w = fopen(filename, "w+");
	fprintf(w, "%d %d %d\n", state->line_no, state->i, state->bootstrap);
	fprintf(w, "%d %d %d\n", state->params.feature_size, state->size.width, state->size.height);
	fprintf(w, "%d %d %d %d\n", state->margin.left, state->margin.top, state->margin.right, state->margin.bottom);
	fclose(w);
	int i, q;
	if (!state->x.positives)
	{
		snprintf(filename, 1024, "%s/positives", directory);
		w = fopen(filename, "wb+");
		fwrite(&state->positives->rnum, sizeof(state->positives->rnum), 1, w);
		fwrite(&state->positives->rsize, sizeof(state->positives->rsize), 1, w);
		for (i = 0; i < state->positives->rnum; i++)
		{
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(state->positives, i);
			assert(a->rows == state->size.height + state->margin.top + state->margin.bottom + 2 && a->cols == state->size.width + state->margin.left + state->margin.right + 2);
			fwrite(a, 1, state->positives->rsize, w);
		}
		fclose(w);
		state->x.positives = 1;
	}
	if (!state->x.negatives)
	{
		assert(state->negatives->rsize == state->positives->rsize);
		snprintf(filename, 1024, "%s/negatives", directory);
		w = fopen(filename, "wb+");
		fwrite(&state->negatives->rnum, sizeof(state->negatives->rnum), 1, w);
		fwrite(&state->negatives->rsize, sizeof(state->negatives->rsize), 1, w);
		for (i = 0; i < state->negatives->rnum; i++)
		{
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(state->negatives, i);
			assert(a->rows == state->size.height + state->margin.top + state->margin.bottom + 2 && a->cols == state->size.width + state->margin.left + state->margin.right + 2);
			fwrite(a, 1, state->negatives->rsize, w);
		}
		fclose(w);
		state->x.negatives = 1;
	}
	if (!state->x.features)
	{
		snprintf(filename, 1024, "%s/features", directory);
		w = fopen(filename, "w+");
		for (i = 0; i < state->params.feature_size; i++)
		{
			ccv_icf_feature_t* feature = state->features + i;
			fprintf(w, "%d %a\n", feature->count, feature->beta);
			for (q = 0; q < feature->count; q++)
				fprintf(w, "%d %a %d %d %d %d\n", feature->channel[q], feature->alpha[q], feature->sat[q * 2].x, feature->sat[q * 2].y, feature->sat[q * 2 + 1].x, feature->sat[q * 2 + 1].y);
		}
		fclose(w);
		state->x.features = 1;
	}
	if (!state->x.example_state)
	{
		snprintf(filename, 1024, "%s/example_state", directory);
		w = fopen(filename, "w+");
		for (i = 0; i < state->positives->rnum + state->negatives->rnum; i++)
			fprintf(w, "%u %la %a\n", (uint32_t)state->example_state[i].correct, state->example_state[i].weight, state->example_state[i].rate);
		fclose(w);
		state->x.example_state = 1;
	}
	if (!state->x.precomputed)
	{
		size_t step = (3 * (state->positives->rnum + state->negatives->rnum) + 3) & -4;
		snprintf(filename, 1024, "%s/precomputed", directory);
		w = fopen(filename, "wb+");
		fwrite(state->precomputed, 1, step * state->params.feature_size, w);
		fclose(w);
		state->x.precomputed = 1;
	}
	if (!state->x.classifier)
	{
		snprintf(filename, 1024, "%s/cascade", directory);
		ccv_icf_write_classifier_cascade(state->classifier, filename);
		state->x.classifier = 1;
	}
}

static void _ccv_icf_read_classifier_cascade_state(const char* directory, ccv_icf_classifier_cascade_state_t* state)
{
	char filename[1024];
	state->line_no = state->i = 0;
	state->bootstrap = 0;
	snprintf(filename, 1024, "%s/state", directory);
	FILE* r = fopen(filename, "r");
	if (r)
	{
		int feature_size;
		fscanf(r, "%d %d %d", &state->line_no, &state->i, &state->bootstrap);
		fscanf(r, "%d %d %d", &feature_size, &state->size.width, &state->size.height);
		fscanf(r, "%d %d %d %d", &state->margin.left, &state->margin.top, &state->margin.right, &state->margin.bottom);
		assert(feature_size == state->params.feature_size);
		fclose(r);
	}
	int i, q;
	snprintf(filename, 1024, "%s/positives", directory);
	r = fopen(filename, "rb");
	state->x.precomputed = state->x.features = state->x.example_state = state->x.classifier = state->x.positives = state->x.negatives = 1;
	if (r)
	{
		int rnum, rsize;
		fread(&rnum, sizeof(rnum), 1, r);
		fread(&rsize, sizeof(rsize), 1, r);
		state->positives = ccv_array_new(rsize, rnum, 0);
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)alloca(rsize);
		for (i = 0; i < rnum; i++)
		{
			fread(a, 1, rsize, r);
			assert(a->rows == state->size.height + state->margin.top + state->margin.bottom + 2 && a->cols == state->size.width + state->margin.left + state->margin.right + 2);
			ccv_array_push(state->positives, a);
		}
		fclose(r);
	}
	snprintf(filename, 1024, "%s/negatives", directory);
	r = fopen(filename, "rb");
	if (r)
	{
		int rnum, rsize;
		fread(&rnum, sizeof(rnum), 1, r);
		fread(&rsize, sizeof(rsize), 1, r);
		state->negatives = ccv_array_new(rsize, rnum, 0);
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)alloca(rsize);
		for (i = 0; i < rnum; i++)
		{
			fread(a, 1, rsize, r);
			assert(a->rows == state->size.height + state->margin.top + state->margin.bottom + 2 && a->cols == state->size.width + state->margin.left + state->margin.right + 2);
			ccv_array_push(state->negatives, a);
		}
		fclose(r);
	}
	snprintf(filename, 1024, "%s/features", directory);
	r = fopen(filename, "r");
	if (r)
	{
		state->features = (ccv_icf_feature_t*)ccmalloc(state->params.feature_size * sizeof(ccv_icf_feature_t));
		for (i = 0; i < state->params.feature_size; i++)
		{
			ccv_icf_feature_t* feature = state->features + i;
			fscanf(r, "%d %a", &feature->count, &feature->beta);
			for (q = 0; q < feature->count; q++)
				fscanf(r, "%d %a %d %d %d %d", &feature->channel[q], &feature->alpha[q], &feature->sat[q * 2].x, &feature->sat[q * 2].y, &feature->sat[q * 2 + 1].x, &feature->sat[q * 2 + 1].y);
		}
		fclose(r);
	}
	snprintf(filename, 1024, "%s/example_state", directory);
	r = fopen(filename, "r");
	if (r)
	{
		state->example_state = (ccv_icf_example_state_t*)ccmalloc((state->positives->rnum + state->negatives->rnum) * sizeof(ccv_icf_example_state_t));
		for (i = 0; i < state->positives->rnum + state->negatives->rnum; i++)
		{
			uint32_t correct;
			double weight;
			float rate;
			fscanf(r, "%u %la %a", &correct, &weight, &rate);
			state->example_state[i].correct = correct;
			state->example_state[i].weight = weight;
			state->example_state[i].rate = rate;
		}
		fclose(r);
	} else
		state->example_state = 0;
	snprintf(filename, 1024, "%s/precomputed", directory);
	r = fopen(filename, "rb");
	if (r)
	{
		size_t step = (3 * (state->positives->rnum + state->negatives->rnum) + 3) & -4;
		state->precomputed = (uint8_t*)ccmalloc(sizeof(uint8_t) * state->params.feature_size * step);
		fread(state->precomputed, 1, step * state->params.feature_size, r);
		fclose(r);
	} else
		state->precomputed = 0;
	snprintf(filename, 1024, "%s/cascade", directory);
	state->classifier = ccv_icf_read_classifier_cascade(filename);
	if (!state->classifier)
	{
		state->classifier = (ccv_icf_classifier_cascade_t*)ccmalloc(sizeof(ccv_icf_classifier_cascade_t));
		state->classifier->count = 0;
		state->classifier->grayscale = state->params.grayscale;
		state->classifier->weak_classifiers = (ccv_icf_decision_tree_t*)ccmalloc(sizeof(ccv_icf_decision_tree_t) * state->params.weak_classifier);
	} else {
		if (state->classifier->count < state->params.weak_classifier)
			state->classifier->weak_classifiers = (ccv_icf_decision_tree_t*)ccrealloc(state->classifier->weak_classifiers, sizeof(ccv_icf_decision_tree_t) * state->params.weak_classifier);
	}
}

#define less_than(s1, s2, aux) ((s1).value < (s2).value)
static CCV_IMPLEMENT_QSORT(_ccv_icf_precomputed_ordering, ccv_icf_value_index_t, less_than)
#undef less_than

static inline void _ccv_icf_3_uint8_to_1_uint1_1_uint23(uint8_t* u8, uint8_t* u1, uint32_t* uint23)
{
	*u1 = (u8[0] >> 7);
	*uint23 = (((uint32_t)(u8[0] & 0x7f)) << 16) | ((uint32_t)(u8[1]) << 8) | u8[2];
}

static inline uint32_t _ccv_icf_3_uint8_to_1_uint23(uint8_t* u8)
{
	return (((uint32_t)(u8[0] & 0x7f)) << 16) | ((uint32_t)(u8[1]) << 8) | u8[2];
}

static inline void _ccv_icf_1_uint1_1_uint23_to_3_uint8(uint8_t u1, uint32_t u23, uint8_t* u8)
{
	u8[0] = ((u1 << 7) | (u23 >> 16)) & 0xff;
	u8[1] = (u23 >> 8) & 0xff;
	u8[2] = u23 & 0xff;
}

static float _ccv_icf_run_feature_on_example(ccv_icf_feature_t* feature, ccv_dense_matrix_t* a)
{
	ccv_dense_matrix_t* icf = 0;
	// we have 1px padding around the image
	ccv_icf(a, &icf, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(icf);
	float* ptr = sat->data.f32;
	int ch = CCV_GET_CHANNEL(sat->type);
	float c = _ccv_icf_run_feature(feature, ptr, sat->cols, ch, 1, 1);
	ccv_matrix_free(sat);
	return c;
}

static uint8_t* _ccv_icf_precompute_features(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives)
{
	int i, j;
	// we use 3 bytes to represent the sorted index, and compute feature result (float) on fly
	size_t step = (3 * (positives->rnum + negatives->rnum) + 3) & -4;
	uint8_t* precomputed = (uint8_t*)ccmalloc(sizeof(uint8_t) * feature_size * step);
	ccv_icf_value_index_t* sortkv = (ccv_icf_value_index_t*)ccmalloc(sizeof(ccv_icf_value_index_t) * (positives->rnum + negatives->rnum));
	printf(" - precompute features using %uM memory temporarily\n", (uint32_t)((sizeof(float) * (positives->rnum + negatives->rnum) * feature_size + sizeof(uint8_t) * feature_size * step) / (1024 * 1024)));
	float* featval = (float*)ccmalloc(sizeof(float) * feature_size * (positives->rnum + negatives->rnum));
	ccv_disable_cache(); // clean up cache so we have enough space to run it
#ifdef USE_DISPATCH
	dispatch_semaphore_t sema = dispatch_semaphore_create(1);
	dispatch_apply(positives->rnum + negatives->rnum, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
#else
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
#endif
#ifdef USE_DISPATCH
		dispatch_semaphore_wait(sema, DISPATCH_TIME_FOREVER);
#endif
		if (i % 37 == 0 || i == positives->rnum + negatives->rnum - 1) // don't flush too fast
			FLUSH(" - precompute %d features through %d%% (%d / %d) examples", feature_size, (int)(i + 1) * 100 / (positives->rnum + negatives->rnum), (int)i + 1, positives->rnum + negatives->rnum);
#ifdef USE_DISPATCH
		dispatch_semaphore_signal(sema);
		int j;
#endif
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(i < positives->rnum ? positives : negatives, i < positives->rnum ? i : i - positives->rnum);
		a->data.u8 = (unsigned char*)(a + 1); // re-host the pointer to the right place
		ccv_dense_matrix_t* icf = 0;
		// we have 1px padding around the image
		ccv_icf(a, &icf, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(icf);
		float* ptr = sat->data.f32;
		int ch = CCV_GET_CHANNEL(sat->type);
		for (j = 0; j < feature_size; j++)
		{
			ccv_icf_feature_t* feature = features + j;
			float c = _ccv_icf_run_feature(feature, ptr, sat->cols, ch, 1, 1);
			assert(isfinite(c));
			featval[(size_t)j * (positives->rnum + negatives->rnum) + i] = c;
		}
		ccv_matrix_free(sat);
#ifdef USE_DISPATCH
	});
	dispatch_release(sema);
#else
	}
#endif
	printf("\n");
	uint8_t* computed = precomputed;
	float* pfeatval = featval;
	for (i = 0; i < feature_size; i++)
	{
		if (i % 37 == 0 || i == feature_size - 1) // don't flush too fast
			FLUSH(" - precompute %d examples through %d%% (%d / %d) features", positives->rnum + negatives->rnum, (i + 1) * 100 / feature_size, i + 1, feature_size);
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
			sortkv[j].value = pfeatval[j], sortkv[j].index = j;
		_ccv_icf_precomputed_ordering(sortkv, positives->rnum + negatives->rnum, 0);
		// the first flag denotes if the subsequent one are equal to the previous one (if so, we have to skip both of them)
		for (j = 0; j < positives->rnum + negatives->rnum - 1; j++)
			_ccv_icf_1_uint1_1_uint23_to_3_uint8(sortkv[j].value == sortkv[j + 1].value, sortkv[j].index, computed + j * 3);
		j = positives->rnum + negatives->rnum - 1;
		_ccv_icf_1_uint1_1_uint23_to_3_uint8(0, sortkv[j].index, computed + j * 3);
		computed += step;
		pfeatval += positives->rnum + negatives->rnum;
	}
	ccfree(featval);
	ccfree(sortkv);
	printf("\n - features are precomputed on examples and will occupy %uM memory\n", (uint32_t)((feature_size * step) / (1024 * 1024)));
	return precomputed;
}

typedef struct {
	uint32_t pass;
	double weigh[4];
	int first_feature;
	uint8_t* lut;
} ccv_icf_decision_tree_cache_t;

static inline float _ccv_icf_compute_threshold_between(ccv_icf_feature_t* feature, uint8_t* computed, ccv_array_t* positives, ccv_array_t* negatives, int index0, int index1)
{
	float c[2];
	uint32_t b[2] = {
		_ccv_icf_3_uint8_to_1_uint23(computed + index0 * 3),
		_ccv_icf_3_uint8_to_1_uint23(computed + index1 * 3),
	};
	ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(b[0] < positives->rnum ? positives : negatives, b[0] < positives->rnum ? b[0] : b[0] - positives->rnum);
	a->data.u8 = (unsigned char*)(a + 1); // re-host the pointer to the right place
	c[0] = _ccv_icf_run_feature_on_example(feature, a);
	a = (ccv_dense_matrix_t*)ccv_array_get(b[1] < positives->rnum ? positives : negatives, b[1] < positives->rnum ? b[1] : b[1] - positives->rnum);
	a->data.u8 = (unsigned char*)(a + 1); // re-host the pointer to the right place
	c[1] = _ccv_icf_run_feature_on_example(feature, a);
	return (c[0] + c[1]) * 0.5;
}

static inline void _ccv_icf_example_correct(ccv_icf_example_state_t* example_state, uint8_t* computed, uint8_t* lut, int leaf, ccv_array_t* positives, ccv_array_t* negatives, int start, int end)
{
	int i;
	for (i = start; i <= end; i++)
	{
		uint32_t index = _ccv_icf_3_uint8_to_1_uint23(computed + i * 3);
		if (!lut || lut[index] == leaf)
			example_state[index].correct = (index < positives->rnum);
	}
}

typedef struct {
	int error_index;
	double error_rate;
	double weigh[2];
	int count[2];
} ccv_icf_first_feature_find_t;

static ccv_icf_decision_tree_cache_t _ccv_icf_find_first_feature(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, uint8_t* precomputed, ccv_icf_example_state_t* example_state, ccv_icf_feature_t* feature)
{
	int i;
	assert(feature != 0);
	ccv_icf_decision_tree_cache_t intermediate_cache;
	double aweigh0 = 0, aweigh1 = 0;
	for (i = 0; i < positives->rnum; i++)
		aweigh1 += example_state[i].weight, example_state[i].correct = 0; // assuming positive examples we get wrong
	for (i = positives->rnum; i < positives->rnum + negatives->rnum; i++)
		aweigh0 += example_state[i].weight, example_state[i].correct = 1; // assuming negative examples we get right
	size_t step = (3 * (positives->rnum + negatives->rnum) + 3) & -4;
	ccv_icf_first_feature_find_t* feature_find = (ccv_icf_first_feature_find_t*)ccmalloc(sizeof(ccv_icf_first_feature_find_t) * feature_size);
#ifdef USE_DISPATCH
	dispatch_apply(feature_size, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
#else
	for (i = 0; i < feature_size; i++)
	{
#endif
		ccv_icf_first_feature_find_t min_find = {
			.error_rate = 1.0,
			.error_index = 0,
			.weigh = {0, 0},
			.count = {0, 0},
		};
		double weigh[2] = {0, 0};
		int count[2] = {0, 0};
		int j;
		uint8_t* computed = precomputed + step * i;
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
		{
			uint8_t skip;
			uint32_t index;
			_ccv_icf_3_uint8_to_1_uint1_1_uint23(computed + j * 3, &skip, &index);
			conditional_assert(j == positives->rnum + negatives->rnum - 1, !skip);
			assert(index >= 0 && index < positives->rnum + negatives->rnum);
			weigh[index < positives->rnum] += example_state[index].weight;
			assert(example_state[index].weight > 0);
			assert(weigh[0] <= aweigh0 + 1e-10 && weigh[1] <= aweigh1 + 1e-10);
			++count[index < positives->rnum];
			if (skip) // the current index is equal to the next one, we cannot differentiate, therefore, skip
				continue;
			double error_rate = ccv_min(weigh[0] + aweigh1 - weigh[1], weigh[1] + aweigh0 - weigh[0]);
			assert(error_rate > 0);
			if (error_rate < min_find.error_rate)
			{
				min_find.error_index = j;
				min_find.error_rate = error_rate;
				min_find.weigh[0] = weigh[0];
				min_find.weigh[1] = weigh[1];
				min_find.count[0] = count[0];
				min_find.count[1] = count[1];
			}
		}
		feature_find[i] = min_find;
#ifdef USE_DISPATCH
	});
#else
	}
#endif
	ccv_icf_first_feature_find_t best = {
		.error_rate = 1.0,
		.error_index = -1,
		.weigh = {0, 0},
		.count = {0, 0},
	};
	int feature_index = 0;
	for (i = 0; i < feature_size; i++)
		if (feature_find[i].error_rate < best.error_rate)
		{
			best = feature_find[i];
			feature_index = i;
		}
	ccfree(feature_find);
	*feature = features[feature_index];
	uint8_t* computed = precomputed + step * feature_index;
	intermediate_cache.lut = (uint8_t*)ccmalloc(positives->rnum + negatives->rnum);
	assert(best.error_index < positives->rnum + negatives->rnum - 1 && best.error_index >= 0);
	if (best.weigh[0] + aweigh1 - best.weigh[1] < best.weigh[1] + aweigh0 - best.weigh[0])
	{
		for (i = 0; i < positives->rnum + negatives->rnum; i++)
			intermediate_cache.lut[_ccv_icf_3_uint8_to_1_uint23(computed + i * 3)] = (i <= best.error_index);
		feature->beta = _ccv_icf_compute_threshold_between(feature, computed, positives, negatives, best.error_index, best.error_index + 1);
		// revert the sign of alpha, after threshold is computed
		for (i = 0; i < feature->count; i++)
			feature->alpha[i] = -feature->alpha[i];
		intermediate_cache.weigh[0] = aweigh0 - best.weigh[0];
		intermediate_cache.weigh[1] = aweigh1 - best.weigh[1];
		intermediate_cache.weigh[2] = best.weigh[0];
		intermediate_cache.weigh[3] = best.weigh[1];
		intermediate_cache.pass = 3;
		if (best.count[0] == 0)
			intermediate_cache.pass &= 2; // only positive examples in the right, no need to build right leaf
		if (best.count[1] == positives->rnum)
			intermediate_cache.pass &= 1; // no positive examples in the left, no need to build left leaf
		if (!(intermediate_cache.pass & 1)) // mark positives in the right as correct, if we don't have right leaf
			_ccv_icf_example_correct(example_state, computed, 0, 0, positives, negatives, 0, best.error_index);
	} else {
		for (i = 0; i < positives->rnum + negatives->rnum; i++)
			intermediate_cache.lut[_ccv_icf_3_uint8_to_1_uint23(computed + i * 3)] = (i > best.error_index);
		feature->beta = -_ccv_icf_compute_threshold_between(feature, computed, positives, negatives, best.error_index, best.error_index + 1);
		intermediate_cache.weigh[0] = best.weigh[0];
		intermediate_cache.weigh[1] = best.weigh[1];
		intermediate_cache.weigh[2] = aweigh0 - best.weigh[0];
		intermediate_cache.weigh[3] = aweigh1 - best.weigh[1];
		intermediate_cache.pass = 3;
		if (best.count[0] == negatives->rnum)
			intermediate_cache.pass &= 2; // only positive examples in the right, no need to build right leaf
		if (best.count[1] == 0)
			intermediate_cache.pass &= 1; // no positive examples in the left, no need to build left leaf
		if (!(intermediate_cache.pass & 1)) // mark positives in the right as correct if we don't have right leaf
			_ccv_icf_example_correct(example_state, computed, 0, 0, positives, negatives, best.error_index + 1, positives->rnum + negatives->rnum - 1);
	}
	intermediate_cache.first_feature = feature_index;
	return intermediate_cache;
}

typedef struct {
	int error_index;
	double error_rate;
	double weigh[2];
} ccv_icf_second_feature_find_t;

static double _ccv_icf_find_second_feature(ccv_icf_decision_tree_cache_t intermediate_cache, int leaf, ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, uint8_t* precomputed, ccv_icf_example_state_t* example_state, ccv_icf_feature_t* feature)
{
	int i;
	size_t step = (3 * (positives->rnum + negatives->rnum) + 3) & -4;
	uint8_t* lut = intermediate_cache.lut;
	double* aweigh = intermediate_cache.weigh + leaf * 2;
	ccv_icf_second_feature_find_t* feature_find = (ccv_icf_second_feature_find_t*)ccmalloc(sizeof(ccv_icf_second_feature_find_t) * feature_size);
#ifdef USE_DISPATCH
	dispatch_apply(feature_size, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t i) {
#else
	for (i = 0; i < feature_size; i++)
	{
#endif
		ccv_icf_second_feature_find_t min_find = {
			.error_rate = 1.0,
			.error_index = 0,
			.weigh = {0, 0},
		};
		double weigh[2] = {0, 0};
		uint8_t* computed = precomputed + step * i;
		int j, k;
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
		{
			uint8_t skip;
			uint32_t index;
			_ccv_icf_3_uint8_to_1_uint1_1_uint23(computed + j * 3, &skip, &index);
			conditional_assert(j == positives->rnum + negatives->rnum - 1, !skip);
			assert(index >= 0 && index < positives->rnum + negatives->rnum);
			// only care about part of the data
			if (lut[index] == leaf)
			{
				uint8_t leaf_skip = 0;
				for (k = j + 1; skip; k++)
				{
					uint32_t new_index;
					_ccv_icf_3_uint8_to_1_uint1_1_uint23(computed + j * 3, &skip, &new_index);
					// if the next equal one is the same leaf, we cannot distinguish them, skip
					if ((leaf_skip = (lut[new_index] == leaf)))
						break;
					conditional_assert(k == positives->rnum + negatives->rnum - 1, !skip);
				}
				weigh[index < positives->rnum] += example_state[index].weight;
				if (leaf_skip)
					continue;
				assert(example_state[index].weight > 0);
				assert(weigh[0] <= aweigh[0] + 1e-10 && weigh[1] <= aweigh[1] + 1e-10);
				double error_rate = ccv_min(weigh[0] + aweigh[1] - weigh[1], weigh[1] + aweigh[0] - weigh[0]);
				if (error_rate < min_find.error_rate)
				{
					min_find.error_index = j;
					min_find.error_rate = error_rate;
					min_find.weigh[0] = weigh[0];
					min_find.weigh[1] = weigh[1];
				}
			}
		}
		feature_find[i] = min_find;
#ifdef USE_DISPATCH
	});
#else
	}
#endif
	ccv_icf_second_feature_find_t best = {
		.error_rate = 1.0,
		.error_index = -1,
		.weigh = {0, 0},
	};
	int feature_index = 0;
	for (i = 0; i < feature_size; i++)
		if (feature_find[i].error_rate < best.error_rate)
		{
			best = feature_find[i];
			feature_index = i;
		}
	ccfree(feature_find);
	*feature = features[feature_index];
	uint8_t* computed = precomputed + step * feature_index;
	assert(best.error_index < positives->rnum + negatives->rnum - 1 && best.error_index >= 0);
	if (best.weigh[0] + aweigh[1] - best.weigh[1] < best.weigh[1] + aweigh[0] - best.weigh[0])
	{
		feature->beta = _ccv_icf_compute_threshold_between(feature, computed, positives, negatives, best.error_index, best.error_index + 1);
		// revert the sign of alpha, after threshold is computed
		for (i = 0; i < feature->count; i++)
			feature->alpha[i] = -feature->alpha[i];
		// mark everything on the right properly
		_ccv_icf_example_correct(example_state, computed, lut, leaf, positives, negatives, 0, best.error_index);
		return best.weigh[1] + aweigh[0] - best.weigh[0];
	} else {
		feature->beta = -_ccv_icf_compute_threshold_between(feature, computed, positives, negatives, best.error_index, best.error_index + 1);
		// mark everything on the right properly
		_ccv_icf_example_correct(example_state, computed, lut, leaf, positives, negatives, best.error_index + 1, positives->rnum + negatives->rnum - 1);
		return best.weigh[0] + aweigh[1] - best.weigh[1];
	}
}

static double _ccv_icf_find_best_weak_classifier(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, uint8_t* precomputed, ccv_icf_example_state_t* example_state, ccv_icf_decision_tree_t* weak_classifier)
{
	// we are building the specific depth-2 decision tree
	ccv_icf_decision_tree_cache_t intermediate_cache = _ccv_icf_find_first_feature(features, feature_size, positives, negatives, precomputed, example_state, weak_classifier->features);
	// find the left feature
	// for the pass, 10 is the left branch, 01 is the right branch
	weak_classifier->pass = intermediate_cache.pass;
	double rate = 0;
	if (weak_classifier->pass & 0x2)
		rate += _ccv_icf_find_second_feature(intermediate_cache, 0, features, feature_size, positives, negatives, precomputed, example_state, weak_classifier->features + 1);
	else
		rate += intermediate_cache.weigh[0]; // the negative weights covered by first feature
	// find the right feature
	if (weak_classifier->pass & 0x1)
		rate += _ccv_icf_find_second_feature(intermediate_cache, 1, features, feature_size, positives, negatives, precomputed, example_state, weak_classifier->features + 2);
	else
		rate += intermediate_cache.weigh[3]; // the positive weights covered by first feature
	ccfree(intermediate_cache.lut);
	return rate;
}

static ccv_array_t* _ccv_icf_collect_validates(gsl_rng* rng, ccv_size_t size, ccv_margin_t margin, ccv_array_t* validatefiles, int grayscale)
{
	ccv_array_t* validates = ccv_array_new(ccv_compute_dense_matrix_size(size.height + margin.top + margin.bottom + 2, size.width + margin.left + margin.right + 2, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), validatefiles->rnum, 0);
	int i;
	// collect tests
	for (i = 0; i < validatefiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(validatefiles, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
		if (image == 0)
		{
			printf("\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
			continue;
		}
		ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, file_info->pose, size, margin, 0, 0, 0);
		feature->sig = 0;
		ccv_array_push(validates, feature);
		ccv_matrix_free(feature);
		ccv_matrix_free(image);
	}
	return validates;
}

static ccv_array_t* _ccv_icf_collect_positives(gsl_rng* rng, ccv_size_t size, ccv_margin_t margin, ccv_array_t* posfiles, int posnum, float deform_angle, float deform_scale, float deform_shift, int grayscale)
{
	ccv_array_t* positives = ccv_array_new(ccv_compute_dense_matrix_size(size.height + margin.top + margin.bottom + 2, size.width + margin.left + margin.right + 2, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), posnum, 0);
	int i, j, q;
	// collect positives (with random deformation)
	for (i = 0; i < posnum;)
	{
		FLUSH(" - collect positives %d%% (%d / %d)", (i + 1) * 100 / posnum, i + 1, posnum);
		double ratio = (double)(posnum - i) / posfiles->rnum;
		for (j = 0; j < posfiles->rnum && i < posnum; j++)
		{
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
			if (image == 0)
			{
				printf("\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
				continue;
			}
			for (q = 0; q < ratio; q++)
				if (q < (int)ratio || gsl_rng_uniform(rng) <= ratio - (int)ratio)
				{
					FLUSH(" - collect positives %d%% (%d / %d)", (i + 1) * 100 / posnum, i + 1, posnum);
					ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, file_info->pose, size, margin, deform_angle, deform_scale, deform_shift);
					feature->sig = 0;
					ccv_array_push(positives, feature);
					ccv_matrix_free(feature);
					++i;
					if (i >= posnum)
						break;
				}
			ccv_matrix_free(image);
		}
	}
	printf("\n");
	return positives;
}

static uint64_t* _ccv_icf_precompute_classifier_cascade(ccv_icf_classifier_cascade_t* cascade, ccv_array_t* positives)
{
	int step = ((cascade->count - 1) >> 6) + 1;
	uint64_t* precomputed = (uint64_t*)ccmalloc(sizeof(uint64_t) * positives->rnum * step);
	uint64_t* result = precomputed;
	int i, j;
	for (i = 0; i < positives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)(ccv_array_get(positives, i));
		a->data.u8 = (uint8_t*)(a + 1);
		ccv_dense_matrix_t* icf = 0;
		ccv_icf(a, &icf, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(icf);
		float* ptr = sat->data.f32;
		int ch = CCV_GET_CHANNEL(sat->type);
		for (j = 0; j < cascade->count; j++)
			if (_ccv_icf_run_weak_classifier(cascade->weak_classifiers + j,  ptr, sat->cols, ch, 1, 1))
				precomputed[j >> 6] |= (1UL << (j & 63));
			else
				precomputed[j >> 6] &= ~(1UL << (j & 63));
		ccv_matrix_free(sat);
		precomputed += step;
	}
	return result;
}

#define less_than(s1, s2, aux) ((s1) > (s2))
static CCV_IMPLEMENT_QSORT(_ccv_icf_threshold_rating, float, less_than)
#undef less_than

static void _ccv_icf_classifier_cascade_soft_with_validates(ccv_array_t* validates, ccv_icf_classifier_cascade_t* cascade, double min_accept)
{
	int i, j;
	int step = ((cascade->count - 1) >> 6) + 1;
	uint64_t* precomputed = _ccv_icf_precompute_classifier_cascade(cascade, validates);
	float* positive_rate = (float*)ccmalloc(sizeof(float) * validates->rnum);
	uint64_t* computed = precomputed;
	for (i = 0; i < validates->rnum; i++)
	{
		positive_rate[i] = 0;
		for (j = 0; j < cascade->count; j++)
		{
			uint64_t accept = computed[j >> 6] & (1UL << (j & 63));
			positive_rate[i] += cascade->weak_classifiers[j].weigh[!!accept];
		}
		computed += step;
	}
	_ccv_icf_threshold_rating(positive_rate, validates->rnum, 0);
	float threshold = positive_rate[ccv_min((int)(min_accept * (validates->rnum + 0.5) - 0.5), validates->rnum - 1)];
	ccfree(positive_rate);
	computed = precomputed;
	// compute the final acceptance per validates / negatives with final threshold
	uint64_t* acceptance = (uint64_t*)cccalloc(((validates->rnum - 1) >> 6) + 1, sizeof(uint64_t));
	int true_positives = 0;
	for (i = 0; i < validates->rnum; i++)
	{
		float rate = 0;
		for (j = 0; j < cascade->count; j++)
		{
			uint64_t accept = computed[j >> 6] & (1UL << (j & 63));
			rate += cascade->weak_classifiers[j].weigh[!!accept];
		}
		if (rate >= threshold)
		{
			acceptance[i >> 6] |= (1UL << (i & 63));
			++true_positives;
		} else
			acceptance[i >> 6] &= ~(1UL << (i & 63));
		computed += step;
	}
	printf(" - at threshold %f, true positive rate: %f%%\n", threshold, (float)true_positives * 100 / validates->rnum);
	float* rate = (float*)cccalloc(validates->rnum, sizeof(float));
	for (j = 0; j < cascade->count; j++)
	{
		computed = precomputed;
		for (i = 0; i < validates->rnum; i++)
		{
			uint64_t correct = computed[j >> 6] & (1UL << (j & 63));
			rate[i] += cascade->weak_classifiers[j].weigh[!!correct];
			computed += step;
		}
		float threshold = FLT_MAX;
		// find a threshold that keeps all accepted validates still acceptable
		for (i = 0; i < validates->rnum; i++)
		{
			uint64_t correct = acceptance[i >> 6] & (1UL << (i & 63));
			if (correct && rate[i] < threshold)
				threshold = rate[i];
		}
		cascade->weak_classifiers[j].threshold = threshold - 1e-10;
	}
	ccfree(rate);
	ccfree(acceptance);
	ccfree(precomputed);
}

typedef struct {
	ccv_point_t point;
	float sum;
} ccv_point_with_sum_t;

static void _ccv_icf_bootstrap_negatives(ccv_icf_classifier_cascade_t* cascade, ccv_array_t* negatives, gsl_rng* rng, ccv_array_t* bgfiles, int negnum, int grayscale, int spread, ccv_icf_param_t params)
{
#ifdef USE_DISPATCH
	__block int i;
#else
	int i;
#endif
#ifdef USE_DISPATCH
	__block int fppi = 0, is = 0;
#else
	int fppi = 0, is = 0;
#endif
	int t = 0;
	for (i = 0; i < negnum;)
	{
		double ratio = (double)(negnum - i) / bgfiles->rnum;
#ifdef USE_DISPATCH
		dispatch_semaphore_t sem = dispatch_semaphore_create(1);
		dispatch_apply(bgfiles->rnum, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t j) {
#else
		size_t j;
		for (j = 0; j < bgfiles->rnum; j++)
		{
#endif
			int k, x, y, q, p;
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccmalloc(ccv_compute_dense_matrix_size(cascade->size.height + 2, cascade->size.width + 2, (grayscale ? CCV_C1 : CCV_C3) | CCV_8U));
#ifdef USE_DISPATCH
			dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#endif
			if (i >= negnum || (spread && ratio < 1 && gsl_rng_uniform(rng) > ratio))
			{
				ccfree(a);
#ifdef USE_DISPATCH
				dispatch_semaphore_signal(sem);
				return;
#else
				continue;
#endif
			}
			FLUSH(" - bootstrap negatives %d%% (%d / %d) [%u / %d] %s", (i + 1) * 100 / negnum, i + 1, negnum, (uint32_t)(j + 1), bgfiles->rnum, spread ? "" : "without statistic balancing");
#ifdef USE_DISPATCH
			gsl_rng* crng = gsl_rng_alloc(gsl_rng_default);
			gsl_rng_set(crng, gsl_rng_get(rng));
			dispatch_semaphore_signal(sem);
#else
			gsl_rng* crng = rng;
#endif
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(bgfiles, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
			if (image == 0)
			{
				printf("\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
				ccfree(a);
#ifdef USE_DISPATCH
				gsl_rng_free(crng);
				return;
#else
				continue;
#endif
			}
			if (ccv_max(image->rows, image->cols) < 800 ||
				image->rows <= (cascade->size.height - cascade->margin.top - cascade->margin.bottom) ||
				image->cols <= (cascade->size.width - cascade->margin.left - cascade->margin.right)) // background is too small, blow it up to next scale
			{
				ccv_dense_matrix_t* blowup = 0;
				ccv_sample_up(image, &blowup, 0, 0, 0);
				ccv_matrix_free(image);
				image = blowup;
			}
			if (image->rows <= (cascade->size.height - cascade->margin.top - cascade->margin.bottom) ||
				image->cols <= (cascade->size.width - cascade->margin.left - cascade->margin.right)) // background is still too small, abort
			{
				ccv_matrix_free(image);
				ccfree(a);
#ifdef USE_DISPATCH
				gsl_rng_free(crng);
				return;
#else
				continue;
#endif
			}
			double scale = pow(2., 1. / (params.interval + 1.));
			int next = params.interval + 1;
			int scale_upto = (int)(log(ccv_min((double)image->rows / (cascade->size.height - cascade->margin.top - cascade->margin.bottom), (double)image->cols / (cascade->size.width - cascade->margin.left - cascade->margin.right))) / log(scale) - DBL_MIN) + 1;
			ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)ccmalloc(scale_upto * sizeof(ccv_dense_matrix_t*));
			memset(pyr, 0, scale_upto * sizeof(ccv_dense_matrix_t*));
#ifdef USE_DISPATCH
			dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#endif
			++is; // how many images are scanned
#ifdef USE_DISPATCH
			dispatch_semaphore_signal(sem);
#endif
			if (t % 2 != 0)
				ccv_flip(image, 0, 0, CCV_FLIP_X);
			if (t % 4 >= 2)
				ccv_flip(image, 0, 0, CCV_FLIP_Y);
			pyr[0] = image;
			for (q = 1; q < ccv_min(params.interval + 1, scale_upto); q++)
				ccv_resample(pyr[0], &pyr[q], 0, (int)(pyr[0]->rows / pow(scale, q)), (int)(pyr[0]->cols / pow(scale, q)), CCV_INTER_AREA);
			for (q = next; q < scale_upto; q++)
				ccv_sample_down(pyr[q - next], &pyr[q], 0, 0, 0);
			for (q = 0; q < scale_upto; q++)
			{
#ifdef USE_DISPATCH
				dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#endif
				if (i >= negnum)
				{
#ifdef USE_DISPATCH
					dispatch_semaphore_signal(sem);
#endif
					ccv_matrix_free(pyr[q]);
					continue;
				}
#ifdef USE_DISPATCH
				dispatch_semaphore_signal(sem);
#endif
				ccv_dense_matrix_t* bordered = 0;
				ccv_border(pyr[q], (ccv_matrix_t**)&bordered, 0, cascade->margin);
				ccv_matrix_free(pyr[q]);
				ccv_dense_matrix_t* icf = 0;
				ccv_icf(bordered, &icf, 0);
				ccv_dense_matrix_t* sat = 0;
				ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
				ccv_matrix_free(icf);
				assert(sat->rows == bordered->rows + 1 && sat->cols == bordered->cols + 1);
				int ch = CCV_GET_CHANNEL(sat->type);
				float* ptr = sat->data.f32 + sat->cols * ch;
				ccv_array_t* seq = ccv_array_new(sizeof(ccv_point_with_sum_t), 64, 0);
				for (y = 1; y < sat->rows - cascade->size.height - 2; y += params.step_through)
				{
					for (x = 1; x < sat->cols - cascade->size.width - 2; x += params.step_through)
					{
						int pass = 1;
						float sum = 0;
						for (p = 0; p < cascade->count; p++)
						{
							ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + p;
							int c = _ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, x, 0);
							sum += weak_classifier->weigh[c];
							if (sum < weak_classifier->threshold)
							{
								pass = 0;
								break;
							}
						}
						if (pass)
						{
							ccv_point_with_sum_t point;
							point.point = ccv_point(x - 1, y - 1);
							point.sum = sum;
							ccv_array_push(seq, &point);
						}
					}
					ptr += sat->cols * ch * params.step_through;
				}
				ccv_matrix_free(sat);
				// shuffle negatives so that we don't have too biased negatives
#ifdef USE_DISPATCH
				dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#endif
				fppi += seq->rnum; // how many detections we have in total
#ifdef USE_DISPATCH
				dispatch_semaphore_signal(sem);
#endif
				if (seq->rnum > 0)
				{
					gsl_ran_shuffle(crng, ccv_array_get(seq, 0), seq->rnum, seq->rsize);
					/* so that we at least collect 10 from each scale */
					for (p = 0; p < (spread ? ccv_min(10, seq->rnum) : seq->rnum); p++) // collect enough negatives from this scale
					{
						a = ccv_dense_matrix_new(cascade->size.height + 2, cascade->size.width + 2, (grayscale ? CCV_C1 : CCV_C3) | CCV_8U, a, 0);
						ccv_point_with_sum_t* point = (ccv_point_with_sum_t*)ccv_array_get(seq, p);
						ccv_slice(bordered, (ccv_matrix_t**)&a, 0, point->point.y, point->point.x, a->rows, a->cols);
						assert(bordered->rows >= point->point.y + a->rows && bordered->cols >= point->point.x + a->cols);
						a->sig = 0;
						// verify the data we sliced is worthy negative
						ccv_dense_matrix_t* icf = 0;
						ccv_icf(a, &icf, 0);
						ccv_dense_matrix_t* sat = 0;
						ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
						ccv_matrix_free(icf);
						float* ptr = sat->data.f32;
						int ch = CCV_GET_CHANNEL(sat->type);
						int pass = 1;
						float sum = 0;
						for (k = 0; k < cascade->count; k++)
						{
							ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + k;
							int c = _ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, 1, 1);
							sum += weak_classifier->weigh[c];
							if (sum < weak_classifier->threshold)
							{
								pass = 0;
								break;
							}
						}
						ccv_matrix_free(sat);
						if (pass)
						{
#ifdef USE_DISPATCH
							dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#endif
							if (i < negnum)
								ccv_array_push(negatives, a);
							++i;
							if (i >= negnum)
							{
#ifdef USE_DISPATCH
								dispatch_semaphore_signal(sem);
#endif
								break;
							}
#ifdef USE_DISPATCH
							dispatch_semaphore_signal(sem);
#endif
						}
					}
				}
				ccv_array_free(seq);
				ccv_matrix_free(bordered);
			}
			ccfree(pyr);
			ccfree(a);
#ifdef USE_DISPATCH
			gsl_rng_free(crng);
		});
		dispatch_release(sem);
#else
		}
#endif
		if ((double)fppi / is <= (double)negnum / bgfiles->rnum) // if the targeted negative per image is bigger than our fppi, we don't prob anymore
			spread = 0;
		++t;
		if (t > (spread ? 4 : 3) && !spread) // we've go over 4 or 3 transformations (original, flip x, flip y, flip x & y, [and original again]), and nothing we can do now
			break;
	}
	printf("\n");
}

static ccv_array_t* _ccv_icf_collect_negatives(gsl_rng* rng, ccv_size_t size, ccv_margin_t margin, ccv_array_t* bgfiles, int negnum, float deform_angle, float deform_scale, float deform_shift, int grayscale)
{
	ccv_array_t* negatives = ccv_array_new(ccv_compute_dense_matrix_size(size.height + margin.top + margin.bottom + 2, size.width + margin.left + margin.right + 2, CCV_8U | (grayscale ? CCV_C1 : CCV_C3)), negnum, 0);
	int i, j, q;
	// randomly collect negatives (with random deformation)
	for (i = 0; i < negnum;)
	{
		FLUSH(" - collect negatives %d%% (%d / %d)", (i + 1) * 100 / negnum, i + 1, negnum);
		double ratio = (double)(negnum - i) / bgfiles->rnum;
		for (j = 0; j < bgfiles->rnum && i < negnum; j++)
		{
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(bgfiles, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | (grayscale ? CCV_IO_GRAY : CCV_IO_RGB_COLOR));
			if (image == 0)
			{
				printf("\n - %s: cannot be open, possibly corrupted\n", file_info->filename);
				continue;
			}
			double max_scale_ratio = ccv_min((double)image->rows / size.height, (double)image->cols / size.width);
			if (max_scale_ratio <= 0.5) // too small to be interesting
				continue;
			for (q = 0; q < ratio; q++)
				if (q < (int)ratio || gsl_rng_uniform(rng) <= ratio - (int)ratio)
				{
					FLUSH(" - collect negatives %d%% (%d / %d)", (i + 1) * 100 / negnum, i + 1, negnum);
					ccv_decimal_pose_t pose;
					double scale_ratio = gsl_rng_uniform(rng) * (max_scale_ratio - 0.5) + 0.5;
					pose.a = size.width * 0.5 * scale_ratio;
					pose.b = size.height * 0.5 * scale_ratio;
					pose.x = gsl_rng_uniform_int(rng, ccv_max((int)(image->cols - pose.a * 2 + 1.5), 1)) + pose.a;
					pose.y = gsl_rng_uniform_int(rng, ccv_max((int)(image->rows - pose.b * 2 + 1.5), 1)) + pose.b;
					pose.roll = pose.pitch = pose.yaw = 0;
					ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, pose, size, margin, deform_angle, deform_scale, deform_shift);
					feature->sig = 0;
					ccv_array_push(negatives, feature);
					ccv_matrix_free(feature);
					++i;
					if (i >= negnum)
						break;
				}
			ccv_matrix_free(image);
		}
	}
	printf("\n");
	return negatives;
}

#ifdef USE_SANITY_ASSERTION
static double _ccv_icf_rate_weak_classifier(ccv_icf_decision_tree_t* weak_classifier, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_example_state_t* example_state)
{
	int i;
	double rate = 0;
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(i < positives->rnum ? positives : negatives, i < positives->rnum ? i : i - positives->rnum);
		a->data.u8 = (uint8_t*)(a + 1); // re-host the pointer to the right place
		ccv_dense_matrix_t* icf = 0;
		// we have 1px padding around the image
		ccv_icf(a, &icf, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(icf);
		float* ptr = sat->data.f32;
		int ch = CCV_GET_CHANNEL(sat->type);
		if (i < positives->rnum)
		{
			if (_ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, 1, 1))
			{
				assert(example_state[i].correct);
				rate += example_state[i].weight;
			} else {
				assert(!example_state[i].correct);
			}
		} else {
			if (!_ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, 1, 1))
			{
				assert(example_state[i].correct);
				rate += example_state[i].weight;
			} else {
				assert(!example_state[i].correct);
			}
		}
		ccv_matrix_free(sat);
	}
	return rate;
}
#endif
#endif

ccv_icf_classifier_cascade_t* ccv_icf_classifier_cascade_new(ccv_array_t* posfiles, int posnum, ccv_array_t* bgfiles, int negnum, ccv_array_t* validatefiles, const char* dir, ccv_icf_new_param_t params)
{
#ifdef HAVE_GSL
	_ccv_icf_check_params(params);
	assert(posfiles->rnum > 0);
	assert(bgfiles->rnum > 0);
	assert(posnum > 0 && negnum > 0);
	printf("with %d positive examples and %d negative examples\n"
		   "positive examples are going to be collected from %d positive images\n"
		   "negative examples are are going to be collected from %d background images\n",
		   posnum, negnum, posfiles->rnum, bgfiles->rnum);
	printf("use color? %s\n", params.grayscale ? "no" : "yes");
	printf("feature pool size : %d\n"
		   "weak classifier count : %d\n"
		   "soft cascade acceptance : %lf\n"
		   "minimum dimension of ICF feature : %d\n"
		   "number of bootstrap : %d\n"
		   "distortion on translation : %f\n"
		   "distortion on rotation : %f\n"
		   "distortion on scale : %f\n"
	       "learn ICF classifier cascade at size %dx%d with margin (%d,%d,%d,%d)\n"
		   "------------------------\n",
		   params.feature_size, params.weak_classifier, params.acceptance, params.min_dimension, params.bootstrap, params.deform_shift, params.deform_angle, params.deform_scale, params.size.width, params.size.height, params.margin.left, params.margin.top, params.margin.right, params.margin.bottom);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	// we will keep all states inside this structure for easier save / resume across process
	// this should work better than ad-hoc one we used in DPM / BBF implementation
	ccv_icf_classifier_cascade_state_t z;
	z.params = params;
	ccv_function_state_begin(_ccv_icf_read_classifier_cascade_state, z, dir);
	z.classifier->grayscale = params.grayscale;
	z.size = params.size;
	z.margin = params.margin;
	z.classifier->size = ccv_size(z.size.width + z.margin.left + z.margin.right, z.size.height + z.margin.top + z.margin.bottom);
	z.features = (ccv_icf_feature_t*)ccmalloc(sizeof(ccv_icf_feature_t) * params.feature_size);
	// generate random features
	for (z.i = 0; z.i < params.feature_size; z.i++)
		_ccv_icf_randomize_feature(rng, z.classifier->size, params.min_dimension, z.features + z.i, params.grayscale);
	z.x.features = 0;
	ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
	z.positives = _ccv_icf_collect_positives(rng, z.size, z.margin, posfiles, posnum, params.deform_angle, params.deform_scale, params.deform_shift, params.grayscale);
	z.x.positives = 0;
	ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
	z.negatives = _ccv_icf_collect_negatives(rng, z.size, z.margin, bgfiles, negnum, params.deform_angle, params.deform_scale, params.deform_shift, params.grayscale);
	z.x.negatives = 0;
	ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
	for (z.bootstrap = 0; z.bootstrap <= params.bootstrap; z.bootstrap++)
	{
		z.example_state = (ccv_icf_example_state_t*)ccmalloc(sizeof(ccv_icf_example_state_t) * (z.negatives->rnum + z.positives->rnum));
		memset(z.example_state, 0, sizeof(ccv_icf_example_state_t) * (z.negatives->rnum + z.positives->rnum));
		for (z.i = 0; z.i < z.positives->rnum + z.negatives->rnum; z.i++)
			z.example_state[z.i].weight = (z.i < z.positives->rnum) ? 0.5 / z.positives->rnum : 0.5 / z.negatives->rnum;
		z.x.example_state = 0;
		ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		z.precomputed = _ccv_icf_precompute_features(z.features, params.feature_size, z.positives, z.negatives);
		z.x.precomputed = 0;
		ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		for (z.i = 0; z.i < params.weak_classifier; z.i++)
		{
			z.classifier->count = z.i + 1;
			printf(" - boost weak classifier %d of %d\n", z.i + 1, params.weak_classifier);
			int j;
			ccv_icf_decision_tree_t weak_classifier;
			double rate = _ccv_icf_find_best_weak_classifier(z.features, params.feature_size, z.positives, z.negatives, z.precomputed, z.example_state, &weak_classifier);
			assert(rate > 0.5); // it has to be better than random chance
#ifdef USE_SANITY_ASSERTION
			double confirm_rate = _ccv_icf_rate_weak_classifier(&weak_classifier, z.positives, z.negatives, z.example_state);
#endif
			double alpha = sqrt((1 - rate) / rate);
			double beta = 1.0 / alpha;
			double c = log(rate / (1 - rate));
			weak_classifier.weigh[0] = -c;
			weak_classifier.weigh[1] = c;
			weak_classifier.threshold = 0;
			double reweigh = 0;
			for (j = 0; j < z.positives->rnum + z.negatives->rnum; j++)
			{
				z.example_state[j].weight *= (z.example_state[j].correct) ? alpha : beta;
				z.example_state[j].rate += weak_classifier.weigh[!((j < z.positives->rnum) ^ z.example_state[j].correct)];
				reweigh += z.example_state[j].weight;
			}
			reweigh = 1.0 / reweigh;
#ifdef USE_SANITY_ASSERTION
			printf(" - on all examples, best feature at rate %lf, confirm rate %lf\n", rate, confirm_rate);
#else
			printf(" - on all examples, best feature at rate %lf\n", rate);
#endif
			// balancing the weight to sum 1.0
			for (j = 0; j < z.positives->rnum + z.negatives->rnum; j++)
				z.example_state[j].weight *= reweigh;
			z.classifier->weak_classifiers[z.i] = weak_classifier;
			// compute the threshold at given acceptance
			float threshold = z.example_state[0].rate;
			for (j = 1; j < z.positives->rnum; j++)
				if (z.example_state[j].rate < threshold)
					threshold = z.example_state[j].rate;
			int true_positives = 0, false_positives = 0;
			for (j = 0; j < z.positives->rnum; j++)
				if (z.example_state[j].rate >= threshold)
					++true_positives;
			for (j = z.positives->rnum; j < z.positives->rnum + z.negatives->rnum; j++)
				if (z.example_state[j].rate >= threshold)
					++false_positives;
			printf(" - at threshold %f, true positive rate: %f%%, false positive rate: %f%% (%d)\n", threshold, (float)true_positives * 100 / z.positives->rnum, (float)false_positives * 100 / z.negatives->rnum, false_positives);
			printf(" - first feature :\n");
			for (j = 0; j < weak_classifier.features[0].count; j++)
				printf(" - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[0].channel[j], weak_classifier.features[0].sat[j * 2].x, weak_classifier.features[0].sat[j * 2].y, weak_classifier.features[0].sat[j * 2 + 1].x, weak_classifier.features[0].sat[j * 2 + 1].y);
			if (weak_classifier.pass & 0x2)
			{
				printf(" - second feature, on left :\n");
				for (j = 0; j < weak_classifier.features[1].count; j++)
					printf(" - | - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[1].channel[j], weak_classifier.features[1].sat[j * 2].x, weak_classifier.features[1].sat[j * 2].y, weak_classifier.features[1].sat[j * 2 + 1].x, weak_classifier.features[1].sat[j * 2 + 1].y);
			}
			if (weak_classifier.pass & 0x1)
			{
				printf(" - second feature, on right :\n");
				for (j = 0; j < weak_classifier.features[2].count; j++)
					printf(" - | - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[2].channel[j], weak_classifier.features[2].sat[j * 2].x, weak_classifier.features[2].sat[j * 2].y, weak_classifier.features[2].sat[j * 2 + 1].x, weak_classifier.features[2].sat[j * 2 + 1].y);
			}
			z.classifier->count = z.i + 1; // update count
			z.classifier->size = ccv_size(z.size.width + z.margin.left + z.margin.right, z.size.height + z.margin.top + z.margin.bottom);
			z.classifier->margin = z.margin;
			if (z.i + 1 < params.weak_classifier)
			{
				z.x.example_state = 0;
				z.x.classifier = 0;
				ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
			}
		}
		if (z.bootstrap < params.bootstrap) // collecting negatives, again
		{
			// free expensive memory
			ccfree(z.example_state);
			z.example_state = 0;
			ccfree(z.precomputed);
			z.precomputed = 0;
			_ccv_icf_classifier_cascade_soft_with_validates(z.positives, z.classifier, 1); // assuming perfect score, what's the soft cascading will be
			int exists = z.negatives->rnum;
			int spread_policy = z.bootstrap < 2; // we don't spread bootstrapping anymore after the first two bootstrappings
			// try to boostrap half negatives from perfect scoring
			_ccv_icf_bootstrap_negatives(z.classifier, z.negatives, rng, bgfiles, (negnum + 1) / 2, params.grayscale, spread_policy, params.detector);
			int leftover = negnum - (z.negatives->rnum - exists);
			if (leftover > 0)
			{
				// if we cannot get enough negative examples, now will use the validates data set to extract more
				ccv_array_t* validates = _ccv_icf_collect_validates(rng, z.size, z.margin, validatefiles, params.grayscale);
				_ccv_icf_classifier_cascade_soft_with_validates(validates, z.classifier, params.acceptance);
				ccv_array_free(validates);
				_ccv_icf_bootstrap_negatives(z.classifier, z.negatives, rng, bgfiles, leftover, params.grayscale, spread_policy, params.detector);
			}
			printf(" - after %d bootstrapping, learn with %d positives and %d negatives\n", z.bootstrap + 1, z.positives->rnum, z.negatives->rnum);
			z.classifier->count = 0; // reset everything
			z.x.negatives = 0;
		} else {
			z.x.example_state = 0;
			z.x.classifier = 0;
			ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		}
	}
	if (z.precomputed)
		ccfree(z.precomputed);
	if (z.example_state)
		ccfree(z.example_state);
	ccfree(z.features);
	ccv_array_free(z.positives);
	ccv_array_free(z.negatives);
	gsl_rng_free(rng);
	ccv_function_state_finish();
	return z.classifier;
#else
	assert(0 && "ccv_icf_classifier_cascade_new requires GSL library support");
	return 0;
#endif
}

void ccv_icf_classifier_cascade_soft(ccv_icf_classifier_cascade_t* cascade, ccv_array_t* posfiles, double acceptance)
{
#ifdef HAVE_GSL
	printf("with %d positive examples\n"
		   "going to accept %.2lf%% positive examples\n",
		   posfiles->rnum, acceptance * 100);
	ccv_size_t size = ccv_size(cascade->size.width - cascade->margin.left - cascade->margin.right, cascade->size.height - cascade->margin.top - cascade->margin.bottom);
	printf("use color? %s\n", cascade->grayscale ? "no" : "yes");
	printf("compute soft cascading thresholds for ICF classifier cascade at size %dx%d with margin (%d,%d,%d,%d)\n"
		   "------------------------\n",
		   size.width, size.height, cascade->margin.left, cascade->margin.top, cascade->margin.right, cascade->margin.bottom);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	/* collect positives */
	double weigh[2] = {
		0, 0
	};
	int i;
	for (i = 0; i < cascade->count; i++)
	{
		ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + i;
		weigh[0] += weak_classifier->weigh[0];
		weigh[1] += weak_classifier->weigh[1];
	}
	weigh[0] = 1 / fabs(weigh[0]), weigh[1] = 1 / fabs(weigh[1]);
	for (i = 0; i < cascade->count; i++)
	{
		ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + i;
		weak_classifier->weigh[0] = weak_classifier->weigh[0] * weigh[0];
		weak_classifier->weigh[1] = weak_classifier->weigh[1] * weigh[1];
	}
	ccv_array_t* validates = _ccv_icf_collect_validates(rng, size, cascade->margin, posfiles, cascade->grayscale);
	/* compute soft cascading thresholds */
	_ccv_icf_classifier_cascade_soft_with_validates(validates, cascade, acceptance);
	ccv_array_free(validates);
	gsl_rng_free(rng);
#else
	assert(0 && "ccv_icf_classifier_cascade_soft requires GSL library support");
#endif
}

static void _ccv_icf_read_classifier_cascade_with_fd(FILE* r, ccv_icf_classifier_cascade_t* cascade)
{
	cascade->type = CCV_ICF_CLASSIFIER_TYPE_A;
	fscanf(r, "%d %d %d %d", &cascade->count, &cascade->size.width, &cascade->size.height, &cascade->grayscale);
	fscanf(r, "%d %d %d %d", &cascade->margin.left, &cascade->margin.top, &cascade->margin.right, &cascade->margin.bottom);
	cascade->weak_classifiers = (ccv_icf_decision_tree_t*)ccmalloc(sizeof(ccv_icf_decision_tree_t) * cascade->count);
	int i, q;
	for (i = 0; i < cascade->count; i++)
	{
		ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + i;
		fscanf(r, "%u %a %a %a", &weak_classifier->pass, &weak_classifier->weigh[0], &weak_classifier->weigh[1], &weak_classifier->threshold);
		fscanf(r, "%d %a", &weak_classifier->features[0].count, &weak_classifier->features[0].beta);
		for (q = 0; q < weak_classifier->features[0].count; q++)
			fscanf(r, "%d %a %d %d %d %d", &weak_classifier->features[0].channel[q], &weak_classifier->features[0].alpha[q], &weak_classifier->features[0].sat[q * 2].x, &weak_classifier->features[0].sat[q * 2].y, &weak_classifier->features[0].sat[q * 2 + 1].x, &weak_classifier->features[0].sat[q * 2 + 1].y);
		if (weak_classifier->pass & 0x2)
		{
			fscanf(r, "%d %a", &weak_classifier->features[1].count, &weak_classifier->features[1].beta);
			for (q = 0; q < weak_classifier->features[1].count; q++)
				fscanf(r, "%d %a %d %d %d %d", &weak_classifier->features[1].channel[q], &weak_classifier->features[1].alpha[q], &weak_classifier->features[1].sat[q * 2].x, &weak_classifier->features[1].sat[q * 2].y, &weak_classifier->features[1].sat[q * 2 + 1].x, &weak_classifier->features[1].sat[q * 2 + 1].y);
		}
		if (weak_classifier->pass & 0x1)
		{
			fscanf(r, "%d %a", &weak_classifier->features[2].count, &weak_classifier->features[2].beta);
			for (q = 0; q < weak_classifier->features[2].count; q++)
				fscanf(r, "%d %a %d %d %d %d", &weak_classifier->features[2].channel[q], &weak_classifier->features[2].alpha[q], &weak_classifier->features[2].sat[q * 2].x, &weak_classifier->features[2].sat[q * 2].y, &weak_classifier->features[2].sat[q * 2 + 1].x, &weak_classifier->features[2].sat[q * 2 + 1].y);
		}
	}
}

static void _ccv_icf_write_classifier_cascade_with_fd(ccv_icf_classifier_cascade_t* cascade, FILE* w)
{
	int i, q;
	fprintf(w, "%d %d %d %d\n", cascade->count, cascade->size.width, cascade->size.height, cascade->grayscale);
	fprintf(w, "%d %d %d %d\n", cascade->margin.left, cascade->margin.top, cascade->margin.right, cascade->margin.bottom);
	for (i = 0; i < cascade->count; i++)
	{
		ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + i;
		fprintf(w, "%u %a %a %a\n", weak_classifier->pass, weak_classifier->weigh[0], weak_classifier->weigh[1], weak_classifier->threshold);
		fprintf(w, "%d %a\n", weak_classifier->features[0].count, weak_classifier->features[0].beta);
		for (q = 0; q < weak_classifier->features[0].count; q++)
			fprintf(w, "%d %a\n%d %d %d %d\n", weak_classifier->features[0].channel[q], weak_classifier->features[0].alpha[q], weak_classifier->features[0].sat[q * 2].x, weak_classifier->features[0].sat[q * 2].y, weak_classifier->features[0].sat[q * 2 + 1].x, weak_classifier->features[0].sat[q * 2 + 1].y);
		if (weak_classifier->pass & 0x2)
		{
			fprintf(w, "%d %a\n", weak_classifier->features[1].count, weak_classifier->features[1].beta);
			for (q = 0; q < weak_classifier->features[1].count; q++)
				fprintf(w, "%d %a\n%d %d %d %d\n", weak_classifier->features[1].channel[q], weak_classifier->features[1].alpha[q], weak_classifier->features[1].sat[q * 2].x, weak_classifier->features[1].sat[q * 2].y, weak_classifier->features[1].sat[q * 2 + 1].x, weak_classifier->features[1].sat[q * 2 + 1].y);
		}
		if (weak_classifier->pass & 0x1)
		{
			fprintf(w, "%d %a\n", weak_classifier->features[2].count, weak_classifier->features[2].beta);
			for (q = 0; q < weak_classifier->features[2].count; q++)
				fprintf(w, "%d %a\n%d %d %d %d\n", weak_classifier->features[2].channel[q], weak_classifier->features[2].alpha[q], weak_classifier->features[2].sat[q * 2].x, weak_classifier->features[2].sat[q * 2].y, weak_classifier->features[2].sat[q * 2 + 1].x, weak_classifier->features[2].sat[q * 2 + 1].y);
		}
	}
}

ccv_icf_classifier_cascade_t* ccv_icf_read_classifier_cascade(const char* filename)
{
	FILE* r = fopen(filename, "r");
	ccv_icf_classifier_cascade_t* cascade = 0;
	if (r)
	{
		cascade = (ccv_icf_classifier_cascade_t*)ccmalloc(sizeof(ccv_icf_classifier_cascade_t));
		_ccv_icf_read_classifier_cascade_with_fd(r, cascade);
		fclose(r);
	}
	return cascade;
}

void ccv_icf_write_classifier_cascade(ccv_icf_classifier_cascade_t* cascade, const char* filename)
{
	FILE* w = fopen(filename, "w+");
	if (w)
	{
		_ccv_icf_write_classifier_cascade_with_fd(cascade, w);
		fclose(w);
	}
}

void ccv_icf_classifier_cascade_free(ccv_icf_classifier_cascade_t* classifier)
{
	ccfree(classifier->weak_classifiers);
	ccfree(classifier);
}

ccv_icf_multiscale_classifier_cascade_t* ccv_icf_read_multiscale_classifier_cascade(const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/multiscale", directory);
	FILE* r = fopen(filename, "r");
	if (r)
	{
		int octave = 0, count = 0, grayscale = 0;
		fscanf(r, "%d %d %d", &octave, &count, &grayscale);
		fclose(r);
		ccv_icf_multiscale_classifier_cascade_t* classifier = (ccv_icf_multiscale_classifier_cascade_t*)ccmalloc(sizeof(ccv_icf_multiscale_classifier_cascade_t) + sizeof(ccv_icf_classifier_cascade_t) * count);
		classifier->type = CCV_ICF_CLASSIFIER_TYPE_B;
		classifier->octave = octave;
		classifier->count = count;
		classifier->grayscale = grayscale;
		classifier->cascade = (ccv_icf_classifier_cascade_t*)(classifier + 1);
		int i;
		for (i = 0; i < count; i++)
		{
			snprintf(filename, 1024, "%s/cascade-%d", directory, i + 1);
			r = fopen(filename, "r");
			if (r)
			{
				ccv_icf_classifier_cascade_t* cascade = classifier->cascade + i;
				_ccv_icf_read_classifier_cascade_with_fd(r, cascade);
				fclose(r);
			}
		}
		return classifier;
	}
	return 0;
}

void ccv_icf_write_multiscale_classifier_cascade(ccv_icf_multiscale_classifier_cascade_t* classifier, const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/multiscale", directory);
	FILE* w = fopen(filename, "w+");
	fprintf(w, "%d %d %d\n", classifier->octave, classifier->count, classifier->grayscale);
	fclose(w);
	int i;
	for (i = 0; i < classifier->count; i++)
	{
		snprintf(filename, 1024, "%s/cascade-%d", directory, i + 1);
		w = fopen(filename, "w+");
		_ccv_icf_write_classifier_cascade_with_fd(classifier->cascade + i, w);
		fclose(w);
	}
}

void ccv_icf_multiscale_classifier_cascade_free(ccv_icf_multiscale_classifier_cascade_t* classifier)
{
	int i;
	for (i = 0; i < classifier->count; i++)
		ccfree(classifier->cascade[i].weak_classifiers);
	ccfree(classifier);
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

static void _ccv_icf_detect_objects_with_classifier_cascade(ccv_dense_matrix_t* a, ccv_icf_classifier_cascade_t** cascades, int count, ccv_icf_param_t params, ccv_array_t* seq[])
{
	int i, j, k, q, x, y;
	int scale_upto = 1;
	for (i = 0; i < count; i++)
		scale_upto = ccv_max(scale_upto, (int)(log(ccv_min((double)a->rows / (cascades[i]->size.height - cascades[i]->margin.top - cascades[i]->margin.bottom), (double)a->cols / (cascades[i]->size.width - cascades[i]->margin.left - cascades[i]->margin.right))) / log(2.) - DBL_MIN) + 1);
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * scale_upto);
	pyr[0] = a;
	for (i = 1; i < scale_upto; i++)
	{
		pyr[i] = 0;
		ccv_sample_down(pyr[i - 1], &pyr[i], 0, 0, 0);
	}
	for (i = 0; i < scale_upto; i++)
	{
		// run it
		for (j = 0; j < count; j++)
		{
			double scale_ratio = pow(2., 1. / (params.interval + 1));
			double scale = 1;
			ccv_icf_classifier_cascade_t* cascade = cascades[j];
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
				rows = bordered->rows;
				cols = bordered->cols;
				ccv_dense_matrix_t* icf = 0;
				ccv_icf(bordered, &icf, 0);
				ccv_matrix_free(bordered);
				ccv_dense_matrix_t* sat = 0;
				ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
				ccv_matrix_free(icf);
				int ch = CCV_GET_CHANNEL(sat->type);
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
						for (q = 0; q < cascade->count; q++)
						{
							ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + q;
							int c = _ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, x, 0);
							sum += weak_classifier->weigh[c];
							if (sum < weak_classifier->threshold)
							{
								pass = 0;
								break;
							}
						}
						if (pass)
						{
							ccv_comp_t comp;
							comp.rect = ccv_rect((int)((x + 0.5) * scale * (1 << i) - 0.5), (int)((y + 0.5) * scale * (1 << i) - 0.5), (cascade->size.width - cascade->margin.left - cascade->margin.right) * scale * (1 << i), (cascade->size.height - cascade->margin.top - cascade->margin.bottom) * scale * (1 << i));
							comp.neighbors = 1;
							comp.classification.id = j + 1;
							comp.classification.confidence = sum;
							ccv_array_push(seq[j], &comp);
						}
					}
					ptr += sat->cols * ch * params.step_through;
				}
				ccv_matrix_free(sat);
				scale *= scale_ratio;
			}
		}
	}

	for (i = 1; i < scale_upto; i++)
		ccv_matrix_free(pyr[i]);
}

static void _ccv_icf_detect_objects_with_multiscale_classifier_cascade(ccv_dense_matrix_t* a, ccv_icf_multiscale_classifier_cascade_t** multiscale_cascade, int count, ccv_icf_param_t params, ccv_array_t* seq[])
{
	int i, j, k, q, x, y, ix, iy, py;
	assert(multiscale_cascade[0]->count % multiscale_cascade[0]->octave == 0);
	ccv_margin_t margin = multiscale_cascade[0]->cascade[multiscale_cascade[0]->count - 1].margin;
	for (i = 1; i < count; i++)
	{
		assert(multiscale_cascade[i]->count % multiscale_cascade[i]->octave == 0);
		assert(multiscale_cascade[i - 1]->grayscale == multiscale_cascade[i]->grayscale);
		assert(multiscale_cascade[i - 1]->count == multiscale_cascade[i]->count);
		assert(multiscale_cascade[i - 1]->octave == multiscale_cascade[i]->octave);
		ccv_icf_classifier_cascade_t* cascade = multiscale_cascade[i]->cascade + multiscale_cascade[i]->count - 1;
		margin.top = ccv_max(margin.top, cascade->margin.top);
		margin.right = ccv_max(margin.right, cascade->margin.right);
		margin.bottom = ccv_max(margin.bottom, cascade->margin.bottom);
		margin.left = ccv_max(margin.left, cascade->margin.left);
	}
	int scale_upto = 1;
	for (i = 0; i < count; i++)
		scale_upto = ccv_max(scale_upto, (int)(log(ccv_min((double)a->rows / (multiscale_cascade[i]->cascade[0].size.height - multiscale_cascade[i]->cascade[0].margin.top - multiscale_cascade[i]->cascade[0].margin.bottom), (double)a->cols / (multiscale_cascade[i]->cascade[0].size.width - multiscale_cascade[i]->cascade[0].margin.left - multiscale_cascade[i]->cascade[0].margin.right))) / log(2.) - DBL_MIN) + 2 - multiscale_cascade[i]->octave);
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * scale_upto);
	pyr[0] = a;
	for (i = 1; i < scale_upto; i++)
	{
		pyr[i] = 0;
		ccv_sample_down(pyr[i - 1], &pyr[i], 0, 0, 0);
	}
	for (i = 0; i < scale_upto; i++)
	{
		ccv_dense_matrix_t* bordered = 0;
		ccv_border(pyr[i], (ccv_matrix_t**)&bordered, 0, margin);
		ccv_dense_matrix_t* icf = 0;
		ccv_icf(bordered, &icf, 0);
		ccv_matrix_free(bordered);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(icf);
		int ch = CCV_GET_CHANNEL(sat->type);
		assert(CCV_GET_DATA_TYPE(sat->type) == CCV_32F);
		// run it
		for (j = 0; j < count; j++)
		{
			double scale_ratio = pow(2., (double)multiscale_cascade[j]->octave / multiscale_cascade[j]->count);
			int starter = i > 0 ? multiscale_cascade[j]->count - (multiscale_cascade[j]->count / multiscale_cascade[j]->octave) : 0;
			double scale = pow(scale_ratio, starter);
			for (k = starter; k < multiscale_cascade[j]->count; k++)
			{
				ccv_icf_classifier_cascade_t* cascade = multiscale_cascade[j]->cascade + k;
				int rows = (int)(pyr[i]->rows / scale + cascade->margin.top + 0.5);
				int cols = (int)(pyr[i]->cols / scale + cascade->margin.left + 0.5);
				int top = margin.top - cascade->margin.top;
				int right = margin.right - cascade->margin.right;
				int bottom = margin.bottom - cascade->margin.bottom;
				int left = margin.left - cascade->margin.left;
				if (sat->rows - top - bottom <= cascade->size.height || sat->cols - left - right <= cascade->size.width)
					break;
				float* ptr = sat->data.f32 + top * sat->cols * ch;
				for (y = 0, iy = py = top; y < rows; y += params.step_through)
				{
					iy = (int)((y + 0.5) * scale + top);
					if (iy >= sat->rows - cascade->size.height - 1)
						break;
					if (iy > py)
					{
						ptr += sat->cols * ch * (iy - py);
						py = iy;
					}
					for (x = 0; x < cols; x += params.step_through)
					{
						ix = (int)((x + 0.5) * scale + left);
						if (ix >= sat->cols - cascade->size.width - 1)
							break;
						int pass = 1;
						float sum = 0;
						for (q = 0; q < cascade->count; q++)
						{
							ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + q;
							int c = _ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, ix, 0);
							sum += weak_classifier->weigh[c];
							if (sum < weak_classifier->threshold)
							{
								pass = 0;
								break;
							}
						}
						if (pass)
						{
							ccv_comp_t comp;
							comp.rect = ccv_rect((int)((x + 0.5) * scale * (1 << i)), (int)((y + 0.5) * scale * (1 << i)), (cascade->size.width - cascade->margin.left - cascade->margin.right) << i, (cascade->size.height - cascade->margin.top - cascade->margin.bottom) << i);
							comp.neighbors = 1;
							comp.classification.id = j + 1;
							comp.classification.confidence = sum;
							ccv_array_push(seq[j], &comp);
						}
					}
				}
				scale *= scale_ratio;
			}
		}
		ccv_matrix_free(sat);
	}

	for (i = 1; i < scale_upto; i++)
		ccv_matrix_free(pyr[i]);
}

ccv_array_t* ccv_icf_detect_objects(ccv_dense_matrix_t* a, void* cascade, int count, ccv_icf_param_t params)
{
	assert(count > 0);
	int i, j, k;
	int type = *(((int**)cascade)[0]);
	for (i = 1; i < count; i++)
	{
		// check all types to be the same
		assert(*(((int**)cascade)[i]) == type);
	}
	ccv_array_t** seq = (ccv_array_t**)alloca(sizeof(ccv_array_t*) * count);
	for (i = 0; i < count; i++)
		seq[i] = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	switch (type)
	{
		case CCV_ICF_CLASSIFIER_TYPE_A:
			_ccv_icf_detect_objects_with_classifier_cascade(a, (ccv_icf_classifier_cascade_t**)cascade, count, params, seq);
			break;
		case CCV_ICF_CLASSIFIER_TYPE_B:
			_ccv_icf_detect_objects_with_multiscale_classifier_cascade(a, (ccv_icf_multiscale_classifier_cascade_t**)cascade, count, params, seq);
			break;
	}
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
			ccv_comp_t* comps = (ccv_comp_t*)cccalloc(sizeof(ccv_comp_t), ncomp + 1);

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
