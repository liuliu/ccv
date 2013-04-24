#include "ccv.h"
#include "ccv_internal.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const ccv_icf_param_t ccv_icf_default_params = {
	.min_neighbors = 2,
	.threshold = 0,
	.flags = 0,
};

// generating the integrate channels features (which combines the grayscale, gradient magnitude, and 6-direction HOG
void ccv_icf(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_literal("ccv_icf"), a->sig, CCV_EOF_SIGN);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_32F | 8, CCV_32F | 8, sig);
	ccv_object_return_if_cached(, db);
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.f32;
	float* mgp = mg->data.f32;
	float* dbp = db->data.f32;
	ccv_zero(db);
	int i, j;
	unsigned char* a_ptr = a->data.u8;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
		{ \
			dbp[0] = _for_get(a_ptr, j, 0); \
			dbp[1] = mgp[j]; \
			float agr = (ccv_clamp(agp[j], 0, 359.99) / 360.0) * 6; \
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
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
}

static void _ccv_icf_randomize_feature(gsl_rng* rng, ccv_size_t size, ccv_icf_feature_t* feature)
{
	feature->count = gsl_rng_uniform_int(rng, CCV_ICF_SAT_MAX - 2) + 2;
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
		} while (!(abs(x1 - x0) > 0 && abs(y1 - y0) > 0 && abs(x1 - x0) < size.width / 2 && abs(y1 - y0) < size.height / 2));
		feature->sat[i * 2].x = ccv_min(x0, x1);
		feature->sat[i * 2].y = ccv_min(y0, y1);
		feature->sat[i * 2 + 1].x = ccv_max(x0, x1);
		feature->sat[i * 2 + 1].y = ccv_max(y0, y1);
		feature->channel[i] = gsl_rng_uniform_int(rng, 7); // 8-channels
		feature->alpha[i] = gsl_rng_uniform(rng) / (float)((feature->sat[i * 2 + 1].x - feature->sat[i * 2].x + 1) * (feature->sat[i * 2 + 1].y - feature->sat[i * 2].y + 1));
	}
}

static void _ccv_icf_check_params(ccv_icf_new_param_t params)
{
	assert(params.interval >= 0);
	assert(params.size.width > 0 && params.size.height > 0);
	assert(params.deform_shift > 0);
	assert(params.deform_angle > 0);
	assert(params.deform_scale > 0);
	assert(params.feature_size > 0);
	assert(params.weight_trimming > 0.5 && params.weight_trimming <= 1.0);
	assert(params.sample_rate > 0 && params.sample_rate <= 1.0);
	assert(params.acceptance > 0 && params.acceptance < 1.0);
}

static ccv_dense_matrix_t* _ccv_icf_capture_feature(gsl_rng* rng, ccv_dense_matrix_t* image, ccv_decimal_pose_t pose, ccv_size_t size, float deform_angle, float deform_scale, float deform_shift)
{
	float rotate_x = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180;
	float rotate_y = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180;
	float rotate_z = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.roll;
	float scale = 1 + deform_scale  - deform_scale * 2 * gsl_rng_uniform(rng);
	float m00 = cosf(rotate_z) * scale;
	float m01 = cosf(rotate_y) * sinf(rotate_z);
	float m02 = (deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) * pose.a + pose.x - image->cols * 0.5;
	float m10 = sinf(rotate_y) * cosf(rotate_z) - cosf(rotate_x) * sinf(rotate_z);
	float m11 = (sinf(rotate_y) * sinf(rotate_z) + cosf(rotate_x) * cosf(rotate_z)) * scale;
	float m12 = (deform_shift * gsl_rng_uniform(rng) - deform_shift) * pose.b + pose.y - image->rows * 0.5;
	float m20 = sinf(rotate_y) * cosf(rotate_z) + sinf(rotate_x) * sinf(rotate_z);
	float m21 = sinf(rotate_y) * sinf(rotate_z) - sinf(rotate_x) * cosf(rotate_z);
	float m22 = cosf(rotate_x) * cosf(rotate_y);
	ccv_dense_matrix_t* b = 0;
	ccv_perspective_transform(image, &b, 0, m00, m01, m02, m10, m11, m12, m20, m21, m22);
	ccv_dense_matrix_t* resize = 0;
	// have 1px border around the grayscale image because we need these to compute correct gradient feature
	float scale_ratio = sqrtf((float)((size.width + 2) * (size.height + 2)) / (pose.a * pose.b * 4));
	ccv_size_t scale_size = {
		.width = (int)((size.width + 2) / scale_ratio + 0.5),
		.height = (int)((size.height + 2) / scale_ratio + 0.5),
	};
	assert(scale_size.width > 0 && scale_size.height > 0);
	ccv_decimal_slice(b, &resize, 0, b->rows * 0.5 - (size.height + 2) / scale_ratio * 0.5, b->cols * 0.5 - (size.width + 2) / scale_ratio * 0.5, scale_size.height, scale_size.width);
	ccv_matrix_free(b);
	b = 0;
	if (scale_ratio > 1)
		ccv_resample(resize, &b, 0, size.height + 2, size.width + 2, CCV_INTER_CUBIC);
	else
		ccv_resample(resize, &b, 0, size.height + 2, size.width + 2, CCV_INTER_AREA);
	ccv_matrix_free(resize);
	return b;
}

#ifdef HAVE_LIBLINEAR
#include <linear.h>
#endif

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
} ccv_icf_classifier_cascade_persistence_state_t;

typedef struct {
	uint32_t index;
	float value;
} ccv_icf_value_index_t;

typedef struct {
	ccv_icf_value_index_t* sort;
	float* lookup;
} ccv_icf_precomputed_collection_t;

typedef struct {
	ccv_function_state_reserve_field;
	int i, j;
	ccv_icf_new_param_t params;
	ccv_icf_multiscale_classifier_cascade_t* classifier;
	ccv_array_t* positives;
	ccv_array_t* negatives;
	ccv_icf_feature_t* features;
	ccv_size_t size;
	double scale;
	ccv_icf_example_state_t* example_state;
	ccv_icf_precomputed_collection_t precomputed;
	ccv_icf_classifier_cascade_persistence_state_t x;
} ccv_icf_classifier_cascade_state_t;

static void _ccv_icf_write_classifier_cascade_state(ccv_icf_classifier_cascade_state_t* state, const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/state", directory);
	FILE* w = fopen(filename, "w+");
	fprintf(w, "%d %d %d\n", state->line_no, state->i, state->j);
	fprintf(w, "%d %d %d %la\n", state->params.feature_size, state->size.width, state->size.height, state->scale);
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
			assert(a->rows == state->size.height + 2 && a->cols == state->size.width + 2);
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
			assert(a->rows == state->size.height + 2 && a->cols == state->size.width + 2);
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
	if (!state->x.classifier)
	{
		ccv_icf_write_classifier_cascade(state->classifier, directory);
		state->x.classifier = 1;
	}
}

static void _ccv_icf_read_classifier_cascade_state(const char* directory, ccv_icf_classifier_cascade_state_t* state)
{
	char filename[1024];
	state->line_no = state->i = state->j = 0;
	snprintf(filename, 1024, "%s/state", directory);
	FILE* r = fopen(filename, "r");
	if (r)
	{
		int feature_size;
		fscanf(r, "%d %d %d", &state->line_no, &state->i, &state->j);
		fscanf(r, "%d %d %d %la", &feature_size, &state->size.width, &state->size.height, &state->scale);
		assert(feature_size == state->params.feature_size);
		fclose(r);
	}
	int i, q;
	snprintf(filename, 1024, "%s/positives", directory);
	r = fopen(filename, "rb");
	state->x.features = state->x.example_state = state->x.classifier = state->x.positives = state->x.negatives = 1;
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
			assert(a->rows == state->size.height + 2 && a->cols == state->size.width + 2);
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
			assert(a->rows == state->size.height + 2 && a->cols == state->size.width + 2);
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
	}
	state->classifier = ccv_icf_read_classifier_cascade(directory);
	if (!state->classifier)
	{
		state->classifier = (ccv_icf_multiscale_classifier_cascade_t*)ccmalloc(sizeof(ccv_icf_multiscale_classifier_cascade_t) + sizeof(ccv_icf_classifier_cascade_t) * (state->params.interval + 1));
		state->classifier->interval = 0;
		state->classifier->cascade = (ccv_icf_classifier_cascade_t*)(state->classifier + 1);
		state->scale = 1;
	} else {
		// we need to realloc it to desired size
		state->classifier = ccrealloc(state->classifier, sizeof(ccv_icf_multiscale_classifier_cascade_t) + sizeof(ccv_icf_classifier_cascade_t) * (state->params.interval + 1));
		state->classifier->cascade = (ccv_icf_classifier_cascade_t*)(state->classifier + 1);
	}
}

#define less_than(s1, s2, aux) ((s1).value < (s2).value)
static CCV_IMPLEMENT_QSORT(_ccv_icf_precomputed_ordering, ccv_icf_value_index_t, less_than)
#undef less_than

static inline float _ccv_icf_run_feature(ccv_icf_feature_t* feature, float* ptr, int cols, int ch, int x, int y)
{
	float c = feature->beta;
	int q;
	for (q = 0; q < feature->count; q++)
		c += (ptr[(feature->sat[q * 2 + 1].x + x + 1 + (feature->sat[q * 2 + 1].y + y + 1) * cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2].x + x + (feature->sat[q * 2 + 1].y + y + 1) * cols) * ch + feature->channel[q]] + ptr[(feature->sat[q * 2].x + x + (feature->sat[q * 2].y + y) * cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2 + 1].x + x + 1 + (feature->sat[q * 2].y + y) * cols) * ch + feature->channel[q]]) * feature->alpha[q];
	return c;
}

static ccv_icf_precomputed_collection_t _ccv_icf_precompute_features(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives)
{
	int i, j;
	ccv_icf_precomputed_collection_t precomputed;
	precomputed.sort = (ccv_icf_value_index_t*)ccmalloc(sizeof(ccv_icf_value_index_t) * feature_size * (positives->rnum + negatives->rnum) + sizeof(float) * feature_size * (positives->rnum + negatives->rnum));
	precomputed.lookup = (float*)(precomputed.sort + feature_size * (positives->rnum + negatives->rnum));
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
		if (i % 97 == 0 || i == positives->rnum + negatives->rnum - 1) // don't flush too fast
			FLUSH(" - precompute %d features through %d%% (%d / %d) examples", feature_size, (i + 1) * 100 / (positives->rnum + negatives->rnum), i + 1, positives->rnum + negatives->rnum);
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
			float c = _ccv_icf_run_feature(features + j, ptr, sat->cols, ch, 1, 1);
			precomputed.lookup[j * (positives->rnum + negatives->rnum) + i] = c;
			precomputed.sort[j * (positives->rnum + negatives->rnum) + i].value = c;
			precomputed.sort[j * (positives->rnum + negatives->rnum) + i].index = i;
		}
		ccv_matrix_free(sat);
	}
	for (i = 0; i < feature_size; i++)
		_ccv_icf_precomputed_ordering(precomputed.sort + i * (positives->rnum + negatives->rnum),  positives->rnum + negatives->rnum, 0);
	printf("\n - features are precomputed on examples\n");
	return precomputed;
}

typedef struct {
	uint32_t pass;
	double weigh[4];
	int sign;
	float threshold;
	int first_feature;
} ccv_icf_decision_tree_cache_t;

static ccv_icf_decision_tree_cache_t _ccv_icf_find_first_feature(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_precomputed_collection_t precomputed, ccv_icf_example_state_t* example_state, ccv_icf_feature_t* feature)
{
	assert(feature != 0);
	ccv_icf_decision_tree_cache_t intermediate_cache;
	int i, j;
	double aweigh[2] = {0};
	for (i = 0; i < positives->rnum; i++)
		aweigh[1] += example_state[i].weight;
	for (i = positives->rnum; i < positives->rnum + negatives->rnum; i++)
		aweigh[0] += example_state[i].weight;
	ccv_icf_value_index_t* computed = precomputed.sort;
	double best_error_rate = 1.0;
	double best_weigh[2] = {0};
	int best_count[2] = {0};
	int feature_index = 0, best_error_index = -1;
	for (i = 0; i < feature_size; i++)
	{
		double weigh[2] = {0};
		double min_weigh[2] = {0};
		int count[2] = {0};
		int min_count[2] = {0};
		double min_error_rate = 1.0;
		int min_error_index = 0;
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
		{
			weigh[computed[j].index < positives->rnum] += example_state[computed[j].index].weight;
			assert(weigh[0] <= aweigh[0] && weigh[1] <= aweigh[1]);
			++count[computed[j].index < positives->rnum];
			double error_rate = ccv_min(weigh[0] + aweigh[1] - weigh[1], weigh[1] + aweigh[0] - weigh[0]);
			assert(error_rate > 0);
			if (error_rate < min_error_rate)
			{
				min_error_index = j;
				min_error_rate = error_rate;
				min_weigh[0] = weigh[0];
				min_weigh[1] = weigh[1];
				min_count[0] = count[0];
				min_count[1] = count[1];
			}
		}
		if (min_error_rate < best_error_rate)
		{
			best_error_rate = min_error_rate;
			best_error_index = min_error_index;
			best_weigh[0] = min_weigh[0];
			best_weigh[1] = min_weigh[1];
			best_count[0] = min_count[0];
			best_count[1] = min_count[1];
			feature_index = i;
		}
		computed += positives->rnum + negatives->rnum;
	}
	*feature = features[feature_index];
	computed = precomputed.sort + feature_index * (positives->rnum + negatives->rnum);
	assert(best_error_index < positives->rnum + negatives->rnum - 1 && best_error_index >= 0);
	intermediate_cache.threshold = (computed[best_error_index].value + computed[best_error_index + 1].value) * 0.5;
	if (best_weigh[0] + aweigh[1] - best_weigh[1] < best_weigh[1] + aweigh[0] - best_weigh[0])
	{
		// revert the sign of alpha
		for (i = 0; i < feature->count; i++)
			feature->alpha[i] = -feature->alpha[i];
		feature->beta = intermediate_cache.threshold;
		intermediate_cache.sign = -1;
		intermediate_cache.weigh[0] = aweigh[0] - best_weigh[0];
		intermediate_cache.weigh[1] = aweigh[1] - best_weigh[1];
		intermediate_cache.weigh[2] = best_weigh[0];
		intermediate_cache.weigh[3] = best_weigh[1];
		intermediate_cache.pass = 3;
		if (best_count[0] == 0)
			intermediate_cache.pass &= 2; // only positive examples in the right, no need to build right leaf
		if (best_count[1] == positives->rnum)
			intermediate_cache.pass &= 1; // no positive examples in the left, no need to build left leaf
	} else {
		feature->beta = -intermediate_cache.threshold;
		intermediate_cache.sign = 1;
		intermediate_cache.weigh[0] = best_weigh[0];
		intermediate_cache.weigh[1] = best_weigh[1];
		intermediate_cache.weigh[2] = aweigh[0] - best_weigh[0];
		intermediate_cache.weigh[3] = aweigh[1] - best_weigh[1];
		intermediate_cache.pass = 3;
		if (best_count[0] == negatives->rnum)
			intermediate_cache.pass &= 2; // only positive examples in the right, no need to build right leaf
		if (best_count[1] == 0)
			intermediate_cache.pass &= 1; // no positive examples in the left, no need to build left leaf
	}
	intermediate_cache.first_feature = feature_index;
	return intermediate_cache;
}

static double _ccv_icf_find_second_feature(ccv_icf_decision_tree_cache_t intermediate_cache, int leaf, ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_precomputed_collection_t precomputed, ccv_icf_example_state_t* example_state, ccv_icf_feature_t* feature)
{
	int i, j;
	ccv_icf_value_index_t* computed = precomputed.sort;
	float* lookup = precomputed.lookup + intermediate_cache.first_feature * (positives->rnum + negatives->rnum);
	double* aweigh = intermediate_cache.weigh + leaf * 2;
	int sign = intermediate_cache.sign * (leaf * 2 - 1);
	double best_error_rate = 1.0;
	double best_weigh[2] = {0};
	int feature_index = 0, best_error_index = -1;
	for (i = 0; i < feature_size; i++)
	{
		double weigh[2] = {0};
		double min_weigh[2] = {0};
		double min_error_rate = 1.0;
		int min_error_index = 0;
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
		{
			// only care about part of the data
			if (lookup[computed[j].index] * sign > sign * intermediate_cache.threshold)
			{
				weigh[computed[j].index < positives->rnum] += example_state[computed[j].index].weight;
				assert(weigh[0] <= aweigh[0] && weigh[1] <= aweigh[1]);
				double error_rate = ccv_min(weigh[0] + aweigh[1] - weigh[1], weigh[1] + aweigh[0] - weigh[0]);
				if (error_rate < min_error_rate)
				{
					min_error_index = j;
					min_error_rate = error_rate;
					min_weigh[0] = weigh[0];
					min_weigh[1] = weigh[1];
				}
			}
		}
		if (min_error_rate < best_error_rate)
		{
			best_error_rate = min_error_rate;
			best_error_index = min_error_index;
			best_weigh[0] = min_weigh[0];
			best_weigh[1] = min_weigh[1];
			feature_index = i;
		}
		computed += positives->rnum + negatives->rnum;
	}
	*feature = features[feature_index];
	computed = precomputed.sort + feature_index * (positives->rnum + negatives->rnum);
	assert(best_error_index < positives->rnum + negatives->rnum - 1 && best_error_index >= 0);
	if (best_weigh[0] + aweigh[1] - best_weigh[1] < best_weigh[1] + aweigh[0] - best_weigh[0])
	{
		// revert the sign of alpha
		for (i = 0; i < feature->count; i++)
			feature->alpha[i] = -feature->alpha[i];
		feature->beta = (computed[best_error_index].value + computed[best_error_index + 1].value) * 0.5;
		return best_weigh[1] + aweigh[0] - best_weigh[0];
	} else {
		feature->beta = -(computed[best_error_index].value + computed[best_error_index + 1].value) * 0.5;
		return best_weigh[0] + aweigh[1] - best_weigh[1];
	}
}

static double _ccv_icf_find_best_weak_classifier(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_precomputed_collection_t precomputed, ccv_icf_example_state_t* example_state, ccv_icf_decision_tree_t* weak_classifier)
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
	return rate;
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

static double _ccv_icf_rate_weak_classifier(ccv_icf_decision_tree_t* weak_classifier, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_precomputed_collection_t precomputed, ccv_icf_example_state_t* example_state)
{
	int i;
	double rate = 0;
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
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
		if (i < positives->rnum)
		{
			if (_ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, 1, 1))
				rate += example_state[i].weight;
		} else {
			if (!_ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, 1, 1))
				rate += example_state[i].weight;
		}
		ccv_matrix_free(sat);
	}
	return rate;
}

static ccv_array_t* _ccv_icf_collect_positives(gsl_rng* rng, ccv_size_t size, ccv_array_t* posfiles, int posnum, float deform_angle, float deform_scale, float deform_shift)
{
	ccv_array_t* positives = ccv_array_new(ccv_compute_dense_matrix_size(size.height + 2, size.width + 2, CCV_8U | CCV_C1), posnum, 0);
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
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			if (image)
			{
				for (q = 0; q < ratio; q++)
					if (q < (int)ratio || gsl_rng_uniform(rng) <= ratio - (int)ratio)
					{
						FLUSH(" - collect positives %d%% (%d / %d)", (i + 1) * 100 / posnum, i + 1, posnum);
						ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, file_info->pose, size, deform_angle, deform_scale, deform_shift);
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
	}
	printf("\n");
	return positives;
}

static ccv_array_t* _ccv_icf_collect_negatives(gsl_rng* rng, ccv_size_t size, ccv_array_t* bgfiles, int negnum, float deform_angle, float deform_scale, float deform_shift)
{
	ccv_array_t* negatives = ccv_array_new(ccv_compute_dense_matrix_size(size.height + 2, size.width + 2, CCV_8U | CCV_C1), negnum, 0);
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
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			if (image)
			{
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
						ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, pose, size, deform_angle, deform_scale, deform_shift);
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
	}
	printf("\n");
	return negatives;
}

#define less_than(s1, s2, aux) ((s1) > (s2))
static CCV_IMPLEMENT_QSORT(_ccv_icf_threshold_rating, float, less_than)
#undef less_than

ccv_icf_multiscale_classifier_cascade_t* ccv_icf_classifier_cascade_new(ccv_array_t* posfiles, int posnum, ccv_array_t* bgfiles, int negnum, const char* dir, ccv_icf_new_param_t params)
{
	_ccv_icf_check_params(params);
	assert(posfiles->rnum > 0);
	assert(bgfiles->rnum > 0);
	assert(posnum > 0 && negnum > 0);
	int k, scale_upto = params.interval + 1;
	double scale_factor = pow(2., 1. / scale_upto);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	// we will keep all states inside this structure for easier save / resume across process
	// this should work better than ad-hoc one we used in DPM / BBF implementation
	ccv_icf_classifier_cascade_state_t z;
	z.params = params;
	ccv_function_state_begin(_ccv_icf_read_classifier_cascade_state, z, dir);
	for (z.i = 0; z.i <= params.interval; z.i++)
	{
		z.size = ccv_size((int)(params.size.width * z.scale + 0.5), (int)(params.size.height * z.scale + 0.5));
		z.classifier->cascade[z.i].size = z.size;
		z.classifier->cascade[z.i].thresholds = 0;
		printf(" - learn icf classifier cascade at size %dx%d\n", z.size.width, z.size.height);
		z.features = (ccv_icf_feature_t*)ccmalloc(sizeof(ccv_icf_feature_t) * params.feature_size);
		// generate random features
		for (z.j = 0; z.j < params.feature_size; z.j++)
			_ccv_icf_randomize_feature(rng, z.size, z.features + z.j);
		z.x.features = 0;
		ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		z.positives = _ccv_icf_collect_positives(rng, z.size, posfiles, posnum, params.deform_angle, params.deform_scale, params.deform_shift);
		z.x.positives = 0;
		ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		z.negatives = _ccv_icf_collect_negatives(rng, z.size, bgfiles, negnum, params.deform_angle, params.deform_scale, params.deform_shift);
		z.x.negatives = 0;
		ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		z.example_state = (ccv_icf_example_state_t*)ccmalloc(sizeof(ccv_icf_example_state_t) * (z.negatives->rnum + z.positives->rnum));
		for (z.j = 0; z.j < z.positives->rnum + z.negatives->rnum; z.j++)
		{
			z.example_state[z.j].weight = (z.j < z.positives->rnum) ? 0.5 / z.positives->rnum : 0.5 / z.negatives->rnum;
			z.example_state[z.j].rate = 0;
		}
		z.precomputed = _ccv_icf_precompute_features(z.features, params.feature_size, z.positives, z.negatives);
		z.classifier->cascade[z.i].weak_classifiers = (ccv_icf_decision_tree_t*)ccmalloc(sizeof(ccv_icf_decision_tree_t) * params.weak_classifier);
		for (z.j = 0; z.j < params.weak_classifier; z.j++)
		{
			z.classifier->cascade[z.i].count = z.j + 1;
			gsl_ran_shuffle(rng, z.example_state, z.positives->rnum + z.negatives->rnum, sizeof(ccv_icf_example_state_t));
			printf(" - boost weak classifier %d of %d\n", z.j + 1, params.weak_classifier);
			ccv_icf_decision_tree_t weak_classifier;
			double rate = _ccv_icf_find_best_weak_classifier(z.features, params.feature_size, z.positives, z.negatives, z.precomputed, z.example_state, &weak_classifier);
			double confirm_rate = _ccv_icf_rate_weak_classifier(&weak_classifier, z.positives, z.negatives, z.precomputed, z.example_state);
			printf(" - %lf %lf\n", rate, confirm_rate);
			assert(rate > 0.5); // it has to be better than random chance
			double alpha = sqrt((1 - rate) / rate);
			double beta = 1.0 / alpha;
			double c = log(rate / (1 - rate));
			weak_classifier.weigh[0] = -c;
			weak_classifier.weigh[1] = c;
			double reweigh = 0;
			for (k = 0; k < z.positives->rnum + z.negatives->rnum; k++)
			{
				z.example_state[k].weight *= (z.example_state[k].correct) ? alpha : beta;
				z.example_state[k].rate += weak_classifier.weigh[!((k < z.positives->rnum) ^ z.example_state[k].correct)];
				reweigh += z.example_state[k].weight;
			}
			reweigh = 1.0 / reweigh;
			printf(" - on all examples, best feature at rate %lf\n", rate);
			// balancing the weight to sum 1.0
			for (k = 0; k < z.positives->rnum + z.negatives->rnum; k++)
				z.example_state[k].weight *= reweigh;
			z.classifier->cascade[z.i].weak_classifiers[z.j] = weak_classifier;
			// compute the threshold at given acceptance
			float* positive_rate = (float*)ccmalloc(sizeof(float) * z.positives->rnum);
			for (k = 0; k < z.positives->rnum; k++)
				positive_rate[k] = z.example_state[k].rate;
			_ccv_icf_threshold_rating(positive_rate, z.positives->rnum, 0);
			float threshold = positive_rate[ccv_min((int)(params.acceptance * (z.positives->rnum + 0.5) - 0.5), z.positives->rnum - 1)];
			int true_positives = 0, false_positives = 0;
			for (k = 0; k < z.positives->rnum; k++)
				if (z.example_state[k].rate >= threshold)
					++true_positives;
			for (k = z.positives->rnum; k < z.positives->rnum + z.negatives->rnum; k++)
				if (z.example_state[k].rate >= threshold)
					++false_positives;
			printf(" - at threshold %f, true positive rate: %f%%, false positive rate: %f%% (%d)\n", threshold, (float)true_positives * 100 / z.positives->rnum, (float)false_positives * 100 / z.negatives->rnum, false_positives);
			ccfree(positive_rate);
			printf(" - first feature -\n");
			for (k = 0; k < weak_classifier.features[0].count; k++)
				printf(" - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[0].channel[k], weak_classifier.features[0].sat[k * 2].x, weak_classifier.features[0].sat[k * 2].y, weak_classifier.features[0].sat[k * 2 + 1].x, weak_classifier.features[0].sat[k * 2 + 1].y);
			if (weak_classifier.pass & 0x2)
			{
				printf(" - second feature, on left -\n");
				for (k = 0; k < weak_classifier.features[1].count; k++)
					printf(" - | - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[1].channel[k], weak_classifier.features[1].sat[k * 2].x, weak_classifier.features[1].sat[k * 2].y, weak_classifier.features[1].sat[k * 2 + 1].x, weak_classifier.features[1].sat[k * 2 + 1].y);
			}
			if (weak_classifier.pass & 0x1)
			{
				printf(" - second feature, on right -\n");
				for (k = 0; k < weak_classifier.features[2].count; k++)
					printf(" - | - %d - (%d, %d) - (%d, %d)\n", weak_classifier.features[2].channel[k], weak_classifier.features[2].sat[k * 2].x, weak_classifier.features[2].sat[k * 2].y, weak_classifier.features[2].sat[k * 2 + 1].x, weak_classifier.features[2].sat[k * 2 + 1].y);
			}
			z.classifier->interval = z.i + 1; // update interval
			z.classifier->cascade[z.i].size = z.size;
			z.classifier->cascade[z.i].thresholds = 0;
			z.x.example_state = 0;
			z.x.classifier = 0;
			ccv_function_state_resume(_ccv_icf_write_classifier_cascade_state, z, dir);
		}
		ccfree(z.features);
		ccv_array_free(z.positives);
		ccv_array_free(z.negatives);
		z.scale *= scale_factor;
	}
	gsl_rng_free(rng);
	ccv_function_state_finish();
	return z.classifier;
}

static uint64_t* _ccv_icf_precompute_classifier_cascade(ccv_icf_classifier_cascade_t* cascade, ccv_array_t* positives, ccv_array_t* negatives)
{
	int step = ((cascade->count - 1) >> 6) + 1;
	uint64_t* precomputed = (uint64_t*)ccmalloc(sizeof(uint64_t) * (positives->rnum + negatives->rnum) * step);
	uint64_t* result = precomputed;
	int i, j;
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)((i < positives->rnum) ? ccv_array_get(positives, i) : ccv_array_get(negatives, i - positives->rnum));
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

void ccv_icf_classifier_cascade_soft(ccv_icf_multiscale_classifier_cascade_t* multiscale_cascade, ccv_array_t* posfiles, int posnum, ccv_array_t* bgfiles, int negnum, const char* dir, ccv_icf_new_param_t params)
{
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	int i, j, k;
	for (i = 0; i < multiscale_cascade->interval; i++)
	{
		ccv_icf_classifier_cascade_t* cascade = multiscale_cascade->cascade + i;
		ccv_array_t* positives = _ccv_icf_collect_positives(rng, cascade->size, posfiles, posnum, params.deform_angle, params.deform_scale, params.deform_shift);
		ccv_array_t* negatives = _ccv_icf_collect_negatives(rng, cascade->size, bgfiles, negnum, params.deform_angle, params.deform_scale, params.deform_shift);
		int step = ((cascade->count - 1) >> 6) + 1;
		uint64_t* precomputed = _ccv_icf_precompute_classifier_cascade(cascade, positives, negatives);
		float* rate = (float*)ccmalloc(sizeof(float) * positives->rnum);
		uint64_t* computed = precomputed;
		for (j = 0; j < positives->rnum; j++)
		{
			rate[j] = 0;
			for (k = 0; k < cascade->count; k++)
			{
				uint64_t accept = computed[k >> 6] & (1UL << (k & 63));
				rate[j] += cascade->weak_classifiers[k].weigh[!!accept];
			}
			computed += step;
		}
		_ccv_icf_threshold_rating(rate, positives->rnum, 0);
		float threshold = rate[ccv_min((int)(params.acceptance * (positives->rnum + 0.5) - 0.5), positives->rnum)];
		ccfree(rate);
		computed = precomputed;
		// compute the final acceptance per positives / negatives with final threshold
		uint64_t* acceptance = (uint64_t*)ccmalloc(sizeof(uint64_t) * (((positives->rnum + negatives->rnum - 1) >> 6) + 1));
		int true_positives = 0, false_positives = 0;
		for (j = 0; j < positives->rnum; j++)
		{
			float rate = 0;
			for (k = 0; k < cascade->count; k++)
			{
				uint64_t accept = computed[k >> 6] & (1UL << (k & 63));
				rate += cascade->weak_classifiers[k].weigh[!!accept];
			}
			if (rate >= threshold)
			{
				acceptance[j >> 6] |= (1UL << (j & 63));
				++true_positives;
			} else
				acceptance[j >> 6] &= ~(1UL << (j & 63));
			computed += step;
		}
		for (j = 0; j < negatives->rnum; j++)
		{
			float rate = 0;
			for (k = 0; k < cascade->count; k++)
			{
				uint64_t accept = computed[k >> 6] & (1UL << (k & 63));
				rate += cascade->weak_classifiers[k].weigh[!!accept];
			}
			if (rate >= threshold)
			{
				acceptance[(j + positives->rnum) >> 6] |= (1UL << ((j + positives->rnum) & 63));
				++false_positives;
			} else
				acceptance[(j + positives->rnum) >> 6] &= ~(1UL << ((j + positives->rnum) & 63));
			computed += step;
		}
		printf(" - at threshold %f, true positive rate: %f%%, false positive rate: %f%% (%d)\n", threshold, (float)true_positives * 100 / positives->rnum, (float)false_positives * 100 / negatives->rnum, false_positives);
		rate = (float*)ccmalloc(sizeof(float) * (positives->rnum + negatives->rnum));
		memset(rate, 0, sizeof(float) * (positives->rnum + negatives->rnum));
		for (k = 0; k < cascade->count; k++)
		{
			computed = precomputed;
			for (j = 0; j < positives->rnum + negatives->rnum; j++)
			{
				uint64_t correct = computed[k >> 6] & (1UL << (k & 63));
				rate[j] += cascade->weak_classifiers[k].weigh[!!correct];
				computed += step;
			}
			float threshold = FLT_MAX;
			// find a threshold that keeps all accepted positives still acceptable
			for (j = 0; j < positives->rnum; j++)
			{
				uint64_t correct = acceptance[j >> 6] & (1UL << (j & 63));
				if (correct && rate[j] < threshold)
					threshold = rate[j];
			}
			// compute how many negatives we prune with this threshold
			for (j = 0; j < negatives->rnum; j++)
			{
			}
		}
		ccfree(rate);
		ccfree(acceptance);
	}
	gsl_rng_free(rng);
}

ccv_icf_multiscale_classifier_cascade_t* ccv_icf_read_classifier_cascade(const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/multiscale", directory);
	FILE* r = fopen(filename, "r");
	if (r)
	{
		int interval = 0;
		fscanf(r, "%d", &interval);
		fclose(r);
		ccv_icf_multiscale_classifier_cascade_t* classifier = (ccv_icf_multiscale_classifier_cascade_t*)ccmalloc(sizeof(ccv_icf_multiscale_classifier_cascade_t) + sizeof(ccv_icf_classifier_cascade_t) * interval);
		classifier->interval = interval;
		classifier->cascade = (ccv_icf_classifier_cascade_t*)(classifier + 1);
		int i, j, q;
		for (i = 0; i < interval; i++)
		{
			snprintf(filename, 1024, "%s/cascade-%d", directory, i + 1);
			r = fopen(filename, "r");
			if (r)
			{
				ccv_icf_classifier_cascade_t* cascade = classifier->cascade + i;
				fscanf(r, "%d %d %d", &cascade->count, &cascade->size.width, &cascade->size.height);
				cascade->weak_classifiers = (ccv_icf_decision_tree_t*)ccmalloc(sizeof(ccv_icf_decision_tree_t) * cascade->count);
				for (j = 0; j < cascade->count; j++)
				{
					ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + j;
					fscanf(r, "%u %a %a", &weak_classifier->pass, &weak_classifier->weigh[0], &weak_classifier->weigh[1]);
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
				int thresholds = 0;
				fscanf(r, "%d", &thresholds);
				if (thresholds > 0)
				{
					cascade->thresholds = (ccv_icf_threshold_t*)ccmalloc(sizeof(ccv_icf_threshold_t) * thresholds);
					for (j = 0; j < thresholds; j++)
						fscanf(r, "%d %a", &cascade->thresholds[j].index, &cascade->thresholds[j].threshold);
				} else
					cascade->thresholds = 0;
				fclose(r);
			}
		}
		return classifier;
	}
	return 0;
}

void ccv_icf_write_classifier_cascade(ccv_icf_multiscale_classifier_cascade_t* classifier, const char* directory)
{
	char filename[1024];
	snprintf(filename, 1024, "%s/multiscale", directory);
	FILE* w = fopen(filename, "w+");
	fprintf(w, "%d\n", classifier->interval);
	fclose(w);
	int i, j, q;
	for (i = 0; i < classifier->interval; i++)
	{
		snprintf(filename, 1024, "%s/cascade-%d", directory, i + 1);
		w = fopen(filename, "w+");
		fprintf(w, "%d %d %d\n", classifier->cascade[i].count, classifier->cascade[i].size.width, classifier->cascade[i].size.height);
		for (j = 0; j < classifier->cascade[i].count; j++)
		{
			ccv_icf_decision_tree_t* weak_classifier = classifier->cascade[i].weak_classifiers + j;
			fprintf(w, "%u %a %a", weak_classifier->pass, weak_classifier->weigh[0], weak_classifier->weigh[1]);
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
		if (classifier->cascade[i].thresholds)
		{
			q = 0;
			for (j = 0; classifier->cascade[i].thresholds[j].index < classifier->cascade[i].count; j++)
				++q;
			fprintf(w, "%d\n", q);
			for (j = 0; j < q; j++)
				fprintf(w, "%d %a\n", classifier->cascade[i].thresholds[j].index, classifier->cascade[i].thresholds[j].threshold);
		} else {
			fprintf(w, "0\n");
		}
		fclose(w);
	}
}

void ccv_icf_classifier_cascade_free(ccv_icf_multiscale_classifier_cascade_t* classifier)
{
}

ccv_array_t* ccv_icf_detect_objects(ccv_dense_matrix_t* a, ccv_icf_multiscale_classifier_cascade_t** multiscale_cascade, int count, ccv_icf_param_t params)
{
	assert(count > 0);
	int i, j, k, q, x, y;
	for (i = 0; i < count - 1; i++)
		assert(multiscale_cascade[i]->interval == multiscale_cascade[i + 1]->interval);
	int min_win = 0x7FFFFFFF;
	for (i = 0; i < count; i++)
		min_win = ccv_min(min_win, ccv_min(multiscale_cascade[i]->cascade[0].size.width, multiscale_cascade[i]->cascade[0].size.height));
	int scale_upto = (int)(log((double)ccv_min(a->rows, a->cols) / min_win) / log(2.));
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * scale_upto);
	pyr[0] = a;
	for (i = 1; i < scale_upto; i++)
	{
		pyr[i] = 0;
		ccv_sample_down(pyr[i - 1], &pyr[i], 0, 0, 0);
	}
	ccv_array_t* seq = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	for (i = 0; i < scale_upto; i++)
	{
		ccv_dense_matrix_t* icf = 0;
		ccv_icf(pyr[i], &icf, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		// run it
		int ch = CCV_GET_CHANNEL(sat->type);
		assert(CCV_GET_DATA_TYPE(sat->type) == CCV_32F);
		for (j = 0; j < count; j++)
			for (k = 0; k < multiscale_cascade[j]->interval; k++)
			{
				ccv_icf_classifier_cascade_t* cascade = multiscale_cascade[j]->cascade + k;
				float* ptr = sat->data.f32;
				for (y = 0; y < sat->rows - cascade->size.height - 1; y++)
				{
					for (x = 0; x < sat->cols - cascade->size.width - 1; x++)
					{
						int pass = 1;
						float sum = 0;
						ccv_icf_threshold_t* thresholds = cascade->thresholds;
						for (q = 0; q < cascade->count; q++)
						{
							ccv_icf_decision_tree_t* weak_classifier = cascade->weak_classifiers + q;
							int c = _ccv_icf_run_weak_classifier(weak_classifier, ptr, sat->cols, ch, x, 0);
							sum += weak_classifier->weigh[c];
							if (q == thresholds->index)
							{
								if (sum < thresholds->threshold)
								{
									pass = 0;
									break;
								}
								++thresholds;
							}
						}
						if (pass)
						{
							ccv_comp_t comp;
							comp.rect = ccv_rect(x << i, y << i, cascade->size.width << i, cascade->size.height << i);
							comp.id = j;
							comp.neighbors = 1;
							comp.confidence = sum;
							ccv_array_push(seq, &comp);
						}
					}
					ptr += sat->cols * ch;
				}
			}
		ccv_matrix_free(icf);
		ccv_matrix_free(sat);
	}
	for (i = 1; i < scale_upto; i++)
		ccv_matrix_free(pyr[i]);
	return seq;
}
