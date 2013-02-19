#include "ccv.h"
#include "ccv_internal.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
	int i;
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
	}
}

static void _ccv_icf_check_params(ccv_icf_new_param_t params)
{
	assert(params.interval >= 0);
	assert(params.size.width > 0 && params.size.height > 0);
	assert(params.deform_shift > 0);
	assert(params.deform_angle > 0);
	assert(params.deform_scale > 0);
	assert(params.C > 0);
	assert(params.feature_size > 0);
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
	ccv_decimal_slice(b, &resize, 0, b->rows * 0.5 - (size.height + 2) / scale_ratio * 0.5, b->cols * 0.5 - (size.width + 2) / scale_ratio * 0.5, (int)((size.height + 2) / scale_ratio + 0.5), (int)((size.width + 2) / scale_ratio + 0.5));
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
	uint8_t binary:1;
	uint8_t correct:1;
	uint32_t index:30;
	double weight;
} ccv_icf_example_state_t;

typedef struct {
	ccv_array_t* positives;
	ccv_array_t* negatives;
	ccv_icf_feature_t* features;
	ccv_size_t size;
	double scale;
	ccv_icf_example_state_t* example_state;
} ccv_icf_classifier_cascade_state_t;

static void _ccv_icf_feature_pre_learn(double C, ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_example_state_t* example_state)
{
#ifdef HAVE_LIBLINEAR
	int i, j, q;
	for (i = 0; i < feature_size; i++)
	{
		ccv_icf_feature_t* feature = features + i;
		struct problem prob;
		prob.n = feature->count + 1;
		prob.bias = 1.0;
		prob.y = malloc(sizeof(prob.y[0]) * (positives->rnum + negatives->rnum));
		prob.x = (struct feature_node**)malloc(sizeof(struct feature_node*) * (positives->rnum + negatives->rnum));
		struct feature_node* feature_cluster = (struct feature_node*)malloc(sizeof(struct feature_node) * (positives->rnum + negatives->rnum) * (feature->count + 2));
		struct feature_node* feature_node = feature_cluster;
		for (j = 0; j < positives->rnum + negatives->rnum; j++)
		{
			ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(example_state[j].binary ? positives : negatives, example_state[j].index);
			ccv_dense_matrix_t* icf = 0;
			ccv_icf(a, &icf, 0);
			ccv_dense_matrix_t* sat = 0;
			ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
			ccv_matrix_free(icf);
			float* ptr = sat->data.f32;
			int ch = CCV_GET_CHANNEL(sat->type);
			for (q = 0; q < feature->count; q++)
			{
				feature_node[q].index = q;
				feature_node[q].value = ptr[(feature->sat[q * 2 + 1].x + 1 + (feature->sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2].x + 1 + (feature->sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature->channel[q]] + ptr[(feature->sat[q * 2].x + 1 + (feature->sat[q * 2].y + 1) * sat->cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2 + 1].x + 1 + (feature->sat[q * 2].y + 1) * sat->cols) * ch + feature->channel[q]];
			}
			ccv_matrix_free(sat);
			feature_node[feature->count].index = feature->count;
			feature_node[feature->count].value = prob.bias;
			feature_node[feature->count + 1].index = -1;
			prob.x[j] = feature_node;
			prob.y[j] = example_state[j].binary ? 1 : -1;
			feature_node += feature->count + 2;
		}
		prob.l = positives->rnum + negatives->rnum;
		struct parameter linear_parameters = { .solver_type = L2R_L1LOSS_SVC_DUAL,
											   .eps = 1e-1,
											   .C = C,
											   .nr_weight = 0,
											   .weight_label = 0,
											   .weight = 0 };
		const char* err = check_parameter(&prob, &linear_parameters);
		if (err)
		{
			printf(" - ERROR: cannot pass check parameter: %s\n", err);
			exit(-1);
		}
		struct model* linear = train(&prob, &linear_parameters);
		for (q = 0; q < feature->count; q++)
			feature->alpha[q] = linear->w[q];
		feature->beta = linear->w[feature->count];
		free_and_destroy_model(&linear);
		free(feature_cluster);
		free(prob.x);
		free(prob.y);
	}
#endif
}

static ccv_icf_feature_t _ccv_icf_find_best_feature(ccv_icf_feature_t* features, int feature_size, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_example_state_t* example_state)
{
	int i, j, q;
	double* rate = (double*)ccmalloc(sizeof(double) * feature_size);
	memset(rate, 0, sizeof(double) * feature_size);
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(example_state[i].binary ? positives : negatives, example_state[i].index);
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
			float c = feature->beta;
			for (q = 0; q < feature->count; q++)
				c += (ptr[(feature->sat[q * 2 + 1].x + 1 + (feature->sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2].x + 1 + (feature->sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature->channel[q]] + ptr[(feature->sat[q * 2].x + 1 + (feature->sat[q * 2].y + 1) * sat->cols) * ch + feature->channel[q]] - ptr[(feature->sat[q * 2 + 1].x + 1 + (feature->sat[q * 2].y + 1) * sat->cols) * ch + feature->channel[q]]) * feature->alpha[q];
			if (c > 0)
				rate[j] += example_state[i].binary ? example_state[i].weight : 0;
			else
				rate[j] += example_state[i].binary ? 0 : example_state[i].weight;
		}
		ccv_matrix_free(sat);
	}
	ccv_icf_feature_t best_feature;
	double best_rate = 0; // at least one feature should be better than 0
	for (i = 0; i < feature_size; i++)
		if (rate[i] > best_rate)
		{
			best_rate = rate[i];
			best_feature = features[i];
		}
	ccfree(rate);
	return best_feature;
}

static double _ccv_icf_rate_feature(ccv_icf_feature_t feature, ccv_array_t* positives, ccv_array_t* negatives, ccv_icf_example_state_t* example_state)
{
	int i, q;
	double rate = 0;
	for (i = 0; i < positives->rnum + negatives->rnum; i++)
	{
		ccv_dense_matrix_t* a = (ccv_dense_matrix_t*)ccv_array_get(example_state[i].binary ? positives : negatives, example_state[i].index);
		ccv_dense_matrix_t* icf = 0;
		// we have 1px padding around the image
		ccv_icf(a, &icf, 0);
		ccv_dense_matrix_t* sat = 0;
		ccv_sat(icf, &sat, 0, CCV_PADDING_ZERO);
		ccv_matrix_free(icf);
		float* ptr = sat->data.f32;
		int ch = CCV_GET_CHANNEL(sat->type);
		float c = feature.beta;
		for (q = 0; q < feature.count; q++)
			c += (ptr[(feature.sat[q * 2 + 1].x + 1 + (feature.sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature.channel[q]] - ptr[(feature.sat[q * 2].x + 1 + (feature.sat[q * 2 + 1].y + 1) * sat->cols) * ch + feature.channel[q]] + ptr[(feature.sat[q * 2].x + 1 + (feature.sat[q * 2].y + 1) * sat->cols) * ch + feature.channel[q]] - ptr[(feature.sat[q * 2 + 1].x + 1 + (feature.sat[q * 2].y + 1) * sat->cols) * ch + feature.channel[q]]) * feature.alpha[q];
		if (c > 0)
		{
			rate += example_state[i].binary ? example_state[i].weight : 0;
			example_state[i].correct = example_state[i].binary;
		} else {
			rate += example_state[i].binary ? 0 : example_state[i].weight;
			example_state[i].correct = !example_state[i].binary;
		}
		ccv_matrix_free(sat);
	}
	return rate;
}

void ccv_icf_classifier_cascade_new(ccv_array_t* posfiles, int posnum, ccv_array_t* bgfiles, int negnum, const char* dir, ccv_icf_new_param_t params)
{
	_ccv_icf_check_params(params);
	int i, j, k;
	int scale_upto = params.interval + 1;
	double scale_factor = pow(2., 1. / scale_upto);
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	// we will keep all states inside this structure for easier save / resume across process
	// this should work better than ad-hoc one we used in DPM / BBF implementation
	ccv_icf_classifier_cascade_state_t cascade_state;
	cascade_state.scale = 1;
	for (i = 0; i < scale_upto; i++)
	{
		cascade_state.size = ccv_size((int)(params.size.width * cascade_state.scale + 0.5), (int)(params.size.height * cascade_state.scale + 0.5));
		cascade_state.features = (ccv_icf_feature_t*)ccmalloc(sizeof(ccv_icf_feature_t) * params.feature_size);
		// generate random features
		for (j = 0; j < params.feature_size; j++)
			_ccv_icf_randomize_feature(rng, cascade_state.size, cascade_state.features + j);
		cascade_state.positives = ccv_array_new(ccv_compute_dense_matrix_size(cascade_state.size.height + 2, cascade_state.size.width + 2, CCV_8U | CCV_C1), posnum, 0);
		cascade_state.negatives = ccv_array_new(ccv_compute_dense_matrix_size(cascade_state.size.height + 2, cascade_state.size.width + 2, CCV_8U | CCV_C1), negnum, 0);
		// collect positives (with random deformation)
		for (j = 0; j < posfiles->rnum; j++)
		{
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			if (image)
			{
				ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, file_info->pose, cascade_state.size, params.deform_angle, params.deform_scale, params.deform_shift);
				feature->sig = 0;
				ccv_array_push(cascade_state.positives, feature);
				ccv_matrix_free(feature);
				ccv_matrix_free(image);
			}
		}
		// randomly collect negatives (with random deformation)
		int npp = negnum / bgfiles->rnum;
		for (j = 0; j < bgfiles->rnum * npp; j += npp)
		{
			ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(bgfiles, j);
			ccv_dense_matrix_t* image = 0;
			ccv_read(file_info->filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			if (image)
			{
				double max_scale_ratio = ccv_min((double)image->rows / cascade_state.size.height, (double)image->cols / cascade_state.size.height);
				for (k = j; k < j + npp; k++)
				{
					ccv_decimal_pose_t pose;
					double scale_ratio = gsl_rng_uniform(rng) * max_scale_ratio;
					pose.a = cascade_state.size.width * 0.5 * scale_ratio;
					pose.b = cascade_state.size.height * 0.5 * scale_ratio;
					pose.x = gsl_rng_uniform_int(rng, (int)(image->cols - pose.a * 2 + 0.5)) + pose.a;
					pose.y = gsl_rng_uniform_int(rng, (int)(image->rows - pose.b * 2 + 0.5)) + pose.b;
					pose.roll = pose.pitch = pose.yaw = 0;
					ccv_dense_matrix_t* feature = _ccv_icf_capture_feature(rng, image, pose, cascade_state.size, params.deform_angle, params.deform_scale, params.deform_shift);
					feature->sig = 0;
					ccv_array_push(cascade_state.negatives, feature);
					ccv_matrix_free(feature);
				}
				ccv_matrix_free(image);
			}
		}
		cascade_state.example_state = (ccv_icf_example_state_t*)ccmalloc(sizeof(ccv_icf_example_state_t) * (cascade_state.negatives->rnum + cascade_state.positives->rnum));
		for (j = 0; j < cascade_state.positives->rnum; j++)
		{
			cascade_state.example_state[j].index = j;
			cascade_state.example_state[j].binary = 1;
			cascade_state.example_state[j].weight = 0.5 / cascade_state.positives->rnum;
		}
		for (j = 0; j < cascade_state.negatives->rnum; j++)
		{
			cascade_state.example_state[cascade_state.positives->rnum + j].index = j;
			cascade_state.example_state[cascade_state.positives->rnum + j].binary = 0;
			cascade_state.example_state[cascade_state.positives->rnum + j].weight = 0.5 / cascade_state.negatives->rnum;
		}
		_ccv_icf_feature_pre_learn(params.C, cascade_state.features, params.feature_size, cascade_state.positives, cascade_state.negatives, cascade_state.example_state);
		for (j = 0; j < params.select_feature_size; j++)
		{
			ccv_icf_feature_t best_feature = _ccv_icf_find_best_feature(cascade_state.features, params.feature_size, cascade_state.positives, cascade_state.negatives, cascade_state.example_state);
			double rate = _ccv_icf_rate_feature(best_feature, cascade_state.positives, cascade_state.negatives, cascade_state.example_state);
			double alpha = sqrt((1 - rate) / rate);
			double beta = 1.0 / alpha;
			double reweigh = 0;
			for (k = 0; k < cascade_state.positives->rnum + cascade_state.negatives->rnum; k++)
			{
				cascade_state.example_state[k].weight *= (cascade_state.example_state[k].correct) ? alpha : beta;
				reweigh += cascade_state.example_state[k].weight;
			}
			reweigh = 1.0 / reweigh;
			// balancing the weight to sum 1.0
			for (k = 0; k < cascade_state.positives->rnum + cascade_state.negatives->rnum; k++)
				cascade_state.example_state[k].weight *= reweigh;
		}
		ccfree(cascade_state.features);
		ccv_array_free(cascade_state.positives);
		ccv_array_free(cascade_state.negatives);
		cascade_state.scale *= scale_factor;
	}
	gsl_rng_free(rng);
}

ccv_array_t* ccv_icf_detect_objects(ccv_dense_matrix_t* a, ccv_icf_multiscale_classifier_cascade_t** multiscale_cascade, int count, ccv_icf_param_t params)
{
	assert(count > 0);
	int i, j, k, p, q, x, y;
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
						for (p = 0; p < cascade->count; p++)
						{
							ccv_icf_feature_t* feature = cascade->features + p;
							float c = feature->beta;
							for (q = 0; q < feature->count; q++)
								c += (ptr[(x + feature->sat[q * 2 + 1].x + feature->sat[q * 2 + 1].y * sat->cols) * ch + feature->channel[q]] - ptr[(x + feature->sat[q * 2].x + feature->sat[q * 2 + 1].y * sat->cols) * ch + feature->channel[q]] + ptr[(x + feature->sat[q * 2].x + feature->sat[q * 2].y * sat->cols) * ch + feature->channel[q]] - ptr[(x + feature->sat[q * 2 + 1].x + feature->sat[q * 2].y * sat->cols) * ch + feature->channel[q]]) * feature->alpha[q];
							sum += c > 0 ? feature->weigh[0] : feature->weigh[1];
							if (p == thresholds->index)
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
