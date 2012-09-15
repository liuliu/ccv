#include "ccv.h"
#include "ccv_internal.h"
#include <sys/time.h>
#include <ctype.h>

// compute f-rate of swt
static double _ccv_evaluate_swt(int n, ccv_dense_matrix_t** images, ccv_array_t** truth, double a, ccv_swt_param_t params, double* precision, double* recall)
{
	int i, j, k;
	double total_f = 0, total_precision = 0, total_recall = 0;
	for (i = 0; i < n; i++)
	{
		ccv_array_t* words = ccv_swt_detect_words(images[i], params);
		double f = 0, precision = 0, recall = 0;
		for (j = 0; j < words->rnum; j++)
		{
			ccv_rect_t* estimate = (ccv_rect_t*)ccv_array_get(words, j);
			int match = 0;
			for (k = 0; k < truth[i]->rnum; k++)
			{
				ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(truth[i], k);
				match = ccv_max(match, ccv_max(ccv_min(target->x + target->width, estimate->x + estimate->width) - ccv_max(target->x, estimate->x), 0) * ccv_max(ccv_min(target->y + target->height, estimate->y + estimate->height) - ccv_max(target->y, estimate->y), 0));
			}
			precision += (double)match / (double)(estimate->width * estimate->height);
		}
		if (words->rnum > 0)
			precision /= words->rnum;
		for (j = 0; j < truth[i]->rnum; j++)
		{
			ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(truth[i], j);
			int match = 0;
			for (k = 0; k < words->rnum; k++)
			{
				ccv_rect_t* estimate = (ccv_rect_t*)ccv_array_get(words, k);
				match = ccv_max(match, ccv_max(ccv_min(target->x + target->width, estimate->x + estimate->width) - ccv_max(target->x, estimate->x), 0) * ccv_max(ccv_min(target->y + target->height, estimate->y + estimate->height) - ccv_max(target->y, estimate->y), 0));
			}
			recall += (double)match / (double)(target->width * target->height);
		}
		ccv_array_free(words);
		if (truth[i]->rnum > 0)
			recall /= truth[i]->rnum;
		if (precision > 0 && recall > 0)
			f = 1 / (a / precision + (1 - a) / recall);
		total_f += f;
		total_precision += precision;
		total_recall += recall;
	}
	total_f /= n;
	total_precision /= n;
	total_recall /= n;
	if (precision)
		*precision = total_precision;
	if (recall)
		*recall = total_recall;
	return total_f;
}

typedef struct {
	double min_value;
	double max_value;
	double step;
} ccv_swt_range_t;

int main(int argc, char** argv)
{
	FILE* r = fopen(argv[1], "rt");
	if (argc == 3)
		chdir(argv[2]);
	int images;
	fscanf(r, "%d", &images);
	int i;
	ccv_enable_default_cache();
	ccv_dense_matrix_t** aof = (ccv_dense_matrix_t**)ccmalloc(sizeof(ccv_dense_matrix_t*) * images);
	ccv_array_t** aow = (ccv_array_t**)ccmalloc(sizeof(ccv_array_t**) * images);
	for (i = 0; i < images; i++)
	{
		char file[1000];
		fscanf(r, "%s", file);
		aof[i] = 0;
		ccv_read(file, aof + i, CCV_IO_GRAY | CCV_IO_ANY_FILE);
		int locations;
		fscanf(r, "%d", &locations);
		int j;
		aow[i] = ccv_array_new(sizeof(ccv_rect_t), locations, 0);
		for (j = 0; j < locations; j++)
		{
			double x, y, width, height;
			fscanf(r, "%lf %lf %lf %lf", &x, &y, &width, &height);
			ccv_rect_t rect = { .x = (int)x, .y = (int)y, .width = (int)width, .height = (int)height };
			ccv_array_push(aow[i], &rect);
		}
	}
	ccv_swt_param_t params = {
		.size = 3,
		.low_thresh = 76,
		.high_thresh = 228,
		.max_height = 500,
		.min_height = 10,
		.min_area = 60,
		.aspect_ratio = 10,
		.variance_ratio = 0.72,
		.thickness_ratio = 1.5,
		.height_ratio = 2,
		.intensity_thresh = 26,
		.distance_ratio = 3,
		.intersect_ratio = 2,
		.letter_thresh = 3,
		.elongate_ratio = 1.6,
		.breakdown = 1,
		.breakdown_ratio = 1.0,
	};
	ccv_swt_range_t size_range = {
		.min_value = 1,
		.max_value = 3,
		.step = 2,
	};
	ccv_swt_range_t low_thresh_range = {
		.min_value = 50,
		.max_value = 150,
		.step = 1,
	};
	ccv_swt_range_t high_thresh_range = {
		.min_value = 200,
		.max_value = 350,
		.step = 1,
	};
	ccv_swt_range_t max_height_range = {
		.min_value = 500,
		.max_value = 500,
		.step = 1,
	};
	ccv_swt_range_t min_height_range = {
		.min_value = 5,
		.max_value = 30,
		.step = 1,
	};
	ccv_swt_range_t min_area_range = {
		.min_value = 10,
		.max_value = 100,
		.step = 1,
	};
	ccv_swt_range_t aspect_ratio_range = {
		.min_value = 5,
		.max_value = 15,
		.step = 1,
	};
	ccv_swt_range_t variance_ratio_range = {
		.min_value = 0.3,
		.max_value = 1.0,
		.step = 0.01,
	};
	ccv_swt_range_t thickness_ratio_range = {
		.min_value = 1.0,
		.max_value = 2.0,
		.step = 0.1,
	};
	ccv_swt_range_t height_ratio_range = {
		.min_value = 1.0,
		.max_value = 3.0,
		.step = 0.1,
	};
	ccv_swt_range_t intensity_thresh_range = {
		.min_value = 1,
		.max_value = 50,
		.step = 1,
	};
	ccv_swt_range_t distance_ratio_range = {
		.min_value = 1.0,
		.max_value = 5.0,
		.step = 0.1,
	};
	ccv_swt_range_t intersect_ratio_range = {
		.min_value = 0.0,
		.max_value = 5.0,
		.step = 0.1,
	};
	ccv_swt_range_t letter_thresh_range = {
		.min_value = 0,
		.max_value = 5,
		.step = 1,
	};
	ccv_swt_range_t elongate_ratio_range = {
		.min_value = 0.1,
		.max_value = 2.5,
		.step = 0.1,
	};
	ccv_swt_range_t breakdown_ratio_range = {
		.min_value = 0.5,
		.max_value = 1.5,
		.step = 0.01,
	};
	double best_f = 0, best_precision = 0, best_recall = 0;
	double a = 0.5;
	double v;
	ccv_swt_param_t best_params = params;
#define optimize(parameter, type, rounding) \
	params = best_params; \
	for (v = parameter##_range.min_value; v <= parameter##_range.max_value; v += parameter##_range.step) \
	{ \
		params.parameter = (type)(v + rounding); \
		double f, recall, precision; \
		f = _ccv_evaluate_swt(images, aof, aow, a, params, &precision, &recall); \
		if (f > best_f) \
		{ \
			best_params = params; \
			best_f = f; \
			best_precision = precision; \
			best_recall = recall; \
		} \
		FLUSH("current f : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; best f : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; at " #parameter " = %lg (%lg <<[%lg, %lg])", f * 100, precision * 100, recall * 100, best_f * 100, best_precision * 100, best_recall * 100, (double)best_params.parameter, v, parameter##_range.min_value, parameter##_range.max_value); \
	}
	int max_round = 10;
	for (i = 0; i < max_round; i++)
	optimize(size, int, 0.5);
	optimize(low_thresh, int, 0.5);
	optimize(high_thresh, int, 0.5);
	optimize(max_height, int, 0.5);
	optimize(min_height, int, 0.5);
	optimize(min_area, int, 0.5);
	optimize(aspect_ratio, double, 0);
	optimize(variance_ratio, double, 0);
	optimize(thickness_ratio, double, 0);
	optimize(height_ratio, double, 0);
	optimize(intensity_thresh, int, 0.5);
	optimize(distance_ratio, double, 0);
	optimize(intersect_ratio, double, 0);
	optimize(letter_thresh, int, 0.5);
	optimize(elongate_ratio, double, 0);
	optimize(breakdown_ratio, double, 0);
	printf("\nAt round %d(of %d) : best parameters for swt is:\n"
		   "\tsize = %d\n"
		   "\tlow_thresh = %d\n"
		   "\thigh_thresh = %d\n"
		   "\tmax_height = %d\n"
		   "\tmin_height = %d\n"
		   "\tmin_area = %d\n"
		   "\taspect_ratio = %lf\n"
		   "\tvariance_ratio = %lf\n"
		   "\tthickness_ratio = %lf\n"
		   "\theight_ratio = %lf\n"
		   "\tintensity_thresh = %d\n"
		   "\tdistance_ratio = %lf\n"
		   "\tintersect_ratio = %lf\n"
		   "\tletter_thresh = %d\n"
		   "\telongate_ratio = %lf\n"
		   "\tbreakdown_ratio = %lf\n",
		   i + 1, max_round,
		   best_params.size,
		   best_params.low_thresh,
		   best_params.high_thresh,
		   best_params.max_height,
		   best_params.min_height,
		   best_params.min_area,
		   best_params.aspect_ratio,
		   best_params.variance_ratio,
		   best_params.thickness_ratio,
		   best_params.height_ratio,
		   best_params.intensity_thresh,
		   best_params.distance_ratio,
		   best_params.intersect_ratio,
		   best_params.letter_thresh,
		   best_params.elongate_ratio,
		   best_params.breakdown_ratio);
#undef optimize
	ccfree(aof);
	ccfree(aow);
	ccv_drain_cache();
	return 0;
}
