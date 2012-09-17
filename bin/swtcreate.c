#include "ccv.h"
#include "ccv_internal.h"
#include <sys/time.h>
#include <ctype.h>

// compute f-rate of swt
static double _ccv_evaluate_swt(ccv_array_t* images, ccv_array_t* gt, double a, ccv_swt_param_t params, double* precision, double* recall)
{
	int i, j, k;
	double total_f = 0, total_precision = 0, total_recall = 0;
	for (i = 0; i < images->rnum; i++)
	{
		char* name = *(char**)ccv_array_get(images, i);
		ccv_array_t* truth = *(ccv_array_t**)ccv_array_get(gt, i);
		ccv_dense_matrix_t* image = 0;
		ccv_read(name, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
		ccv_array_t* words = ccv_swt_detect_words(image, params);
		ccv_matrix_free(image);
		double f = 0, precision = 0, recall = 0;
		for (j = 0; j < words->rnum; j++)
		{
			ccv_rect_t* estimate = (ccv_rect_t*)ccv_array_get(words, j);
			int match = 0;
			for (k = 0; k < truth->rnum; k++)
			{
				ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(truth, k);
				match = ccv_max(match, ccv_max(ccv_min(target->x + target->width, estimate->x + estimate->width) - ccv_max(target->x, estimate->x), 0) * ccv_max(ccv_min(target->y + target->height, estimate->y + estimate->height) - ccv_max(target->y, estimate->y), 0));
			}
			precision += (double)match / (double)(estimate->width * estimate->height);
		}
		if (words->rnum > 0)
			precision /= words->rnum;
		for (j = 0; j < truth->rnum; j++)
		{
			ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(truth, j);
			int match = 0;
			for (k = 0; k < words->rnum; k++)
			{
				ccv_rect_t* estimate = (ccv_rect_t*)ccv_array_get(words, k);
				match = ccv_max(match, ccv_max(ccv_min(target->x + target->width, estimate->x + estimate->width) - ccv_max(target->x, estimate->x), 0) * ccv_max(ccv_min(target->y + target->height, estimate->y + estimate->height) - ccv_max(target->y, estimate->y), 0));
			}
			recall += (double)match / (double)(target->width * target->height);
		}
		ccv_array_free(words);
		if (truth->rnum > 0)
			recall /= truth->rnum;
		if (precision > 0 && recall > 0)
			f = 1 / (a / precision + (1 - a) / recall);
		total_f += f;
		total_precision += precision;
		total_recall += recall;
	}
	total_f /= images->rnum;
	total_precision /= images->rnum;
	total_recall /= images->rnum;
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
	ccv_enable_cache(1024 * 1024 * 1024);
	ccv_array_t* aof = ccv_array_new(sizeof(char*), 64, 0);
	ccv_array_t* aow = ccv_array_new(sizeof(ccv_array_t*), 64, 0);
	ccv_array_t* cw = 0;
	char* file = (char*)malloc(1024);
	size_t len = 1024;
	ssize_t read;
	while ((read = getline(&file, &len, r)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		double x, y, width, height;
		int recognized = sscanf(file, "%lf %lf %lf %lf", &x, &y, &width, &height);
		if (recognized == 4)
		{
			ccv_rect_t rect = {
				.x = (int)(x + 0.5),
				.y = (int)(y + 0.5),
				.width = (int)(width + 0.5),
				.height = (int)(height + 0.5)
			};
			ccv_array_push(cw, &rect);
		} else {
			char* name = (char*)malloc(ccv_min(1023, strlen(file)) + 1);
			strncpy(name, file, ccv_min(1023, strlen(file)) + 1);
			ccv_array_push(aof, &name);
			cw = ccv_array_new(sizeof(ccv_rect_t), 1, 0);
			ccv_array_push(aow, &cw);
		}
	}
	free(file);
	printf("loaded %d images for parameter search\n", aof->rnum);
	int i;
	ccv_swt_param_t params = {
		.interval = 0,
		.same_word_thresh = { 0.5, 0.9 },
		.min_neighbors = 0,
		.scale_invariant = 0,
		.size = 3,
		.low_thresh = 74,
		.high_thresh = 212,
		.max_height = 500,
		.min_height = 12,
		.min_area = 95,
		.letter_occlude_thresh = 4,
		.aspect_ratio = 6,
		.std_ratio = 0.78,
		.thickness_ratio = 1.8,
		.height_ratio = 2.0,
		.intensity_thresh = 33,
		.distance_ratio = 3.4,
		.intersect_ratio = 1.2,
		.letter_thresh = 3,
		.elongate_ratio = 1.3,
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
	ccv_swt_range_t letter_occlude_thresh_range = {
		.min_value = 0,
		.max_value = 5,
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
	ccv_swt_range_t std_ratio_range = {
		.min_value = 0.1,
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
		f = _ccv_evaluate_swt(aof, aow, a, params, &precision, &recall); \
		if (f > best_f) \
		{ \
			best_params = params; \
			best_f = f; \
			best_precision = precision; \
			best_recall = recall; \
		} \
		FLUSH("current f : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; best f : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; at " #parameter " = %lg (%lg <- [%lg, %lg])", f * 100, precision * 100, recall * 100, best_f * 100, best_precision * 100, best_recall * 100, (double)best_params.parameter, v, parameter##_range.min_value, parameter##_range.max_value); \
	} \
	printf("\n");
	int max_round = 10;
	for (i = 0; i < max_round; i++)
	{
		optimize(size, int, 0.5);
		optimize(low_thresh, int, 0.5);
		optimize(high_thresh, int, 0.5);
		optimize(max_height, int, 0.5);
		optimize(min_height, int, 0.5);
		optimize(min_area, int, 0.5);
		optimize(letter_occlude_thresh, int, 0.5);
		optimize(aspect_ratio, double, 0);
		optimize(std_ratio, double, 0);
		optimize(thickness_ratio, double, 0);
		optimize(height_ratio, double, 0);
		optimize(intensity_thresh, int, 0.5);
		optimize(distance_ratio, double, 0);
		optimize(intersect_ratio, double, 0);
		optimize(letter_thresh, int, 0.5);
		optimize(elongate_ratio, double, 0);
		optimize(breakdown_ratio, double, 0);
		printf("At round %d(of %d) : best parameters for swt is:\n"
			   "\tsize = %d\n"
			   "\tlow_thresh = %d\n"
			   "\thigh_thresh = %d\n"
			   "\tmax_height = %d\n"
			   "\tmin_height = %d\n"
			   "\tmin_area = %d\n"
			   "\tletter_occlude_thresh = %d\n"
			   "\taspect_ratio = %lf\n"
			   "\tstd_ratio = %lf\n"
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
			   best_params.letter_occlude_thresh,
			   best_params.aspect_ratio,
			   best_params.std_ratio,
			   best_params.thickness_ratio,
			   best_params.height_ratio,
			   best_params.intensity_thresh,
			   best_params.distance_ratio,
			   best_params.intersect_ratio,
			   best_params.letter_thresh,
			   best_params.elongate_ratio,
			   best_params.breakdown_ratio);
	}
#undef optimize
	for (i = 0; i < aof->rnum; i++)
	{
		char* name = *(char**)ccv_array_get(aof, i);
		free(name);
		ccv_array_t* cw = *(ccv_array_t**)ccv_array_get(aow, i);
		ccv_array_free(cw);
	}
	ccv_array_free(aof);
	ccv_array_free(aow);
	ccv_drain_cache();
	return 0;
}
