#include "ccv.h"
#include "ccv_internal.h"
#include <sys/time.h>
#include <ctype.h>
#include <getopt.h>

static void exit_with_help()
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    swtreate input-text [OPTION...]\n\n"
	"  \033[1mARGUMENTS\033[0m\n\n"
	"    input-text: text file contains a list of text locations in format:\n"
	"                <file name>\\newline\n"
	"                x y width height <of one text location>\\newline\n"
	"                x y width height <of another text location>\\newline\n\n"
	"  \033[1mOPTIONS\033[0m\n\n"
	"    --size : the window size of canny filter [DEFAULT TO 1,2,3]\n"
	"    --low-thresh : the low threshold of canny filter [DEFAULT TO 50,1,150]\n"
	"    --high-thresh : the high threshold of canny filter [DEFAULT TO 200,1,350]\n"
	"    --max-height : the maximal height of letter [DEFAULT TO 500,1,500]\n"
	"    --min-height : the minimal height of letter [DEFAULT TO 5,1,30]\n"
	"    --min-area : the minimal area of letter [DEFAULT TO 10,1,100]\n"
	"    --aspect-ratio : the aspect ratio of letter [DEFAULT TO 5,1,15]\n"
	"    --std-ratio : the maximal std to mean ratio of letter [DEFAULT TO 0.1,0.01,1.0]\n"
	"    --thickness-ratio : the maximal allowance of thickness change between two letters [DEFAULT TO 1,0.1,2]\n"
	"    --height-ratio : the maximal allowance of height change between two letters [DEFAULT TO 1,0.1,3]\n"
	"    --intensity-thresh : how much intensity tolerance between two letters [DEFAULT TO 1,1,50]\n"
	"    --letter-occlude-thresh : how many letters one letter rectangle can occlude [DEFAULT TO 0,1,5]\n"
	"    --distance-ratio : the distance between two letters comparing to their width [DEFAULT TO 1,0.1,5]\n"
	"    --intersect-ratio : how much in the y-axis two letters intersect with each other [DEFAULT TO 0,0.1,5]\n"
	"    --letter-thresh : how many letters in minimal should one text line contains [DEFAULT TO 0,1,5]\n"
	"    --elongate-ratio : what's the minimal ratio between text line's width and height [DEFAULT TO 0.1,0.1,2.5]\n"
	"    --breakdown-ratio : what's the ratio for text line to break down into words [DEFAULT TO 0.5,0.01,1.5]\n"
	"    --breakdown : support to break text lines down to words [DEFAULT TO 1]\n"
	"    --iterations : how many iterations for the search [DEFAULT TO 10]\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	);
	exit(-1);
}

static double one_g = 0.8;
static double one_d = 0.4;
static double om_one = 0.8;
static double center_diff_thr = 1.0;

// compute harmonic mean of precision / recall of swt
static double _ccv_evaluate_wolf(ccv_array_t* detected, ccv_array_t* gt, double a, ccv_swt_param_t params, double* precision, double* recall)
{
	int i, j, k;
	int total_detected = 0, total_truth = 0;
	double total_precision = 0, total_recall = 0;
	for (i = 0; i < detected->rnum; i++)
	{
		ccv_array_t* words = *(ccv_array_t**)ccv_array_get(detected, i);
		ccv_array_t* truth = *(ccv_array_t**)ccv_array_get(gt, i);
		if (words->rnum == 0 || truth->rnum == 0)
		{
			ccv_array_free(words);
			continue;
		}
		total_detected += words->rnum;
		total_truth += truth->rnum;
		int* cG = (int*)ccmalloc(sizeof(int) * truth->rnum);
		int* cD = (int*)ccmalloc(sizeof(int) * words->rnum);
		memset(cG, 0, sizeof(int) * truth->rnum);
		memset(cD, 0, sizeof(int) * words->rnum);
		double* mG = (double*)ccmalloc(sizeof(double) * truth->rnum * words->rnum);
		double* mD = (double*)ccmalloc(sizeof(double) * truth->rnum * words->rnum);
		memset(mG, 0, sizeof(double) * truth->rnum * words->rnum);
		memset(mD, 0, sizeof(double) * truth->rnum * words->rnum);
		for (j = 0; j < truth->rnum; j++)
		{
			ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(truth, j);
			for (k = 0; k < words->rnum; k++)
			{
				ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(words, k);
				int match = ccv_max(ccv_min(target->x + target->width, rect->x + rect->width) - ccv_max(target->x, rect->x), 0) * ccv_max(ccv_min(target->y + target->height, rect->y + rect->height) - ccv_max(target->y, rect->y), 0);
				if (match > 0)
				{
					mG[j * words->rnum + k] = (double)match / (double)(rect->width * rect->height);
					mD[k * truth->rnum + j] = (double)match / (double)(target->width * target->height);
					++cG[j];
					++cD[k];
				}
			}
		}
		unsigned char* tG = (unsigned char*)ccmalloc(truth->rnum);
		unsigned char* tD = (unsigned char*)ccmalloc(words->rnum);
		memset(tG, 0, truth->rnum);
		memset(tD, 0, words->rnum);
		// one to one match
		for (j = 0; j < truth->rnum; j++)
		{
			if (cG[j] != 1)
				continue;
			ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(truth, j);
			for (k = 0; k < words->rnum; k++)
			{
				if (cD[j] != 1)
					continue;
				ccv_rect_t* target = (ccv_rect_t*)ccv_array_get(words, k);
				if (mG[j * words->rnum + k] >= one_g && mD[k * truth->rnum + j] >= one_d)
				{
					double dx = (target->x + target->width * 0.5) - (rect->x + rect->width * 0.5);
					double dy = (target->y + target->height * 0.5) - (rect->y + rect->height * 0.5);
					double d = sqrt(dx * dx + dy * dy) * 2.0 / (sqrt(target->width * target->width + target->height * target->height) + sqrt(rect->width * rect->width + rect->height * rect->height));
					if (d < center_diff_thr)
					{
						total_recall += 1.0;
						total_precision += 1.0;
						tG[j] = tD[k] = 1;
					}
				}
			}
		}
		int* many = (int*)ccmalloc(sizeof(int) * ccv_max(words->rnum, truth->rnum));
		// one to many match, starts with ground truth
		for (j = 0; j < truth->rnum; j++)
		{
			if (tG[j] || cG[j] <= 1)
				continue;
			double one_sum = 0;
			int no_many = 0;
			for (k = 0; k < words->rnum; k++)
			{
				if (tD[k])
					continue;
				double many_single = mD[k * truth->rnum + j];
				if (many_single >= one_d)
				{
					one_sum += mG[j * words->rnum + k];
					many[no_many] = k;
					++no_many;
				}
			}
			if (no_many == 1)
			{
				// degrade to one to one match
				if (mG[j * words->rnum + many[0]] >= one_g && mD[many[0] * truth->rnum + j] >= one_d)
				{
					total_recall += 1.0;
					total_precision += 1.0;
					tG[j] = tD[many[0]] = 1;
				}
			} else if (one_sum >= one_g) {
				for (k = 0; k < no_many; k++)
					tD[many[k]] = 1;
				total_recall += om_one;
				total_precision += om_one * no_many;
			}
		}
		// one to many match, with estimate
		for (k = 0; k < words->rnum; k++)
		{
			if (tD[k] || cD[k] <= 1)
				continue;
			double one_sum = 0;
			int no_many = 0;
			for (j = 0; j < truth->rnum; j++)
			{
				if (tG[j])
					continue;
				double many_single = mG[j * words->rnum + k];
				if (many_single >= one_g)
				{
					one_sum += mD[k * truth->rnum + j];
					many[no_many] = j;
					++no_many;
				}
			}
			if (no_many == 1)
			{
				// degrade to one to one match
				if (mG[many[0] * words->rnum + k] >= one_g && mD[k * truth->rnum + many[0]] >= one_d)
				{
					total_recall += 1.0;
					total_precision += 1.0;
					tG[many[0]] = tD[k] = 1;
				}
			} else if (one_sum >= one_g) {
				for (j = 0; j < no_many; j++)
					tG[many[j]] = 1;
				total_recall += om_one * no_many;
				total_precision += om_one;
			}
		}
		ccv_array_free(words);
		ccfree(many);
		ccfree(tG);
		ccfree(tD);
		ccfree(cG);
		ccfree(cD);
		ccfree(mG);
		ccfree(mD);
	}
	total_precision /= total_detected;
	total_recall /= total_truth;
	if (precision)
		*precision = total_precision;
	if (recall)
		*recall = total_recall;
	return 1.0 / (a / total_precision + (1.0 - a) / total_recall);
}

typedef struct {
	double min_value;
	double max_value;
	double step;
	int enable;
} ccv_swt_range_t;

static void decode_range(const char* arg, ccv_swt_range_t* range)
{
}

int main(int argc, char** argv)
{
	static struct option swt_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* optional parameters */
		{"size", 1, 0, 0},
		{"low-thresh", 1, 0, 0},
		{"high-thresh", 1, 0, 0},
		{"max-height", 1, 0, 0},
		{"min-height", 1, 0, 0},
		{"min-area", 1, 0, 0},
		{"aspect-ratio", 1, 0, 0},
		{"std-ratio", 1, 0, 0},
		{"thickness-ratio", 1, 0, 0},
		{"height-ratio", 1, 0, 0},
		{"intensity-thresh", 1, 0, 0},
		{"letter-occlude-thresh", 1, 0, 0},
		{"distance-ratio", 1, 0, 0},
		{"intersect-ratio", 1, 0, 0},
		{"letter-thresh", 1, 0, 0},
		{"elongate-ratio", 1, 0, 0},
		{"breakdown-ratio", 1, 0, 0},
		{"breakdown", 1, 0, 0},
		{"iterations", 1, 0, 0},
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	if (argc <= 1)
		exit_with_help();
	ccv_swt_param_t params = {
		.interval = 0,
		.same_word_thresh = { 0.5, 0.9 },
		.min_neighbors = 0,
		.scale_invariant = 0,
		.size = 3,
		.low_thresh = 65,
		.high_thresh = 212,
		.max_height = 500,
		.min_height = 14,
		.min_area = 94,
		.letter_occlude_thresh = 4,
		.aspect_ratio = 9,
		.std_ratio = 0.95,
		.thickness_ratio = 1.7,
		.height_ratio = 1.8,
		.intensity_thresh = 34,
		.distance_ratio = 3.3,
		.intersect_ratio = 1.2,
		.letter_thresh = 4,
		.elongate_ratio = 1.9,
		.breakdown = 1,
		.breakdown_ratio = 0.82,
	};
	ccv_swt_range_t size_range = {
		.min_value = 1,
		.max_value = 3,
		.step = 2,
		.enable = 1,
	};
	ccv_swt_range_t low_thresh_range = {
		.min_value = 50,
		.max_value = 150,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t high_thresh_range = {
		.min_value = 200,
		.max_value = 350,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t max_height_range = {
		.min_value = 500,
		.max_value = 500,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t min_height_range = {
		.min_value = 5,
		.max_value = 30,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t min_area_range = {
		.min_value = 10,
		.max_value = 100,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t letter_occlude_thresh_range = {
		.min_value = 0,
		.max_value = 5,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t aspect_ratio_range = {
		.min_value = 5,
		.max_value = 15,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t std_ratio_range = {
		.min_value = 0.1,
		.max_value = 1.0,
		.step = 0.01,
		.enable = 1,
	};
	ccv_swt_range_t thickness_ratio_range = {
		.min_value = 1.0,
		.max_value = 2.0,
		.step = 0.1,
		.enable = 1,
	};
	ccv_swt_range_t height_ratio_range = {
		.min_value = 1.0,
		.max_value = 3.0,
		.step = 0.1,
		.enable = 1,
	};
	ccv_swt_range_t intensity_thresh_range = {
		.min_value = 1,
		.max_value = 50,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t distance_ratio_range = {
		.min_value = 1.0,
		.max_value = 5.0,
		.step = 0.1,
		.enable = 1,
	};
	ccv_swt_range_t intersect_ratio_range = {
		.min_value = 0.0,
		.max_value = 5.0,
		.step = 0.1,
		.enable = 1,
	};
	ccv_swt_range_t letter_thresh_range = {
		.min_value = 0,
		.max_value = 5,
		.step = 1,
		.enable = 1,
	};
	ccv_swt_range_t elongate_ratio_range = {
		.min_value = 0.1,
		.max_value = 2.5,
		.step = 0.1,
		.enable = 1,
	};
	ccv_swt_range_t breakdown_ratio_range = {
		.min_value = 0.5,
		.max_value = 1.5,
		.step = 0.01,
		.enable = 1,
	};
	int i, j, k, iterations = 1; // 10;
	while (getopt_long_only(argc - 1, argv + 1, "", swt_options, &k) != -1)
	{
		switch (k)
		{
			case 0:
				exit_with_help();
			case 1:
				decode_range(optarg, &size_range);
				break;
			case 2:
				decode_range(optarg, &low_thresh_range);
				break;
			case 3:
				decode_range(optarg, &high_thresh_range);
				break;
			case 4:
				decode_range(optarg, &max_height_range);
				break;
			case 5:
				decode_range(optarg, &min_height_range);
				break;
			case 6:
				decode_range(optarg, &min_area_range);
				break;
			case 7:
				decode_range(optarg, &aspect_ratio_range);
				break;
			case 8:
				decode_range(optarg, &std_ratio_range);
				break;
			case 9:
				decode_range(optarg, &thickness_ratio_range);
				break;
			case 10:
				decode_range(optarg, &height_ratio_range);
				break;
			case 11:
				decode_range(optarg, &intensity_thresh_range);
				break;
			case 12:
				decode_range(optarg, &letter_occlude_thresh_range);
				break;
			case 13:
				decode_range(optarg, &distance_ratio_range);
				break;
			case 14:
				decode_range(optarg, &intersect_ratio_range);
				break;
			case 15:
				decode_range(optarg, &letter_thresh_range);
				break;
			case 16:
				decode_range(optarg, &elongate_ratio_range);
				break;
			case 17:
				decode_range(optarg, &breakdown_ratio_range);
				break;
			case 18:
				params.breakdown = !!atoi(optarg);
				break;
			case 19:
				iterations = atoi(optarg);
				break;
			case 20:
				chdir(optarg);
				break;
		}
	}
	FILE* r = fopen(argv[1], "rt");
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
	double best_f = 0, best_precision = 0, best_recall = 0;
	double a = 0.5;
	double v;
	ccv_swt_param_t best_params = params;
#define optimize(parameter, type, rounding) \
	if (parameter##_range.enable) \
	{ \
		params = best_params; \
		int total_iterations = 0; \
		for (v = parameter##_range.min_value; v <= parameter##_range.max_value; v += parameter##_range.step) \
			++total_iterations; \
		ccv_array_t** detected = (ccv_array_t**)alloca(sizeof(ccv_array_t*) * total_iterations); \
		for (v = parameter##_range.min_value, j = 0; v <= parameter##_range.max_value; v += parameter##_range.step, j++) \
			detected[j] = ccv_array_new(sizeof(ccv_array_t*), 64, 0); \
		for (j = 0; j < aof->rnum; j++) \
		{ \
			char* name = *(char**)ccv_array_get(aof, j); \
			ccv_dense_matrix_t* image = 0; \
			ccv_read(name, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE); \
			for (v = parameter##_range.min_value, k = 0; v <= parameter##_range.max_value; v += parameter##_range.step, k++) \
			{ \
				params.parameter = (type)(v + rounding); \
				ccv_array_t* words = ccv_swt_detect_words(image, params); \
				ccv_array_push(detected[k], &words); \
				FLUSH("perform SWT on %s (%d / %d) for " #parameter " = (%lg <- [%lg, %lg])", name, j + 1, aof->rnum, v, parameter##_range.min_value, parameter##_range.max_value); \
			} \
			ccv_matrix_free(image); \
		} \
		for (v = parameter##_range.min_value, j = 0; v <= parameter##_range.max_value; v += parameter##_range.step, j++) \
		{ \
			params.parameter = (type)(v + rounding); \
			double f, recall, precision; \
			f = _ccv_evaluate_wolf(detected[j], aow, a, params, &precision, &recall); \
			ccv_array_free(detected[j]); \
			if (f > best_f) \
			{ \
				best_params = params; \
				best_f = f; \
				best_precision = precision; \
				best_recall = recall; \
			} \
			FLUSH("current harmonic mean : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; best harmonic mean : %.2lf%%, precision : %.2lf%%, recall : %.2lf%% ; at " #parameter " = %lg (%lg <- [%lg, %lg])", f * 100, precision * 100, recall * 100, best_f * 100, best_precision * 100, best_recall * 100, (double)best_params.parameter, v, parameter##_range.min_value, parameter##_range.max_value); \
		} \
		printf("\n"); \
	}
	for (i = 0; i < iterations; i++)
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
		printf("At iteration %d(of %d) : best parameters for swt is:\n"
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
			   i + 1, iterations,
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
