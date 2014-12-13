#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

static void exit_with_help(void)
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    dpmcreate [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --positive-list : text file contains a list of positive files in format:\n"
	"                      <file name> x y width height \\newline\n"
	"    --background-list : text file contains a list of image files that don't contain any target objects\n"
	"    --negative-count : the number of negative examples we should collect from background files to initialize SVM\n"
	"    --model-component : the number of root filters in our mixture model\n"
	"    --model-part : the number of part filters for each root filter\n"
	"    --working-dir : the directory to save progress and produce result model\n"
	"    --symmetric : 0 or 1, whether to exploit symmetric property of the object\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	"    --iterations : how many iterations are needed for stochastic gradient descent [DEFAULT TO 1000]\n"
	"    --root-relabels : how many relabel procedures are needed for root model optimization [DEFAULT TO 20]\n"
	"    --data-minings : how many data mining procedures are needed for discovering hard examples [DEFAULT TO 50]\n"
	"    --relabels : how many relabel procedures are needed for part model optimization [DEFAULT TO 10]\n"
	"    --alpha : the step size for stochastic gradient descent [DEFAULT TO 0.01]\n"
	"    --alpha-ratio : decrease the step size for each iteration [DEFAULT TO 0.995]\n"
	"    --margin-c : the famous C in SVM [DEFAULT TO 0.002]\n"
	"    --balance : to balance the weight of positive examples and negative examples [DEFAULT TO 1.5]\n"
	"    --negative-cache-size : the cache size for negative examples it should be smaller than negative-count and larger than 100 [DEFAULT TO 2000]\n"
	"    --include-overlap : the percentage of overlap between expected bounding box and the bounding box from detection. Beyond this threshold, it is ensured to be the same object [DEFAULT TO 0.7]\n"
	"    --grayscale : 0 or 1, whether to exploit color in a given image [DEFAULT TO 0]\n"
	"    --discard-estimating-constant : 0 or 1, when estimating bounding boxes, discarding constant (which may be accumulated error) [DEFAULT TO 1]\n"
	"    --percentile-breakdown : 0.00 - 1.00, the percentile use for breakdown threshold [DEFAULT TO 0.05]\n\n"
	);
	exit(-1);
}

int main(int argc, char** argv)
{
	static struct option dpm_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"positive-list", 1, 0, 0},
		{"background-list", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		{"negative-count", 1, 0, 0},
		{"model-component", 1, 0, 0},
		{"model-part", 1, 0, 0},
		{"symmetric", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{"iterations", 1, 0, 0},
		{"root-relabels", 1, 0, 0},
		{"data-minings", 1, 0, 0},
		{"relabels", 1, 0, 0},
		{"alpha", 1, 0, 0},
		{"alpha-ratio", 1, 0, 0},
		{"balance", 1, 0, 0},
		{"negative-cache-size", 1, 0, 0},
		{"margin-c", 1, 0, 0},
		{"percentile-breakdown", 1, 0, 0},
		{"include-overlap", 1, 0, 0},
		{"grayscale", 1, 0, 0},
		{"discard-estimating-constant", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* background_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	int negative_count = 0;
	ccv_dpm_param_t detector = { .interval = 8, .min_neighbors = 0, .flags = 0, .threshold = 0.0 };
	ccv_dpm_new_param_t params = {
		.components = 0,
		.detector = detector,
		.parts = 0,
		.min_area = 3000,
		.max_area = 5000,
		.symmetric = 1,
		.alpha = 0.01,
		.balance = 1.5,
		.alpha_ratio = 0.995,
		.iterations = 1000,
		.data_minings = 50,
		.root_relabels = 20,
		.relabels = 10,
		.negative_cache_size = 2000,
		.C = 0.002,
		.percentile_breakdown = 0.05,
		.include_overlap = 0.7,
		.grayscale = 0,
		.discard_estimating_constant = 1,
	};
	int i, k;
	while (getopt_long_only(argc, argv, "", dpm_options, &k) != -1)
	{
		switch (k)
		{
			case 0:
				exit_with_help();
			case 1:
				positive_list = optarg;
				break;
			case 2:
				background_list = optarg;
				break;
			case 3:
				working_dir = optarg;
				break;
			case 4:
				negative_count = atoi(optarg);
				break;
			case 5:
				params.components = atoi(optarg);
				break;
			case 6:
				params.parts = atoi(optarg);
				break;
			case 7:
				params.symmetric = !!atoi(optarg);
				break;
			case 8:
				base_dir = optarg;
				break;
			case 9:
				params.iterations = atoi(optarg);
				break;
			case 10:
				params.root_relabels = atoi(optarg);
				break;
			case 11:
				params.data_minings = atoi(optarg);
			case 12:
				params.relabels = atoi(optarg);
				break;
			case 13:
				params.alpha = atof(optarg);
				break;
			case 14:
				params.alpha_ratio = atof(optarg);
				break;
			case 15:
				params.balance = atof(optarg);
				break;
			case 16:
				params.negative_cache_size = atoi(optarg);
				break;
			case 17:
				params.C = atof(optarg);
				break;
			case 18:
				params.percentile_breakdown = atof(optarg);
				break;
			case 19:
				params.include_overlap = atof(optarg);
				break;
			case 20:
				params.grayscale = !!atoi(optarg);
				break;
			case 21:
				params.discard_estimating_constant = !!atoi(optarg);
				break;
		}
	}
	assert(positive_list != 0);
	assert(background_list != 0);
	assert(working_dir != 0);
	assert(negative_count > 0);
	assert(params.components > 0);
	assert(params.parts > 0);
	ccv_enable_cache(512 * 1024 * 1024);
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
	FILE* r1 = fopen(background_list, "r");
	assert(r1 && "background-list doesn't exists");
	char* file = (char*)malloc(1024);
	int x, y, width, height;
	int capacity = 32, size = 0;
	char** posfiles = (char**)ccmalloc(sizeof(char*) * capacity);
	ccv_rect_t* bboxes = (ccv_rect_t*)ccmalloc(sizeof(ccv_rect_t) * capacity);
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	while (fscanf(r0, "%s %d %d %d %d", file, &x, &y, &width, &height) != EOF)
	{
		posfiles[size] = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(posfiles[size], base_dir, 1024);
			posfiles[size][dirlen - 1] = '/';
		}
		strncpy(posfiles[size] + dirlen, file, 1024 - dirlen);
		bboxes[size] = ccv_rect(x, y, width, height);
		++size;
		if (size >= capacity)
		{
			capacity *= 2;
			posfiles = (char**)ccrealloc(posfiles, sizeof(char*) * capacity);
			bboxes = (ccv_rect_t*)ccrealloc(bboxes, sizeof(ccv_rect_t) * capacity);
		}
	}
	int posnum = size;
	fclose(r0);
	size_t len = 1024;
	ssize_t read;
	capacity = 32, size = 0;
	char** bgfiles = (char**)ccmalloc(sizeof(char*) * capacity);
	while ((read = getline(&file, &len, r1)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		bgfiles[size] = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(bgfiles[size], base_dir, 1024);
			bgfiles[size][dirlen - 1] = '/';
		}
		strncpy(bgfiles[size] + dirlen, file, 1024 - dirlen);
		++size;
		if (size >= capacity)
		{
			capacity *= 2;
			bgfiles = (char**)ccrealloc(bgfiles, sizeof(char*) * capacity);
		}
	}
	fclose(r1);
	int bgnum = size;
	free(file);
	ccv_dpm_mixture_model_new(posfiles, bboxes, posnum, bgfiles, bgnum, negative_count, working_dir, params);
	for (i = 0; i < posnum; i++)
		free(posfiles[i]);
	ccfree(posfiles);
	ccfree(bboxes);
	for (i = 0; i < bgnum; i++)
		free(bgfiles[i]);
	ccfree(bgfiles);
	ccv_disable_cache();
	return 0;
}
