#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

static void exit_with_help(void)
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    bbfcreate [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --positive-list : text file contains a list of positive files (cropped and scaled to the same size)\n"
	"    --background-list : text file contains a list of image files that don't contain any target objects\n"
	"    --negative-count : the number of negative examples we should collect from background files to initialize SVM\n"
	"    --working-dir : the directory to save progress and produce result model\n"
	"    --width : the width of positive image\n"
	"    --height : the height of positive image\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	"    --layer : how many layers needed for cascade classifier [DEFAULT TO 24]\n"
	"    --positive-criteria : what's the percentage of positive examples need to pass for the next layer [DEFAULT TO 0.9975]\n"
	"    --negative-criteria : what's the percentage of negative examples need to reject for the next layer [DEFAULT TO 0.5]\n"
	"    --balance : the balance weight for positive examples v.s. negative examples [DEFAULT TO 1.0]\n"
	"    --feature-number : how big our feature pool should be [DEFAULT TO 100 (thus, 100 * 100 = 10000 features)]\n\n"
	);
	exit(-1);
}

int main(int argc, char** argv)
{
	static struct option bbf_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"positive-list", 1, 0, 0},
		{"background-list", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		{"negative-count", 1, 0, 0},
		{"width", 1, 0, 0},
		{"height", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{"layer", 1, 0, 0},
		{"positive-criteria", 1, 0, 0},
		{"negative-criteria", 1, 0, 0},
		{"balance", 1, 0, 0},
		{"feature-number", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* background_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	int negnum = 0;
	int width = 0, height = 0;
	ccv_bbf_new_param_t params = {
		.pos_crit = 0.9975,
		.neg_crit = 0.50,
		.balance_k = 1.0,
		.layer = 24,
		.feature_number = 100,
		.optimizer = CCV_BBF_GENETIC_OPT | CCV_BBF_FLOAT_OPT,
	};
	int i, k;
	while (getopt_long_only(argc, argv, "", bbf_options, &k) != -1)
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
				negnum = atoi(optarg);
				break;
			case 5:
				width = atoi(optarg);
				break;
			case 6:
				height = atoi(optarg);
				break;
			case 7:
				base_dir = optarg;
				break;
			case 8:
				params.layer = atoi(optarg);
				break;
			case 9:
				params.pos_crit = atof(optarg);
				break;
			case 10:
				params.neg_crit = atof(optarg);
				break;
			case 11:
				params.balance_k = atof(optarg);
				break;
			case 12:
				params.feature_number = atoi(optarg);
				break;
		}
	}
	assert(positive_list != 0);
	assert(background_list != 0);
	assert(working_dir != 0);
	assert(negnum > 0);
	assert(width > 0 && height > 0);
	ccv_enable_default_cache();
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
	FILE* r1 = fopen(background_list, "r");
	assert(r1 && "background-list doesn't exists");
	char* file = (char*)malloc(1024);
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	size_t len = 1024;
	ssize_t read;
	int capacity = 32, size = 0;
	ccv_dense_matrix_t** posimg = (ccv_dense_matrix_t**)ccmalloc(sizeof(ccv_dense_matrix_t*) * capacity);
	while ((read = getline(&file, &len, r0)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		char* posfile = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(posfile, base_dir, 1024);
			posfile[dirlen - 1] = '/';
		}
		strncpy(posfile + dirlen, file, 1024 - dirlen);
		posimg[size] = 0;
		ccv_read(posfile, &posimg[size], CCV_IO_GRAY | CCV_IO_ANY_FILE);
		if (posimg != 0)
		{
			++size;
			if (size >= capacity)
			{
				capacity *= 2;
				posimg = (ccv_dense_matrix_t**)ccrealloc(posimg, sizeof(ccv_dense_matrix_t*) * capacity);
			}
		}
	}
	fclose(r0);
	int posnum = size;
	capacity = 32;
	size = 0;
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
	ccv_bbf_classifier_cascade_new(posimg, posnum, bgfiles, bgnum, negnum, ccv_size(width, height), working_dir, params);
	for (i = 0; i < bgnum; i++)
		free(bgfiles[i]);
	for (i = 0; i < posnum; i++)
		ccv_matrix_free(&posimg[i]);
	free(posimg);
	free(bgfiles);
	ccv_disable_cache();
	return 0;
}
