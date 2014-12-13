#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

static void exit_with_help(void)
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    icfcreate [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --positive-list : text file contains a list of positive files in format:\n"
	"                      <file name> center-x center-y horizontal-axis-length vertical-axis-length object-roll object-pitch object-yaw \\newline\n"
	"    --background-list : text file contains a list of image files that don't contain any target objects\n"
	"    --negative-count : the number of negative examples we should collect from background files for boosting\n"
	"    --working-dir : the directory to save progress and produce result model\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	);
	exit(0);
}

int main(int argc, char** argv)
{
	static struct option scd_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"positive-list", 1, 0, 0},
		{"background-list", 1, 0, 0},
		{"negative-count", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		/* optional parameters */
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* background_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	int negative_count = 0;
	int k;
	while (getopt_long_only(argc, argv, "", scd_options, &k) != -1)
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
				negative_count = atoi(optarg);
				break;
			case 4:
				working_dir = optarg;
				break;
			case 5:
				base_dir = optarg;
		}
	}
	assert(positive_list != 0);
	assert(background_list != 0);
	assert(working_dir != 0);
	assert(negative_count > 0);
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
	FILE* r1 = fopen(background_list, "r");
	assert(r1 && "background-list doesn't exists");
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* posfiles = ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
	char* file = (char*)malloc(1024);
	size_t len = 1024;
	ssize_t read;
	while ((read = getline(&file, &len, r0)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		ccv_file_info_t file_info;
		file_info.filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(file_info.filename, base_dir, 1024);
			file_info.filename[dirlen - 1] = '/';
		}
		strncpy(file_info.filename + dirlen, file, 1024 - dirlen);
		ccv_array_push(posfiles, &file_info);
	}
	fclose(r0);
	ccv_array_t* hard_mine = (ccv_array_t*)ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
	while ((read = getline(&file, &len, r1)) != -1)
	{
		while(read > 1 && isspace(file[read - 1]))
			read--;
		file[read] = 0;
		ccv_file_info_t file_info;
		file_info.filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(file_info.filename, base_dir, 1024);
			file_info.filename[dirlen - 1] = '/';
		}
		strncpy(file_info.filename + dirlen, file, 1024 - dirlen);
		ccv_array_push(hard_mine, &file_info);
	}
	fclose(r1);
	free(file);
	ccv_scd_train_param_t params = {
		.boosting = 10,
		.size = ccv_size(48, 48),
		.feature = {
			.base = ccv_size(8, 8),
			.range_through = 4,
			.step_through = 4,
		},
		.stop_criteria = {
			.hit_rate = 0.995,
			.false_positive_rate = 0.5,
			.accu_false_positive_rate = 1e-7,
			.auc_crit = 1e-5,
			.maximum_feature = 2048,
			.prune_stage = 3,
			.prune_feature = 4,
		},
		.weight_trimming = 0.98,
		.C = 0.0005,
		.grayscale = 0,
	};
	ccv_scd_classifier_cascade_t* cascade = ccv_scd_classifier_cascade_new(posfiles, hard_mine, negative_count, working_dir, params);
	ccv_scd_classifier_cascade_write(cascade, working_dir);
	return 0;
}
