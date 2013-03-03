#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

void exit_with_help()
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    icfcreate [OPTION...]\n\n"
	);
	exit(-1);
}

int main(int argc, char** argv)
{
	static struct option icf_options[] = {
		/* help */
		{"help", 0, 0, 0},
		/* required parameters */
		{"positive-list", 1, 0, 0},
		{"background-list", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		{"negative-count", 1, 0, 0},
		{"positive-count", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* background_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	int negative_count = 0;
	int positive_count = 0;
	ccv_icf_param_t detector = { .min_neighbors = 0, .flags = 0, .threshold = 0.0 };
	ccv_icf_new_param_t params = {
		.detector = detector,
	};
	int i, k;
	while (getopt_long_only(argc, argv, "", icf_options, &k) != -1)
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
				positive_count = atoi(optarg);
				break;
			case 6:
				base_dir = optarg;
				break;
		}
	}
	assert(positive_list != 0);
	assert(background_list != 0);
	assert(working_dir != 0);
	assert(negative_count > 0);
	ccv_enable_cache(512 * 1024 * 1024);
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
	FILE* r1 = fopen(background_list, "r");
	assert(r1 && "background-list doesn't exists");
	char* file = (char*)malloc(1024);
	ccv_decimal_pose_t pose;
	ccv_array_t* posfiles = ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	// roll pitch yaw
	while (fscanf(r0, "%s %f %f %f %f %f %f %f", file, &pose.x, &pose.y, &pose.a, &pose.b, &pose.roll, &pose.pitch, &pose.yaw) != EOF)
	{
		ccv_file_info_t file_info;
		file_info.filename = (char*)ccmalloc(1024);
		if (base_dir != 0)
		{
			strncpy(file_info.filename, base_dir, 1024);
			file_info.filename[dirlen - 1] = '/';
		}
		strncpy(file_info.filename + dirlen, file, 1024 - dirlen);
		file_info.pose = pose;
		ccv_array_push(posfiles, &file_info);
	}
	fclose(r0);
	size_t len = 1024;
	ssize_t read;
	ccv_array_t* bgfiles = (ccv_array_t*)ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
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
		ccv_array_push(bgfiles, &file_info);
	}
	fclose(r1);
	free(file);
	params.interval = 5;
	params.size.width = 20;
	params.size.height = 60;
	params.deform_shift = 0.01;
	params.deform_angle = 1;
	params.deform_scale = 0.1;
	params.C = 0.002;
	params.feature_size = 15000;
	params.select_feature_size = 1500;
	ccv_icf_multiscale_classifier_cascade_t* classifier = ccv_icf_classifier_cascade_new(posfiles, positive_count, bgfiles, negative_count, working_dir, params);
	ccv_icf_write_classifier_cascade(classifier, working_dir);
	for (i = 0; i < posfiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, i);
		free(file_info->filename);
	}
	ccv_array_free(posfiles);
	for (i = 0; i < bgfiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(bgfiles, i);
		free(file_info->filename);
	}
	ccv_array_free(bgfiles);
	ccv_disable_cache();
	return 0;
}
