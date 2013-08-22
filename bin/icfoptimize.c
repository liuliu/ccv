#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

void exit_with_help()
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    icfoptimize [OPTION...]\n\n"
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
		{"working-dir", 1, 0, 0},
		{"acceptance", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	double acceptance = 0;
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
				working_dir = optarg;
				break;
			case 3:
				acceptance = atof(optarg);
				break;
			case 4:
				base_dir = optarg;
				break;
		}
	}
	assert(positive_list != 0);
	assert(working_dir != 0);
	ccv_enable_cache(512 * 1024 * 1024);
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
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
		// blow up pose a little bit for INRIA data (16px on four strides)
		file_info.pose = pose;
		ccv_array_push(posfiles, &file_info);
	}
	fclose(r0);
	free(file);
	params.grayscale = 0;
	params.margin = ccv_margin(5, 5, 5, 5);
	params.size = ccv_size(20, 60);
	params.deform_shift = 0;
	params.deform_angle = 0;
	params.deform_scale = 0;
	params.feature_size = 50000;
	params.weak_classifier = 2000;
	params.acceptance = acceptance;
	ccv_icf_classifier_cascade_t* cascade = ccv_icf_read_classifier_cascade(working_dir);
	ccv_icf_classifier_cascade_soft(cascade, posfiles, working_dir, params);
	ccv_icf_write_classifier_cascade(cascade, working_dir);
	for (i = 0; i < posfiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, i);
		free(file_info->filename);
	}
	ccv_array_free(posfiles);
	ccv_disable_cache();
	return 0;
}
