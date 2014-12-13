#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

static void exit_with_help(void)
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    icfoptimize [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --positive-list : text file contains a list of positive files in format:\n"
	"                      <file name> center-x center-y horizontal-axis-length vertical-axis-length object-roll object-pitch object-yaw \\newline\n"
	"    --acceptance : what percentage of positive examples that we should accept for soft cascading\n"
	"    --classifier-cascade : the model file that we will compute soft cascading thresholds on\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n\n"
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
		{"classifier-cascade", 1, 0, 0},
		{"acceptance", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* classifier_cascade = 0;
	char* base_dir = 0;
	double acceptance = 0;
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
				classifier_cascade = optarg;
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
	assert(classifier_cascade != 0);
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
	ccv_icf_classifier_cascade_t* cascade = ccv_icf_read_classifier_cascade(classifier_cascade);
	assert(cascade && "classifier cascade doesn't exists");
	ccv_icf_classifier_cascade_soft(cascade, posfiles, acceptance);
	ccv_icf_write_classifier_cascade(cascade, classifier_cascade);
	for (i = 0; i < posfiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(posfiles, i);
		free(file_info->filename);
	}
	ccv_array_free(posfiles);
	ccv_disable_cache();
	return 0;
}
