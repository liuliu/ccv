#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

void exit_with_help()
{
	printf(
	"\n  \033[1mUSAGE\033[0m\n\n    icfcreate [OPTION...]\n\n"
	"  \033[1mREQUIRED OPTIONS\033[0m\n\n"
	"    --positive-list : text file contains a list of positive files in format:\n"
	"                      <file name> center-x center-y horizontal-axis-length vertical-axis-length object-roll object-pitch object-yaw \\newline\n"
	"    --positive-count : the number of positive examples we should collect from positive files with certain distortion\n"
	"    --validate-list : text file contains a list of positive files in following format but only used for soft cascading:\n"
	"                      <file name> center-x center-y horizontal-axis-length vertical-axis-length object-roll object-pitch object-yaw \\newline\n"
	"    --acceptance : what percentage of validate examples that we should accept for soft cascading\n"
	"    --background-list : text file contains a list of image files that don't contain any target objects\n"
	"    --negative-count : the number of negative examples we should collect from background files for boosting\n"
	"    --size : size of object in pixel formatted as WxH\n"
	"    --margin : margin for object when extracting from given images, formatted as left,top,right,bottom\n"
	"    --feature-size : the number of features that we randomly generates and later pooling from\n"
	"    --weak-classifier-count : the number of weak classifiers in the boosted model\n"
	"    --working-dir : the directory to save progress and produce result model\n\n"
	"  \033[1mOTHER OPTIONS\033[0m\n\n"
	"    --base-dir : change the base directory so that the program can read images from there\n"
	"    --grayscale : 0 or 1, whether to exploit color in a given image [DEFAULT TO 0]\n"
	"    --deform-shift : [DEFAULT TO 1]\n"
	"    --deform-angle : [DEFAULT TO 0]\n"
	"    --deform-scale : [DEFAULT TO 0.075]\n"
	"    --min-dimension : [DEFAULT TO 2]\n"
	"    --bootstrap : [DEFAULT TO 3]\n\n"
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
		{"validate-list", 1, 0, 0},
		{"working-dir", 1, 0, 0},
		{"negative-count", 1, 0, 0},
		{"positive-count", 1, 0, 0},
		{"acceptance", 1, 0, 0},
		{"size", 1, 0, 0},
		{"margin", 1, 0, 0},
		{"feature-size", 1, 0, 0},
		{"weak-classifier-count", 1, 0, 0},
		/* optional parameters */
		{"base-dir", 1, 0, 0},
		{0, 0, 0, 0}
	};
	char* positive_list = 0;
	char* background_list = 0;
	char* test_list = 0;
	char* working_dir = 0;
	char* base_dir = 0;
	int negative_count = 0;
	int positive_count = 0;
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
				background_list = optarg;
				break;
			case 3:
				test_list = optarg;
				break;
			case 4:
				working_dir = optarg;
				break;
			case 5:
				negative_count = atoi(optarg);
				break;
			case 6:
				positive_count = atoi(optarg);
				break;
			case 7:
				acceptance = atof(optarg);
				break;
			case 8:
				base_dir = optarg;
				break;
		}
	}
	assert(positive_list != 0);
	assert(background_list != 0);
	assert(test_list != 0);
	assert(working_dir != 0);
	assert(negative_count > 0);
	ccv_enable_cache(512 * 1024 * 1024);
	FILE* r0 = fopen(positive_list, "r");
	assert(r0 && "positive-list doesn't exists");
	FILE* r1 = fopen(background_list, "r");
	assert(r1 && "background-list doesn't exists");
	FILE* r2 = fopen(test_list, "r");
	assert(r2 && "validate-list doesn't exists");
	char* file = (char*)malloc(1024);
	ccv_decimal_pose_t pose;
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	ccv_array_t* posfiles = ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
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
	ccv_array_t* validatefiles = ccv_array_new(sizeof(ccv_file_info_t), 32, 0);
	// roll pitch yaw
	while (fscanf(r2, "%s %f %f %f %f %f %f %f", file, &pose.x, &pose.y, &pose.a, &pose.b, &pose.roll, &pose.pitch, &pose.yaw) != EOF)
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
		ccv_array_push(validatefiles, &file_info);
	}
	fclose(r2);
	free(file);
	params.grayscale = 0;
	params.margin = ccv_margin(5, 5, 5, 5);
	params.size = ccv_size(20, 60);
	params.min_dimension = 2;
	params.deform_shift = 1;
	params.deform_angle = 0;
	params.deform_scale = 0.075;
	params.feature_size = 50000;
	params.weak_classifier = 2000;
	params.acceptance = acceptance;
	params.bootstrap = 3;
	params.detector = ccv_icf_default_params;
	params.detector.step_through = 4; // for faster negatives bootstrap time
	ccv_icf_classifier_cascade_t* classifier = ccv_icf_classifier_cascade_new(posfiles, positive_count, bgfiles, negative_count, validatefiles, working_dir, params);
	char filename[1024];
	snprintf(filename, 1024, "%s/final-cascade", working_dir);
	ccv_icf_write_classifier_cascade(classifier, filename);
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
	for (i = 0; i < validatefiles->rnum; i++)
	{
		ccv_file_info_t* file_info = (ccv_file_info_t*)ccv_array_get(validatefiles, i);
		free(file_info->filename);
	}
	ccv_array_free(validatefiles);
	ccv_disable_cache();
	return 0;
}
