#include "ccv.h"
#include <ctype.h>

void exit_with_help()
{
	printf(
	"USAGE: dpmcreate positive_set_file background_set_file negative_number model_components model_parts output [change_to_directory]\n"
	"DESCRIPTION:\n"
	"- positive_set_file: text file contains a list of positive files in format:\n"
	"-                    <file name> x y width height \\newline\n"
	"- background_set_file: text file contains a list of image files that don't contain any target objects\n"
	"- negative_number: the number of negative examples we should collect from background files to initialize SVM\n"
	"- model_components: the number of root filters in our mixture model\n"
	"- model_parts: the number of part filters for each root filter\n"
	"- output: the output model file\n"
	"- change_to_directory: change the base directory so that the program can read images from there\n"
	);
	exit(-1);
}

int main(int argc, char** argv)
{
	if (argc != 7 && argc != 8)
		exit_with_help();
	ccv_enable_cache(512 * 1024 * 1024);
	FILE* r0 = fopen(argv[1], "r");
	FILE* r1 = fopen(argv[2], "r");
	char* file = (char*)malloc(1024);
	int x, y, width, height;
	int capacity = 32, size = 0;
	char** posfiles = (char**)ccmalloc(sizeof(char*) * capacity);
	ccv_rect_t* bboxes = (ccv_rect_t*)ccmalloc(sizeof(ccv_rect_t) * capacity);
	int dirlen = (argc == 8) ? strlen(argv[7]) + 1 : 0;
	while (fscanf(r0, "%s %d %d %d %d", file, &x, &y, &width, &height) != EOF)
	{
		posfiles[size] = (char*)ccmalloc(1024);
		if (argc == 8)
		{
			strncpy(posfiles[size], argv[7], 1024);
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
		if (argc == 8)
		{
			strncpy(bgfiles[size], argv[7], 1024);
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
	int negnum = atoi(argv[3]);
	int components = atoi(argv[4]);
	int parts = atoi(argv[5]);
	free(file);
	ccv_dpm_param_t detector = { .interval = 8, .min_neighbors = 0, .flags = 0, .threshold = 0.0 };
	ccv_dpm_new_param_t params = { .components = components,
								   .detector = detector,
								   .parts = parts,
								   .min_area = 3000,
								   .max_area = 5000,
								   .symmetric = 1,
								   .alpha = 0.1,
								   .balance = 1.75,
								   .alpha_ratio = 0.9,
								   .iterations = 15,
								   .relabels = 5,
								   .C = 0.002,
								   .percentile_breakdown = 0.05,
								   .overlap = 0.75,
								   .grayscale = 0 };
	ccv_dpm_mixture_model_new(posfiles, bboxes, posnum, bgfiles, bgnum, negnum, argv[6], params);
	free(posfiles);
	free(bboxes);
	free(bgfiles);
	ccv_disable_cache();
	return 0;
}
