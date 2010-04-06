#include "ccv.h"
#include <string.h>

void write_code(ccv_sgf_classifier_cascade_t* cascade)
{

	printf("ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade()\n"
		   "{\n"
		   "	ccv_sgf_classifier_cascade_t* cascade = (ccv_sgf_classifier_cascade_t*)mlloc(sizeof(ccv_sgf_classifier_cascade_t));\n"
		   "	cascade->count = %d;\n"
		   "	cascade->size = ccv_size(%d, %d);\n"
		   "	cascade->stage_classifier = (ccv_sgf_stage_classifier_t*)malloc(cascade->count * sizeof(ccv_sgf_stage_classifier_t));\n",
			cascade->count, cascade->size.width, cascade->size.height);
	int i, j, k;
	for (i = 0; i < cascade->count; i++)
	{
		printf("	{\n");
		printf("		ccv_sgf_stage_classifier_t* classifier = cascade->stage_classifier + %d;\n", i);
		printf("		classifier->count = %d;\n"
			   "		classifier->threshold = %f;\n",
			   cascade->stage_classifier[i].count, cascade->stage_classifier[i].threshold);
		printf("		classifier->feature = (ccv_sgf_feature_t*)malloc(classifier->count * sizeof(ccv_sgf_feature_t));\n"
			   "		classifier->alpha = (float*)malloc(classifier->count * 2 * sizeof(float));\n");
		for (j = 0; j < cascade->stage_classifier[i].count; j++)
		{
			printf("		classifier->feature[%d].size = %d;\n",
				   j, cascade->stage_classifier[i].feature[j].size);
			for (k = 0; k < cascade->stage_classifier[i].feature[j].size; k++)
			{
				printf("		classifier->feature[%d].px[%d] = %d;\n"
					   "		classifier->feature[%d].py[%d] = %d;\n"
					   "		classifier->feature[%d].pz[%d] = %d;\n",
					   j, k, cascade->stage_classifier[i].feature[j].px[k],
					   j, k, cascade->stage_classifier[i].feature[j].py[k],
					   j, k, cascade->stage_classifier[i].feature[j].pz[k]);
				printf("		classifier->feature[%d].nx[%d] = %d;\n"
					   "		classifier->feature[%d].ny[%d] = %d;\n"
					   "		classifier->feature[%d].nz[%d] = %d;\n",
					   j, k, cascade->stage_classifier[i].feature[j].nx[k],
					   j, k, cascade->stage_classifier[i].feature[j].ny[k],
					   j, k, cascade->stage_classifier[i].feature[j].nz[k]);
			}
			printf("		classifier->alpha[%d] = %f;\n"
				   "		classifier->alpha[%d] = %f;\n",
				   j * 2, cascade->stage_classifier[i].alpha[j * 2], j * 2 + 1, cascade->stage_classifier[i].alpha[j * 2 + 1]);
		}
		printf("	}\n");
	}
	printf("	return cascade;\n}");
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	ccv_sgf_classifier_cascade_t* cascade = ccv_load_sgf_classifier_cascade(argv[1]);
	if (strcmp(argv[2], "bin") == 0)
	{
		assert(argc >= 4);
		int len = ccv_sgf_classifier_cascade_write_binary(cascade, NULL, 0);
		char* s = malloc(len);
		ccv_sgf_classifier_cascade_write_binary(cascade, s, len);
		FILE* w = fopen(argv[3], "w");
		fwrite(s, 1, len, w);
		fclose(w);
		free(s);
	} else if (strcmp(argv[2], "code") == 0) {
		write_code(cascade);
	} else {
		int len = ccv_sgf_classifier_cascade_write_binary(cascade, NULL, 0);
		char* s = malloc(len);
		ccv_sgf_classifier_cascade_write_binary(cascade, s, len);
		int i;
		for (i = 0; i < len; i++)
			printf("\\x%x", (unsigned char)s[i]);
		fflush(NULL);
		free(s);
	}
	return 0;
}
