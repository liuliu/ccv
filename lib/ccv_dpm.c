#include "ccv.h"
#include "ccv_internal.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}


ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_root_classifier_t** _classifier, int count, ccv_dpm_param_t params)
{
	int hr = a->rows / params.size.height;
	int wr = a->cols / params.size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale));
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	if (params.size.height != _classifier[0]->root.size.height * 8 || params.size.width != _classifier[0]->root.size.width * 8)
		ccv_resample(a, &pyr[0], 0, a->rows * _classifier[0]->root.size.height * 8 / params.size.height, a->cols * _classifier[0]->root.size.width * 8 / params.size.width, CCV_INTER_AREA);
	else
		pyr[0] = a;
	int i, j;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	for (i = 0; i < scale_upto + next * 2; i++)
	{
		ccv_dense_matrix_t* hog = 0;
		_ccv_hog(pyr[i], &hog, 0, 9, 8);
		ccv_matrix_free(hog);
	}
	if (params.size.height != _classifier[0]->root.size.height * 8 || params.size.width != _classifier[0]->root.size.width * 8)
		ccv_matrix_free(pyr[0]);
	 for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);
	return 0;
}

ccv_dpm_root_classifier_t* ccv_load_dpm_root_classifier(const char* directory)
{
	FILE* r = fopen(directory, "r");
	if (r == 0)
		return 0;
	ccv_dpm_root_classifier_t* root_classifier = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t));
	memset(root_classifier, 0, sizeof(ccv_dpm_root_classifier_t));
	fscanf(r, "%d %d", &root_classifier->root.size.width, &root_classifier->root.size.height);
	root_classifier->root.w = (float*)ccmalloc(sizeof(float) * root_classifier->root.size.width * root_classifier->root.size.height * 32);
	int i, j, k;
	for (i = 0; i < root_classifier->root.size.width * root_classifier->root.size.height * 32; i++)
		fscanf(r, "%f", &root_classifier->root.w[i]);
	/*
	for (j = 0; j < root_classifier->root.size.width * root_classifier->root.size.height; j++)
	{
		i = 31;
		printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 27; i < 31; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 18; i < 27; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
		for (i = 0; i < 18; i++)
			printf("%f ", root_classifier->root.w[i * root_classifier->root.size.width * root_classifier->root.size.height + j]);
	}
	printf("\n");
	*/
	fclose(r);
	return root_classifier;
}
