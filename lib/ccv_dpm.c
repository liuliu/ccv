#include "ccv.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}

// this is specific HOG computation for dpm, I may later change it to ccv_hog
static void __ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size)
{
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.fl;
	float* mgp = mg->data.fl;
	int i, j;
	ccv_dense_matrix_t* bn = ccv_dense_matrix_new(a->rows / size, a->cols / size * 108, CCV_32F | CCV_C1, 0, 0);
	float* bnp = bn->data.fl;
	memset(bnp, 0, bn->rows * bn->step);
	for (i = 0; i < a->rows; i++)
	{
		bnp = bn->data.fl + i / size * bn->cols;
		for (j = 0; j < a->cols / size; j++)
		{
			bnp[j * 108] = 0;
		}
		agp += a->cols;
		mgp += a->cols;
	}
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
	if (params.size.height != _classifier[0]->size.height || params.size.width != _classifier[0]->size.width)
		ccv_resample(a, &pyr[0], 0, a->rows * _classifier[0]->size.height / params.size.height, a->cols * _classifier[0]->size.width / params.size.width, CCV_INTER_AREA);
	else
		pyr[0] = a;
	int i, j, k, t, x, y, q;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	if (params.size.height != _classifier[0]->size.height || params.size.width != _classifier[0]->size.width)
		ccv_matrix_free(pyr[0]);
	for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);
}
