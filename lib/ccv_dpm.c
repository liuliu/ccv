#include "ccv.h"
#include "ccv_internal.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}


ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_mixture_model_t** _model, int count, ccv_dpm_param_t params)
{
	int i, j;
	for (i = 0; i < count; i++)
	{
	}
	int hr = a->rows; // / params.size.height;
	int wr = a->cols; // / params.size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale));
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	pyr[0] = a;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	for (i = 0; i < scale_upto + next * 2; i++)
	{
		ccv_dense_matrix_t* hog = 0;
		ccv_hog(pyr[i], &hog, 0, 9, 8);
		ccv_matrix_free(hog);
	}
	for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);
	return 0;
}

/* rewind format from matlab
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
ccv_dpm_mixture_model_t* ccv_load_dpm_mixture_model(const char* directory)
{
	FILE* r = fopen(directory, "r");
	if (r == 0)
		return 0;
	int count;
	fscanf(r, "%d", &count);
	ccv_dpm_root_classifier_t* root_classifier = (ccv_dpm_root_classifier_t*)ccmalloc(sizeof(ccv_dpm_root_classifier_t) * count);
	memset(root_classifier, 0, sizeof(ccv_dpm_root_classifier_t) * count);
	int i, j, k;
	size_t size = sizeof(ccv_dpm_mixture_model_t) + sizeof(ccv_dpm_root_classifier_t) * count;
	/* the format is easy, but I tried to copy all data into one memory region */
	for (i = 0; i < count; i++)
	{
		int rows, cols;
		fscanf(r, "%d %d", &rows, &cols);
		root_classifier->root.w = ccv_dense_matrix_new(rows, cols, CCV_64F | 32, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 32)), 0);
		size += ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 32);
		for (j = 0; j < rows * cols * 32; j++)
			fscanf(r, "%lf", &root_classifier[i].root.w->data.f64[j]);
		root_classifier->root.w->sig = ccv_matrix_generate_signature((char*)root_classifier->root.w->data.u8, root_classifier->root.w->rows * root_classifier->root.w->step, 0);
		fscanf(r, "%d", &root_classifier[i].count);
		ccv_dpm_part_classifier_t* part_classifier = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count);
		size += sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count;
		for (j = 0; j < root_classifier[i].count; j++)
		{
			fscanf(r, "%d %d %d", &part_classifier[j].x, &part_classifier[j].y, &part_classifier[j].z);
			fscanf(r, "%lf %lf %lf %lf", &part_classifier[j].dx, &part_classifier[j].dy, &part_classifier[j].dxx, &part_classifier[j].dyy);
			fscanf(r, "%d %d", &rows, &cols);
			part_classifier[j].w = ccv_dense_matrix_new(rows, cols, CCV_64F | 32, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 32)), 0);
			size += ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 32);
			for (k = 0; k < rows * cols * 32; j++)
				fscanf(r, "%lf", &part_classifier[j].w->data.f64[j]);
			part_classifier[j].w->sig = ccv_matrix_generate_signature((char*)part_classifier[j].w->data.u8, part_classifier[j].w->rows * part_classifier[j].w->step, 0);
		}
		root_classifier[i].part = part_classifier;
	}
	unsigned char* m = (unsigned char*)ccmalloc(size);
	ccv_dpm_mixture_model_t* model = (ccv_dpm_mixture_model_t*)m;
	m += sizeof(ccv_dpm_mixture_model_t);
	model->count = count;
	model->root = (ccv_dpm_root_classifier_t*)m;
	m += sizeof(ccv_dpm_root_classifier_t) * model->count;
	memcpy(model->root, root_classifier, sizeof(ccv_dpm_root_classifier_t) * model->count);
	ccfree(root_classifier);
	for (i = 0; i < model->count; i++)
	{
		ccv_dpm_part_classifier_t* part_classifier = model->root[i].part;
		model->root[i].part = (ccv_dpm_part_classifier_t*)m;
		m += sizeof(ccv_dpm_part_classifier_t) * model->root[i].count;
		memcpy(model->root[i].part, part_classifier, sizeof(ccv_dpm_part_classifier_t) * model->root[i].count);
		ccfree(part_classifier);
	}
	for (i = 0; i < model->count; i++)
	{
		ccv_dense_matrix_t* w = model->root[i].root.w;
		model->root[i].root.w = (ccv_dense_matrix_t*)m;
		m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
		memcpy(model->root[i].root.w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
		ccfree(w);
		for (j = 0; j < model->root[i].count; j++)
		{
			w = model->root[i].part[j].w;
			model->root[i].part[j].w = (ccv_dense_matrix_t*)m;
			m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
			memcpy(model->root[i].part[j].w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
			ccfree(w);
		}
	}
	fclose(r);
	return model;
}
