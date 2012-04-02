#include "ccv.h"
#include "ccv_internal.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}

ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_mixture_model_t** _model, int count, ccv_dpm_param_t params)
{
	int c, i, j, k, x, y;
	ccv_size_t size = ccv_size(a->cols, a->rows);
	for (c = 0; c < count; c++)
	{
		ccv_dpm_mixture_model_t* model = _model[c];
		for (i = 0; i < model->count; i++)
		{
			size.width = ccv_min(model->root[i].root.w->cols * 8, size.width);
			size.height = ccv_min(model->root[i].root.w->rows * 8, size.height);
		}
	}
	int hr = a->rows / size.height;
	int wr = a->cols / size.width;
	double scale = pow(2., 1. / (params.interval + 1.));
	int next = params.interval + 1;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(scale)) - next;
	if (scale_upto < 1) // image is too small to be interesting
		return 0;
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next) * sizeof(ccv_dense_matrix_t*));
	pyr[0] = a;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	ccv_dense_matrix_t* hog = 0;
	ccv_hog(pyr[0], &hog, 0, 9, 8);
	pyr[0] = hog;
	for (i = 1; i < scale_upto + next; i++)
	{
		hog = 0;
		ccv_hog(pyr[i], &hog, 0, 9, 8);
		ccv_matrix_free(pyr[i]);
		pyr[i] = hog;
	}
	for (c = 0; c < count; c++)
	{
		ccv_dpm_mixture_model_t* model = _model[c];
		double scale_x = 1.0;
		double scale_y = 1.0;
		for (i = next; i < scale_upto + next; i++)
		{
			for (j = 0; j < model->count; j++)
			{
				ccv_dpm_root_classifier_t* root = model->root + j;
				ccv_dense_matrix_t* response = 0;
				ccv_filter(pyr[i], root->root.w, &response, 0, CCV_NO_PADDING);
				root->root.feature = 0;
				ccv_flatten(response, (ccv_matrix_t**)&root->root.feature, 0, 0);
				ccv_matrix_free(response);
				int rwh = root->root.w->rows / 2;
				int rww = root->root.w->cols / 2;
				for (k = 0; k < root->count; k++)
				{
					ccv_dpm_part_classifier_t* part = root->part + k;
					ccv_dense_matrix_t* response = 0;
					ccv_filter(pyr[i - next], part->w, &response, 0, CCV_NO_PADDING);
					ccv_dense_matrix_t* feature = 0;
					ccv_flatten(response, (ccv_matrix_t**)&feature, 0, 0);
					ccv_matrix_free(response);
					part->feature = 0;
					ccv_distance_transform(feature, &part->feature, 0, part->dx, part->dy, part->dxx, part->dyy, CCV_NEGATE | CCV_GSEDT);
					ccv_matrix_free(feature);
					int offy = part->y + part->w->rows / 2 - rwh * 2;
					int miny = part->w->rows / 2, maxy = part->feature->rows - part->w->rows / 2;
					int offx = part->x + part->w->cols / 2 - rww * 2;
					int minx = part->w->cols / 2, maxx = part->feature->cols - part->w->cols / 2;
					double* f_ptr = root->root.feature->data.f64 + root->root.feature->cols * rwh;
					for (y = rwh; y < root->root.feature->rows - rwh; y++)
					{
						int iy = ccv_clamp(y * 2 + offy, miny, maxy);
						for (x = rww; x < root->root.feature->cols - rww; x++)
						{
							int ix = ccv_clamp(x * 2 + offx, minx, maxx);
							f_ptr[x] -= part->feature->data.f64[iy * part->feature->cols + ix];
						}
						f_ptr += root->root.feature->cols;
					}
				}
				double* f_ptr = root->root.feature->data.f64 + root->root.feature->cols * rwh;
				for (y = rwh; y < root->root.feature->rows - rwh; y++)
				{
					for (x = rww; x < root->root.feature->cols - rww; x++)
						if (f_ptr[x] + root->beta > params.threshold)
						{
							printf("%lf at %d %d\n", f_ptr[x], (int)(x * 8 * 2 * scale_x), (int)(y * 8 * 2 * scale_y));
						}
					f_ptr += root->root.feature->cols;
				}
				/*
				response = 0;
				ccv_slice(root->root.feature, (ccv_matrix_t**)&response, 0, 7, 2, root->root.feature->rows - 14, root->root.feature->cols - 4);
				ccv_dense_matrix_t* visual = 0;
				ccv_visualize(response, &visual, 0);
				ccv_matrix_free(response);
				ccv_write(visual, "root.png", 0, CCV_IO_PNG_FILE, 0);
				ccv_matrix_free(visual);
				*/
				printf("finish level %d\n", i - next);
				for (k = 0; k < root->count; k++)
					ccv_matrix_free(root->part[k].feature);
				ccv_matrix_free(root->root.feature);
			}
			scale_x *= scale;
			scale_y *= scale;
		}
	}
	for (i = 0; i < scale_upto + next; i++)
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
		fscanf(r, "%lf", &root_classifier[i].beta);
		root_classifier[i].root.w = ccv_dense_matrix_new(rows, cols, CCV_64F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 31)), 0);
		size += ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 31);
		for (j = 0; j < rows * cols * 31; j++)
			fscanf(r, "%lf", &root_classifier[i].root.w->data.f64[j]);
		root_classifier[i].root.w->sig = ccv_matrix_generate_signature((char*)root_classifier[i].root.w->data.u8, root_classifier[i].root.w->rows * root_classifier[i].root.w->step, 0);
		fscanf(r, "%d", &root_classifier[i].count);
		ccv_dpm_part_classifier_t* part_classifier = (ccv_dpm_part_classifier_t*)ccmalloc(sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count);
		size += sizeof(ccv_dpm_part_classifier_t) * root_classifier[i].count;
		for (j = 0; j < root_classifier[i].count; j++)
		{
			fscanf(r, "%d %d %d", &part_classifier[j].x, &part_classifier[j].y, &part_classifier[j].z);
			fscanf(r, "%lf %lf %lf %lf", &part_classifier[j].dx, &part_classifier[j].dy, &part_classifier[j].dxx, &part_classifier[j].dyy);
			fscanf(r, "%d %d", &rows, &cols);
			part_classifier[j].w = ccv_dense_matrix_new(rows, cols, CCV_64F | 31, ccmalloc(ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 31)), 0);
			size += ccv_compute_dense_matrix_size(rows, cols, CCV_64F | 31);
			for (k = 0; k < rows * cols * 31; k++)
				fscanf(r, "%lf", &part_classifier[j].w->data.f64[k]);
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
	for (i = 0; i < model->count; i++)
	{
		ccv_dpm_part_classifier_t* part_classifier = model->root[i].part;
		model->root[i].part = (ccv_dpm_part_classifier_t*)m;
		m += sizeof(ccv_dpm_part_classifier_t) * model->root[i].count;
		memcpy(model->root[i].part, part_classifier, sizeof(ccv_dpm_part_classifier_t) * model->root[i].count);
	}
	for (i = 0; i < model->count; i++)
	{
		ccv_dense_matrix_t* w = model->root[i].root.w;
		model->root[i].root.w = (ccv_dense_matrix_t*)m;
		m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
		memcpy(model->root[i].root.w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
		model->root[i].root.w->data.u8 = (unsigned char*)(model->root[i].root.w + 1);
		ccfree(w);
		for (j = 0; j < model->root[i].count; j++)
		{
			w = model->root[i].part[j].w;
			model->root[i].part[j].w = (ccv_dense_matrix_t*)m;
			m += ccv_compute_dense_matrix_size(w->rows, w->cols, w->type);
			memcpy(model->root[i].part[j].w, w, ccv_compute_dense_matrix_size(w->rows, w->cols, w->type));
			model->root[i].part[j].w->data.u8 = (unsigned char*)(model->root[i].part[j].w + 1);
			ccfree(w);
		}
	}
	fclose(r);
	return model;
}
