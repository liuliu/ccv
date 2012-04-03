#include "ccv.h"
#include "ccv_internal.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}

static int _ccv_is_equal(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.25 + 0.5);

	return r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

static int _ccv_is_equal_same_class(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.25 + 0.5);

	return r2->id == r1->id &&
		   r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
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
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	memset(pyr, 0, (scale_upto + next * 2) * sizeof(ccv_dense_matrix_t*));
	pyr[next] = a;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[next], &pyr[next + i], 0, (int)(pyr[next]->rows / pow(scale, i)), (int)(pyr[next]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next; i++)
		ccv_sample_down(pyr[i], &pyr[i + next], 0, 0, 0);
	ccv_dense_matrix_t* hog = 0;
	ccv_hog(pyr[next], &hog, 0, 9, 8);
	pyr[next] = hog;
	/* a more efficient way to generate up-scaled hog (using smaller size) */
	for (i = 0; i < next; i++)
	{
		hog = 0;
		ccv_hog(pyr[i + next], &hog, 0, 9, 4 /* this is */);
		pyr[i] = hog;
	}
	for (i = next + 1; i < scale_upto + next * 2; i++)
	{
		hog = 0;
		ccv_hog(pyr[i], &hog, 0, 9, 8);
		ccv_matrix_free(pyr[i]);
		pyr[i] = hog;
	}
	ccv_array_t* idx_seq;
	ccv_array_t* seq = ccv_array_new(64, sizeof(ccv_comp_t));
	ccv_array_t* seq2 = ccv_array_new(64, sizeof(ccv_comp_t));
	ccv_array_t* result_seq = ccv_array_new(64, sizeof(ccv_comp_t));
	for (c = 0; c < count; c++)
	{
		ccv_dpm_mixture_model_t* model = _model[c];
		double scale_x = 1.0;
		double scale_y = 1.0;
		for (i = next; i < scale_upto + next * 2; i++)
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
							ccv_comp_t comp;
							comp.rect = ccv_rect((int)((x - rww) * 8 * scale_x + 0.5), (int)((y - rwh) * 8 * scale_y + 0.5), (int)(root->root.w->cols * 8 * scale_x + 0.5), (int)(root->root.w->rows * 8 * scale_y + 0.5));
							comp.id = c;
							comp.neighbors = 1;
							comp.confidence = f_ptr[x] + root->beta;
							ccv_array_push(seq, &comp);
						}
					f_ptr += root->root.feature->cols;
				}
				for (k = 0; k < root->count; k++)
					ccv_matrix_free(root->part[k].feature);
				ccv_matrix_free(root->root.feature);
			}
			scale_x *= scale;
			scale_y *= scale;
		}
		/* the following code from OpenCV's haar feature implementation */
		if(params.min_neighbors == 0)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(seq, i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			idx_seq = 0;
			ccv_array_clear(seq2);
			// group retrieved rectangles in order to filter out noise
			int ncomp = ccv_array_group(seq, &idx_seq, _ccv_is_equal_same_class, 0);
			ccv_comp_t* comps = (ccv_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_comp_t));
			memset(comps, 0, (ncomp + 1) * sizeof(ccv_comp_t));

			// count number of neighbors
			for(i = 0; i < seq->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq, i);
				int idx = *(int*)ccv_array_get(idx_seq, i);

				if (comps[idx].neighbors == 0)
					comps[idx].confidence = r1.confidence;

				++comps[idx].neighbors;

				comps[idx].rect.x += r1.rect.x;
				comps[idx].rect.y += r1.rect.y;
				comps[idx].rect.width += r1.rect.width;
				comps[idx].rect.height += r1.rect.height;
				comps[idx].id = r1.id;
				comps[idx].confidence = ccv_max(comps[idx].confidence, r1.confidence);
			}

			// calculate average bounding box
			for(i = 0; i < ncomp; i++)
			{
				int n = comps[i].neighbors;
				if(n >= params.min_neighbors)
				{
					ccv_comp_t comp;
					comp.rect.x = (comps[i].rect.x * 2 + n) / (2 * n);
					comp.rect.y = (comps[i].rect.y * 2 + n) / (2 * n);
					comp.rect.width = (comps[i].rect.width * 2 + n) / (2 * n);
					comp.rect.height = (comps[i].rect.height * 2 + n) / (2 * n);
					comp.neighbors = comps[i].neighbors;
					comp.id = comps[i].id;
					comp.confidence = comps[i].confidence;
					ccv_array_push(seq2, &comp);
				}
			}

			// filter out small face rectangles inside large face rectangles
			for(i = 0; i < seq2->rnum; i++)
			{
				ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(seq2, i);
				int flag = 1;

				for(j = 0; j < seq2->rnum; j++)
				{
					ccv_comp_t r2 = *(ccv_comp_t*)ccv_array_get(seq2, j);
					int distance = (int)(r2.rect.width * 0.25 + 0.5);

					if(i != j &&
					   r1.id == r2.id &&
					   r1.rect.x >= r2.rect.x - distance &&
					   r1.rect.y >= r2.rect.y - distance &&
					   r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
					   r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
					   (r2.neighbors > ccv_max(3, r1.neighbors) || r1.neighbors < 3))
					{
						flag = 0;
						break;
					}
				}

				if(flag)
					ccv_array_push(result_seq, &r1);
			}
			ccv_array_free(idx_seq);
			ccfree(comps);
		}
	}

	for (i = 0; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);

	ccv_array_free(seq);
	ccv_array_free(seq2);

	ccv_array_t* result_seq2;
	/* the following code from OpenCV's haar feature implementation */
	if (params.flags & CCV_DPM_NO_NESTED)
	{
		result_seq2 = ccv_array_new(64, sizeof(ccv_comp_t));
		idx_seq = 0;
		// group retrieved rectangles in order to filter out noise
		int ncomp = ccv_array_group(result_seq, &idx_seq, _ccv_is_equal, 0);
		ccv_comp_t* comps = (ccv_comp_t*)ccmalloc((ncomp + 1) * sizeof(ccv_comp_t));
		memset(comps, 0, (ncomp + 1) * sizeof(ccv_comp_t));

		// count number of neighbors
		for(i = 0; i < result_seq->rnum; i++)
		{
			ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(result_seq, i);
			int idx = *(int*)ccv_array_get(idx_seq, i);

			if (comps[idx].neighbors == 0 || comps[idx].confidence < r1.confidence)
			{
				comps[idx].confidence = r1.confidence;
				comps[idx].neighbors = 1;
				comps[idx].rect = r1.rect;
				comps[idx].id = r1.id;
			}
		}

		// calculate average bounding box
		for(i = 0; i < ncomp; i++)
			if(comps[i].neighbors)
				ccv_array_push(result_seq2, &comps[i]);

		ccv_array_free(result_seq);
		ccfree(comps);
	} else {
		result_seq2 = result_seq;
	}

	return result_seq2;
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
