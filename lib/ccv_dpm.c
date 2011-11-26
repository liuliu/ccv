#include "ccv.h"

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params)
{
}

// this is specific HOG computation for dpm, I may later change it to ccv_hog
static void _ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size)
{
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.fl;
	float* mgp = mg->data.fl;
	int i, j, k;
	int rows = a->rows / size;
	int cols = a->cols / size;
	ccv_dense_matrix_t* cn = ccv_dense_matrix_new(rows, cols * sbin * 2, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* ca = ccv_dense_matrix_new(rows, cols, CCV_64F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_new(rows, cols * (4 + sbin * 3), CCV_32F | CCV_C1, 0, 0);
	ccv_zero(cn);
	float* cnp = cn->data.fl;
	int sizec = 0;
	for (i = 0; i < rows * size; i++)
	{
		for (j = 0; j < cols; j++)
		{
			for (k = j * size; k < j * size + size; k++)
			{
				int ag0, ag1;
				float agr;
				agr = (agp[k] / 360.0f) * (sbin * 2);
				ag0 = (int)agr;
				ag1 = (ag0 + 1 < sbin * 2) ? ag0 + 1 : 0;
				agr = agr - ag0;
				cnp[ag0] += (1.0f - agr) * mgp[k] / 255.0f;
				cnp[ag1] += agr * mgp[k] / 255.0f;
			}
			cnp += 2 * sbin;
		}
		agp += a->cols;
		mgp += a->cols;
		if (++sizec < size)
			cnp -= cn->cols;
		else
			sizec = 0;
	}
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
	cnp = cn->data.fl;
	double* cap = ca->data.db;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			*cap = 0;
			for (k = 0; k < sbin * 2; k++)
			{
				*cap += (*cnp) * (*cnp);
				cnp++;
			}
			cap++;
		}
	}
	cnp = cn->data.fl;
	cap = ca->data.db;
	ccv_zero(db);
	float* dbp = db->data.fl;
	// normalize sbin direction-sensitive and sbin * 2 insensitive over 4 normalization factor
	// accumulating them over sbin * 2 + sbin + 4 channels
	// TNA - truncation - normalization - accumulation
#define TNA(idx, a, b, c, d) \
	{ \
		float norm = 1.0f / sqrt(cap[a] + cap[b] + cap[c] + cap[d] + 1e-4f); \
		for (k = 0; k < sbin * 2; k++) \
		{ \
			float v = 0.5f * ccv_min(cnp[k] * norm, 0.2f); \
			dbp[sbin + k] += v; \
			dbp[sbin * 3 + idx] += v; \
		} \
		dbp[sbin * 3 + idx] *= 0.2357f; \
		for (k = 0; k < sbin; k++) \
		{ \
			float v = 0.5f * ccv_min((cnp[k] + cnp[k + sbin]) * norm, 0.2f); \
			dbp[k] += v; \
		} \
	}
	TNA(0, 0, 0, 0, 0);
	TNA(1, 1, 1, 0, 0);
	TNA(2, 0, ca->cols, ca->cols, 0);
	TNA(3, 1, ca->cols + 1, ca->cols, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (j = 1; j < cols - 1; j++)
	{
		TNA(0, -1, -1, 0, 0);
		TNA(1, 1, 1, 0, 0);
		TNA(2, -1, ca->cols - 1, ca->cols, 0);
		TNA(3, 1, ca->cols + 1, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, -1, -1, 0, 0);
	TNA(1, 0, 0, 0, 0);
	TNA(2, -1, ca->cols - 1, ca->cols, 0);
	TNA(3, 0, ca->cols, ca->cols, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (i = 1; i < rows - 1; i++)
	{
		TNA(0, 0, -ca->cols, -ca->cols, 0);
		TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
		TNA(2, 0, ca->cols, ca->cols, 0);
		TNA(3, 1, ca->cols + 1, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
		for (j = 1; j < cols - 1; j++)
		{
			TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
			TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
			TNA(2, -1, ca->cols - 1, ca->cols, 0);
			TNA(3, 1, ca->cols + 1, ca->cols, 0);
			cnp += 2 * sbin;
			dbp += 3 * sbin + 4;
			cap++;
		}
		TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
		TNA(1, 0, -ca->cols, -ca->cols, 0);
		TNA(2, -1, ca->cols - 1, ca->cols, 0);
		TNA(3, 0, ca->cols, ca->cols, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, 0, -ca->cols, -ca->cols, 0);
	TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
	TNA(2, 0, 0, 0, 0);
	TNA(3, 1, 1, 0, 0);
	cnp += 2 * sbin;
	dbp += 3 * sbin + 4;
	cap++;
	for (j = 1; j < cols - 1; j++)
	{
		TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
		TNA(1, 1, -ca->cols + 1, -ca->cols, 0);
		TNA(2, -1, -1, 0, 0);
		TNA(3, 1, 1, 0, 0);
		cnp += 2 * sbin;
		dbp += 3 * sbin + 4;
		cap++;
	}
	TNA(0, -1, -ca->cols - 1, -ca->cols, 0);
	TNA(1, 0, -ca->cols, -ca->cols, 0);
	TNA(2, -1, -1, 0, 0);
	TNA(3, 0, 0, 0, 0);
#undef TNA
	ccv_matrix_free(cn);
	ccv_matrix_free(ca);
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
	if (params.size.height != _classifier[0]->root.size.height || params.size.width != _classifier[0]->root.size.width)
		ccv_resample(a, &pyr[0], 0, a->rows * _classifier[0]->root.size.height / params.size.height, a->cols * _classifier[0]->root.size.width / params.size.width, CCV_INTER_AREA);
	else
		pyr[0] = a;
	int i, j;
	for (i = 1; i <= params.interval; i++)
		ccv_resample(pyr[0], &pyr[i], 0, (int)(pyr[0]->rows / pow(scale, i)), (int)(pyr[0]->cols / pow(scale, i)), CCV_INTER_AREA);
	for (i = next; i < scale_upto + next * 2; i++)
		ccv_sample_down(pyr[i - next], &pyr[i], 0, 0, 0);
	/*
	for (i = 0; i < scale_upto + next * 2; i++)
	{
		ccv_dense_matrix_t* hog = 0;
		_ccv_hog(pyr[i], &hog, 0, 9, 8);
		ccv_matrix_free(hog);
	}
	*/
	ccv_dense_matrix_t* hog = 0;
	_ccv_hog(pyr[0], &hog, 0, 9, 8);
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(pyr[0]->rows, pyr[0]->cols, CCV_8U | CCV_C1, 0, 0);
	unsigned char* bptr = b->data.ptr;
	for (i = 0; i < b->rows; i++)
	{
		if (i >= hog->rows * 8)
			break;
		for (j = 0; j < b->cols; j++)
		{
			int k = (i / 8) * hog->cols + (j / 8) * 31 + 2;
			bptr[j] = ccv_clamp(100 * hog->data.fl[k], 0, 255);
		}
		bptr += b->step;
	}
	ccv_serialize(b, "hog.png", 0, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(hog);
	ccv_matrix_free(b);
	if (params.size.height != _classifier[0]->root.size.height || params.size.width != _classifier[0]->root.size.width)
		ccv_matrix_free(pyr[0]);
	 for (i = 1; i < scale_upto + next * 2; i++)
		ccv_matrix_free(pyr[i]);
	return 0;
}

ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory)
{
}
