#include "ccv.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

static inline int __ccv_run_sgf_feature(ccv_sgf_feature_t* feature, int* step, int** i32c8)
{
#define pf_at(i) (*(i32c8[feature->pz[i]] + feature->px[i] + feature->py[i] * step[feature->pz[i]]))
#define nf_at(i) (*(i32c8[feature->nz[i]] + feature->nx[i] + feature->ny[i] * step[feature->nz[i]]))
	int pmin = pf_at(0), nmax = nf_at(0);
	/* check if every point in P > every point in N, and take a shortcut */
	if (pmin <= nmax)
		return 0;
	int i;
	for (i = 1; i < feature->size; i++)
	{
		if (feature->pz[i] >= 0)
		{
			int p = pf_at(i);
			if (p < pmin)
			{
				if (p <= nmax)
					return 0;
				pmin = p;
			}
		}
		if (feature->nz[i] >= 0)
		{
			int n = nf_at(i);
			if (n > nmax)
			{
				if (pmin <= n)
					return 0;
				nmax = n;
			}
		}
	}
#undef pf_at
#undef nf_at
	return 1;
}

#define HOG_BORDER_SIZE (2)

static int __ccv_prepare_background_data(ccv_sgf_classifier_cascade_t* cascade, char** bgfiles, int bgnum, int** negdata, int negnum)
{
	int t, i, j, k;
	int negperbg = negnum / bgnum + 1;
	int negtotal = 0;
	int isizs0 = cascade->size.width * cascade->size.height * 8;
	int isizs1 = ((cascade->size.width >> 1) - HOG_BORDER_SIZE) * ((cascade->size.height >> 1) - HOG_BORDER_SIZE) * 8;
	int steps[] = { cascade->size.width * 8, ((cascade->size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	printf("preparing negative data ...  0%%");
	int* idcheck = (int*)malloc(negnum * sizeof(int));

	gsl_rng_env_setup();

	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(rng, (unsigned long int)idcheck);

	ccv_size_t imgsz = ccv_size(cascade->size.width + HOG_BORDER_SIZE * 2, cascade->size.height + HOG_BORDER_SIZE * 2);
	int rneg = negtotal;
	for (t = 0; negtotal < negnum; t++)
	{
		for (i = 0; i < bgnum; i++)
		{
			negperbg = (t < 2) ? (negnum - negtotal) / (bgnum - i) + 1 : negnum - negtotal;
			ccv_dense_matrix_t* image = NULL;
			ccv_unserialize(bgfiles[i], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
			assert(image->type & CCV_C1);
			if (image == NULL)
			{
				printf("\n%s file corrupted\n", bgfiles[i]);
				continue;
			}
			if (t % 2 != 0)
				ccv_flip(image, NULL, CCV_FLIP_X);
			ccv_array_t* detected = ccv_sgf_detect_objects(image, &cascade, 1, 0, 0, cascade->size);
			for (j = 0; j < ccv_min(detected->rnum, negperbg); j++)
			{
				int r = gsl_rng_uniform_int(rng, detected->rnum);
				int flag = 1;
				ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(detected, r);
				while (flag) {
					flag = 0;
					for (k = 0; k < j; ++k)
						if (r == idcheck[k])
						{
							flag = 1;
							r = gsl_rng_uniform_int(rng, detected->rnum);
							break;
						}
					rect = (ccv_rect_t*)ccv_array_get(detected, r);
					if ((rect->x < 0) || (rect->y < 0) || (rect->width + rect->x >= image->cols) || (rect->height + rect->y >= image->rows))
					{
						flag = 1;
						r = gsl_rng_uniform_int(rng, detected->rnum);
					}
				}
				idcheck[j] = r;
				ccv_dense_matrix_t* temp = NULL;
				ccv_dense_matrix_t* imgs0 = NULL;
				ccv_dense_matrix_t* imgs1 = NULL;
				ccv_slice(image, &temp, rect->y, rect->x, rect->height, rect->width);
				ccv_resample(temp, &imgs0, imgsz.height, imgsz.width, CCV_INTER_AREA);
				ccv_matrix_free(temp);
				ccv_sample_down(imgs0, &imgs1);

				negdata[negtotal] = (int*)malloc((isizs0 + isizs1) * sizeof(int));
				int* i32c8s0 = negdata[negtotal];
				int* i32c8s1 = negdata[negtotal] + isizs0;
				int* i32c8[] = { i32c8s0, i32c8s1 };
				ccv_dense_matrix_t des0 = ccv_dense_matrix(imgs0->rows - HOG_BORDER_SIZE * 2, (imgs0->cols - HOG_BORDER_SIZE * 2) * 8, CCV_32S | CCV_C1, i32c8s0, NULL);
				ccv_dense_matrix_t des1 = ccv_dense_matrix(imgs1->rows - HOG_BORDER_SIZE * 2, (imgs1->cols - HOG_BORDER_SIZE * 2) * 8, CCV_32S | CCV_C1, i32c8s1, NULL);
				ccv_dense_matrix_t* des0p = &des0;
				ccv_dense_matrix_t* des1p = &des1;
				ccv_hog(imgs0, &des0p, HOG_BORDER_SIZE * 2 + 1);
				ccv_hog(imgs1, &des1p, HOG_BORDER_SIZE * 2 + 1);
				ccv_matrix_free(imgs0);
				ccv_matrix_free(imgs1);

	/*
				for ( int y = 0; y < cascade->size.height; ++y )
					for ( int x = 0; x < cascade->size.width; ++x )
						if ( i32c8s0[x * 8 + 1 + y * cascade->size.width * 8] > 255 )
							out8u->data.ptr[x + y * out8u->step] = 255;
						else if ( i32c8s0[x * 8 + 1 + y * cascade->size.width * 8] < 0 )
							out8u->data.ptr[x + y * out8u->step] = 0;
						else
							out8u->data.ptr[x + y * out8u->step] = i32c8s0[x * 8 + 1 + y * cascade->size.width * 8];
				cvShowImage("image", imgs0 );
				cvShowImage("output", out8u);
				cvWaitKey(0);
	*/
				flag = 1;
				ccv_sgf_stage_classifier_t* classifier = cascade->stage_classifier;
				for (k = 0; k < cascade->count; ++k, ++classifier)
				{
					float sum = 0;
					float* alpha = classifier->alpha;
					ccv_sgf_feature_t* feature = classifier->feature;
					for (k = 0; k < classifier->count; ++k, alpha += 2, ++feature)
						sum += alpha[__ccv_run_sgf_feature(feature, steps, i32c8)];
					if (sum < classifier->threshold)
					{
						flag = 0;
						break;
					}
				}
				if (!flag)
					free(negdata[negtotal]);
				else {
					++negtotal;
					if (negtotal >= negnum)
						break;
				}
			}
			ccv_array_free(detected);
			ccv_matrix_free(image);
			ccv_garbage_collect();
			printf("\rpreparing negative data ... %2d%%", 100 * negtotal / negnum);
			fflush(NULL);
			if (negtotal >= negnum)
				break;
		}
		if (rneg == negtotal)
			break;
		rneg = negtotal;
	}
	gsl_rng_free(rng);
	free(idcheck);
	ccv_garbage_collect();
	printf("\n");
	return negtotal;
}

static void __ccv_prepare_positive_data(ccv_dense_matrix_t** posimg, int** posdata, ccv_size_t size, int posnum)
{
	printf("preparing positive data ...  0%%");
	int i;
	for (i = 0; i < posnum; i++)
	{
		ccv_dense_matrix_t* imgs0 = posimg[i];
		ccv_dense_matrix_t* imgs1 = NULL;
		ccv_sample_down(imgs0, &imgs1);
		int isizs0 = size.width * size.height * 8;
		int isizs1 = ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;

		posdata[i] = (int*)malloc((isizs0 + isizs1) * sizeof(int));
		int* i32c8s0 = posdata[i];
		int* i32c8s1 = posdata[i] + isizs0;
		ccv_dense_matrix_t des0 = ccv_dense_matrix(imgs0->rows - HOG_BORDER_SIZE * 2, (imgs0->cols - HOG_BORDER_SIZE * 2) * 8, CCV_32S | CCV_C1, i32c8s0, NULL);
		ccv_dense_matrix_t des1 = ccv_dense_matrix(imgs1->rows - HOG_BORDER_SIZE * 2, (imgs1->cols - HOG_BORDER_SIZE * 2) * 8, CCV_32S | CCV_C1, i32c8s1, NULL);
		ccv_dense_matrix_t* des0p = &des0;
		ccv_dense_matrix_t* des1p = &des1;
		ccv_hog(imgs0, &des0p, HOG_BORDER_SIZE * 2 + 1);
		ccv_hog(imgs1, &des1p, HOG_BORDER_SIZE * 2 + 1);

		printf("\rpreparing positive data ... %2d%%", 100 * (i + 1) / posnum);
		fflush(NULL);
/*
		for ( int y = 0; y < size.height; ++y )
			for ( int x = 0; x < size.width; ++x )
				if ( i32c8s0[x * 8 + 1 + y * size.width * 8] > 255 )
					out8u->data.ptr[x + y * out8u->step] = 255;
				else if ( i32c8s0[x * 8 + 1 + y * size.width * 8] < 0 )
					out8u->data.ptr[x + y * out8u->step] = 0;
				else
					out8u->data.ptr[x + y * out8u->step] = i32c8s0[x * 8 + 1 + y * size.width * 8];
		cvShowImage("image", imgs0 );
		cvShowImage("output", out8u);
		cvWaitKey(0);
*/
		ccv_matrix_free(imgs1);
	}
	ccv_garbage_collect();
	printf("\n");
}

typedef struct {
	double fitness;
	int pk, nk;
	int age;
	double error;
	ccv_sgf_feature_t feature;
} ccv_sgf_gene_t;

static inline void __ccv_sgf_genetic_fitness(ccv_sgf_gene_t* gene)
{
	gene->fitness = (1 - gene->error) * exp(-0.01 * gene->age) * exp((gene->pk + gene->nk) * log(1.015));
}

static inline double __ccv_sgf_error_rate(ccv_sgf_feature_t* feature, int** posdata, int posnum, int** negdata, int negnum, ccv_size_t size, double* pw, double* nw)
{
	int i;
	int isizs0 = size.width * size.height * 8;
	int steps[] = { size.width * 8, ((size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	double error = 0;
	for (i = 0; i < posnum; i++)
	{
		int* i32c8[] = { posdata[i], posdata[i] + isizs0 };
		if (!__ccv_run_sgf_feature(feature, steps, i32c8))
			error += pw[i];
	}
	for (i = 0; i < negnum; i++)
	{
		int* i32c8[] = { negdata[i], negdata[i] + isizs0 };
		if ( __ccv_run_sgf_feature(feature, steps, i32c8))
			error += nw[i];
	}
	return error;
}

#define less_than(a, b, aux) ((a) < (b))
CCV_IMPLEMENT_QSORT(__ccv_sort_32f, float, less_than)
#undef less_than

static void __ccv_sgf_eval_data(ccv_sgf_stage_classifier_t* classifier, int** posdata, int posnum, int** negdata, int negnum, ccv_size_t size, float* peval, float* neval)
{
	int i, j;
	int isizs0 = size.width * size.height * 8;
	int steps[] = { size.width * 8, ((size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	for (i = 0; i < posnum; i++)
	{
		int* i32c8[] = { posdata[i], posdata[i] + isizs0 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_sgf_feature_t* feature = classifier->feature;
		for (j = 0; j < classifier->count; ++j, alpha += 2, ++feature)
			sum += alpha[__ccv_run_sgf_feature(feature, steps, i32c8)];
		peval[i] = sum;
	}
	for (i = 0; i < negnum; i++)
	{
		int* i32c8[] = { negdata[i], negdata[i] + isizs0 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_sgf_feature_t* feature = classifier->feature;
		for (j = 0; j < classifier->count; ++j, alpha += 2, ++feature)
			sum += alpha[__ccv_run_sgf_feature(feature, steps, i32c8)];
		neval[i] = sum;
	}
}

static void __ccv_prune_positive_data(ccv_sgf_classifier_cascade_t* cascade, int** posdata, int* posnum, ccv_size_t size)
{
	float* peval = (float*)malloc(*posnum * sizeof(float));
	int i, j, k;
	for (i = 0; i < cascade->count; i++)
	{
		__ccv_sgf_eval_data(cascade->stage_classifier + i, posdata, *posnum, 0, 0, size, peval, 0);
		k = 0;
		for (j = 0; j < *posnum; j++)
			if (peval[j] >= cascade->stage_classifier[i].threshold)
			{
				posdata[k] = posdata[j];
				++k;
			} else {
				free(posdata[j]);
			}
		*posnum = k;
	}
	free(peval);
}

static inline int __ccv_sgf_exist_gene_feature(ccv_sgf_gene_t* gene, int x, int y, int z)
{
	int i;
	for ( i = 0; i < gene->pk; ++i )
		if ( z == gene->feature.pz[i] && x == gene->feature.px[i] && y == gene->feature.py[i] )
			return 1;
	for ( i = 0; i < gene->nk; ++i )
		if ( z == gene->feature.nz[i] && x == gene->feature.nx[i] && y == gene->feature.ny[i] )
			return 1;
	return 0;
}

#define less_than(fit1, fit2, aux) ((fit1).fitness < (fit2).fitness)
static CCV_IMPLEMENT_QSORT(__ccv_sgf_genetic_qsort, ccv_sgf_gene_t, less_than)
#undef less_than

static inline void __ccv_sgf_randomize_gene(gsl_rng* rng, ccv_sgf_gene_t* gene, int* rows, int* steps)
{
	int i;
	do {
		gene->pk = gsl_rng_uniform_int(rng, CCV_SGF_POINT_MAX - 1) + 1;
		gene->nk = gsl_rng_uniform_int(rng, CCV_SGF_POINT_MAX - 1) + 1;
	} while (gene->pk + gene->nk < CCV_SGF_POINT_MIN); /* a hard restriction of at least 3 points have to be examed */
	gene->feature.size = ccv_max(gene->pk, gene->nk);
	gene->age = 0;
	for (i = 0; i < CCV_SGF_POINT_MAX; i++)
	{
		gene->feature.pz[i] = -1;
		gene->feature.nz[i] = -1;
	}
	int x, y, z;
	for (i = 0; i < gene->pk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 2);
			x = gsl_rng_uniform_int(rng, steps[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while (__ccv_sgf_exist_gene_feature(gene, x, y, z));
		gene->feature.pz[i] = z;
		gene->feature.px[i] = x;
		gene->feature.py[i] = y;
	}
	for (i = 0; i < gene->nk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 2);
			x = gsl_rng_uniform_int(rng, steps[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while ( __ccv_sgf_exist_gene_feature(gene, x, y, z));
		gene->feature.nz[i] = z;
		gene->feature.nx[i] = x;
		gene->feature.ny[i] = y;
	}
}

static ccv_sgf_feature_t __ccv_sgf_genetic_optimize(int** posdata, int posnum, int** negdata, int negnum, int ftnum, ccv_size_t size, double* pw, double* nw)
{
	/* seed (random method) */
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	gsl_rng_set(rng, (unsigned long int)(pw[0] + nw[0]));
	int i, j;
	int pnum = ftnum * 100;
	ccv_sgf_gene_t* gene = (ccv_sgf_gene_t*)malloc(pnum * sizeof(ccv_sgf_gene_t));
	int rows[] = { size.height, (size.height >> 1) - HOG_BORDER_SIZE };
	int steps[] = { size.width * 8, ((size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	for (i = 0; i < pnum; i++)
		__ccv_sgf_randomize_gene(rng, &gene[i], rows, steps);
#ifdef HAVE_OPENMP
#pragma omp parallel for private(i) num_threads(8) schedule(dynamic)
#endif
	for (i = 0; i < pnum; i++)
	{
		gene[i].error = __ccv_sgf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
		__ccv_sgf_genetic_fitness(&gene[i]);
	}
	double best_err = 1;
	ccv_sgf_feature_t best;
	int rnum = ftnum * 39;//99;//49;//39; /* number of randomize */
	int mnum = ftnum * 40;//0;//50;//40; /* number of mutation */
	int hnum = ftnum * 20;//0;//0;//20; /* number of hybrid */
	/* iteration stop crit : best no change in 40 iterations */
	int it = 0, t;
	for (t = 0 ; it < 40; ++it, ++t)
	{
		__ccv_sgf_genetic_qsort(gene, pnum, 0);
		for (i = 0; i < ftnum; i++)
			++gene[i].age;
		for (i = ftnum; i < ftnum + mnum; i++)
		{
			int parent = gsl_rng_uniform_int(rng, ftnum);
			memcpy(gene + i, gene + parent, sizeof(ccv_sgf_gene_t));
			/* three mutation strategy : 1. add, 2. remove, 3. refine */
			int pnm, pn = gsl_rng_uniform_int(rng, 2);
			int* pnk[] = { &gene[i].pk, &gene[i].nk };
			int* pnx[] = { gene[i].feature.px, gene[i].feature.nx };
			int* pny[] = { gene[i].feature.py, gene[i].feature.ny };
			int* pnz[] = { gene[i].feature.pz, gene[i].feature.nz };
			int x, y, z;
			int victim, decay = 1;
			do {
				switch (gsl_rng_uniform_int(rng, 3))
				{
					case 0: /* add */
						if (gene[i].pk == CCV_SGF_POINT_MAX && gene[i].nk == CCV_SGF_POINT_MAX)
							break;
						while (*pnk[pn] + 1 > CCV_SGF_POINT_MAX)
							pn = gsl_rng_uniform_int(rng, 2);
						do {
							z = gsl_rng_uniform_int(rng, 2);
							x = gsl_rng_uniform_int(rng, steps[z]);
							y = gsl_rng_uniform_int(rng, rows[z]);
						} while (__ccv_sgf_exist_gene_feature(&gene[i], x, y, z));
						pnz[pn][*pnk[pn]] = z;
						pnx[pn][*pnk[pn]] = x;
						pny[pn][*pnk[pn]] = y;
						++(*pnk[pn]);
						gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
						decay = gene[i].age = 0;
						break;
					case 1: /* remove */
						if (gene[i].pk + gene[i].nk <= CCV_SGF_POINT_MIN) /* at least 3 points have to be examed */
							break;
						while (*pnk[pn] - 1 <= 0) // || *pnk[pn] + *pnk[!pn] - 1 < CCV_SGF_POINT_MIN)
							pn = gsl_rng_uniform_int(rng, 2);
						victim = gsl_rng_uniform_int(rng, *pnk[pn]);
						for (j = victim; j < *pnk[pn] - 1; j++)
						{
							pnz[pn][j] = pnz[pn][j + 1];
							pnx[pn][j] = pnx[pn][j + 1];
							pny[pn][j] = pny[pn][j + 1];
						}
						pnz[pn][*pnk[pn] - 1] = -1;
						--(*pnk[pn]);
						gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
						decay = gene[i].age = 0;
						break;
					case 2: /* refine */
						pnm = gsl_rng_uniform_int(rng, *pnk[pn]);
						do {
							z = gsl_rng_uniform_int(rng, 2);
							x = gsl_rng_uniform_int(rng, steps[z]);
							y = gsl_rng_uniform_int(rng, rows[z]);
						} while (__ccv_sgf_exist_gene_feature(&gene[i], x, y, z));
						pnz[pn][pnm] = z;
						pnx[pn][pnm] = x;
						pny[pn][pnm] = y;
						decay = gene[i].age = 0;
						break;
				}
			} while (decay);
		}
		for (i = ftnum + mnum; i < ftnum + mnum + hnum; i++)
		{
			/* hybrid strategy: taking positive points from dad, negative points from mum */
			int dad, mum;
			do {
				dad = gsl_rng_uniform_int(rng, ftnum);
				mum = gsl_rng_uniform_int(rng, ftnum);
			} while (dad == mum || gene[dad].pk + gene[mum].nk < CCV_SGF_POINT_MIN); /* at least 3 points have to be examed */
			for (j = 0; j < CCV_SGF_POINT_MAX; j++)
			{
				gene[i].feature.pz[j] = -1;
				gene[i].feature.nz[j] = -1;
			}
			gene[i].pk = gene[dad].pk;
			for (j = 0; j < gene[i].pk; j++)
			{
				gene[i].feature.pz[j] = gene[dad].feature.pz[j];
				gene[i].feature.px[j] = gene[dad].feature.px[j];
				gene[i].feature.py[j] = gene[dad].feature.py[j];
			}
			gene[i].nk = gene[mum].nk;
			for (j = 0; j < gene[i].nk; j++)
			{
				gene[i].feature.nz[j] = gene[mum].feature.nz[j];
				gene[i].feature.nx[j] = gene[mum].feature.nx[j];
				gene[i].feature.ny[j] = gene[mum].feature.ny[j];
			}
			gene[i].feature.size = ccv_max(gene[i].pk, gene[i].nk);
			gene[i].age = 0;
		}
		for (i = ftnum + mnum + hnum; i < ftnum + mnum + hnum + rnum; i++)
			__ccv_sgf_randomize_gene(rng, &gene[i], rows, steps);
#ifdef HAVE_OPENMP
#pragma omp parallel for private(i) num_threads(8) schedule(dynamic)
#endif
		for (i = 0; i < pnum; i++)
		{
			gene[i].error = __ccv_sgf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
			__ccv_sgf_genetic_fitness(&gene[i]);
		}
		int min_id = 0;
		double min_err = gene[0].error;
		for (i = 1; i < pnum; i++)
			if (gene[i].error < min_err)
			{
				min_id = i;
				min_err = gene[i].error;
			}
		if (min_err < best_err)
		{
			best_err = min_err;
			memcpy(&best, &gene[min_id].feature, sizeof(best));
			printf("best sgf feature with error %f\n|-size: %d\n|-positive point: ", best_err, best.size);
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.px[i], best.py[i], best.pz[i]);
			printf("\n|-negative point: ");
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.nx[i], best.ny[i], best.nz[i]);
			printf("\n");
			it = 0;
		}
		printf("minimum error achieved in round %d(%d) : %f\n", t, it, min_err);
	}
	gsl_rng_free(rng);
	return best;
}

int __ccv_read_sgf_stage_classifier(const char* file, ccv_sgf_stage_classifier_t* classifier)
{
	FILE* r = fopen(file, "r");
	if (r == NULL) return -1;
	int stat = 0;
	stat |= fscanf(r, "%d", &classifier->count);
	union { float fl; int i; } fli;
	stat |= fscanf(r, "%d", &fli.i);
	classifier->threshold = fli.fl;
	classifier->feature = (ccv_sgf_feature_t*)malloc(classifier->count * sizeof(ccv_sgf_feature_t));
	classifier->alpha = (float*)malloc(classifier->count * 2 * sizeof(float));
	int i, j;
	for (i = 0; i < classifier->count; i++)
	{
		stat |= fscanf(r, "%d", &classifier->feature[i].size);
		for (j = 0; j < classifier->feature[i].size; j++)
		{
			stat |= fscanf(r, "%d %d %d", &classifier->feature[i].px[j], &classifier->feature[i].py[j], &classifier->feature[i].pz[j]);
			stat |= fscanf(r, "%d %d %d", &classifier->feature[i].nx[j], &classifier->feature[i].ny[j], &classifier->feature[i].nz[j]);
		}
		union { float fl; int i; } flia, flib;
		stat |= fscanf(r, "%d %d", &flia.i, &flib.i);
		classifier->alpha[i * 2] = flia.fl;
		classifier->alpha[i * 2 + 1] = flib.fl;
	}
	fclose(r);
	return 0;
}

int __ccv_write_sgf_stage_classifier(const char* file, ccv_sgf_stage_classifier_t* classifier)
{
	FILE* w = fopen(file, "w");
	if (w == NULL) return -1;
	fprintf(w, "%d\n", classifier->count);
	union { float fl; int i; } fli;
	fli.fl = classifier->threshold;
	fprintf(w, "%d\n", fli.i);
	int i, j;
	for (i = 0; i < classifier->count; i++)
	{
		fprintf(w, "%d\n", classifier->feature[i].size);
		for (j = 0; j < classifier->feature[i].size; j++)
		{
			fprintf(w, "%d %d %d\n", classifier->feature[i].px[j], classifier->feature[i].py[j], classifier->feature[i].pz[j]);
			fprintf(w, "%d %d %d\n", classifier->feature[i].nx[j], classifier->feature[i].ny[j], classifier->feature[i].nz[j]);
		}
		union { float fl; int i; } flia, flib;
		flia.fl = classifier->alpha[i * 2];
		flib.fl = classifier->alpha[i * 2 + 1];
		fprintf(w, "%d %d\n", flia.i, flib.i);
	}
	fclose(w);
	return 0;
}

static int __ccv_read_background_data(const char* file, int** negdata, int* negnum, ccv_size_t size)
{
	int stat = 0;
	FILE* r = fopen(file, "r");
	if (r == NULL) return -1;
	stat |= fscanf(r, "%d", negnum);
	int i, j;
	int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
	for (i = 0; i < *negnum; i++)
	{
		negdata[i] = (int*)malloc(isizs01 * sizeof(int));
		for (j = 0; j < isizs01; j++)
			stat |= fscanf(r, "%d", &negdata[i][j]);
	}
	fclose(r);
	return 0;
}

static int __ccv_write_background_data(const char* file, int** negdata, int negnum, ccv_size_t size)
{
	FILE* w = fopen(file, "w");
	if (w == NULL) return -1;
	fprintf(w, "%d\n", negnum);
	int i, j;
	int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
	for (i = 0; i < negnum; i++)
	{
		for (j = 0; j < isizs01; j++)
			fprintf(w, "%d ", negdata[i][j]);
		fprintf(w, "\n");
	}
	fclose(w);
	return 0;
}

static int __ccv_resume_sgf_cascade_training_state(const char* file, int* i, int* k, int* bg, double* pw, double* nw, int posnum, int negnum)
{
	int stat = 0;
	FILE* r = fopen(file, "r");
	if (r == NULL) return -1;
	stat |= fscanf(r, "%d %d %d", i, k, bg);
	int j;
	union { double db; int i[2]; } dbi;
	for (j = 0; j < posnum; j++)
	{
		stat |= fscanf(r, "%d %d", &dbi.i[0], &dbi.i[1]);
		pw[j] = dbi.db;
	}
	for (j = 0; j < negnum; j++)
	{
		stat |= fscanf(r, "%d %d", &dbi.i[0], &dbi.i[1]);
		nw[j] = dbi.db;
	}
	fclose(r);
	return 0;
}

static int __ccv_save_sgf_cacade_training_state(const char* file, int i, int k, int bg, double* pw, double* nw, int posnum, int negnum)
{
	FILE* w = fopen(file, "w");
	if (w == NULL) return -1;
	fprintf(w, "%d %d %d\n", i, k, bg);
	int j;
	union { double db; int i[2]; } dbi;
	for (j = 0; j < posnum; ++j)
	{
		dbi.db = pw[j];
		fprintf(w, "%d %d ", dbi.i[0], dbi.i[1]);
	}
	fprintf(w, "\n");
	for (j = 0; j < negnum; ++j)
	{
		dbi.db = nw[j];
		fprintf(w, "%d %d ", dbi.i[0], dbi.i[1]);
	}
	fprintf(w, "\n");
	fclose(w);
	return 0;
}

void ccv_sgf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_sgf_param_t params)
{
	int i, j, k;
	/* allocate memory for usage */
	ccv_sgf_classifier_cascade_t* cascade = (ccv_sgf_classifier_cascade_t*)malloc(sizeof(ccv_sgf_classifier_cascade_t));
	cascade->count = 0;
	cascade->size = size;
	cascade->stage_classifier = (ccv_sgf_stage_classifier_t*)malloc(sizeof(ccv_sgf_stage_classifier_t));
	int** posdata = (int**)malloc(posnum * sizeof(int*));
	__ccv_prepare_positive_data(posimg, posdata, cascade->size, posnum);
	int** negdata = (int**)malloc(negnum * sizeof(int*));

	double* pw = (double*)malloc(posnum * sizeof(double));
	double* nw = (double*)malloc(negnum * sizeof(double));
	float* peval = (float*)malloc(posnum * sizeof(float));
	float* neval = (float*)malloc(negnum * sizeof(float));
	double inv_balance_k = 1. / params.balance_k;
	/* balance factor k, and weighted with 0.01 */
	params.balance_k *= 0.01;
	inv_balance_k *= 0.01;

	int isizs0 = cascade->size.width * cascade->size.height * 8;
	int steps[] = { cascade->size.width * 8, ((cascade->size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	
	i = 0;
	k = 0;
	int bg = 0;
	int cacheK = 10;
	/* state resume code */
	char buf[1024];
	sprintf(buf, "%s/stat.txt", dir);
	__ccv_resume_sgf_cascade_training_state(buf, &i, &k, &bg, pw, nw, posnum, negnum);
	if (i > 0)
	{
		cascade->count = i;
		free(cascade->stage_classifier);
		cascade->stage_classifier = (ccv_sgf_stage_classifier_t*)malloc(i * sizeof(ccv_sgf_stage_classifier_t));
		for (j = 0; j < i; j++)
		{
			sprintf(buf, "%s/stage-%d.txt", dir, j);
			__ccv_read_sgf_stage_classifier(buf, &cascade->stage_classifier[j]);
		}
	}
	if (k > 0)
		cacheK = k;
	int rneg;
	if (bg)
	{
		sprintf(buf, "%s/negs.txt", dir);
		__ccv_read_background_data(buf, negdata, &rneg, cascade->size);
	}
	for (; i < params.layer; i++)
	{
		if (!bg)
		{
			rneg = __ccv_prepare_background_data(cascade, bgfiles, bgnum, negdata, negnum);
			/* save state of background data */
			sprintf(buf, "%s/negs.txt", dir);
			__ccv_write_background_data(buf, negdata, rneg, cascade->size);
			bg = 1;
		}
		double totalw;
		/* save state of cascade : level, weight etc. */
		sprintf(buf, "%s/stat.txt", dir);
		__ccv_save_sgf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
		ccv_sgf_stage_classifier_t classifier;
		if (k > 0)
		{
			/* resume state of classifier */
			sprintf( buf, "%s/stage-%d.txt", dir, i );
			__ccv_read_sgf_stage_classifier(buf, &classifier);
		} else {
			/* initialize classifier */
			totalw = params.balance_k * posnum + inv_balance_k * rneg;
			for (j = 0; j < posnum; j++)
				pw[j] = params.balance_k / totalw;
			for (j = 0; j < rneg; j++)
				nw[j] = inv_balance_k / totalw;
			classifier.count = k;
			classifier.threshold = 0;
			classifier.feature = (ccv_sgf_feature_t*)malloc(cacheK * sizeof(ccv_sgf_feature_t));
			classifier.alpha = (float*)malloc(cacheK * 2 * sizeof(float));
		}
		__ccv_prune_positive_data(cascade, posdata, &posnum, cascade->size);
		printf("%d postivie data and %d negative data in training\n", posnum, rneg);
		for (; ; k++)
		{
			/* get overall true-positive, false-positive rate and threshold */
			double tp = 0, fp = 0, etp = 0, efp = 0;
			__ccv_sgf_eval_data(&classifier, posdata, posnum, negdata, rneg, cascade->size, peval, neval);
			__ccv_sort_32f(peval, posnum, 0);
			classifier.threshold = peval[(int)((1. - params.pos_crit) * posnum)] - 1e-6;
			for (j = 0; j < posnum; j++)
			{
				if (peval[j] >= 0)
					++tp;
				if (peval[j] >= classifier.threshold)
					++etp;
			}
			tp /= posnum; etp /= posnum;
			for (j = 0; j < rneg; j++)
			{
				if (neval[j] >= 0)
					++fp;
				if (neval[j] >= classifier.threshold)
					++efp;
			}
			fp /= rneg; efp /= rneg;
			printf("stage classifier real TP rate : %f, FP rate : %f\n", tp, fp);
			printf("stage classifier TP rate : %f, FP rate : %f at threshold : %f\n", etp, efp, classifier.threshold);
			if (k > 0)
			{
				/* save classifier state */
				sprintf(buf, "%s/stage-%d.txt", dir, i);
				__ccv_write_sgf_stage_classifier(buf, &classifier);
				sprintf(buf, "%s/stat.txt", dir);
				__ccv_save_sgf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
			}
			if (etp > params.pos_crit && efp < params.neg_crit)
				break;
			/* TODO: more post-process is needed in here */

			/* select the best feature in current distribution through genetic algorithm optimization */
			ccv_sgf_feature_t best = __ccv_sgf_genetic_optimize(posdata, posnum, negdata, rneg, params.feature_number, cascade->size, pw, nw);
			double err = __ccv_sgf_error_rate(&best, posdata, posnum, negdata, rneg, cascade->size, pw, nw);
			double rw = (1 - err) / err;
			totalw = 0;
			/* reweight */
			for (j = 0; j < posnum; j++)
			{
				int* i32c8[] = { posdata[j], posdata[j] + isizs0 };
				if (!__ccv_run_sgf_feature(&best, steps, i32c8))
					pw[j] *= rw;
				pw[j] *= params.balance_k;
				totalw += pw[j];
			}
			for (j = 0; j < rneg; j++)
			{
				int* i32c8[] = { negdata[j], negdata[j] + isizs0 };
				if (__ccv_run_sgf_feature(&best, steps, i32c8))
					nw[j] *= rw;
				nw[j] *= inv_balance_k;
				totalw += nw[j];
			}
			for (j = 0; j < posnum; j++)
				pw[j] = pw[j] / totalw;
			for (j = 0; j < rneg; j++)
				nw[j] = nw[j] / totalw;
			double c = log(rw);
			printf("coefficient of feature %d: %f\n", k + 1, c);
			classifier.count = k + 1;
			/* resizing classifier */
			if (k >= cacheK)
			{
				ccv_sgf_feature_t* feature = (ccv_sgf_feature_t*)malloc(cacheK * 2 * sizeof(ccv_sgf_feature_t));
				memcpy(feature, classifier.feature, cacheK * sizeof(ccv_sgf_feature_t));
				free(classifier.feature);
				float* alpha = (float*)malloc(cacheK * 4 * sizeof(float));
				memcpy(alpha, classifier.alpha, cacheK * 2 * sizeof(float));
				free(classifier.alpha);
				classifier.feature = feature;
				classifier.alpha = alpha;
				cacheK *= 2;
			}
			/* setup new feature */
			classifier.feature[k] = best;
			classifier.alpha[k * 2] = -c;
			classifier.alpha[k * 2 + 1] = c;
		}
		cascade->count = i + 1;
		ccv_sgf_stage_classifier_t* stage_classifier = (ccv_sgf_stage_classifier_t*)malloc(cascade->count * sizeof(ccv_sgf_stage_classifier_t));
		memcpy(stage_classifier, cascade->stage_classifier, i * sizeof(ccv_sgf_stage_classifier_t));
		free(cascade->stage_classifier);
		stage_classifier[i] = classifier;
		cascade->stage_classifier = stage_classifier;
		k = 0;
		bg = 0;
		for (j = 0; j < rneg; j++)
			free(negdata[j]);
	}

	free(neval);
	free(peval);
	free(nw);
	free(pw);
	free(negdata);
	free(posdata);
	free(cascade);
}

static int __ccv_is_equal(const void* _r1, const void* _r2, void* data)
{
	const ccv_sgf_comp_t* r1 = (const ccv_sgf_comp_t*)_r1;
	const ccv_sgf_comp_t* r2 = (const ccv_sgf_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.5 + 0.5);

	return r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

static int __ccv_is_equal_same_class(const void* _r1, const void* _r2, void* data)
{
	const ccv_sgf_comp_t* r1 = (const ccv_sgf_comp_t*)_r1;
	const ccv_sgf_comp_t* r2 = (const ccv_sgf_comp_t*)_r2;
	int distance = (int)(r1->rect.width * 0.5 + 0.5);

	return r2->id == r1->id &&
		   r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= (int)(r1->rect.width * 1.5 + 0.5) &&
		   (int)(r2->rect.width * 1.5 + 0.5) >= r1->rect.width;
}

ccv_array_t* ccv_sgf_detect_objects(ccv_dense_matrix_t* a, ccv_sgf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size)
{
	int hr = a->rows / min_size.height;
	int wr = a->cols / min_size.width;
	int scale_upto = (int)(log((double)ccv_min(hr, wr)) / log(sqrt(2.)));
	/* generate scale-down HOG images */
	ccv_dense_matrix_t** pyr = (ccv_dense_matrix_t**)alloca((scale_upto + 2) * sizeof(ccv_dense_matrix_t*));
	if (min_size.height != _cascade[0]->size.height || min_size.width != _cascade[0]->size.width)
	{
		pyr[0] = NULL;
		ccv_resample(a, &pyr[0], a->rows * _cascade[0]->size.height / min_size.height, a->cols * _cascade[0]->size.width / min_size.width, CCV_INTER_AREA);
	} else
		pyr[0] = a;
	double sqrt_2 = sqrt(2.);
	pyr[1] = NULL;
	ccv_resample(pyr[0], &pyr[1], (int)(pyr[0]->rows / sqrt_2), (int)(pyr[0]->cols / sqrt_2), CCV_INTER_AREA);
	int i, j, k, t, x, y;
	for (i = 2; i < scale_upto + 2; i += 2)
	{
		pyr[i] = NULL;
		ccv_sample_down(pyr[i - 2], &pyr[i]);
	}
	for ( i = 3; i < scale_upto + 2; i += 2 )
	{
		pyr[i] = NULL;
		ccv_sample_down(pyr[i - 2], &pyr[i]);
	}
	int* cols = (int*)alloca((scale_upto + 2) * sizeof(int));
	int* rows = (int*)alloca((scale_upto + 2) * sizeof(int));
	ccv_dense_matrix_t** hogs = (ccv_dense_matrix_t**)alloca((scale_upto + 2) * sizeof(ccv_dense_matrix_t*));
	for (i = 0; i < scale_upto + 2; i++)
	{
		rows[i] = pyr[i]->rows;
		cols[i] = pyr[i]->cols;
		hogs[i] = NULL;
		ccv_hog(pyr[i], &hogs[i], 2 * HOG_BORDER_SIZE + 1);
	}
	for (i = 1; i < scale_upto + 2; i++)
		ccv_matrix_free(pyr[i]);
	if ( min_size.height != _cascade[0]->size.height || min_size.width != _cascade[0]->size.width )
		ccv_matrix_free(pyr[0]);

	ccv_array_t* idx_seq;
	ccv_array_t* seq = ccv_array_new(64, sizeof(ccv_sgf_comp_t));
	ccv_array_t* seq2 = ccv_array_new(64, sizeof(ccv_sgf_comp_t));
	ccv_array_t* result_seq = ccv_array_new(64, sizeof(ccv_sgf_comp_t));
	/* detect in multi scale */
	for (t = 0; t < count; t++)
	{
		ccv_sgf_classifier_cascade_t* cascade = _cascade[t];
		float scale_x = (float) min_size.width / (float) cascade->size.width;
		float scale_y = (float) min_size.height / (float) cascade->size.height;
		ccv_array_clear(seq);
		for (i = 0; i < scale_upto; i++)
		{
			int i_rows = rows[i + 2] - HOG_BORDER_SIZE * 2 - (cascade->size.height >> 1);
			int steps[] = { (cols[i] - HOG_BORDER_SIZE * 2) * 8, (cols[i + 2] - HOG_BORDER_SIZE * 2) * 8 };
			int cols_pads1 = cols[i + 2] - HOG_BORDER_SIZE * 2 - (cascade->size.width >> 1);
			int pads1 = (cascade->size.width >> 1) * 8;
			int pads0 = steps[0] * 2 - (cols_pads1 << 1) * 8;
			int* i32c8p[] = { hogs[i]->data.i, hogs[i + 2]->data.i };
			for (y = 0; y < i_rows; y++)
			{
				for (x = 0; x < cols_pads1; x++)
				{
					float sum;
					int flag = 1;
					ccv_sgf_stage_classifier_t* classifier = cascade->stage_classifier;
					for (j = 0; j < cascade->count; ++j, ++classifier)
					{
						sum = 0;
						float* alpha = classifier->alpha;
						ccv_sgf_feature_t* feature = classifier->feature;
						for (k = 0; k < classifier->count; ++k, alpha += 2, ++feature)
							sum += alpha[__ccv_run_sgf_feature(feature, steps, i32c8p)];
						if (sum < classifier->threshold)
						{
							flag = 0;
							break;
						}
					}
					if (flag)
					{
						ccv_sgf_comp_t comp;
						comp.rect = ccv_rect((int)((x * 2 + HOG_BORDER_SIZE) * scale_x), (int)((y * 2 + HOG_BORDER_SIZE) * scale_y), (int)(cascade->size.width * scale_x), (int)(cascade->size.height * scale_y));
						comp.id = t;
						comp.neighbors = 1;
						comp.confidence = sum;
						ccv_array_push(seq, &comp);
					}
					i32c8p[0] += 16;
					i32c8p[1] += 8;
				}
				i32c8p[0] += pads0;
				i32c8p[1] += pads1;
			}
			scale_x *= sqrt_2;
			scale_y *= sqrt_2;
		}

		/* the following code from OpenCV's haar feature implementation */
		if(min_neighbors == 0)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_sgf_comp_t* comp = (ccv_sgf_comp_t*)ccv_array_get(seq, i);
				ccv_array_push(result_seq, comp);
			}
		} else {
			idx_seq = NULL;
			ccv_array_clear(seq2);
			// group retrieved rectangles in order to filter out noise
			int ncomp = ccv_array_group(seq, &idx_seq, __ccv_is_equal_same_class, 0);
			ccv_sgf_comp_t* comps = (ccv_sgf_comp_t*)malloc((ncomp + 1) * sizeof(ccv_sgf_comp_t));
			memset(comps, 0, (ncomp + 1) * sizeof(ccv_sgf_comp_t));

			// count number of neighbors
			for(i = 0; i < seq->rnum; i++)
			{
				ccv_sgf_comp_t r1 = *(ccv_sgf_comp_t*)ccv_array_get(seq, i);
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
				if(n >= min_neighbors)
				{
					ccv_sgf_comp_t comp;
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
				ccv_sgf_comp_t r1 = *(ccv_sgf_comp_t*)ccv_array_get(seq2, i);
				int flag = 1;

				for(j = 0; j < seq2->rnum; j++)
				{
					ccv_sgf_comp_t r2 = *(ccv_sgf_comp_t*)ccv_array_get(seq2, j);
					int distance = (int)(r2.rect.width * 0.5 + 0.5);

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
			free(comps);
		}
	}

	ccv_array_free(seq);
	ccv_array_free(seq2);

	ccv_array_t* result_seq2;
	/* the following code from OpenCV's haar feature implementation */
	if (flags & CCV_SGF_NO_NESTED)
	{
		result_seq2 = ccv_array_new(64, sizeof(ccv_sgf_comp_t));
		idx_seq = NULL;
		// group retrieved rectangles in order to filter out noise
		int ncomp = ccv_array_group(result_seq, &idx_seq, __ccv_is_equal, 0);
		ccv_sgf_comp_t* comps = (ccv_sgf_comp_t*)malloc((ncomp + 1) * sizeof(ccv_sgf_comp_t));
		memset(comps, 0, (ncomp + 1) * sizeof(ccv_sgf_comp_t));

		// count number of neighbors
		for(i = 0; i < result_seq->rnum; i++)
		{
			ccv_sgf_comp_t r1 = *(ccv_sgf_comp_t*)ccv_array_get(result_seq, i);
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
		free(comps);
	} else {
		result_seq2 = result_seq;
	}

	for ( i = 0; i < scale_upto + 2; ++i )
		ccv_matrix_free(hogs[i]);

	return result_seq2;
}

ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade(const char* directory)
{
	ccv_sgf_classifier_cascade_t* cascade = (ccv_sgf_classifier_cascade_t*)malloc(sizeof(ccv_sgf_classifier_cascade_t));
	char buf[1024];
	sprintf(buf, "%s/cascade.txt", directory);
	int s, i;
	FILE* r = fopen(buf, "r");
	if (r != NULL)
		s = fscanf(r, "%d %d %d", &cascade->count, &cascade->size.width, &cascade->size.height);
	cascade->stage_classifier = (ccv_sgf_stage_classifier_t*)malloc(cascade->count * sizeof(ccv_sgf_stage_classifier_t));
	for (i = 0; i < cascade->count; i++)
	{
		sprintf(buf, "%s/stage-%d.txt", directory, i);
		if (__ccv_read_sgf_stage_classifier(buf, &cascade->stage_classifier[i]) < 0)
		{
			cascade->count = i;
			break;
		}
	}
	return cascade;
}

void ccv_sgf_classifier_cascade_free(ccv_sgf_classifier_cascade_t* cascade)
{
	int i;
	for (i = 0; i < cascade->count; ++i)
	{
		free(cascade->stage_classifier[i].feature);
		free(cascade->stage_classifier[i].alpha);
	}
	free(cascade->stage_classifier);
	free(cascade);
}
