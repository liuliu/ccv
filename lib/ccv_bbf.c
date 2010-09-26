#include "ccv.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif
#ifndef _WIN32
#include <sys/time.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

static unsigned int __ccv_bbf_time_measure()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

static inline int __ccv_run_bbf_feature(ccv_bbf_feature_t* feature, int* step, unsigned char** u8)
{
#define pf_at(i) (*(u8[feature->pz[i]] + feature->px[i] + feature->py[i] * step[feature->pz[i]]))
#define nf_at(i) (*(u8[feature->nz[i]] + feature->nx[i] + feature->ny[i] * step[feature->nz[i]]))
	unsigned char pmin = pf_at(0), nmax = nf_at(0);
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

typedef struct {
	double fitness;
	int pk, nk;
	int age;
	double error;
	ccv_bbf_feature_t feature;
} ccv_bbf_gene_t;

static inline void __ccv_bbf_genetic_fitness(ccv_bbf_gene_t* gene)
{
	gene->fitness = (1 - gene->error) * exp(-0.01 * gene->age) * exp((gene->pk + gene->nk) * log(1.015));
}

static inline int __ccv_bbf_exist_gene_feature(ccv_bbf_gene_t* gene, int x, int y, int z)
{
	int i;
	for (i = 0; i < gene->pk; i++)
		if (z == gene->feature.pz[i] && x == gene->feature.px[i] && y == gene->feature.py[i])
			return 1;
	for (i = 0; i < gene->nk; i++)
		if (z == gene->feature.nz[i] && x == gene->feature.nx[i] && y == gene->feature.ny[i])
			return 1;
	return 0;
}

static inline void __ccv_bbf_randomize_gene(gsl_rng* rng, ccv_bbf_gene_t* gene, int* rows, int* cols)
{
	int i;
	do {
		gene->pk = gsl_rng_uniform_int(rng, CCV_BBF_POINT_MAX - 1) + 1;
		gene->nk = gsl_rng_uniform_int(rng, CCV_BBF_POINT_MAX - 1) + 1;
	} while (gene->pk + gene->nk < CCV_BBF_POINT_MIN); /* a hard restriction of at least 3 points have to be examed */
	gene->feature.size = ccv_max(gene->pk, gene->nk);
	gene->age = 0;
	for (i = 0; i < CCV_BBF_POINT_MAX; i++)
	{
		gene->feature.pz[i] = -1;
		gene->feature.nz[i] = -1;
	}
	int x, y, z;
	for (i = 0; i < gene->pk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 3);
			x = gsl_rng_uniform_int(rng, cols[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while (__ccv_bbf_exist_gene_feature(gene, x, y, z));
		gene->feature.pz[i] = z;
		gene->feature.px[i] = x;
		gene->feature.py[i] = y;
	}
	for (i = 0; i < gene->nk; i++)
	{
		do {
			z = gsl_rng_uniform_int(rng, 3);
			x = gsl_rng_uniform_int(rng, cols[z]);
			y = gsl_rng_uniform_int(rng, rows[z]);
		} while ( __ccv_bbf_exist_gene_feature(gene, x, y, z));
		gene->feature.nz[i] = z;
		gene->feature.nx[i] = x;
		gene->feature.ny[i] = y;
	}
}

static ccv_bbf_feature_t __ccv_bbf_convex_optimize(int** posdata, int posnum, int** negdata, int negnum, int ftnum, ccv_size_t size, double* pw, double* nw)
{
	/* seed (random method) */
	gsl_rng_env_setup();
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	union { unsigned long int li; double db; } dbli;
	dbli.db = pw[0] + nw[0];
	gsl_rng_set(rng, dbli.li);
	int i, j, k, m;
	int rows[] = { size.height, size.height >> 1, size.height >> 2 };
	int cols[] = { size.width, size.width >> 1, size.width >> 2 };
	ccv_bbf_gene_t* gene = (ccv_bbf_gene_t*)malloc((rows[0] * cols[0] + rows[1] * cols[1] + rows[2] * cols[2]) * sizeof(ccv_bbf_gene_t));
	int pnum;
	double best_err;
	ccv_bbf_feature_t best;
	int z = gsl_rng_uniform_int(rng, 2);
	int x = gsl_rng_uniform_int(rng, steps[z]);
	int y = gsl_rng_uniform_int(rng, rows[z]);
	m = 0;
	for (i = 0; i < 3; i++)
		for (j = 0; j < cols[i]; j++)
			for (k = 0; k < rows[i]; k++)
				if (i != z && j != x && k != y)
				{
					gene[m].pk = gene[m].nk = 1;
					gene[m].feature.pz[0] = z;
					gene[m].feature.px[0] = x;
					gene[m].feature.py[0] = y;
					gene[m].feature.nz[0] = i;
					gene[m].feature.nx[0] = j;
					gene[m].feature.ny[0] = k;
					gene[m].feature.size = 1;
					m++;
				}
	unsigned int timer = __ccv_bbf_time_measure();
#ifdef USE_OPENMP
#pragma omp parallel for private(i) schedule(dynamic)
#endif
	for (i = 0; i < pnum; i++)
		gene[i].error = __ccv_bbf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
	timer = __ccv_bbf_time_measure() - timer;
	/* iteration stop crit : best no change in 40 iterations */
	int it = 0, t;
	for (t = 0 ; it < 40; ++it, ++t)
	{
		int min_id = 0;
		double min_err = gene[0].error;
		for (i = 1; i < pnum; i++)
			if (gene[i].error < min_err)
			{
				min_id = i;
				min_err = gene[i].error;
			}
		min_err = gene[min_id].error = __ccv_bbf_error_rate(&gene[min_id].feature, posdata, posnum, negdata, negnum, size, pw, nw);
		if (min_err < best_err)
		{
			best_err = min_err;
			memcpy(&best, &gene[min_id].feature, sizeof(best));
			printf("best bbf feature with error %f\n|-size: %d\n|-positive point: ", best_err, best.size);
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.px[i], best.py[i], best.pz[i]);
			printf("\n|-negative point: ");
			for (i = 0; i < best.size; i++)
				printf("(%d %d %d), ", best.nx[i], best.ny[i], best.nz[i]);
			printf("\n");
			it = 0;
		}
		printf("minimum error achieved in round %d(%d) : %f with %d ms\n", t, it, min_err, timer / 1000);
		timer = __ccv_bbf_time_measure();
#ifdef USE_OPENMP
#pragma omp parallel for private(i) schedule(dynamic)
#endif
		for (i = 0; i < pnum; i++)
			gene[i].error = __ccv_bbf_error_rate(&gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw);
		timer = __ccv_bbf_time_measure() - timer;
		for (i = 0; i < pnum; i++)
			__ccv_bbf_genetic_fitness(&gene[i]);
	}
	gsl_rng_free(rng);
	return best;
}

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_param_t params)
{
}

ccv_array_t* ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size)
{
}

ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory)
{
}

ccv_bbf_classifier_cascade_t* ccv_bbf_classifier_cascade_read_binary(char* s)
{
}

int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen)
{
}

void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade)
{
}
