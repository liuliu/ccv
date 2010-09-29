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

static int __ccv_read_bbf_stage_classifier(const char* file, ccv_bbf_stage_classifier_t* classifier)
{
	FILE* r = fopen(file, "r");
	if (r == 0) return -1;
	int stat = 0;
	stat |= fscanf(r, "%d", &classifier->count);
	union { float fl; int i; } fli;
	stat |= fscanf(r, "%d", &fli.i);
	classifier->threshold = fli.fl;
	classifier->feature = (ccv_bbf_feature_t*)malloc(classifier->count * sizeof(ccv_bbf_feature_t));
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

static int __ccv_write_bbf_stage_classifier(const char* file, ccv_bbf_stage_classifier_t* classifier)
{
	FILE* w = fopen(file, "wb");
	if (w == 0) return -1;
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
	FILE* r = fopen(file, "rb");
	if (r == 0) return -1;
	stat |= fread(negnum, sizeof(int), 1, r);
	int i;
	int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
	for (i = 0; i < *negnum; i++)
	{
		negdata[i] = (int*)malloc(isizs01 * sizeof(int));
		stat |= fread(negdata[i], sizeof(int), isizs01, r);
	}
	fclose(r);
	return 0;
}

static int __ccv_write_background_data(const char* file, int** negdata, int negnum, ccv_size_t size)
{
	FILE* w = fopen(file, "w");
	if (w == 0) return -1;
	fwrite(&negnum, sizeof(int), 1, w);
	int i;
	int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
	for (i = 0; i < negnum; i++)
		fwrite(negdata[i], sizeof(int), isizs01, w);
	fclose(w);
	return 0;
}

static int __ccv_resume_bbf_cascade_training_state(const char* file, int* i, int* k, int* bg, double* pw, double* nw, int posnum, int negnum)
{
	int stat = 0;
	FILE* r = fopen(file, "r");
	if (r == 0) return -1;
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

static int __ccv_save_bbf_cacade_training_state(const char* file, int i, int k, int bg, double* pw, double* nw, int posnum, int negnum)
{
	FILE* w = fopen(file, "w");
	if (w == 0) return -1;
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

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_param_t params)
{
	int i, j, k;
	/* allocate memory for usage */
	ccv_bbf_classifier_cascade_t* cascade = (ccv_bbf_classifier_cascade_t*)malloc(sizeof(ccv_bbf_classifier_cascade_t));
	cascade->count = 0;
	cascade->size = size;
	cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)malloc(sizeof(ccv_bbf_stage_classifier_t));
	int** posdata = (int**)malloc(posnum * sizeof(int*));
	int** negdata = (int**)malloc(negnum * sizeof(int*));
	double* pw = (double*)malloc(posnum * sizeof(double));
	double* nw = (double*)malloc(negnum * sizeof(double));
	float* peval = (float*)malloc(posnum * sizeof(float));
	float* neval = (float*)malloc(negnum * sizeof(float));
	double inv_balance_k = 1. / params.balance_k;
	/* balance factor k, and weighted with 0.01 */
	params.balance_k *= 0.01;
	inv_balance_k *= 0.01;

	int isizs0 = cascade->size.width * cascade->size.height;
	int cols[] = { cascade->size.width, cascade->size.width >> 1, cascade->size.width >> 2 };
	
	i = 0;
	k = 0;
	int bg = 0;
	int cacheK = 10;
	/* state resume code */
	char buf[1024];
	sprintf(buf, "%s/stat.txt", dir);
	__ccv_resume_bbf_cascade_training_state(buf, &i, &k, &bg, pw, nw, posnum, negnum);
	if (i > 0)
	{
		cascade->count = i;
		free(cascade->stage_classifier);
		cascade->stage_classifier = (ccv_bbf_stage_classifier_t*)malloc(i * sizeof(ccv_bbf_stage_classifier_t));
		for (j = 0; j < i; j++)
		{
			sprintf(buf, "%s/stage-%d.txt", dir, j);
			__ccv_read_bbf_stage_classifier(buf, &cascade->stage_classifier[j]);
		}
	}
	if (k > 0)
		cacheK = k;
	int rpos, rneg;
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
		__ccv_save_bbf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
		ccv_bbf_stage_classifier_t classifier;
		if (k > 0)
		{
			/* resume state of classifier */
			sprintf( buf, "%s/stage-%d.txt", dir, i );
			__ccv_read_bbf_stage_classifier(buf, &classifier);
		} else {
			/* initialize classifier */
			for (j = 0; j < posnum; j++)
				pw[j] = params.balance_k;
			for (j = 0; j < rneg; j++)
				nw[j] = inv_balance_k;
			classifier.count = k;
			classifier.threshold = 0;
			classifier.feature = (ccv_bbf_feature_t*)malloc(cacheK * sizeof(ccv_bbf_feature_t));
			classifier.alpha = (float*)malloc(cacheK * 2 * sizeof(float));
		}
		__ccv_prepare_positive_data(posimg, posdata, cascade->size, posnum);
		rpos = __ccv_prune_positive_data(cascade, posdata, posnum, cascade->size);
		printf("%d postivie data and %d negative data in training\n", rpos, rneg);
		/* reweight to 1.00 */
		totalw = 0;
		for (j = 0; j < rpos; j++)
			totalw += pw[j];
		for (j = 0; j < rneg; j++)
			totalw += nw[j];
		for (j = 0; j < rpos; j++)
			pw[j] = pw[j] / totalw;
		for (j = 0; j < rneg; j++)
			nw[j] = nw[j] / totalw;
		for (; ; k++)
		{
			/* get overall true-positive, false-positive rate and threshold */
			double tp = 0, fp = 0, etp = 0, efp = 0;
			__ccv_bbf_eval_data(&classifier, posdata, rpos, negdata, rneg, cascade->size, peval, neval);
			__ccv_sort_32f(peval, rpos, 0);
			classifier.threshold = peval[(int)((1. - params.pos_crit) * rpos)] - 1e-6;
			for (j = 0; j < rpos; j++)
			{
				if (peval[j] >= 0)
					++tp;
				if (peval[j] >= classifier.threshold)
					++etp;
			}
			tp /= rpos; etp /= rpos;
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
				__ccv_write_bbf_stage_classifier(buf, &classifier);
				sprintf(buf, "%s/stat.txt", dir);
				__ccv_save_bbf_cacade_training_state(buf, i, k, bg, pw, nw, posnum, negnum);
			}
			if (etp > params.pos_crit && efp < params.neg_crit)
				break;
			/* TODO: more post-process is needed in here */

			/* select the best feature in current distribution through genetic algorithm optimization */
			ccv_bbf_feature_t best = __ccv_bbf_convex_optimize(posdata, rpos, negdata, rneg, params.feature_number, cascade->size, pw, nw);
			double err = __ccv_bbf_error_rate(&best, posdata, rpos, negdata, rneg, cascade->size, pw, nw);
			double rw = (1 - err) / err;
			totalw = 0;
			/* reweight */
			for (j = 0; j < rpos; j++)
			{
				int* i32c8[] = { posdata[j], posdata[j] + isizs0 };
				if (!__ccv_run_bbf_feature(&best, steps, i32c8))
					pw[j] *= rw;
				pw[j] *= params.balance_k;
				totalw += pw[j];
			}
			for (j = 0; j < rneg; j++)
			{
				int* i32c8[] = { negdata[j], negdata[j] + isizs0 };
				if (__ccv_run_bbf_feature(&best, steps, i32c8))
					nw[j] *= rw;
				nw[j] *= inv_balance_k;
				totalw += nw[j];
			}
			for (j = 0; j < rpos; j++)
				pw[j] = pw[j] / totalw;
			for (j = 0; j < rneg; j++)
				nw[j] = nw[j] / totalw;
			double c = log(rw);
			printf("coefficient of feature %d: %f\n", k + 1, c);
			classifier.count = k + 1;
			/* resizing classifier */
			if (k >= cacheK)
			{
				ccv_bbf_feature_t* feature = (ccv_bbf_feature_t*)malloc(cacheK * 2 * sizeof(ccv_bbf_feature_t));
				memcpy(feature, classifier.feature, cacheK * sizeof(ccv_bbf_feature_t));
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
		ccv_bbf_stage_classifier_t* stage_classifier = (ccv_bbf_stage_classifier_t*)malloc(cascade->count * sizeof(ccv_bbf_stage_classifier_t));
		memcpy(stage_classifier, cascade->stage_classifier, i * sizeof(ccv_bbf_stage_classifier_t));
		free(cascade->stage_classifier);
		stage_classifier[i] = classifier;
		cascade->stage_classifier = stage_classifier;
		k = 0;
		bg = 0;
		for (j = 0; j < rpos; j++)
			free(posdata[j]);
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
