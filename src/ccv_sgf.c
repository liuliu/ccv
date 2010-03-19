#include "ccv.h"
#include <omp.h>

static inline int __ccv_run_sgf_feature(ccv_sgf_feature_t* feature, int* step, int** i32c8)
{
#define pf_at(i) (*(i32c8[feature->pz[i]] + feature->px[i] + feature->py[i] * step[feature->pz[i]]))
#define nf_at(i) (*(i32c8[feature->nz[i]] + feature->nx[i] + feature->ny[i] * step[feature->nz[i]]))
	int pmin = pf_at(0), nmax = nf_at(0);
	/* check if every point in P > every point in N, and take a shortcut */
	if ( pmin <= nmax )
		return 0;
	int i;
	for ( i = 1; i < feature->size; ++i )
	{
		if ( feature->pz[i] >= 0 )
		{
			int p = pf_at(i);
			if ( p < pmin )
			{
				if ( p <= nmax )
					return 0;
				pmin = p;
			}
		}
		if ( feature->nz[i] >= 0 )
		{
			int n = nf_at(i);
			if ( n > nmax )
			{
				if ( pmin <= n )
					return 0;
				nmax = n;
			}
		}
	}
#undef pf_at
#undef nf_at
	return 1;
}

static int __ccv_prepare_background_data(ccv_sgf_classifier_cascade_t* cascade, char** bgfiles, int bgnum, int** negdata, int negnum)
{
	int t, i, j, k, x, y;
	int negperbg = negnum / bgnum + 1;
	int negtotal = 0;
	int isizs0 = cascade->size.width * cascade->size.height * 8;
	int isizs1 = ((cascade->size.width >> 1) - HOG_BORDER_SIZE) * ((cascade->size.height >> 1) - HOG_BORDER_SIZE) * 8;
	int steps[] = { cascade->size.width * 8, ((cascade->size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	printf("Preparing negative data ...  0%%");
	CvMemStorage* parent = cvCreateMemStorage(NULL);
	int* idcheck = (int*)cvAlloc( negnum * sizeof(idcheck[0]) );
	CvRNG rng = cvRNG( (int64)idcheck );
	CvMat* imgs0 = cvCreateMat( cascade->size.height + HOG_BORDER_SIZE * 2, cascade->size.width + HOG_BORDER_SIZE * 2, CV_8UC1 );
	CvMat* imgs1 = cvCreateMat( imgs0->rows >> 1, imgs0->cols >> 1, CV_8UC1 );
	int rneg = negtotal;
	for ( t = 0; negtotal < negnum; ++t )
	{
		for ( i = 0; i < bgnum; ++i )
		{
			negperbg = ( t < 2 ) ? (negnum - negtotal) / ( bgnum - i) + 1 : negnum - negtotal;
			IplImage* image = cvLoadImage( bgfiles[i], CV_LOAD_IMAGE_GRAYSCALE );
			if ( image == NULL )
			{
				printf("\n%s file corrupted\n", bgfiles[i]);
				continue;
			}
			if ( t % 2 != 0 )
				cvFlip( image, NULL, 1 );
			CvMemStorage* storage = cvCreateChildMemStorage( parent );
			CvSeq* detected = cvSGFDetectObjects( image, &cascade, 1, storage, 0, 0, cascade->size );
			for ( j = 0; j < MIN( detected->total, negperbg ); ++j )
			{
				int r = cvRandInt( &rng ) % detected->total;
				int flag = 1;
				CvRect* rect = (CvRect*)cvGetSeqElem( detected, r );
				while ( flag ) {
					flag = 0;
					for ( k = 0; k < j; ++k )
						if ( r == idcheck[k] )
						{
							flag = 1;
							r = cvRandInt( &rng ) % detected->total;
							break;
						}
					rect = (CvRect*)cvGetSeqElem( detected, r );
					if ( (rect->x < 0) || (rect->y < 0) || (rect->width + rect->x >= image->width) || (rect->height + rect->y >= image->height) )
					{
						flag = 1;
						r = cvRandInt( &rng ) % detected->total;
					}
				}
				idcheck[j] = r;
				cvSetImageROI( image, *rect );
				CvMat* temp = cvCreateMat( rect->height, rect->width, CV_8UC1 );
				cvCopy( image, temp );
				cvResize( temp, imgs0, CV_INTER_AREA );
				cvReleaseMat( &temp );
				cvPyrDown( imgs0, imgs1 );

				negdata[negtotal] = (int*)cvAlloc( (isizs0 + isizs1) * sizeof(int) );
				int* i32c8s0 = negdata[negtotal];
				int* i32c8s1 = negdata[negtotal] + isizs0;
				int* i32c8[] = { i32c8s0, i32c8s1 };

				icvCreateHOG( imgs0, i32c8s0 );
				icvCreateHOG( imgs1, i32c8s1 );

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
				CvSGFStageClassifier* classifier = cascade->stage_classifier;
				for ( k = 0; k < cascade->count; ++k, ++classifier )
				{
					float sum = 0;
					float* alpha = classifier->alpha;
					CvSGFeature* feature = classifier->feature;
					for ( k = 0; k < classifier->count; ++k, alpha += 2, ++feature )
						sum += alpha[icvRunSGFeature( feature, steps, i32c8 )];
					if ( sum < classifier->threshold )
					{
						flag = 0;
						break;
					}
				}
				if ( !flag )
					cvFree( &negdata[negtotal] );
				else {
					++negtotal;
					if ( negtotal >= negnum )
						break;
				}
			}

			cvReleaseMemStorage( &storage );
			cvReleaseImage( &image );
			printf( "\rPreparing negative data ... %2d%%", 100 * negtotal / negnum );
			fflush(NULL);
			if ( negtotal >= negnum )
				break;
		}
		if ( rneg == negtotal )
			break;
		rneg = negtotal;
	}
	cvFree( &idcheck );
	cvReleaseMat( &imgs1 );
	cvReleaseMat( &imgs0 );
	cvReleaseMemStorage( &parent );
	printf("\n");
	return negtotal;
}

static void __ccv_prepare_positive_data(ccv_dense_matrix_t** posimg, int** posdata, ccv_size_t size, int posnum)
{
	printf("Preparing positive data ...  0%%");
	int i;
	for ( i = 0; i < posnum; i++ )
	{
		CvMat imghdr, *imgs0 = cvGetMat( posimg[i], &imghdr );
		CvMat* imgs1 = cvCreateMat( imgs0->rows >> 1, imgs0->cols >> 1, CV_8UC1 );
		cvPyrDown( imgs0, imgs1 );
		int isizs0 = size.width * size.height * 8;
		int isizs1 = ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;

		posdata[i] = (int*)cvAlloc( (isizs0 + isizs1) * sizeof(int) );
		int* i32c8s0 = posdata[i];
		int* i32c8s1 = posdata[i] + isizs0;

		icvCreateHOG( imgs0, i32c8s0 );
		icvCreateHOG( imgs1, i32c8s1 );

		printf( "\rPreparing positive data ... %2d%%", 100 * (i + 1) / posnum );
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
		cvReleaseMat( &imgs1 );
	}
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
	for ( i = 0; i < posnum; ++i )
	{
		int* i32c8[] = { posdata[i], posdata[i] + isizs0 };
		if ( !icvRunSGFeature( feature, steps, i32c8 ) )
			error += pw[i];
	}
	for ( i = 0; i < negnum; ++i )
	{
		int* i32c8[] = { negdata[i], negdata[i] + isizs0 };
		if ( icvRunSGFeature( feature, steps, i32c8 ) )
			error += nw[i];
	}
	return error;
}

#define less_than( a, b ) ((a) < (b))
CCV_IMPLEMENT_QSORT(__ccv_sort_32f, float, less_than)
#undef less_than

static void __ccv_sgf_eval_data(ccv_sgf_stage_classifier_t* classifier, int** posdata, int posnum, int** negdata, int negnum, ccv_size_t size, float* peval, float* neval)
{
	int i, j;
	int isizs0 = size.width * size.height * 8;
	int steps[] = { size.width * 8, ((size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	for ( i = 0; i < posnum; ++i )
	{
		int* i32c8[] = { posdata[i], posdata[i] + isizs0 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_sgf_feature_t* feature = classifier->feature;
		for ( j = 0; j < classifier->count; ++j, alpha += 2, ++feature )
			sum += alpha[icvRunSGFeature( feature, steps, i32c8 )];
		peval[i] = sum;
	}
	for ( i = 0; i < negnum; ++i )
	{
		int* i32c8[] = { negdata[i], negdata[i] + isizs0 };
		float sum = 0;
		float* alpha = classifier->alpha;
		ccv_sgf_feature_t* feature = classifier->feature;
		for ( j = 0; j < classifier->count; ++j, alpha += 2, ++feature )
			sum += alpha[icvRunSGFeature( feature, steps, i32c8 )];
		neval[i] = sum;
	}
}

static void __ccv_prune_positive_data(ccv_sgf_classifier_cascade_t* cascade, int** posdata, int* posnum, ccv_size_t size)
{
	float* peval = (float*)cvAlloc( *posnum * sizeof(peval[0]) );
	int i, j, k;
	for ( i = 0; i < cascade->count; ++i )
	{
		icvSGFEvalData( cascade->stage_classifier + i, posdata, *posnum, 0, 0, size, peval, 0 );
		k = 0;
		for ( j = 0; j < *posnum; ++j )
			if ( peval[j] >= cascade->stage_classifier[i].threshold )
			{
				posdata[k] = posdata[j];
				++k;
			} else {
				cvFree( &posdata[j] );
			}
		*posnum = k;
	}
	cvFree( &peval );
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

#define less_than(fit1, fit2) ((fit1).fitness < (fit2).fitness)
static CCV_IMPLEMENT_QSORT(__ccv_sgf_genetic_qsort, ccv_sgf_gene_t, less_than)
#undef less_than

static inline void __ccv_sgf_randomize_gene(CvRNG* rng, ccv_sgf_gene_t* gene, int* rows, int* steps)
{
	int i, j;
	do {
		gene->pk = cvRandInt(rng) % (CV_SGF_POINT_MAX - 1) + 1;
		gene->nk = cvRandInt(rng) % (CV_SGF_POINT_MAX - 1) + 1;
	} while ( gene->pk + gene->nk < CV_SGF_POINT_MIN ); /* a hard restriction of at least 3 points have to be examed */
	gene->feature.size = MAX( gene->pk, gene->nk );
	gene->age = 0;
	for ( i = 0; i < CV_SGF_POINT_MAX; ++i )
	{
		gene->feature.pz[i] = -1;
		gene->feature.nz[i] = -1;
	}
	int x, y, z;
	for ( i = 0; i < gene->pk; ++i )
	{
		do {
			z = cvRandInt(rng) % 2;
			x = cvRandInt(rng) % steps[z];
			y = cvRandInt(rng) % rows[z];
		} while ( icvSGFExistGeneFeature( gene, x, y, z ) );
		gene->feature.pz[i] = z;
		gene->feature.px[i] = x;
		gene->feature.py[i] = y;
	}
	for ( i = 0; i < gene->nk; ++i )
	{
		do {
			z = cvRandInt(rng) % 2;
			x = cvRandInt(rng) % steps[z];
			y = cvRandInt(rng) % rows[z];
		} while ( icvSGFExistGeneFeature( gene, x, y, z ) );
		gene->feature.nz[i] = z;
		gene->feature.nx[i] = x;
		gene->feature.ny[i] = y;
	}
}

static ccv_sgf_feature_t __ccv_sgf_genetic_optimize(int** posdata, int posnum, int** negdata, int negnum, int ftnum, ccv_size_t size, double* pw, double* nw)
{
	/* seed (random method) */
	CvRNG rng = cvRNG((int64)posdata);
	int i, j;
	int pnum = ftnum * 100;
	CvSGFGene* gene = (CvSGFGene*)cvAlloc( pnum * sizeof(gene[0]) );
	int rows[] = { size.height, (size.height >> 1) - HOG_BORDER_SIZE };
	int steps[] = { size.width * 8, ((size.width >> 1) - HOG_BORDER_SIZE) * 8 };
	for ( i = 0; i < pnum; ++i )
		icvSGFRandomizeGene( &rng, &gene[i], rows, steps );
	int nthreads = cvGetNumThreads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
	for ( i = 0; i < pnum; ++i )
	{
		gene[i].error = icvSGFErrorRate( &gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw );
		icvSGFGeneticFitness( &gene[i] );
	}
	double best_err = 1;
	CvSGFeature best;
	int rnum = ftnum * 39;//99;//49;//39; /* number of randomize */
	int mnum = ftnum * 40;//0;//50;//40; /* number of mutation */
	int hnum = ftnum * 20;//0;//0;//20; /* number of hybrid */
	/* iteration stop crit : best no change in 40 iterations */
	int it = 0, t;
	for ( t = 0 ; it < 40; ++it, ++t )
	{
		icvSGFGeneticQSort( gene, pnum, 0 );
		for ( i = 0; i < ftnum; ++i )
			++gene[i].age;
		for ( i = ftnum; i < ftnum + mnum; ++i )
		{
			int parent = cvRandInt(&rng) % ftnum;
			memcpy( gene + i, gene + parent, sizeof(gene[0]) );
			/* three mutation strategy : 1. add, 2. remove, 3. refine */
			int pnm, pn = cvRandInt(&rng) % 2;
			int* pnk[] = { &gene[i].pk, &gene[i].nk };
			int* pnx[] = { gene[i].feature.px, gene[i].feature.nx };
			int* pny[] = { gene[i].feature.py, gene[i].feature.ny };
			int* pnz[] = { gene[i].feature.pz, gene[i].feature.nz };
			int x, y, z;
			switch ( cvRandInt(&rng) % 3 )
			{
				case 0: /* add */
					if ( gene[i].pk == CV_SGF_POINT_MAX && gene[i].nk == CV_SGF_POINT_MAX )
						break;
					while ( *pnk[pn] + 1 > CV_SGF_POINT_MAX )
						pn = cvRandInt(&rng) % 2;
					do {
						z = cvRandInt(&rng) % 2;
						x = cvRandInt(&rng) % steps[z];
						y = cvRandInt(&rng) % rows[z];
					} while ( icvSGFExistGeneFeature( &gene[i], x, y, z ) );
					pnz[pn][*pnk[pn]] = z;
					pnx[pn][*pnk[pn]] = x;
					pny[pn][*pnk[pn]] = y;
					++(*pnk[pn]);
					gene[i].feature.size = MAX( gene[i].pk, gene[i].nk );
					gene[i].age = 0;
					break;
				case 1: /* remove */
					if ( gene[i].pk + gene[i].nk <= CV_SGF_POINT_MIN ) /* at least 3 points have to be examed */
						break;
					while ( *pnk[pn] - 1 <= 0 || *pnk[pn] + *pnk[!pn] - 1 < CV_SGF_POINT_MIN )
						pn = cvRandInt(&rng) % 2;
					for ( j = cvRandInt(&rng) % *pnk[pn]; j < *pnk[pn] - 1; ++j )
					{
						pnz[pn][j] = pnz[pn][j + 1];
						pnx[pn][j] = pnx[pn][j + 1];
						pny[pn][j] = pny[pn][j + 1];
					}
					pnz[pn][*pnk[pn] - 1] = -1;
					--(*pnk[pn]);
					gene[i].feature.size = MAX( gene[i].pk, gene[i].nk );
					gene[i].age = 0;
					break;
				case 2: /* refine */
					pnm = cvRandInt(&rng) % *pnk[pn];
					do {
						z = cvRandInt(&rng) % 2;
						x = cvRandInt(&rng) % steps[z];
						y = cvRandInt(&rng) % rows[z];
					} while ( icvSGFExistGeneFeature( &gene[i], x, y, z ) );
					pnz[pn][pnm] = z;
					pnx[pn][pnm] = x;
					pny[pn][pnm] = y;
					gene[i].age = 0;
					break;
			}
		}
		for ( i = ftnum + mnum; i < ftnum + mnum + hnum; ++i )
		{
			/* hybrid strategy: taking positive points from dad, negative points from mum */
			int dad, mum;
			do {
				dad = cvRandInt(&rng) % ftnum;
				mum = cvRandInt(&rng) % ftnum;
			} while ( dad == mum || gene[dad].pk + gene[mum].nk < CV_SGF_POINT_MIN ); /* at least 3 points have to be examed */
			for ( j = 0; j < CV_SGF_POINT_MAX; ++j )
			{
				gene[i].feature.pz[j] = -1;
				gene[i].feature.nz[j] = -1;
			}
			gene[i].pk = gene[dad].pk;
			for ( j = 0; j < gene[i].pk; ++j )
			{
				gene[i].feature.pz[j] = gene[dad].feature.pz[j];
				gene[i].feature.px[j] = gene[dad].feature.px[j];
				gene[i].feature.py[j] = gene[dad].feature.py[j];
			}
			gene[i].nk = gene[mum].nk;
			for ( j = 0; j < gene[i].nk; ++j )
			{
				gene[i].feature.nz[j] = gene[mum].feature.nz[j];
				gene[i].feature.nx[j] = gene[mum].feature.nx[j];
				gene[i].feature.ny[j] = gene[mum].feature.ny[j];
			}
			gene[i].feature.size = MAX( gene[i].pk, gene[i].nk );
			gene[i].age = 0;
		}
		for ( i = ftnum + mnum + hnum; i < ftnum + mnum + hnum + rnum; ++i )
			icvSGFRandomizeGene( &rng, &gene[i], rows, steps );
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
#endif
		for ( i = 0; i < pnum; ++i )
		{
			gene[i].error = icvSGFErrorRate( &gene[i].feature, posdata, posnum, negdata, negnum, size, pw, nw );
			icvSGFGeneticFitness( &gene[i] );
		}
		int min_id = 0;
		double min_err = gene[0].error;
		for ( i = 1; i < pnum; ++i )
			if ( gene[i].error < min_err )
			{
				min_id = i;
				min_err = gene[i].error;
			}
		if ( min_err < best_err )
		{
			best_err = min_err;
			memcpy( &best, &gene[min_id].feature, sizeof(best) );
			printf("Best SGFeature with error %f\n|-size: %d\n|-positive point: ", best_err, best.size);
			for ( i = 0; i < best.size; ++i)
				printf("(%d %d %d), ", best.px[i], best.py[i], best.pz[i]);
			printf("\n|-negative point: ");
			for ( i = 0; i < best.size; ++i)
				printf("(%d %d %d), ", best.nx[i], best.ny[i], best.nz[i]);
			printf("\n");
			it = 0;
		}
		printf( "Minimum error achieved in round %d(%d) : %f\n", t, it, min_err );
	}
	return best;
}

int __ccv_read_sgf_stage_classifier( const char* file, ccv_sgf_stage_classifier_t* classifier )
{
	FILE* R = fopen( file, "r" );
	int stat = 0;
	if ( R != NULL )
	{
		stat |= fscanf( R, "%d", &classifier->count );
		stat |= fscanf( R, "%f", &classifier->threshold );
		classifier->feature = (CvSGFeature*)cvAlloc( classifier->count * sizeof(classifier->feature[0]) );
		classifier->alpha = (float*)cvAlloc( classifier->count * 2 * sizeof(classifier->alpha[0]) );
		int i, j;
		for ( i = 0; i < classifier->count; ++i )
		{
			stat |= fscanf( R, "%d", &classifier->feature[i].size );
			for ( j = 0; j < classifier->feature[i].size; ++j )
			{
				stat |= fscanf( R, "%d %d %d", &classifier->feature[i].px[j], &classifier->feature[i].py[j], &classifier->feature[i].pz[j] );
				stat |= fscanf( R, "%d %d %d", &classifier->feature[i].nx[j], &classifier->feature[i].ny[j], &classifier->feature[i].nz[j] );
			}
			stat |= fscanf( R, "%f %f", &classifier->alpha[i * 2], &classifier->alpha[i * 2 + 1] );
		}
		fclose( R );
		return 1;
	}
	return 0;
}

int __ccv_write_sgf_stage_classifier(const char* file, ccv_sgf_stage_classifier_t* classifier)
{
	FILE* W = fopen( file, "w" );
	if ( W != NULL )
	{
		fprintf( W, "%d\n", classifier->count );
		fprintf( W, "%f\n", classifier->threshold );
		int i, j;
		for ( i = 0; i < classifier->count; ++i )
		{
			fprintf( W, "%d\n", classifier->feature[i].size );
			for ( j = 0; j < classifier->feature[i].size; ++j )
			{
				fprintf( W, "%d %d %d\n", classifier->feature[i].px[j], classifier->feature[i].py[j], classifier->feature[i].pz[j] );
				fprintf( W, "%d %d %d\n", classifier->feature[i].nx[j], classifier->feature[i].ny[j], classifier->feature[i].nz[j] );
			}
			fprintf( W, "%f %f\n", classifier->alpha[i * 2], classifier->alpha[i * 2 + 1] );
		}
		fclose( W );
		return 1;
	}
	return 0;
}

static int __ccv_read_background_data(const char* file, int** negdata, int* negnum, ccv_size_t size)
{
	int stat = 0;
	FILE* R = fopen( file, "r" );
	if ( R != NULL )
	{
		stat |= fscanf( R, "%d", negnum );
		int i, j;
		int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
		for ( i = 0; i < *negnum; ++i )
		{
			negdata[i] = (int*)cvAlloc( isizs01 * sizeof(int) );
			for ( j = 0; j < isizs01; ++j )
				stat |= fscanf( R, "%d", &negdata[i][j] );
		}
		fclose( R );
		return 1;
	}
	return 0;
}

static int __ccv_write_background_data(const char* file, int** negdata, int negnum, ccv_size_t size)
{
	FILE* W = fopen( file, "w" );
	if ( W != NULL )
	{
		fprintf( W, "%d\n", negnum );
		int i, j;
		int isizs01 = size.width * size.height * 8 + ((size.width >> 1) - HOG_BORDER_SIZE) * ((size.height >> 1) - HOG_BORDER_SIZE) * 8;
		for ( i = 0; i < negnum; ++i )
		{
			for ( j = 0; j < isizs01; ++j )
				fprintf( W, "%d ", negdata[i][j] );
			fprintf( W, "\n" );
		}
		fclose( W );
		return 1;
	}
	return 0;
}

static int __ccv_resume_sgf_cascade_training_state(const char* file, int* i, int* k, int* bg, double* pw, double* nw, int posnum, int negnum)
{
	int stat = 0;
	FILE* R = fopen( file, "r" );
	if ( R != NULL )
	{
		stat |= fscanf( R, "%d %d %d", i, k, bg );
		int j;
		for ( j = 0; j < posnum; ++j )
			stat |= fscanf( R, "%le", &pw[j] );
		for ( j = 0; j < negnum; ++j )
			stat |= fscanf( R, "%le", &nw[j] );
		fclose( R );
		return 1;
	}
	return 0;
}

static int __ccv_save_sgf_cacade_training_state(const char* file, int i, int k, int bg, double* pw, double* nw, int posnum, int negnum)
{
	FILE* W = fopen( file, "w" );
	if ( W != NULL )
	{
		fprintf( W, "%d %d %d\n", i, k, bg );
		int j;
		for ( j = 0; j < posnum; ++j )
			fprintf( W, "%le ", pw[j] );
		fprintf(W, "\n");
		for ( j = 0; j < negnum; ++j )
			fprintf( W, "%le ", nw[j] );
		fprintf(W, "\n");
		fclose( W );
		return 1;
	}
	return 0;
}

void ccv_sgf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_sgf_params_t params)
{
	int i, j, k;
	/* allocate memory for usage */
	CvSGFClassifierCascade* cascade = (CvSGFClassifierCascade*)cvAlloc( sizeof(cascade[0]) );
	cascade->count = 0;
	cascade->size = size;
	cascade->stage_classifier = (CvSGFStageClassifier*)cvAlloc( sizeof(cascade->stage_classifier[0]) );
	int** posdata = (int**)cvAlloc( posnum * sizeof(posdata[0]) );
	icvPreparePositiveData( posimg, posdata, cascade->size, posnum );
	int** negdata = (int**)cvAlloc( negnum * sizeof(negdata[0]) );

	double* pw = (double*)cvAlloc( posnum * sizeof(pw[0]) );
	double* nw = (double*)cvAlloc( negnum * sizeof(nw[0]) );
	float* peval = (float*)cvAlloc( posnum * sizeof(peval[0]) );
	float* neval = (float*)cvAlloc( negnum * sizeof(neval[0]) );
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
	sprintf( buf, "%s/stat.txt", dir );
	icvResumeSGFCascadeTrainingState( buf, &i, &k, &bg, pw, nw, posnum, negnum );
	if ( i > 0 )
	{
		cascade->count = i;
		cvFree( &cascade->stage_classifier );
		cascade->stage_classifier = (CvSGFStageClassifier*)cvAlloc( i * sizeof(cascade->stage_classifier[0]) );
		for ( j = 0; j < i; ++j )
		{
			sprintf( buf, "%s/stage-%d.txt", dir, j );
			cvReadSGFStageClassifier( buf, &cascade->stage_classifier[j] );
		}
	}
	if ( k > 0 )
		cacheK = k;
	int rneg;
	if ( bg )
	{
		sprintf( buf, "%s/negs.txt", dir );
		icvReadBackgroundData( buf, negdata, &rneg, cascade->size );
	}
	for ( ; i < params.layer; ++i )
	{
		if ( !bg )
		{
			rneg = icvPrepareBackgroundData( cascade, bgfiles, bgnum, negdata, negnum );
			/* save state of background data */
			sprintf( buf, "%s/negs.txt", dir );
			icvWriteBackgroundData( buf, negdata, rneg, cascade->size );
			bg = 1;
		}
		double totalw;
		/* save state of cascade : level, weight etc. */
		sprintf( buf, "%s/stat.txt", dir );
		icvSaveSGFCacadeTrainingState( buf, i, k, bg, pw, nw, posnum, negnum );
		CvSGFStageClassifier classifier;
		if ( k > 0 )
		{
			/* resume state of classifier */
			sprintf( buf, "%s/stage-%d.txt", dir, i );
			cvReadSGFStageClassifier( buf, &classifier );
		} else {
			/* initialize classifier */
			totalw = params.balance_k * posnum + inv_balance_k * rneg;
			for ( j = 0; j < posnum; ++j )
				pw[j] = params.balance_k / totalw;
			for ( j = 0; j < rneg; ++j )
				nw[j] = inv_balance_k / totalw;
			classifier.count = k;
			classifier.threshold = 0;
			classifier.feature = (CvSGFeature*)cvAlloc( cacheK * sizeof(classifier.feature[0]) );
			classifier.alpha = (float*)cvAlloc( cacheK * 2 * sizeof(classifier.alpha[0]) );
		}
		icvPrunePositiveData( cascade, posdata, &posnum, cascade->size );
		printf("%d Postivie Data and %d Negative Data in Training\n", posnum, rneg);
		for ( ; ; ++k )
		{
			/* get overall true-positive, false-positive rate and threshold */
			double tp = 0, fp = 0, etp = 0, efp = 0;
			icvSGFEvalData( &classifier, posdata, posnum, negdata, rneg, cascade->size, peval, neval );
			icvSort_32f( peval, posnum, 0 );
			classifier.threshold = peval[(int)((1. - params.pos_crit) * posnum)] - 1e-6;
			for ( j = 0; j < posnum; ++j )
			{
				if ( peval[j] >= 0 )
					++tp;
				if ( peval[j] >= classifier.threshold )
					++etp;
			}
			tp /= posnum; etp /= posnum;
			for ( j = 0; j < rneg; ++j )
			{
				if ( neval[j] >= 0 )
					++fp;
				if ( neval[j] >= classifier.threshold )
					++efp;
			}
			fp /= rneg; efp /= rneg;
			printf( "Stage Classifier Real TP rate : %f, FP rate : %f\n", tp, fp );
			printf( "Stage Classifier TP rate : %f, FP rate : %f at Threshold : %f\n", etp, efp, classifier.threshold );
			if ( k > 0 )
			{
				/* save classifier state */
				sprintf( buf, "%s/stage-%d.txt", dir, i );
				cvWriteSGFStageClassifier( buf, &classifier );
				sprintf( buf, "%s/stat.txt", dir );
				icvSaveSGFCacadeTrainingState( buf, i, k, bg, pw, nw, posnum, negnum );
			}
			if ( etp > params.pos_crit && efp < params.neg_crit )
				break;
			/* TODO: more post-process is needed in here */

			/* select the best feature in current distribution through genetic algorithm optimization */
			CvSGFeature best = icvSGFGeneticOptimize( posdata, posnum, negdata, rneg, params.feature_number, cascade->size, pw, nw );
			double err = icvSGFErrorRate( &best, posdata, posnum, negdata, rneg, cascade->size, pw, nw );
			double rw = (1 - err) / err;
			totalw = 0;
			/* reweight */
			for ( j = 0; j < posnum; ++j )
			{
				int* i32c8[] = { posdata[j], posdata[j] + isizs0 };
				if ( !icvRunSGFeature( &best, steps, i32c8 ) )
					pw[j] *= rw;
				pw[j] *= params.balance_k;
				totalw += pw[j];
			}
			for ( j = 0; j < rneg; ++j )
			{
				int* i32c8[] = { negdata[j], negdata[j] + isizs0 };
				if ( icvRunSGFeature( &best, steps, i32c8 ) )
					nw[j] *= rw;
				nw[j] *= inv_balance_k;
				totalw += nw[j];
			}
			for ( j = 0; j < posnum; ++j )
				pw[j] = pw[j] / totalw;
			for ( j = 0; j < rneg; ++j )
				nw[j] = nw[j] / totalw;
			double c = log( rw );
			printf( "Coefficient of Feature %d: %f\n", k + 1, c );
			classifier.count = k + 1;
			/* resizing classifier */
			if ( k >= cacheK )
			{
				CvSGFeature* feature = (CvSGFeature*)cvAlloc( cacheK * 2 * sizeof(feature[0]) );
				memcpy( feature, classifier.feature, cacheK * sizeof(feature[0]) );
				cvFree( &classifier.feature );
				float* alpha = (float*)cvAlloc( cacheK * 4 * sizeof(alpha[0]) );
				memcpy( alpha, classifier.alpha, cacheK * 2 * sizeof(alpha[0]) );
				cvFree( &classifier.alpha );
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
		CvSGFStageClassifier* stage_classifier = (CvSGFStageClassifier*)cvAlloc( cascade->count * sizeof(stage_classifier[0]) );
		memcpy( stage_classifier, cascade->stage_classifier, i * sizeof(stage_classifier[0]) );
		cvFree( &cascade->stage_classifier );
		stage_classifier[i] = classifier;
		cascade->stage_classifier = stage_classifier;
		k = 0;
		bg = 0;
		for ( j = 0; j < rneg; ++j )
			cvFree( &negdata[j] );
	}

	cvFree( &neval );
	cvFree( &peval );
	cvFree( &nw );
	cvFree( &pw );
	cvFree( &negdata );
	cvFree( &posdata );
	cvFree( &cascade );
}

static int is_equal(const void* _r1, const void* _r2, void*)
{
	const ccv_sgf_comp_t* r1 = (const ccv_sgf_comp_t*)_r1;
	const ccv_sgf_comp_t* r2 = (const ccv_sgf_comp_t*)_r2;
	int distance = cvRound( r1->rect.width * 0.5 );

	return r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= cvRound( r1->rect.width * 1.5 ) &&
		   cvRound( r2->rect.width * 1.5 ) >= r1->rect.width;
}

static int is_equal_same_class(const void* _r1, const void* _r2, void*)
{
	const ccv_sgf_comp_t* r1 = (const ccv_sgf_comp_t*)_r1;
	const ccv_sgf_comp_t* r2 = (const ccv_sgf_comp_t*)_r2;
	int distance = cvRound( r1->rect.width * 0.5 );

	return r2->id == r1->id &&
		   r2->rect.x <= r1->rect.x + distance &&
		   r2->rect.x >= r1->rect.x - distance &&
		   r2->rect.y <= r1->rect.y + distance &&
		   r2->rect.y >= r1->rect.y - distance &&
		   r2->rect.width <= cvRound( r1->rect.width * 1.5 ) &&
		   cvRound( r2->rect.width * 1.5 ) >= r1->rect.width;
}

ccv_array_t* ccv_sgf_detect_objects(ccv_dense_matrix_t* a, ccv_sgf_classifier_cascade** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size)
{
	CvMat imghdr, *img = cvGetMat( _img, &imghdr );

	int hr = img->rows / min_size.height;
	int wr = img->cols / min_size.width;
	int scale_upto = (int)( log( (double)MIN( hr, wr ) ) / log( sqrt(2.) ) );
	/* generate scale-down HOG images */
	CvMat** pyr = (CvMat**)cvAlloc( (scale_upto + 2) * sizeof(pyr[0]) );
	if ( min_size.height != _cascade[0]->size.height || min_size.width != _cascade[0]->size.width )
	{
		pyr[0] = cvCreateMat( img->rows * _cascade[0]->size.height / min_size.height, img->cols * _cascade[0]->size.width / min_size.width, CV_8UC1 );
		cvResize( img, pyr[0], CV_INTER_AREA );
	} else
		pyr[0] = img;
	double sqrt_2 = sqrt(2.);
	pyr[1] = cvCreateMat( (int)(pyr[0]->rows / sqrt_2), (int)(pyr[0]->cols / sqrt_2), CV_8UC1 );
	cvResize( pyr[0], pyr[1], CV_INTER_AREA );
	int i, j, k, t, x, y;
	for ( i = 2; i < scale_upto + 2; i += 2 )
	{
		pyr[i] = cvCreateMat( pyr[i - 2]->rows >> 1, pyr[i - 2]->cols >> 1, CV_8UC1 );
		cvPyrDown( pyr[i - 2], pyr[i] );
	}
	for ( i = 3; i < scale_upto + 2; i += 2 )
	{
		pyr[i] = cvCreateMat( pyr[i - 2]->rows >> 1, pyr[i - 2]->cols >> 1, CV_8UC1 );
		cvPyrDown( pyr[i - 2], pyr[i] );
	}
	int* cols = (int*)cvAlloc( (scale_upto + 2) * sizeof(cols[0]) );
	int* rows = (int*)cvAlloc( (scale_upto + 2) * sizeof(rows[0]) );
	int** i32c8s = (int**)cvAlloc( (scale_upto + 2) * sizeof(i32c8s[0]) );
	for ( i = 0; i < scale_upto + 2; ++i )
	{
		rows[i] = pyr[i]->rows;
		cols[i] = pyr[i]->cols;
		i32c8s[i] = (int*)cvAlloc( (pyr[i]->rows - HOG_BORDER_SIZE * 2) * (pyr[i]->cols - HOG_BORDER_SIZE * 2) * 8 * sizeof(i32c8s[i][0]) );
		icvCreateHOG( pyr[i], i32c8s[i] );
	}
	for ( i = 1; i < scale_upto + 2; ++i )
		cvReleaseMat( &pyr[i] );
	if ( min_size.height != _cascade[0]->size.height || min_size.width != _cascade[0]->size.width )
		cvReleaseMat( &pyr[0] );
	cvFree( &pyr );

	CvMemStorage* temp_storage = cvCreateChildMemStorage( storage );
	CvSeq* idx_seq;
	CvSeq* seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSGFComp), temp_storage );
	CvSeq* seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSGFComp), temp_storage );
	CvSeq* result_seq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSGFComp), temp_storage );
	/* detect in multi scale */
	for ( t = 0; t < count; ++t )
	{
		CvSGFClassifierCascade* cascade = _cascade[t];
		float scale_x = (float) min_size.width / (float) cascade->size.width;
		float scale_y = (float) min_size.height / (float) cascade->size.height;
		cvClearSeq( seq );
		for ( i = 0; i < scale_upto; ++i )
		{
			int i_rows = rows[i + 2] - HOG_BORDER_SIZE * 2 - (cascade->size.height >> 1);
			int steps[] = { (cols[i] - HOG_BORDER_SIZE * 2) * 8, (cols[i + 2] - HOG_BORDER_SIZE * 2) * 8 };
			int cols_pads1 = cols[i + 2] - HOG_BORDER_SIZE * 2 - (cascade->size.width >> 1);
			int pads1 = (cascade->size.width >> 1) * 8;
			int pads0 = steps[0] * 2 - (cols_pads1 << 1) * 8;
			int* i32c8p[] = { i32c8s[i], i32c8s[i + 2] };
			for ( y = 0; y < i_rows; ++y )
			{
				for ( x = 0; x < cols_pads1; ++x )
				{
					float sum;
					int flag = 1;
					CvSGFStageClassifier* classifier = cascade->stage_classifier;
					for ( j = 0; j < cascade->count; ++j, ++classifier )
					{
						sum = 0;
						float* alpha = classifier->alpha;
						CvSGFeature* feature = classifier->feature;
						for ( k = 0; k < classifier->count; ++k, alpha += 2, ++feature )
							sum += alpha[icvRunSGFeature( feature, steps, i32c8p )];
						if ( sum < classifier->threshold )
						{
							flag = 0;
							break;
						}
					}
					if ( flag )
					{
						CvSGFComp comp;
						comp.rect = cvRect( (int) ((x * 2 + HOG_BORDER_SIZE) * scale_x), (int) ((y * 2 + HOG_BORDER_SIZE) * scale_y), (int) (cascade->size.width * scale_x), (int) (cascade->size.height * scale_y) );
						comp.id = t;
						comp.neighbors = 1;
						comp.confidence = sum;
						cvSeqPush( seq, &comp );
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
		if( min_neighbors == 0 )
		{
			for( i = 0; i < seq->total; i++ )
			{
				CvSGFComp* comp = (CvSGFComp*)cvGetSeqElem( seq, i );
				cvSeqPush( result_seq, comp );
			}
		} else {
			idx_seq = 0;
			cvClearSeq( seq2 );
			// group retrieved rectangles in order to filter out noise
			int ncomp = cvSeqPartition( seq, 0, &idx_seq, is_equal_same_class, 0 );
			CvSGFComp* comps = (CvSGFComp*)cvAlloc( (ncomp + 1) * sizeof(comps[0]) );
			memset( comps, 0, (ncomp + 1) * sizeof(comps[0]));

			// count number of neighbors
			for( i = 0; i < seq->total; ++i )
			{
				CvSGFComp r1 = *(CvSGFComp*)cvGetSeqElem( seq, i );
				int idx = *(int*)cvGetSeqElem( idx_seq, i );

				if (comps[idx].neighbors == 0)
					comps[idx].confidence = r1.confidence;

				++comps[idx].neighbors;

				comps[idx].rect.x += r1.rect.x;
				comps[idx].rect.y += r1.rect.y;
				comps[idx].rect.width += r1.rect.width;
				comps[idx].rect.height += r1.rect.height;
				comps[idx].id = r1.id;
				comps[idx].confidence = MAX( comps[idx].confidence, r1.confidence );
			}

			// calculate average bounding box
			for( i = 0; i < ncomp; ++i )
			{
				int n = comps[i].neighbors;
				if( n >= min_neighbors )
				{
					CvSGFComp comp;
					comp.rect.x = (comps[i].rect.x * 2 + n)/(2 * n);
					comp.rect.y = (comps[i].rect.y * 2 + n)/(2 * n);
					comp.rect.width = (comps[i].rect.width * 2 + n)/(2 * n);
					comp.rect.height = (comps[i].rect.height * 2 + n)/(2 * n);
					comp.neighbors = comps[i].neighbors;
					comp.id = comps[i].id;
					comp.confidence = comps[i].confidence;

					cvSeqPush( seq2, &comp );
				}
			}

			// filter out small face rectangles inside large face rectangles
			for( i = 0; i < seq2->total; ++i )
			{
				CvSGFComp r1 = *(CvSGFComp*)cvGetSeqElem( seq2, i );
				int flag = 1;

				for( j = 0; j < seq2->total; ++j )
				{
					CvSGFComp r2 = *(CvSGFComp*)cvGetSeqElem( seq2, j );
					int distance = cvRound( r2.rect.width * 0.5 );

					if( i != j &&
						r1.id == r2.id &&
						r1.rect.x >= r2.rect.x - distance &&
						r1.rect.y >= r2.rect.y - distance &&
						r1.rect.x + r1.rect.width <= r2.rect.x + r2.rect.width + distance &&
						r1.rect.y + r1.rect.height <= r2.rect.y + r2.rect.height + distance &&
						(r2.neighbors > MAX( 3, r1.neighbors ) || r1.neighbors < 3) )
					{
						flag = 0;
						break;
					}
				}

				if( flag )
					cvSeqPush( result_seq, &r1 );
			}
			cvFree( &comps );
		}
	}

	CvSeq* result_seq2 = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSGFComp), storage );
	/* the following code from OpenCV's haar feature implementation */
	if( flags & CV_SGF_NO_NESTED )
	{
		idx_seq = 0;
		// group retrieved rectangles in order to filter out noise
		int ncomp = cvSeqPartition( result_seq, 0, &idx_seq, is_equal, 0 );
		CvSGFComp* comps = (CvSGFComp*)cvAlloc( (ncomp + 1) * sizeof(comps[0]) );
		memset( comps, 0, (ncomp + 1) * sizeof(comps[0]));

		// count number of neighbors
		for( i = 0; i < result_seq->total; ++i )
		{
			CvSGFComp r1 = *(CvSGFComp*)cvGetSeqElem( result_seq, i );
			int idx = *(int*)cvGetSeqElem( idx_seq, i );

			if (comps[idx].neighbors == 0 || comps[idx].confidence < r1.confidence)
			{
				comps[idx].confidence = r1.confidence;
				comps[idx].neighbors = 1;
				comps[idx].rect = r1.rect;
				comps[idx].id = r1.id;
			}
		}

		// calculate average bounding box
		for( i = 0; i < ncomp; ++i )
			if( comps[i].neighbors )
				cvSeqPush( result_seq2, &comps[i] );

		cvFree( &comps );
	} else {
		for( i = 0; i < result_seq->total; i++ )
		{
			CvSGFComp* comp = (CvSGFComp*)cvGetSeqElem( result_seq, i );
			cvSeqPush( result_seq2, comp );
		}
	}

	cvReleaseMemStorage( &temp_storage );

	for ( i = 0; i < scale_upto + 2; ++i )
		cvFree( &i32c8s[i] );
	cvFree( &i32c8s );

	return result_seq2;
}

ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade(const char* directory)
{
	CvSGFClassifierCascade* cascade = (CvSGFClassifierCascade*)cvAlloc( sizeof(CvSGFClassifierCascade) );
	char buf[1024];
	sprintf( buf, "%s/cascade.txt", directory );
	int s, i;
	FILE* r = fopen( buf, "r" );
	if ( r != NULL )
		s = fscanf( r, "%d %d %d", &cascade->count, &cascade->size.width, &cascade->size.height );
	cascade->stage_classifier = (CvSGFStageClassifier*)cvAlloc( cascade->count * sizeof(cascade->stage_classifier[0]) );
	for ( i = 0; i < cascade->count; ++i )
	{
		sprintf( buf, "%s/stage-%d.txt", directory, i );
		if ( !cvReadSGFStageClassifier( buf, &cascade->stage_classifier[i] ) )
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
