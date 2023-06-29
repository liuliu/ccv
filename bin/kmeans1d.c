#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>
#include <3rdparty/dsfmt/dSFMT.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_dense_matrix_t* const a = ccv_dense_matrix_new(1, 1000000, CCV_32F | CCV_C1, 0, 0);
	int i;
	for (i = 0; i < 1000000; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	int* const clusters = (int*)ccmalloc(sizeof(int) * 1000000);
	double* const centroids = (double*)ccmalloc(sizeof(double) * 128);
	unsigned int elapsed_time = get_current_time();
	ccv_kmeans1d(a, 128, clusters, centroids);
	elapsed_time = get_current_time() - elapsed_time;
	printf("elapsed: %ums\n", elapsed_time);
	ccfree(centroids);
	ccfree(clusters);
	ccv_matrix_free(a);
	return 0;
}
