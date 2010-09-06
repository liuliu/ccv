#include "ccv.h"

ccv_array_t* ccv_sift(ccv_dense_matrix_t* a, ccv_sift_param_t params)
{
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves);
	memset(g, sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves, 0);
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	memset(dog, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves, 0);
	int i, j;
	g[0] = a;
	for (i = 1; i < params.noctaves; i++)
		ccv_sample_down(g[(i - 1) * params.nlevels], &g[i * params.nlevels]);
	for (i = 1; i < params.nlevels * params.noctaves; i++)
		ccv_matrix_free(g[i]);
}
