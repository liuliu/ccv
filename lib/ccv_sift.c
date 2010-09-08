#include "ccv.h"

ccv_array_t* ccv_sift(ccv_dense_matrix_t* a, ccv_sift_param_t params)
{
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves);
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves);
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	int i, j;
	ccv_convert(a, &g[0], CCV_32S, 8, 0);
	for (j = 1; j < params.nlevels; j++)
	{
		ccv_blur(g[j - 1], &g[j], params.sigma);
		ccv_substract(g[j - 1], g[j], &dog[j - 1]);
		if (j > 1)
			ccv_matrix_free(g[j - 1]);
	}
	ccv_matrix_free(g[params.nlevels - 1]);
	for (i = 1; i < params.noctaves; i++)
	{
		ccv_sample_down(g[(i - 1) * params.nlevels], &g[i * params.nlevels]);
		ccv_matrix_free(g[(i - 1) * params.nlevels]);
		for (j = 1; j < params.nlevels; j++)
		{
			ccv_blur(g[i * params.nlevels + j - 1], &g[i * params.nlevels + j], params.sigma);
			ccv_substract(g[i * params.nlevels + j - 1], g[i * params.nlevels + j], &dog[i * (params.nlevels - 1) + j - 1]);
			if (j > 1)
				ccv_matrix_free(g[i * params.nlevels + j - 1]);
		}
		ccv_matrix_free(g[i * params.nlevels + params.nlevels - 1]);
	}
	for (i = 0; i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
}
