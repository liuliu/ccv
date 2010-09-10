#include "ccv.h"

ccv_array_t* ccv_sift(ccv_dense_matrix_t* a, ccv_sift_param_t params)
{
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves);
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * params.nlevels * params.noctaves);
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	int i, j;
	double sigmak = pow(2.0, 1.0 / (params.nlevels - 3));
	double sigma0 = 1.6 * sigmak;
	double dsigma0 = sigma0 * sqrt(1.0 - 1.0 / (sigmak * sigmak));
	ccv_convert(a, &g[0], CCV_32S, 8, 0);
	for (j = 1; j < params.nlevels; j++)
	{
		double sd = dsigma0 * pow(sigmak, j);
		ccv_blur(g[j - 1], &g[j], sd);
		ccv_substract(g[j], g[j - 1], &dog[j - 1]);
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
			double sd = dsigma0 * pow(sigmak, j);
			ccv_blur(g[i * params.nlevels + j - 1], &g[i * params.nlevels + j], sd);
			ccv_substract(g[i * params.nlevels + j], g[i * params.nlevels + j - 1], &dog[i * (params.nlevels - 1) + j - 1]);
			if (j > 1)
				ccv_matrix_free(g[i * params.nlevels + j - 1]);
		}
		ccv_matrix_free(g[i * params.nlevels + params.nlevels - 1]);
	}
	ccv_matrix_free(g[(params.noctaves - 1) * params.nlevels]);
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(a->rows, a->cols, CCV_8U | CCV_C1, 0, 0);
	memset(imx->data.ptr, 0, imx->rows * imx->step);
	int s = 1;
	for (i = 0; i < params.noctaves; i++)
	{
		int x, y;
		int rows = dog[i * (params.nlevels - 1)]->rows;
		int cols = dog[i * (params.nlevels - 1)]->cols;
		int step = dog[i * (params.nlevels - 1)]->step;
		for (j = 1; j < params.nlevels - 2; j++)
		{
			unsigned char* bptr = dog[i * (params.nlevels - 1) + j - 1]->data.ptr + step;
			unsigned char* cptr = dog[i * (params.nlevels - 1) + j]->data.ptr + step;
			unsigned char* uptr = dog[i * (params.nlevels - 1) + j + 1]->data.ptr + step;
			for (y = 1; y < rows - 1; y++)
			{
				for (x = 1; x < cols - 1; x++)
				{
					int v = ((int*)cptr)[x];
					if ((v < -params.peak_threshold && v < ((int*)cptr)[x - 1] && v < ((int*)cptr)[x + 1] &&
						 v < ((int*)(cptr - step))[x - 1] && v < ((int*)(cptr - step))[x] && v < ((int*)(cptr - step))[x + 1] &&
						 v < ((int*)(cptr + step))[x - 1] && v < ((int*)(cptr + step))[x] && v < ((int*)(cptr + step))[x + 1] &&
						 v < ((int*)bptr)[x - 1] && v < ((int*)bptr)[x] && v < ((int*)bptr)[x + 1] &&
						 v < ((int*)(bptr - step))[x - 1] && v < ((int*)(bptr - step))[x] && v < ((int*)(bptr - step))[x + 1] &&
						 v < ((int*)(bptr + step))[x - 1] && v < ((int*)(bptr + step))[x] && v < ((int*)(bptr + step))[x + 1] &&
						 v < ((int*)uptr)[x - 1] && v < ((int*)uptr)[x] && v < ((int*)uptr)[x + 1] &&
						 v < ((int*)(uptr - step))[x - 1] && v < ((int*)(uptr - step))[x] && v < ((int*)(uptr - step))[x + 1] &&
						 v < ((int*)(uptr + step))[x - 1] && v < ((int*)(uptr + step))[x] && v < ((int*)(uptr + step))[x + 1]) ||
						(v > params.peak_threshold && v > ((int*)cptr)[x - 1] && v > ((int*)cptr)[x + 1] &&
						 v > ((int*)(cptr - step))[x - 1] && v > ((int*)(cptr - step))[x] && v > ((int*)(cptr - step))[x + 1] &&
						 v > ((int*)(cptr + step))[x - 1] && v > ((int*)(cptr + step))[x] && v > ((int*)(cptr + step))[x + 1] &&
						 v > ((int*)bptr)[x - 1] && v > ((int*)bptr)[x] && v > ((int*)bptr)[x + 1] &&
						 v > ((int*)(bptr - step))[x - 1] && v > ((int*)(bptr - step))[x] && v > ((int*)(bptr - step))[x + 1] &&
						 v > ((int*)(bptr + step))[x - 1] && v > ((int*)(bptr + step))[x] && v > ((int*)(bptr + step))[x + 1] &&
						 v > ((int*)uptr)[x - 1] && v > ((int*)uptr)[x] && v > ((int*)uptr)[x + 1] &&
						 v > ((int*)(uptr - step))[x - 1] && v > ((int*)(uptr - step))[x] && v > ((int*)(uptr - step))[x + 1] &&
						 v > ((int*)(uptr + step))[x - 1] && v > ((int*)(uptr + step))[x] && v > ((int*)(uptr + step))[x + 1]))
					{
						imx->data.ptr[x * s + y * s * imx->step] = 255;
					}
				}
				bptr += step;
				cptr += step;
				uptr += step;
			}
		}
		s *= 2;
	}
	int len;
	ccv_serialize(imx, "keypoint.png", &len, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(imx);
	for (i = 0; i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
}
