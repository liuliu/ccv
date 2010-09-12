#include "ccv.h"

inline static int __ccv_keypoint_interpolate(float N9[3][9], float te, ccv_keypoint_t* kp)
{
	float Dxx = (N9[1][3] - 2 * N9[1][4] + N9[1][5]) * 4; 
	float Dyy = (N9[1][1] - 2 * N9[1][4] + N9[1][7]) * 4;
	float Dxy = N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0];
	float Dxxyy = (Dxx + Dyy) * (Dxx + Dyy);
	float Dxyxy = (Dxx * Dyy - Dxy * Dxy);
	if (Dxxyy * te >= (te + 1) * (te + 1) * Dxyxy || (Dxxyy * Dxyxy) < 0)
		return -1;
	float Dx = (N9[1][3] - N9[1][5]) * 2;
	float Dy = (N9[1][1] - N9[1][7]) * 2;
	return 0;
}

ccv_array_t* ccv_sift(ccv_dense_matrix_t* a, ccv_sift_param_t params)
{
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * params.noctaves);
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * params.noctaves);
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	int i, j;
	double sigma0 = 1.6;
	double sigmak = pow(2.0, 1.0 / (params.nlevels - 3));
	double dsigma0 = sigma0 * sigmak * sqrt(1.0 - 1.0 / (sigmak * sigmak));
	double sd = sqrt(sigma0 * sigma0 - 0.25);
	ccv_convert(a, &g[0], CCV_32F, 0, 0);
	ccv_blur(g[0], &g[1], sd);
	for (j = 1; j < params.nlevels; j++)
	{
		sd = dsigma0 * pow(sigmak, j - 1);
		ccv_blur(g[j], &g[j + 1], sd);
		ccv_substract(g[j + 1], g[j], &dog[j - 1]);
		ccv_matrix_free(g[j]);
	}
	ccv_matrix_free(g[params.nlevels]);
	for (i = 1; i < params.noctaves; i++)
	{
		ccv_sample_down(g[(i - 1) * (params.nlevels + 1)], &g[i * (params.nlevels + 1)]);
		ccv_matrix_free(g[(i - 1) * (params.nlevels + 1)]);
		sd = sqrt(sigma0 * sigma0 - 0.25);
		ccv_blur(g[i * (params.nlevels + 1)], &g[i * (params.nlevels + 1) + 1], sd);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[i * (params.nlevels + 1) + j], &g[i * (params.nlevels + 1) + j + 1], sd);
			ccv_substract(g[i * (params.nlevels + 1) + j + 1], g[i * (params.nlevels + 1) + j], &dog[i * (params.nlevels - 1) + j - 1]);
			ccv_matrix_free(g[i * (params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[i * (params.nlevels + 1) + params.nlevels]);
	}
	ccv_matrix_free(g[(params.noctaves - 1) * (params.nlevels + 1)]);
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(a->rows, a->cols, CCV_8U | CCV_C1, 0, 0);
	memset(imx->data.ptr, 0, imx->rows * imx->step);
	int s = 1;
	int t = 0;
	for (i = 0; i < params.noctaves; i++)
	{
		int x, y;
		int rows = dog[i * (params.nlevels - 1)]->rows;
		int cols = dog[i * (params.nlevels - 1)]->cols;
		for (j = 1; j < params.nlevels - 2; j++)
		{
			float* bf = dog[i * (params.nlevels - 1) + j - 1]->data.fl + cols;
			float* cf = dog[i * (params.nlevels - 1) + j]->data.fl + cols;
			float* uf = dog[i * (params.nlevels - 1) + j + 1]->data.fl + cols;
			for (y = 1; y < rows - 1; y++)
			{
				for (x = 1; x < cols - 1; x++)
				{
					float v = cf[x];
#define locality_if(CMP, SGN) \
	(v CMP ## = SGN params.peak_threshold && v CMP cf[x - 1] && v CMP cf[x + 1] && \
	 v CMP cf[x - cols - 1] && v CMP cf[x - cols] && v CMP cf[x - cols + 1] && \
	 v CMP cf[x + cols - 1] && v CMP cf[x + cols] && v CMP cf[x + cols + 1] && \
	 v CMP bf[x - 1] && v CMP bf[x] && v CMP bf[x + 1] && \
	 v CMP bf[x - cols - 1] && v CMP bf[x - cols] && v CMP bf[x - cols + 1] && \
	 v CMP bf[x + cols - 1] && v CMP bf[x + cols] && v CMP bf[x + cols + 1] && \
	 v CMP uf[x - 1] && v CMP uf[x] && v CMP uf[x + 1] && \
	 v CMP uf[x - cols - 1] && v CMP uf[x - cols] && v CMP uf[x - cols + 1] && \
	 v CMP uf[x + cols - 1] && v CMP uf[x + cols] && v CMP uf[x + cols + 1])
					if (locality_if(<, -) || locality_if(>, +))
					{
						float N9[3][9] = { { bf[x - cols - 1], bf[x - cols], bf[x - cols + 1],
											 bf[x - 1], bf[x], bf[x + 1],
											 bf[x + cols - 1], bf[x + cols], bf[x + cols + 1] },
										   { cf[x - cols - 1], cf[x - cols], cf[x - cols + 1],
											 cf[x - 1], v, cf[x + 1],
											 cf[x + cols - 1], cf[x + cols], cf[x + cols + 1] },
										   { uf[x - cols - 1], uf[x - cols], uf[x - cols + 1],
											 uf[x - 1], uf[x], uf[x + 1],
											 uf[x + cols - 1], uf[x + cols], uf[x + cols + 1] } };
						ccv_keypoint_t kp;
						if (__ccv_keypoint_interpolate(N9, params.edge_threshold, &kp) == 0)
						{
							imx->data.ptr[x * s + y * s * imx->step] = 255;
							t++;
						}
					}
#undef locality_if
				}
				bf += cols;
				cf += cols;
				uf += cols;
			}
		}
		s *= 2;
	}
	printf("%d\n", t);
	int len;
	ccv_serialize(imx, "keypoint.png", &len, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(imx);
	for (i = 0; i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
	return 0;
}
