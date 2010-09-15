#include "ccv.h"

inline static int __ccv_keypoint_interpolate(float N9[3][9], float te, int ix, int iy, int is, ccv_keypoint_t* kp)
{
	double Dxx = N9[1][3] - 2 * N9[1][4] + N9[1][5]; 
	double Dyy = N9[1][1] - 2 * N9[1][4] + N9[1][7];
	double Dxy = (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25;
	double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
	if (score >= (te + 1) * (te + 1) / te  || score < 0)
		return -1;
	double Dx = (N9[1][5] - N9[1][3]) * 0.5;
	double Dy = (N9[1][7] - N9[1][1]) * 0.5;
	double Ds = (N9[2][4] - N9[0][4]) * 0.5;
	double Dxs = (N9[2][5] + N9[0][3] - N9[2][3] - N9[0][5]) * 0.25;
	double Dys = (N9[2][7] + N9[0][1] - N9[2][1] - N9[0][7]) * 0.25;
	double Dss = N9[0][4] - 2 * N9[1][4] + N9[2][4];
	double A[3][3] = { { Dxx, Dxy, Dxs },
					   { Dxy, Dyy, Dys },
					   { Dxs, Dys, Dss } };
	double b[3] = { -Dx, -Dy, -Ds };
	/* Gauss elimination */
	int i, j, ii, jj;
	for(j = 0; j < 3; j++)
	{
		double maxa = 0;
		double maxabsa = 0;
		int maxi = -1;
		double tmp;

		/* look for the maximally stable pivot */
		for (i = j; i < 3; i++)
		{
			double a = A[i][j];
			double absa = fabs(a);
			if (absa > maxabsa)
			{
				maxa = a;
				maxabsa = absa;
				maxi = i;
			}
		}

		/* if singular give up */
		if (maxabsa < 1e-10f)
		{
			b[0] = b[1] = b[2] = 0;
			break;
		}

		i = maxi;

		/* swap j-th row with i-th row and normalize j-th row */
		for(jj = j; jj < 3; jj++)
		{
			tmp = A[i][jj];
			A[i][jj] = A[j][jj];
			A[j][jj] = tmp;
			A[j][jj] /= maxa;
		}
		tmp = b[j];
		b[j] = b[i];
		b[i] = tmp;
		b[j] /= maxa;

		/* elimination */
		for (ii = j + 1; ii < 3; ii++)
		{
			double x = A[ii][j];
			for (jj = j; jj < 3; jj++)
				A[ii][jj] -= x * A[j][jj];
			b[ii] -= x * b[j];
		}
	}

	/* backward substitution */
	for (i = 2; i > 0; i--)
	{
		double x = b[i];
		for (ii = i - 1; ii >= 0; ii++)
		  b[ii] -= x * A[ii][i];
	}
	kp->x = (ix + b[0]) * is;
	kp->y = (iy + b[1]) * is;
	kp->regular.scale = is + b[2];
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
						if (__ccv_keypoint_interpolate(N9, params.edge_threshold, x, y, s, &kp) == 0 && kp.x >= 0 && kp.x <= a->cols -1 && kp.y >= 0 && kp.y <= a->rows - 1)
						{
							int ix = (int)(kp.x + 0.5);
							int iy = (int)(kp.y + 0.5);
							imx->data.ptr[ix + iy * imx->step] = 255;
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
