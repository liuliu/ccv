#include "ccv.h"

inline static double __ccv_keypoint_interpolate(float N9[3][9], int ix, int iy, int is, ccv_keypoint_t* kp)
{
	double Dxx = N9[1][3] - 2 * N9[1][4] + N9[1][5]; 
	double Dyy = N9[1][1] - 2 * N9[1][4] + N9[1][7];
	double Dxy = (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25;
	double score = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
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
		for (ii = i - 1; ii >= 0; ii--)
		  b[ii] -= x * A[ii][i];
	}
	kp->x = ix + ccv_min(ccv_max(b[0], -1), 1);
	kp->y = iy + ccv_min(ccv_max(b[1], -1), 1);
	kp->regular.scale = is + b[2];
	return score;
}

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** _keypoints, ccv_dense_matrix_t** _desc, int type, ccv_sift_param_t params)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * params.noctaves);
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * params.noctaves);
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * params.noctaves);
	ccv_dense_matrix_t** th = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * params.noctaves);
	memset(th, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * params.noctaves);
	ccv_dense_matrix_t** md = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * params.noctaves);
	memset(md, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * params.noctaves);
	ccv_array_t* keypoints = *_keypoints;
	int custom_keypoints = 0;
	if (keypoints == 0)
		keypoints = *_keypoints = ccv_array_new(10, sizeof(ccv_keypoint_t));
	else
		custom_keypoints = 1;
	int i, j, k, x, y;
	double sigma0 = 1.6;
	double sigmak = pow(2.0, 1.0 / (params.nlevels - 3));
	double dsigma0 = sigma0 * sigmak * sqrt(1.0 - 1.0 / (sigmak * sigmak));
	double sd = sqrt(sigma0 * sigma0 - 0.25);
	g[0] = a;
	ccv_blur(g[0], &g[1], CCV_32F | CCV_C1, sd);
	for (j = 1; j < params.nlevels; j++)
	{
		sd = dsigma0 * pow(sigmak, j - 1);
		ccv_blur(g[j], &g[j + 1], 0, sd);
		ccv_substract(g[j + 1], g[j], (ccv_matrix_t**)&dog[j - 1], 0);
		if (j > 1 && j < params.nlevels - 1)
			ccv_gradient(g[j], &th[j - 2], 0, &md[j - 2], 0);
		ccv_matrix_free(g[j]);
	}
	ccv_matrix_free(g[params.nlevels]);
	for (i = 1; i < params.noctaves; i++)
	{
		ccv_sample_down(g[(i - 1) * (params.nlevels + 1)], &g[i * (params.nlevels + 1)], 0);
		if (i - 1 > 0)
			ccv_matrix_free(g[(i - 1) * (params.nlevels + 1)]);
		sd = sqrt(sigma0 * sigma0 - 0.25);
		g[i * (params.nlevels + 1) + 1] = ccv_dense_matrix_new(g[i * (params.nlevels + 1)]->rows, g[i * (params.nlevels + 1)]->cols, CCV_C1 | CCV_32F, 0, 0);
		ccv_blur(g[i * (params.nlevels + 1)], &g[i * (params.nlevels + 1) + 1], CCV_32F | CCV_C1, sd);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[i * (params.nlevels + 1) + j], &g[i * (params.nlevels + 1) + j + 1], 0, sd);
			ccv_substract(g[i * (params.nlevels + 1) + j + 1], g[i * (params.nlevels + 1) + j], (ccv_matrix_t**)&dog[i * (params.nlevels - 1) + j - 1], 0);
			if (j > 1 && j < params.nlevels - 1)
				ccv_gradient(g[i * (params.nlevels + 1) + j], &th[i * (params.nlevels - 3) + j - 2], 0, &md[i * (params.nlevels - 3) + j - 2], 0);
			ccv_matrix_free(g[i * (params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[i * (params.nlevels + 1) + params.nlevels]);
	}
	ccv_matrix_free(g[(params.noctaves - 1) * (params.nlevels + 1)]);
	int s = 1;
	if (!custom_keypoints)
	{
		for (i = 0; i < params.noctaves; i++)
		{
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
							ccv_keypoint_t kp;
							int ix = x, iy = y;
							double score = -1;
							int cvg = 0;
							int offset = ix + (iy - y) * cols;
							for (k = 0; k < 5; k++)
							{
								offset = ix + (iy - y) * cols;
								float N9[3][9] = { { bf[offset - cols - 1], bf[offset - cols], bf[offset - cols + 1],
													 bf[offset - 1], bf[offset], bf[offset + 1],
													 bf[offset + cols - 1], bf[offset + cols], bf[offset + cols + 1] },
												   { cf[offset - cols - 1], cf[offset - cols], cf[offset - cols + 1],
													 cf[offset - 1], cf[offset], cf[offset + 1],
													 cf[offset + cols - 1], cf[offset + cols], cf[offset + cols + 1] },
												   { uf[offset - cols - 1], uf[offset - cols], uf[offset - cols + 1],
													 uf[offset - 1], uf[offset], uf[offset + 1],
													 uf[offset + cols - 1], uf[offset + cols], uf[offset + cols + 1] } };
								score = __ccv_keypoint_interpolate(N9, ix, iy, j, &kp);
								if (kp.x >= 0 && kp.x <= cols - 1 && kp.y >= 0 && kp.y <= rows - 1)
								{
									int nx = (int)(kp.x + 0.5);
									int ny = (int)(kp.y + 0.5);
									if (ix == nx && iy == ny)
										break;
									ix = nx;
									iy = ny;
								} else {
									cvg = -1;
									break;
								}
							}
							if (cvg == 0 && fabs(cf[offset]) > params.peak_threshold && score >= 0 && score < (params.edge_threshold + 1) * (params.edge_threshold + 1) / params.edge_threshold)
							{
								kp.x *= s;
								kp.y *= s;
								kp.octave = i;
								kp.level = j;
								ccv_array_push(keypoints, &kp);
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
	}
	double const winf = 1.5;
	for (i = 0; i < keypoints->rnum; i++)
	{
		ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
		double s = pow(2.0, kp->octave);
		double x = kp->x / s;
		double y = kp->y / s;
		double sigma = sigma0 * sigmak * pow(2.0, kp->level / (params.nlevels - 3));
		int ix = (int)(x + 0.5);
		int iy = (int)(y + 0.5);
		double const sigmaw = winf * sigma;
		int wz = ccv_max((int)(3.0 * sigmaw + 0.5), 1);
		ccv_dense_matrix_t* tho = th[kp->octave * (params.nlevels - 3) + kp->level - 1];
		ccv_dense_matrix_t* mdo = md[kp->octave * (params.nlevels - 3) + kp->level - 1];
		if (ix >= 0 && ix < tho->cols && iy >=0 && iy < tho->rows)
		{
			for (y = -wz; y <= wz; y++)
			{
				for (x = ix - wz; x <= ix + wz; x++)
				{
				}
			}
		}
	}
	for (i = 0; i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
	for (i = 0; i < (params.nlevels - 3) * params.noctaves; i++)
	{
		ccv_matrix_free(th[i]);
		ccv_matrix_free(md[i]);
	}
}
