/* The code is adopted from VLFeat with heavily rewrite.
 * The original code is licenced under 2-clause BSD license,
 * should be compatible with New BSD Licence used by ccv.
 * The original Copyright:
 *
 * Copyright (C) 2007-12, Andrea Vedaldi and Brian Fulkerson
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ccv.h"
#include "ccv_internal.h"

const ccv_sift_param_t ccv_sift_default_params = {
	.noctaves = 3,
	.nlevels = 6,
	.up2x = 1,
	.edge_threshold = 10,
	.norm_threshold = 0,
	.peak_threshold = 0,
};

inline static double _ccv_keypoint_interpolate(float N9[3][9], int ix, int iy, int is, ccv_keypoint_t* kp)
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
		int maxi = j;
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

static float _ccv_mod_2pi(float x)
{
	while (x > 2 * CCV_PI)
		x -= 2 * CCV_PI;
	while (x < 0)
		x += 2 * CCV_PI;
	return x;
}

static int _ccv_floor(float x)
{
	int xi = (int)x;
	if (x >= 0 || (float)xi == x)
		return xi;
	return xi - 1;
}

#define EXPN_SZ  256 /* fast_expn table size */
#define EXPN_MAX 25.0 /* fast_expn table max */
static double _ccv_expn_tab[EXPN_SZ + 1]; /* fast_expn table */
static int _ccv_expn_init = 0;

static inline double _ccv_expn(double x)
{
	double a, b, r;
	int i;
	assert(0 <= x && x <= EXPN_MAX);
	if (x > EXPN_MAX)
		return 0.0;
	x *= EXPN_SZ / EXPN_MAX;
	i = (int)x;
	r = x - i;
	a = _ccv_expn_tab[i];
	b = _ccv_expn_tab[i + 1];
	return a + r * (b - a);
}

static void _ccv_precomputed_expn()
{
	int i;
	for(i = 0; i < EXPN_SZ + 1; i++)
		_ccv_expn_tab[i] = exp(-(double)i * (EXPN_MAX / EXPN_SZ));
	_ccv_expn_init = 1;
}

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** _keypoints, ccv_dense_matrix_t** _desc, int type, ccv_sift_param_t params)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	ccv_dense_matrix_t** g = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(g, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels + 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** dog = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(dog, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 1) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** th = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(th, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	ccv_dense_matrix_t** md = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	memset(md, 0, sizeof(ccv_dense_matrix_t*) * (params.nlevels - 3) * (params.up2x ? params.noctaves + 1 : params.noctaves));
	if (params.up2x)
	{
		g += params.nlevels + 1;
		dog += params.nlevels - 1;
		th += params.nlevels - 3;
		md += params.nlevels - 3;
	}
	ccv_array_t* keypoints = *_keypoints;
	int custom_keypoints = 0;
	if (keypoints == 0)
		keypoints = *_keypoints = ccv_array_new(sizeof(ccv_keypoint_t), 10, 0);
	else
		custom_keypoints = 1;
	int i, j, k, x, y;
	double sigma0 = 1.6;
	double sigmak = pow(2.0, 1.0 / (params.nlevels - 3));
	double dsigma0 = sigma0 * sigmak * sqrt(1.0 - 1.0 / (sigmak * sigmak));
	if (params.up2x)
	{
		ccv_sample_up(a, &g[-(params.nlevels + 1)], 0, 0, 0);
		/* since there is a gaussian filter in sample_up function already,
		 * the default sigma for upsampled image is sqrt(2) */
		double sd = sqrt(sigma0 * sigma0 - 2.0);
		ccv_blur(g[-(params.nlevels + 1)], &g[-(params.nlevels + 1) + 1], CCV_32F | CCV_C1, sd);
		ccv_matrix_free(g[-(params.nlevels + 1)]);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[-(params.nlevels + 1) + j], &g[-(params.nlevels + 1) + j + 1], 0, sd);
			ccv_subtract(g[-(params.nlevels + 1) + j + 1], g[-(params.nlevels + 1) + j], (ccv_matrix_t**)&dog[-(params.nlevels - 1) + j - 1], 0);
			if (j > 1 && j < params.nlevels - 1)
				ccv_gradient(g[-(params.nlevels + 1) + j], &th[-(params.nlevels - 3) + j - 2], 0, &md[-(params.nlevels - 3) + j - 2], 0, 1, 1);
			ccv_matrix_free(g[-(params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[-1]);
	}
	double sd = sqrt(sigma0 * sigma0 - 0.25);
	g[0] = a;
	/* generate gaussian pyramid (g, dog) & gradient pyramid (th, md) */
	ccv_blur(g[0], &g[1], CCV_32F | CCV_C1, sd);
	for (j = 1; j < params.nlevels; j++)
	{
		sd = dsigma0 * pow(sigmak, j - 1);
		ccv_blur(g[j], &g[j + 1], 0, sd);
		ccv_subtract(g[j + 1], g[j], (ccv_matrix_t**)&dog[j - 1], 0);
		if (j > 1 && j < params.nlevels - 1)
			ccv_gradient(g[j], &th[j - 2], 0, &md[j - 2], 0, 1, 1);
		ccv_matrix_free(g[j]);
	}
	ccv_matrix_free(g[params.nlevels]);
	for (i = 1; i < params.noctaves; i++)
	{
		ccv_sample_down(g[(i - 1) * (params.nlevels + 1)], &g[i * (params.nlevels + 1)], 0, 0, 0);
		if (i - 1 > 0)
			ccv_matrix_free(g[(i - 1) * (params.nlevels + 1)]);
		sd = sqrt(sigma0 * sigma0 - 0.25);
		ccv_blur(g[i * (params.nlevels + 1)], &g[i * (params.nlevels + 1) + 1], CCV_32F | CCV_C1, sd);
		for (j = 1; j < params.nlevels; j++)
		{
			sd = dsigma0 * pow(sigmak, j - 1);
			ccv_blur(g[i * (params.nlevels + 1) + j], &g[i * (params.nlevels + 1) + j + 1], 0, sd);
			ccv_subtract(g[i * (params.nlevels + 1) + j + 1], g[i * (params.nlevels + 1) + j], (ccv_matrix_t**)&dog[i * (params.nlevels - 1) + j - 1], 0);
			if (j > 1 && j < params.nlevels - 1)
				ccv_gradient(g[i * (params.nlevels + 1) + j], &th[i * (params.nlevels - 3) + j - 2], 0, &md[i * (params.nlevels - 3) + j - 2], 0, 1, 1);
			ccv_matrix_free(g[i * (params.nlevels + 1) + j]);
		}
		ccv_matrix_free(g[i * (params.nlevels + 1) + params.nlevels]);
	}
	ccv_matrix_free(g[(params.noctaves - 1) * (params.nlevels + 1)]);
	if (!custom_keypoints)
	{
		/* detect keypoint */
		for (i = (params.up2x ? -1 : 0); i < params.noctaves; i++)
		{
			double s = pow(2.0, i);
			int rows = dog[i * (params.nlevels - 1)]->rows;
			int cols = dog[i * (params.nlevels - 1)]->cols;
			for (j = 1; j < params.nlevels - 2; j++)
			{
				float* bf = dog[i * (params.nlevels - 1) + j - 1]->data.f32 + cols;
				float* cf = dog[i * (params.nlevels - 1) + j]->data.f32 + cols;
				float* uf = dog[i * (params.nlevels - 1) + j + 1]->data.f32 + cols;
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
							/* iteratively converge to meet subpixel accuracy */
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
								score = _ccv_keypoint_interpolate(N9, ix, iy, j, &kp);
								if (kp.x >= 1 && kp.x <= cols - 2 && kp.y >= 1 && kp.y <= rows - 2)
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
							if (cvg == 0 && fabs(cf[offset]) > params.peak_threshold && score >= 0 && score < (params.edge_threshold + 1) * (params.edge_threshold + 1) / params.edge_threshold && kp.regular.scale > 0 && kp.regular.scale < params.nlevels - 1)
							{
								kp.x *= s;
								kp.y *= s;
								kp.octave = i;
								kp.level = j;
								kp.regular.scale = sigma0 * sigmak * pow(2.0, kp.regular.scale / (double)(params.nlevels - 3));
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
		}
	}
	/* repeatable orientation/angle (p.s. it will push more keypoints (with different angles) to array) */
	float const winf = 1.5;
	double bins[36];
	int kpnum = keypoints->rnum;
	if (!_ccv_expn_init)
		_ccv_precomputed_expn();
	for (i = 0; i < kpnum; i++)
	{
		ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
		float ds = pow(2.0, kp->octave);
		float dx = kp->x / ds;
		float dy = kp->y / ds;
		int ix = (int)(dx + 0.5);
		int iy = (int)(dy + 0.5);
		float const sigmaw = winf * kp->regular.scale;
		int wz = ccv_max((int)(3.0 * sigmaw + 0.5), 1);
		ccv_dense_matrix_t* tho = th[kp->octave * (params.nlevels - 3) + kp->level - 1];
		ccv_dense_matrix_t* mdo = md[kp->octave * (params.nlevels - 3) + kp->level - 1];
		assert(tho->rows == mdo->rows && tho->cols == mdo->cols);
		if (ix >= 0 && ix < tho->cols && iy >=0 && iy < tho->rows)
		{
			float* theta = tho->data.f32 + ccv_max(iy - wz, 0) * tho->cols;
			float* magnitude = mdo->data.f32 + ccv_max(iy - wz, 0) * mdo->cols;
			memset(bins, 0, 36 * sizeof(double));
			/* oriented histogram with bilinear interpolation */
			for (y = ccv_max(iy - wz, 0); y <= ccv_min(iy + wz, tho->rows - 1); y++)
			{
				for (x = ccv_max(ix - wz, 0); x <= ccv_min(ix + wz, tho->cols - 1); x++)
				{
					float r2 = (x - dx) * (x - dx) + (y - dy) * (y - dy);
					if (r2 > wz * wz + 0.6)
						continue;
					float weight = _ccv_expn(r2 / (2.0 * sigmaw * sigmaw));
					float fbin = theta[x] * 0.1;
					int ibin = _ccv_floor(fbin - 0.5);
					float rbin = fbin - ibin - 0.5;
					/* bilinear interpolation */
					bins[(ibin + 36) % 36] += (1 - rbin) * magnitude[x] * weight;
					bins[(ibin + 1) % 36] += rbin * magnitude[x] * weight;
				}
				theta += tho->cols;
				magnitude += mdo->cols;
			}
			/* smoothing histogram */
			for (j = 0; j < 6; j++)
			{
				double first = bins[0];
				double prev = bins[35];
				for (k = 0; k < 35; k++)
				{
					double nb = (prev + bins[k] + bins[k + 1]) / 3.0;
					prev = bins[k];
					bins[k] = nb;
				}
				bins[35] = (prev + bins[35] + first) / 3.0;
			}
			int maxib = 0;
			for (j = 1; j < 36; j++)
				if (bins[j] > bins[maxib])
					maxib = j;
			double maxb = bins[maxib];
			double bm = bins[(maxib + 35) % 36];
			double bp = bins[(maxib + 1) % 36];
			double di = -0.5 * (bp - bm) / (bp + bm - 2 * maxb);
			kp->regular.angle = 2 * CCV_PI * (maxib + di + 0.5) / 36.0;
			maxb *= 0.8;
			for (j = 0; j < 36; j++)
				if (j != maxib)
				{
					bm = bins[(j + 35) % 36];
					bp = bins[(j + 1) % 36];
					if (bins[j] > maxb && bins[j] > bm && bins[j] > bp)
					{
						di = -0.5 * (bp - bm) / (bp + bm - 2 * bins[j]);
						ccv_keypoint_t nkp = *kp;
						nkp.regular.angle = 2 * CCV_PI * (j + di + 0.5) / 36.0;
						ccv_array_push(keypoints, &nkp);
					}
				}
		}
	}
	/* calculate descriptor */
	if (_desc != 0)
	{
		ccv_dense_matrix_t* desc = *_desc = ccv_dense_matrix_new(keypoints->rnum, 128, CCV_32F | CCV_C1, 0, 0);
		float* fdesc = desc->data.f32;
		memset(fdesc, 0, sizeof(float) * keypoints->rnum * 128);
		for (i = 0; i < keypoints->rnum; i++)
		{
			ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
			float ds = pow(2.0, kp->octave);
			float dx = kp->x / ds;
			float dy = kp->y / ds;
			int ix = (int)(dx + 0.5);
			int iy = (int)(dy + 0.5);
			double SBP = 3.0 * kp->regular.scale;
			int wz = ccv_max((int)(SBP * sqrt(2.0) * 2.5 + 0.5), 1);
			ccv_dense_matrix_t* tho = th[kp->octave * (params.nlevels - 3) + kp->level - 1];
			ccv_dense_matrix_t* mdo = md[kp->octave * (params.nlevels - 3) + kp->level - 1];
			assert(tho->rows == mdo->rows && tho->cols == mdo->cols);
			assert(ix >= 0 && ix < tho->cols && iy >=0 && iy < tho->rows);
			float* theta = tho->data.f32 + ccv_max(iy - wz, 0) * tho->cols;
			float* magnitude = mdo->data.f32 + ccv_max(iy - wz, 0) * mdo->cols;
			float ca = cos(kp->regular.angle);
			float sa = sin(kp->regular.angle);
			float sigmaw = 2.0;
			/* sidenote: NBP = 4, NBO = 8 */
			for (y = ccv_max(iy - wz, 0); y <= ccv_min(iy + wz, tho->rows - 1); y++)
			{
				for (x = ccv_max(ix - wz, 0); x <= ccv_min(ix + wz, tho->cols - 1); x++)
				{
					float nx = (ca * (x - dx) + sa * (y - dy)) / SBP;
					float ny = (-sa * (x - dx) + ca * (y - dy)) / SBP;
					float nt = 8.0 * _ccv_mod_2pi(theta[x] * CCV_PI / 180.0 - kp->regular.angle) / (2.0 * CCV_PI);
					float weight = _ccv_expn((nx * nx + ny * ny) / (2.0 * sigmaw * sigmaw));
					int binx = _ccv_floor(nx - 0.5);
					int biny = _ccv_floor(ny - 0.5);
					int bint = _ccv_floor(nt);
					float rbinx = nx - (binx + 0.5);
					float rbiny = ny - (biny + 0.5);
					float rbint = nt - bint;
					int dbinx, dbiny, dbint;
					/* Distribute the current sample into the 8 adjacent bins*/
					for(dbinx = 0; dbinx < 2; dbinx++)
						for(dbiny = 0; dbiny < 2; dbiny++)
							for(dbint = 0; dbint < 2; dbint++)
								if (binx + dbinx >= -2 && binx + dbinx < 2 && biny + dbiny >= -2 && biny + dbiny < 2)
									fdesc[(2 + biny + dbiny) * 32 + (2 + binx + dbinx) * 8 + (bint + dbint) % 8] += weight * magnitude[x] * fabs(1 - dbinx - rbinx) * fabs(1 - dbiny - rbiny) * fabs(1 - dbint - rbint);
				}
				theta += tho->cols;
				magnitude += mdo->cols;
			}
			ccv_dense_matrix_t tm = ccv_dense_matrix(1, 128, CCV_32F | CCV_C1, fdesc, 0);
			ccv_dense_matrix_t* tmp = &tm;
 			double norm = ccv_normalize(&tm, (ccv_matrix_t**)&tmp, 0, CCV_L2_NORM);
			int num = (ccv_min(iy + wz, tho->rows - 1) - ccv_max(iy - wz, 0) + 1) * (ccv_min(ix + wz, tho->cols - 1) - ccv_max(ix - wz, 0) + 1);
			if (params.norm_threshold && norm < params.norm_threshold * num)
			{
				for (j = 0; j < 128; j++)
					fdesc[j] = 0;
			} else {
				for (j = 0; j < 128; j++)
					if (fdesc[j] > 0.2)
						fdesc[j] = 0.2;
				ccv_normalize(&tm, (ccv_matrix_t**)&tmp, 0, CCV_L2_NORM);
			}
			fdesc += 128;
		}
	}
	for (i = (params.up2x ? -(params.nlevels - 1) : 0); i < (params.nlevels - 1) * params.noctaves; i++)
		ccv_matrix_free(dog[i]);
	for (i = (params.up2x ? -(params.nlevels - 3) : 0); i < (params.nlevels - 3) * params.noctaves; i++)
	{
		ccv_matrix_free(th[i]);
		ccv_matrix_free(md[i]);
	}
}
