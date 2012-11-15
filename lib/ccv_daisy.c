#include "ccv.h"
#include "ccv_internal.h"

/* the method is adopted from original author's published C++ code under BSD Licence.
 * Here is the copyright:
 * //////////////////////////////////////////////////////////////////////////
 * // Software License Agreement (BSD License)                             //
 * //                                                                      //
 * // Copyright (c) 2009                                                   //
 * // Engin Tola                                                           //
 * // web   : http://cvlab.epfl.ch/~tola                                   //
 * // email : engin.tola@epfl.ch                                           //
 * //                                                                      //
 * // All rights reserved.                                                 //
 * //                                                                      //
 * // Redistribution and use in source and binary forms, with or without   //
 * // modification, are permitted provided that the following conditions   //
 * // are met:                                                             //
 * //                                                                      //
 * //  * Redistributions of source code must retain the above copyright    //
 * //    notice, this list of conditions and the following disclaimer.     //
 * //  * Redistributions in binary form must reproduce the above           //
 * //    copyright notice, this list of conditions and the following       //
 * //    disclaimer in the documentation and/or other materials provided   //
 * //    with the distribution.                                            //
 * //  * Neither the name of the EPFL nor the names of its                 //
 * //    contributors may be used to endorse or promote products derived   //
 * //    from this software without specific prior written permission.     //
 * //                                                                      //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  //
 * // "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    //
 * // LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    //
 * // FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       //
 * // COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,  //
 * // INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, //
 * // BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;     //
 * // LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER     //
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT   //
 * // LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    //
 * // ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE      //
 * // POSSIBILITY OF SUCH DAMAGE.                                          //
 * //                                                                      //
 * // See licence.txt file for more details                                //
 * //////////////////////////////////////////////////////////////////////////
 */

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_daisy_param_t params)
{
	int grid_point_number = params.rad_q_no * params.th_q_no + 1;
	int desc_size = grid_point_number * params.hist_th_q_no;
	char* identifier = (char*)alloca(ccv_max(sizeof(ccv_daisy_param_t) + 9, sizeof(double) * params.rad_q_no));
	memset(identifier, 0, ccv_max(sizeof(ccv_daisy_param_t) + 9, sizeof(double) * params.rad_q_no));
	memcpy(identifier, "ccv_daisy", 9);
	memcpy(identifier + 9, &params, sizeof(ccv_daisy_param_t));
	uint64_t sig = (a->sig == 0) ? 0 : ccv_cache_generate_signature(identifier, sizeof(ccv_daisy_param_t) + 9, a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_32F | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols * desc_size, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	int layer_size = a->rows * a->cols;
	int cube_size = layer_size * params.hist_th_q_no;
	float* workspace_memory = (float*)ccmalloc(cube_size * (params.rad_q_no + 2) * sizeof(float));
	/* compute_cube_sigmas */
	int i, j, k, r, t;
	double* cube_sigmas = (double*)identifier;
	double r_step = params.radius / (double)params.rad_q_no;
	for (i = 0; i < params.rad_q_no; i++)
		cube_sigmas[i] = (i + 1) * r_step * 0.5;
	/* compute_grid_points */
	double t_step = 2 * 3.141592654 / params.th_q_no;
	double* grid_points = (double*)alloca(grid_point_number * 2 * sizeof(double));
	for (i = 0; i < params.rad_q_no; i++)
		for (j = 0; j < params.th_q_no; j++)
		{
			grid_points[(i * params.th_q_no + 1 + j) * 2] = sin(j * t_step) * (i + 1) * r_step;
			grid_points[(i * params.th_q_no + 1 + j) * 2 + 1] = cos(j * t_step) * (i + 1) * r_step;
		}
	/* TODO: require 0.5 gaussian smooth before gradient computing */
	/* NOTE: the default sobel already applied a sigma = 0.85 gaussian blur by using a
	 * | -1  0  1 |   |  0  0  0 |   | 1  2  1 |
	 * | -2  0  2 | = | -1  0  1 | * | 2  4  2 |
	 * | -1  0  1 |   |  0  0  0 |   | 1  2  1 | */
	ccv_dense_matrix_t* dx = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_sobel(a, &dx, 0, 1, 0);
	ccv_dense_matrix_t* dy = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_sobel(a, &dy, 0, 0, 1);
	double sobel_sigma = sqrt(0.5 / -log(0.5));
	double sigma_init = 1.6;
	double sigma = sqrt(sigma_init * sigma_init - sobel_sigma * sobel_sigma);
	/* layered_gradient & smooth_layers */
	for (k = params.hist_th_q_no - 1; k >= 0; k--)
	{
		float radius = k * 2 * 3.141592654 / params.th_q_no;
		float kcos = cos(radius);
		float ksin = sin(radius);
		float* w_ptr = workspace_memory + cube_size + (k - 1) * layer_size;
		for (i = 0; i < layer_size; i++)
			w_ptr[i] = ccv_max(0, kcos * dx->data.f32[i] + ksin * dy->data.f32[i]);
		ccv_dense_matrix_t src = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, w_ptr, 0);
		ccv_dense_matrix_t des = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, w_ptr + layer_size, 0);
		ccv_dense_matrix_t* desp = &des;
		ccv_blur(&src, &desp, 0, sigma);
	}
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
	/* compute_smoothed_gradient_layers & compute_histograms (rearrange memory) */
	for (k = 0; k < params.rad_q_no; k++)
	{
		sigma = (k == 0) ? cube_sigmas[0] : sqrt(cube_sigmas[k] * cube_sigmas[k] - cube_sigmas[k - 1] * cube_sigmas[k - 1]);
		float* src_ptr = workspace_memory + (k + 1) * cube_size;
		float* des_ptr = src_ptr + cube_size;
		for (i = 0; i < params.hist_th_q_no; i++)
		{
			ccv_dense_matrix_t src = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, src_ptr + i * layer_size, 0);
			ccv_dense_matrix_t des = ccv_dense_matrix(a->rows, a->cols, CCV_32F | CCV_C1, des_ptr + i * layer_size, 0);
			ccv_dense_matrix_t* desp = &des;
			ccv_blur(&src, &desp, 0, sigma);
		}
		float* his_ptr = src_ptr - cube_size;
		for (i = 0; i < layer_size; i++)
			for (j = 0; j < params.hist_th_q_no; j++)
				his_ptr[i * params.hist_th_q_no + j] = src_ptr[i + j * layer_size];
	}
	/* petals of the flower */
	memset(db->data.u8, 0, db->rows * db->step);
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			float* a_ptr = workspace_memory + i * params.hist_th_q_no * a->cols + j * params.hist_th_q_no;
			float* b_ptr = db->data.f32 + i * db->cols + j * desc_size;
			memcpy(b_ptr, a_ptr, params.hist_th_q_no * sizeof(float));
			for (r = 0; r < params.rad_q_no; r++)
			{
				int rdt = r * params.th_q_no + 1;
				for (t = rdt; t < rdt + params.th_q_no; t++)
				{
					double y = i + grid_points[t * 2];
					double x = j + grid_points[t * 2 + 1];
					int iy = (int)(y + 0.5);
					int ix = (int)(x + 0.5);
					float* bh = b_ptr + t * params.hist_th_q_no;
					if (iy < 0 || iy >= a->rows || ix < 0 || ix >= a->cols)
						continue;
					// bilinear interpolation
					int jy = (int)y;
					int jx = (int)x;
					float yr = y - jy, _yr = 1 - yr;
					float xr = x - jx, _xr = 1 - xr;
					if (jy >= 0 && jy < a->rows && jx >= 0 && jx < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + jy * params.hist_th_q_no * a->cols + jx * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * _yr * _xr;
					}
					if (jy + 1 >= 0 && jy + 1 < a->rows && jx >= 0 && jx < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + (jy + 1) * params.hist_th_q_no * a->cols + jx * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * yr * _xr;
					}
					if (jy >= 0 && jy < a->rows && jx + 1 >= 0 && jx + 1 < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + jy * params.hist_th_q_no * a->cols + (jx + 1) * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * _yr * xr;
					}
					if (jy + 1 >= 0 && jy + 1 < a->rows && jx + 1 >= 0 && jx + 1 < a->cols)
					{
						float* ah = workspace_memory + (r + 1) * cube_size + (jy + 1) * params.hist_th_q_no * a->cols + (jx + 1) * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							bh[k] += ah[k] * yr * xr;
					}
				}
			}
		}
	ccfree(workspace_memory);
	for (i = 0; i < a->rows; i++)
		for (j = 0; j < a->cols; j++)
		{
			float* b_ptr = db->data.f32 + i * db->cols + j * desc_size;
			float norm;
			int iter, changed;
			switch (params.normalize_method)
			{
				case CCV_DAISY_NORMAL_PARTIAL:
					for (t = 0; t < grid_point_number; t++)
					{
						norm = 0;
						float* bh = b_ptr + t * params.hist_th_q_no;
						for (k = 0; k < params.hist_th_q_no; k++)
							norm += bh[k] * bh[k];
						if (norm > 1e-6)
						{
							norm = 1.0 / sqrt(norm);
							for (k = 0; k < params.hist_th_q_no; k++)
								bh[k] *= norm;
						}
					}
					break;
				case CCV_DAISY_NORMAL_FULL:
					norm = 0;
					for (t = 0; t < desc_size; t++)
						norm += b_ptr[t] * b_ptr[t];
					if (norm > 1e-6)
					{
						norm = 1.0 / sqrt(norm);
						for (t = 0; t < desc_size; t++)
							b_ptr[t] *= norm;
					}
					break;
				case CCV_DAISY_NORMAL_SIFT:
					for (iter = 0, changed = 1; changed && iter < 5; iter++)
					{
						norm = 0;
						for (t = 0; t < desc_size; t++)
							norm += b_ptr[t] * b_ptr[t];
						changed = 0;
						if (norm > 1e-6)
						{
							norm = 1.0 / sqrt(norm);
							for (t = 0; t < desc_size; t++)
							{
								b_ptr[t] *= norm;
								if (b_ptr[t] < params.normalize_threshold)
								{
									b_ptr[t] = params.normalize_threshold;
									changed = 1;
								}
							}
						}
					}
					break;
			}
		}
}
