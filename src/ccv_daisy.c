#include "ccv.h"

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

void __ccv_gaussian_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int filter_size, double sigma)
{
}

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, ccv_daisy_param_t params)
{
	int grid_point_number = params.rad_q_no * params.th_q_no + 1;
	int desc_size = grid_point_number * params.hist_th_q_no;
	int sig[5];
	char* identifier = (char*)alloca(ccv_max(sizeof(ccv_daisy_param_t) + 9, sizeof(double) * params.rad_q_no));
	memset(identifier, 0, 64);
	memcpy(identifier, "ccv_daisy", 9);
	memcpy(identifier + 9, &params, sizeof(ccv_daisy_param_t));
	ccv_matrix_generate_signature(identifier, sizeof(ccv_daisy_param_t) + 9, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(a->rows, a->cols * desc_size, CCV_32F | CCV_C1, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
	}
	/* compute_cube_sigmas */
	int i, j, k;
	double* cube_sigmas = (double*)identifier;
	double r_step = (double)params.rad_q_no / params.radius;
	for (i = 0; i < params.rad_q_no; i++)
		cube_sigmas[i] = (i + 1) * r_step / 2;
	/* compute_grid_points */
	double t_step = 2 * 3.141592654 / params.th_q_no;
	double* grid_points = (double*)alloca(grid_point_number * 2);
	for (i = 0; i < params.rad_q_no; i++)
		for (j = 0; j < params.th_q_no; j++)
		{
			grid_points[(i * params.th_q_no + 1 + j) * 2] = cos(j * t_step) * (i + 1) * r_step;
			grid_points[(i * params.th_q_no + 1 + j) * 2 + 1] = sin(j * t_step) * (i + 1) * r_step;
		}
	/* TODO: require 0.5 gaussian smooth before gradient computing */
	ccv_dense_matrix_t* dx = NULL;
	ccv_sobel(a, &dx, 1, 0);
	ccv_dense_matrix_t* dy = NULL;
	ccv_sobel(a, &dy, 0, 1);
	/* layered_gradient */
	for (k = 0; k < params.th_q_no; k++)
	{
		float radius = k * 2 * 3.141592654 / params.th_q_no;
		float kcos = cos(radius);
		float ksin = sin(radius);
		float* b_ptr = db->data.fl + k * a->rows * a->cols;
		for (i = 0; i < a->rows * a->cols; i++)
			b_ptr[i] = ccv_max(0, kcos * dx->data.fl[i] + ksin * dy->data.fl[i]);
	}
}
