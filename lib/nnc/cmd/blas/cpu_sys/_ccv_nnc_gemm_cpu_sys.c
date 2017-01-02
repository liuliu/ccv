#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include "../_ccv_nnc_gemm_cpu_opt.h"

int _ccv_nnc_gemm_forw_cpu_sys(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const w, const ccv_nnc_tensor_view_t* const bias, ccv_nnc_tensor_view_t* const b)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	assert(!CCV_IS_TENSOR_VIEW(w));
	assert(!CCV_IS_TENSOR_VIEW(bias));
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	ccv_dense_matrix_t am = ccv_dense_matrix(ccv_max(1, a->info.dim[1]), a->info.dim[0], CCV_32F | CCV_C1, a->data.u8, 0);
	assert(!CCV_IS_TENSOR_VIEW(b));
	int bias_count = ccv_nnc_tensor_count(bias->info);
	assert(b->info.dim[0] == bias_count);
	assert(b->info.dim[2] == 0); // It is a 2-d array.
	assert(ccv_max(1, b->info.dim[1]) == ccv_max(1, a->info.dim[1]));
	ccv_dense_matrix_t bm = ccv_dense_matrix(ccv_max(1, b->info.dim[1]), b->info.dim[0], CCV_32F | CCV_C1, b->data.u8, 0);
	ccv_dense_matrix_t* dbm = &bm;
	// copy bias into each row.
	int i;
	for (i = 0; i < ccv_max(1, b->info.dim[1]); i++)
		memcpy(bm.data.f32 + i * b->info.dim[0], bias->data.f32, sizeof(float) * b->info.dim[0]);
	assert(a->info.dim[0] == w->info.dim[0]);
	assert(b->info.dim[0] == w->info.dim[1]);
	ccv_dense_matrix_t wm = ccv_dense_matrix(b->info.dim[0], a->info.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
	ccv_gemm(&am, &wm, 1, dbm, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&dbm, 0); // supply b as matrix C is allowed
	return CCV_NNC_EXEC_SUCCESS;
#else
	return CCV_NNC_EXEC_INVALID;
#endif
}

int _ccv_nnc_gemm_back_cpu_sys(const ccv_nnc_tensor_view_t* const g, const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const w, ccv_nnc_tensor_view_t* const dw, ccv_nnc_tensor_view_t* const bias, ccv_nnc_tensor_view_t* const h, const int flags)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	assert(!CCV_IS_TENSOR_VIEW(g));
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(!CCV_IS_TENSOR_VIEW(dw));
	assert(!CCV_IS_TENSOR_VIEW(bias));
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(dw->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	assert(ccv_max(1, a->info.dim[1]) == ccv_max(1, g->info.dim[1]));
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	ccv_dense_matrix_t gm = ccv_dense_matrix(ccv_max(1, g->info.dim[1]), g->info.dim[0], CCV_32F | CCV_C1, g->data.u8, 0);
	assert(bias->info.dim[0] == g->info.dim[0]);
	ccv_dense_matrix_t am = ccv_dense_matrix(ccv_max(1, a->info.dim[1]), a->info.dim[0], CCV_32F | CCV_C1, a->data.u8, 0);
	int i, j;
	float* gp = g->data.f32;
	float* bp = bias->data.f32;
	for (i = 0; i < ccv_max(1, g->info.dim[1]); i++)
	{
		for (j = 0; j < g->info.dim[0]; j++)
			bp[j] += gp[j];
		gp += g->info.dim[0];
	}
	assert(a->info.dim[0] == w->info.dim[0]);
	assert(g->info.dim[0] == w->info.dim[1]);
	ccv_dense_matrix_t dwm = ccv_dense_matrix(g->info.dim[0], a->info.dim[0], CCV_32F | CCV_C1, dw->data.u8, 0);
	ccv_dense_matrix_t* ddwm = &dwm;
	ccv_gemm(&gm, &am, 1, ddwm, 1, CCV_A_TRANSPOSE, (ccv_matrix_t**)&ddwm, 0);
	if (h && w)
	{
		assert(!CCV_IS_TENSOR_VIEW(h));
		assert(!CCV_IS_TENSOR_VIEW(w));
		assert(h->info.dim[0] == a->info.dim[0]);
		assert(ccv_max(1, h->info.dim[1]) == ccv_max(1, a->info.dim[1]));
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		ccv_dense_matrix_t wm = ccv_dense_matrix(g->info.dim[0], h->info.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
		ccv_dense_matrix_t hm = ccv_dense_matrix(ccv_max(1, h->info.dim[1]), h->info.dim[0], CCV_32F | CCV_C1, h->data.u8, 0);
		ccv_dense_matrix_t* dhm = &hm;
		ccv_gemm(&gm, &wm, 1, 0, 0, 0 /* No transpose */, (ccv_matrix_t**)&dhm, 0);
	}
	return CCV_NNC_EXEC_SUCCESS;
#else
	return CCV_NNC_EXEC_INVALID;
#endif
}
