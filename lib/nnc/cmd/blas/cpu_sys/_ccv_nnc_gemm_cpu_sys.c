#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include "../_ccv_nnc_gemm_cpu_opt.h"

int _ccv_nnc_gemm_forw_cpu_sys(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const w, const ccv_nnc_tensor_view_t* const bias, ccv_nnc_tensor_view_t* const b)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(!CCV_IS_TENSOR_VIEW(w));
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(b->info.dim[2] == 0); // It is a 2-d array.
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	const int* bdim = (b_nd == 1) ? b->info.dim : b->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	ccv_dense_matrix_t am = ccv_dense_matrix(batch_size, adim[0], CCV_32F | CCV_C1, a->data.u8, 0);
	assert(!bias || bdim[0] == ccv_nnc_tensor_count(bias->info));
	assert(batch_size == (b_nd == 1) ? 1 : ccv_max(1, b->info.dim[0]));
	ccv_dense_matrix_t bm = ccv_dense_matrix(batch_size, bdim[0], CCV_32F | CCV_C1, b->data.u8, 0);
	ccv_dense_matrix_t* dbm = &bm;
	// copy bias into each row.
	if (bias)
	{
		int i;
		for (i = 0; i < batch_size; i++)
			memcpy(bm.data.f32 + i * bdim[0], bias->data.f32, sizeof(float) * bdim[0]);
	} else
		memset(bm.data.f32, 0, sizeof(float) * bdim[0] * batch_size);
	assert(bdim[0] == w->info.dim[0]);
	assert(adim[0] == w->info.dim[1]);
	ccv_dense_matrix_t wm = ccv_dense_matrix(bdim[0], adim[0], CCV_32F | CCV_C1, w->data.u8, 0);
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
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		memset(dw->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(w->info));
		if (bias)
			memset(bias->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(bias->info));
	}
	assert(a->info.dim[2] == 0); // It is a 2-d array.
	assert(g->info.dim[2] == 0); // It is a 2-d array.
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == 1 || a_nd == 2);
	const int* adim = (a_nd == 1) ? a->info.dim : a->info.dim + 1;
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 1 || g_nd == 2);
	const int* gdim = (g_nd == 1) ? g->info.dim : g->info.dim + 1;
	const int batch_size = a_nd == 1 ? 1 : ccv_max(1, a->info.dim[0]);
	assert(batch_size == (g_nd == 1) ? 1 : ccv_max(1, g->info.dim[0]));
	assert(!bias || bias->info.dim[0] == gdim[0]);
	ccv_dense_matrix_t gm = ccv_dense_matrix(batch_size, gdim[0], CCV_32F | CCV_C1, g->data.u8, 0);
	ccv_dense_matrix_t am = ccv_dense_matrix(batch_size, adim[0], CCV_32F | CCV_C1, a->data.u8, 0);
	int i, j;
	float* gp = g->data.f32;
	if (bias)
	{
		float* bp = bias->data.f32;
		for (i = 0; i < batch_size; i++)
		{
			for (j = 0; j < gdim[0]; j++)
				bp[j] += gp[j];
			gp += gdim[0];
		}
	}
	assert(gdim[0] == w->info.dim[0]);
	assert(adim[0] == w->info.dim[1]);
	ccv_dense_matrix_t dwm = ccv_dense_matrix(gdim[0], adim[0], CCV_32F | CCV_C1, dw->data.u8, 0);
	ccv_dense_matrix_t* ddwm = &dwm;
	ccv_gemm(&gm, &am, 1, ddwm, 1, CCV_A_TRANSPOSE, (ccv_matrix_t**)&ddwm, 0);
	if (h && w)
	{
		assert(!CCV_IS_TENSOR_VIEW(h));
		assert(!CCV_IS_TENSOR_VIEW(w));
		assert(h->info.dim[2] == 0); // It is a 2-d array.
		assert(h->info.dim[0] == a->info.dim[0]);
		const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
		assert(h_nd == 1 || h_nd == 2);
		const int* hdim = (h_nd == 1) ? h->info.dim : h->info.dim + 1;
		assert(hdim[0] == adim[0]);
		assert(batch_size == (h_nd == 1) ? 1 : ccv_max(1, h->info.dim[0]));
		ccv_dense_matrix_t wm = ccv_dense_matrix(gdim[0], hdim[0], CCV_32F | CCV_C1, w->data.u8, 0);
		ccv_dense_matrix_t hm = ccv_dense_matrix(batch_size, hdim[0], CCV_32F | CCV_C1, h->data.u8, 0);
		ccv_dense_matrix_t* dhm = &hm;
		ccv_gemm(&gm, &wm, 1, 0, 0, 0 /* No transpose */, (ccv_matrix_t**)&dhm, 0);
	}
	return CCV_NNC_EXEC_SUCCESS;
#else
	return CCV_NNC_EXEC_INVALID;
#endif
}
