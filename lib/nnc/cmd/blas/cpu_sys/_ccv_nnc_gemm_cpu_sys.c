#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "../_ccv_nnc_gemm_cpu_opt.h"
#if HAVE_ACCELERATE_FRAMEWORK
#include <Accelerate/Accelerate.h>
#elif HAVE_CBLAS
#include <cblas.h>
#endif

static inline void _ccv_nnc_gbmm_and_bias(const float* const ones, const float* const a, const int a_nd, const int* const adim, const int* const astride, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, const float* const bias, const int bias_nd, const int* const biasdim, const int* const biasstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int transa, const int transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int bias_rows_inc, const int b_rows_inc)
{
	int i;
	if (b_nd <= 3)
	{
		for (i = 0; i < b_batch_size; i++)
		{
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, b_cols, b_rows, 1, 1.0, bias + i * bias_batch_inc, bias_rows_inc, ones, 1, 0.0, b + i * b_batch_inc, b_rows_inc);
			cblas_sgemm(CblasColMajor, transa, transb, b_cols, b_rows, a_cols, 1.0, w + i * w_batch_inc, lda_inc, a + i * a_batch_inc, ldb_inc, 1.0, b + i * b_batch_inc, b_rows_inc);
		}
		return;
	}
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	if (bias_nd > 3)
		{ assert(biasdim[0] == 1 || dim == biasdim[0]); }
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_gbmm_and_bias(ones,
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			bias_nd > 3 ? bias + i * biasstride[0] : bias, bias_nd > 3 ? bias_nd - 1 : bias_nd, bias_nd > 3 ? biasdim + 1 : biasdim, bias_nd > 3 ? biasstride + 1 : biasstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
	}
}

static inline void _ccv_nnc_gbmm(const float* const a, const int a_nd, const int* const adim, const int* const astride, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int transa, const int transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int b_rows_inc)
{
	int i;
	if (b_nd <= 3)
	{
		for (i = 0; i < b_batch_size; i++)
			cblas_sgemm(CblasColMajor, transa, transb, b_cols, b_rows, a_cols, 1.0, w + i * w_batch_inc, lda_inc, a + i * a_batch_inc, ldb_inc, 0.0, b + i * b_batch_inc, b_rows_inc);
		return;
	}
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_gbmm(
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
}

int _ccv_nnc_gemm_forw_cpu_sys(const int transpose_a[2], const int transpose_b[2], const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const w, const ccv_nnc_tensor_view_t* const bias, ccv_nnc_tensor_view_t* const b)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
	int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
	int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
	int b_batch_size, b_rows, b_cols, b_batch_inc, b_rows_inc, b_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	ccv_nnc_tensor_get_matrix_params(b->info, CCV_IS_TENSOR_VIEW(b) ? b->stride : 0, b->info.dim, no_transpose, &b_batch_size, &b_rows, &b_cols, &b_batch_inc, &b_rows_inc, &b_cols_inc);
	assert(a_batch_size == b_batch_size);
	assert(a_batch_size == b_batch_size || a_batch_size == 1);
	if (a_batch_size == 1 && b_batch_size > 1)
		a_batch_inc = 0;
	assert(w_batch_size == a_batch_size || w_batch_size == 1);
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
	assert(a_rows == b_rows);
	assert(a_cols == w_rows);
	assert(w_cols == b_cols);
	const int is_transpose_a = ccv_nnc_is_matrix_transpose(a->info, transpose_a);
	const int is_transpose_w = ccv_nnc_is_matrix_transpose(w->info, transpose_b);
	int astride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
	int wstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
	int bstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
	const int* astride;
	if (CCV_IS_TENSOR_VIEW(a))
		astride = a->stride;
	else {
		ccv_nnc_tensor_get_stride(a->info.dim, astride_from_dim);
		astride = astride_from_dim;
	}
	const int* wstride;
	if (CCV_IS_TENSOR_VIEW(w))
		wstride = w->stride;
	else {
		ccv_nnc_tensor_get_stride(w->info.dim, wstride_from_dim);
		wstride = wstride_from_dim;
	}
	const int* bstride;
	if (CCV_IS_TENSOR_VIEW(b))
		bstride = b->stride;
	else {
		ccv_nnc_tensor_get_stride(b->info.dim, bstride_from_dim);
		bstride = bstride_from_dim;
	}
	const int transa = is_transpose_w ? CblasTrans : CblasNoTrans;
	const int transb = is_transpose_a ? CblasTrans : CblasNoTrans;
	const int lda_inc = is_transpose_w ? w_cols_inc : w_rows_inc;
	const int ldb_inc = is_transpose_a ? a_cols_inc : a_rows_inc;
	if (bias)
	{
		float* const ones = (float*)ccmalloc(sizeof(float) * b_rows);
		int i;
		for (i = 0; i < b_rows; i++)
			ones[i] = 1;
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_batch_size == b_batch_size || bias_batch_size == 1);
		if (bias_batch_size == 1 && b_batch_size > 1)
			bias_batch_inc = 0;
		assert(bias_cols == b_cols);
		const int* biasstride;
		int biasstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		if (CCV_IS_TENSOR_VIEW(bias))
			biasstride = bias->stride;
		else {
			ccv_nnc_tensor_get_stride(bias->info.dim, biasstride_from_dim);
			biasstride = biasstride_from_dim;
		}
		_ccv_nnc_gbmm_and_bias(ones, a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, bias->data.f32, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
		ccfree(ones);
	} else {
		_ccv_nnc_gbmm(a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
#else
	return CCV_NNC_EXEC_INVALID;
#endif
}

int _ccv_nnc_gemm_back_cpu_sys(const int transpose_a[2], const int transpose_b[2], const ccv_nnc_tensor_view_t* const g, const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const w, ccv_nnc_tensor_view_t* const dw, ccv_nnc_tensor_view_t* const bias, ccv_nnc_tensor_view_t* const h, const int flags)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 2-d or 3-d array.
	int g_batch_size, g_rows, g_cols, g_batch_inc, g_rows_inc, g_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->stride : 0, g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
	int i;
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_cols == g_cols);
		assert(bias_batch_size == 1 || bias_batch_size == g_batch_size);
		if (bias_batch_size == 1 && g_batch_size > 1)
			bias_batch_inc = 0;
		float* const ones = (float*)ccmalloc(sizeof(float) * g_rows);
		for (i = 0; i < g_rows; i++)
			ones[i] = 1;
		if (g_batch_size > 1 && bias_batch_size == g_batch_size)
		{
			for (i = 0; i < g_batch_size; i++)
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, bias_cols, bias_rows, g_rows, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, ones, g_rows, 0.0, bias->data.f32 + i * bias_batch_inc, bias_rows_inc);
		} else {
			cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, bias_cols, bias_rows, g_rows, 1.0, g->data.f32, g_rows_inc, ones, g_rows, 0.0, bias->data.f32, bias_rows_inc);
			// We cannot use strided batched alternative because on write, the data could race to the same position
			for (i = 1; i < g_batch_size; i++)
				cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, bias_cols, bias_rows, g_rows, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, ones, g_rows, 1.0, bias->data.f32, bias_rows_inc);
		}
		ccfree(ones);
	}
	if (dw)
	{
		const int is_transpose_a = ccv_nnc_is_matrix_transpose(a->info, transpose_a);
		const int is_transpose_w = ccv_nnc_is_matrix_transpose(dw->info, transpose_b);
		int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
		int dw_batch_size, dw_rows, dw_cols, dw_batch_inc, dw_rows_inc, dw_cols_inc;
		ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
		ccv_nnc_tensor_get_matrix_params(dw->info, CCV_IS_TENSOR_VIEW(dw) ? dw->stride : 0, dw->info.dim, transpose_b, &dw_batch_size, &dw_rows, &dw_cols, &dw_batch_inc, &dw_rows_inc, &dw_cols_inc);
		assert(a_rows == g_rows);
		assert(a_cols == dw_rows);
		assert(dw_cols == g_cols);
		assert(a_batch_size == g_batch_size || a_batch_size == 1);
		if (a_batch_size == 1 && g_batch_size > 1)
			a_batch_inc = 0;
		assert(dw_batch_size == g_batch_size || dw_batch_size == 1);
		if (dw_batch_size == 1 && g_batch_size > 1)
			dw_batch_inc = 0;
		if (g_batch_size > 1 && g_batch_size == dw_batch_size)
		{
			if (is_transpose_w)
			{
				const int transa = is_transpose_a ? CblasTrans : CblasNoTrans;
				const int lda_inc = is_transpose_a ? a_cols_inc : a_rows_inc;
				for (i = 0; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, transa, CblasTrans, dw_rows, dw_cols, a_rows, 1.0, a->data.f32 + i * a_batch_inc, lda_inc, g->data.f32 + i * g_batch_inc, g_rows_inc, 0.0, dw->data.f32 + i * dw_batch_inc, dw_cols_inc);
			} else {
				const int transb = is_transpose_a ? CblasNoTrans : CblasTrans;
				const int ldb_inc = is_transpose_a ? a_cols_inc : a_rows_inc;
				for (i = 0; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, CblasNoTrans, transb, dw_cols, dw_rows, a_rows, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, a->data.f32 + i * a_batch_inc, ldb_inc, 0.0, dw->data.f32 + i * dw_batch_inc, dw_rows_inc);
			}
		} else {
			if (is_transpose_w)
			{
				const int transa = is_transpose_a ? CblasTrans : CblasNoTrans;
				const int lda_inc = is_transpose_a ? a_cols_inc : a_rows_inc;
				cblas_sgemm(CblasColMajor, transa, CblasTrans, dw_rows, dw_cols, a_rows, 1.0, a->data.f32, lda_inc, g->data.f32, g_rows_inc, 0.0, dw->data.f32, dw_cols_inc);
				for (i = 1; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, transa, CblasTrans, dw_rows, dw_cols, a_rows, 1.0, a->data.f32 + i * a_batch_inc, lda_inc, g->data.f32 + i * g_batch_inc, g_rows_inc, 1.0, dw->data.f32, dw_cols_inc);
			} else {
				const int transb = is_transpose_a ? CblasNoTrans : CblasTrans;
				const int ldb_inc = is_transpose_a ? a_cols_inc : a_rows_inc;
				cblas_sgemm(CblasColMajor, CblasNoTrans, transb, dw_cols, dw_rows, a_rows, 1.0, g->data.f32, g_rows_inc, a->data.f32, ldb_inc, 0.0, dw->data.f32, dw_rows_inc);
				for (i = 1; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, CblasNoTrans, transb, dw_cols, dw_rows, a_rows, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, a->data.f32 + i * a_batch_inc, ldb_inc, 1.0, dw->data.f32, dw_rows_inc);
			}
		}
	}
	if (h)
	{
		const int is_transpose_h = ccv_nnc_is_matrix_transpose(h->info, transpose_a);
		const int is_transpose_w = ccv_nnc_is_matrix_transpose(w->info, transpose_b);
		int h_batch_size, h_rows, h_cols, h_batch_inc, h_rows_inc, h_cols_inc;
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		ccv_nnc_tensor_get_matrix_params(h->info, CCV_IS_TENSOR_VIEW(h) ? h->stride : 0, h->info.dim, transpose_a, &h_batch_size, &h_rows, &h_cols, &h_batch_inc, &h_rows_inc, &h_cols_inc);
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		assert(h_rows == g_rows);
		assert(h_cols == w_rows);
		assert(w_cols == g_cols);
		assert(h_batch_size == g_batch_size || h_batch_size == 1);
		if (h_batch_size == 1 && g_batch_size > 1)
			h_batch_inc = 0;
		assert(w_batch_size == g_batch_size || w_batch_size == 1);
		if (w_batch_size == 1 && g_batch_size > 1)
			w_batch_inc = 0;
		if (g_batch_size > 1 && g_batch_size == h_batch_size)
		{
			if (is_transpose_h)
			{
				const int transb = is_transpose_w ? CblasTrans : CblasNoTrans;
				const int ldb_inc = is_transpose_w ? w_cols_inc : w_rows_inc;
				for (i = 0; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, CblasTrans, transb, h_rows, h_cols, g_cols, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, w->data.f32 + i * w_batch_inc, ldb_inc, 0.0, h->data.f32 + i * h_batch_inc, h_cols_inc);
			} else {
				const int transa = is_transpose_w ? CblasNoTrans : CblasTrans;
				const int lda_inc = is_transpose_w ? w_cols_inc : w_rows_inc;
				for (i = 0; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, transa, CblasNoTrans, h_cols, h_rows, g_cols, 1.0, w->data.f32 + i * w_batch_inc, lda_inc, g->data.f32 + i * g_batch_inc, g_rows_inc, 0.0, h->data.f32 + i * h_batch_inc, h_rows_inc);
			}
		} else {
			if (is_transpose_h)
			{
				const int transb = is_transpose_w ? CblasTrans : CblasNoTrans;
				const int ldb_inc = is_transpose_w ? w_cols_inc : w_rows_inc;
				cblas_sgemm(CblasColMajor, CblasTrans, transb, h_rows, h_cols, g_cols, 1.0, g->data.f32, g_rows_inc, w->data.f32, ldb_inc, 0.0, h->data.f32, h_cols_inc);
				for (i = 1; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, CblasTrans, transb, h_rows, h_cols, g_cols, 1.0, g->data.f32 + i * g_batch_inc, g_rows_inc, w->data.f32 + i * w_batch_inc, ldb_inc, 1.0, h->data.f32, h_cols_inc);
			} else {
				const int transa = is_transpose_w ? CblasNoTrans : CblasTrans;
				const int lda_inc = is_transpose_w ? w_cols_inc : w_rows_inc;
				cblas_sgemm(CblasColMajor, transa, CblasNoTrans, h_cols, h_rows, g_cols, 1.0, w->data.f32, lda_inc, g->data.f32, g_rows_inc, 0.0, h->data.f32, h_rows_inc);
				for (i = 1; i < g_batch_size; i++)
					cblas_sgemm(CblasColMajor, transa, CblasNoTrans, h_cols, h_rows, g_cols, 1.0, w->data.f32 + i * w_batch_inc, lda_inc, g->data.f32 + i * g_batch_inc, g_rows_inc, 1.0, h->data.f32, h_rows_inc);
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
#else
	return CCV_NNC_EXEC_INVALID;
#endif
}
