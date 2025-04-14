#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static inline void _ccv_nnc_segmented_bmm_and_bias(const float* const a, const float* const w, const float* const bias, float* const b, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int bias_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int bias_rows_inc, const int b_rows_inc, const int* const indices, const int* const counts, const int bincount)
{
	assert(b_batch_size == 1);
	int n, i;
	int off = 0;
	for (n = 0; n < bincount; n++)
	{
		const float* const ap = a + off * a_rows_inc;
		const float* const wp = w + indices[n] * w_batch_inc;
		const float* const biasp = bias + indices[n] * bias_batch_inc;
		float* const bp = b + off * b_rows_inc;
		const int rowcount = counts[n];
		off += rowcount;
		for (i = 0; i < rowcount; i++)
		{
			const float* const api = ap + i * a_rows_inc;
			const float* const biaspi = biasp;
			float* const bpi = bp + i * b_rows_inc;
			parallel_for(j, b_cols) {
				float v = biaspi[j * bias_cols_inc];
				const float* const wpj = wp + j * w_cols_inc;
				int k;
				for (k = 0; k < a_cols; k++)
					v += wpj[k * w_rows_inc] * api[k * a_cols_inc];
				bpi[j * b_cols_inc] = v;
			} parallel_endfor
		}
	}
}

static inline void _ccv_nnc_segmented_gbmm_and_bias(const float* const a, const int a_nd, const int* const adim, const int* const astride, const int* const indices, const int* const counts, const int bincount, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, const float* const bias, const int bias_nd, const int* const biasdim, const int* const biasstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int bias_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int bias_rows_inc, const int b_rows_inc)
{
	if (b_nd <= 3)
	{
		_ccv_nnc_segmented_bmm_and_bias(a, w, bias, b, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc, indices, counts, bincount);
		return;
	}
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	if (bias_nd > 3)
		{ assert(biasdim[0] == 1 || dim == biasdim[0]); }
	int i;
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_segmented_gbmm_and_bias(
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			indices, counts, bincount,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			bias_nd > 3 ? bias + i * biasstride[0] : bias, bias_nd > 3 ? bias_nd - 1 : bias_nd, bias_nd > 3 ? biasdim + 1 : biasdim, bias_nd > 3 ? biasstride + 1 : biasstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc);
	}
}

static inline void _ccv_nnc_segmented_bmm(const float* const a, const float* const w, float* const b, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int b_rows_inc, const int* const indices, const int* const counts, const int bincount)
{
	assert(b_batch_size == 1);
	int n, i;
	int off = 0;
	for (n = 0; n < bincount; n++)
	{
		const float* const ap = a + off * a_rows_inc;
		const float* const wp = w + indices[n] * w_batch_inc;
		float* const bp = b + off * b_rows_inc;
		const int rowcount = counts[n];
		off += rowcount;
		for (i = 0; i < rowcount; i++)
		{
			const float* const api = ap + i * a_rows_inc;
			float* const bpi = bp + i * b_rows_inc;
			parallel_for(j, b_cols) {
				float v = 0;
				const float* const wpj = wp + j * w_cols_inc;
				int k;
				for (k = 0; k < a_cols; k++)
					v += wpj[k * w_rows_inc] * api[k * a_cols_inc];
				bpi[j * b_cols_inc] = v;
			} parallel_endfor
		}
	}
}

static inline void _ccv_nnc_segmented_gbmm(const float* const a, const int a_nd, const int* const adim, const int* const astride, const int* const indices, const int* const counts, const int bincount, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int b_rows_inc)
{
	if (b_nd <= 3)
	{
		_ccv_nnc_segmented_bmm(a, w, b, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc, indices, counts, bincount);
		return;
	}
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	int i;
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_segmented_gbmm(
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			indices, counts, bincount,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc);
	}
}

static int _ccv_nnc_segmented_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* indices = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* counts = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* bias = input_size > 4 ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0;
	// Copy the most of parameters, but reshape the dimension of a to a vector.
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
	int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
	int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
	int b_batch_size, b_rows, b_cols, b_batch_inc, b_rows_inc, b_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	ccv_nnc_tensor_get_matrix_params(b->info, CCV_IS_TENSOR_VIEW(b) ? b->stride : 0, b->info.dim, no_transpose, &b_batch_size, &b_rows, &b_cols, &b_batch_inc, &b_rows_inc, &b_cols_inc);
	assert(a_batch_size == 1); // Currently, a cannot be batched (no broadcast support too).
	assert(a_batch_size == b_batch_size);
	assert(a_rows == b_rows);
	assert(a_cols == w_rows);
	assert(w_cols == b_cols);
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
	const int bincount = ccv_nnc_tensor_count(indices->info);
	assert(ccv_nnc_tensor_count(counts->info) == bincount);
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(CCV_IS_TENSOR_CONTIGUOUS(counts));
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		const int bias_nd = ccv_nnc_tensor_nd(bias->info.dim);
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		if (bias_nd == 2) // For nd == 2, we expand rows to 1 and assign that to batch.
		{
			bias_batch_size = bias_rows;
			bias_rows = 1;
			bias_batch_inc = bias_rows_inc;
			bias_rows_inc = 0;
		}
		assert(bias_batch_size == w_batch_size);
		assert(bias_cols == b_cols);
		const int* biasstride;
		int biasstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		if (CCV_IS_TENSOR_VIEW(bias))
			biasstride = bias->stride;
		else {
			ccv_nnc_tensor_get_stride(bias->info.dim, biasstride_from_dim);
			biasstride = biasstride_from_dim;
		}
		_ccv_nnc_segmented_gbmm_and_bias(a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, indices->data.i32, counts->data.i32, bincount, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, bias->data.f32, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc);
	} else {
		_ccv_nnc_segmented_gbmm(a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, indices->data.i32, counts->data.i32, bincount, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_segmented_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_GEMM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_gemm_back;
}
