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

static inline void _ccv_nnc_bmm_and_bias(const float* const a, const float* const w, const float* const bias, float* const b, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int bias_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int bias_rows_inc, const int b_rows_inc)
{
	int n, i;
	for (n = 0; n < b_batch_size; n++)
	{
		const float* const ap = a + n * a_batch_inc;
		const float* const wp = w + n * w_batch_inc;
		const float* const biasp = bias + n * bias_batch_inc;
		float* const bp = b + n * b_batch_inc;
		for (i = 0; i < b_rows; i++)
		{
			const float* const api = ap + i * a_rows_inc;
			const float* const biaspi = biasp + i * bias_rows_inc;
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

static inline void _ccv_nnc_gbmm_and_bias(const float* const a, const int a_nd, const int* const adim, const int* const astride, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, const float* const bias, const int bias_nd, const int* const biasdim, const int* const biasstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int bias_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int bias_rows_inc, const int b_rows_inc)
{
	if (b_nd <= 3)
	{
		_ccv_nnc_bmm_and_bias(a, w, bias, b, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc);
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
		_ccv_nnc_gbmm_and_bias(
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			bias_nd > 3 ? bias + i * biasstride[0] : bias, bias_nd > 3 ? bias_nd - 1 : bias_nd, bias_nd > 3 ? biasdim + 1 : biasdim, bias_nd > 3 ? biasstride + 1 : biasstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc);
	}
}

static inline void _ccv_nnc_bmm(const float* const a, const float* const w, float* const b, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int b_rows_inc)
{
	int n, i;
	for (n = 0; n < b_batch_size; n++)
	{
		const float* const ap = a + n * a_batch_inc;
		const float* const wp = w + n * w_batch_inc;
		float* const bp = b + n * b_batch_inc;
		for (i = 0; i < b_rows; i++)
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

static inline void _ccv_nnc_gbmm(const float* const a, const int a_nd, const int* const adim, const int* const astride, const float* const w, const int w_nd, const int* const wdim, const int* const wstride, float* const b, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int a_cols_inc, const int w_cols_inc, const int b_cols_inc, const int a_rows_inc, const int w_rows_inc, const int b_rows_inc)
{
	if (b_nd <= 3)
	{
		_ccv_nnc_bmm(a, w, b, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc);
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
		_ccv_nnc_gbmm(
			a_nd > 3 ? a + i * astride[0] : a, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			w_nd > 3 ? w + i * wstride[0] : w, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			b + i * bstride[0], b_nd - 1, bdim + 1, bstride + 1, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc);
	}
}

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
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
	assert(ccv_max(a_batch_size, w_batch_size) == b_batch_size);
	assert(a_batch_size == b_batch_size || a_batch_size == 1);
	if (a_batch_size == 1 && b_batch_size > 1)
		a_batch_inc = 0;
	assert(w_batch_size == b_batch_size || w_batch_size == 1);
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
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
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_batch_size == b_batch_size || bias_batch_size == 1);
		if (bias_batch_size == 1 && b_batch_size > 1)
			bias_batch_inc = 0;
		if (bias_rows == 1 && b_rows > 1)
			bias_rows_inc = 0;
		assert(bias_cols == b_cols);
		const int* biasstride;
		int biasstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		if (CCV_IS_TENSOR_VIEW(bias))
			biasstride = bias->stride;
		else {
			ccv_nnc_tensor_get_stride(bias->info.dim, biasstride_from_dim);
			biasstride = biasstride_from_dim;
		}
		_ccv_nnc_gbmm_and_bias(a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, bias->data.f32, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, bias_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, bias_rows_inc, b_rows_inc);
	} else {
		_ccv_nnc_gbmm(a->data.f32, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w->data.f32, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, b->data.f32, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, a_cols_inc, w_cols_inc, b_cols_inc, a_rows_inc, w_rows_inc, b_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* bias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 2-d or 3-d array.
	int g_batch_size, g_rows, g_cols, g_batch_inc, g_rows_inc, g_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->stride : 0, g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
	int n, i;
	if (bias)
	{
		if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
			ccv_nnc_tensor_zero(bias);
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_cols == g_cols);
		assert(bias_batch_size == 1 || bias_batch_size == g_batch_size);
		if (bias_batch_size == 1 && g_batch_size > 1)
			bias_batch_inc = 0;
		if (bias_rows == 1 && g_rows > 1)
			bias_rows_inc = 0;
		int j;
		for (n = 0; n < g_batch_size; n++)
		{
			const float* const gp = g->data.f32 + n * g_batch_inc;
			float* const bp = bias->data.f32 + n * bias_batch_inc;
			for (i = 0; i < g_rows; i++)
			{
				const float* const gpi = gp + i * g_rows_inc;
				float* const bpi = bp + i * bias_rows_inc;
				for (j = 0; j < g_cols; j++)
					bpi[j * bias_cols_inc] += gpi[j * g_cols_inc];
			}
		}
	}
	if (dw)
	{
		if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
			ccv_nnc_tensor_zero(dw);
		const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
		int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
		int dw_batch_size, dw_rows, dw_cols, dw_batch_inc, dw_rows_inc, dw_cols_inc;
		ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
		ccv_nnc_tensor_get_matrix_params(dw->info, CCV_IS_TENSOR_VIEW(dw) ? dw->stride : 0, dw->info.dim, cmd.info.blas.transpose_b, &dw_batch_size, &dw_rows, &dw_cols, &dw_batch_inc, &dw_rows_inc, &dw_cols_inc);
		assert(a_rows == g_rows);
		assert(a_cols == dw_rows);
		assert(dw_cols == g_cols);
		assert(a_batch_size == g_batch_size || a_batch_size == 1);
		if (a_batch_size == 1 && g_batch_size > 1)
			a_batch_inc = 0;
		assert(dw_batch_size == g_batch_size || dw_batch_size == 1);
		if (dw_batch_size == 1 && g_batch_size > 1)
			dw_batch_inc = 0;
		for (n = 0; n < g_batch_size; n++)
		{
			const float* const gp = g->data.f32 + n * g_batch_inc;
			const float* const ap = a->data.f32 + n * a_batch_inc;
			float* const dwp = dw->data.f32 + n * dw_batch_inc;
			for (i = 0; i < a_rows; i++)
			{
				const float* const gpi = gp + i * g_rows_inc;
				const float* const api = ap + i * a_rows_inc;
				parallel_for(j, g_cols) {
					const float v = gpi[j * g_cols_inc];
					float* dwpj = dwp + j * dw_cols_inc;
					int k;
					for (k = 0; k < a_cols; k++)
						dwpj[k * dw_rows_inc] += api[k * a_cols_inc] * v;
				} parallel_endfor
			}
		}
	}
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h)
	{
		const int zero_h = !(flags & CCV_NNC_ACCUMULATE_OUTPUT); // reset the gradients to 0
		const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2];
		int h_batch_size, h_rows, h_cols, h_batch_inc, h_rows_inc, h_cols_inc;
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		ccv_nnc_tensor_get_matrix_params(h->info, CCV_IS_TENSOR_VIEW(h) ? h->stride : 0, h->info.dim, cmd.info.blas.transpose_a, &h_batch_size, &h_rows, &h_cols, &h_batch_inc, &h_rows_inc, &h_cols_inc);
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		assert(h_cols == w_rows);
		assert(w_cols == g_cols);
		assert(h_batch_size == g_batch_size || h_batch_size == 1);
		if (h_batch_size == 1 && g_batch_size > 1)
			h_batch_inc = 0;
		assert(w_batch_size == g_batch_size || w_batch_size == 1);
		if (w_batch_size == 1 && g_batch_size > 1)
			w_batch_inc = 0;
		for (n = 0; n < g_batch_size; n++)
		{
			const float* const gp = g->data.f32 + n * g_batch_inc;
			const float* const wp = w->data.f32 + n * w_batch_inc;
			float* const hp = h->data.f32 + n * h_batch_inc;
			for (i = 0; i < h_rows; i++)
			{
				const float* const gpi = gp + i * g_rows_inc;
				float* const hpi = hp + i * h_rows_inc;
				parallel_for(j, h_cols) {
					const float* const wpj = wp + j * w_rows_inc;
					float v = zero_h ? 0 : hpi[j * h_cols_inc];
					int k;
					for (k = 0; k < g_cols; k++)
						v += wpj[k * w_cols_inc] * gpi[k * g_cols_inc];
					hpi[j * h_cols_inc] = v;
				} parallel_endfor
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
}
