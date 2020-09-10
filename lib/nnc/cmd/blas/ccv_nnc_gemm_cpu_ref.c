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
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	ccv_nnc_tensor_get_matrix_params(b->info, CCV_IS_TENSOR_VIEW(b) ? b->inc : b->info.dim, no_transpose, &b_batch_size, &b_rows, &b_cols, &b_batch_inc, &b_rows_inc, &b_cols_inc);
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
	int n, i;
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->inc : bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_batch_size == b_batch_size || bias_batch_size == 1);
		if (bias_batch_size == 1 && b_batch_size > 1)
			bias_batch_inc = 0;
		if (bias_rows == 1 && b_rows > 1)
			bias_rows_inc = 0;
		assert(bias_cols == b_cols);
		for (n = 0; n < b_batch_size; n++)
		{
			const float* const ap = a->data.f32 + n * a_batch_inc;
			const float* const wp = w->data.f32 + n * w_batch_inc;
			const float* const biasp = bias->data.f32 + n * bias_batch_inc;
			float* const bp = b->data.f32 + n * b_batch_inc;
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
	} else {
		for (n = 0; n < b_batch_size; n++)
		{
			const float* const ap = a->data.f32 + n * a_batch_inc;
			const float* const wp = w->data.f32 + n * w_batch_inc;
			float* const bp = b->data.f32 + n * b_batch_inc;
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
	ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
	int n, i;
	if (bias)
	{
		if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
			ccv_nnc_tensor_zero(bias);
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->inc : bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
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
		ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->inc : a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
		ccv_nnc_tensor_get_matrix_params(dw->info, CCV_IS_TENSOR_VIEW(dw) ? dw->inc : dw->info.dim, cmd.info.blas.transpose_b, &dw_batch_size, &dw_rows, &dw_cols, &dw_batch_inc, &dw_rows_inc, &dw_cols_inc);
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
		ccv_nnc_tensor_get_matrix_params(h->info, CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim, cmd.info.blas.transpose_a, &h_batch_size, &h_rows, &h_cols, &h_batch_inc, &h_rows_inc, &h_cols_inc);
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
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
