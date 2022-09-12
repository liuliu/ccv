extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
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
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size));
	static const float one = 1;
	static const float zero = 0;
	const int transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
	const int transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->inc : bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_batch_size == b_batch_size || bias_batch_size == 1);
		if (bias_batch_size == 1 && b_batch_size > 1)
			bias_batch_inc = 0;
		assert(bias_cols == b_cols);
		const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, b_rows, b->info.datatype);
		if (b_batch_size == 1)
		{
			CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, &one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, device_ones, ccv_nnc_cuda_datatype(b->info.datatype), 1, &zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			const cublasOperation_t transa = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
			const cublasOperation_t transb = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
			const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
			const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &one, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
#if CUDA_VERSION >= 9100
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, &one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, bias_batch_inc, device_ones, ccv_nnc_cuda_datatype(b->info.datatype), 1, 0, &zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
			int i;
			for (i = 0; i < b_batch_size; i++)
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, &one, bias->data.u8 + CCV_GET_DATA_TYPE_SIZE(bias->info.datatype) * i * bias_batch_inc, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, device_ones, ccv_nnc_cuda_datatype(b->info.datatype), 1, &zero, b->data.u8 + CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * i * b_batch_inc, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
			const cublasOperation_t transa = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
			const cublasOperation_t transb = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
			const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
			const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
#if CUDA_VERSION >= 9100
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, w_batch_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, a_batch_inc, &one, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
			for (i = 0; i < b_batch_size; i++)
				CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &one, b->data.u8 + CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * i * b_batch_inc, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
		}
	} else {
		if (b_batch_size == 1)
		{
			const cublasOperation_t transa = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
			const cublasOperation_t transb = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
			const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
			const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			const cublasOperation_t transa = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
			const cublasOperation_t transb = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
			const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
			const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
#if CUDA_VERSION >= 9100
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, w_batch_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, a_batch_inc, &zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
			int i;
			for (i = 0; i < b_batch_size; i++)
				CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, &one, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &zero, b->data.u8 + CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * i * b_batch_inc, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#endif
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
	static const float one = 1;
	static const float zero = 0;
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size));
	int g_batch_size, g_rows, g_cols, g_batch_inc, g_rows_inc, g_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->inc : g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
	int i;
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->inc : bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_cols == g_cols);
		assert(bias_batch_size == 1 || bias_batch_size == g_batch_size);
		if (bias_batch_size == 1 && g_batch_size > 1)
			bias_batch_inc = 0;
		const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, g_rows, bias->info.datatype);
		if (g_batch_size > 1 && bias_batch_size == g_batch_size)
		{
#if CUDA_VERSION >= 9100
			if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
				CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, 0, &zero, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, bias_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			else
				CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, 0, &one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, bias_batch_inc, bias_batch_size, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
			for (i = 0; i < g_batch_size; i++)
			{
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, &zero, bias->data.u8 + CCV_GET_DATA_TYPE_SIZE(bias->info.datatype) * i * bias_batch_inc, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, &one, bias->data.u8 + CCV_GET_DATA_TYPE_SIZE(bias->info.datatype) * i * bias_batch_inc, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
#endif
		} else {
			if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, &zero, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			else
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, &one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			// We cannot use strided batched alternative because on write, the data could race to the same position
			for (i = 1; i < g_batch_size; i++)
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bias_cols, bias_rows, g_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(bias->info.datatype), g_rows, &one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, ccv_nnc_cuda_compute_datatype(bias->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
	}
	if (dw)
	{
		const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
		const int transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
		const int transpose_w = ccv_nnc_is_matrix_transpose(dw->info, cmd.info.blas.transpose_b);
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
		if (g_batch_size > 1 && g_batch_size == dw_batch_size)
		{
			if (transpose_w)
			{
				const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int lda_inc = transpose_a ? a_cols_inc : a_rows_inc;
#if CUDA_VERSION >= 9100
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, a_batch_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, &zero, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, a_batch_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
				for (i = 0; i < g_batch_size; i++)
				{
					if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
						CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &zero, dw->data.u8 + CCV_GET_DATA_TYPE_SIZE(dw->info.datatype) * i * dw_batch_inc, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					else
						CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, dw->data.u8 + CCV_GET_DATA_TYPE_SIZE(dw->info.datatype) * i * dw_batch_inc, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				}
#endif
			} else {
				const cublasOperation_t transb = transpose_a ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
#if CUDA_VERSION >= 9100
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, a_batch_inc, &zero, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, a_batch_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
				for (i = 0; i < g_batch_size; i++)
				{
					if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
						CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &zero, dw->data.u8 + CCV_GET_DATA_TYPE_SIZE(dw->info.datatype) * i * dw_batch_inc, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					else
						CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &one, dw->data.u8 + CCV_GET_DATA_TYPE_SIZE(dw->info.datatype) * i * dw_batch_inc, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				}
#endif
			}
		} else {
			if (transpose_w)
			{
				const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int lda_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &zero, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, &one, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transb = transpose_a ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &zero, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, a->data.u8 + CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, &one, dw->data.u8, ccv_nnc_cuda_datatype(dw->info.datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		}
	}
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h)
	{
		const int transpose_h = ccv_nnc_is_matrix_transpose(h->info, cmd.info.blas.transpose_a);
		const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2];
		const int transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
		int h_batch_size, h_rows, h_cols, h_batch_inc, h_rows_inc, h_cols_inc;
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		ccv_nnc_tensor_get_matrix_params(h->info, CCV_IS_TENSOR_VIEW(h) ? h->inc : h->info.dim, cmd.info.blas.transpose_a, &h_batch_size, &h_rows, &h_cols, &h_batch_inc, &h_rows_inc, &h_cols_inc);
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->inc : w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
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
			if (transpose_h)
			{
				const cublasOperation_t transb = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int ldb_inc = transpose_w ? w_cols_inc : w_rows_inc;
#if CUDA_VERSION >= 9100
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, w_batch_inc, &zero, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, h_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, w_batch_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, h_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
				for (i = 0; i < g_batch_size; i++)
				{
					if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
						CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, &zero, h->data.u8 + CCV_GET_DATA_TYPE_SIZE(h->info.datatype) * i * h_batch_inc, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					else
						CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, &one, h->data.u8 + CCV_GET_DATA_TYPE_SIZE(h->info.datatype) * i * h_batch_inc, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				}
#endif
			} else {
				const cublasOperation_t transa = transpose_w ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
#if CUDA_VERSION >= 9100
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, w_batch_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, &zero, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, h_batch_inc, h_batch_size, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, w_batch_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, g_batch_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, h_batch_inc, h_batch_size, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
#else
				for (i = 0; i < g_batch_size; i++)
				{
					if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
						CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &zero, h->data.u8 + CCV_GET_DATA_TYPE_SIZE(h->info.datatype) * i * h_batch_inc, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
					else
						CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, h->data.u8 + CCV_GET_DATA_TYPE_SIZE(h->info.datatype) * i * h_batch_inc, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				}
#endif
			}
		} else {
			if (transpose_h)
			{
				const cublasOperation_t transb = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int ldb_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, &zero, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, &one, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), ldb_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transa = transpose_w ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &zero, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, g->data.u8, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, &one, w->data.u8 + CCV_GET_DATA_TYPE_SIZE(w->info.datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w->info.datatype), lda_inc, g->data.u8 + CCV_GET_DATA_TYPE_SIZE(g->info.datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g->info.datatype), g_rows_inc, &one, h->data.u8, ccv_nnc_cuda_datatype(h->info.datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h->info.datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
#endif
}
