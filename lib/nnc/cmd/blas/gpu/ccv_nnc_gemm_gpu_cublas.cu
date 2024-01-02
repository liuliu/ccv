extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

static inline void _ccv_nnc_gbmm_and_bias(cublasHandle_t cublas, const void* const ones, const unsigned char* const a, const int a_datatype, const int a_nd, const int* const adim, const int* const astride, const unsigned char* const w, const int w_datatype, const int w_nd, const int* const wdim, const int* const wstride, unsigned char* const bias, const int bias_datatype, const int bias_nd, const int* const biasdim, const int* const biasstride, unsigned char* const b, const int b_datatype, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const cublasOperation_t transa, const cublasOperation_t transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int bias_rows_inc, const int b_rows_inc)
{
	static const half one_f16 = 1;
	static const float one_f32 = 1;
	static const double one_f64 = 1;
	static const double zero_f64 = 0;
	const void* zero = &zero_f64;
	const void* one;
	switch (ccv_nnc_cuda_compute_datatype(b_datatype))
	{
		case CUBLAS_COMPUTE_16F:
			one = &one_f16;
			break;
		case CUBLAS_COMPUTE_32F:
			one = &one_f32;
			break;
		case CUBLAS_COMPUTE_64F:
			one = &one_f64;
			break;
		default:
			assert(0);
	}
	if (b_nd <= 3)
	{
		if (b_batch_size == 1)
		{
			CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, one, bias, ccv_nnc_cuda_datatype(bias_datatype), bias_rows_inc, ones, ccv_nnc_cuda_datatype(b_datatype), 1, zero, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, one, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, one, bias, ccv_nnc_cuda_datatype(bias_datatype), bias_rows_inc, bias_batch_inc, ones, ccv_nnc_cuda_datatype(b_datatype), 1, 0, zero, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, w_batch_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, a_batch_inc, one, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
		return;
	}
	int i;
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	if (bias_nd > 3)
		{ assert(biasdim[0] == 1 || dim == biasdim[0]); }
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_gbmm_and_bias(cublas, ones,
			(a_nd > 3 && adim[0] > 1) ? a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * astride[0] : a, a_datatype, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			(w_nd > 3 && wdim[0] > 1) ? w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * wstride[0] : w, w_datatype, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			bias_nd > 3 ? bias + CCV_GET_DATA_TYPE_SIZE(bias_datatype) * i * biasstride[0] : bias, bias_datatype, bias_nd > 3 ? bias_nd - 1 : bias_nd, bias_nd > 3 ? biasdim + 1 : biasdim, bias_nd > 3 ? biasstride + 1 : biasstride,
			b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * i * bstride[0], b_datatype, b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
	}
}

static inline void _ccv_nnc_gbmm(cublasHandle_t cublas, const unsigned char* const a, const int a_datatype, const int a_nd, const int* const adim, const int* const astride, const unsigned char* const w, const int w_datatype, const int w_nd, const int* const wdim, const int* const wstride, unsigned char* const b, const int b_datatype, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const cublasOperation_t transa, const cublasOperation_t transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int b_rows_inc)
{
	static const half one_f16 = 1;
	static const float one_f32 = 1;
	static const double one_f64 = 1;
	static const double zero_f64 = 0;
	const void* zero = &zero_f64;
	const void* one;
	switch (ccv_nnc_cuda_compute_datatype(b_datatype))
	{
		case CUBLAS_COMPUTE_16F:
			one = &one_f16;
			break;
		case CUBLAS_COMPUTE_32F:
			one = &one_f32;
			break;
		case CUBLAS_COMPUTE_64F:
			one = &one_f64;
			break;
		default:
			assert(0);
	}
	if (b_nd <= 3)
	{
		if (b_batch_size == 1)
		{
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, zero, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, w_batch_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, a_batch_inc, zero, b, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, b_batch_inc, b_batch_size, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
		return;
	}
	int i;
	const int dim = bdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	for (i = 0; i < dim; i++)
	{
		_ccv_nnc_gbmm(cublas,
			(a_nd > 3 && adim[0] > 1) ? a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * astride[0] : a, a_datatype, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			(w_nd > 3 && wdim[0] > 1) ? w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * wstride[0] : w, w_datatype, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * i * bstride[0], b_datatype, b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
}

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
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
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
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size));
	const int transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
	const int transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
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
	const cublasOperation_t transa = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
	const cublasOperation_t transb = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
	const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
	const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
	size_t a_data_size = 0;
	if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t a_params = a->info;
		const int palette_datatype = (a_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_a_params = a_params;
		depalettize_a_params.datatype = palette_datatype;
		depalettize_a_params.reserved = 0;
		a_data_size = ccv_nnc_tensor_data_size(depalettize_a_params);
	}
	size_t w_data_size = 0;
	if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t w_params = w->info;
		const int palette_datatype = (w_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_w_params = w_params;
		depalettize_w_params.datatype = palette_datatype;
		depalettize_w_params.reserved = 0;
		w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
	}
	void* workspace = 0;
	if (a_data_size + w_data_size > 0)
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, a_data_size + w_data_size, CCV_TENSOR_GPU_MEMORY);
	unsigned char* a_data = a->data.u8;
	if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t a_params = a->info;
		const size_t count = ccv_nnc_tensor_count(a_params);
		const int palette_datatype = (a_params.datatype & 0xff) << 12;
		const int qbits = (a_params.datatype & 0xf00) >> 8;
		const int number_in_blocks = a_params.reserved;
		a_data = (unsigned char*)workspace;
		ccv_nnc_compat_depalettize(a->data.u8, palette_datatype, ccv_nnc_tensor_data_size_without_padding(a_params), qbits, number_in_blocks, a_data, count, stream_context);
	}
	unsigned char* w_data = w->data.u8;
	if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t w_params = w->info;
		const size_t count = ccv_nnc_tensor_count(w_params);
		const int palette_datatype = (w_params.datatype & 0xff) << 12;
		const int qbits = (w_params.datatype & 0xf00) >> 8;
		const int number_in_blocks = w_params.reserved;
		w_data = (unsigned char*)workspace + a_data_size;
		ccv_nnc_compat_depalettize(w->data.u8, palette_datatype, ccv_nnc_tensor_data_size_without_padding(w_params), qbits, number_in_blocks, w_data, count, stream_context);
	}
	if (bias)
	{
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
		const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, b_rows, b->info.datatype);
		_ccv_nnc_gbmm_and_bias(cublas, device_ones, a_data, a->info.datatype, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w_data, w->info.datatype, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, bias->data.u8, bias->info.datatype, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, b->data.u8, b->info.datatype, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
	} else {
		_ccv_nnc_gbmm(cublas, a_data, a->info.datatype, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, w_data, w->info.datatype, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, b->data.u8, b->info.datatype, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static inline void _ccv_nnc_gbmm_dbias(cublasHandle_t cublas, const int flags, const void* const device_ones, const unsigned char* const g, const int g_datatype, const int g_nd, const int* const gdim, const int* const gstride, unsigned char* const dbias, const int dbias_datatype, const int dbias_nd, const int* const dbiasdim, const int* const dbiasstride, const int g_batch_size, const int dbias_batch_size, const int g_batch_inc, const int dbias_batch_inc, const int dbias_rows, const int dbias_cols, const int g_rows, const int g_rows_inc, const int dbias_rows_inc)
{
	static const half one_f16 = 1;
	static const float one_f32 = 1;
	static const double one_f64 = 1;
	static const double zero_f64 = 0;
	const void* zero = &zero_f64;
	const void* one;
	switch (ccv_nnc_cuda_compute_datatype(dbias_datatype))
	{
		case CUBLAS_COMPUTE_16F:
			one = &one_f16;
			break;
		case CUBLAS_COMPUTE_32F:
			one = &one_f32;
			break;
		case CUBLAS_COMPUTE_64F:
			one = &one_f64;
			break;
		default:
			assert(0);
	}
	int i;
	if (g_nd <= 3)
	{
		if (g_batch_size > 1 && dbias_batch_size == g_batch_size)
		{
			if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
				CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dbias_cols, dbias_rows, g_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, device_ones, ccv_nnc_cuda_datatype(dbias_datatype), g_rows, 0, zero, dbias, ccv_nnc_cuda_datatype(dbias_datatype), dbias_rows_inc, dbias_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dbias_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			else
				CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dbias_cols, dbias_rows, g_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, device_ones, ccv_nnc_cuda_datatype(dbias_datatype), g_rows, 0, one, dbias, ccv_nnc_cuda_datatype(dbias_datatype), dbias_rows_inc, dbias_batch_inc, dbias_batch_size, ccv_nnc_cuda_compute_datatype(dbias_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dbias_cols, dbias_rows, g_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(dbias_datatype), g_rows, zero, dbias, ccv_nnc_cuda_datatype(dbias_datatype), dbias_rows_inc, ccv_nnc_cuda_compute_datatype(dbias_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			else
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dbias_cols, dbias_rows, g_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(dbias_datatype), g_rows, one, dbias, ccv_nnc_cuda_datatype(dbias_datatype), dbias_rows_inc, ccv_nnc_cuda_compute_datatype(dbias_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			// We cannot use strided batched alternative because on write, the data could race to the same position
			for (i = 1; i < g_batch_size; i++)
				CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, dbias_cols, dbias_rows, g_rows, one, g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, device_ones, ccv_nnc_cuda_datatype(dbias_datatype), g_rows, one, dbias, ccv_nnc_cuda_datatype(dbias_datatype), dbias_rows_inc, ccv_nnc_cuda_compute_datatype(dbias_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
		return;
	}
	const int dim = gdim[0];
	if (dbias_nd > 3)
		{ assert(dbiasdim[0] == 1 || dim == dbiasdim[0]); }
	for (i = 0; i < dim; i++)
	{
		const int flags_override = (i == 0 || (i * dbiasstride[0] > 0 && dbias_nd > 3)) ? flags : (flags | CCV_NNC_ACCUMULATE_OUTPUT);
		_ccv_nnc_gbmm_dbias(cublas, flags_override, device_ones,
			g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * gstride[0], g_datatype, g_nd - 1, gdim + 1, gstride + 1,
			dbias_nd > 3 ? dbias + CCV_GET_DATA_TYPE_SIZE(dbias_datatype) * i * dbiasstride[0] : dbias, dbias_datatype, dbias_nd > 3 ? dbias_nd - 1 : dbias_nd, dbias_nd > 3 ? dbiasdim + 1 : dbiasdim, dbias_nd > 3 ? dbiasstride + 1 : dbiasstride,
			g_batch_size, dbias_batch_size, g_batch_inc, dbias_batch_inc, dbias_rows, dbias_cols, g_rows, g_rows_inc, dbias_rows_inc);
	}
}

static inline void _ccv_nnc_gbmm_dw(cublasHandle_t cublas, const int flags, const unsigned char* const g, const int g_datatype, const int g_nd, const int* const gdim, const int* const gstride, const unsigned char* const a, const int a_datatype, const int a_nd, const int* const adim, const int* const astride, unsigned char* const dw, const int dw_datatype, const int dw_nd, const int* const dwdim, const int* const dwstride, const int g_batch_size, const int dw_batch_size, const int transpose_a, const int transpose_w, const int g_batch_inc, const int a_batch_inc, const int dw_batch_inc, const int dw_rows, const int dw_cols, const int a_rows, const int g_rows_inc, const int a_cols_inc, const int a_rows_inc, const int dw_cols_inc, const int dw_rows_inc)
{
	static const half one_f16 = 1;
	static const float one_f32 = 1;
	static const double one_f64 = 1;
	static const double zero_f64 = 0;
	const void* zero = &zero_f64;
	const void* one;
	switch (ccv_nnc_cuda_compute_datatype(dw_datatype))
	{
		case CUBLAS_COMPUTE_16F:
			one = &one_f16;
			break;
		case CUBLAS_COMPUTE_32F:
			one = &one_f32;
			break;
		case CUBLAS_COMPUTE_64F:
			one = &one_f64;
			break;
		default:
			assert(0);
	}
	int i;
	if (g_nd <= 3)
	{
		if (g_batch_size > 1 && g_batch_size == dw_batch_size)
		{
			if (transpose_w)
			{
				const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int lda_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, one, a, ccv_nnc_cuda_datatype(a_datatype), lda_inc, a_batch_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, zero, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_cols_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, one, a, ccv_nnc_cuda_datatype(a_datatype), lda_inc, a_batch_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_cols_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transb = transpose_a ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, a_batch_inc, zero, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_rows_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, a_batch_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_rows_inc, dw_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		} else {
			if (transpose_w)
			{
				const cublasOperation_t transa = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int lda_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, one, a, ccv_nnc_cuda_datatype(a_datatype), lda_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, zero, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, one, a, ccv_nnc_cuda_datatype(a_datatype), lda_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_T, dw_rows, dw_cols, a_rows, one, a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a_datatype), lda_inc, g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_cols_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transb = transpose_a ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int ldb_inc = transpose_a ? a_cols_inc : a_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, zero, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, a, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, transb, dw_cols, dw_rows, a_rows, one, g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * a_batch_inc, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, one, dw, ccv_nnc_cuda_datatype(dw_datatype), dw_rows_inc, ccv_nnc_cuda_compute_datatype(dw_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		}
		return;
	}
	const int dim = gdim[0];
	if (a_nd > 3)
		{ assert(adim[0] == 1 || dim == adim[0]); }
	if (dw_nd > 3)
		{ assert(dwdim[0] == 1 || dim == dwdim[0]); }
	for (i = 0; i < dim; i++)
	{
		const int flags_override = (i == 0 || (i * dwstride[0] > 0 && dw_nd > 3)) ? flags : (flags | CCV_NNC_ACCUMULATE_OUTPUT);
		_ccv_nnc_gbmm_dw(cublas, flags_override,
			g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * gstride[0], g_datatype, g_nd - 1, gdim + 1, gstride + 1,
			a_nd > 3 ? a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * astride[0] : a, a_datatype, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			dw_nd > 3 ? dw + CCV_GET_DATA_TYPE_SIZE(dw_datatype) * i * dwstride[0] : dw, dw_datatype, dw_nd > 3 ? dw_nd - 1 : dw_nd, dw_nd > 3 ? dwdim + 1 : dwdim, dw_nd > 3 ? dwstride + 1 : dwstride,
			g_batch_size, dw_batch_size, transpose_a, transpose_w, g_batch_inc, a_batch_inc, dw_batch_inc, dw_rows, dw_cols, a_rows, g_rows_inc, a_cols_inc, a_rows_inc, dw_cols_inc, dw_rows_inc);
	}
}

static inline void _ccv_nnc_gbmm_h(cublasHandle_t cublas, const int flags, const unsigned char* const g, const int g_datatype, const int g_nd, const int* const gdim, const int* const gstride, const unsigned char* const w, const int w_datatype, const int w_nd, const int* const wdim, const int* const wstride, unsigned char* const h, const int h_datatype, const int h_nd, const int* const hdim, const int* const hstride, const int g_batch_size, const int h_batch_size, const int transpose_h, const int transpose_w, const int g_batch_inc, const int w_batch_inc, const int h_batch_inc, const int h_rows, const int h_cols, const int g_cols, const int g_rows_inc, const int w_cols_inc, const int w_rows_inc, const int h_cols_inc, const int h_rows_inc)
{
	static const half one_f16 = 1;
	static const float one_f32 = 1;
	static const double one_f64 = 1;
	static const double zero_f64 = 0;
	const void* zero = &zero_f64;
	const void* one;
	switch (ccv_nnc_cuda_compute_datatype(h_datatype))
	{
		case CUBLAS_COMPUTE_16F:
			one = &one_f16;
			break;
		case CUBLAS_COMPUTE_32F:
			one = &one_f32;
			break;
		case CUBLAS_COMPUTE_64F:
			one = &one_f64;
			break;
		default:
			assert(0);
	}
	int i;
	if (g_nd <= 3)
	{
		if (g_batch_size > 1 && g_batch_size == h_batch_size)
		{
			if (transpose_h)
			{
				const cublasOperation_t transb = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int ldb_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, w, ccv_nnc_cuda_datatype(w_datatype), ldb_inc, w_batch_inc, zero, h, ccv_nnc_cuda_datatype(h_datatype), h_cols_inc, h_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, w, ccv_nnc_cuda_datatype(w_datatype), ldb_inc, w_batch_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_cols_inc, h_batch_inc, g_batch_size, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transa = transpose_w ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, w_batch_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, zero, h, ccv_nnc_cuda_datatype(h_datatype), h_rows_inc, h_batch_inc, h_batch_size, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmStridedBatchedEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, w_batch_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, g_batch_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_rows_inc, h_batch_inc, h_batch_size, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		} else {
			if (transpose_h)
			{
				const cublasOperation_t transb = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
				const int ldb_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, w, ccv_nnc_cuda_datatype(w_datatype), ldb_inc, zero, h, ccv_nnc_cuda_datatype(h_datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, one, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, w, ccv_nnc_cuda_datatype(w_datatype), ldb_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_T, transb, h_rows, h_cols, g_cols, one, g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w_datatype), ldb_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_cols_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			} else {
				const cublasOperation_t transa = transpose_w ? CUBLAS_OP_N : CUBLAS_OP_T;
				const int lda_inc = transpose_w ? w_cols_inc : w_rows_inc;
				if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, zero, h, ccv_nnc_cuda_datatype(h_datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				else
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, one, w, ccv_nnc_cuda_datatype(w_datatype), lda_inc, g, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
				for (i = 1; i < g_batch_size; i++)
					CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, CUBLAS_OP_N, h_cols, h_rows, g_cols, one, w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * w_batch_inc, ccv_nnc_cuda_datatype(w_datatype), lda_inc, g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * g_batch_inc, ccv_nnc_cuda_datatype(g_datatype), g_rows_inc, one, h, ccv_nnc_cuda_datatype(h_datatype), h_rows_inc, ccv_nnc_cuda_compute_datatype(h_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			}
		}
		return;
	}
	const int dim = gdim[0];
	if (w_nd > 3)
		{ assert(wdim[0] == 1 || dim == wdim[0]); }
	if (h_nd > 3)
		{ assert(hdim[0] == 1 || dim == hdim[0]); }
	for (i = 0; i < dim; i++)
	{
		const int flags_override = (i == 0 || (i * hstride[0] > 0 && h_nd > 3)) ? flags : (flags | CCV_NNC_ACCUMULATE_OUTPUT);
		_ccv_nnc_gbmm_h(cublas, flags_override,
			g + CCV_GET_DATA_TYPE_SIZE(g_datatype) * i * gstride[0], g_datatype, g_nd - 1, gdim + 1, gstride + 1,
			w_nd > 3 ? w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * wstride[0] : w, w_datatype, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			h_nd > 3 ? h + CCV_GET_DATA_TYPE_SIZE(h_datatype) * i * hstride[0] : h, h_datatype, h_nd > 3 ? h_nd - 1 : h_nd, h_nd > 3 ? hdim + 1 : hdim, h_nd > 3 ? hstride + 1 : hstride,
			g_batch_size, h_batch_size, transpose_h, transpose_w, g_batch_inc, w_batch_inc, h_batch_inc, h_rows, h_cols, g_cols, g_rows_inc, w_cols_inc, w_rows_inc, h_cols_inc, h_rows_inc);
	}
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, bias updates
	assert(input_size >= 2 && output_size >= 1);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* dw = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* bias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 2-d or 3-d array.
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size));
	int g_batch_size, g_rows, g_cols, g_batch_inc, g_rows_inc, g_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(g->info, CCV_IS_TENSOR_VIEW(g) ? g->stride : 0, g->info.dim, no_transpose, &g_batch_size, &g_rows, &g_cols, &g_batch_inc, &g_rows_inc, &g_cols_inc);
	if (bias)
	{
		int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
		ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
		assert(bias_cols == g_cols);
		assert(bias_batch_size == 1 || bias_batch_size == g_batch_size);
		if (bias_batch_size == 1 && g_batch_size > 1)
			bias_batch_inc = 0;
		int gstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int biasstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		const int* gstride;
		if (CCV_IS_TENSOR_VIEW(g))
			gstride = g->stride;
		else {
			ccv_nnc_tensor_get_stride(g->info.dim, gstride_from_dim);
			gstride = gstride_from_dim;
		}
		const int* biasstride;
		if (CCV_IS_TENSOR_VIEW(bias))
			biasstride = bias->stride;
		else {
			ccv_nnc_tensor_get_stride(bias->info.dim, biasstride_from_dim);
			biasstride = biasstride_from_dim;
		}
		const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, g_rows, bias->info.datatype);
		_ccv_nnc_gbmm_dbias(cublas, flags, device_ones, g->data.u8, g->info.datatype, ccv_nnc_tensor_nd(g->info.dim), g->info.dim, gstride, bias->data.u8, bias->info.datatype, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, g_batch_size, bias_batch_size, g_batch_inc, bias_batch_inc, bias_rows, bias_cols, g_rows, g_rows_inc, bias_rows_inc);
	}
	size_t a_data_size = 0;
	if (dw && CCV_GET_DATA_TYPE(inputs[1]->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t a_params = inputs[1]->info;
		const int palette_datatype = (a_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_a_params = a_params;
		depalettize_a_params.datatype = palette_datatype;
		depalettize_a_params.reserved = 0;
		a_data_size = ccv_nnc_tensor_data_size(depalettize_a_params);
	}
	size_t w_data_size = 0;
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (h && CCV_GET_DATA_TYPE(inputs[2]->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t w_params = inputs[2]->info;
		const int palette_datatype = (w_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_w_params = w_params;
		depalettize_w_params.datatype = palette_datatype;
		depalettize_w_params.reserved = 0;
		w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
	}
	void* workspace = 0;
	if (a_data_size + w_data_size > 0)
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, a_data_size + w_data_size, CCV_TENSOR_GPU_MEMORY);
	if (dw)
	{
		const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1];
		unsigned char* a_data = a->data.u8;
		if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t a_params = a->info;
			const size_t count = ccv_nnc_tensor_count(a_params);
			const int palette_datatype = (a_params.datatype & 0xff) << 12;
			const int qbits = (a_params.datatype & 0xf00) >> 8;
			const int number_in_blocks = a_params.reserved;
			a_data = (unsigned char*)workspace;
			ccv_nnc_compat_depalettize(a->data.u8, palette_datatype, ccv_nnc_tensor_data_size_without_padding(a_params), qbits, number_in_blocks, a_data, count, stream_context);
		}
		const int transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
		const int transpose_w = ccv_nnc_is_matrix_transpose(dw->info, cmd.info.blas.transpose_b);
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
		int gstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int astride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int dwstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		const int* gstride;
		if (CCV_IS_TENSOR_VIEW(g))
			gstride = g->stride;
		else {
			ccv_nnc_tensor_get_stride(g->info.dim, gstride_from_dim);
			gstride = gstride_from_dim;
		}
		const int* astride;
		if (CCV_IS_TENSOR_VIEW(a))
			astride = a->stride;
		else {
			ccv_nnc_tensor_get_stride(a->info.dim, astride_from_dim);
			astride = astride_from_dim;
		}
		const int* dwstride;
		if (CCV_IS_TENSOR_VIEW(dw))
			dwstride = dw->stride;
		else {
			ccv_nnc_tensor_get_stride(dw->info.dim, dwstride_from_dim);
			dwstride = dwstride_from_dim;
		}
		_ccv_nnc_gbmm_dw(cublas, flags, g->data.u8, g->info.datatype, ccv_nnc_tensor_nd(g->info.dim), g->info.dim, gstride, a_data, a->info.datatype, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, dw->data.u8, dw->info.datatype, ccv_nnc_tensor_nd(dw->info.dim), dw->info.dim, dwstride, g_batch_size, dw_batch_size, transpose_a, transpose_w, g_batch_inc, a_batch_inc, dw_batch_inc, dw_rows, dw_cols, a_rows, g_rows_inc, a_cols_inc, a_rows_inc, dw_cols_inc, dw_rows_inc);
	}
	if (h)
	{
		const int transpose_h = ccv_nnc_is_matrix_transpose(h->info, cmd.info.blas.transpose_a);
		const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2];
		unsigned char* w_data = w->data.u8;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			const size_t count = ccv_nnc_tensor_count(w_params);
			const int palette_datatype = (w_params.datatype & 0xff) << 12;
			const int qbits = (w_params.datatype & 0xf00) >> 8;
			const int number_in_blocks = w_params.reserved;
			w_data = (unsigned char*)workspace + a_data_size;
			ccv_nnc_compat_depalettize(w->data.u8, palette_datatype, ccv_nnc_tensor_data_size_without_padding(w_params), qbits, number_in_blocks, w_data, count, stream_context);
		}
		const int transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
		int h_batch_size, h_rows, h_cols, h_batch_inc, h_rows_inc, h_cols_inc;
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		ccv_nnc_tensor_get_matrix_params(h->info, CCV_IS_TENSOR_VIEW(h) ? h->stride : 0, h->info.dim, cmd.info.blas.transpose_a, &h_batch_size, &h_rows, &h_cols, &h_batch_inc, &h_rows_inc, &h_cols_inc);
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		assert(h_rows == g_rows);
		assert(h_cols == w_rows);
		assert(w_cols == g_cols);
		assert(h_batch_size == g_batch_size || h_batch_size == 1);
		if (h_batch_size == 1 && g_batch_size > 1)
			h_batch_inc = 0;
		assert(w_batch_size == g_batch_size || w_batch_size == 1);
		if (w_batch_size == 1 && g_batch_size > 1)
			w_batch_inc = 0;
		int gstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int wstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		int hstride_from_dim[CCV_NNC_MAX_DIM_ALLOC];
		const int* gstride;
		if (CCV_IS_TENSOR_VIEW(g))
			gstride = g->stride;
		else {
			ccv_nnc_tensor_get_stride(g->info.dim, gstride_from_dim);
			gstride = gstride_from_dim;
		}
		const int* wstride;
		if (CCV_IS_TENSOR_VIEW(w))
			wstride = w->stride;
		else {
			ccv_nnc_tensor_get_stride(w->info.dim, wstride_from_dim);
			wstride = wstride_from_dim;
		}
		const int* hstride;
		if (CCV_IS_TENSOR_VIEW(h))
			hstride = h->stride;
		else {
			ccv_nnc_tensor_get_stride(h->info.dim, hstride_from_dim);
			hstride = hstride_from_dim;
		}
		_ccv_nnc_gbmm_h(cublas, flags, g->data.u8, g->info.datatype, ccv_nnc_tensor_nd(g->info.dim), g->info.dim, gstride, w_data, w->info.datatype, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, h->data.u8, h->info.datatype, ccv_nnc_tensor_nd(h->info.dim), h->info.dim, hstride, g_batch_size, h_batch_size, transpose_h, transpose_w, g_batch_inc, w_batch_inc, h_batch_inc, h_rows, h_cols, g_cols, g_rows_inc, w_cols_inc, w_rows_inc, h_cols_inc, h_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
#endif
}
