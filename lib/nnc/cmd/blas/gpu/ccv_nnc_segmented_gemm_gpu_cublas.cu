extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

static inline void _ccv_nnc_segmented_gbmm_and_bias(cublasHandle_t cublas, const void* const ones, const unsigned char* const a, const int a_datatype, const int a_nd, const int* const adim, const int* const astride, const int* const host_indices, const int* const host_counts, const int bincount, const unsigned char* const w, const int w_datatype, const int w_nd, const int* const wdim, const int* const wstride, unsigned char* const bias, const int bias_datatype, const int bias_nd, const int* const biasdim, const int* const biasstride, unsigned char* const b, const int b_datatype, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const cublasOperation_t transa, const cublasOperation_t transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int bias_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int bias_rows_inc, const int b_rows_inc)
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
		case CUBLAS_COMPUTE_32F_FAST_TF32:
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
		assert(b_batch_size == 1);
		int i;
		int off = 0;
		for (i = 0; i < bincount; i++)
		{
			const unsigned char* const ap = a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * off * ldb_inc;
			const unsigned char* const wp = w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * host_indices[i] * w_batch_inc;
			const unsigned char* const biasp = bias + CCV_GET_DATA_TYPE_SIZE(bias_datatype) * host_indices[i] * bias_batch_inc;
			unsigned char* const bp = b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * off * b_rows_inc;
			const int rowcount = host_counts[i];
			off += rowcount;
			CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, rowcount, 1, one, biasp, ccv_nnc_cuda_datatype(bias_datatype), bias_rows_inc, ones, ccv_nnc_cuda_datatype(b_datatype), 1, zero, bp, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, rowcount, a_cols, one, wp, ccv_nnc_cuda_datatype(w_datatype), lda_inc, ap, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, one, bp, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
		_ccv_nnc_segmented_gbmm_and_bias(cublas, ones,
			(a_nd > 3 && adim[0] > 1) ? a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * astride[0] : a, a_datatype, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			host_indices, host_counts, bincount,
			(w_nd > 3 && wdim[0] > 1) ? w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * wstride[0] : w, w_datatype, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			bias_nd > 3 ? bias + CCV_GET_DATA_TYPE_SIZE(bias_datatype) * i * biasstride[0] : bias, bias_datatype, bias_nd > 3 ? bias_nd - 1 : bias_nd, bias_nd > 3 ? biasdim + 1 : biasdim, bias_nd > 3 ? biasstride + 1 : biasstride,
			b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * i * bstride[0], b_datatype, b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
	}
}

static inline void _ccv_nnc_segmented_gbmm(cublasHandle_t cublas, const unsigned char* const a, const int a_datatype, const int a_nd, const int* const adim, const int* const astride, const int* const host_indices, const int* const host_counts, const int bincount, const unsigned char* const w, const int w_datatype, const int w_nd, const int* const wdim, const int* const wstride, unsigned char* const b, const int b_datatype, const int b_nd, const int* const bdim, const int* const bstride, const int b_batch_size, const cublasOperation_t transa, const cublasOperation_t transb, const int lda_inc, const int ldb_inc, const int a_batch_inc, const int w_batch_inc, const int b_batch_inc, const int b_rows, const int b_cols, const int a_cols, const int b_rows_inc)
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
		case CUBLAS_COMPUTE_32F_FAST_TF32:
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
		assert(b_batch_size == 1);
		int i;
		int off = 0;
		for (i = 0; i < bincount; i++)
		{
			const unsigned char* const ap = a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * off * ldb_inc;
			const unsigned char* const wp = w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * host_indices[i] * w_batch_inc;
			unsigned char* const bp = b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * off * b_rows_inc;
			const int rowcount = host_counts[i];
			off += rowcount;
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, rowcount, a_cols, one, wp, ccv_nnc_cuda_datatype(w_datatype), lda_inc, ap, ccv_nnc_cuda_datatype(a_datatype), ldb_inc, zero, bp, ccv_nnc_cuda_datatype(b_datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b_datatype), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
		_ccv_nnc_segmented_gbmm(cublas,
			(a_nd > 3 && adim[0] > 1) ? a + CCV_GET_DATA_TYPE_SIZE(a_datatype) * i * astride[0] : a, a_datatype, a_nd > 3 ? a_nd - 1 : a_nd, a_nd > 3 ? adim + 1 : adim, a_nd > 3 ? astride + 1 : astride,
			host_indices, host_counts, bincount,
			(w_nd > 3 && wdim[0] > 1) ? w + CCV_GET_DATA_TYPE_SIZE(w_datatype) * i * wstride[0] : w, w_datatype, w_nd > 3 ? w_nd - 1 : w_nd, w_nd > 3 ? wdim + 1 : wdim, w_nd > 3 ? wstride + 1 : wstride,
			b + CCV_GET_DATA_TYPE_SIZE(b_datatype) * i * bstride[0], b_datatype, b_nd - 1, bdim + 1, bstride + 1, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
}

static int _ccv_nnc_segmented_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* indices = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* counts = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* bias = input_size > 4 ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0;
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
	int a_datatype = a->info.datatype;
	if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t a_params = a->info;
		a_datatype = (a_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_a_params = a_params;
		depalettize_a_params.datatype = a_datatype;
		depalettize_a_params.reserved = 0;
		a_data_size = ccv_nnc_tensor_data_size(depalettize_a_params);
	}
	size_t w_data_size = 0;
	int w_datatype = w->info.datatype;
	if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t w_params = w->info;
		w_datatype = (w_params.datatype & 0xff) << 12;
		ccv_nnc_tensor_param_t depalettize_w_params = w_params;
		depalettize_w_params.datatype = w_datatype;
		depalettize_w_params.reserved = 0;
		w_data_size = ccv_nnc_tensor_data_size(depalettize_w_params);
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int bincount = ccv_nnc_tensor_count(indices->info);
	assert(ccv_nnc_tensor_count(counts->info) == bincount);
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(CCV_IS_TENSOR_CONTIGUOUS(counts));
	int* const host_indices = (int*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(int) * bincount * 2, CCV_TENSOR_CPU_MEMORY);
	int* const host_counts = host_indices + bincount;
	cudaMemcpyAsync(host_indices, indices->data.i32, sizeof(int) * bincount, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(host_counts, counts->data.i32, sizeof(int) * bincount, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	const size_t cublas_size = ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size);
	void* workspace = 0;
	if (a_data_size + w_data_size > 0)
		workspace = ccv_nnc_stream_context_get_workspace(stream_context, cublas_size + a_data_size + w_data_size, CCV_TENSOR_GPU_MEMORY);
	unsigned char* a_data = a->data.u8;
	if (CCV_GET_DATA_TYPE(a->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t a_params = a->info;
		const size_t count = ccv_nnc_tensor_count(a_params);
		const int qbits = (a_params.datatype & 0xf00) >> 8;
		const int number_in_blocks = a_params.reserved;
		a_data = (unsigned char*)workspace + cublas_size;
		ccv_nnc_compat_depalettize(a->data.u8, a_datatype, ccv_nnc_tensor_data_size_without_padding(a_params), qbits, number_in_blocks, a_data, count, stream_context);
	}
	unsigned char* w_data = w->data.u8;
	if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
	{
		ccv_nnc_tensor_param_t w_params = w->info;
		const size_t count = ccv_nnc_tensor_count(w_params);
		const int qbits = (w_params.datatype & 0xf00) >> 8;
		const int number_in_blocks = w_params.reserved;
		w_data = (unsigned char*)workspace + cublas_size + a_data_size;
		ccv_nnc_compat_depalettize(w->data.u8, w_datatype, ccv_nnc_tensor_data_size_without_padding(w_params), qbits, number_in_blocks, w_data, count, stream_context);
	}
	// Check if we can shortcut this and use dequantize_mul_mat_vec which will be faster for gmmv.
	cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
	ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, cublas_size);
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
		const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, b_rows, b->info.datatype);
		// Explicit sync to make sure the data now is on the host.
		_ccv_nnc_segmented_gbmm_and_bias(cublas, device_ones, a_data, a_datatype, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, host_indices, host_counts, bincount, w_data, w_datatype, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, bias->data.u8, bias->info.datatype, ccv_nnc_tensor_nd(bias->info.dim), bias->info.dim, biasstride, b->data.u8, b->info.datatype, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, bias_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, bias_rows_inc, b_rows_inc);
	} else {
		_ccv_nnc_segmented_gbmm(cublas, a_data, a_datatype, ccv_nnc_tensor_nd(a->info.dim), a->info.dim, astride, host_indices, host_counts, bincount, w_data, w_datatype, ccv_nnc_tensor_nd(w->info.dim), w->info.dim, wstride, b->data.u8, b->info.datatype, ccv_nnc_tensor_nd(b->info.dim), b->info.dim, bstride, b_batch_size, transa, transb, lda_inc, ldb_inc, a_batch_inc, w_batch_inc, b_batch_inc, b_rows, b_cols, a_cols, b_rows_inc);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_segmented_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_gemm_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_GEMM_BACKWARD, CCV_NNC_BACKEND_GPU_CUBLAS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_gemm_back;
#endif
}
