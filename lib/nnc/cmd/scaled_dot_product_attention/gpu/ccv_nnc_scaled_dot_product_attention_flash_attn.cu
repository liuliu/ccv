extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA_SM80
#include <nnc/gpu/3rdparty/flash_attn/flash_api.h>

static int _ccv_nnc_scaled_dot_product_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// NNC notation:
	// C = sm(Q * K^T) * V
	//
	// MFA notation:
	// O = sm(Q * K^T) * V
	assert(input_size >= 3);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 3 ? (ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const weights = input_size > 4 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 5 ? (ccv_nnc_tensor_view_t*)inputs[5] : 0;
	const int is_varlen = cmd.info.scaled_dot_product_attention.is_varlen;
	ccv_nnc_tensor_view_t* const q_seq_offsets = is_varlen && input_size > 6 ? (ccv_nnc_tensor_view_t*)inputs[6] : 0;
	ccv_nnc_tensor_view_t* const k_seq_offsets = is_varlen && input_size > 7 ? (ccv_nnc_tensor_view_t*)inputs[7] : 0;
	if (bias) // bias always requires a weight matrix.
		{ assert(weights); }
	if (is_varlen && (attn_mask || weights || bias || !q_seq_offsets || !k_seq_offsets))
		return CCV_NNC_EXEC_INVALID;

	ccv_nnc_tensor_view_t* const saved_softmax_lse = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_tensor_view_t* const o = (weights) ? (ccv_nnc_tensor_view_t*)outputs[2] : (ccv_nnc_tensor_view_t*)outputs[0];
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int o_nd = ccv_nnc_tensor_nd(o->info.dim);
	assert(o_nd == 3 || o_nd == 4);
	assert(q_nd == k_nd && k_nd == v_nd && v_nd == o_nd);
	if (is_varlen && q_nd != 4)
		return CCV_NNC_EXEC_INVALID;

	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int odim[CCV_NNC_MAX_DIM_ALLOC];
	int amdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(o, odim);

	assert(q->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(k->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(v->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);
	if (attn_mask) {
		// MFA does not support fused transposes on the mask.
		assert(attn_mask->info.format == CCV_TENSOR_FORMAT_NHWC);
	}

	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(o));
	if (is_varlen) {
		assert(q_seq_offsets->info.datatype == CCV_32S);
		assert(k_seq_offsets->info.datatype == CCV_32S);
		assert(CCV_IS_TENSOR_CONTIGUOUS(q_seq_offsets));
		assert(CCV_IS_TENSOR_CONTIGUOUS(k_seq_offsets));
	}

	if (attn_mask) {
		assert(CCV_IS_TENSOR_CONTIGUOUS(attn_mask));
	}

	int batch_size;
	int R;
	int C;
	int Hq;
	int Hk;
	int D;
	if (is_varlen) {
		batch_size = ccv_nnc_tensor_count(q_seq_offsets->info) - 1;
		assert(batch_size > 0);
		assert(ccv_nnc_tensor_count(k_seq_offsets->info) == batch_size + 1);
		assert(cmd.info.scaled_dot_product_attention.max_seqlen_q > 0);
		assert(cmd.info.scaled_dot_product_attention.max_seqlen_k > 0);
		assert(qdim[0] == 1);
		assert(kdim[0] == 1);
		assert(vdim[0] == 1);
		assert(odim[0] == 1);
		assert(odim[1] == qdim[1]);
		R = cmd.info.scaled_dot_product_attention.max_seqlen_q;
		C = cmd.info.scaled_dot_product_attention.max_seqlen_k;
		Hq = qdim[2];
		Hk = kdim[2];
		assert(Hq >= Hk);
		assert(Hq % Hk == 0);
		D = qdim[3];
		assert(D == kdim[3]);
		assert(D == vdim[3]);
		assert(D == odim[3]);
		assert(kdim[1] == vdim[1]);
		assert(Hk == vdim[2]);
		assert(Hq == odim[2]);
	} else if (q_nd == 3) {
		batch_size = qdim[1];
		assert(batch_size == kdim[1]);
		R = qdim[2];
		C = kdim[2];
		Hq = Hk = 1;
		D = qdim[3];
		assert(D == kdim[3]);
	} else if (q_nd == 4) {
		batch_size = qdim[0];
		assert(batch_size == kdim[0]);
		R = qdim[1];
		C = kdim[1];
		Hq = qdim[2];
		Hk = kdim[2];
		assert(Hq >= Hk);
		assert(Hq % Hk == 0);
		D = qdim[3];
		assert(D == kdim[3]);
	}

	if (attn_mask) {
		// MFA can support am_nd == 2 and broadcast batch=1 -> batch=batch_size, but
		// wait until that occurs in practice before doing so.
		const int am_nd = ccv_nnc_tensor_nd(attn_mask->info.dim);
		assert(am_nd == 3 || am_nd == 4); // [batch_size, R, C]

		// MFA does not support attention mask broadcasting (where the R dimension
		// of Q > 1, but the R dimension of the mask == 1).
		ccv_nnc_tensor_view_get_dim(attn_mask, amdim);
		if (am_nd == 3)
		{
			assert(amdim[1] == batch_size || amdim[1] == 1);
			amdim[0] = amdim[1];
			amdim[1] = 1;
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		} else {
			assert(amdim[0] == batch_size || amdim[0] == 1);
			assert(amdim[1] == 1);
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		}
	}
	int weights_datatype = 0;
	if (weights)
		weights_datatype = CCV_GET_DATA_TYPE(weights->info.datatype) == CCV_QX ? ((weights->info.datatype & 0xff) << 12) : weights->info.datatype;

	const int is_same_dtype =
		(q->info.datatype == k->info.datatype) &&
		(q->info.datatype == v->info.datatype) &&
		(q->info.datatype == o->info.datatype) &&
		(weights ? (q->info.datatype == weights_datatype) : 1) &&
		(bias ? (q->info.datatype == bias->info.datatype) : 1);

	assert(is_same_dtype);

	Flash_fwd_params params;
	memset(&params, 0, sizeof(params));
	params.is_bf16 = q->info.datatype == CCV_16BF;
	params.q_ptr = q->data.u8;
	params.k_ptr = k->data.u8;
	params.v_ptr = v->data.u8;
	params.q_row_stride = D * Hq;
	params.k_row_stride = D * Hk;
	params.v_row_stride = D * Hk;
	params.q_head_stride = D;
	params.k_head_stride = D;
	params.v_head_stride = D;
	params.q_batch_stride = R * Hq * D;
	params.k_batch_stride = C * Hk * D;
	params.v_batch_stride = C * Hk * D;
	auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
	params.seqlen_q = R;
	params.seqlen_q_rounded = round_multiple(R, 128);
	params.seqlen_k = C;
	params.seqlen_k_rounded = round_multiple(C, 128);
	params.cu_seqlens_q = is_varlen ? q_seq_offsets->data.i32 : 0;
	params.cu_seqlens_k = is_varlen ? k_seq_offsets->data.i32 : 0;
	params.d = D;
	assert(D % 8 == 0);
	params.d_rounded = round_multiple(D, 32);
	params.o_ptr = o->data.u8;
	params.o_row_stride = D * Hq;
	params.o_head_stride = D;
	params.o_batch_stride = R * Hq * D;
	params.b = batch_size;
	params.h = Hq;
	params.h_k = Hk;
	params.h_h_k_ratio = Hq / Hk;
	params.scale_softmax = cmd.info.scaled_dot_product_attention.scale;
	params.scale_softmax_log2 = cmd.info.scaled_dot_product_attention.scale * M_LOG2E;
	params.is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	params.p_dropout = 1;
	params.p_dropout_in_uint8_t = 255;
	params.rp_dropout = 1;
	params.scale_softmax_rp_dropout = params.scale_softmax;
	params.window_size_left = ccv_max(R, C);
	params.window_size_right = params.is_causal ? 0 : ccv_max(R, C);
	params.is_seqlens_k_cumulative = true;
	const int block_n = D <= 64 ? 256 : (D <= 128 ? 128 : 64);
	const int num_n_blocks = (C + block_n - 1) / block_n;
	// Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
	// In any case we don't expect seqlen_q to be larger than 64 for inference.
	const int num_m_blocks = (R + 64 - 1) / 64;
	const ccv_nnc_cuda_device_prop_t props = ccv_nnc_gpu_device_props();
	// Only enable splitkv if R is 1.
	params.num_splits = (!is_varlen && R == 1) ? num_splits_heuristic(batch_size * Hq * num_m_blocks, props.multi_processor_count * 2, num_n_blocks, 128) : 1;
	if (saved_softmax_lse)
		params.softmax_lse_ptr = saved_softmax_lse->data.u8;
	if (params.num_splits > 1)
	{
		if (saved_softmax_lse)
		{
			float* const workspace = (float*)ccv_nnc_stream_context_get_workspace(stream_context, (params.num_splits * batch_size * Hq * R + params.num_splits * batch_size * Hq * R * params.d_rounded) * sizeof(float), CCV_TENSOR_GPU_MEMORY);
			params.softmax_lseaccum_ptr = workspace;
			params.oaccum_ptr = workspace + params.num_splits * batch_size * Hq * R;
		} else {
			float* const workspace = (float*)ccv_nnc_stream_context_get_workspace(stream_context, (batch_size * Hq * R + params.num_splits * batch_size * Hq * R + params.num_splits * batch_size * Hq * R * params.d_rounded) * sizeof(float), CCV_TENSOR_GPU_MEMORY);
			params.softmax_lse_ptr = workspace;
			params.softmax_lseaccum_ptr = workspace + batch_size * Hq * R;
			params.oaccum_ptr = workspace + batch_size * Hq * R + params.num_splits * batch_size * Hq * R;
		}
	} else if (!saved_softmax_lse) {
		void* const workspace = ccv_nnc_stream_context_get_workspace(stream_context, batch_size * Hq * R * sizeof(float), CCV_TENSOR_GPU_MEMORY);
		params.softmax_lse_ptr = workspace;
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	run_mha_fwd(params, stream, false);
	CUDA_ENFORCE(cudaGetLastError());
	if (weights)
	{
		const ccv_nnc_tensor_view_t* a = o;
		const ccv_nnc_tensor_view_t* w = weights;
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
		assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
		assert(CCV_IS_TENSOR_CONTIGUOUS(b));
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == 3);
		int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
		const int w_nd = ccv_nnc_tensor_nd(w->info.dim);
		const int transpose_w[2] = {
			w_nd - 2, w_nd - 1
		};
		ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, transpose_w, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
		int a_rows, a_cols;
		if (o_nd == 3) {
			a_rows = odim[1] * odim[2];
			a_cols = odim[3];
		} else if (q_nd == 4) {
			a_rows = odim[0] * odim[1];
			a_cols = odim[2] * odim[3];
		}
		int b_rows, b_cols, b_rows_inc;
		b_rows = b->info.dim[0] * b->info.dim[1];
		b_cols = b->info.dim[2];
		b_rows_inc = b_cols;
		assert(a_rows == b_rows);
		assert(a_cols == w_rows);
		assert(w_cols == b_cols);

		const cublasOperation_t transa = CUBLAS_OP_T;
		const cublasOperation_t transb = CUBLAS_OP_N;
		const int lda_inc = w_cols_inc;
		const int ldb_inc = a_cols;
		size_t w_data_size = 0;
		int w_datatype = w->info.datatype;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			w_datatype = (w_params.datatype & 0xff) << 12;
			w_data_size = ccv_nnc_compat_qx_dense_data_size(w_params);
		}
		const size_t cublas_size = ccv_nnc_cublas_workspace_size_in_bytes(inputs, input_size, outputs, output_size);
		void* workspace = 0;
		if (w_data_size > 0)
			workspace = ccv_nnc_stream_context_get_workspace(stream_context, cublas_size + w_data_size, CCV_TENSOR_GPU_MEMORY);
		unsigned char* w_data = w->data.u8;
		if (CCV_GET_DATA_TYPE(w->info.datatype) == CCV_QX)
		{
			ccv_nnc_tensor_param_t w_params = w->info;
			w_data = (unsigned char*)workspace + cublas_size;
			ccv_nnc_compat_decode_qx(w->data.u8, w_params, w_data, stream_context);
		}
		cublasHandle_t cublas = ccv_nnc_stream_context_get_cublas(stream_context);
		static const half one_f16 = 1;
		static const float one_f32 = 1;
		static const double one_f64 = 1;
		static const double zero_f64 = 0;
		const void* zero = &zero_f64;
		const void* one;
		const int is_downcast = ((cmd.info.scaled_dot_product_attention.flags & CCV_NNC_GEMM_16F) && b->info.datatype == CCV_16F);
		switch (ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast))
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
		ccv_nnc_stream_context_set_cublas_workspace(cublas, stream_context, cublas_size);
		if (bias)
		{
			int bias_batch_size, bias_rows, bias_cols, bias_batch_inc, bias_rows_inc, bias_cols_inc;
			const static int no_transpose[2] = {};
			ccv_nnc_tensor_get_matrix_params(bias->info, CCV_IS_TENSOR_VIEW(bias) ? bias->stride : 0, bias->info.dim, no_transpose, &bias_batch_size, &bias_rows, &bias_cols, &bias_batch_inc, &bias_rows_inc, &bias_cols_inc);
			assert(bias_batch_size == 1);
			assert(bias_cols == b_cols);
			assert(CCV_IS_TENSOR_CONTIGUOUS(bias));
			const void* const device_ones = ccv_nnc_stream_context_get_ones(stream_context, b_rows, b->info.datatype);
			CUBLAS_ENFORCE(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, b_cols, b_rows, 1, one, bias->data.u8, ccv_nnc_cuda_datatype(bias->info.datatype), bias_rows_inc, device_ones, ccv_nnc_cuda_datatype(b->info.datatype), 1, zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w_data, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, one, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		} else {
			CUBLAS_ENFORCE(cublasGemmEx(cublas, transa, transb, b_cols, b_rows, a_cols, one, w_data, ccv_nnc_cuda_datatype(w_datatype), lda_inc, a->data.u8, ccv_nnc_cuda_datatype(a->info.datatype), ldb_inc, zero, b->data.u8, ccv_nnc_cuda_datatype(b->info.datatype), b_rows_inc, ccv_nnc_cuda_compute_datatype(b->info.datatype, is_downcast), CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_sum_out(const int B, const int Hk, const int r, const int D, const NUM* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(i, B * Hk * D) {
		const int j = i / D;
		const int k = i % D;
		const NUM* const arow = a + j * r * D + k;
		float accum = (float)arow[0];
		for (int l = 1; l < r; l++)
			accum += (float)arow[l * D];
		b[i] = (NUM)accum;
	}
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	// NNC notation:
	// C = sm(Q * K^T) * V
	//
	// MFA notation:
	// O = sm(Q * K^T) * V
	assert(input_size >= 6);
	assert(output_size >= 3);
	if (cmd.info.scaled_dot_product_attention.is_varlen)
		return CCV_NNC_EXEC_INVALID;
	ccv_nnc_tensor_view_t* const d_o = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const attn_mask = input_size > 6 ? (ccv_nnc_tensor_view_t*)inputs[6] : 0;
	ccv_nnc_tensor_view_t* const weights = input_size > 7 ? (ccv_nnc_tensor_view_t*)inputs[7] : 0;
	ccv_nnc_tensor_view_t* const bias = input_size > 8 ? (ccv_nnc_tensor_view_t*)inputs[8] : 0;
	if (bias) // bias always requires a weight matrix.
		{ assert(weights); }
	ccv_nnc_tensor_view_t* const o = input_size > 9 ? (ccv_nnc_tensor_view_t*)inputs[9] : 0;
	ccv_nnc_tensor_view_t* const saved_softmax_lse = input_size > 10 ? (ccv_nnc_tensor_view_t*)inputs[10] : 0;
	// ccv_nnc_tensor_view_t* const qkv = input_size > 11 ? (ccv_nnc_tensor_view_t*)inputs[11] : 0;
	ccv_nnc_tensor_view_t* const dq = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dk = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dv = (ccv_nnc_tensor_view_t*)outputs[2];

	// Things we don't support.
	if (weights != 0)
		return CCV_NNC_EXEC_INVALID;

	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int o_nd = ccv_nnc_tensor_nd(o->info.dim);
	assert(o_nd == 3 || o_nd == 4);
	assert(q_nd == k_nd && k_nd == v_nd && v_nd == o_nd);

	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int odim[CCV_NNC_MAX_DIM_ALLOC];
	int amdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(o, odim);

	assert(q->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(k->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(v->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(o->info.format == CCV_TENSOR_FORMAT_NHWC);
	if (attn_mask) {
		// MFA does not support fused transposes on the mask.
		assert(attn_mask->info.format == CCV_TENSOR_FORMAT_NHWC);
	}

	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(o));
	assert(CCV_IS_TENSOR_CONTIGUOUS(d_o));
	assert(CCV_IS_TENSOR_CONTIGUOUS(saved_softmax_lse));

	if (attn_mask) {
		assert(CCV_IS_TENSOR_CONTIGUOUS(attn_mask));
	}

	int batch_size;
	int R;
	int C;
	int Hq;
	int Hk;
	int D;
	if (q_nd == 3) {
		batch_size = qdim[1];
		assert(batch_size == kdim[1]);
		R = qdim[2];
		C = kdim[2];
		Hq = Hk = 1;
		D = qdim[3];
		assert(D == kdim[3]);
	} else if (q_nd == 4) {
		batch_size = qdim[0];
		assert(batch_size == kdim[0]);
		R = qdim[1];
		C = kdim[1];
		Hq = qdim[2];
		Hk = kdim[2];
		assert(Hq >= Hk);
		assert(Hq % Hk == 0);
		D = qdim[3];
		assert(D == kdim[3]);
	}

	if (attn_mask) {
		// MFA can support am_nd == 2 and broadcast batch=1 -> batch=batch_size, but
		// wait until that occurs in practice before doing so.
		const int am_nd = ccv_nnc_tensor_nd(attn_mask->info.dim);
		assert(am_nd == 3 || am_nd == 4); // [batch_size, R, C]

		// MFA does not support attention mask broadcasting (where the R dimension
		// of Q > 1, but the R dimension of the mask == 1).
		ccv_nnc_tensor_view_get_dim(attn_mask, amdim);
		if (am_nd == 3)
		{
			assert(amdim[1] == batch_size || amdim[1] == 1);
			amdim[0] = amdim[1];
			amdim[1] = 1;
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		} else {
			assert(amdim[0] == batch_size || amdim[0] == 1);
			assert(amdim[1] == 1);
			assert(amdim[2] == R);
			assert(amdim[3] == C);
		}
	}
	int weights_datatype = 0;
	if (weights)
		weights_datatype = CCV_GET_DATA_TYPE(weights->info.datatype) == CCV_QX ? ((weights->info.datatype & 0xff) << 12) : weights->info.datatype;

	const int is_same_dtype =
		(q->info.datatype == k->info.datatype) &&
		(q->info.datatype == v->info.datatype) &&
		(q->info.datatype == o->info.datatype) &&
		(weights ? (q->info.datatype == weights_datatype) : 1) &&
		(bias ? (q->info.datatype == bias->info.datatype) : 1);

	assert(is_same_dtype);

	Flash_bwd_params params;
	memset(&params, 0, sizeof(params));
	params.is_bf16 = q->info.datatype == CCV_16BF;
	params.q_ptr = q->data.u8;
	params.k_ptr = k->data.u8;
	params.v_ptr = v->data.u8;
	params.q_row_stride = D * Hq;
	params.k_row_stride = D * Hk;
	params.v_row_stride = D * Hk;
	params.q_head_stride = D;
	params.k_head_stride = D;
	params.v_head_stride = D;
	params.q_batch_stride = R * Hq * D;
	params.k_batch_stride = C * Hk * D;
	params.v_batch_stride = C * Hk * D;
	auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
	params.seqlen_q = R;
	params.seqlen_q_rounded = round_multiple(R, 128);
	params.seqlen_k = C;
	params.seqlen_k_rounded = round_multiple(C, 128);
	params.d = D;
	assert(D % 8 == 0);
	params.d_rounded = round_multiple(D, 32);
	params.o_ptr = o->data.u8;
	params.o_row_stride = D * Hq;
	params.o_head_stride = D;
	params.o_batch_stride = R * Hq * D;
	params.b = batch_size;
	params.h = Hq;
	params.h_k = Hk;
	params.h_h_k_ratio = Hq / Hk;
	params.scale_softmax = cmd.info.scaled_dot_product_attention.scale;
	params.scale_softmax_log2 = cmd.info.scaled_dot_product_attention.scale * M_LOG2E;
	params.is_causal = cmd.info.scaled_dot_product_attention.is_causal;
	params.p_dropout = 1;
	params.p_dropout_in_uint8_t = 255;
	params.rp_dropout = 1;
	params.scale_softmax_rp_dropout = params.scale_softmax;
	params.window_size_left = ccv_max(R, C);
	params.window_size_right = params.is_causal ? 0 : ccv_max(R, C);
	params.is_seqlens_k_cumulative = true;
	params.dq_ptr = dq->data.u8;
	params.dk_ptr = dk->data.u8;
	params.dv_ptr = dv->data.u8;
	params.dq_row_stride = D * Hq;
	params.dk_row_stride = D * Hq; // This is not a typo, dk / dv is expanded and we sum it later.
	params.dv_row_stride = D * Hq;
	params.dq_head_stride = D;
	params.dk_head_stride = D;
	params.dv_head_stride = D;
	params.dq_batch_stride = R * Hq * D;
	params.dk_batch_stride = C * Hq * D;
	params.dv_batch_stride = C * Hq * D;
	params.do_ptr = d_o->data.u8;
	params.do_row_stride = D * Hq;
	params.do_head_stride = D;
	params.do_batch_stride = R * Hq * D;
	params.deterministic = cmd.info.scaled_dot_product_attention.deterministic;

	size_t dq_accum_size;
	if (params.deterministic)
	{
		const ccv_nnc_cuda_device_prop_t props = ccv_nnc_gpu_device_props();
		const int nsplits = (props.multi_processor_count + batch_size * Hq - 1) / (batch_size * Hq);
		dq_accum_size = sizeof(float) * nsplits * batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
		params.dq_accum_split_stride = batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
	} else {
		dq_accum_size = sizeof(float) * batch_size * params.seqlen_q_rounded * Hq * params.d_rounded;
		params.dq_accum_split_stride = 0;
	}

	params.softmax_lse_ptr = saved_softmax_lse->data.u8;
	if (Hq != Hk)
	{
		unsigned char* const workspace = (unsigned char*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size + sizeof(short) * batch_size * Hq * C * D * 2, CCV_TENSOR_GPU_MEMORY);
		params.dsoftmax_sum = workspace;
		params.dq_accum_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded;
		params.dk_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size;
		params.dv_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size + sizeof(short) * batch_size * Hq * C * D;
	} else {
		unsigned char* const workspace = (unsigned char*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * batch_size * Hq * params.seqlen_q_rounded + dq_accum_size, CCV_TENSOR_GPU_MEMORY);
		params.dsoftmax_sum = workspace;
		params.dq_accum_ptr = workspace + sizeof(float) * batch_size * Hq * params.seqlen_q_rounded;
		params.dk_accum_ptr = 0;
		params.dv_accum_ptr = 0;
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (params.deterministic)
		cudaMemsetAsync(params.dq_accum_ptr, 0, dq_accum_size, stream);
	run_mha_bwd(params, stream);
	CUDA_ENFORCE(cudaGetLastError());
	if (Hq != Hk)
	{
		_ccv_nnc_sum_out<<<CUDA_GET_BLOCKS(batch_size * C * Hk * D), CUDA_NUM_THREADS, 0, stream>>>(batch_size * C, Hk, Hq / Hk, D, (__half*)params.dk_ptr, (__half*)dk->data.f16);
		_ccv_nnc_sum_out<<<CUDA_GET_BLOCKS(batch_size * C * Hk * D), CUDA_NUM_THREADS, 0, stream>>>(batch_size * C, Hk, Hq / Hk, D, (__half*)params.dv_ptr, (__half*)dv->data.f16);
		CUDA_ENFORCE(cudaGetLastError());
	}
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
#endif
}
