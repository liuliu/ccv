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
	if (bias) // bias always requires a weight matrix.
		{ assert(weights); }

	ccv_nnc_tensor_view_t* const saved_softmax = NULL;
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

	if (saved_softmax) {
		// MFA does not support a backward pass and cannot store the intermediate
		// softmax. If this is required, fall back to MPSGraph (if will never occur
		// during inference).
		assert(false);
		return CCV_NNC_EXEC_INVALID;
	}

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

	Flash_fwd_params params;
	memset(&params, 0, sizeof(params));
	params.is_bf16 = false;
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
	// TODO: Support num_splits.
	const int block_n = D <= 64 ? 256 : (D <= 128 ? 128 : 64);
	const int num_n_blocks = (C + block_n - 1) / block_n;
	// Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
	// In any case we don't expect seqlen_q to be larger than 64 for inference.
	const int num_m_blocks = (R + 64 - 1) / 64;
	const ccv_nnc_cuda_device_prop_t props = ccv_nnc_gpu_device_props();
	params.num_splits = num_splits_heuristic(batch_size * Hq * num_m_blocks, props.multi_processor_count, num_n_blocks, 128);
	if (params.num_splits > 1)
	{
		float* const workspace = (float*)ccv_nnc_stream_context_get_workspace(stream_context, (batch_size * Hq * R + params.num_splits * batch_size * Hq * R + params.num_splits * batch_size * Hq * R * params.d_rounded) * sizeof(float), CCV_TENSOR_GPU_MEMORY);
		params.softmax_lse_ptr = workspace;
		params.softmax_lseaccum_ptr = workspace + batch_size * Hq * R;
		params.oaccum_ptr = workspace + batch_size * Hq * R + params.num_splits * batch_size * Hq * R;
	} else {
		void* const workspace = ccv_nnc_stream_context_get_workspace(stream_context, batch_size * Hq * R * sizeof(float), CCV_TENSOR_GPU_MEMORY);
		params.softmax_lse_ptr = workspace;
	}
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	run_mha_fwd(params, stream, false);
	CUDA_ENFORCE(cudaGetLastError());

	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA_SM80
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
#endif
}
