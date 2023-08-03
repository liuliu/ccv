#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

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

	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int ostride[CCV_NNC_MAX_DIM_ALLOC];
	int amstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(o, ostride);

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
	int H;
	int D;
	if (q_nd == 3) {
		batch_size = qdim[0];
		assert(batch_size == kdim[0]);
		R = qdim[1];
		C = kdim[1];
		H = 1;
		D = qdim[2];
		assert(D == kdim[2]);
	} else if (q_nd == 4) {
		batch_size = qdim[0];
		assert(batch_size == kdim[0]);
		R = qdim[1];
		C = kdim[1];
		H = qdim[2];
		assert(H == kdim[2]);
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
		ccv_nnc_tensor_view_get_stride(attn_mask, amstride);
    if (am_nd == 3)
    {
      assert(amdim[1] == batch_size);
      amdim[0] = amdim[1];
      amdim[1] = 1;
      assert(amdim[2] == R);
      assert(amdim[3] == C);
    } else {
      assert(amdim[0] == batch_size);
      assert(amdim[1] == 1);
      assert(amdim[2] == R);
      assert(amdim[3] == C);
    }
	}

	const int is_same_dtype =
		(q->info.datatype == k->info.datatype) &&
		(q->info.datatype == v->info.datatype) &&
		(q->info.datatype == o->info.datatype) &&
		(weights ? (q->info.datatype == weights->info.datatype) : 1) &&
		(bias ? (q->info.datatype == bias->info.datatype) : 1);
	assert(is_same_dtype);

	uint32_t mtl_data_type = UINT32_MAX;
	switch (q->info.datatype) {
		case CCV_16F: {
			mtl_data_type = 16;
			break;
		}
		case CCV_32F: {
			mtl_data_type = 3;
			break;
		}
		default: {
			assert(false);
			break;
		}
	}

	ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
	if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION)) {
		assert(false); // MFA is required.
	}

	int attention_is_batched = (batch_size > 1);
	ccv_nnc_mfa_attention_params_t params = {
		.data_type = mtl_data_type,
		.R = (uint32_t)R,
		.C = (uint32_t)C,
		.H = (uint32_t)H,
		.D = (uint32_t)D,
		.Q_trans = false,
		.K_trans = true,
		.V_trans = false,
		.O_trans = false,
		.alpha = cmd.info.scaled_dot_product_attention.scale,
		.batched = (attention_is_batched ? 1 : 0),
		.masked = (attn_mask != NULL ? 1 : 0),

		.batch_dims_q = { 0 },
		.batch_dims_mask = { 0 },
	};
	if (attention_is_batched) {
		params.batch_dims_q[0] = batch_size;
		params.batch_dims_q[1] = 0;
		params.batch_dims_mask[0] = amdim[0];
		params.batch_dims_mask[1] = 0;
	}
	ccv_nnc_mfa_prepare_attention(context, params);

	mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
	mtl_buffer_t* mask_buffer = NULL;
	if (params.masked) {
		mask_buffer = mpgetbuffer((ccv_nnc_tensor_t*)attn_mask);
	}
	mtl_buffer_t* tensors[6] = {
		mpgetbuffer((ccv_nnc_tensor_t*)q),
		mpgetbuffer((ccv_nnc_tensor_t*)k),
		mpgetbuffer((ccv_nnc_tensor_t*)v),
		mpgetbuffer((ccv_nnc_tensor_t*)o),
		mask_buffer,
		NULL,
	};
	size_t tensor_offsets[5] = {
		q->dataof,
		k->dataof,
		v->dataof,
		o->dataof,
		attn_mask ? attn_mask->dataof : 0,
	};
	ccv_nnc_mfa_encode_attention(context, params, command_batch, tensors, tensor_offsets);

	// NNC notation:
	// D = C * W^T + bias
	//
	// MFA notation:
	// A <- O
	// B <- feedforward weights
	// D <- bias
	// C = A * B^T + D
	//
	// For MFA, C is the output of GEMM, not the output of attention. D stands for
	// "data" (arbitrary user-defined data). In this case, the user-defined data
	// is the bias vector.
	if (weights) {
		int *adim = odim;
		ccv_nnc_tensor_view_t* const a = o; // left input matrix
		ccv_nnc_tensor_view_t* const b = weights; // weights
		ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];

		const int b_nd = ccv_nnc_tensor_nd(weights->info.dim);
		assert(b_nd == 2);
		assert(CCV_IS_TENSOR_CONTIGUOUS(bias));
		const int c_nd = ccv_nnc_tensor_nd(c->info.dim);
		assert(c_nd == 3);

		int cdim[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_dim(c, cdim);

		const int attention_batch_size = batch_size;
		const int gemm_batch_size = cdim[1];
		int gemm_is_batched = (gemm_batch_size > 1);
		if (attention_is_batched) {
			assert(gemm_is_batched);
			assert(attention_batch_size == gemm_batch_size);
		}
		if (!gemm_is_batched) {
			assert(!attention_is_batched);
		}

		// The C matrix of the GEMM cannot be transposed, so the assume the C matrix
		// is NHWC.
		assert(c->info.format == CCV_TENSOR_FORMAT_NHWC);
		int M = cdim[2];
		int N = cdim[3];
		int K = H * D;

		assert(adim[0] == attention_batch_size);
		assert(adim[1] == M);
		if (H > 1) {
			assert(adim[2] * adim[3] == K);
		} else {
			assert(adim[2] == K);
		}

		// We assume the weights matrix is square.
		assert(K == N);
		assert(b->info.dim[0] == N);
		assert(b->info.dim[1] == K);

		if (bias) {
			const int bias_nd = ccv_nnc_tensor_nd(bias->info.dim);

			// Since the weights matrix doesn't have a batch dimension, the bias
			// vector doesn't either.
			assert(bias_nd == 1);
			assert(CCV_IS_TENSOR_CONTIGUOUS(bias));
			assert(bias->info.dim[0] == N);
		}

		ccv_nnc_mfa_gemm_params_t params = {
			.data_type = mtl_data_type,
			.M = (uint32_t)M,
			.N = (uint32_t)N,
			.K = (uint32_t)K,
			.A_trans = false,
			.B_trans = true,
			.D_trans = false,
			.alpha = (float)1.0,
			.beta = (float)0.0,
			.batched = (gemm_is_batched ? 1 : 0),
			.fused_activation_function = 0,
			.fused_bias = (bias ? 1 : 0),

			.batch_dims_a = { 0 },
			.batch_dims_b = { 0 },
			.batch_dims_d = { 0 },
		};
		if (gemm_is_batched) {
			params.batch_dims_a[0] = gemm_batch_size;
			params.batch_dims_a[1] = 0;
		}
		ccv_nnc_mfa_prepare_gemm(context, params);

		mtl_buffer_t* bias_buffer = NULL;
		if (bias) {
			bias_buffer = mpgetbuffer((ccv_nnc_tensor_t*)bias);
		}
		mtl_buffer_t* tensors[5] = {
			mpgetbuffer((ccv_nnc_tensor_t*)a), // A
			mpgetbuffer((ccv_nnc_tensor_t*)b), // B
			mpgetbuffer((ccv_nnc_tensor_t*)c), // C
			bias_buffer, // D
			NULL,
		};
		size_t tensor_offsets[4] = {
			a->dataof, // A offset
			b->dataof, // B offset
			c->dataof, // C offset
			bias ? bias->dataof : 0, // D offset
		};
		ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);
	}

	ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_attention_back;
}
