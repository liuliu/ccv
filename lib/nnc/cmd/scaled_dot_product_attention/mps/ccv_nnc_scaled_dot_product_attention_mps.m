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
	int H;
	int D;
	if (q_nd == 3) {
		batch_size = qdim[1];
		assert(batch_size == kdim[1]);
		R = qdim[2];
		C = kdim[2];
		H = 1;
		D = qdim[3];
		assert(D == kdim[3]);
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

	@autoreleasepool {
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION)) {
			assert(false); // MFA is required.
			return CCV_NNC_EXEC_INVALID;
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
			params.batch_dims_mask[0] = attn_mask ? amdim[0] : batch_size;
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

			if (o_nd == 3)
			{
				assert(adim[1] == attention_batch_size);
				assert(adim[2] == M);
			} else {
				assert(adim[0] == attention_batch_size);
				assert(adim[1] == M);
			}
			if (H > 1) {
				assert(adim[2] * adim[3] == K);
			} else {
				assert(adim[3] == K);
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
			mtl_buffer_t* weights_data = mpgetbuffer((ccv_nnc_tensor_t*)weights);
			size_t weights_dataof = weights->dataof;
			if (CCV_GET_DATA_TYPE(weights->info.datatype) == CCV_QX)
			{
				ccv_nnc_tensor_param_t weights_params = weights->info;
				const int palette_datatype = (weights_params.datatype & 0xff) << 12;
				ccv_nnc_tensor_param_t depalettize_weights_params = weights_params;
				depalettize_weights_params.datatype = palette_datatype;
				depalettize_weights_params.reserved = 0;
				size_t weights_data_size = ccv_nnc_tensor_data_size(depalettize_weights_params);
				const size_t count = ccv_nnc_tensor_count(weights_params);
				const int qbits = (weights_params.datatype & 0xf00) >> 8;
				const int number_in_blocks = weights_params.reserved;
				ccv_nnc_mfa_depalettize_params_t weights_depalettize_params = {
					.data_type = mtl_data_type,
					.qbits = (uint32_t)qbits,
					.number_in_blocks = (uint32_t)number_in_blocks,
					.length = (uint64_t)count,
				};
				ccv_nnc_mfa_prepare_depalettize(context, weights_depalettize_params);
				weights_data = ccv_nnc_mfa_request_scratch(context, weights_data_size);
				weights_dataof = 0;
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer((ccv_nnc_tensor_t*)weights), // A
					(mtl_buffer_t*)weights_data, // B
					NULL,
				};
				size_t tensor_offsets[2] = {
					weights->dataof, // A offset
					0, // B offset
				};
				ccv_nnc_mfa_encode_depalettize(context, weights_depalettize_params, command_batch, tensors, tensor_offsets);
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
				weights_data, // B
				mpgetbuffer((ccv_nnc_tensor_t*)c), // C
				bias_buffer, // D
				NULL,
			};
			size_t tensor_offsets[4] = {
				a->dataof, // A offset
				weights_dataof, // B offset
				c->dataof, // C offset
				bias ? bias->dataof : 0, // D offset
			};
			ccv_nnc_mfa_encode_gemm(context, params, command_batch, tensors, tensor_offsets);
		}

		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 6);
	assert(!cmd.info.scaled_dot_product_attention.is_causal);
	ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const q = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const k = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const dq = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dk = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const dv = (ccv_nnc_tensor_view_t*)outputs[2];
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	assert(q_nd == 3 || q_nd == 4);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	assert(k_nd == 3 || k_nd == 4);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	assert(v_nd == 3 || v_nd == 4);
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd == 3 || g_nd == 4);
	const int dq_nd = ccv_nnc_tensor_nd(dq->info.dim);
	assert(dq_nd == 3 || dq_nd == 4);
	assert(dq_nd == q_nd);
	const int dk_nd = ccv_nnc_tensor_nd(dk->info.dim);
	assert(dk_nd == 3 || dk_nd == 4);
	assert(dk_nd == k_nd);
	const int dv_nd = ccv_nnc_tensor_nd(dv->info.dim);
	assert(dv_nd == 3 || dv_nd == 4);
	assert(dv_nd == v_nd);
	assert(q_nd == k_nd && k_nd == v_nd && v_nd == g_nd);
	// Assuming this is float 32.
	int qdim[CCV_NNC_MAX_DIM_ALLOC];
	int kdim[CCV_NNC_MAX_DIM_ALLOC];
	int vdim[CCV_NNC_MAX_DIM_ALLOC];
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int dqdim[CCV_NNC_MAX_DIM_ALLOC];
	int dkdim[CCV_NNC_MAX_DIM_ALLOC];
	int dvdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(q, qdim);
	ccv_nnc_tensor_view_get_dim(k, kdim);
	ccv_nnc_tensor_view_get_dim(v, vdim);
	ccv_nnc_tensor_view_get_dim(g, gdim);
	ccv_nnc_tensor_view_get_dim(dq, dqdim);
	ccv_nnc_tensor_view_get_dim(dk, dkdim);
	ccv_nnc_tensor_view_get_dim(dv, dvdim);
	if (q_nd == 3)
	{
		qdim[0] = qdim[1], qdim[1] = qdim[2], qdim[2] = 1;
		kdim[0] = kdim[1], kdim[1] = kdim[2], kdim[2] = 1;
		vdim[0] = vdim[1], vdim[1] = vdim[2], vdim[2] = 1;
		gdim[0] = gdim[1], gdim[1] = gdim[2], gdim[2] = 1;
		dqdim[0] = dqdim[1], dqdim[1] = dqdim[2], dqdim[2] = 1;
		dkdim[0] = dkdim[1], dkdim[1] = dkdim[2], dkdim[2] = 1;
		dvdim[0] = dvdim[1], dvdim[1] = dvdim[2], dvdim[2] = 1;
	}
	assert(qdim[0] == kdim[0] && kdim[0] == vdim[0] && vdim[0] == gdim[0]);
	assert(qdim[2] == kdim[2] && kdim[2] == vdim[2] && vdim[2] == gdim[2]);
	assert(qdim[3] == kdim[3]);
	assert(kdim[1] == vdim[1]);
	assert(gdim[1] == qdim[1]);
	assert(gdim[3] == vdim[3]);
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	int qstride[CCV_NNC_MAX_DIM_ALLOC];
	int kstride[CCV_NNC_MAX_DIM_ALLOC];
	int vstride[CCV_NNC_MAX_DIM_ALLOC];
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	int dqstride[CCV_NNC_MAX_DIM_ALLOC];
	int dkstride[CCV_NNC_MAX_DIM_ALLOC];
	int dvstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(q, qstride);
	ccv_nnc_tensor_view_get_stride(k, kstride);
	ccv_nnc_tensor_view_get_stride(v, vstride);
	ccv_nnc_tensor_view_get_stride(g, gstride);
	ccv_nnc_tensor_view_get_stride(dq, dqstride);
	ccv_nnc_tensor_view_get_stride(dk, dkstride);
	ccv_nnc_tensor_view_get_stride(dv, dvstride);
	if (q_nd == 3)
	{
		qstride[0] = qstride[1], qstride[1] = qstride[2], qstride[2] = qstride[3];
		kstride[0] = kstride[1], kstride[1] = kstride[2], kstride[2] = kstride[3];
		vstride[0] = vstride[1], vstride[1] = vstride[2], vstride[2] = vstride[3];
		gstride[0] = gstride[1], gstride[1] = gstride[2], gstride[2] = gstride[3];
		dqstride[0] = dqstride[1], dqstride[1] = dqstride[2], dqstride[2] = dqstride[3];
		dkstride[0] = dkstride[1], dkstride[1] = dkstride[2], dkstride[2] = dkstride[3];
		dvstride[0] = dvstride[1], dvstride[1] = dvstride[2], dvstride[2] = dvstride[3];
	}
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[4];
		const int* gdim_r = gdim;
		const int* gstride_r = gstride;
		const int* qdim_r = qdim;
		const int* qstride_r = qstride;
		const int* kdim_r = kdim;
		const int* kstride_r = kstride;
		const int* vdim_r = vdim;
		const int* vstride_r = vstride;
		const float scale = cmd.info.scaled_dot_product_attention.scale;

		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, gdim_r, gstride_r, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, gdim_r, gstride_r);
			[inputShapedTypes addObject:mps_g_shape];

			MPSGraphTensor* mps_input_q;
			MPSGraphTensor* mps_q = ccv_nnc_mps_graph_tensor_input(graph, q, qdim_r, qstride_r, &mps_input_q);
			[inputTensors addObject:mps_input_q];
			MPSGraphShapedType* mps_q_shape = ccv_nnc_mps_graph_tensor_input_shape(q, qdim_r, qstride_r);
			[inputShapedTypes addObject:mps_q_shape];

			MPSGraphTensor* mps_input_k;
			MPSGraphTensor* mps_k = ccv_nnc_mps_graph_tensor_input(graph, k, kdim_r, kstride_r, &mps_input_k);
			[inputTensors addObject:mps_input_k];
			MPSGraphShapedType* mps_k_shape = ccv_nnc_mps_graph_tensor_input_shape(k, kdim_r, kstride_r);
			[inputShapedTypes addObject:mps_k_shape];

			MPSGraphTensor* mps_input_v;
			MPSGraphTensor* mps_v = ccv_nnc_mps_graph_tensor_input(graph, v, vdim_r, vstride_r, &mps_input_v);
			[inputTensors addObject:mps_input_v];
			MPSGraphShapedType* mps_v_shape = ccv_nnc_mps_graph_tensor_input_shape(v, vdim_r, vstride_r);
			[inputShapedTypes addObject:mps_v_shape];

			MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:ccv_nnc_mps_datatype(q->info.datatype)];
			mps_q = [graph multiplicationWithPrimaryTensor:mps_scale secondaryTensor:[graph transposeTensor:mps_q dimension:1 withDimension:2 name:nil] name:nil];
			mps_k = [graph transposeTensor:mps_k dimension:1 withDimension:2 name:nil];
			MPSGraphTensor* mps_kt = [graph transposeTensor:mps_k dimension:2 withDimension:3 name:nil];
			mps_v = [graph transposeTensor:mps_v dimension:1 withDimension:2 name:nil];
			MPSGraphTensor* mps_qk = [graph matrixMultiplicationWithPrimaryTensor:mps_q secondaryTensor:mps_kt name:nil];
			MPSGraphTensor* mps_softmax = [graph softMaxWithTensor:mps_qk axis:3 name:nil];
			mps_g = [graph transposeTensor:mps_g dimension:1 withDimension:2 name:nil];
			MPSGraphTensor* mps_softmaxt = [graph transposeTensor:mps_softmax dimension:2 withDimension:3 name:nil];
			MPSGraphTensor* mps_dv = [graph matrixMultiplicationWithPrimaryTensor:mps_softmaxt secondaryTensor:mps_g name:nil];
			mps_v = [graph transposeTensor:mps_v dimension:2 withDimension:3 name:nil];
			MPSGraphTensor* mps_dsoftmax = [graph matrixMultiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_v name:nil];
			MPSGraphTensor* mulTensor = [graph multiplicationWithPrimaryTensor:mps_softmax secondaryTensor:mps_dsoftmax name:nil];
			MPSGraphTensor* mulSumTensor = [graph reductionSumWithTensor:mulTensor axis:-1 name:nil];
			MPSGraphTensor* gradSubTensor = [graph subtractionWithPrimaryTensor:mps_dsoftmax secondaryTensor:mulSumTensor name:nil];
			MPSGraphTensor* mps_dqk = [graph multiplicationWithPrimaryTensor:mps_softmax secondaryTensor:gradSubTensor name:nil];
			MPSGraphTensor* mps_dq = [graph multiplicationWithPrimaryTensor:mps_scale secondaryTensor:[graph matrixMultiplicationWithPrimaryTensor:mps_dqk secondaryTensor:mps_k name:nil] name:nil];
			mps_dqk = [graph transposeTensor:mps_dqk dimension:2 withDimension:3 name:nil];
			MPSGraphTensor* mps_dk = [graph matrixMultiplicationWithPrimaryTensor:mps_dqk secondaryTensor:mps_q name:nil];
			mps_dq = [graph transposeTensor:mps_dq dimension:1 withDimension:2 name:nil];
			[resultTensors addObject:mps_dq];
			mps_dk = [graph transposeTensor:mps_dk dimension:1 withDimension:2 name:nil];
			[resultTensors addObject:mps_dk];
			mps_dv = [graph transposeTensor:mps_dv dimension:1 withDimension:2 name:nil];
			[resultTensors addObject:mps_dv];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, gdim, gstride);
		MPSGraphTensorData* data_q = ccv_nnc_mps_graph_tensor_data(q, qdim, qstride);
		MPSGraphTensorData* data_k = ccv_nnc_mps_graph_tensor_data(k, kdim, kstride);
		MPSGraphTensorData* data_v = ccv_nnc_mps_graph_tensor_data(v, vdim, vstride);
		MPSGraphTensorData* data[] = {data_g, data_q, data_k, data_v};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]]], (ccv_nnc_tensor_view_t*[]){ dq, dk, dv }, (int*[]){ dqdim, dkdim, dvdim }, (int*[]){ dqstride, dkstride, dvstride }, 3);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
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
