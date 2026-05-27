#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif

static int _ccv_nnc_sparse_indexed_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6 || input_size == 7);
	assert(output_size == 1);
	const int attention_sinks = cmd.info.sparse_indexed_attention.attention_sinks;
	if ((attention_sinks != 0) != (input_size == 7))
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const q = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const dense_k = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const dense_v = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const sparse_k = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* const sparse_v = (const ccv_nnc_tensor_view_t*)inputs[4];
	const ccv_nnc_tensor_view_t* const indices = (const ccv_nnc_tensor_view_t*)inputs[5];
	const ccv_nnc_tensor_view_t* const sinks = attention_sinks ? (const ccv_nnc_tensor_view_t*)inputs[6] : 0;
	ccv_nnc_tensor_view_t* const out = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(dense_k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(dense_v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(sparse_k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(sparse_v));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(CCV_IS_TENSOR_CONTIGUOUS(out));
	if (sinks)
		assert(CCV_IS_TENSOR_CONTIGUOUS(sinks));
	if (q->info.datatype != dense_k->info.datatype || q->info.datatype != dense_v->info.datatype || q->info.datatype != sparse_k->info.datatype || q->info.datatype != sparse_v->info.datatype || q->info.datatype != out->info.datatype)
		return CCV_NNC_EXEC_INVALID;
	if (sinks && sinks->info.datatype != q->info.datatype)
		return CCV_NNC_EXEC_INVALID;
	if (indices->info.datatype != CCV_32S)
		return CCV_NNC_EXEC_INVALID;
	uint32_t mtl_data_type;
	if (q->info.datatype == CCV_16F)
		mtl_data_type = 16;
	else if (q->info.datatype == CCV_16BF)
		mtl_data_type = 121;
	else
		return CCV_NNC_EXEC_INVALID;
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	const int dense_k_nd = ccv_nnc_tensor_nd(dense_k->info.dim);
	const int dense_v_nd = ccv_nnc_tensor_nd(dense_v->info.dim);
	const int sparse_k_nd = ccv_nnc_tensor_nd(sparse_k->info.dim);
	const int sparse_v_nd = ccv_nnc_tensor_nd(sparse_v->info.dim);
	const int indices_nd = ccv_nnc_tensor_nd(indices->info.dim);
	const int out_nd = ccv_nnc_tensor_nd(out->info.dim);
	assert(q_nd == 3);
	assert(dense_k_nd == 2);
	assert(dense_v_nd == 2);
	assert(sparse_k_nd == 2);
	assert(sparse_v_nd == 2);
	assert(indices_nd == 1 || indices_nd == 2);
	assert(out_nd == 3);
	const int T = q->info.dim[0];
	const int H = q->info.dim[1];
	const int D = q->info.dim[2];
	const int dense_rows = dense_k->info.dim[0];
	const int sparse_rows = sparse_k->info.dim[0];
	const int K = (indices_nd == 1) ? 0 : indices->info.dim[1];
	if (H != 64 || (D != 512 && D != 128))
		return CCV_NNC_EXEC_INVALID;
	assert(dense_k->info.dim[1] == D);
	assert(dense_v->info.dim[0] == dense_rows);
	assert(dense_v->info.dim[1] == D);
	assert(sparse_k->info.dim[1] == D);
	assert(sparse_v->info.dim[0] == sparse_rows);
	assert(sparse_v->info.dim[1] == D);
	assert(indices->info.dim[0] == T);
	assert(out->info.dim[0] == T);
	assert(out->info.dim[1] == H);
	assert(out->info.dim[2] == D);
	if (mpgetbuffer((ccv_nnc_tensor_t*)dense_k) != mpgetbuffer((ccv_nnc_tensor_t*)dense_v) || dense_k->dataof != dense_v->dataof)
		return CCV_NNC_EXEC_INVALID;
	if (mpgetbuffer((ccv_nnc_tensor_t*)sparse_k) != mpgetbuffer((ccv_nnc_tensor_t*)sparse_v) || sparse_k->dataof != sparse_v->dataof)
		return CCV_NNC_EXEC_INVALID;
	uint32_t sink_head_stride = 0;
	if (sinks)
	{
		const int sink_count = ccv_nnc_tensor_count(sinks->info);
		assert(sink_count == 1 || sink_count == H);
		sink_head_stride = (sink_count == 1) ? 0 : 1;
	}
	@autoreleasepool {
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) || !ccv_nnc_mfa_has_neural_accelerators(context))
			return CCV_NNC_EXEC_INVALID;
		if (mtl_data_type == 121 && !ccv_nnc_mfa_neural_accelerators_support_bfloat(context))
			return CCV_NNC_EXEC_INVALID;
		if (cmd.algorithm >= 5)
			return CCV_NNC_EXEC_INVALID;
		const uint32_t variant = (cmd.algorithm < 0) ? ((D == 128) ? 4 : 2) : (uint32_t)cmd.algorithm;
		if ((D == 128 && (variant != 4 || K == 0)) || (D == 512 && variant >= 4))
			return CCV_NNC_EXEC_INVALID;
		const ccv_nnc_mfa_sparse_indexed_attention_params_t params = {
			.data_type = mtl_data_type,
			.T = (uint32_t)T,
			.dense_rows = (uint32_t)dense_rows,
			.sparse_rows = (uint32_t)sparse_rows,
			.H = (uint32_t)H,
			.D = (uint32_t)D,
			.K = (uint32_t)K,
			.scale = cmd.info.sparse_indexed_attention.scale,
			.is_causal = (uint8_t)(cmd.info.sparse_indexed_attention.is_causal != 0),
			.attention_sinks = (uint8_t)(attention_sinks != 0),
			.sink_head_stride = sink_head_stride,
			.variant = variant,
		};
		ccv_nnc_mfa_prepare_sparse_indexed_attention(context, params);
		mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		mtl_buffer_t* tensors[9] = {
			mpgetbuffer((ccv_nnc_tensor_t*)q),
			mpgetbuffer((ccv_nnc_tensor_t*)dense_k),
			mpgetbuffer((ccv_nnc_tensor_t*)dense_v),
			mpgetbuffer((ccv_nnc_tensor_t*)sparse_k),
			mpgetbuffer((ccv_nnc_tensor_t*)sparse_v),
			mpgetbuffer((ccv_nnc_tensor_t*)indices),
			attention_sinks ? mpgetbuffer((ccv_nnc_tensor_t*)sinks) : 0,
			mpgetbuffer((ccv_nnc_tensor_t*)out),
			NULL,
		};
		size_t tensor_offsets[8] = {
			q->dataof,
			dense_k->dataof,
			dense_v->dataof,
			sparse_k->dataof,
			sparse_v->dataof,
			indices->dataof,
			attention_sinks ? sinks->dataof : 0,
			out->dataof,
		};
		ccv_nnc_mfa_encode_sparse_indexed_attention(context, params, command_batch, tensors, tensor_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sparse_indexed_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 5;
	registry->exec = _ccv_nnc_sparse_indexed_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sparse_indexed_attention_back;
}
