#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

enum {
	CCV_NNC_MFA_MTL_DATA_TYPE_FLOAT = 3,
	CCV_NNC_MFA_MTL_DATA_TYPE_HALF = 16,
	CCV_NNC_MFA_MTL_DATA_TYPE_BFLOAT = 121,
};

// Elements between consecutive tokens (dimension 1).
static uint32_t _ccv_nnc_gated_delta_row_stride(const ccv_nnc_tensor_view_t* const tensor)
{
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	if (CCV_IS_TENSOR_VIEW(tensor))
		return (uint32_t)tensor->stride[1];
	uint32_t stride = 1;
	int i;
	for (i = 2; i < nd; i++)
		stride *= tensor->info.dim[i];
	return stride;
}

// The kernels stride token rows only: everything below a row is packed, and batches follow
// each other row by row.
static int _ccv_nnc_gated_delta_rows_are_packed(const ccv_nnc_tensor_view_t* const tensor)
{
	if (!CCV_IS_TENSOR_VIEW(tensor) || CCV_IS_TENSOR_CONTIGUOUS(tensor))
		return 1;
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	int i;
	int stride = 1;
	for (i = nd - 1; i >= 2; i--)
	{
		if (tensor->stride[i] != stride)
			return 0;
		stride *= tensor->info.dim[i];
	}
	return tensor->stride[1] >= stride && (tensor->info.dim[0] == 1 || tensor->stride[0] == tensor->stride[1] * tensor->info.dim[1]);
}

static int _ccv_nnc_gated_delta_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int sigmoid_gate = cmd.info.gated_delta.sigmoid_gate;
	assert(input_size == (sigmoid_gate ? 8 : 6));
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const q = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const k = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const v = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const log_decay = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* const beta = (const ccv_nnc_tensor_view_t*)inputs[4];
	const ccv_nnc_tensor_view_t* const state_in = (const ccv_nnc_tensor_view_t*)inputs[5];
	const ccv_nnc_tensor_view_t* const y = (const ccv_nnc_tensor_view_t*)outputs[0];
	const ccv_nnc_tensor_view_t* const state_out = (const ccv_nnc_tensor_view_t*)outputs[1];
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	const int v_nd = ccv_nnc_tensor_nd(v->info.dim);
	const int log_decay_nd = ccv_nnc_tensor_nd(log_decay->info.dim);
	const int beta_nd = ccv_nnc_tensor_nd(beta->info.dim);
	const int state_in_nd = ccv_nnc_tensor_nd(state_in->info.dim);
	const int y_nd = ccv_nnc_tensor_nd(y->info.dim);
	const int state_out_nd = ccv_nnc_tensor_nd(state_out->info.dim);
	assert(q_nd == 4);
	assert(k_nd == 4);
	assert(v_nd == 4);
	assert(log_decay_nd == 3 || log_decay_nd == 4);
	assert(beta_nd == 3);
	assert(state_in_nd == 4);
	assert(y_nd == 4);
	assert(state_out_nd == 4);
	const int B = q->info.dim[0];
	const int T = q->info.dim[1];
	const int Hk = q->info.dim[2];
	const int Dk = q->info.dim[3];
	const int Hv = v->info.dim[2];
	const int Dv = v->info.dim[3];
	const int state_checkpoint_count = cmd.info.gated_delta.state_checkpoint_count;
	const int state_history_count = state_checkpoint_count + 1;
	assert(state_checkpoint_count >= 0);
	assert(state_checkpoint_count < T);
	assert(k->info.dim[0] == B);
	assert(k->info.dim[1] == T);
	assert(k->info.dim[2] == Hk);
	assert(k->info.dim[3] == Dk);
	assert(v->info.dim[0] == B);
	assert(v->info.dim[1] == T);
	assert(log_decay->info.dim[0] == B);
	assert(log_decay->info.dim[1] == T);
	assert(log_decay->info.dim[2] == Hv);
	assert(log_decay_nd == 3 || log_decay->info.dim[3] == Dk);
	assert(beta->info.dim[0] == B);
	assert(beta->info.dim[1] == T);
	assert(beta->info.dim[2] == Hv);
	assert(state_in->info.dim[0] == B);
	assert(state_in->info.dim[1] == Hv);
	assert(state_in->info.dim[2] == Dv);
	assert(state_in->info.dim[3] == Dk);
	assert(y->info.dim[0] == B);
	assert(y->info.dim[1] == T);
	assert(y->info.dim[2] == Hv);
	assert(y->info.dim[3] == Dv);
	assert(state_out->info.dim[0] == B);
	assert(state_out->info.dim[1] == Hv * state_history_count);
	assert(state_out->info.dim[2] == Dv);
	assert(state_out->info.dim[3] == Dk);
	assert(Hv % Hk == 0);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		const int input_datatype = q->info.datatype;
		// A raw sigmoid gate may come straight from a projection in the input datatype.
		if ((input_datatype != CCV_32F && input_datatype != CCV_16F && input_datatype != CCV_16BF) ||
			k->info.datatype != input_datatype || v->info.datatype != input_datatype ||
			(log_decay->info.datatype != CCV_32F && !(sigmoid_gate && log_decay->info.datatype == input_datatype)) || (beta->info.datatype != CCV_32F && beta->info.datatype != input_datatype) ||
			state_in->info.datatype != CCV_32F || y->info.datatype != input_datatype || state_out->info.datatype != CCV_32F)
			return CCV_NNC_EXEC_INVALID;
		if (!_ccv_nnc_gated_delta_rows_are_packed(q) || !_ccv_nnc_gated_delta_rows_are_packed(k) || !_ccv_nnc_gated_delta_rows_are_packed(v) ||
			!_ccv_nnc_gated_delta_rows_are_packed(log_decay) || !_ccv_nnc_gated_delta_rows_are_packed(beta) ||
			!CCV_IS_TENSOR_CONTIGUOUS(state_in) || !CCV_IS_TENSOR_CONTIGUOUS(y) || !CCV_IS_TENSOR_CONTIGUOUS(state_out))
			return CCV_NNC_EXEC_INVALID;
		if (sigmoid_gate)
		{
			const ccv_nnc_tensor_view_t* const gate_bias = (const ccv_nnc_tensor_view_t*)inputs[6];
			const ccv_nnc_tensor_view_t* const gate_scale = (const ccv_nnc_tensor_view_t*)inputs[7];
			if (!cmd.info.gated_delta.log_decay || log_decay_nd != 4 ||
				gate_bias->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(gate_bias) || ccv_nnc_tensor_count(gate_bias->info) != Hv * Dk ||
				gate_scale->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(gate_scale) || ccv_nnc_tensor_count(gate_scale->info) != Hv)
				return CCV_NNC_EXEC_INVALID;
		}
		ccv_nnc_mfa_gated_delta_params_t params = {
			.batch_size = (uint32_t)B,
			.sequence_length = (uint32_t)T,
			.key_head_count = (uint32_t)Hk,
			.value_head_count = (uint32_t)Hv,
			.key_dim = (uint32_t)Dk,
			.value_dim = (uint32_t)Dv,
			.data_type = input_datatype == CCV_16F ? CCV_NNC_MFA_MTL_DATA_TYPE_HALF : (input_datatype == CCV_16BF ? CCV_NNC_MFA_MTL_DATA_TYPE_BFLOAT : CCV_NNC_MFA_MTL_DATA_TYPE_FLOAT),
			.beta_data_type = beta->info.datatype == CCV_16F ? CCV_NNC_MFA_MTL_DATA_TYPE_HALF : (beta->info.datatype == CCV_16BF ? CCV_NNC_MFA_MTL_DATA_TYPE_BFLOAT : CCV_NNC_MFA_MTL_DATA_TYPE_FLOAT),
			.state_checkpoint_count = (uint32_t)state_checkpoint_count,
			.decay_data_type = log_decay->info.datatype == CCV_16F ? CCV_NNC_MFA_MTL_DATA_TYPE_HALF : (log_decay->info.datatype == CCV_16BF ? CCV_NNC_MFA_MTL_DATA_TYPE_BFLOAT : CCV_NNC_MFA_MTL_DATA_TYPE_FLOAT),
			.q_row_stride = _ccv_nnc_gated_delta_row_stride(q),
			.k_row_stride = _ccv_nnc_gated_delta_row_stride(k),
			.v_row_stride = _ccv_nnc_gated_delta_row_stride(v),
			.decay_row_stride = _ccv_nnc_gated_delta_row_stride(log_decay),
			.beta_row_stride = _ccv_nnc_gated_delta_row_stride(beta),
			.q_scale = cmd.info.gated_delta.q_scale,
			.gate_lower_bound = cmd.info.gated_delta.gate_lower_bound,
			.log_decay = cmd.info.gated_delta.log_decay != 0,
			.channel_decay = log_decay_nd == 4,
			.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			.qk_l2norm = cmd.info.gated_delta.qk_l2norm != 0,
			.sigmoid_gate = sigmoid_gate != 0,
			.sigmoid_beta = cmd.info.gated_delta.sigmoid_beta != 0,
		};
		ccv_nnc_mfa_prepare_gated_delta(context, params);
		mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		// The sigmoid gate's bias and scale follow at 9 and 10; without the gate, the decay stands in.
		mtl_buffer_t* tensors[11] = {
			mpgetbuffer(inputs[0]),
			mpgetbuffer(inputs[1]),
			mpgetbuffer(inputs[2]),
			mpgetbuffer(inputs[3]),
			mpgetbuffer(inputs[4]),
			mpgetbuffer(inputs[5]),
			mpgetbuffer(outputs[0]),
			mpgetbuffer(outputs[1]),
			NULL,
			mpgetbuffer(inputs[sigmoid_gate ? 6 : 3]),
			mpgetbuffer(inputs[sigmoid_gate ? 7 : 3]),
		};
		size_t tensor_offsets[11] = {
			q->dataof,
			k->dataof,
			v->dataof,
			log_decay->dataof,
			beta->dataof,
			state_in->dataof,
			y->dataof,
			state_out->dataof,
			0,
			sigmoid_gate ? ((const ccv_nnc_tensor_view_t*)inputs[6])->dataof : log_decay->dataof,
			sigmoid_gate ? ((const ccv_nnc_tensor_view_t*)inputs[7])->dataof : log_decay->dataof,
		};
		ccv_nnc_mfa_encode_gated_delta(context, params, command_batch, tensors, tensor_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GATED_DELTA_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gated_delta_forw;
}
