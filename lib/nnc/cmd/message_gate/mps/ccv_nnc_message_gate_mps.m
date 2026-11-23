#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_message_gate_mtl_data_type(const int datatype)
{
	switch (datatype)
	{
		case CCV_32F:
			return 3; // MTLDataTypeFloat
		case CCV_16F:
			return 16; // MTLDataTypeHalf
		case CCV_16BF:
			return 121; // MTLDataTypeBFloat
	}
	return 0;
}

// The byte range a tensor takes of its buffer.
static int _ccv_nnc_message_gate_overlaps(const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b)
{
	// On MPS, data.u8 is the buffer and dataof the offset into it.
	if (a->data.u8 != b->data.u8)
		return 0;
	const off_t a_end = a->dataof + (off_t)ccv_nnc_tensor_data_size_without_padding(a->info);
	const off_t b_end = b->dataof + (off_t)ccv_nnc_tensor_data_size_without_padding(b->info);
	return a->dataof < b_end && b->dataof < a_end;
}

// A projection is dense, or 8-bit rowwise, its scales for rows of its last axis, which is then the k
// of a row here; its type, or its row scales' when quantized, is returned as a Metal data type, 0
// when unsupported.
static int _ccv_nnc_message_gate_weight_type(const ccv_nnc_tensor_t* const weight, const int k, int* const quantized)
{
	*quantized = CCV_GET_DATA_TYPE(weight->info.datatype) == CCV_QX;
	const int nd = ccv_nnc_tensor_nd(weight->info.dim);
	if (*quantized && ((weight->info.datatype & 0xf00) != CCV_NNC_QX_8I_ROWWISE || nd < 1 || weight->info.dim[nd - 1] != k))
		return 0;
	return _ccv_nnc_message_gate_mtl_data_type(*quantized ? (weight->info.datatype & 0xff) << 12 : weight->info.datatype);
}

static int _ccv_nnc_message_gate_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 5 || output_size != 3)
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < 5; i++)
		if (!inputs[i] || !CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 3; i++)
		if (!outputs[i] || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return CCV_NNC_EXEC_INVALID;
	ccv_nnc_tensor_t* const y = outputs[0];
	// Scratch: the normalized state and message side by side, and the mix.
	ccv_nnc_tensor_t* const normalized = outputs[1];
	ccv_nnc_tensor_t* const mix = outputs[2];
	const ccv_nnc_tensor_t* const state = inputs[0];
	const ccv_nnc_tensor_t* const message = inputs[1];
	const ccv_nnc_tensor_t* const down = inputs[2];
	const ccv_nnc_tensor_t* const forget = inputs[3];
	const ccv_nnc_tensor_t* const gain = inputs[4];
	const int datatype = state->info.datatype;
	if (!_ccv_nnc_message_gate_mtl_data_type(datatype) || message->info.datatype != datatype || y->info.datatype != datatype)
		return CCV_NNC_EXEC_INVALID;
	// Dense shapes only matter through their counts: C is the state's last axis, R the down
	// projection's first.
	const int channels = state->info.dim[ccv_nnc_tensor_nd(state->info.dim) - 1];
	const int rank = down->info.dim[0];
	int down_quantized;
	const int down_weight_type = _ccv_nnc_message_gate_weight_type(down, channels * 2, &down_quantized);
	// The forget and gain projections are both dense, or both 8-bit rowwise, in the same type.
	if (!down_weight_type || forget->info.datatype != gain->info.datatype)
		return CCV_NNC_EXEC_INVALID;
	int quantized;
	const int weight_type = _ccv_nnc_message_gate_weight_type(forget, rank, &quantized);
	if (!weight_type || !_ccv_nnc_message_gate_weight_type(gain, rank, &quantized))
		return CCV_NNC_EXEC_INVALID;
	const size_t state_size = ccv_nnc_tensor_count(state->info);
	if (channels < 4 || channels % 4 != 0 || rank < 4 || rank % 4 != 0 || rank > CCV_NNC_MFA_MESSAGE_GATE_MAX_RANK || state_size % channels != 0)
		return CCV_NNC_EXEC_INVALID;
	const size_t rows = state_size / channels;
	if (rows > UINT32_MAX || ccv_nnc_tensor_count(message->info) != state_size || ccv_nnc_tensor_count(y->info) != state_size ||
		ccv_nnc_tensor_count(down->info) != (size_t)rank * channels * 2 ||
		ccv_nnc_tensor_count(forget->info) != (size_t)channels * rank || ccv_nnc_tensor_count(gain->info) != (size_t)channels * rank ||
		normalized->info.datatype != CCV_32F || ccv_nnc_tensor_count(normalized->info) != state_size * 2 ||
		mix->info.datatype != CCV_32F || ccv_nnc_tensor_count(mix->info) != rows * rank)
		return CCV_NNC_EXEC_INVALID;
	// The output may be the state or the message in place, since the mix has read them before it is
	// written and each of its values reads only theirs at its position, but no other overlap.
	for (i = 0; i < 5; i++)
		if (_ccv_nnc_message_gate_overlaps(y, inputs[i]) && (i >= 2 || y->dataof != inputs[i]->dataof))
			return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			return CCV_NNC_EXEC_INVALID;
		// A quantized projection's row scales start at the first multiple of 128 bytes after its
		// signed bytes.
		const uint64_t down_scale_offset = down_quantized ? (((uint64_t)rank * channels * 2 + 127) & ~(uint64_t)127) : 0;
		const uint64_t scale_offset = quantized ? (((uint64_t)channels * rank + 127) & ~(uint64_t)127) : 0;
		ccv_nnc_mfa_message_gate_params_t params = {
			.data_type = (uint64_t)_ccv_nnc_message_gate_mtl_data_type(datatype),
			.down_data_type = (uint64_t)down_weight_type,
			.weight_data_type = (uint64_t)weight_type,
			.down_quantized = (uint8_t)down_quantized,
			.quantized = (uint8_t)quantized,
			.mix_given = 0,
			.rows = (uint32_t)rows,
			.channels = (uint32_t)channels,
			.rank = (uint32_t)rank,
			.epsilon = cmd.info.message_gate.epsilon,
			.down_scale_offset = down_scale_offset,
			.forget_scale_offset = scale_offset,
			.gain_scale_offset = scale_offset,
		};
		mtl_buffer_t* tensors[7] = {};
		size_t offsets[7] = {};
		for (i = 0; i < 5; i++)
		{
			tensors[i] = mpgetbuffer(inputs[i]);
			offsets[i] = inputs[i]->dataof;
		}
		tensors[5] = mpgetbuffer(y);
		offsets[5] = y->dataof;
		tensors[6] = mpgetbuffer(mix);
		offsets[6] = mix->dataof;
		if (rows > CCV_NNC_MFA_MESSAGE_GATE_FEW_ROWS)
		{
			// Many rows: the normalized rows, then the down projection by the GEMM command, as a Dense.
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* normalize_tensors[3] = { tensors[0], tensors[1], mpgetbuffer(normalized) };
			size_t normalize_offsets[3] = { offsets[0], offsets[1], normalized->dataof };
			ccv_nnc_mfa_encode_message_gate_normalize(context, params, command_batch, normalize_tensors, normalize_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			ccv_nnc_tensor_t normalized_rows = ccv_nnc_tensor(normalized->data.u8, GPU_TENSOR_NHWC(000, 32F, (int)rows, channels * 2), 0);
			normalized_rows.dataof = normalized->dataof;
			ccv_nnc_tensor_t mix_rows = ccv_nnc_tensor(mix->data.u8, GPU_TENSOR_NHWC(000, 32F, (int)rows, rank), 0);
			mix_rows.dataof = mix->dataof;
			// The down projection as [R, 2C] in NHWC, whatever its shape and format are.
			ccv_nnc_tensor_param_t down_params = down->info;
			down_params.format = CCV_TENSOR_FORMAT_NHWC;
			memset(down_params.dim, 0, sizeof(down_params.dim));
			down_params.dim[0] = rank;
			down_params.dim[1] = channels * 2;
			ccv_nnc_tensor_t down_rows = ccv_nnc_tensor(down->data.u8, down_params, 0);
			down_rows.dataof = down->dataof;
			if (ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(&normalized_rows, &down_rows), TENSOR_LIST(&mix_rows), stream_context) != CCV_NNC_EXEC_SUCCESS)
				return CCV_NNC_EXEC_INVALID;
			params.mix_given = 1;
		}
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		ccv_nnc_mfa_encode_message_gate(context, params, command_batch, tensors, offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MESSAGE_GATE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_message_gate_forw;
}
