#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

// A projection [N, K] is dense in 32F or 16F, or 8-bit rowwise: N * K signed bytes, then a scale for
// each row in the base type, from the first multiple of 128 bytes after them. Its scales are for rows
// of its last axis, which is then K.
static int _ccv_nnc_message_gate_weight_is_supported(const ccv_nnc_tensor_t* const weight, const int k)
{
	const int datatype = weight->info.datatype;
	if (CCV_GET_DATA_TYPE(datatype) == CCV_QX)
	{
		const int nd = ccv_nnc_tensor_nd(weight->info.dim);
		if ((datatype & 0xf00) != CCV_NNC_QX_8I_ROWWISE || nd < 1 || weight->info.dim[nd - 1] != k)
			return 0;
		const int base = (datatype & 0xff) << 12;
		return base == CCV_32F || base == CCV_16F;
	}
	return datatype == CCV_32F || datatype == CCV_16F;
}

static float _ccv_nnc_message_gate_scalar(const void* const data, const int datatype, const size_t i)
{
	if (datatype == CCV_16F)
	{
		float value;
		ccv_half_precision_to_float((uint16_t*)data + i, &value, 1);
		return value;
	}
	return ((const float*)data)[i];
}

// Row n of a projection whose rows are k wide, decoded into row.
static void _ccv_nnc_message_gate_weight_row(const ccv_nnc_tensor_t* const weight, const int n, const int k, float* const row)
{
	const int datatype = weight->info.datatype;
	int i;
	if (CCV_GET_DATA_TYPE(datatype) == CCV_QX)
	{
		const size_t count = ccv_nnc_tensor_count(weight->info);
		const int8_t* const q = (const int8_t*)weight->data.u8 + (size_t)n * k;
		const float scale = _ccv_nnc_message_gate_scalar(weight->data.u8 + ((count + 127) & ~(size_t)127), (datatype & 0xff) << 12, n);
		for (i = 0; i < k; i++)
			row[i] = (float)q[i] * scale;
	} else {
		for (i = 0; i < k; i++)
			row[i] = _ccv_nnc_message_gate_scalar(weight->data.u8, datatype, (size_t)n * k + i);
	}
}

// The state with a message gated in: mix = down [RMSNorm(state), RMSNorm(message)]^T, forget =
// tanh(mix forget^T), gain = tanh(mix gain^T), and y = state - forget * state + gain * message.
static int _ccv_nnc_message_gate_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 5);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* const state = inputs[0];
	const ccv_nnc_tensor_t* const message = inputs[1];
	const ccv_nnc_tensor_t* const down = inputs[2];
	const ccv_nnc_tensor_t* const forget = inputs[3];
	const ccv_nnc_tensor_t* const gain = inputs[4];
	ccv_nnc_tensor_t* const y = outputs[0];
	int i;
	// All are read as packed.
	for (i = 0; i < 5; i++)
		if (!CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	// The state, the message and the output are 32F; 16F is registered for the projections.
	if (state->info.datatype != CCV_32F || message->info.datatype != CCV_32F || y->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(y))
		return CCV_NNC_EXEC_INVALID;
	// Dense shapes only matter through their counts: C is the state's last axis, R the down
	// projection's first.
	const int channels = state->info.dim[ccv_nnc_tensor_nd(state->info.dim) - 1];
	const int rank = down->info.dim[0];
	if (!_ccv_nnc_message_gate_weight_is_supported(down, channels * 2) || !_ccv_nnc_message_gate_weight_is_supported(forget, rank) || !_ccv_nnc_message_gate_weight_is_supported(gain, rank))
		return CCV_NNC_EXEC_INVALID;
	const size_t state_size = ccv_nnc_tensor_count(state->info);
	const int rows = state_size / channels;
	assert(ccv_nnc_tensor_count(message->info) == state_size);
	assert(ccv_nnc_tensor_count(y->info) == state_size);
	assert(ccv_nnc_tensor_count(down->info) == (size_t)rank * channels * 2);
	assert(ccv_nnc_tensor_count(forget->info) == (size_t)channels * rank);
	assert(ccv_nnc_tensor_count(gain->info) == (size_t)channels * rank);
	const float epsilon = cmd.info.message_gate.epsilon;
	// The mixes of all rows first, as the output may be the state or the message in place.
	float* const mix = (float*)ccmalloc(sizeof(float) * rows * rank);
	parallel_for(t, rows) {
		const float* const s = state->data.f32 + (size_t)t * channels;
		const float* const m = message->data.f32 + (size_t)t * channels;
		double ss = 0, mm = 0;
		int c, r;
		for (c = 0; c < channels; c++)
			ss += (double)s[c] * s[c], mm += (double)m[c] * m[c];
		const float inv_s = 1 / sqrtf((float)(ss / channels) + epsilon);
		const float inv_m = 1 / sqrtf((float)(mm / channels) + epsilon);
		float* const row = (float*)ccmalloc(sizeof(float) * channels * 2);
		for (r = 0; r < rank; r++)
		{
			_ccv_nnc_message_gate_weight_row(down, r, channels * 2, row);
			float ds = 0, dm = 0;
			for (c = 0; c < channels; c++)
				ds += row[c] * s[c], dm += row[channels + c] * m[c];
			mix[(size_t)t * rank + r] = ds * inv_s + dm * inv_m;
		}
		ccfree(row);
	} parallel_endfor
	parallel_for(c, channels) {
		float* const weights = (float*)ccmalloc(sizeof(float) * rank * 2);
		_ccv_nnc_message_gate_weight_row(forget, c, rank, weights);
		_ccv_nnc_message_gate_weight_row(gain, c, rank, weights + rank);
		int t, r;
		for (t = 0; t < rows; t++)
		{
			const float* const m = mix + (size_t)t * rank;
			float f = 0, g = 0;
			for (r = 0; r < rank; r++)
			{
				f += m[r] * weights[r];
				g += m[r] * weights[rank + r];
			}
			const size_t index = (size_t)t * channels + c;
			const float x = state->data.f32[index];
			y->data.f32[index] = x - tanhf(f) * x + tanhf(g) * message->data.f32[index];
		}
		ccfree(weights);
	} parallel_endfor
	ccfree(mix);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_message_gate_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MESSAGE_GATE_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_message_gate_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MESSAGE_GATE_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_message_gate_back;
}
