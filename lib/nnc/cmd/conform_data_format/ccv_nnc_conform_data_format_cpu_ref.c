#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>

#define CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE (64)

static inline float _ccv_nnc_conform_data_format_e4m3_value(const int i)
{
	static const float exp_scale[16] = {
		0.0f, 0.015625f, 0.03125f, 0.0625f,
		0.125f, 0.25f, 0.5f, 1.0f,
		2.0f, 4.0f, 8.0f, 16.0f,
		32.0f, 64.0f, 128.0f, 256.0f,
	};
	const int exp = (i >> 3) & 0xf;
	const int mant = i & 0x7;
	return exp == 0 ? (float)mant * 0.001953125f : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static inline float _ccv_nnc_conform_data_format_e4m3_dequant(const float x)
{
	const float sign = x < 0 ? -1.0f : 1.0f;
	const float ax = fminf(fabsf(x), 448.0f);
	int lo = 0;
	int hi = 126;
	while (lo < hi)
	{
		const int mid = (lo + hi + 1) >> 1;
		if (_ccv_nnc_conform_data_format_e4m3_value(mid) <= ax)
			lo = mid;
		else
			hi = mid - 1;
	}
	int best = lo;
	if (best < 126)
	{
		const float best_diff = ax - _ccv_nnc_conform_data_format_e4m3_value(best);
		const float next_diff = _ccv_nnc_conform_data_format_e4m3_value(best + 1) - ax;
		if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0))
			++best;
	}
	return sign * _ccv_nnc_conform_data_format_e4m3_value(best);
}

static int _ccv_nnc_conform_data_format_e4m3_row(const float* const ap, float* const bp, const int head_dim, const int preserved_tail)
{
	const int prefix = head_dim - preserved_tail;
	int offset;
	for (offset = 0; offset < prefix; offset += CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE)
	{
		float amax = 1.0e-4f;
		int i;
		for (i = 0; i < CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE; i++)
		{
			const float value = fabsf(ap[offset + i]);
			if (!isfinite(value))
				return 0;
			amax = ccv_max(amax, value);
		}
		const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
		for (i = 0; i < CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE; i++)
		{
			const float normalized = ccv_clamp(ap[offset + i] / scale, -448.0f, 448.0f);
			bp[offset + i] = _ccv_nnc_conform_data_format_e4m3_dequant(normalized) * scale;
		}
	}
	if (preserved_tail > 0 && ap != bp)
		memcpy(bp + prefix, ap + prefix, sizeof(float) * preserved_tail);
	return 1;
}

static int _ccv_nnc_conform_data_format_validate(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_t* const a, const ccv_nnc_tensor_t* const b, int* const head_dim)
{
	if (cmd.info.conform_data_format.datatype != CCV_NNC_FP8_E4M3 || !a || !b)
		return 0;
	if (a->info.datatype != CCV_32F || b->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b))
		return 0;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	if (a_nd <= 0 || a_nd != b_nd)
		return 0;
	int i;
	for (i = 0; i < a_nd; i++)
		if (a->info.dim[i] != b->info.dim[i])
			return 0;
	*head_dim = a->info.dim[a_nd - 1];
	const int preserved_tail = cmd.info.conform_data_format.preserved_tail;
	if (*head_dim <= 0 || preserved_tail < 0 || preserved_tail > *head_dim || ((*head_dim - preserved_tail) % CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE) != 0)
		return 0;
	const size_t count = ccv_nnc_tensor_count(a->info);
	return count % *head_dim == 0;
}

static int _ccv_nnc_conform_data_format_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 1 || output_size != 1 || !inputs || !outputs)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_t* const a = inputs[0];
	ccv_nnc_tensor_t* const b = outputs[0];
	int head_dim;
	if (!_ccv_nnc_conform_data_format_validate(cmd, a, b, &head_dim))
		return CCV_NNC_EXEC_INVALID;
	const size_t rows = ccv_nnc_tensor_count(a->info) / head_dim;
	const float* const ap = a->data.f32;
	float* const bp = b->data.f32;
	const int preserved_tail = cmd.info.conform_data_format.preserved_tail;
	size_t i;
	for (i = 0; i < rows; i++)
		if (!_ccv_nnc_conform_data_format_e4m3_row(ap + i * head_dim, bp + i * head_dim, head_dim, preserved_tail))
			return CCV_NNC_EXEC_INVALID;
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conform_data_format_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size < 1 || output_size != 1 || !inputs || !outputs)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_t* const g = inputs[0];
	ccv_nnc_tensor_t* const h = outputs[0];
	int head_dim;
	if (!_ccv_nnc_conform_data_format_validate(cmd, g, h, &head_dim))
		return CCV_NNC_EXEC_INVALID;
	if (g->data.f32 != h->data.f32)
		memcpy(h->data.f32, g->data.f32, sizeof(float) * ccv_nnc_tensor_count(g->info));
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conform_data_format_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conform_data_format_back;
}
