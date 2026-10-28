#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "ccv_nnc_hyper_connection_internal.h"
#include <float.h>
#include <math.h>

static void _ccv_nnc_hyper_connection_split_row(const float* const mix, const float* const scale, const float* const base, float* const pre, float* const post, float* const comb, const int hc, const int iterations, const float epsilon)
{
	int i, j;
	for (i = 0; i < hc; i++)
		pre[i] = 1.0f / (1.0f + expf(-(mix[i] * scale[0] + base[i]))) + epsilon;
	for (i = 0; i < hc; i++)
		post[i] = 2.0f / (1.0f + expf(-(mix[hc + i] * scale[1] + base[hc + i])));
	for (i = 0; i < hc; i++)
	{
		float row_max = -FLT_MAX;
		for (j = 0; j < hc; j++)
		{
			const int k = i * hc + j;
			comb[k] = mix[2 * hc + k] * scale[2] + base[2 * hc + k];
			row_max = ccv_max(row_max, comb[k]);
		}
		float row_sum = 0;
		for (j = 0; j < hc; j++)
		{
			const int k = i * hc + j;
			comb[k] = expf(comb[k] - row_max);
			row_sum += comb[k];
		}
		for (j = 0; j < hc; j++)
			comb[i * hc + j] = comb[i * hc + j] / row_sum + epsilon;
	}
	for (j = 0; j < hc; j++)
	{
		float sum = 0;
		for (i = 0; i < hc; i++)
			sum += comb[i * hc + j];
		for (i = 0; i < hc; i++)
			comb[i * hc + j] /= sum + epsilon;
	}
	int iter;
	for (iter = 1; iter < iterations; iter++)
	{
		for (i = 0; i < hc; i++)
		{
			float sum = 0;
			for (j = 0; j < hc; j++)
				sum += comb[i * hc + j];
			for (j = 0; j < hc; j++)
				comb[i * hc + j] /= sum + epsilon;
		}
		for (j = 0; j < hc; j++)
		{
			float sum = 0;
			for (i = 0; i < hc; i++)
				sum += comb[i * hc + j];
			for (i = 0; i < hc; i++)
				comb[i * hc + j] /= sum + epsilon;
		}
	}
}

static int _ccv_nnc_hyper_connection_mix(const ccv_nnc_cmd_t cmd, ccv_nnc_tensor_t* const* const inputs, ccv_nnc_tensor_t* const* const outputs)
{
	const int hc = cmd.info.hyper_connection.count;
	const ccv_nnc_tensor_view_t* const residual = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const incoming_pre = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const projection = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const scale = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* const base = (const ccv_nnc_tensor_view_t*)inputs[4];
	const ccv_nnc_tensor_view_t* const norm_scale = (const ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const pre = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const post = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const comb = (ccv_nnc_tensor_view_t*)outputs[2];
	ccv_nnc_tensor_view_t* const collapsed = (ccv_nnc_tensor_view_t*)outputs[3];
	int i;
	for (i = 0; i < 6; i++)
		if (inputs[i]->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(inputs[i]))
			return CCV_NNC_EXEC_INVALID;
	for (i = 0; i < 4; i++)
		if (outputs[i]->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(outputs[i]))
			return CCV_NNC_EXEC_INVALID;
	const int iterations = cmd.info.hyper_connection.sinkhorn_iterations;
	const float epsilon = cmd.info.hyper_connection.epsilon;
	const float norm_epsilon = cmd.info.hyper_connection.norm_epsilon;
	if (iterations <= 0 || !(epsilon >= 0) || !(norm_epsilon > 0))
		return CCV_NNC_EXEC_INVALID;
	if (!_ccv_nnc_hyper_connection_mix_shapes_are_valid(hc, residual->info, incoming_pre->info, projection->info, scale->info, base->info, norm_scale->info, pre->info, post->info, comb->info, collapsed->info))
		return CCV_NNC_EXEC_INVALID;
	const int nd = ccv_nnc_tensor_nd(residual->info.dim);
	const int hidden = residual->info.dim[nd - 1];
	const int flat = hc * hidden;
	const int mix_dim = 2 * hc + hc * hc;
	const size_t rows = ccv_nnc_tensor_count(residual->info) / flat;
	size_t row;
	for (row = 0; row < rows; row++)
	{
		const float* const x = residual->data.f32 + row * flat;
		double sum_squares = 0;
		int k, j;
		for (j = 0; j < flat; j++)
			sum_squares += (double)x[j] * x[j];
		const float rstd = 1.0f / sqrtf((float)(sum_squares / flat) + norm_epsilon);
		float mix[16 * 16 + 32];
		for (k = 0; k < mix_dim; k++)
		{
			double dot = 0;
			for (j = 0; j < flat; j++)
				dot += (double)projection->data.f32[k * flat + j] * x[j];
			mix[k] = (float)dot * rstd;
		}
		_ccv_nnc_hyper_connection_split_row(mix, scale->data.f32, base->data.f32, pre->data.f32 + row * hc, post->data.f32 + row * hc, comb->data.f32 + row * hc * hc, hc, iterations, epsilon);
		float* const out = collapsed->data.f32 + row * hidden;
		double collapsed_squares = 0;
		for (j = 0; j < hidden; j++)
		{
			float sum = 0;
			for (k = 0; k < hc; k++)
				sum += x[k * hidden + j] * incoming_pre->data.f32[row * hc + k];
			out[j] = sum;
			collapsed_squares += (double)sum * sum;
		}
		const float collapsed_rstd = 1.0f / sqrtf((float)(collapsed_squares / hidden) + norm_epsilon);
		for (j = 0; j < hidden; j++)
			out[j] = out[j] * collapsed_rstd * norm_scale->data.f32[j];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_hyper_connection_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (!((input_size == 3 && output_size == 3) || (input_size == 4 && (output_size == 1 || output_size == 3)) || (input_size == 5 && output_size == 1) || (input_size == 6 && output_size == 4)))
		return CCV_NNC_EXEC_INVALID;
	if (outputs[0]->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const int hc = cmd.info.hyper_connection.count;
	if (hc <= 0 || hc > 16)
		return CCV_NNC_EXEC_INVALID;
	if (input_size == 6)
		return _ccv_nnc_hyper_connection_mix(cmd, inputs, outputs);
	if (output_size == 1)
	{
		const ccv_nnc_tensor_view_t* const block = (const ccv_nnc_tensor_view_t*)inputs[0];
		const ccv_nnc_tensor_view_t* const residual = (const ccv_nnc_tensor_view_t*)inputs[1];
		const ccv_nnc_tensor_view_t* const post = (const ccv_nnc_tensor_view_t*)inputs[2];
		const ccv_nnc_tensor_view_t* const comb = (const ccv_nnc_tensor_view_t*)inputs[3];
		const ccv_nnc_tensor_view_t* const addend = input_size == 5 ? (const ccv_nnc_tensor_view_t*)inputs[4] : 0;
		ccv_nnc_tensor_view_t* const expanded = (ccv_nnc_tensor_view_t*)outputs[0];
		if (block->info.datatype != CCV_32F || residual->info.datatype != CCV_32F || post->info.datatype != CCV_32F || comb->info.datatype != CCV_32F || expanded->info.datatype != CCV_32F)
			return CCV_NNC_EXEC_INVALID;
		if (addend && (!_ccv_nnc_hyper_connection_addend_shape_is_valid(block->info, addend->info) || !CCV_IS_TENSOR_CONTIGUOUS(addend)))
			return CCV_NNC_EXEC_INVALID;
		if (!CCV_IS_TENSOR_CONTIGUOUS(block) || !CCV_IS_TENSOR_CONTIGUOUS(residual) || !CCV_IS_TENSOR_CONTIGUOUS(post) || !CCV_IS_TENSOR_CONTIGUOUS(comb) || !CCV_IS_TENSOR_CONTIGUOUS(expanded))
			return CCV_NNC_EXEC_INVALID;
		if (!_ccv_nnc_hyper_connection_expand_shapes_are_valid(hc, block->info, residual->info, post->info, comb->info, expanded->info))
			return CCV_NNC_EXEC_INVALID;
		const int nd = ccv_nnc_tensor_nd(residual->info.dim);
		const int hidden = residual->info.dim[nd - 1];
		const size_t residual_count = ccv_nnc_tensor_count(residual->info);
		const size_t rows = residual_count / ((size_t)hc * hidden);
		size_t row;
		for (row = 0; row < rows; row++)
		{
			int i, d;
			for (i = 0; i < hc; i++)
				for (d = 0; d < hidden; d++)
				{
					const float sum = addend ? block->data.f32[row * hidden + d] + addend->data.f32[row * hidden + d] : block->data.f32[row * hidden + d];
					float value = sum * post->data.f32[row * hc + i];
					int j;
					for (j = 0; j < hc; j++)
						value += comb->data.f32[(row * hc + j) * hc + i] * residual->data.f32[(row * hc + j) * hidden + d];
					expanded->data.f32[(row * hc + i) * hidden + d] = value;
				}
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	const ccv_nnc_tensor_view_t* const mix = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const scale = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const base = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const pre = input_size == 3 ? (ccv_nnc_tensor_view_t*)outputs[0] : 0;
	ccv_nnc_tensor_view_t* const post = (ccv_nnc_tensor_view_t*)outputs[input_size == 3 ? 1 : 0];
	ccv_nnc_tensor_view_t* const comb = (ccv_nnc_tensor_view_t*)outputs[input_size == 3 ? 2 : 1];
	if (mix->info.datatype != CCV_32F || scale->info.datatype != CCV_32F || base->info.datatype != CCV_32F || (pre && pre->info.datatype != CCV_32F) || post->info.datatype != CCV_32F || comb->info.datatype != CCV_32F)
		return CCV_NNC_EXEC_INVALID;
	if (!CCV_IS_TENSOR_CONTIGUOUS(mix) || !CCV_IS_TENSOR_CONTIGUOUS(scale) || !CCV_IS_TENSOR_CONTIGUOUS(base) || (pre && !CCV_IS_TENSOR_CONTIGUOUS(pre)) || !CCV_IS_TENSOR_CONTIGUOUS(post) || !CCV_IS_TENSOR_CONTIGUOUS(comb))
		return CCV_NNC_EXEC_INVALID;
	const int iterations = cmd.info.hyper_connection.sinkhorn_iterations;
	const float epsilon = cmd.info.hyper_connection.epsilon;
	if (iterations <= 0 || !(epsilon >= 0))
		return CCV_NNC_EXEC_INVALID;
	const int mix_dim = 2 * hc + hc * hc;
	const size_t mix_count = ccv_nnc_tensor_count(mix->info);
	const size_t rows = mix_count / mix_dim;
	const ccv_nnc_tensor_view_t* const residual = input_size == 4 ? (const ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const weighted = input_size == 4 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	if (residual && (residual->info.datatype != CCV_32F || weighted->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(residual) || !CCV_IS_TENSOR_CONTIGUOUS(weighted)))
		return CCV_NNC_EXEC_INVALID;
	if (!_ccv_nnc_hyper_connection_split_shapes_are_valid(hc, mix->info, scale->info, base->info, residual ? &residual->info : 0, pre ? &pre->info : 0, post->info, comb->info, weighted ? &weighted->info : 0))
		return CCV_NNC_EXEC_INVALID;
	int hidden = 0;
	if (residual)
	{
		const int nd = ccv_nnc_tensor_nd(residual->info.dim);
		hidden = residual->info.dim[nd - 1];
	}
	size_t row;
	for (row = 0; row < rows; row++)
	{
		float local_pre[16];
		float* const row_pre = pre ? pre->data.f32 + row * hc : local_pre;
		_ccv_nnc_hyper_connection_split_row(mix->data.f32 + row * mix_dim, scale->data.f32, base->data.f32, row_pre, post->data.f32 + row * hc, comb->data.f32 + row * hc * hc, hc, iterations, epsilon);
		if (residual)
		{
			int d, i;
			for (d = 0; d < hidden; d++)
			{
				float sum = 0;
				for (i = 0; i < hc; i++)
					sum += residual->data.f32[(row * hc + i) * hidden + d] * row_pre[i];
				weighted->data.f32[row * hidden + d] = sum;
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_hyper_connection_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_hyper_connection_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_HYPER_CONNECTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_hyper_connection_back;
}
