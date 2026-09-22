#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <math.h>
#include "../_ccv_nnc_cpu_ref.h"

// Independent, deliberately slow oracle. Mean V + log(block length) is
// algebraically equivalent to summed V + denominator multiplicity.
// FP32 inputs use double-precision pooling and accumulation to independently
// compute routing and attention.
static int _ccv_nnc_sol_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert((input_size == 3 || input_size == 4) && output_size == 1);
	int is_dense_attention = 0;
	if (input_size == 4)
	{
		if (CCV_TENSOR_GET_MEMORY(inputs[3]->info.type) != CCV_TENSOR_CPU_MEMORY || inputs[3]->info.datatype != CCV_32S || ccv_nnc_tensor_count(inputs[3]->info) != 1)
			return CCV_NNC_EXEC_INVALID;
		is_dense_attention = inputs[3]->data.i32[0] != 0;
	}
	const int N = inputs[0]->info.dim[0], T = inputs[0]->info.dim[1], H = inputs[0]->info.dim[2], D = inputs[0]->info.dim[3];
	const int B = cmd.info.sol_attention.block_size;
	const int QB = cmd.info.sol_attention.query_block_size > 0 ? cmd.info.sol_attention.query_block_size : B;
	const int start = cmd.info.sol_attention.approximation_start, end = cmd.info.sol_attention.approximation_end;
	if (N <= 0 || T <= 0 || H <= 0 || D <= 0 || B <= 0 || cmd.info.sol_attention.query_block_size < 0 || cmd.info.sol_attention.local_block_radius < 1 || start < 0 || end < start || end > T || !isfinite(cmd.info.sol_attention.scale) || !isfinite(cmd.info.sol_attention.tau))
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < 4; i++)
	{
		const ccv_nnc_tensor_t* const tensor = i < 3 ? inputs[i] : outputs[0];
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_CPU_MEMORY || tensor->info.format != CCV_TENSOR_FORMAT_NHWC || tensor->info.datatype != CCV_32F || ccv_nnc_tensor_nd(tensor->info.dim) != 4 || !CCV_IS_TENSOR_CONTIGUOUS(tensor) || tensor->info.dim[0] != N || tensor->info.dim[1] != T || tensor->info.dim[2] != H || tensor->info.dim[3] != D)
			return CCV_NNC_EXEC_INVALID;
	}
	if (is_dense_attention)
	{
		ccv_nnc_cmd_t dense = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(cmd.info.sol_attention.scale, 0);
		dense.info.scaled_dot_product_attention.flags = cmd.info.sol_attention.flags;
		return _ccv_nnc_scaled_dot_product_attention_forw_cpu_ref(dense, hint, flags, inputs, 3, outputs, output_size, stream_context);
	}
	const float* const q = inputs[0]->data.f32;
	const float* const k = inputs[1]->data.f32;
	const float* const v = inputs[2]->data.f32;
	float* const output = outputs[0]->data.f32;
	const int J = (T + B - 1) / B;
	const int QJ = (T + QB - 1) / QB;
	double* const scratch = (double*)cccalloc((size_t)(QJ + 2 * J + 3) * D + J, sizeof(double));
	double* const qm = scratch;
	double* const km = qm + QJ * D;
	double* const vm = km + J * D;
	double* const mean = vm + J * D;
	double* const variance = mean + D;
	double* const acc = variance + D;
	double* const routes = acc + D;
	int n, h, j, d, t, r;
	const double scale = cmd.info.sol_attention.scale;
	for (n = 0; n < N; n++)
		for (h = 0; h < H; h++)
		{
			memset(scratch, 0, (size_t)(QJ + 2 * J + 3) * D * sizeof(double));
			for (j = 0; j < QJ; j++)
			{
				const int count = ccv_min(QB, T - j * QB);
				for (t = j * QB; t < j * QB + count; t++)
					for (d = 0; d < D; d++)
						qm[j * D + d] += q[((size_t)n * T * H + t * H + h) * D + d] / (double)count;
			}
			for (j = 0; j < J; j++)
			{
				const int count = ccv_min(B, T - j * B);
				for (t = j * B; t < j * B + count; t++)
					for (d = 0; d < D; d++)
					{
						const size_t offset = ((size_t)n * T * H + t * H + h) * D + d;
						km[j * D + d] += k[offset] / (double)count;
						vm[j * D + d] += v[offset] / (double)count;
					}
				for (d = 0; d < D; d++)
				{
					mean[d] += km[j * D + d] / J;
					variance[d] += km[j * D + d] * km[j * D + d] / J;
				}
			}
			for (d = 0; d < D; d++)
				variance[d] = ccv_max(0, variance[d] - mean[d] * mean[d]);
			for (r = 0; r < T; r++)
			{
				const int qb = r / QB;
				double mu = 0, var = 0;
				for (d = 0; d < D; d++)
				{
					mu += qm[qb * D + d] * mean[d];
					var += qm[qb * D + d] * qm[qb * D + d] * variance[d];
				}
				// Upstream's epsilon is in base-2 logit units.
				const double g = scale * 1.4426950408889634;
				const double threshold = g * mu + cmd.info.sol_attention.tau * sqrt(g * g * var + 1e-6);
				for (j = 0; j < J; j++)
				{
					double score = 0;
					for (d = 0; d < D; d++)
						score += qm[qb * D + d] * km[j * D + d];
					routes[j] = qb * QB < start || ccv_min((qb + 1) * QB, T) > end || j * B < start || ccv_min((j + 1) * B, T) > end || abs(qb * QB / B - j) <= cmd.info.sol_attention.local_block_radius || score * g > threshold;
				}
				double maximum = -INFINITY, denominator = 0;
				memset(acc, 0, D * sizeof(double));
				for (j = 0; j < J; j++)
				{
					const int count = ccv_min(B, T - j * B);
					for (t = 0; t < (routes[j] ? count : 1); t++)
					{
						double score = 0;
						for (d = 0; d < D; d++)
							score += q[((size_t)n * T * H + r * H + h) * D + d] * (routes[j] ? k[((size_t)n * T * H + (j * B + t) * H + h) * D + d] : km[j * D + d]);
						score = score * scale + (routes[j] ? 0 : log(count));
						const double next_maximum = ccv_max(maximum, score), correction = exp(maximum - next_maximum), weight = exp(score - next_maximum);
						denominator = denominator * correction + weight;
						for (d = 0; d < D; d++)
							acc[d] = acc[d] * correction + weight * (routes[j] ? v[((size_t)n * T * H + (j * B + t) * H + h) * D + d] : vm[j * D + d]);
						maximum = next_maximum;
					}
				}
				for (d = 0; d < D; d++)
					output[((size_t)n * T * H + r * H + h) * D + d] = acc[d] / denominator;
			}
		}
	ccfree(scratch);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sol_attention_forw;
}
