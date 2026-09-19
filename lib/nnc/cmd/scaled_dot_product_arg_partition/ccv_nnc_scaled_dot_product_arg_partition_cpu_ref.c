#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <float.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static inline int _ccv_nnc_sdpap_visible_count(const int C, const int t, const int is_causal, const int compression_ratio, const int query_offset)
{
	if (!is_causal)
		return C;
	int visible = (query_offset + t + 1) / compression_ratio;
	if (visible < 0)
		visible = 0;
	else if (visible > C)
		visible = C;
	return visible;
}

static inline int _ccv_nnc_sdpap_better(const float a_score, const int a_idx, const float b_score, const int b_idx)
{
	return a_score > b_score || (a_score == b_score && (b_idx < 0 || a_idx < b_idx));
}

static void _ccv_nnc_sdpap_insert_topk(const float score, const int idx, const int kth, float* const top_scores, int* const top_indices, int* const top_count)
{
	if (*top_count == kth && !_ccv_nnc_sdpap_better(score, idx, top_scores[kth - 1], top_indices[kth - 1]))
		return;
	int pos = (*top_count < kth) ? (*top_count)++ : kth - 1;
	while (pos > 0 && _ccv_nnc_sdpap_better(score, idx, top_scores[pos - 1], top_indices[pos - 1]))
	{
		top_scores[pos] = top_scores[pos - 1];
		top_indices[pos] = top_indices[pos - 1];
		--pos;
	}
	top_scores[pos] = score;
	top_indices[pos] = idx;
}

#define less_than(a, b, aux) ((a) < (b))
static CCV_IMPLEMENT_QSORT(_ccv_nnc_sdpap_sort_ids, int, less_than)
#undef less_than

static int _ccv_nnc_scaled_dot_product_arg_partition_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3 || input_size == 4);
	assert(output_size == 1 || output_size == 2);
	const ccv_nnc_tensor_view_t* const q = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const k = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const head_w = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const selected = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(head_w));
	assert(CCV_IS_TENSOR_CONTIGUOUS(selected));
	assert(q->info.datatype == CCV_32F);
	assert(k->info.datatype == CCV_32F);
	assert(head_w->info.datatype == CCV_32F);
	assert(selected->info.datatype == CCV_32S);
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	const int head_w_nd = ccv_nnc_tensor_nd(head_w->info.dim);
	const int selected_nd = ccv_nnc_tensor_nd(selected->info.dim);
	assert(q_nd == 3);
	assert(head_w_nd == 2);
	assert(selected_nd == 2);
	const int T = q->info.dim[0];
	const int H = q->info.dim[1];
	const int D = q->info.dim[2];
	const int C = k->info.dim[0];
	const int kth = cmd.info.scaled_dot_product_arg_partition.kth;
	const float scale = cmd.info.scaled_dot_product_arg_partition.scale;
	const int is_causal = cmd.info.scaled_dot_product_arg_partition.is_causal;
	const int compression_ratio = cmd.info.scaled_dot_product_arg_partition.compression_ratio;
	const int query_offset = cmd.info.scaled_dot_product_arg_partition.query_offset;
	assert(C == 0 || k_nd == 2);
	assert(C == 0 || k->info.dim[1] == D);
	assert(head_w->info.dim[0] == T);
	assert(head_w->info.dim[1] == H);
	assert(selected->info.dim[0] == T);
	assert(selected->info.dim[1] == kth);
	assert(kth > 0);
	assert(compression_ratio > 0);
	const int block_size = cmd.info.scaled_dot_product_arg_partition.candidate_block_size;
	const int pool_size = cmd.info.scaled_dot_product_arg_partition.candidate_kth;
	const ccv_nnc_tensor_view_t* const candidates = input_size == 4 ? (const ccv_nnc_tensor_view_t*)inputs[3] : 0;
	ccv_nnc_tensor_view_t* const pool = output_size == 2 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	if (candidates || pool)
	{
		assert(block_size > 0 && pool_size > 0);
		assert(!candidates || !pool);
		const ccv_nnc_tensor_view_t* const ids = candidates ? candidates : pool;
		assert(CCV_IS_TENSOR_CONTIGUOUS(ids));
		assert(ids->info.datatype == CCV_32S);
		assert(ccv_nnc_tensor_nd(ids->info.dim) == 2 && ids->info.dim[0] == T && ids->info.dim[1] == pool_size);
	}
	const int pool_work = candidates || pool ? pool_size : 0;
	float* const top_scores = (float*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(float) * (kth + pool_work) + sizeof(int) * (kth + pool_work * 2), CCV_TENSOR_CPU_MEMORY);
	float* const pool_scores = top_scores + kth;
	int* const top_indices = (int*)(pool_scores + pool_work);
	int* const pool_indices = top_indices + kth;
	int* const input_ids = pool_indices + pool_work;
	int t, h, d, c;
	for (t = 0; t < T; t++)
	{
		int* const selected_t = selected->data.i32 + t * kth;
		for (d = 0; d < kth; d++)
			selected_t[d] = -1;
		if (pool)
			for (d = 0; d < pool_size; d++)
				pool->data.i32[t * pool_size + d] = -1;
		const int visible = _ccv_nnc_sdpap_visible_count(C, t, is_causal, compression_ratio, query_offset);
		int top_count = 0, pool_count = 0;
		if (visible <= 0)
			continue;
		if (candidates)
		{
			memcpy(input_ids, candidates->data.i32 + t * pool_size, sizeof(int) * pool_size);
			_ccv_nnc_sdpap_sort_ids(input_ids, pool_size, 0);
		}
		const int size = candidates || pool ? block_size : visible;
		const int blocks = candidates ? pool_size : (visible + size - 1) / size;
		int b;
		for (b = 0; b < blocks; b++)
		{
			const int block = candidates ? input_ids[b] : b;
			if (block < 0 || block > (visible - 1) / size || (candidates && b > 0 && block == input_ids[b - 1]))
				continue;
			float block_score = -FLT_MAX;
			const int end = ccv_min((int64_t)(block + 1) * size, visible);
			for (c = block * size; c < end; c++)
			{
				float score = 0;
				// Full-width enumeration needs scores only when publishing block maxima.
				if (C > kth || pool)
					for (h = 0; h < H; h++)
					{
						const float* const q_ptr = q->data.f32 + (t * H + h) * D;
						const float* const k_ptr = k->data.f32 + c * D;
						float dot = 0;
						for (d = 0; d < D; d++)
							dot += q_ptr[d] * k_ptr[d];
						if (dot > 0)
							score += dot * head_w->data.f32[t * H + h] * scale;
					}
				block_score = ccv_max(block_score, score);
				if (C <= kth)
					selected_t[top_count++] = c;
				else
					_ccv_nnc_sdpap_insert_topk(score, c, kth, top_scores, top_indices, &top_count);
			}
			if (pool)
				_ccv_nnc_sdpap_insert_topk(block == (visible - 1) / size ? INFINITY : block_score, block, pool_size, pool_scores, pool_indices, &pool_count);
		}
		if (C > kth)
			memcpy(selected_t, top_indices, sizeof(int) * top_count);
		if (C > kth && cmd.info.scaled_dot_product_arg_partition.sort_indices)
			_ccv_nnc_sdpap_sort_ids(selected_t, top_count, 0);
		if (pool)
		{
			_ccv_nnc_sdpap_sort_ids(pool_indices, pool_count, 0);
			memcpy(pool->data.i32 + t * pool_size, pool_indices, sizeof(int) * pool_count);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_arg_partition_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_arg_partition_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_arg_partition_back;
}
