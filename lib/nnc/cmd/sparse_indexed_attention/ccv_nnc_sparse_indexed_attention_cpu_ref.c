#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <float.h>
#include <math.h>
#include <string.h>

static inline void _ccv_nnc_sparse_indexed_attention_attend_row(const float* const q, const float* const k, const float* const v, const int D, const float scale, double* const maxval, double* const sumval, double* const acc)
{
	int d;
	double score = 0;
	for (d = 0; d < D; d++)
		score += (double)q[d] * (double)k[d];
	score *= (double)scale;
	const double old_maxval = *maxval;
	const double new_maxval = ccv_max(old_maxval, score);
	const double old_scale = exp(old_maxval - new_maxval);
	const double row_scale = exp(score - new_maxval);
	for (d = 0; d < D; d++)
		acc[d] = acc[d] * old_scale + (double)v[d] * row_scale;
	*sumval = *sumval * old_scale + row_scale;
	*maxval = new_maxval;
}

static inline void _ccv_nnc_sparse_indexed_attention_attend_sink(const float sink, const int D, double* const maxval, double* const sumval, double* const acc)
{
	const double score = (double)sink;
	const double old_maxval = *maxval;
	const double new_maxval = ccv_max(old_maxval, score);
	const double old_scale = exp(old_maxval - new_maxval);
	int d;
	for (d = 0; d < D; d++)
		acc[d] *= old_scale;
	*sumval = *sumval * old_scale + exp(score - new_maxval);
	*maxval = new_maxval;
}

static int _ccv_nnc_sparse_indexed_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6 || input_size == 7);
	assert(output_size == 1);
	const int attention_sinks = cmd.info.sparse_indexed_attention.attention_sinks;
	const int is_causal = cmd.info.sparse_indexed_attention.is_causal;
	const int sliding_window = cmd.info.sparse_indexed_attention.sliding_window;
	if ((attention_sinks != 0) != (input_size == 7))
		return CCV_NNC_EXEC_INVALID;
	if (sliding_window < 0 || (sliding_window > 0 && !is_causal))
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
	assert(q->info.datatype == CCV_32F);
	assert(dense_k->info.datatype == CCV_32F);
	assert(dense_v->info.datatype == CCV_32F);
	assert(sparse_k->info.datatype == CCV_32F);
	assert(sparse_v->info.datatype == CCV_32F);
	assert(indices->info.datatype == CCV_32S);
	assert(out->info.datatype == CCV_32F);
	if (sinks)
		assert(sinks->info.datatype == CCV_32F);
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
	uint32_t sink_head_stride = 0;
	if (sinks)
	{
		const int sink_count = ccv_nnc_tensor_count(sinks->info);
		assert(sink_count == 1 || sink_count == H);
		sink_head_stride = (sink_count == 1) ? 0 : 1;
	}
	double* const acc = (double*)ccv_nnc_stream_context_get_workspace(stream_context, sizeof(double) * D, CCV_TENSOR_CPU_MEMORY);
	const float scale = cmd.info.sparse_indexed_attention.scale;
	int t, h, d, r, i;
	for (t = 0; t < T; t++)
	{
		int dense_start = 0;
		int dense_end = dense_rows;
		if (is_causal)
		{
			dense_end = dense_rows - T + t + 1;
			if (dense_end < 0)
				dense_end = 0;
			else if (dense_end > dense_rows)
				dense_end = dense_rows;
			if (sliding_window > 0 && dense_end > sliding_window)
				dense_start = dense_end - sliding_window;
		}
		for (h = 0; h < H; h++)
		{
			memset(acc, 0, sizeof(double) * D);
			double maxval = -DBL_MAX;
			double sumval = 0;
			const float* const q_ptr = q->data.f32 + (t * H + h) * D;
			for (r = dense_start; r < dense_end; r++)
				_ccv_nnc_sparse_indexed_attention_attend_row(q_ptr, dense_k->data.f32 + r * D, dense_v->data.f32 + r * D, D, scale, &maxval, &sumval, acc);
			const int* const index_ptr = indices->data.i32 + t * K;
			for (i = 0; i < K; i++)
			{
				const int idx = index_ptr[i];
				if (idx < 0)
					break;
				if (idx >= sparse_rows)
					return CCV_NNC_EXEC_INVALID;
				_ccv_nnc_sparse_indexed_attention_attend_row(q_ptr, sparse_k->data.f32 + idx * D, sparse_v->data.f32 + idx * D, D, scale, &maxval, &sumval, acc);
			}
			if (sinks)
				_ccv_nnc_sparse_indexed_attention_attend_sink(sinks->data.f32[h * sink_head_stride], D, &maxval, &sumval, acc);
			float* const out_ptr = out->data.f32 + (t * H + h) * D;
			if (sumval == 0)
				memset(out_ptr, 0, sizeof(float) * D);
			else
				for (d = 0; d < D; d++)
					out_ptr[d] = (float)(acc[d] / sumval);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sparse_indexed_attention_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sparse_indexed_attention_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SPARSE_INDEXED_ATTENTION_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sparse_indexed_attention_back;
}
