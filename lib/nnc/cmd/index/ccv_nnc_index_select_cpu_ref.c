#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_index_select_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd <= 2);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == 1);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd <= 2);
	const int a_cols = a_nd < 2 ? 1 : a->info.dim[1];
	const int a_cols_inc = CCV_IS_TENSOR_VIEW(a) ? (a_nd < 2 ? 1 : a->stride[0]) : a_cols;
	const int a_rows = a->info.dim[0];
	const int b_cols = b_nd < 2 ? 1 : b->info.dim[1];
	const int b_cols_inc = CCV_IS_TENSOR_VIEW(b) ? (b_nd < 2 ? 1 : b->stride[0]) : b_cols;
	const int b_rows = b->info.dim[0];
	assert(b_rows == indices->info.dim[0]);
	assert(a_cols == b_cols);
	assert(indices->info.datatype == CCV_32S);
	assert(a->info.datatype == b->info.datatype);
	assert(a->info.datatype == CCV_32F || a->info.datatype == CCV_16F);
	const size_t data_size = CCV_GET_DATA_TYPE_SIZE(a->info.datatype);
	parallel_for(i, b_rows) {
		const int index = indices->data.i32[i];
		assert(index < a_rows);
		uint8_t* const bp = b->data.u8 + data_size * b_cols_inc * i;
		uint8_t* const ap = a->data.u8 + data_size * a_cols_inc * index;
		memcpy(bp, ap, data_size * a_cols);
	} parallel_endfor
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_index_select_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size <= 2);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
	assert(g_nd <= 2);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)inputs[2];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == 1);
	const ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
	assert(h_nd <= 2);
	ccv_nnc_tensor_zero((ccv_nnc_tensor_t*)h);
	if (output_size >= 2 && outputs[1])
		ccv_nnc_tensor_zero(outputs[1]);
	const int g_cols = g_nd < 2 ? 1 : g->info.dim[1];
	const int g_cols_inc = CCV_IS_TENSOR_VIEW(g) ? (g_nd < 2 ? 1 : g->stride[0]) : g_cols;
	const int g_rows = g->info.dim[0];
	const int h_cols = h_nd < 2 ? 1 : h->info.dim[1];
	const int h_cols_inc = CCV_IS_TENSOR_VIEW(h) ? (h_nd < 2 ? 1 : h->stride[0]) : h_cols;
	const int h_rows = h->info.dim[0];
	assert(g_rows == indices->info.dim[0]);
	assert(g_cols == h_cols);
	assert(indices->info.datatype == CCV_32S);
	assert(g->info.datatype == h->info.datatype);
	assert(g->info.datatype == CCV_32F || g->info.datatype == CCV_16F);
	int i;
	if (g->info.datatype == CCV_32F)
	{
		for (i = 0; i < g_rows; i++)
		{
			const int index = indices->data.i32[i];
			assert(index < h_rows);
			float* const hp = h->data.f32 + h_cols_inc * index;
			float* const gp = g->data.f32 + g_cols_inc * i;
			parallel_for(j, g_cols) {
				hp[j] += gp[j];
			} parallel_endfor
		}
	} else {
		for (i = 0; i < g_rows; i++)
		{
			const int index = indices->data.i32[i];
			assert(index < h_rows);
			ccv_float16_t* const hp = h->data.f16 + h_cols_inc * index;
			ccv_float16_t* const gp = g->data.f16 + g_cols_inc * i;
			parallel_for(j, g_cols) {
				float t, v;
				ccv_half_precision_to_float((uint16_t*)gp + j, &t, 1);
				ccv_half_precision_to_float((uint16_t*)hp + j, &v, 1);
				v += t;
				ccv_float_to_half_precision(&v, (uint16_t*)hp + j, 1);
			} parallel_endfor
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_back;
}
