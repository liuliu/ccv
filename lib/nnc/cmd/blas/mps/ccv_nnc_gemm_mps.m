#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_gemm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(!bias || (bias->info.dim[1] == 0 || bias->info.dim[2] == 0 || bias->info.dim[3] == 0)); // It is a 1-d array
	int a_batch_size, a_rows, a_cols, a_batch_inc, a_rows_inc, a_cols_inc;
	int w_batch_size, w_rows, w_cols, w_batch_inc, w_rows_inc, w_cols_inc;
	int b_batch_size, b_rows, b_cols, b_batch_inc, b_rows_inc, b_cols_inc;
	const static int no_transpose[2] = {};
	ccv_nnc_tensor_get_matrix_params(a->info, CCV_IS_TENSOR_VIEW(a) ? a->stride : 0, a->info.dim, cmd.info.blas.transpose_a, &a_batch_size, &a_rows, &a_cols, &a_batch_inc, &a_rows_inc, &a_cols_inc);
	ccv_nnc_tensor_get_matrix_params(w->info, CCV_IS_TENSOR_VIEW(w) ? w->stride : 0, w->info.dim, cmd.info.blas.transpose_b, &w_batch_size, &w_rows, &w_cols, &w_batch_inc, &w_rows_inc, &w_cols_inc);
	ccv_nnc_tensor_get_matrix_params(b->info, CCV_IS_TENSOR_VIEW(b) ? b->stride : 0, b->info.dim, no_transpose, &b_batch_size, &b_rows, &b_cols, &b_batch_inc, &b_rows_inc, &b_cols_inc);
	assert(a_batch_size == b_batch_size);
	assert(a_batch_size == b_batch_size || a_batch_size == 1);
	if (a_batch_size == 1 && b_batch_size > 1)
		a_batch_inc = 0;
	assert(w_batch_size == a_batch_size || w_batch_size == 1);
	if (w_batch_size == 1 && b_batch_size > 1)
		w_batch_inc = 0;
	assert(a_rows == b_rows);
	assert(a_cols == w_rows);
	assert(w_cols == b_cols);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	memcpy(adim, a->info.dim, sizeof(adim));
	if (CCV_IS_TENSOR_VIEW(a))
		memcpy(astride, a->stride, sizeof(astride));
	if (ccv_nnc_tensor_nd(adim) < 2)
	{
		adim[1] = 1;
		astride[1] = 1;
	}
	assert(ccv_nnc_tensor_nd(w->info.dim) >= 2);
	const int is_transpose_a = ccv_nnc_is_matrix_transpose(a->info, cmd.info.blas.transpose_a);
	const int is_transpose_w = ccv_nnc_is_matrix_transpose(w->info, cmd.info.blas.transpose_b);
	@autoreleasepool {
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
		MPSGraphTensor* mps_input_w;
		MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, w->info.dim, w->stride, &mps_input_w);
		if (is_transpose_a)
			mps_a = [graph transposeTensor:mps_a dimension:-2 withDimension:-1 name:nil];
		if (is_transpose_w)
			mps_w = [graph transposeTensor:mps_w dimension:-2 withDimension:-1 name:nil];
		MPSGraphTensor* mps_b = [graph matrixMultiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_w name:nil];
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
		MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data(w, w->info.dim, w->stride);
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		if (bias)
		{
			MPSGraphTensor* mps_input_bias;
			MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, bias, bias->info.dim, bias->stride, &mps_input_bias);
			// Add support broadcast directly.
			mps_b = [graph additionWithPrimaryTensor:mps_b secondaryTensor:mps_bias name:nil];
			MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, bias->info.dim, bias->stride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a, mps_input_w: data_w, mps_input_bias: data_bias}, mps_b, b, b->info.dim, b->stride);
		} else
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a, mps_input_w: data_w}, mps_b, b, b->info.dim, b->stride);
		[graph release];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gemm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gemm_back;
}
