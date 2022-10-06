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

static int _ccv_nnc_upsample_nearest_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	assert(a->info.format == b->info.format);
	assert(a->info.datatype == b->info.datatype);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSGraph *graph = [MPSGraph new];
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
			MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim[CCV_NNC_MAX_DIM]), @(bdim[CCV_NNC_MAX_DIM + 1])] mode:MPSGraphResizeNearest centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_b, b, bdim, bstride);
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		@autoreleasepool {
			MPSGraph *graph = [MPSGraph new];
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
			MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim[CCV_NNC_MAX_DIM - 1]), @(bdim[CCV_NNC_MAX_DIM])] mode:MPSGraphResizeNearest centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_b, b, bdim, bstride);
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_bilinear_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(b->info.dim) <= CCV_NNC_MAX_DIM + 2);
	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	assert(CCV_NNC_MAX_DIM == 2); // Need to change this logic for CCV_NNC_MAX_DIM == other number.
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	assert(a->info.format == b->info.format);
	assert(a->info.datatype == b->info.datatype);
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSGraph *graph = [MPSGraph new];
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
			MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim[CCV_NNC_MAX_DIM]), @(bdim[CCV_NNC_MAX_DIM + 1])] mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_b, b, bdim, bstride);
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		@autoreleasepool {
			MPSGraph *graph = [MPSGraph new];
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim, astride, &mps_input_a);
			MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim[CCV_NNC_MAX_DIM - 1]), @(bdim[CCV_NNC_MAX_DIM])] mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_b, b, bdim, bstride);
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_NEAREST)
		return _ccv_nnc_upsample_nearest_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	else if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_BILINEAR)
		return _ccv_nnc_upsample_bilinear_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_forw;
}
