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
	int* adim_r = adim;
	int* astride_r = astride;
	int* bdim_r = bdim;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim_r[CCV_NNC_MAX_DIM]), @(bdim_r[CCV_NNC_MAX_DIM + 1])] mode:MPSGraphResizeNearest centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim_r[CCV_NNC_MAX_DIM - 1]), @(bdim_r[CCV_NNC_MAX_DIM])] mode:MPSGraphResizeNearest centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
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
	int* adim_r = adim;
	int* astride_r = astride;
	int* bdim_r = bdim;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim_r[CCV_NNC_MAX_DIM]), @(bdim_r[CCV_NNC_MAX_DIM + 1])] mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph resizeTensor:mps_a size:@[@(bdim_r[CCV_NNC_MAX_DIM - 1]), @(bdim_r[CCV_NNC_MAX_DIM])] mode:MPSGraphResizeBilinear centerResult:YES alignCorners:NO layout:MPSGraphTensorNamedDataLayoutNHWC name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
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

static int _ccv_nnc_upsample_bilinear_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
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
	int* adim_r = adim;
	int* astride_r = astride;
	int* bdim_r = bdim;
	int* bstride_r = bstride;
	NSMutableArray<NSNumber*>* inputSize = [NSMutableArray new];	
	for (int i = 0; i < CCV_NNC_MAX_DIM + 2; i++) {
		[inputSize addObject:@(adim_r[i])];
	}
	
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, bdim_r, bstride_r, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, bdim_r, bstride_r);
				[inputShapedTypes addObject:mps_b_shape];
				
				MPSGraphTensor* inputSizeTensor = [graph constantWithScalar:0 shape:inputSize dataType:ccv_nnc_mps_datatype(b->info.datatype)];

				MPSGraphTensor* mps_a = [graph resizeWithGradientTensor:mps_b 
                                       input:inputSizeTensor 
                                        mode:MPSGraphResizeBilinear 
                                centerResult:YES
                                alignCorners:NO 
                                      layout:MPSGraphTensorNamedDataLayoutNCHW
                                        name:nil];

				[resultTensors addObject:mps_a];
			});
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, bdim_r, bstride_r);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_b], &a, (int*[]){ adim_r }, (int*[]){ astride_r }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		assert(inputSize.count == 4);
		// for unknown reason, MPS handling NHWC as NHCW... 
		// explicitly transpose input and output for NHWC
		[inputSize exchangeObjectAtIndex:2 withObjectAtIndex:3];
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, bdim_r, bstride_r, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, bdim_r, bstride_r);
				[inputShapedTypes addObject:mps_b_shape];
				// NHWC to NHCW
				mps_b = [graph transposeTensor:mps_b dimension:-1 withDimension:-2 name:nil];
				MPSGraphTensor* inputSizeTensor = [graph constantWithScalar:0 shape:inputSize dataType:ccv_nnc_mps_datatype(b->info.datatype)];

				MPSGraphTensor* mps_a = [graph resizeWithGradientTensor:mps_b 
                                       input:inputSizeTensor 
                                        mode:MPSGraphResizeBilinear 
                                centerResult:YES
                                alignCorners:NO 
                                      layout:MPSGraphTensorNamedDataLayoutNHWC
                                        name:nil];
				// NHCW to NHWC
				mps_a = [graph transposeTensor:mps_a dimension:-1 withDimension:-2 name:nil];
				[resultTensors addObject:mps_a];
			});
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, bdim_r, bstride_r);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_b], &a , (int*[]){ adim_r }, (int*[]){ astride_r }, 1);

			
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
			
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_nearest_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
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
	int* adim_r = adim;
	int* astride_r = astride;
	int* bdim_r = bdim;
	int* bstride_r = bstride;
	NSMutableArray<NSNumber*>* inputSize = [NSMutableArray new];	

	for (int i = 0; i < CCV_NNC_MAX_DIM + 2; i++) {
		[inputSize addObject:@(adim_r[i])];
	}

	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, bdim_r, bstride_r, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, bdim_r, bstride_r);
				[inputShapedTypes addObject:mps_b_shape];
				
				MPSGraphTensor* inputSizeTensor = [graph constantWithScalar:0 shape:inputSize dataType:ccv_nnc_mps_datatype(b->info.datatype)];

				MPSGraphTensor* mps_a = [graph resizeWithGradientTensor:mps_b 
                                       input:inputSizeTensor 
                                        mode:MPSGraphResizeNearest 
                                centerResult:YES
                                alignCorners:NO 
                                      layout:MPSGraphTensorNamedDataLayoutNCHW
                                        name:nil];

				[resultTensors addObject:mps_a];
			});
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, bdim, bstride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_b], &a, (int*[]){ adim }, (int*[]){ astride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	} else {
		assert(a->info.format == CCV_TENSOR_FORMAT_NHWC);
		assert(inputSize.count == 4);
		// for unknown reason, MPS handling NHWC as NHCW... 
		// explicitly transpose input and output for NHWC
		[inputSize exchangeObjectAtIndex:2 withObjectAtIndex:3]; 

		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, bdim_r, bstride_r, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, bdim_r, bstride_r);
				[inputShapedTypes addObject:mps_b_shape];
				
				MPSGraphTensor* inputSizeTensor = [graph constantWithScalar:0 shape:inputSize dataType:ccv_nnc_mps_datatype(b->info.datatype)];
				// NHWC to NHCW
				mps_b = [graph transposeTensor:mps_b dimension:-1 withDimension:-2 name:nil];

				MPSGraphTensor* mps_a = [graph resizeWithGradientTensor:mps_b 
                                       input:inputSizeTensor 
                                        mode:MPSGraphResizeNearest 
                                centerResult:YES
                                alignCorners:NO 
                                      layout:MPSGraphTensorNamedDataLayoutNHWC
                                        name:nil];
				// NHCW to NHWC
				mps_a = [graph transposeTensor:mps_a dimension:-1 withDimension:-2 name:nil];

				[resultTensors addObject:mps_a];
			});
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, bdim, bstride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_b], &a, (int*[]){ adim }, (int*[]){ astride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_upsample_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_NEAREST)
		return _ccv_nnc_upsample_nearest_back(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
	else if (cmd.info.upsample.type == CCV_NNC_UPSAMPLE_BILINEAR)
		return _ccv_nnc_upsample_bilinear_back(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
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

REGISTER_COMMAND_BACKEND(CCV_NNC_UPSAMPLE_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_upsample_back;
}
