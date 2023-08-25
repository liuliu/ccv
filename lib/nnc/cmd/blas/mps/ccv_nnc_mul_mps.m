#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include <Foundation/Foundation.h>
#include <assert.h>
#include <stdio.h>
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const float p = cmd.info.blas.a[0];
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	if (inputs[1] == 0)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			if (p == 1)
			{
				MPSGraph* graph = [MPSGraph new];
				graph.options = MPSGraphOptionsSynchronizeResults;
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				if (mps_a != mps_input_a)
					ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, c, c->info.dim, c->stride);
				else
					ccv_nnc_mps_export_data(data_a, command_buffer, c, c->info.dim, c->stride);
			} else {
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[1];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[2];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			if (p != 1)
			{
				MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				mps_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
			}
			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			[inputTensors addObject:mps_input_b];
			MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
			[inputShapedTypes addObject:mps_b_shape];
			MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
			[resultTensors addObject:mps_c];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
		MPSGraphTensorData* data[] = {data_a, data_b};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}


static int _ccv_nnc_mul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	int no_broadcasting = 1;
	assert(input_size >= 1);
	assert(output_size >= 1);

	if (outputs[0])
	{
		assert(input_size >= 3 && inputs[2]);
		ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)outputs[0], gdim);
		ccv_nnc_tensor_view_get_broadcast_dim((ccv_nnc_tensor_view_t*)inputs[2], gdim);
	}
	
	if (no_broadcasting && output_size > 1 && outputs[1])
	{
		assert(inputs[1]);
		ccv_nnc_tensor_view_get_dim((ccv_nnc_tensor_view_t*)inputs[1], gdim);
		ccv_nnc_tensor_view_get_broadcast_dim((ccv_nnc_tensor_view_t*)outputs[1], gdim);
	}

	const float p = cmd.info.blas.a[0];
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0] ? : 0;
	const ccv_nnc_tensor_view_t* const b1 = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b2 = (ccv_nnc_tensor_view_t*)inputs[1];

	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const h = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	const int b2_nd = ccv_nnc_tensor_nd(b1->info.dim);
	const int b1_nd = ccv_nnc_tensor_nd(b2->info.dim);
	const int g_nd = ccv_max(b2_nd, b1_nd);
	const int offset = CCV_NNC_MAX_DIM + 2 - g_nd;
	if (a)
		assert(b1); // b1 must be not nil when calculate a

	if (h) 
		assert(b2); // b2 must be not nil when calculate h

	@autoreleasepool {
		NSMutableArray<NSNumber*>* mps_g_shape = [[NSMutableArray new] autorelease];	
		for (int i = offset; i < CCV_NNC_MAX_DIM + 2; i++){
			[mps_g_shape addObject:@(gdim[i])]; // still need mps_g_shape for target broadcast shape
			gdim[i-offset] = gdim[i]; // move forward to align info.dim format 
		}               
		const int* gdim_a = gdim;

		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);

		if (a && b1)
		{
			ccv_nnc_mps_graph_key_t a_key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(a_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {			
				MPSGraphTensor* mps_input_b1;
				MPSGraphTensor* mps_b1 = ccv_nnc_mps_graph_tensor_input(graph, b1, b1->info.dim, b1->stride, &mps_input_b1);
				[inputTensors addObject:mps_input_b1];
				MPSGraphShapedType* mps_b1_shape = ccv_nnc_mps_graph_tensor_input_shape(b1, b1->info.dim, b1->stride);
				[inputShapedTypes addObject:mps_b1_shape];

				MPSGraphTensor* mps_g;
				if (g) {
					MPSGraphTensor* mps_input_g;
					mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];
				} else {
					// empty mps_g for target broadcast shape with 1.0 for each elements	 
					mps_g = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				}		

				MPSGraphTensor* mps_a = mps_g;
				if (p != 1)
				{
					MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(b1->info.datatype)];
					mps_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
				}

				// reshape
				if (mps_b1.shape.count < mps_g_shape.count) {
					NSMutableArray<NSNumber*>* b1_broadcastable_shape = mps_b1.shape.mutableCopy; 
					// padding left as [1, ..., 1, N] until b1 and g aligned, before broadcast
					for (int i = 0; i < mps_g_shape.count - mps_b1.shape.count; i++) {
						[b1_broadcastable_shape insertObject:@(1) atIndex:0]; 
					}
					mps_b1 = [graph reshapeTensor:mps_b1 withShape:b1_broadcastable_shape name:nil];
					[b1_broadcastable_shape release];
				}
				// broadcast
				mps_b1 = [graph broadcastTensor:mps_b1 toShape:mps_g_shape name:nil];
				// multiply 
				mps_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_b1 name:nil];

				NSMutableArray<NSNumber*>* da_axes = [NSMutableArray new];	
				const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
				for (int i = 0; i < a_nd; i++) {
					if (a->info.dim[i] != gdim_a[i])
						[da_axes addObject:@(i)];
				}
				// reduce
				mps_a = [graph reductionSumWithTensor:mps_a axes:da_axes name:nil];
				[da_axes release];
				[resultTensors addObject:mps_a];
			});
			NSArray<MPSGraphTensorData*>* inputs_array = nil;
			MPSGraphTensorData* data_b1 = ccv_nnc_mps_graph_tensor_data(b1, b1->info.dim, b1->stride);
			if (g) {
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data[] = { data_b1, data_g };
				inputs_array = @[data[indices[0]], data[indices[1]]];
			} else {
				MPSGraphTensorData* data[] = { data_b1 };
				inputs_array = @[data[indices[0]]];
			}
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, inputs_array, (ccv_nnc_tensor_view_t* []){ a }, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1);
		}

		if (h && b2)
		{
			ccv_nnc_mps_graph_key_t h_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(h_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {			
				MPSGraphTensor* mps_input_b2;
				MPSGraphTensor* mps_b2 = ccv_nnc_mps_graph_tensor_input(graph, b2, b2->info.dim, b2->stride, &mps_input_b2);
				[inputTensors addObject:mps_input_b2];
				MPSGraphShapedType* mps_b2_shape = ccv_nnc_mps_graph_tensor_input_shape(b2, b2->info.dim, b2->stride);
				[inputShapedTypes addObject:mps_b2_shape];
				MPSGraphTensor* mps_g;
				if (g) {
					MPSGraphTensor* mps_input_g;
					mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];
				} else {
					// empty mps_g for target broadcast shape with 1.0 for each elements	 
					mps_g = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				}		

				MPSGraphTensor* mps_h = mps_g;
				if (p != 1)
				{
					MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(b2->info.datatype)];
					mps_h = [graph multiplicationWithPrimaryTensor:mps_h secondaryTensor:mps_p name:nil];
				}

				// reshape
				if (mps_b2.shape.count < mps_g_shape.count) {
					// padding left as [1, ..., 1, N] until b2 and g aligned, before broadcast
					NSMutableArray<NSNumber*>* b2_broadcastable_shape = mps_b2.shape.mutableCopy;
					for (int i = 0; i < mps_g_shape.count - mps_b2.shape.count; i++) {
						[b2_broadcastable_shape insertObject:@(1) atIndex:0];
					}
					mps_b2 = [graph reshapeTensor:mps_b2 withShape:b2_broadcastable_shape name:nil];
					[b2_broadcastable_shape release];
				}
				// broadcast
				mps_b2 = [graph broadcastTensor:mps_b2 toShape:mps_g_shape name:nil];
				// multiply
				mps_h = [graph multiplicationWithPrimaryTensor:mps_h secondaryTensor:mps_b2 name:nil];

				NSMutableArray<NSNumber*>* dh_axes = [NSMutableArray new];	
				const int h_nd = ccv_nnc_tensor_nd(h->info.dim);
				for (int i = 0; i < h_nd; i++) {
					if (h->info.dim[i] != gdim_a[i])
						[dh_axes addObject:@(i)];
				}
				// reduce
				mps_h = [graph reductionSumWithTensor:mps_h axes:dh_axes name:nil];
				[dh_axes release];
				[resultTensors addObject:mps_h];
			});
			NSArray<MPSGraphTensorData*>* inputs_array = nil;
			MPSGraphTensorData* data_b2 = ccv_nnc_mps_graph_tensor_data(b2, b2->info.dim, b2->stride);
			if (g) {
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data[] = { data_b2, data_g };
				inputs_array = @[data[indices[0]], data[indices[1]]];
			} else {
				MPSGraphTensorData* data[] = { data_b2 };
				inputs_array = @[data[indices[0]]];
			}
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, inputs_array, (ccv_nnc_tensor_view_t* []){ h }, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1);
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MUL_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mul_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mul_forw;
}

static int _ccv_nnc_scalar_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const float p = cmd.info.blas.a[0];
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		if (p == 1)
		{
			MPSGraph* graph = [MPSGraph new];
			graph.options = MPSGraphOptionsSynchronizeResults;
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, c, c->info.dim, c->stride);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, c, c->info.dim, c->stride);
			[graph release];
		} else if (p == 0.5 || p == 2 || p == 1.0 / 3 || p == 3 || p == 1.0 / 10 || p == 10) { // Only create specialized kernels for special p values.
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
				[resultTensors addObject:mps_c];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
		} else {
			ccv_nnc_cmd_t cmd_without_p = cmd;
			cmd_without_p.info.blas.a[0] = 0;
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd_without_p, 1, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			id<MTLBuffer> p_buffer;
			if (a->info.datatype == CCV_16F)
			{
				uint16_t p_bytes;
				ccv_float_to_half_precision(&p, &p_bytes, 1);
#ifdef __x86_64__
				p_buffer = [ccv_nnc_default_device() newBufferWithBytes:&p_bytes length:sizeof(uint16_t) options:MTLResourceStorageModePrivate];
#else
				p_buffer = [ccv_nnc_default_device() newBufferWithBytes:&p_bytes length:sizeof(uint16_t) options:MTLResourceStorageModeShared];
#endif
			} else {
#ifdef __x86_64__
				p_buffer = [ccv_nnc_default_device() newBufferWithBytes:&p length:sizeof(float) options:MTLResourceStorageModePrivate];
#else
				p_buffer = [ccv_nnc_default_device() newBufferWithBytes:&p length:sizeof(float) options:MTLResourceStorageModeShared];
#endif
			}
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_p = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype) name:nil];
				[inputTensors addObject:mps_p];
				MPSGraphShapedType* mps_p_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				[inputShapedTypes addObject:mps_p_shape];
				[mps_p_shape release];
				MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
				[resultTensors addObject:mps_c];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_p = [[MPSGraphTensorData alloc] initWithMTLBuffer:p_buffer shape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensorData* data[] = {data_a, data_p};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
			[data_p release];
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scalar_mul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const float p = cmd.info.blas.a[0];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];

	if (inputs[0] == 0)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphShapedType* mps_c_shape = ccv_nnc_mps_graph_tensor_input_shape(c, c->info.dim, c->stride);

				MPSGraphTensor* mps_c = [graph constantWithScalar:p shape:mps_c_shape.shape dataType:ccv_nnc_mps_datatype(c->info.datatype)];
				[resultTensors addObject:mps_c];
			});
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}

	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		if (p == 1)
		{
			MPSGraph* graph = [MPSGraph new];
			graph.options = MPSGraphOptionsSynchronizeResults;
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, c, c->info.dim, c->stride);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, c, c->info.dim, c->stride);
			[graph release];
		} else {
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
				[resultTensors addObject:mps_c];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALAR_MUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scalar_mul_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALAR_MUL_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scalar_mul_back;
}
