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

static int _ccv_nnc_add_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
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
				[graph release];
			} else {
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
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
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
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
			if (q != 1)
			{
				MPSGraphTensor* mps_q = [graph constantWithScalar:q dataType:ccv_nnc_mps_datatype(b->info.datatype)];
				mps_b = [graph multiplicationWithPrimaryTensor:mps_b secondaryTensor:mps_q name:nil];
			}
			MPSGraphTensor* mps_c = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
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

static int _ccv_nnc_add_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const b = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[2];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
			[inputShapedTypes addObject:mps_g_shape];

			MPSGraphTensor* mps_a = mps_g;
			if (p != 1)
			{
				MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(g->info.datatype)];
				mps_a = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_p name:nil];
			}
			NSMutableArray<NSNumber*>* da_axes = [NSMutableArray new];
			for (int i = 0; i < a_nd; i++) {
				if (a->info.dim[i] != g->info.dim[i])
					[da_axes addObject:@(i)];
			}
			mps_a = [graph reductionSumWithTensor:mps_a axes:da_axes name:nil];
			[resultTensors addObject:mps_a];

			if (b) {
				MPSGraphTensor* mps_b = mps_g;
				if (q != 1)
				{
					MPSGraphTensor* mps_q = [graph constantWithScalar:q dataType:ccv_nnc_mps_datatype(g->info.datatype)];
					mps_b = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_q name:nil];
				}
				NSMutableArray<NSNumber*>* db_axes = [NSMutableArray new];
				const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
				for (int i = 0; i < b_nd; i++) {
					if (b->info.dim[i] != g->info.dim[i])
						[db_axes addObject:@(i)];
				}
				mps_b = [graph reductionSumWithTensor:mps_b axes:db_axes name:nil];
				[resultTensors addObject:mps_b];
			}
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data[] = {data_g};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], (ccv_nnc_tensor_view_t* []){ a, b }, (int*[]){ a->info.dim,b->info.dim }, (int*[]){ a->stride, b->stride }, 2);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_back;
}