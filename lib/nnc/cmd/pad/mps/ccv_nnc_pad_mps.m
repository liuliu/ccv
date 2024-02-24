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

static int _ccv_nnc_pad_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		const int* const begin = cmd.info.size.dim;
		const int* const end = cmd.info.pad.end;
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(a_nd == ccv_nnc_tensor_nd(b->info.dim));
		const int type = cmd.info.pad.type;
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			NSMutableArray* leftPadding = [NSMutableArray new];
			NSMutableArray* rightPadding = [NSMutableArray new];
			for (int i = 0; i < a_nd; i++)
			{
				[leftPadding addObject:@(begin[i])];
				[rightPadding addObject:@(end[i])];
			}
			MPSGraphPaddingMode paddingMode;
			if (type == CCV_NNC_PAD_ZERO)
				paddingMode = MPSGraphPaddingModeZero;
			else
				paddingMode = MPSGraphPaddingModeClampToEdge;
			MPSGraphTensor* mps_b = [graph padTensor:mps_a withPaddingMode:paddingMode leftPadding:leftPadding rightPadding:rightPadding constantValue:0 name:nil];
			[leftPadding release];
			[rightPadding release];
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_pad_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t a_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		const int* const begin = cmd.info.size.dim;
		const int* const end = cmd.info.pad.end;
		const int g_nd = ccv_nnc_tensor_nd(g->info.dim);
		assert(g_nd == ccv_nnc_tensor_nd(a->info.dim));
		const int type = cmd.info.pad.type;
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(a_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
			[inputShapedTypes addObject:mps_g_shape];
			NSMutableArray* leftPadding = [NSMutableArray new];
			NSMutableArray* rightPadding = [NSMutableArray new];
			// This works because we don't support negative padding in forward pass.
			for (int i = 0; i < g_nd; i++)
			{
				[leftPadding addObject:@(-begin[i])];
				[rightPadding addObject:@(-end[i])];
			}
			MPSGraphPaddingMode paddingMode = MPSGraphPaddingModeZero;
			MPSGraphTensor* mps_a = [graph padTensor:mps_g withPaddingMode:paddingMode leftPadding:leftPadding rightPadding:rightPadding constantValue:0 name:nil];
			[leftPadding release];
			[rightPadding release];
			[resultTensors addObject:mps_a];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data[] = {data_g};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], (ccv_nnc_tensor_view_t* []){ a }, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1, 0);
		
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_PAD_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_pad_back;
}
