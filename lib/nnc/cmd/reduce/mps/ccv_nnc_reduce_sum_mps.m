#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_reduce_sum_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_view_t atv = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t btv = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t* tvs[] = {
		&atv, &btv
	};
	ccv_nnc_tensor_view_alignment(tvs, 2);
	const int a_nd = ccv_nnc_tensor_nd(atv.info.dim);
	int noop = 1;
	int i;
	for (i = 0; noop && i < a_nd; i++)
		noop = btv.info.dim[i] != atv.info.dim[i];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		if (noop)
		{
			MPSGraph* graph = [MPSGraph new];
			graph.options = MPSGraphOptionsSynchronizeResults;
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &atv, atv.info.dim, atv.stride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&atv, atv.info.dim, atv.stride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, &btv, btv.info.dim, btv.stride);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, &btv, btv.info.dim, btv.stride);
			[graph release];
		} else {
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &atv, atv.info.dim, atv.stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(&atv, atv.info.dim, atv.stride);
				[inputShapedTypes addObject:mps_a_shape];
				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				int i;
				for (i = 0; i < a_nd; i++)
					if (btv.info.dim[i] != atv.info.dim[i])
						[axes addObject:@(i)];
				MPSGraphTensor* mps_b = [graph reductionSumWithTensor:mps_a axes:axes name:nil];
				[axes release];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&atv, atv.info.dim, atv.stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &tvs[1], (int*[]){ btv.info.dim }, (int*[]){ btv.stride }, 1, 0);
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_REDUCE_SUM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_reduce_sum_forw;
}
