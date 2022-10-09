#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_gelu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_b;
			if (cmd.info.gelu.tanh)
			{
				MPSGraphTensor* mps_x_3 = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:[graph squareWithTensor:mps_a name:nil] name:nil];
				MPSGraphTensor* mps_c0 = [graph constantWithScalar:0.044715 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_mul0 = [graph multiplicationWithPrimaryTensor:mps_x_3 secondaryTensor:mps_c0 name:nil];
				MPSGraphTensor* mps_x_sum = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_mul0 name:nil];
				MPSGraphTensor* mps_c1 = [graph constantWithScalar:0.797884560802865355 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_mul1 = [graph multiplicationWithPrimaryTensor:mps_x_sum secondaryTensor:mps_c1 name:nil];
				MPSGraphTensor* mps_tanh = [graph tanhWithTensor:mps_mul1 name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_tanh secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_half = [graph constantWithScalar:0.5 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				mps_b = [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_sum secondaryTensor:mps_a name:nil] secondaryTensor:mps_half name:nil];
			} else {
				MPSGraphTensor* mps_c = [graph constantWithScalar:0.70710678118654752440 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_x = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_c name:nil];
				MPSGraphTensor* mps_erf = [graph erfWithTensor:mps_x name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_erf secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_half = [graph constantWithScalar:0.5 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				mps_b = [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_sum secondaryTensor:mps_a name:nil] secondaryTensor:mps_half name:nil];
			}
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], b, b->info.dim, b->stride);
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GELU_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gelu_forw;
}
