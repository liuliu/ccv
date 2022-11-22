#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_softmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		if (a_nd <= 2 && b_nd <= 2 && !ccv_nnc_is_a13_and_below()) // Simple case, we use MPS directly.
		{
			assert(a_nd > 0);
			assert(b_nd > 0);
			id<MTLBuffer> a_buffer = mpgetbuffer((ccv_nnc_tensor_t*)a);
			const int a_rows = a_nd == 1 ? 1 : a->info.dim[0];
			const int a_cols = a_nd == 1 ? a->info.dim[0] : a->info.dim[1];
			const size_t a_row_bytes = (CCV_IS_TENSOR_VIEW(a) && a_nd == 2) ? a->stride[0] : CCV_GET_DATA_TYPE_SIZE(a->info.datatype) * a_cols;
			MPSMatrix* inputMatrix = [[MPSMatrix alloc] initWithBuffer:a_buffer offset:a->dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:a_rows columns:a_cols rowBytes:a_row_bytes dataType:ccv_nnc_mps_datatype(a->info.datatype)]];
			id<MTLBuffer> b_buffer = mpgetbuffer((ccv_nnc_tensor_t*)b);
			const int b_rows = b_nd == 1 ? 1 : b->info.dim[0];
			const int b_cols = b_nd == 1 ? b->info.dim[0] : b->info.dim[1];
			const size_t b_row_bytes = (CCV_IS_TENSOR_VIEW(b) && b_nd == 2) ? b->stride[0] : CCV_GET_DATA_TYPE_SIZE(b->info.datatype) * b_cols;
			MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:b_buffer offset:b->dataof descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:b_rows columns:b_cols rowBytes:b_row_bytes dataType:ccv_nnc_mps_datatype(b->info.datatype)]];
			MPSMatrixSoftMax* softmax = [[MPSMatrixSoftMax alloc] initWithDevice:ccv_nnc_default_device()];
			[inputMatrix synchronizeOnCommandBuffer:command_buffer];
			[softmax encodeToCommandBuffer:command_buffer inputMatrix:inputMatrix resultMatrix:resultMatrix];
			[resultMatrix synchronizeOnCommandBuffer:command_buffer];
			[inputMatrix release];
			[resultMatrix release];
			[softmax release];
		} else {
			// Otherwise, use MPSGraph.
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph softMaxWithTensor:mps_a axis:-1 name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOFTMAX_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_softmax_forw;
}
