#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_mse_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
    printf(_ccv_nnc_mse_forw);
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(c, cstride);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = astride[CCV_NNC_MAX_DIM];
	const int bstep = bstride[CCV_NNC_MAX_DIM];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cstride[CCV_NNC_MAX_DIM];

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

			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			[inputTensors addObject:mps_input_b];
			MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
			[inputShapedTypes addObject:mps_b_shape];
            
            MPSGraphTensor* diffTensor = [graph subtractionWithPrimaryTensor:mps_a
                                                          secondaryTensor:mps_b
                                                                     name:nil];
            MPSGraphTensor* diffSquareTensor = [graph squareWithTensor:diffTensor name:nil];
            MPSGraphTensor* mps_c = ccv_nnc_mps_reduce_tensor(diffSquareTensor,cmd.info.mse.reduce_op, graph, a->info.dim);
			[resultTensors addObject:mps_c];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
		MPSGraphTensorData* data[] = {data_a, data_b};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1);
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mse_forw;
}
