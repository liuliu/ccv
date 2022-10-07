#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_layer_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 3);
	ccv_nnc_tensor_view_t at = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t scalet = ccv_nnc_get_tensor_view(inputs[1]);
	ccv_nnc_tensor_view_t biast = ccv_nnc_get_tensor_view(inputs[2]);
	ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t saved_meant = ccv_nnc_get_tensor_view(outputs[1]);
	ccv_nnc_tensor_view_t saved_inv_stdt = ccv_nnc_get_tensor_view(outputs[2]);
	ccv_nnc_tensor_view_alignment((ccv_nnc_tensor_view_t*[]){
		&at,
		&saved_meant,
		&saved_inv_stdt,
		&bt
	}, 4);
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &at, at.info.dim, at.stride, &mps_input_a);
		MPSGraphTensor* mps_input_scale;
		MPSGraphTensor* mps_scale = ccv_nnc_mps_graph_tensor_input(graph, &scalet, scalet.info.dim, scalet.stride, &mps_input_scale);
		MPSGraphTensor* mps_input_bias;
		MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, &biast, biast.info.dim, biast.stride, &mps_input_bias);
		// I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&at, at.info.dim, at.stride);
		MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(&scalet, scalet.info.dim, scalet.stride);
		MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(&biast, biast.info.dim, biast.stride);
		int i;
		NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
		const int rnd = ccv_nnc_tensor_nd(saved_meant.info.dim);
		for (i = 0; i < rnd; i++)
			if (at.info.dim[i] != saved_meant.info.dim[i])
				[axes addObject:@(i)];
		MPSGraphTensor* mps_saved_mean = [graph meanOfTensor:mps_a axes:axes name:nil];
		MPSGraphTensor* mps_saved_inv_std = [graph varianceOfTensor:mps_a meanTensor:mps_saved_mean axes:axes name:nil];
		[axes release];
		const float epsilon = cmd.info.lnorm.epsilon;
		MPSGraphTensor* mps_b = [graph normalizationWithTensor:mps_a meanTensor:mps_saved_mean varianceTensor:mps_saved_inv_std gammaTensor:mps_scale betaTensor:mps_bias epsilon:epsilon name:nil];
		ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a, mps_input_scale: data_scale, mps_input_bias: data_bias}, mps_b, &bt, bt.info.dim, bt.stride);
		[graph release];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_forw;
}
