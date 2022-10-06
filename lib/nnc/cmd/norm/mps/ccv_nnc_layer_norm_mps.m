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
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const scale = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const bias = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)outputs[2];
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(saved_mean, rdim);
	assert(ccv_nnc_tensor_view_check_dim(saved_inv_std, rdim));
	assert(ccv_nnc_tensor_view_check_dim(b, adim));
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
		MPSGraphTensor* mps_input_scale;
		MPSGraphTensor* mps_scale = ccv_nnc_mps_graph_tensor_input(graph, scale, scale->info.dim, scale->stride, &mps_input_scale);
		MPSGraphTensor* mps_input_bias;
		MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, bias, bias->info.dim, bias->stride, &mps_input_bias);
		// I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(scale, scale->info.dim, scale->stride);
		MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, bias->info.dim, bias->stride);
		int i;
		NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
		const int rnd = ccv_nnc_tensor_nd(rdim);
		for (i = 0; i < rnd; i++)
			if (rdim[i] != adim[i])
				[axes addObject:@(i)];
		MPSGraphTensor* mps_saved_mean = [graph meanOfTensor:mps_a axes:axes name:nil];
		MPSGraphTensor* mps_saved_inv_std = [graph varianceOfTensor:mps_a meanTensor:mps_saved_mean axes:axes name:nil];
		[axes release];
		const float epsilon = cmd.info.lnorm.epsilon;
		MPSGraphTensor* mps_b = [graph normalizationWithTensor:mps_a meanTensor:mps_saved_mean varianceTensor:mps_saved_inv_std gammaTensor:mps_scale betaTensor:mps_bias epsilon:epsilon name:nil];
		ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a, mps_input_scale: data_scale, mps_input_bias: data_bias}, mps_b, b);
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
