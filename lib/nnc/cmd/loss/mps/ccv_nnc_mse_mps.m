#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_mse_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
    ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
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

			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			[inputTensors addObject:mps_input_b];
			MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
			[inputShapedTypes addObject:mps_b_shape];
			MPSGraphTensor* diff_tensor = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
			MPSGraphTensor* diff_square_tensor = [graph squareWithTensor:diff_tensor name:nil];
			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
			int i;
			for (i = 0; i < a_nd; i++) {
				if (a->info.dim[i] != c->info.dim[i])
					[axes addObject:@(i)];
			}
			MPSGraphTensor* mps_c;
			if (cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN) {
				mps_c = [graph meanOfTensor:diff_square_tensor axes:axes name:nil];
			} else {
				assert(cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_SUM);
				mps_c = [graph reductionSumWithTensor:diff_square_tensor axes:axes name:nil];
			}
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

static int _ccv_nnc_mse_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const hb = output_size >= 2 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	if (ha)
		{ assert(ccv_nnc_tensor_view_check_dim(ha, dim)); }
	if (hb)
		{ assert(ccv_nnc_tensor_view_check_dim(hb, dim)); }
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const float inv_mean_2 = cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN ? 2.0 / (float)count : 2.0;
	assert(cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN || cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_SUM);

	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
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
			MPSGraphTensor *mps_inv_mean =  [graph constantWithScalar:inv_mean_2 dataType:mps_a.dataType];

			if (g) {
				MPSGraphTensor* mps_input_g;
				MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
				[inputTensors addObject:mps_input_g];
				MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
				[inputShapedTypes addObject:mps_g_shape];

				NSMutableArray<NSNumber*>* g_broadcastable_shape = [NSMutableArray new];  // [N]
				for (int i = 0; i < a_nd; i++) {
					if (g->info.dim[i] > 0) {
						[g_broadcastable_shape addObject:@(g->info.dim[i])];
					}
					if (g->info.dim[i] != a->info.dim[i]) {     // [N]
						[g_broadcastable_shape addObject:@(1)]; // [N, 1]
					}
				}
				mps_g = [graph reshapeTensor:mps_g withShape:g_broadcastable_shape name:nil];

				if (ha) {
					MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
					MPSGraphTensor* mps_ha = [graph multiplicationWithPrimaryTensor:diff secondaryTensor:mps_g name:nil];
					mps_ha = [graph multiplicationWithPrimaryTensor:mps_ha secondaryTensor:mps_inv_mean name:nil];
					[resultTensors addObject:mps_ha];
				}

				if (hb) {
					MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:mps_b secondaryTensor:mps_a name:nil];
					MPSGraphTensor* mps_hb = [graph multiplicationWithPrimaryTensor:diff secondaryTensor:mps_g name:nil];
					mps_hb = [graph multiplicationWithPrimaryTensor:mps_hb secondaryTensor:mps_inv_mean name:nil];
					[resultTensors addObject:mps_hb];
				}

			} else {

				if (ha) {
					MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
					MPSGraphTensor* mps_ha = [graph multiplicationWithPrimaryTensor:diff secondaryTensor:mps_inv_mean name:nil];
					[resultTensors addObject:mps_ha];
				}

				if (hb) {
					MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:mps_b secondaryTensor:mps_a name:nil];
					MPSGraphTensor* mps_hb = [graph multiplicationWithPrimaryTensor:diff secondaryTensor:mps_inv_mean name:nil];
					[resultTensors addObject:mps_hb];
				}
			}
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
		MPSGraphTensorData* data[] = {data_a, data_b, data_g};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){ ha, hb }, (int*[]){ ha->info.dim, hb->info.dim }, (int*[]){ ha->stride, hb->stride }, 2);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MSE_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mse_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mse_forw;
}
