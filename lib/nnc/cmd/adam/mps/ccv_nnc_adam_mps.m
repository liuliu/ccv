#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_adam_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	assert(output_size >= 3);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const u = (ccv_nnc_tensor_view_t*)outputs[2];
	const int step = cmd.info.adam.step;
	const float rate = cmd.info.adam.rate;
	const float scale = cmd.info.adam.scale;
	const float beta1 = cmd.info.adam.beta1;
	const float beta2 = cmd.info.adam.beta2;
	const float decay = cmd.info.adam.decay;
	const float epsilon = cmd.info.adam.epsilon;
	assert(step >= 1);
	const float rate_inv_bias_correction1 = rate / (1 - powf(beta1, step));
	const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_cmd_t cmd_without_step = cmd;
		cmd_without_step.info.adam.step = 0;
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd_without_step, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[6];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
			[inputShapedTypes addObject:mps_g_shape];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_input_m;
			MPSGraphTensor* mps_m = ccv_nnc_mps_graph_tensor_input(graph, m, m->info.dim, m->stride, &mps_input_m);
			[inputTensors addObject:mps_input_m];
			MPSGraphShapedType* mps_m_shape = ccv_nnc_mps_graph_tensor_input_shape(m, m->info.dim, m->stride);
			[inputShapedTypes addObject:mps_m_shape];
			MPSGraphTensor* mps_input_v;
			MPSGraphTensor* mps_v = ccv_nnc_mps_graph_tensor_input(graph, v, v->info.dim, v->stride, &mps_input_v);
			[inputTensors addObject:mps_input_v];
			MPSGraphShapedType* mps_v_shape = ccv_nnc_mps_graph_tensor_input_shape(v, v->info.dim, v->stride);
			[inputShapedTypes addObject:mps_v_shape];
			MPSGraphTensor* mps_rate_inv_bias_correction1 = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(m->info.datatype) name:nil];
			[inputTensors addObject:mps_rate_inv_bias_correction1];
			MPSGraphShapedType* mps_rate_inv_bias_correction1_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(m->info.datatype)];
			[inputShapedTypes addObject:mps_rate_inv_bias_correction1_shape];
			[mps_rate_inv_bias_correction1_shape release];
			MPSGraphTensor* mps_inv_bias_correction2 = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(v->info.datatype) name:nil];
			[inputTensors addObject:mps_inv_bias_correction2];
			MPSGraphShapedType* mps_inv_bias_correction2_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(v->info.datatype)];
			[inputShapedTypes addObject:mps_inv_bias_correction2_shape];
			[mps_inv_bias_correction2_shape release];
			MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:ccv_nnc_mps_datatype(g->info.datatype)];
			mps_g = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_scale name:nil];
			MPSGraphTensor* mps_decay = [graph constantWithScalar:decay dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_decay_x_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_decay name:nil];
			mps_g = [graph additionWithPrimaryTensor:mps_g secondaryTensor:mps_decay_x_a name:nil];
			MPSGraphTensor* mps_beta1 = [graph constantWithScalar:beta1 dataType:ccv_nnc_mps_datatype(m->info.datatype)];
			MPSGraphTensor* mps_beta2 = [graph constantWithScalar:beta2 dataType:ccv_nnc_mps_datatype(v->info.datatype)];
			MPSGraphTensor* mps_1_beta1 = [graph constantWithScalar:1 - beta1 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
			MPSGraphTensor* mps_1_beta2 = [graph constantWithScalar:1 - beta2 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
			mps_m = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_beta1 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_1_beta1 name:nil] name:nil];
			mps_v = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_beta2 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:[graph squareWithTensor:mps_g name:nil] secondaryTensor:mps_1_beta2 name:nil] name:nil];
			MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_b = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:[graph divisionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_rate_inv_bias_correction1 name:nil] secondaryTensor:[graph additionWithPrimaryTensor:[graph squareRootWithTensor:[graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_inv_bias_correction2 name:nil] name:nil] secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
			[resultTensors addObject:mps_b];
			[resultTensors addObject:mps_m];
			[resultTensors addObject:mps_v];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_m = ccv_nnc_mps_graph_tensor_data(m, m->info.dim, m->stride);
		MPSGraphTensorData* data_v = ccv_nnc_mps_graph_tensor_data(v, v->info.dim, v->stride);
		MPSGraphTensorData* data_rate_inv_bias_correction1 = ccv_nnc_mps_graph_constant_data(rate_inv_bias_correction1, m->info.datatype);
		MPSGraphTensorData* data_inv_bias_correction2 = ccv_nnc_mps_graph_constant_data(inv_bias_correction2, v->info.datatype);
		MPSGraphTensorData* data[] = {data_g, data_a, data_m, data_v, data_rate_inv_bias_correction1, data_inv_bias_correction2};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]], data[indices[4]], data[indices[5]]], (ccv_nnc_tensor_view_t* []){ b, n, u }, (int*[]){ b->info.dim, n->info.dim, u->info.dim }, (int*[]){ b->stride, n->stride, u->stride }, 3);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adam_forw;
}
