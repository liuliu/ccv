#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_adamw_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 4);
	assert(output_size >= 3);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const m = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const v = (ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* const vm = input_size >= 5 ? (ccv_nnc_tensor_view_t*)inputs[4] : 0;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const n = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const u = (ccv_nnc_tensor_view_t*)outputs[2];
	ccv_nnc_tensor_view_t* const um = output_size >= 4 ? (ccv_nnc_tensor_view_t*)outputs[3] : 0;
	const int step = cmd.info.adam.step;
	const float rate = cmd.info.adam.rate;
	const float scale = cmd.info.adam.scale;
	const float beta1 = cmd.info.adam.beta1;
	const float beta2 = cmd.info.adam.beta2;
	const float decay = cmd.info.adam.decay;
	const float epsilon = cmd.info.adam.epsilon;
	assert(step >= 1);
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(inputs[0]->info.datatype == outputs[0]->info.datatype) &&
				(inputs[0]->info.datatype == outputs[1]->info.datatype) &&
				(inputs[0]->info.datatype == outputs[2]->info.datatype) &&
				(!um || inputs[0]->info.datatype == um->info.datatype) &&
				(inputs[0]->info.datatype == inputs[1]->info.datatype) &&
				(inputs[0]->info.datatype == inputs[2]->info.datatype) &&
				(!vm || inputs[0]->info.datatype == vm->info.datatype) &&
				(inputs[0]->info.datatype == inputs[3]->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (a->info.datatype) {
				case CCV_16F: {
					mtl_data_type = 16;
					break;
				}
				case CCV_32F: {
					mtl_data_type = 3;
					break;
				}
				default: {
					use_mfa = false;
					fallback_reason = "Unsupported data type.";
					break;
				}
			}
		}

		if (use_mfa) {
			if (!CCV_IS_TENSOR_CONTIGUOUS(inputs[0]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(outputs[0]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(outputs[1]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(outputs[2]) ||
					(!um || !CCV_IS_TENSOR_CONTIGUOUS(um)) ||
					!CCV_IS_TENSOR_CONTIGUOUS(inputs[1]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(inputs[2]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(inputs[3]) ||
					(!vm || !CCV_IS_TENSOR_CONTIGUOUS(vm)))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_adam_params_t params = {
				.data_type = mtl_data_type,
				.adamw = 1,
				.amsgrad = cmd.info.adam.amsgrad,
				.step = step,
				.rate = rate,
				.scale = scale,
				.beta1 = beta1,
				.beta2 = beta2,
				.decay = decay,
				.epsilon = epsilon,
				.length = ccv_nnc_tensor_count(a->info),
			};
			ccv_nnc_mfa_prepare_adam(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[10] = {
				mpgetbuffer(inputs[0]), // gradient
				mpgetbuffer(inputs[1]), // source
				mpgetbuffer(outputs[0]), // destination
				mpgetbuffer(inputs[2]), // mom
				mpgetbuffer(inputs[3]), // vel
				mpgetbuffer(outputs[1]), // new_mom
				mpgetbuffer(outputs[2]), // new_vel
				NULL,
			};
			size_t tensor_offsets[9] = {
				g->dataof,
				a->dataof,
				b->dataof,
				m->dataof,
				v->dataof,
				n->dataof,
				u->dataof,
			};
			if (vm && um)
			{
				tensors[7] = mpgetbuffer(vm);
				tensors[8] = mpgetbuffer(um);
				tensor_offsets[7] = vm->dataof;
				tensor_offsets[8] = um->dataof;
			}
			ccv_nnc_mfa_encode_adam(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			const float rate_inv_bias_correction1 = rate / (1 - powf(beta1, step));
			const float inv_bias_correction2 = 1. / (1 - powf(beta2, step));
			const float rate_decay = rate * decay;
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_cmd_t cmd_without_step = cmd;
			cmd_without_step.info.adam.step = 0;
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd_without_step, 0, hint, flags, inputs, input_size, outputs, output_size);
			if (cmd.info.adam.amsgrad)
			{
				int indices[7];
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
					MPSGraphTensor* mps_input_vm;
					MPSGraphTensor* mps_vm = ccv_nnc_mps_graph_tensor_input(graph, vm, vm->info.dim, vm->stride, &mps_input_vm);
					[inputTensors addObject:mps_input_vm];
					MPSGraphShapedType* mps_vm_shape = ccv_nnc_mps_graph_tensor_input_shape(vm, vm->info.dim, vm->stride);
					[inputShapedTypes addObject:mps_vm_shape];
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
					MPSGraphTensor* mps_beta1 = [graph constantWithScalar:beta1 dataType:ccv_nnc_mps_datatype(m->info.datatype)];
					MPSGraphTensor* mps_beta2 = [graph constantWithScalar:beta2 dataType:ccv_nnc_mps_datatype(v->info.datatype)];
					MPSGraphTensor* mps_1_beta1 = [graph constantWithScalar:1 - beta1 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
					MPSGraphTensor* mps_1_beta2 = [graph constantWithScalar:1 - beta2 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
					mps_m = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_beta1 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_1_beta1 name:nil] name:nil];
					mps_v = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_beta2 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:[graph squareWithTensor:mps_g name:nil] secondaryTensor:mps_1_beta2 name:nil] name:nil];
					MPSGraphTensor* mps_v_hat = [graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_inv_bias_correction2 name:nil];
					MPSGraphTensor* mps_v_max_hat = [graph maximumWithPrimaryTensor:mps_v_hat secondaryTensor:mps_vm name:nil];
					MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_rate_decay = [graph constantWithScalar:rate_decay dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_rate_decay_x_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_rate_decay name:nil];
					MPSGraphTensor* mps_b = [graph subtractionWithPrimaryTensor:[graph subtractionWithPrimaryTensor: mps_a secondaryTensor:mps_rate_decay_x_a name:nil] secondaryTensor:[graph divisionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_rate_inv_bias_correction1 name:nil] secondaryTensor:[graph additionWithPrimaryTensor:[graph squareRootWithTensor:mps_v_max_hat name:nil] secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
					[resultTensors addObject:mps_b];
					[resultTensors addObject:mps_m];
					[resultTensors addObject:mps_v];
					[resultTensors addObject:mps_v_max_hat];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_m = ccv_nnc_mps_graph_tensor_data(m, m->info.dim, m->stride);
				MPSGraphTensorData* data_v = ccv_nnc_mps_graph_tensor_data(v, v->info.dim, v->stride);
				MPSGraphTensorData* data_vm = ccv_nnc_mps_graph_tensor_data(vm, vm->info.dim, vm->stride);
				MPSGraphTensorData* data_rate_inv_bias_correction1 = ccv_nnc_mps_graph_constant_data(rate_inv_bias_correction1, m->info.datatype);
				MPSGraphTensorData* data_inv_bias_correction2 = ccv_nnc_mps_graph_constant_data(inv_bias_correction2, v->info.datatype);
				MPSGraphTensorData* data[] = {data_g, data_a, data_m, data_v, data_vm, data_rate_inv_bias_correction1, data_inv_bias_correction2};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]], data[indices[4]], data[indices[5]], data[indices[6]]], (ccv_nnc_tensor_view_t* []){ b, n, u, um }, (int*[]){ b->info.dim, n->info.dim, u->info.dim, um->info.dim }, (int*[]){ b->stride, n->stride, u->stride, um->stride }, 4, 1);
			} else {
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
					MPSGraphTensor* mps_beta1 = [graph constantWithScalar:beta1 dataType:ccv_nnc_mps_datatype(m->info.datatype)];
					MPSGraphTensor* mps_beta2 = [graph constantWithScalar:beta2 dataType:ccv_nnc_mps_datatype(v->info.datatype)];
					MPSGraphTensor* mps_1_beta1 = [graph constantWithScalar:1 - beta1 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
					MPSGraphTensor* mps_1_beta2 = [graph constantWithScalar:1 - beta2 dataType:ccv_nnc_mps_datatype(g->info.datatype)];
					mps_m = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_beta1 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_1_beta1 name:nil] name:nil];
					mps_v = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_beta2 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:[graph squareWithTensor:mps_g name:nil] secondaryTensor:mps_1_beta2 name:nil] name:nil];
					MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_rate_decay = [graph constantWithScalar:rate_decay dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_rate_decay_x_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_rate_decay name:nil];
					MPSGraphTensor* mps_b = [graph subtractionWithPrimaryTensor:[graph subtractionWithPrimaryTensor: mps_a secondaryTensor:mps_rate_decay_x_a name:nil] secondaryTensor:[graph divisionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_m secondaryTensor:mps_rate_inv_bias_correction1 name:nil] secondaryTensor:[graph additionWithPrimaryTensor:[graph squareRootWithTensor:[graph multiplicationWithPrimaryTensor:mps_v secondaryTensor:mps_inv_bias_correction2 name:nil] name:nil] secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
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
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]], data[indices[4]], data[indices[5]]], (ccv_nnc_tensor_view_t* []){ b, n, u }, (int*[]){ b->info.dim, n->info.dim, u->info.dim }, (int*[]){ b->stride, n->stride, u->stride }, 3, 1);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADAMW_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_adamw_forw;
}
