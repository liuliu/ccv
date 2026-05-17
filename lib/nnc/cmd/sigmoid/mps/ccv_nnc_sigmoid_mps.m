#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_sigmoid_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	if (a->info.dim[0] == 0 || b->info.dim[0] == 0)
		return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(a->info.datatype == b->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (a->info.datatype) {
				case CCV_16F: {
					mtl_data_type = 16;
					break;
				}
				case CCV_16BF: {
					mtl_data_type = 121;
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
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) ||
					!CCV_IS_TENSOR_CONTIGUOUS(b))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_sigmoid_params_t params = {
				.gradient = 0,
				.data_type = mtl_data_type
			};
			const size_t count = ccv_nnc_tensor_count(a->info);
			params.length = count;
			ccv_nnc_mfa_prepare_sigmoid(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[2] = {
				a->dataof,
				b->dataof
			};
			ccv_nnc_mfa_encode_sigmoid(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_b = [graph sigmoidWithTensor:mps_a name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sigmoid_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (g->info.dim[0] == 0 || b->info.dim[0] == 0 || h->info.dim[0] == 0)
		return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(g->info.datatype == b->info.datatype) &&
				(g->info.datatype == h->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (g->info.datatype) {
				case CCV_16F: {
					mtl_data_type = 16;
					break;
				}
				case CCV_16BF: {
					mtl_data_type = 121;
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
			if (!CCV_IS_TENSOR_CONTIGUOUS(g) ||
					!CCV_IS_TENSOR_CONTIGUOUS(b) ||
					!CCV_IS_TENSOR_CONTIGUOUS(h))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_sigmoid_params_t params = {
				.gradient = 1,
				.data_type = mtl_data_type
			};
			const size_t count = ccv_nnc_tensor_count(g->info);
			params.length = count;
			ccv_nnc_mfa_prepare_sigmoid(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[3] = {
				g->dataof,
				b->dataof,
				h->dataof
			};
			ccv_nnc_mfa_encode_sigmoid(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_g;
				MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
				[inputTensors addObject:mps_input_g];
				MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
				[inputShapedTypes addObject:mps_g_shape];

				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
				[inputShapedTypes addObject:mps_b_shape];

				MPSGraphTensor *ones =  [graph constantWithScalar:1.0 dataType:mps_b.dataType];
				MPSGraphTensor* b_minus = [graph subtractionWithPrimaryTensor:ones secondaryTensor:mps_b name:nil];
				MPSGraphTensor* multiply = [graph multiplicationWithPrimaryTensor:b_minus secondaryTensor:mps_b name:nil];
				MPSGraphTensor* mps_h = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:multiply name:nil];
				[resultTensors addObject:mps_h];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data[] = {data_g, data_b};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGMOID_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sigmoid_back;
}
