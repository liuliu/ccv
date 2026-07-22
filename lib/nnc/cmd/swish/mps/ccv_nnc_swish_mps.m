#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_swish_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const float beta = cmd.info.swish.beta;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
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
			ccv_nnc_mfa_swish_params_t params = {
				.gradient = 0,
				.beta = beta,
				.data_type = mtl_data_type,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			const size_t count = ccv_nnc_tensor_count(a->info);
			params.length = count;
			ccv_nnc_mfa_prepare_swish(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer(inputs[0]), // source
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[2] = {
				a->dataof,
				b->dataof
			};
			ccv_nnc_mfa_encode_swish(context, params, command_batch, tensors, tensor_offsets);
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
				MPSGraphTensor* mps_b;
				if (beta == 1)
				{
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_a name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					mps_b = [graph divisionWithPrimaryTensor:mps_a secondaryTensor:mps_denom name:nil];
				} else {
					MPSGraphTensor* mps_a_f32 = mps_a;
					if (a->info.datatype != CCV_32F)
						mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
					MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_beta_a = [graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_beta name:nil];
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_a name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					MPSGraphTensor* mps_b_f32 = [graph divisionWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_denom name:nil];
					if (b->info.datatype != CCV_32F)
						mps_b_f32 = [graph castTensor:mps_b_f32 toType:ccv_nnc_mps_datatype(b->info.datatype) name:@"mps_b"];
					mps_b = mps_b_f32;
				}
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_swish_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const float beta = cmd.info.swish.beta;
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0]; // gradient
	assert(CCV_IS_TENSOR_CONTIGUOUS(g));

	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));

	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(h));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && g->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == g->info.dim[i]);
		assert(g->info.dim[i] == h->info.dim[i]);
	}

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
				(g->info.datatype == a->info.datatype) &&
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
					!CCV_IS_TENSOR_CONTIGUOUS(a) ||
					!CCV_IS_TENSOR_CONTIGUOUS(h))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_swish_params_t params = {
				.gradient = 1,
				.beta = beta,
				.data_type = mtl_data_type,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			const size_t count = ccv_nnc_tensor_count(g->info);
			params.length = count;
			ccv_nnc_mfa_prepare_swish(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]), // gradient
				mpgetbuffer(inputs[1]), // source
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[3] = {
				g->dataof,
				a->dataof,
				h->dataof
			};
			ccv_nnc_mfa_encode_swish(context, params, command_batch, tensors, tensor_offsets);
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

				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];

				MPSGraphTensor* mps_h;
				if (beta == 1)
				{
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_a name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					MPSGraphTensor* mps_y = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
					MPSGraphTensor* mps_y_2 = [graph multiplicationWithPrimaryTensor:mps_y secondaryTensor:mps_y name:nil];
					MPSGraphTensor* mps_y_diff = [graph subtractionWithPrimaryTensor:mps_y secondaryTensor:mps_y_2 name:nil];
					MPSGraphTensor* mps_multiply = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_y_diff name:nil];
					MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_y secondaryTensor:mps_multiply name:nil];
					mps_h = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_sum name:nil];
				} else {
					MPSGraphTensor* mps_a_f32 = mps_a;
					if (a->info.datatype != CCV_32F)
						mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
					MPSGraphTensor* mps_g_f32 = mps_g;
					if (g->info.datatype != CCV_32F)
						mps_g_f32 = [graph castTensor:mps_g toType:MPSDataTypeFloat32 name:@"mps_g_float"];
					MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_beta_a = [graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_beta name:nil];
					MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_a name:nil];
					MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
					MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
					MPSGraphTensor* mps_y = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
					MPSGraphTensor* mps_y_2 = [graph multiplicationWithPrimaryTensor:mps_y secondaryTensor:mps_y name:nil];
					MPSGraphTensor* mps_y_diff = [graph subtractionWithPrimaryTensor:mps_y secondaryTensor:mps_y_2 name:nil];
					MPSGraphTensor* mps_beta_x = [graph multiplicationWithPrimaryTensor:mps_beta secondaryTensor:mps_a_f32 name:nil];
					MPSGraphTensor* mps_multiply = [graph multiplicationWithPrimaryTensor:mps_beta_x secondaryTensor:mps_y_diff name:nil];
					MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_y secondaryTensor:mps_multiply name:nil];
					MPSGraphTensor* mps_h_f32 = [graph multiplicationWithPrimaryTensor:mps_g_f32 secondaryTensor:mps_sum name:nil];
					if (h->info.datatype != CCV_32F)
						mps_h_f32 = [graph castTensor:mps_h_f32 toType:ccv_nnc_mps_datatype(h->info.datatype) name:@"mps_h"];
					mps_h = mps_h_f32;
				}

				[resultTensors addObject:mps_h];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data[] = {data_g, data_a};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_back;
}
