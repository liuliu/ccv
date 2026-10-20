#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_tanh_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	if (a->info.dim[0] == 0 || b->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const float soft_cap = cmd.info.tanh.soft_cap > 0 ? cmd.info.tanh.soft_cap : 1;
	const size_t count = ccv_nnc_tensor_count(a->info);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(b) && a->info.datatype == b->info.datatype && count <= UINT32_MAX && ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const ccv_nnc_mfa_tanh_params_t params = {
				.gradient = 0,
				.soft_cap = soft_cap,
				.data_type = a->info.datatype == CCV_32F ? 3 : (a->info.datatype == CCV_16BF ? 121 : 16),
				.length = (uint32_t)count,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_tanh(context, params);
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[2] = {
				a->dataof,
				b->dataof,
			};
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			ccv_nnc_mfa_encode_tanh(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_a_f32 = mps_a;
			if (a->info.datatype != CCV_32F)
				mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
			MPSGraphTensor* mps_b_f32;
			if (soft_cap != 1)
			{
				MPSGraphTensor* mps_soft_cap = [graph constantWithScalar:soft_cap dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_a_capped = [graph divisionWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_soft_cap name:nil];
				MPSGraphTensor* mps_tanh = [graph tanhWithTensor:mps_a_capped name:nil];
				mps_b_f32 = [graph multiplicationWithPrimaryTensor:mps_tanh secondaryTensor:mps_soft_cap name:nil];
			} else
				mps_b_f32 = [graph tanhWithTensor:mps_a_f32 name:nil];
			MPSGraphTensor* mps_b = mps_b_f32;
			if (b->info.datatype != CCV_32F)
				mps_b = [graph castTensor:mps_b_f32 toType:ccv_nnc_mps_datatype(b->info.datatype) name:@"mps_b"];
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_tanh_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (g->info.dim[0] == 0 || b->info.dim[0] == 0 || h->info.dim[0] == 0)
		return CCV_NNC_EXEC_SUCCESS;
	const float soft_cap = cmd.info.tanh.soft_cap > 0 ? cmd.info.tanh.soft_cap : 1;
	const size_t count = ccv_nnc_tensor_count(b->info);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (CCV_IS_TENSOR_CONTIGUOUS(g) && CCV_IS_TENSOR_CONTIGUOUS(b) && CCV_IS_TENSOR_CONTIGUOUS(h) && g->info.datatype == h->info.datatype && b->info.datatype == h->info.datatype && count <= UINT32_MAX && ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const ccv_nnc_mfa_tanh_params_t params = {
				.gradient = 1,
				.soft_cap = soft_cap,
				.data_type = h->info.datatype == CCV_32F ? 3 : (h->info.datatype == CCV_16BF ? 121 : 16),
				.length = (uint32_t)count,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_tanh(context, params);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[3] = {
				g->dataof,
				b->dataof,
				h->dataof,
			};
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			ccv_nnc_mfa_encode_tanh(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
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

			MPSGraphTensor* mps_g_f32 = mps_g;
			if (g->info.datatype != CCV_32F)
				mps_g_f32 = [graph castTensor:mps_g toType:MPSDataTypeFloat32 name:@"mps_g_float"];
			MPSGraphTensor* mps_b_f32 = mps_b;
			if (b->info.datatype != CCV_32F)
				mps_b_f32 = [graph castTensor:mps_b toType:MPSDataTypeFloat32 name:@"mps_b_float"];
			// d(soft_cap * tanh(x / soft_cap)) / dx = 1 - (y / soft_cap)^2.
			if (soft_cap != 1)
			{
				MPSGraphTensor* mps_soft_cap = [graph constantWithScalar:soft_cap dataType:MPSDataTypeFloat32];
				mps_b_f32 = [graph divisionWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_soft_cap name:nil];
			}
			MPSGraphTensor *ones =  [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
			MPSGraphTensor* b_squared = [graph squareWithTensor:mps_b_f32 name:nil];
			MPSGraphTensor* b_squared_minus = [graph subtractionWithPrimaryTensor:ones secondaryTensor:b_squared name:nil];
			MPSGraphTensor* mps_h = [graph multiplicationWithPrimaryTensor:mps_g_f32 secondaryTensor:b_squared_minus name:nil];
			if (h->info.datatype != CCV_32F)
				mps_h = [graph castTensor:mps_h toType:ccv_nnc_mps_datatype(h->info.datatype) name:@"mps_h"];
			[resultTensors addObject:mps_h];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
		MPSGraphTensorData* data[] = {data_g, data_b};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TANH_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_tanh_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_TANH_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_tanh_back;
}
