#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_fill_if_less_than_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 3 || output_size != 1)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const selector = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const threshold = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	if (!a || !selector || !threshold || !b ||
		a->info.datatype != selector->info.datatype ||
		a->info.datatype != threshold->info.datatype ||
		a->info.datatype != b->info.datatype ||
		ccv_nnc_tensor_nd(a->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(selector->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(threshold->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(b->info.dim) > CCV_NNC_MAX_DIM + 2)
		return CCV_NNC_EXEC_INVALID;
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	if (!ccv_nnc_tensor_view_check_dim(b, adim) ||
		!ccv_nnc_tensor_view_check_broadcast_dim(selector, adim) ||
		!ccv_nnc_tensor_view_check_broadcast_dim(threshold, adim))
		return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		const size_t count = ccv_nnc_tensor_count(a->info);
		uint32_t mtl_data_type = UINT32_MAX;
		switch (a->info.datatype) {
			case CCV_32F:
				mtl_data_type = 3;
				break;
			case CCV_16F:
				mtl_data_type = 16;
				break;
			case CCV_16BF:
				mtl_data_type = 121;
				break;
		}
		const int use_mfa = ccv_nnc_mfa_context_supported(context) &&
			!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA) &&
			mtl_data_type != UINT32_MAX && count > 0 && count <= UINT32_MAX &&
			ccv_nnc_tensor_view_check_dim(selector, adim) &&
			ccv_nnc_tensor_nd(threshold->info.dim) == 1 && threshold->info.dim[0] == 1 &&
			CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(selector) &&
			CCV_IS_TENSOR_CONTIGUOUS(threshold) && CCV_IS_TENSOR_CONTIGUOUS(b);
		if (use_mfa)
		{
			const ccv_nnc_mfa_fill_if_less_than_params_t params = {
				.data_type = mtl_data_type,
				.length = (uint32_t)count,
				.fill = cmd.info.fill_if_less_than.value,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_fill_if_less_than(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[1]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(outputs[0]),
				0,
			};
			size_t tensor_offsets[] = {
				a->dataof,
				selector->dataof,
				threshold->dataof,
				b->dataof,
			};
			ccv_nnc_mfa_encode_fill_if_less_than(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* const mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride)];

			MPSGraphTensor* mps_input_selector;
			MPSGraphTensor* const mps_selector = ccv_nnc_mps_graph_tensor_input(graph, selector, selector->info.dim, selector->stride, &mps_input_selector);
			[inputTensors addObject:mps_input_selector];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(selector, selector->info.dim, selector->stride)];

			MPSGraphTensor* mps_input_threshold;
			MPSGraphTensor* const mps_threshold = ccv_nnc_mps_graph_tensor_input(graph, threshold, threshold->info.dim, threshold->stride, &mps_input_threshold);
			[inputTensors addObject:mps_input_threshold];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(threshold, threshold->info.dim, threshold->stride)];

			MPSGraphTensor* const predicate = [graph lessThanWithPrimaryTensor:mps_selector secondaryTensor:mps_threshold name:nil];
			MPSGraphTensor* const fill = [graph constantWithScalar:cmd.info.fill_if_less_than.value dataType:[mps_a dataType]];
			[resultTensors addObject:[graph selectWithPredicateTensor:predicate truePredicateTensor:fill falsePredicateTensor:mps_a name:nil]];
		});
		MPSGraphTensorData* const data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* const data_selector = ccv_nnc_mps_graph_tensor_data(selector, selector->info.dim, selector->stride);
		MPSGraphTensorData* const data_threshold = ccv_nnc_mps_graph_tensor_data(threshold, threshold->info.dim, threshold->stride);
		MPSGraphTensorData* const data[] = { data_a, data_selector, data_threshold };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_fill_if_less_than_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size < 4 || output_size < 1)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const selector = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const threshold = (const ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	if (!g || !selector || !threshold || !h ||
		g->info.datatype != selector->info.datatype ||
		g->info.datatype != threshold->info.datatype ||
		g->info.datatype != h->info.datatype ||
		ccv_nnc_tensor_nd(g->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(selector->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(threshold->info.dim) > CCV_NNC_MAX_DIM + 2 ||
		ccv_nnc_tensor_nd(h->info.dim) > CCV_NNC_MAX_DIM + 2)
		return CCV_NNC_EXEC_INVALID;
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	if (!ccv_nnc_tensor_view_check_dim(h, gdim) ||
		!ccv_nnc_tensor_view_check_broadcast_dim(selector, gdim) ||
		!ccv_nnc_tensor_view_check_broadcast_dim(threshold, gdim))
		return CCV_NNC_EXEC_INVALID;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		const size_t count = ccv_nnc_tensor_count(g->info);
		uint32_t mtl_data_type = UINT32_MAX;
		switch (g->info.datatype) {
			case CCV_32F:
				mtl_data_type = 3;
				break;
			case CCV_16F:
				mtl_data_type = 16;
				break;
			case CCV_16BF:
				mtl_data_type = 121;
				break;
		}
		const int use_mfa = ccv_nnc_mfa_context_supported(context) &&
			!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA) &&
			mtl_data_type != UINT32_MAX && count > 0 && count <= UINT32_MAX &&
			ccv_nnc_tensor_view_check_dim(selector, gdim) &&
			ccv_nnc_tensor_nd(threshold->info.dim) == 1 && threshold->info.dim[0] == 1 &&
			CCV_IS_TENSOR_CONTIGUOUS(g) && CCV_IS_TENSOR_CONTIGUOUS(selector) &&
			CCV_IS_TENSOR_CONTIGUOUS(threshold) && CCV_IS_TENSOR_CONTIGUOUS(h);
		if (use_mfa)
		{
			const ccv_nnc_mfa_fill_if_less_than_params_t params = {
				.data_type = mtl_data_type,
				.length = (uint32_t)count,
				.fill = 0,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_fill_if_less_than(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(inputs[3]),
				mpgetbuffer(outputs[0]),
				0,
			};
			size_t tensor_offsets[] = {
				g->dataof,
				selector->dataof,
				threshold->dataof,
				h->dataof,
			};
			ccv_nnc_mfa_encode_fill_if_less_than(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* const mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride)];

			MPSGraphTensor* mps_input_selector;
			MPSGraphTensor* const mps_selector = ccv_nnc_mps_graph_tensor_input(graph, selector, selector->info.dim, selector->stride, &mps_input_selector);
			[inputTensors addObject:mps_input_selector];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(selector, selector->info.dim, selector->stride)];

			MPSGraphTensor* mps_input_threshold;
			MPSGraphTensor* const mps_threshold = ccv_nnc_mps_graph_tensor_input(graph, threshold, threshold->info.dim, threshold->stride, &mps_input_threshold);
			[inputTensors addObject:mps_input_threshold];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(threshold, threshold->info.dim, threshold->stride)];

			MPSGraphTensor* const predicate = [graph lessThanWithPrimaryTensor:mps_selector secondaryTensor:mps_threshold name:nil];
			MPSGraphTensor* const zero = [graph constantWithScalar:0 dataType:[mps_g dataType]];
			[resultTensors addObject:[graph selectWithPredicateTensor:predicate truePredicateTensor:zero falsePredicateTensor:mps_g name:nil]];
		});
		MPSGraphTensorData* const data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* const data_selector = ccv_nnc_mps_graph_tensor_data(selector, selector->info.dim, selector->stride);
		MPSGraphTensorData* const data_threshold = ccv_nnc_mps_graph_tensor_data(threshold, threshold->info.dim, threshold->stride);
		MPSGraphTensorData* const data[] = { data_g, data_selector, data_threshold };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FILL_IF_LESS_THAN_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_fill_if_less_than_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_FILL_IF_LESS_THAN_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_fill_if_less_than_back;
}
