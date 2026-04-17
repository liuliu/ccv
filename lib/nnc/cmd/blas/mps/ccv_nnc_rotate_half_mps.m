#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_rotate_half_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(nd == ccv_nnc_tensor_nd(b->info.dim));
	assert(a->info.dim[nd - 1] == b->info.dim[nd - 1]);
	assert((a->info.dim[nd - 1] % 2) == 0);
	@autoreleasepool {
		bool use_mfa = true;
		const char* fallback_reason = 0;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			use_mfa = false;
			fallback_reason = "Disabled.";
		}
		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa)
		{
			if (a->info.datatype != b->info.datatype)
			{
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}
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
				default:
					use_mfa = false;
					fallback_reason = "Unsupported data type.";
					break;
			}
		}
		const size_t count = ccv_nnc_tensor_count(b->info);
		if (use_mfa)
		{
			if (ccv_nnc_tensor_count(a->info) != count)
			{
				use_mfa = false;
				fallback_reason = "Mismatched tensor count.";
			}
		}
		if (use_mfa)
		{
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa)
		{
			(void)fallback_reason;
			const uint32_t dim = (uint32_t)a->info.dim[nd - 1];
			ccv_nnc_mfa_rotate_half_params_t params = {
				.data_type = mtl_data_type,
				.row_count = (uint32_t)(count / dim),
				.dim = dim,
			};
			ccv_nnc_mfa_prepare_rotate_half(context, params);
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[3] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[2] = {
				a->dataof,
				b->dataof,
			};
			ccv_nnc_mfa_encode_rotate_half(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		(void)fallback_reason;
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_a numSplits:2 axis:(nd - 1) name:nil];
			MPSGraphTensor* mps_b = [graph concatTensor:mps_a_splits[1] withTensor:mps_a_splits[0] dimension:(nd - 1) name:nil];
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data[] = { data_a };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rotate_half_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return _ccv_nnc_rotate_half_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ROTATE_HALF_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rotate_half_back;
}
