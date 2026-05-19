#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static inline int _ccv_nnc_walsh_hadamard_transform_is_power_of_two(const int x)
{
	return x > 0 && (x & (x - 1)) == 0;
}

static int _ccv_nnc_walsh_hadamard_transform_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(a_nd == b_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		assert(a->info.dim[i] == b->info.dim[i]);
	const int dim = a->info.dim[a_nd - 1];
	assert(_ccv_nnc_walsh_hadamard_transform_is_power_of_two(dim));
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;
		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa)
		{
			if (a->info.datatype != b->info.datatype)
				use_mfa = false;
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
					break;
			}
		}
		const size_t count = ccv_nnc_tensor_count(b->info);
		if (use_mfa)
		{
			if (ccv_nnc_tensor_count(a->info) != count)
				use_mfa = false;
		}
		if (use_mfa)
		{
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b))
				use_mfa = false;
		}
		if (use_mfa)
		{
			if (dim > 8192)
				use_mfa = false;
		}
		if (use_mfa)
		{
			ccv_nnc_mfa_walsh_hadamard_transform_params_t params = {
				.data_type = mtl_data_type,
				.row_count = (uint32_t)(count / dim),
				.dim = (uint32_t)dim,
				.scale = cmd.info.walsh_hadamard_transform.scale,
			};
			ccv_nnc_mfa_prepare_walsh_hadamard_transform(context, params);
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
			ccv_nnc_mfa_encode_walsh_hadamard_transform(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		return CCV_NNC_EXEC_INVALID;
	}
	return CCV_NNC_EXEC_INVALID;
}

static int _ccv_nnc_walsh_hadamard_transform_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return _ccv_nnc_walsh_hadamard_transform_forw(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_WALSH_HADAMARD_TRANSFORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_walsh_hadamard_transform_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_WALSH_HADAMARD_TRANSFORM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_walsh_hadamard_transform_back;
}
