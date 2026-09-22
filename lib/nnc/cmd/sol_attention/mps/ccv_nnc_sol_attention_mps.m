#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/mps/ccv_nnc_mps.h"
#include <math.h>
#include "../../_ccv_nnc_mps.h"

static int _ccv_nnc_sol_attention_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert((input_size == 3 || input_size == 4) && output_size == 1);
	int is_dense_attention = 0;
	if (input_size == 4)
	{
		if (CCV_TENSOR_GET_MEMORY(inputs[3]->info.type) != CCV_TENSOR_CPU_MEMORY || inputs[3]->info.datatype != CCV_32S || ccv_nnc_tensor_count(inputs[3]->info) != 1)
			return CCV_NNC_EXEC_INVALID;
		is_dense_attention = inputs[3]->data.i32[0] != 0;
	}
	const int N = inputs[0]->info.dim[0], T = inputs[0]->info.dim[1], H = inputs[0]->info.dim[2];
	const int B = cmd.info.sol_attention.block_size;
	const int QB = cmd.info.sol_attention.query_block_size > 0 ? cmd.info.sol_attention.query_block_size : B;
	const int start = cmd.info.sol_attention.approximation_start, end = cmd.info.sol_attention.approximation_end;
	if (N <= 0 || T <= 0 || H <= 0 || (B != 16 && B != 32 && B != 64) || cmd.info.sol_attention.query_block_size < 0 || cmd.info.sol_attention.local_block_radius < 1 || (QB != 16 && QB != 32 && QB != 64) || QB > B || start < 0 || end < start || end > T || !isfinite(cmd.info.sol_attention.scale) || !isfinite(cmd.info.sol_attention.tau))
		return CCV_NNC_EXEC_INVALID;
	int i;
	for (i = 0; i < 4; i++)
	{
		const ccv_nnc_tensor_t* const tensor = i < 3 ? inputs[i] : outputs[0];
		if (CCV_TENSOR_GET_MEMORY(tensor->info.type) != CCV_TENSOR_GPU_MEMORY || tensor->info.format != CCV_TENSOR_FORMAT_NHWC || tensor->info.datatype != CCV_16F || ccv_nnc_tensor_nd(tensor->info.dim) != 4 || !CCV_IS_TENSOR_CONTIGUOUS(tensor) || tensor->info.dim[0] != N || tensor->info.dim[1] != T || tensor->info.dim[2] != H || tensor->info.dim[3] != 128)
			return CCV_NNC_EXEC_INVALID;
	}
	if (is_dense_attention)
	{
		ccv_nnc_cmd_t dense = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(cmd.info.sol_attention.scale, 0);
		dense.info.scaled_dot_product_attention.flags = cmd.info.sol_attention.flags | CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		return _ccv_nnc_scaled_dot_product_attention_forw_mps(dense, hint, flags, inputs, 3, outputs, output_size, stream_context);
	}
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if ((ccv_nnc_flags() & (CCV_NNC_DISABLE_MFA | CCV_NNC_DISABLE_MFA_ATTENTION)) || !ccv_nnc_mfa_context_supported(context))
			return CCV_NNC_EXEC_INVALID;
		const ccv_nnc_mfa_sol_attention_params_t params = { .N = N, .T = T, .H = H, .block_size = B, .approximation_start = start, .approximation_end = end, .scale = cmd.info.sol_attention.scale, .tau = cmd.info.sol_attention.tau, .query_block_size = QB, .local_block_radius = cmd.info.sol_attention.local_block_radius, .use_neural_accelerators = !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) && ccv_nnc_mfa_has_neural_accelerators(context) };
		mtl_buffer_t* tensors[] = { mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]), mpgetbuffer(outputs[0]) };
		size_t offsets[] = { inputs[0]->dataof, inputs[1]->dataof, inputs[2]->dataof, outputs[0]->dataof };
		mtl_command_batch_t* const batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		ccv_nnc_mfa_encode_sol_attention(context, params, batch, tensors, offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SOL_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_16F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY | CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sol_attention_forw;
}
