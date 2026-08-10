#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/mps/ccv_nnc_mps.h"

static uint64_t _ccv_nnc_segmented_swiglu_mtl_datatype(const int datatype)
{
	switch (datatype) {
		case CCV_16F:
			return 16;
		case CCV_16BF:
			return 121;
		case CCV_32F:
			return 3;
		default:
			return UINT64_MAX;
	}
}

static int _ccv_nnc_segmented_swiglu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const indices = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const counts = (const ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const gate_w = (const ccv_nnc_tensor_view_t*)inputs[3];
	const ccv_nnc_tensor_view_t* const up_w = (const ccv_nnc_tensor_view_t*)inputs[4];
	const ccv_nnc_tensor_view_t* const route_weight = (const ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const output = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int w_nd = ccv_nnc_tensor_nd(gate_w->info.dim);
	const int up_w_nd = ccv_nnc_tensor_nd(up_w->info.dim);
	const int output_nd = ccv_nnc_tensor_nd(output->info.dim);
	if (a_nd < 2 || w_nd < 3 || up_w_nd < 3 || output_nd < 2)
		return CCV_NNC_EXEC_INVALID;
	const int k_dim = a->info.dim[a_nd - 1];
	const int n_dim = gate_w->info.dim[w_nd - 2];
	if (k_dim <= 0 || n_dim <= 0)
		return CCV_NNC_EXEC_INVALID;
	const uint32_t input_rows = (uint32_t)(ccv_nnc_tensor_count(a->info) / k_dim);
	const uint32_t M = (uint32_t)ccv_nnc_tensor_count(route_weight->info);
	const uint32_t K = (uint32_t)k_dim;
	const uint32_t N = (uint32_t)n_dim;
	const uint32_t broadcast_input = input_rows == 1 && M > 1;
	const int input_shape_supported = input_rows == M || input_rows == 1;
	const uint32_t expert_count = (uint32_t)(ccv_nnc_tensor_count(gate_w->info) / ((size_t)N * K));
	const uint32_t bincount = (uint32_t)ccv_nnc_tensor_count(indices->info);
	const int gate_subtype = gate_w->info.datatype & 0xf00;
	const int up_subtype = up_w->info.datatype & 0xf00;
	const int gate_rowwise = gate_subtype == CCV_NNC_QX_8I_ROWWISE || gate_subtype == CCV_NNC_QX_8I_ROWWISE_X;
	const int up_rowwise = up_subtype == CCV_NNC_QX_8I_ROWWISE || up_subtype == CCV_NNC_QX_8I_ROWWISE_X;
	const uint32_t gate_format = gate_subtype == CCV_NNC_QX_8I_ROWWISE_X ? (uint32_t)gate_w->info.reserved : 0;
	const uint32_t up_format = up_subtype == CCV_NNC_QX_8I_ROWWISE_X ? (uint32_t)up_w->info.reserved : 0;
	const int rowwise_shape_supported = gate_format == 0 ? (K % 4) == 0 : (K % 256) == 0 && (N % 256) == 0;
	const uint64_t mtl_datatype = _ccv_nnc_segmented_swiglu_mtl_datatype(a->info.datatype);
	ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
	const int direct_decode =
		input_shape_supported && M == bincount && M > 0 && N > 0 && K > 0 && expert_count > 0 &&
		gate_rowwise && up_rowwise && gate_format == up_format && rowwise_shape_supported &&
		gate_w->info.datatype == up_w->info.datatype &&
		((gate_w->info.datatype & 0xff) << 12) == a->info.datatype &&
		gate_w->info.dim[w_nd - 1] == (int)K &&
		ccv_nnc_tensor_count(gate_w->info) == (size_t)expert_count * N * K &&
		up_w->info.dim[up_w_nd - 2] == (int)N && up_w->info.dim[up_w_nd - 1] == (int)K &&
		ccv_nnc_tensor_count(up_w->info) == (size_t)expert_count * N * K &&
		ccv_nnc_tensor_count(counts->info) == bincount &&
		ccv_nnc_tensor_count(route_weight->info) == M &&
		indices->info.datatype == CCV_32S && counts->info.datatype == CCV_32S &&
		output->info.dim[output_nd - 1] == (int)N && output->info.datatype == a->info.datatype &&
		ccv_nnc_tensor_count(output->info) == (size_t)M * N &&
		route_weight->info.datatype == a->info.datatype && mtl_datatype != UINT64_MAX &&
		CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(indices) &&
		CCV_IS_TENSOR_CONTIGUOUS(counts) && CCV_IS_TENSOR_CONTIGUOUS(gate_w) &&
		CCV_IS_TENSOR_CONTIGUOUS(up_w) && CCV_IS_TENSOR_CONTIGUOUS(route_weight) &&
		CCV_IS_TENSOR_CONTIGUOUS(output) && ccv_nnc_mfa_context_supported(context) &&
		!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA);
	if (direct_decode)
	{
		if (gate_format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
			gate_format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
			gate_format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS ||
			gate_format == CCV_NNC_QX_8I_ROWWISE_Q2_K)
		{
			const ccv_nnc_mfa_segmented_int8_swiglu_params_t params = {
				.data_type = mtl_datatype,
				.format = gate_format,
				.M = M,
				.N = N,
				.K = K,
				.expert_count = expert_count,
				.bincount = bincount,
				.broadcast_input = broadcast_input,
				.clamp = cmd.info.segmented_swiglu.clamp,
			};
			ccv_nnc_mfa_prepare_segmented_int8_swiglu(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[8] = {
				mpgetbuffer((ccv_nnc_tensor_t*)gate_w),
				mpgetbuffer((ccv_nnc_tensor_t*)up_w),
				mpgetbuffer((ccv_nnc_tensor_t*)a),
				mpgetbuffer((ccv_nnc_tensor_t*)indices),
				mpgetbuffer((ccv_nnc_tensor_t*)counts),
				mpgetbuffer((ccv_nnc_tensor_t*)route_weight),
				mpgetbuffer((ccv_nnc_tensor_t*)output),
				NULL,
			};
			size_t tensor_offsets[7] = {
				gate_w->dataof, up_w->dataof, a->dataof, indices->dataof,
				counts->dataof, route_weight->dataof, output->dataof,
			};
			ccv_nnc_mfa_encode_segmented_int8_swiglu(
				context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		const size_t intermediate_size = (size_t)M * N * CCV_GET_DATA_TYPE_SIZE(a->info.datatype);
		mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(context, intermediate_size);
		const ccv_nnc_mfa_segmented_int8_gemv_params_t gemv_params = {
			.data_type = mtl_datatype,
			.format = gate_format,
			.M = M,
			.N = N,
			.K = K,
			.expert_count = expert_count,
			.bincount = bincount,
			.broadcast_input = broadcast_input,
		};
		const ccv_nnc_mfa_swish_mul_params_t swish_params = {
			.beta = 1,
			.scale = 1,
			.clamp = cmd.info.segmented_swiglu.clamp,
			.a_data_type = mtl_datatype,
			.b_data_type = mtl_datatype,
			.weight_data_type = mtl_datatype,
			.length = M * N,
			.weight_count = M,
			.weighted = 1,
			.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
		};
		ccv_nnc_mfa_prepare_segmented_int8_gemv(context, gemv_params);
		ccv_nnc_mfa_prepare_swish_mul(context, swish_params);
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		mtl_buffer_t* gate_tensors[6] = {
			mpgetbuffer((ccv_nnc_tensor_t*)a), mpgetbuffer((ccv_nnc_tensor_t*)indices),
			mpgetbuffer((ccv_nnc_tensor_t*)counts), mpgetbuffer((ccv_nnc_tensor_t*)gate_w),
			scratch, NULL,
		};
		size_t gate_offsets[5] = { a->dataof, indices->dataof, counts->dataof, gate_w->dataof, 0 };
		ccv_nnc_mfa_encode_segmented_int8_gemv(context, gemv_params, command_batch, gate_tensors, gate_offsets);
		mtl_buffer_t* up_tensors[6] = {
			mpgetbuffer((ccv_nnc_tensor_t*)a), mpgetbuffer((ccv_nnc_tensor_t*)indices),
			mpgetbuffer((ccv_nnc_tensor_t*)counts), mpgetbuffer((ccv_nnc_tensor_t*)up_w),
			mpgetbuffer((ccv_nnc_tensor_t*)output), NULL,
		};
		size_t up_offsets[5] = { a->dataof, indices->dataof, counts->dataof, up_w->dataof, output->dataof };
		ccv_nnc_mfa_encode_segmented_int8_gemv(context, gemv_params, command_batch, up_tensors, up_offsets);
		mtl_buffer_t* swish_tensors[5] = {
			mpgetbuffer((ccv_nnc_tensor_t*)output), scratch, mpgetbuffer((ccv_nnc_tensor_t*)route_weight),
			mpgetbuffer((ccv_nnc_tensor_t*)output), NULL,
		};
		size_t swish_offsets[4] = { output->dataof, 0, route_weight->dataof, output->dataof };
		ccv_nnc_mfa_encode_swish_mul(context, swish_params, command_batch, swish_tensors, swish_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		return CCV_NNC_EXEC_SUCCESS;
	}

	const int grouped_supported =
		input_rows == M && M > 0 && N > 0 && K > 0 && expert_count > 0 && bincount > 0 &&
		gate_w->info.datatype == up_w->info.datatype && gate_format == up_format &&
		gate_w->info.dim[w_nd - 1] == (int)K &&
		ccv_nnc_tensor_count(gate_w->info) == (size_t)expert_count * N * K &&
		up_w->info.dim[up_w_nd - 2] == (int)N && up_w->info.dim[up_w_nd - 1] == (int)K &&
		ccv_nnc_tensor_count(up_w->info) == (size_t)expert_count * N * K &&
		ccv_nnc_tensor_count(counts->info) == bincount &&
		ccv_nnc_tensor_count(route_weight->info) == M &&
		indices->info.datatype == CCV_32S && counts->info.datatype == CCV_32S &&
		output->info.dim[output_nd - 1] == (int)N &&
		ccv_nnc_tensor_count(output->info) == (size_t)M * N &&
		output->info.datatype == a->info.datatype && route_weight->info.datatype == a->info.datatype &&
		mtl_datatype != UINT64_MAX && CCV_IS_TENSOR_CONTIGUOUS(a) &&
		CCV_IS_TENSOR_CONTIGUOUS(indices) && CCV_IS_TENSOR_CONTIGUOUS(counts) &&
		CCV_IS_TENSOR_CONTIGUOUS(gate_w) && CCV_IS_TENSOR_CONTIGUOUS(up_w) &&
		CCV_IS_TENSOR_CONTIGUOUS(route_weight) && CCV_IS_TENSOR_CONTIGUOUS(output) &&
		ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA);
	if (!grouped_supported)
		return CCV_NNC_EXEC_INVALID;
	const int dense_weights =
		CCV_GET_DATA_TYPE(gate_w->info.datatype) != CCV_QX && gate_w->info.datatype == a->info.datatype;
	const int rowwise_weights =
		gate_rowwise && up_rowwise &&
		((gate_w->info.datatype & 0xff) << 12) == a->info.datatype;
	if (!dense_weights && !rowwise_weights)
		return CCV_NNC_EXEC_INVALID;
	const int use_neural_accelerators =
		!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) &&
		ccv_nnc_mfa_has_neural_accelerators(context) &&
		(mtl_datatype != 121 || ccv_nnc_mfa_neural_accelerators_support_bfloat(context));
	const ccv_nnc_mfa_swish_mul_params_t swish_params = {
		.beta = 1,
		.scale = 1,
		.clamp = cmd.info.segmented_swiglu.clamp,
		.a_data_type = mtl_datatype,
		.b_data_type = mtl_datatype,
		.weight_data_type = mtl_datatype,
		.length = M * N,
		.weight_count = M,
		.weighted = 1,
		.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
	};
	ccv_nnc_mfa_prepare_swish_mul(context, swish_params);
	const size_t intermediate_size = (size_t)M * N * CCV_GET_DATA_TYPE_SIZE(a->info.datatype);
	const ccv_nnc_tensor_view_t* const weights[2] = { gate_w, up_w };
	if (rowwise_weights && use_neural_accelerators)
	{
		const ccv_nnc_mfa_segmented_scaled_gemm_params_t gemm_params = {
			.data_type = mtl_datatype,
			.M = (uint32_t)(M + ccv_max((int)bincount - 2, 0)) / ccv_max((int)bincount - 1, 1),
			.N = N,
			.K = K,
			.originalM = M,
			.fused_bias = 0,
			.use_neural_accelerators = 1,
			.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			.expert_count = expert_count,
			.bincount = bincount,
		};
		ccv_nnc_mfa_prepare_segmented_scaled_gemm(context, gemm_params);
		size_t scratch_offset = ccv_nnc_mfa_segmented_scaled_gemm_reserved_scratch_size(gemm_params);
		ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t decode_params = {};
		size_t decoded_weight_size = 0;
		if (gate_format)
		{
			decode_params = (ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t){
				.data_type = mtl_datatype,
				.format = gate_format,
				.row_length = K,
				.rows_per_expert = N,
				.expert_count = expert_count,
				.bincount = bincount,
			};
			scratch_offset = ccv_max(scratch_offset,
				ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(decode_params));
			const ccv_nnc_tensor_param_t dense_weight_params = {
				.type = CCV_TENSOR_GPU_MEMORY,
				.format = CCV_TENSOR_FORMAT_NHWC,
				.datatype = a->info.datatype,
				.dim = { (int)(expert_count * N), (int)K, 0 },
			};
			decoded_weight_size = ccv_nnc_tensor_data_size_without_padding(
				ccv_nnc_tensor_8i_rowwise(dense_weight_params));
		}
		const size_t decoded_weight_offset = scratch_offset;
		const size_t intermediate_offset = decoded_weight_offset + decoded_weight_size;
		mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(
			context, intermediate_offset + intermediate_size);
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		int projection;
		for (projection = 0; projection < 2; projection++)
		{
			mtl_buffer_t* weight_data = mpgetbuffer((ccv_nnc_tensor_t*)weights[projection]);
			size_t weight_dataof = weights[projection]->dataof;
			if (gate_format)
			{
				mtl_buffer_t* decode_tensors[6] = {
					weight_data, mpgetbuffer((ccv_nnc_tensor_t*)indices),
					mpgetbuffer((ccv_nnc_tensor_t*)counts), scratch, scratch, NULL,
				};
				size_t decode_offsets[5] = {
					weight_dataof, indices->dataof, counts->dataof, decoded_weight_offset, 0,
				};
				ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(
					context, decode_params, command_batch, decode_tensors, decode_offsets);
				weight_data = scratch;
				weight_dataof = decoded_weight_offset;
			}
			mtl_buffer_t* const destination = projection == 0 ? scratch : mpgetbuffer((ccv_nnc_tensor_t*)output);
			const size_t destination_offset = projection == 0 ? intermediate_offset : output->dataof;
			mtl_buffer_t* gemm_tensors[6] = {
				mpgetbuffer((ccv_nnc_tensor_t*)a), mpgetbuffer((ccv_nnc_tensor_t*)indices),
				mpgetbuffer((ccv_nnc_tensor_t*)counts), weight_data, destination, NULL,
			};
			size_t gemm_offsets[5] = {
				a->dataof, indices->dataof, counts->dataof, weight_dataof, destination_offset,
			};
			ccv_nnc_mfa_encode_segmented_scaled_gemm(
				context, gemm_params, command_batch, gemm_tensors, gemm_offsets);
		}
		mtl_buffer_t* swish_tensors[5] = {
			mpgetbuffer((ccv_nnc_tensor_t*)output), scratch,
			mpgetbuffer((ccv_nnc_tensor_t*)route_weight), mpgetbuffer((ccv_nnc_tensor_t*)output), NULL,
		};
		size_t swish_offsets[4] = { output->dataof, intermediate_offset, route_weight->dataof, output->dataof };
		ccv_nnc_mfa_encode_swish_mul(context, swish_params, command_batch, swish_tensors, swish_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		return CCV_NNC_EXEC_SUCCESS;
	}
	const ccv_nnc_mfa_segmented_gemm_params_t gemm_params = {
		.data_type = mtl_datatype,
		.M = (uint32_t)(M + ccv_max((int)bincount - 2, 0)) / ccv_max((int)bincount - 1, 1),
		.N = N,
		.K = K,
		.originalM = M,
		.A_trans = 0,
		.B_trans = 1,
		.D_trans = 0,
		.fused_bias = 0,
		.register_float = 1,
		.use_neural_accelerators = use_neural_accelerators,
		.expert_count = expert_count,
		.bincount = bincount,
	};
	const size_t scratch_offset = ccv_nnc_mfa_segmented_gemm_reserved_scratch_size(gemm_params);
	const size_t decoded_weight_size = rowwise_weights ?
		(size_t)expert_count * N * K * CCV_GET_DATA_TYPE_SIZE(a->info.datatype) : 0;
	const size_t decoded_weight_offset = scratch_offset;
	const size_t intermediate_offset = decoded_weight_offset + decoded_weight_size;
	mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(context, intermediate_offset + intermediate_size);
	ccv_nnc_mfa_dequantize_8i_rowwise_params_t rowwise_decode_params = {};
	ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t rowwise_x_decode_params = {};
	if (rowwise_weights)
	{
		if (gate_format)
		{
			rowwise_x_decode_params = (ccv_nnc_mfa_dequantize_8i_rowwise_x_fp_params_t){
				.data_type = mtl_datatype,
				.format = gate_format,
				.row_length = K,
				.length = (uint64_t)expert_count * N * K,
			};
			ccv_nnc_mfa_prepare_dequantize_8i_rowwise_x_fp(context, rowwise_x_decode_params);
		} else {
			rowwise_decode_params = (ccv_nnc_mfa_dequantize_8i_rowwise_params_t){
				.data_type = mtl_datatype,
				.row_length = K,
				.length = (uint64_t)expert_count * N * K,
			};
			ccv_nnc_mfa_prepare_dequantize_8i_rowwise(context, rowwise_decode_params);
		}
	}
	mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
	int projection;
	for (projection = 0; projection < 2; projection++)
	{
		mtl_buffer_t* weight_data = mpgetbuffer((ccv_nnc_tensor_t*)weights[projection]);
		size_t weight_dataof = weights[projection]->dataof;
		if (rowwise_weights)
		{
			mtl_buffer_t* decode_tensors[3] = { weight_data, scratch, NULL };
			size_t decode_offsets[2] = { weight_dataof, decoded_weight_offset };
			if (gate_format)
				ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_fp(
					context, rowwise_x_decode_params, command_batch, decode_tensors, decode_offsets);
			else
				ccv_nnc_mfa_encode_dequantize_8i_rowwise(
					context, rowwise_decode_params, command_batch, decode_tensors, decode_offsets);
			weight_data = scratch;
			weight_dataof = decoded_weight_offset;
		}
		mtl_buffer_t* const destination = projection == 0 ? scratch : mpgetbuffer((ccv_nnc_tensor_t*)output);
		const size_t destination_offset = projection == 0 ? intermediate_offset : output->dataof;
		mtl_buffer_t* gemm_tensors[6] = {
			mpgetbuffer((ccv_nnc_tensor_t*)a), mpgetbuffer((ccv_nnc_tensor_t*)indices),
			mpgetbuffer((ccv_nnc_tensor_t*)counts), weight_data, destination, NULL,
		};
		size_t gemm_offsets[5] = {
			a->dataof, indices->dataof, counts->dataof, weight_dataof, destination_offset,
		};
		ccv_nnc_mfa_encode_segmented_gemm(context, gemm_params, command_batch, gemm_tensors, gemm_offsets);
	}
	mtl_buffer_t* swish_tensors[5] = {
		mpgetbuffer((ccv_nnc_tensor_t*)output), scratch,
		mpgetbuffer((ccv_nnc_tensor_t*)route_weight), mpgetbuffer((ccv_nnc_tensor_t*)output), NULL,
	};
	size_t swish_offsets[4] = { output->dataof, intermediate_offset, route_weight->dataof, output->dataof };
	ccv_nnc_mfa_encode_swish_mul(context, swish_params, command_batch, swish_tensors, swish_offsets);
	ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_segmented_swiglu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_32S | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_swiglu_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SEGMENTED_SWIGLU_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_QX | CCV_32S | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_segmented_swiglu_back;
}
