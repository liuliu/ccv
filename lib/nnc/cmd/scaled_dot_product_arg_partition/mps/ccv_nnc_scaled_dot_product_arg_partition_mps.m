#include "ccv.h"
#include "ccv_internal.h"
#include <float.h>
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif

static int _ccv_nnc_scaled_dot_product_arg_partition_enumerate(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int T, const int C, const int kth, const int compression_ratio, ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_tensor_view_t* const selected = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			if (METAL_LOG_LEVEL(context) >= 3)
				ccv_nnc_mfa_log_message("SDPAP: MFA enumeration.");
			const ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t params = {
				.loadM = (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M) != 0,
				.T = (uint32_t)T,
				.C = (uint32_t)C,
				.kth = (uint32_t)kth,
				.compression_ratio = (uint32_t)compression_ratio,
				.query_offset = cmd.info.scaled_dot_product_arg_partition.query_offset,
				.is_causal = (uint8_t)(cmd.info.scaled_dot_product_arg_partition.is_causal != 0),
			};
			ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition_enumerate(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[2] = {
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[1] = {
				selected->dataof,
			};
			ccv_nnc_mfa_encode_scaled_dot_product_arg_partition_enumerate(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		if (METAL_LOG_LEVEL(context) >= 3)
			ccv_nnc_mfa_log_message("SDPAP: MPSGraph enumeration.");
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, 0, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* input_tensors, NSMutableArray<MPSGraphShapedType*>* input_shapes, NSMutableArray<MPSGraphTensor*>* result_tensors) {
			NSArray<NSNumber*>* const shape = @[@(T), @(kth)];
			MPSGraphTensor* const position = [graph coordinateAlongAxis:1 withShape:shape name:nil];
			MPSGraphTensor* valid;
			if (cmd.info.scaled_dot_product_arg_partition.is_causal)
			{
				MPSGraphTensor* const token = [graph coordinateAlongAxis:0 withShape:shape name:nil];
				MPSGraphTensor* const token_f = [graph castTensor:token toType:MPSDataTypeFloat32 name:nil];
				MPSGraphTensor* const position_f = [graph castTensor:position toType:MPSDataTypeFloat32 name:nil];
				MPSGraphTensor* const query_offset = [graph constantWithScalar:(float)(cmd.info.scaled_dot_product_arg_partition.query_offset + 1) dataType:MPSDataTypeFloat32];
				MPSGraphTensor* const ratio = [graph constantWithScalar:(float)compression_ratio dataType:MPSDataTypeFloat32];
				MPSGraphTensor* visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph additionWithPrimaryTensor:token_f secondaryTensor:query_offset name:nil] secondaryTensor:ratio name:nil] name:nil];
				visible = [graph maximumWithPrimaryTensor:visible secondaryTensor:[graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32] name:nil];
				visible = [graph minimumWithPrimaryTensor:visible secondaryTensor:[graph constantWithScalar:(float)C dataType:MPSDataTypeFloat32] name:nil];
				valid = [graph lessThanWithPrimaryTensor:position_f secondaryTensor:visible name:nil];
			} else {
				MPSGraphTensor* const position_f = [graph castTensor:position toType:MPSDataTypeFloat32 name:nil];
				valid = [graph lessThanWithPrimaryTensor:position_f secondaryTensor:[graph constantWithScalar:(float)C dataType:MPSDataTypeFloat32] name:nil];
			}
			MPSGraphTensor* const position_i32 = [graph castTensor:position toType:MPSDataTypeInt32 name:nil];
			MPSGraphTensor* const minus_one = [graph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];
			[result_tensors addObject:[graph selectWithPredicateTensor:valid truePredicateTensor:position_i32 falsePredicateTensor:minus_one name:nil]];
		});
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[], &selected, (int*[]){ selected->info.dim }, (int*[]){ selected->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_arg_partition_candidates_graph(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int T = inputs[0]->info.dim[0], H = inputs[0]->info.dim[1], D = inputs[0]->info.dim[2], C = inputs[1]->info.dim[0];
	const int kth = cmd.info.scaled_dot_product_arg_partition.kth;
	const int block_size = cmd.info.scaled_dot_product_arg_partition.candidate_block_size;
	const int pool_size = cmd.info.scaled_dot_product_arg_partition.candidate_kth;
	@autoreleasepool {
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[4];
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* input_tensors, NSMutableArray<MPSGraphShapedType*>* input_shapes, NSMutableArray<MPSGraphTensor*>* result_tensors) {
			MPSGraphTensor* tensors[4];
			int i;
			for (i = 0; i < input_size; i++)
			{
				const ccv_nnc_tensor_view_t* const tensor = (const ccv_nnc_tensor_view_t*)inputs[i];
				MPSGraphTensor* input;
				tensors[i] = ccv_nnc_mps_graph_tensor_input(graph, tensor, tensor->info.dim, tensor->stride, &input);
				[input_tensors addObject:input];
				[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(tensor, tensor->info.dim, tensor->stride)];
				if (i < 3 && tensor->info.datatype != CCV_32F)
					tensors[i] = [graph castTensor:tensors[i] toType:MPSDataTypeFloat32 name:nil];
			}
			MPSGraphTensor* const zero = [graph constantWithScalar:0 dataType:MPSDataTypeFloat32];
			MPSGraphTensor* const negative = [graph constantWithScalar:-FLT_MAX dataType:MPSDataTypeFloat32];
			MPSGraphTensor* const minus_one = [graph constantWithScalar:-1 dataType:MPSDataTypeInt32];
			MPSGraphTensor* const width = [graph constantWithScalar:C dataType:MPSDataTypeInt32];
			MPSGraphTensor* const q = [graph reshapeTensor:tensors[0] withShape:@[@(T * H), @(D)] name:nil];
			MPSGraphTensor* const kt = [graph transposeTensor:tensors[1] dimension:0 withDimension:1 name:nil];
			MPSGraphTensor* dot = [graph matrixMultiplicationWithPrimaryTensor:q secondaryTensor:kt name:nil];
			dot = [graph reshapeTensor:dot withShape:@[@(T), @(H), @(C)] name:nil];
			MPSGraphTensor* const weights = [graph reshapeTensor:tensors[2] withShape:@[@(T), @(H), @1] name:nil];
			MPSGraphTensor* const weighted = [graph multiplicationWithPrimaryTensor:[graph maximumWithPrimaryTensor:dot secondaryTensor:zero name:nil] secondaryTensor:weights name:nil];
			MPSGraphTensor* scores = [graph reshapeTensor:[graph reductionSumWithTensor:weighted axis:1 name:nil] withShape:@[@(T), @(C)] name:nil];
			scores = [graph multiplicationWithPrimaryTensor:scores secondaryTensor:[graph constantWithScalar:cmd.info.scaled_dot_product_arg_partition.scale dataType:MPSDataTypeFloat32] name:nil];
			MPSGraphTensor* visible = [graph constantWithScalar:C shape:@[@(T), @1] dataType:MPSDataTypeFloat32];
			if (cmd.info.scaled_dot_product_arg_partition.is_causal)
			{
				MPSGraphTensor* const row = [graph castTensor:[graph coordinateAlongAxis:0 withShape:@[@(T), @1] name:nil] toType:MPSDataTypeFloat32 name:nil];
				MPSGraphTensor* const end = [graph additionWithPrimaryTensor:row secondaryTensor:[graph constantWithScalar:cmd.info.scaled_dot_product_arg_partition.query_offset + 1 dataType:MPSDataTypeFloat32] name:nil];
				visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:end secondaryTensor:[graph constantWithScalar:cmd.info.scaled_dot_product_arg_partition.compression_ratio dataType:MPSDataTypeFloat32] name:nil] name:nil];
				visible = [graph minimumWithPrimaryTensor:[graph maximumWithPrimaryTensor:visible secondaryTensor:zero name:nil] secondaryTensor:[graph constantWithScalar:C dataType:MPSDataTypeFloat32] name:nil];
			}
			MPSGraphTensor* const positions = [graph castTensor:[graph coordinateAlongAxis:1 withShape:@[@1, @(C)] name:nil] toType:MPSDataTypeFloat32 name:nil];
			MPSGraphTensor* const reachable = [graph lessThanWithPrimaryTensor:positions secondaryTensor:visible name:nil];
			MPSGraphTensor* eligible = reachable;
			scores = [graph selectWithPredicateTensor:reachable truePredicateTensor:scores falsePredicateTensor:negative name:nil];
			MPSGraphTensor* pool = nil;
			if (output_size == 2)
			{
				const int blocks = (C + block_size - 1) / block_size;
				const int keep = ccv_min(pool_size, blocks);
				MPSGraphTensor* padded = scores;
				if (blocks * block_size > C)
					padded = [graph concatTensors:@[scores, [graph constantWithScalar:-FLT_MAX shape:@[@(T), @(blocks * block_size - C)] dataType:MPSDataTypeFloat32]] dimension:1 name:nil];
				MPSGraphTensor* block_scores = [graph reshapeTensor:[graph reductionMaximumWithTensor:[graph reshapeTensor:padded withShape:@[@(T), @(blocks), @(block_size)] name:nil] axis:2 name:nil] withShape:@[@(T), @(blocks)] name:nil];
				MPSGraphTensor* const newest = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph subtractionWithPrimaryTensor:visible secondaryTensor:[graph constantWithScalar:1 dataType:MPSDataTypeFloat32] name:nil] secondaryTensor:[graph constantWithScalar:block_size dataType:MPSDataTypeFloat32] name:nil] name:nil];
				MPSGraphTensor* const block_ids = [graph castTensor:[graph coordinateAlongAxis:1 withShape:@[@1, @(blocks)] name:nil] toType:MPSDataTypeFloat32 name:nil];
				block_scores = [graph selectWithPredicateTensor:[graph equalWithPrimaryTensor:block_ids secondaryTensor:newest name:nil] truePredicateTensor:[graph constantWithScalar:INFINITY dataType:MPSDataTypeFloat32] falsePredicateTensor:block_scores name:nil];
				NSArray<MPSGraphTensor*>* const top = keep == blocks ? @[[graph sortWithTensor:block_scores axis:1 descending:YES name:nil], [graph argSortWithTensor:block_scores axis:1 descending:YES name:nil]] : [graph topKWithSourceTensor:block_scores k:keep name:nil];
				MPSGraphTensor* const sentinel = [graph constantWithScalar:blocks dataType:MPSDataTypeInt32];
				pool = [graph selectWithPredicateTensor:[graph greaterThanWithPrimaryTensor:top[0] secondaryTensor:negative name:nil] truePredicateTensor:top[1] falsePredicateTensor:sentinel name:nil];
				pool = [graph sortWithTensor:pool axis:1 descending:NO name:nil];
				pool = [graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:pool secondaryTensor:sentinel name:nil] truePredicateTensor:pool falsePredicateTensor:minus_one name:nil];
				if (keep < pool_size)
					pool = [graph concatTensors:@[pool, [graph constantWithScalar:-1 shape:@[@(T), @(pool_size - keep)] dataType:MPSDataTypeInt32]] dimension:1 name:nil];
			}
			if (input_size == 4)
			{
				const int blocks = (C + block_size - 1) / block_size;
				MPSGraphTensor* const zero_i = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
				MPSGraphTensor* const upper = [graph constantWithScalar:blocks dataType:MPSDataTypeInt32];
				MPSGraphTensor* const nonnegative = [graph greaterThanOrEqualToWithPrimaryTensor:tensors[3] secondaryTensor:zero_i name:nil];
				MPSGraphTensor* const valid = [graph selectWithPredicateTensor:nonnegative truePredicateTensor:[graph lessThanWithPrimaryTensor:tensors[3] secondaryTensor:upper name:nil] falsePredicateTensor:[graph constantWithScalar:0 dataType:MPSDataTypeBool] name:nil];
				MPSGraphTensor* const safe_ids = [graph selectWithPredicateTensor:valid truePredicateTensor:tensors[3] falsePredicateTensor:zero_i name:nil];
				MPSGraphTensor* const row_ids = [graph castTensor:[graph coordinateAlongAxis:0 withShape:@[@(T), @(pool_size)] name:nil] toType:MPSDataTypeInt32 name:nil];
				MPSGraphTensor* const flat_ids = [graph reshapeTensor:[graph additionWithPrimaryTensor:safe_ids secondaryTensor:[graph multiplicationWithPrimaryTensor:row_ids secondaryTensor:upper name:nil] name:nil] withShape:@[@(T * pool_size)] name:nil];
				MPSGraphTensor* const updates = [graph reshapeTensor:[graph castTensor:valid toType:MPSDataTypeInt32 name:nil] withShape:@[@(T * pool_size)] name:nil];
				// Add rather than set: invalid slots clamped to block zero must not erase it.
				MPSGraphTensor* mask = [graph scatterWithUpdatesTensor:updates indicesTensor:flat_ids shape:@[@(T * blocks)] axis:0 mode:MPSGraphScatterModeAdd name:nil];
				mask = [graph reshapeTensor:mask withShape:@[@(T), @(blocks), @1] name:nil];
				mask = [graph broadcastTensor:mask toShape:@[@(T), @(blocks), @(block_size)] name:nil];
				mask = [graph reshapeTensor:mask withShape:@[@(T), @(blocks * block_size)] name:nil];
				if (blocks * block_size > C)
					mask = [graph sliceTensor:mask dimension:1 start:0 length:C name:nil];
				MPSGraphTensor* const member = [graph greaterThanWithPrimaryTensor:mask secondaryTensor:zero_i name:nil];
				eligible = [graph selectWithPredicateTensor:reachable truePredicateTensor:member falsePredicateTensor:[graph constantWithScalar:0 dataType:MPSDataTypeBool] name:nil];
				scores = [graph selectWithPredicateTensor:member truePredicateTensor:scores falsePredicateTensor:negative name:nil];
			}
			const int keep = ccv_min(kth, C);
			MPSGraphTensor* selected;
			if (C <= kth) {
				selected = [graph selectWithPredicateTensor:eligible truePredicateTensor:[graph castTensor:positions toType:MPSDataTypeInt32 name:nil] falsePredicateTensor:width name:nil];
				// Compact holes in a restricted pool without computing scores.
				if (input_size == 4)
					selected = [graph sortWithTensor:selected axis:1 descending:NO name:nil];
				selected = [graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:selected secondaryTensor:width name:nil] truePredicateTensor:selected falsePredicateTensor:minus_one name:nil];
			} else {
				NSArray<MPSGraphTensor*>* const top = keep == C ? @[[graph sortWithTensor:scores axis:1 descending:YES name:nil], [graph argSortWithTensor:scores axis:1 descending:YES name:nil]] : [graph topKWithSourceTensor:scores k:keep name:nil];
				selected = [graph selectWithPredicateTensor:[graph greaterThanWithPrimaryTensor:top[0] secondaryTensor:negative name:nil] truePredicateTensor:top[1] falsePredicateTensor:width name:nil];
				if (cmd.info.scaled_dot_product_arg_partition.sort_indices)
					selected = [graph sortWithTensor:selected axis:1 descending:NO name:nil];
				selected = [graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:selected secondaryTensor:width name:nil] truePredicateTensor:selected falsePredicateTensor:minus_one name:nil];
			}
			if (keep < kth)
				selected = [graph concatTensors:@[selected, [graph constantWithScalar:-1 shape:@[@(T), @(kth - keep)] dataType:MPSDataTypeInt32]] dimension:1 name:nil];
			[result_tensors addObject:selected];
			if (pool)
				[result_tensors addObject:pool];
		});
		NSMutableArray<MPSGraphTensorData*>* data = [NSMutableArray new];
		int i;
		for (i = 0; i < input_size; i++)
		{
			const ccv_nnc_tensor_view_t* const tensor = (const ccv_nnc_tensor_view_t*)inputs[indices[i]];
			[data addObject:ccv_nnc_mps_graph_tensor_data(tensor, tensor->info.dim, tensor->stride)];
		}
		ccv_nnc_tensor_view_t* result[2] = { (ccv_nnc_tensor_view_t*)outputs[0], output_size == 2 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0 };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, data, result, (int*[]){ result[0]->info.dim, result[1] ? result[1]->info.dim : 0 }, (int*[]){ result[0]->stride, result[1] ? result[1]->stride : 0 }, output_size, 0);
		[data release];
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_arg_partition_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3 || input_size == 4);
	assert(output_size == 1 || output_size == 2);
	const ccv_nnc_tensor_view_t* const q = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const k = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const head_w = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const selected = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(q));
	assert(CCV_IS_TENSOR_CONTIGUOUS(k));
	assert(CCV_IS_TENSOR_CONTIGUOUS(head_w));
	assert(CCV_IS_TENSOR_CONTIGUOUS(selected));
	assert(selected->info.datatype == CCV_32S);
	const int q_nd = ccv_nnc_tensor_nd(q->info.dim);
	const int k_nd = ccv_nnc_tensor_nd(k->info.dim);
	const int head_w_nd = ccv_nnc_tensor_nd(head_w->info.dim);
	const int selected_nd = ccv_nnc_tensor_nd(selected->info.dim);
	assert(q_nd == 3);
	assert(head_w_nd == 2);
	assert(selected_nd == 2);
	const int T = q->info.dim[0];
	const int H = q->info.dim[1];
	const int D = q->info.dim[2];
	const int C = k->info.dim[0];
	const int kth = cmd.info.scaled_dot_product_arg_partition.kth;
	const int compression_ratio = cmd.info.scaled_dot_product_arg_partition.compression_ratio;
	assert(C == 0 || k_nd == 2);
	assert(C == 0 || k->info.dim[1] == D);
	assert(head_w->info.dim[0] == T);
	assert(head_w->info.dim[1] == H);
	assert(selected->info.dim[0] == T);
	assert(selected->info.dim[1] == kth);
	assert(kth > 0);
	assert(compression_ratio > 0);
	const int block_size = cmd.info.scaled_dot_product_arg_partition.candidate_block_size;
	const int pool_size = cmd.info.scaled_dot_product_arg_partition.candidate_kth;
	const int extended = input_size == 4 || output_size == 2 || cmd.info.scaled_dot_product_arg_partition.sort_indices;
	if (input_size == 4 || output_size == 2)
	{
		assert(input_size == 3 || output_size == 1);
		assert(block_size > 0 && pool_size > 0);
		const ccv_nnc_tensor_t* const ids = input_size == 4 ? inputs[3] : outputs[1];
		assert(CCV_IS_TENSOR_CONTIGUOUS(ids));
		assert(ids->info.datatype == CCV_32S);
		assert(ccv_nnc_tensor_nd(ids->info.dim) == 2 && ids->info.dim[0] == T && ids->info.dim[1] == pool_size);
	}
	if (C == 0 || (C <= kth && !extended))
	{
		const int status = _ccv_nnc_scaled_dot_product_arg_partition_enumerate(cmd, hint, flags, inputs, input_size, outputs, 1, T, C, kth, compression_ratio, stream_context);
		if (status != CCV_NNC_EXEC_SUCCESS || output_size == 1)
			return status;
		return _ccv_nnc_scaled_dot_product_arg_partition_enumerate(cmd, hint, flags, inputs, input_size, outputs + 1, 1, T, 0, pool_size, compression_ratio, stream_context);
	}
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();
		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;
		const int use_neural_accelerators = ccv_nnc_mfa_has_neural_accelerators(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa)
		{
			if (q->info.datatype != k->info.datatype || q->info.datatype != head_w->info.datatype)
				use_mfa = false;
			else if (q->info.datatype == CCV_32F) {
				mtl_data_type = 3;
			} else if (q->info.datatype == CCV_16F) {
				mtl_data_type = 16;
			} else if (q->info.datatype == CCV_16BF) {
				mtl_data_type = 121;
				if (!ccv_nnc_mfa_neural_accelerators_support_bfloat(context))
					use_mfa = false;
			} else {
				use_mfa = false;
			}
		}
		// Both score shaders tile over H, but their dot-product width is fixed at 128.
		if (use_mfa && (D != 128 || kth > 1024 || (extended && pool_size > 2048)))
			use_mfa = false;
		if (use_mfa)
		{
			const ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params = {
				.candidate_block_size = block_size,
				.candidate_count = pool_size,
				.has_candidates = input_size == 4,
				.output_candidates = output_size == 2,
				.sort_indices = cmd.info.scaled_dot_product_arg_partition.sort_indices,
				.data_type = mtl_data_type,
				.T = (uint32_t)T,
				.C = (uint32_t)C,
				.H = (uint32_t)H,
				.D = (uint32_t)D,
				.kth = (uint32_t)kth,
				.compression_ratio = (uint32_t)compression_ratio,
				.query_offset = cmd.info.scaled_dot_product_arg_partition.query_offset,
				.scale = cmd.info.scaled_dot_product_arg_partition.scale,
				.is_causal = (uint8_t)(cmd.info.scaled_dot_product_arg_partition.is_causal != 0),
				.use_neural_accelerators = (uint8_t)use_neural_accelerators,
				.loadM = (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M) != 0,
			};
			if (METAL_LOG_LEVEL(context) >= 3)
				ccv_nnc_mfa_log_message(use_neural_accelerators ? "SDPAP: MFA neural accelerators." : "SDPAP: MFA generic.");
			ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition(context, params);
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[7] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[1]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(outputs[0]),
				input_size == 4 ? mpgetbuffer(inputs[3]) : NULL,
				output_size == 2 ? mpgetbuffer(outputs[1]) : NULL,
				NULL,
			};
			size_t tensor_offsets[6] = {
				q->dataof,
				k->dataof,
				head_w->dataof,
				selected->dataof,
				input_size == 4 ? inputs[3]->dataof : 0,
				output_size == 2 ? outputs[1]->dataof : 0,
			};
			ccv_nnc_mfa_encode_scaled_dot_product_arg_partition(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
		if (METAL_LOG_LEVEL(context) >= 3)
			ccv_nnc_mfa_log_message("SDPAP: MPSGraph scoring.");
		if (extended)
			return _ccv_nnc_scaled_dot_product_arg_partition_candidates_graph(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_q;
			MPSGraphTensor* mps_q = ccv_nnc_mps_graph_tensor_input(graph, q, q->info.dim, q->stride, &mps_input_q);
			[inputTensors addObject:mps_input_q];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(q, q->info.dim, q->stride)];
			MPSGraphTensor* mps_input_k;
			MPSGraphTensor* mps_k = ccv_nnc_mps_graph_tensor_input(graph, k, k->info.dim, k->stride, &mps_input_k);
			[inputTensors addObject:mps_input_k];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(k, k->info.dim, k->stride)];
			MPSGraphTensor* mps_input_head_w;
			MPSGraphTensor* mps_head_w = ccv_nnc_mps_graph_tensor_input(graph, head_w, head_w->info.dim, head_w->stride, &mps_input_head_w);
			[inputTensors addObject:mps_input_head_w];
			[inputShapedTypes addObject:ccv_nnc_mps_graph_tensor_input_shape(head_w, head_w->info.dim, head_w->stride)];
			const int k_eff = ccv_min(kth, C);
			MPSGraphTensor* mps_selected;
			if (k_eff == 0)
			{
				mps_selected = [graph constantWithScalar:-1.0f shape:@[@(T), @(kth)] dataType:MPSDataTypeInt32];
			} else {
					mps_q = q->info.datatype == CCV_32F ? mps_q : [graph castTensor:mps_q toType:MPSDataTypeFloat32 name:@"q_float"];
					mps_k = k->info.datatype == CCV_32F ? mps_k : [graph castTensor:mps_k toType:MPSDataTypeFloat32 name:@"k_float"];
					mps_head_w = head_w->info.datatype == CCV_32F ? mps_head_w : [graph castTensor:mps_head_w toType:MPSDataTypeFloat32 name:@"head_w_float"];
					MPSGraphTensor* mps_q_2d = [graph reshapeTensor:mps_q withShape:@[@(T * H), @(D)] name:nil];
					MPSGraphTensor* mps_kt = [graph transposeTensor:mps_k dimension:0 withDimension:1 name:nil];
					MPSGraphTensor* mps_dot = [graph matrixMultiplicationWithPrimaryTensor:mps_q_2d secondaryTensor:mps_kt name:nil];
					mps_dot = [graph reshapeTensor:mps_dot withShape:@[@(T), @(H), @(C)] name:nil];
				MPSGraphTensor* mps_zero = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_positive_dot = [graph maximumWithPrimaryTensor:mps_dot secondaryTensor:mps_zero name:nil];
					MPSGraphTensor* mps_head_w_3d = [graph reshapeTensor:mps_head_w withShape:@[@(T), @(H), @1] name:nil];
					MPSGraphTensor* mps_weighted = [graph multiplicationWithPrimaryTensor:mps_positive_dot secondaryTensor:mps_head_w_3d name:nil];
					MPSGraphTensor* mps_scores = [graph reductionSumWithTensor:mps_weighted axis:1 name:nil];
					mps_scores = [graph reshapeTensor:mps_scores withShape:@[@(T), @(C)] name:nil];
					MPSGraphTensor* mps_scale = [graph constantWithScalar:cmd.info.scaled_dot_product_arg_partition.scale dataType:MPSDataTypeFloat32];
				mps_scores = [graph multiplicationWithPrimaryTensor:mps_scores secondaryTensor:mps_scale name:nil];
				if (cmd.info.scaled_dot_product_arg_partition.is_causal)
				{
					NSArray<NSNumber*>* score_shape = @[@(T), @(C)];
					MPSGraphTensor* mps_t = [graph coordinateAlongAxis:0 withShape:score_shape name:nil];
					MPSGraphTensor* mps_c = [graph coordinateAlongAxis:1 withShape:score_shape name:nil];
					MPSGraphTensor* mps_t_f = [graph castTensor:mps_t toType:MPSDataTypeFloat32 name:nil];
					MPSGraphTensor* mps_c_f = [graph castTensor:mps_c toType:MPSDataTypeFloat32 name:nil];
					MPSGraphTensor* mps_query_offset = [graph constantWithScalar:(float)(cmd.info.scaled_dot_product_arg_partition.query_offset + 1) dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_ratio = [graph constantWithScalar:(float)compression_ratio dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph additionWithPrimaryTensor:mps_t_f secondaryTensor:mps_query_offset name:nil] secondaryTensor:mps_ratio name:nil] name:nil];
					MPSGraphTensor* mps_visible_min = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible_max = [graph constantWithScalar:(float)C dataType:MPSDataTypeFloat32];
					mps_visible = [graph maximumWithPrimaryTensor:mps_visible secondaryTensor:mps_visible_min name:nil];
					mps_visible = [graph minimumWithPrimaryTensor:mps_visible secondaryTensor:mps_visible_max name:nil];
					MPSGraphTensor* mps_valid = [graph lessThanWithPrimaryTensor:mps_c_f secondaryTensor:mps_visible name:nil];
					MPSGraphTensor* mps_neg = [graph constantWithScalar:-3.402823466e+38f dataType:MPSDataTypeFloat32];
					mps_scores = [graph selectWithPredicateTensor:mps_valid truePredicateTensor:mps_scores falsePredicateTensor:mps_neg name:nil];
				}
					if (k_eff == C)
						mps_selected = [graph argSortWithTensor:mps_scores axis:1 descending:YES name:nil];
					else {
						NSArray<MPSGraphTensor*>* result = [graph topKWithSourceTensor:mps_scores k:k_eff name:nil];
						mps_selected = result[1];
					}
				if (cmd.info.scaled_dot_product_arg_partition.is_causal)
				{
					NSArray<NSNumber*>* topk_shape = @[@(T), @(k_eff)];
					MPSGraphTensor* mps_t = [graph coordinateAlongAxis:0 withShape:topk_shape name:nil];
					MPSGraphTensor* mps_pos = [graph coordinateAlongAxis:1 withShape:topk_shape name:nil];
					MPSGraphTensor* mps_t_f = [graph castTensor:mps_t toType:MPSDataTypeFloat32 name:nil];
					MPSGraphTensor* mps_pos_f = [graph castTensor:mps_pos toType:MPSDataTypeFloat32 name:nil];
					MPSGraphTensor* mps_query_offset = [graph constantWithScalar:(float)(cmd.info.scaled_dot_product_arg_partition.query_offset + 1) dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_ratio = [graph constantWithScalar:(float)compression_ratio dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph additionWithPrimaryTensor:mps_t_f secondaryTensor:mps_query_offset name:nil] secondaryTensor:mps_ratio name:nil] name:nil];
					MPSGraphTensor* mps_visible_min = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible_max = [graph constantWithScalar:(float)C dataType:MPSDataTypeFloat32];
					mps_visible = [graph maximumWithPrimaryTensor:mps_visible secondaryTensor:mps_visible_min name:nil];
					mps_visible = [graph minimumWithPrimaryTensor:mps_visible secondaryTensor:mps_visible_max name:nil];
					MPSGraphTensor* mps_valid_position = [graph lessThanWithPrimaryTensor:mps_pos_f secondaryTensor:mps_visible name:nil];
					MPSGraphTensor* mps_minus_one = [graph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];
					mps_selected = [graph selectWithPredicateTensor:mps_valid_position truePredicateTensor:mps_selected falsePredicateTensor:mps_minus_one name:nil];
				}
				if (k_eff < kth)
				{
					MPSGraphTensor* mps_tail = [graph constantWithScalar:-1.0f shape:@[@(T), @(kth - k_eff)] dataType:MPSDataTypeInt32];
					mps_selected = [graph concatTensors:@[mps_selected, mps_tail] dimension:1 name:nil];
				}
			}
			[resultTensors addObject:mps_selected];
		});
		MPSGraphTensorData* data_q = ccv_nnc_mps_graph_tensor_data(q, q->info.dim, q->stride);
		MPSGraphTensorData* data_k = ccv_nnc_mps_graph_tensor_data(k, k->info.dim, k->stride);
		MPSGraphTensorData* data_head_w = ccv_nnc_mps_graph_tensor_data(head_w, head_w->info.dim, head_w->stride);
		MPSGraphTensorData* data[] = { data_q, data_k, data_head_w };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &selected, (int*[]){ selected->info.dim }, (int*[]){ selected->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_scaled_dot_product_arg_partition_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_arg_partition_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_scaled_dot_product_arg_partition_back;
}
