#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "nnc/mps/ccv_nnc_mps.h"

static uint32_t _ccv_nnc_moe_routing_mfa_datatype(const int datatype)
{
	switch (datatype)
	{
		case CCV_32F:
			return 3;
		case CCV_16F:
			return 16;
		case CCV_16BF:
			return 121;
		default:
			return UINT32_MAX;
	}
}

static MPSGraphTensor* _ccv_nnc_moe_routing_stable_log1p(MPSGraph* const graph, MPSGraphTensor* const x)
{
	MPSGraphTensor* const one = [graph constantWithScalar:1.0 dataType:x.dataType];
	MPSGraphTensor* const xp1 = [graph additionWithPrimaryTensor:one secondaryTensor:x name:nil];
	MPSGraphTensor* const log_xp1 = [graph logarithmWithTensor:xp1 name:nil];
	MPSGraphTensor* const denominator = [graph subtractionWithPrimaryTensor:xp1 secondaryTensor:one name:nil];
	MPSGraphTensor* const quotient = [graph divisionWithPrimaryTensor:log_xp1 secondaryTensor:denominator name:nil];
	MPSGraphTensor* const y = [graph multiplicationWithPrimaryTensor:x secondaryTensor:quotient name:nil];
	MPSGraphTensor* const unchanged = [graph equalWithPrimaryTensor:xp1 secondaryTensor:one name:nil];
	return [graph selectWithPredicateTensor:unchanged truePredicateTensor:x falsePredicateTensor:y name:nil];
}

static MPSGraphTensor* _ccv_nnc_moe_routing_probabilities(MPSGraph* const graph, MPSGraphTensor* const logits)
{
	MPSGraphTensor* const zero = [graph constantWithScalar:0.0 dataType:logits.dataType];
	MPSGraphTensor* const positive = [graph maximumWithPrimaryTensor:logits secondaryTensor:zero name:nil];
	MPSGraphTensor* const negative_absolute = [graph negativeWithTensor:[graph absoluteWithTensor:logits name:nil] name:nil];
	MPSGraphTensor* const exponential = [graph exponentWithTensor:negative_absolute name:nil];
	MPSGraphTensor* const softplus = [graph additionWithPrimaryTensor:positive secondaryTensor:_ccv_nnc_moe_routing_stable_log1p(graph, exponential) name:nil];
	return [graph squareRootWithTensor:softplus name:nil];
}

static void _ccv_nnc_moe_routing_unique_segments(MPSGraph* const graph, MPSGraphTensor* const sorted_experts, const int pair_count, const int group_count, MPSGraphTensor** const expert_indices, MPSGraphTensor** const expert_counts)
{
	MPSGraphTensor* const front = [graph sliceTensor:sorted_experts dimension:0 start:0 length:pair_count - 1 name:nil];
	MPSGraphTensor* const back = [graph sliceTensor:sorted_experts dimension:0 start:1 length:pair_count - 1 name:nil];
	MPSGraphTensor* const changed = [graph notEqualWithPrimaryTensor:back secondaryTensor:front name:nil];
	MPSGraphTensor* const mask = [graph castTensor:changed toType:MPSDataTypeInt32 name:nil];
	MPSGraphTensor* const scanned = [graph cumulativeSumWithTensor:mask axis:0 name:nil];
	MPSGraphTensor* const zero = [graph constantWithScalar:0.0 shape:@[@1] dataType:MPSDataTypeInt32];
	MPSGraphTensor* const segment_indices = [graph concatTensors:@[zero, scanned] dimension:0 name:nil];
	MPSGraphTensor* const first = [graph constantWithScalar:1.0 shape:@[@1] dataType:MPSDataTypeBool];
	MPSGraphTensor* const starts = [graph concatTensors:@[first, changed] dimension:0 name:nil];
	MPSGraphTensor* const dummy_index = [graph constantWithScalar:group_count dataType:MPSDataTypeInt32];
	MPSGraphTensor* const scatter_indices = [graph selectWithPredicateTensor:starts truePredicateTensor:segment_indices falsePredicateTensor:dummy_index name:nil];
	MPSGraphTensor* const minus_one = [graph constantWithScalar:-1.0 dataType:MPSDataTypeInt32];
	MPSGraphTensor* const scatter_updates = [graph selectWithPredicateTensor:starts truePredicateTensor:sorted_experts falsePredicateTensor:minus_one name:nil];
	MPSGraphTensor* const empty_experts = [graph constantWithScalar:-1.0 shape:@[@(group_count + 1)] dataType:MPSDataTypeInt32];
	MPSGraphTensor* const experts_with_dummy = [graph scatterWithDataTensor:empty_experts updatesTensor:scatter_updates indicesTensor:scatter_indices axis:0 mode:MPSGraphScatterModeSet name:nil];
	*expert_indices = [graph sliceTensor:experts_with_dummy dimension:0 start:0 length:group_count name:nil];
	MPSGraphTensor* const ones = [graph constantWithScalar:1.0 shape:@[@(pair_count)] dataType:MPSDataTypeInt32];
	*expert_counts = [graph scatterWithUpdatesTensor:ones indicesTensor:segment_indices shape:@[@(group_count)] axis:0 mode:MPSGraphScatterModeAdd name:nil];
}

static int _ccv_nnc_moe_routing_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 3 || output_size != 5)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const logits = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const route = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const activation = (const ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const gathered = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const route_weights = (ccv_nnc_tensor_view_t*)outputs[1];
	ccv_nnc_tensor_view_t* const token_indices = (ccv_nnc_tensor_view_t*)outputs[2];
	ccv_nnc_tensor_view_t* const expert_indices = (ccv_nnc_tensor_view_t*)outputs[3];
	ccv_nnc_tensor_view_t* const expert_counts = (ccv_nnc_tensor_view_t*)outputs[4];
	const int kth = cmd.info.moe_routing.kth;
	const int token_count = logits->info.dim[0];
	const int expert_count = logits->info.dim[1];
	const int hidden = activation->info.dim[1];
	const int pair_count = token_count * kth;
	const int group_count = ccv_min(pair_count, expert_count);
	const int single_input_token = token_count == 1 && (cmd.info.moe_routing.flags & CCV_NNC_MOE_ROUTING_SINGLE_INPUT_TOKEN);
	const int gathered_rows = single_input_token ? 1 : pair_count;
	if (kth <= 0 || expert_count < kth || token_count <= 0 || hidden <= 0 || cmd.info.moe_routing.weight_scale <= 0 ||
		(cmd.info.moe_routing.preselected != 0 && cmd.info.moe_routing.preselected != 1) ||
		ccv_nnc_tensor_nd(logits->info.dim) != 2 || ccv_nnc_tensor_nd(activation->info.dim) != 2 || activation->info.dim[0] != token_count ||
		(logits->info.datatype != CCV_32F && logits->info.datatype != CCV_16F) || route_weights->info.datatype != CCV_32F ||
		token_indices->info.datatype != CCV_32S || expert_indices->info.datatype != CCV_32S ||
		expert_counts->info.datatype != CCV_32S || activation->info.datatype != gathered->info.datatype ||
		(activation->info.datatype != CCV_32F && activation->info.datatype != CCV_16F && activation->info.datatype != CCV_16BF) ||
		gathered->info.dim[0] != gathered_rows || gathered->info.dim[1] != hidden ||
		ccv_nnc_tensor_count(route_weights->info) != pair_count || ccv_nnc_tensor_count(token_indices->info) != pair_count ||
		ccv_nnc_tensor_count(expert_indices->info) != group_count || ccv_nnc_tensor_count(expert_counts->info) != group_count ||
		!CCV_IS_TENSOR_CONTIGUOUS(logits) || !CCV_IS_TENSOR_CONTIGUOUS(route) ||
		!CCV_IS_TENSOR_CONTIGUOUS(activation) || !CCV_IS_TENSOR_CONTIGUOUS(gathered) ||
		!CCV_IS_TENSOR_CONTIGUOUS(route_weights) || !CCV_IS_TENSOR_CONTIGUOUS(token_indices) ||
		!CCV_IS_TENSOR_CONTIGUOUS(expert_indices) || !CCV_IS_TENSOR_CONTIGUOUS(expert_counts))
		return CCV_NNC_EXEC_INVALID;
	if (cmd.info.moe_routing.preselected)
	{
		if (route->info.datatype != CCV_32S || ccv_nnc_tensor_nd(route->info.dim) != 2 || route->info.dim[0] != token_count || route->info.dim[1] != kth)
			return CCV_NNC_EXEC_INVALID;
	} else if ((route->info.datatype != CCV_32F && route->info.datatype != CCV_16F) ||
		route->info.datatype != logits->info.datatype || ccv_nnc_tensor_nd(route->info.dim) != 1 ||
		route->info.dim[0] != expert_count) {
		return CCV_NNC_EXEC_INVALID;
	}
	if (token_count == 1 && kth <= 32 && expert_count <= 256)
	{
		@autoreleasepool {
			ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
			if (ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			{
				const uint32_t activation_data_type = _ccv_nnc_moe_routing_mfa_datatype(activation->info.datatype);
				const uint32_t routing_data_type = _ccv_nnc_moe_routing_mfa_datatype(logits->info.datatype);
				const ccv_nnc_mfa_moe_routing_params_t params = {
					.activation_data_type = activation_data_type,
					.routing_data_type = routing_data_type,
					.expert_count = (uint32_t)expert_count,
					.kth = (uint32_t)kth,
					.hidden = (uint32_t)hidden,
					.weight_scale = cmd.info.moe_routing.weight_scale,
					.preselected = (uint32_t)cmd.info.moe_routing.preselected,
					.single_input_token = (uint32_t)single_input_token,
				};
				ccv_nnc_mfa_prepare_moe_routing(context, params);
				mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				mtl_buffer_t* tensors[9] = {
					mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(inputs[2]),
					mpgetbuffer(outputs[0]), mpgetbuffer(outputs[1]), mpgetbuffer(outputs[2]),
					mpgetbuffer(outputs[3]), mpgetbuffer(outputs[4]), NULL,
				};
				size_t tensor_offsets[8] = {
					logits->dataof, route->dataof, activation->dataof, gathered->dataof,
					route_weights->dataof, token_indices->dataof, expert_indices->dataof,
					expert_counts->dataof,
				};
				ccv_nnc_mfa_encode_moe_routing(context, params, command_batch, tensors, tensor_offsets);
				ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
				return CCV_NNC_EXEC_SUCCESS;
			}
		}
	}
	@autoreleasepool {
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int input_indices[3];
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, input_indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* input_tensors, NSMutableArray<MPSGraphShapedType*>* input_shapes, NSMutableArray<MPSGraphTensor*>* result_tensors) {
			MPSGraphTensor* input_logits;
			MPSGraphTensor* const mps_logits = ccv_nnc_mps_graph_tensor_input(graph, logits, logits->info.dim, logits->stride, &input_logits);
			[input_tensors addObject:input_logits];
			[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(logits, logits->info.dim, logits->stride)];
			MPSGraphTensor* input_route;
			MPSGraphTensor* const mps_route = ccv_nnc_mps_graph_tensor_input(graph, route, route->info.dim, route->stride, &input_route);
			[input_tensors addObject:input_route];
			[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(route, route->info.dim, route->stride)];
			MPSGraphTensor* input_activation;
			MPSGraphTensor* const mps_activation = ccv_nnc_mps_graph_tensor_input(graph, activation, activation->info.dim, activation->stride, &input_activation);
			[input_tensors addObject:input_activation];
			[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(activation, activation->info.dim, activation->stride)];

			MPSGraphTensor* const probabilities = _ccv_nnc_moe_routing_probabilities(graph, mps_logits);
			MPSGraphTensor* selected;
			if (cmd.info.moe_routing.preselected)
				selected = mps_route;
			else {
				MPSGraphTensor* const selection_scores = [graph additionWithPrimaryTensor:probabilities secondaryTensor:mps_route name:nil];
				NSArray<MPSGraphTensor*>* const top_k = [graph topKWithSourceTensor:selection_scores axis:1 k:kth name:nil];
				selected = top_k[1];
			}
			NSArray<NSNumber*>* const selected_shape = @[@(token_count), @(kth)];
			MPSGraphTensor* const token_grid = [graph coordinateAlongAxis:0 withShape:selected_shape name:nil];
			MPSGraphTensor* const expert_stride = [graph constantWithScalar:expert_count dataType:MPSDataTypeInt32];
			MPSGraphTensor* const probability_indices = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:token_grid secondaryTensor:expert_stride name:nil] secondaryTensor:selected name:nil];
			MPSGraphTensor* const flat_probabilities = [graph reshapeTensor:probabilities withShape:@[@(token_count * expert_count)] name:nil];
			MPSGraphTensor* const flat_probability_indices = [graph reshapeTensor:probability_indices withShape:@[@(pair_count)] name:nil];
			MPSGraphTensor* selected_probabilities = [graph gatherAlongAxis:0 withUpdatesTensor:flat_probabilities indicesTensor:flat_probability_indices name:nil];
			selected_probabilities = [graph reshapeTensor:selected_probabilities withShape:selected_shape name:nil];
			MPSGraphTensor* denominator = [graph reductionSumWithTensor:selected_probabilities axis:1 name:nil];
			denominator = [graph maximumWithPrimaryTensor:denominator secondaryTensor:[graph constantWithScalar:6.103515625e-5f dataType:selected_probabilities.dataType] name:nil];
			denominator = [graph reshapeTensor:denominator withShape:@[@(token_count), @1] name:nil];
			MPSGraphTensor* normalized_weights = [graph divisionWithPrimaryTensor:selected_probabilities secondaryTensor:denominator name:nil];
			normalized_weights = [graph multiplicationWithPrimaryTensor:normalized_weights secondaryTensor:[graph constantWithScalar:cmd.info.moe_routing.weight_scale dataType:normalized_weights.dataType] name:nil];
			MPSGraphTensor* const flat_selected = [graph reshapeTensor:selected withShape:@[@(pair_count)] name:nil];
			MPSGraphTensor* const flat_weights = [graph reshapeTensor:normalized_weights withShape:@[@(pair_count)] name:nil];

			MPSGraphTensor* ordered_weights;
			MPSGraphTensor* ordered_tokens;
			MPSGraphTensor* ordered_activations;
			MPSGraphTensor* grouped_experts;
			MPSGraphTensor* grouped_counts;
			if (token_count == 1)
			{
				ordered_weights = flat_weights;
				ordered_tokens = [graph constantWithScalar:0.0 shape:@[@(kth)] dataType:MPSDataTypeInt32];
				ordered_activations = [graph broadcastTensor:mps_activation toShape:@[@(single_input_token ? 1 : kth), @(hidden)] name:nil];
				grouped_experts = flat_selected;
				grouped_counts = [graph constantWithScalar:1.0 shape:@[@(kth)] dataType:MPSDataTypeInt32];
			} else {
				MPSGraphTensor* const ordered_experts = [graph sortWithTensor:flat_selected axis:0 descending:NO name:nil];
				MPSGraphTensor* const order = [graph argSortWithTensor:flat_selected axis:0 descending:NO name:nil];
				ordered_weights = [graph gatherAlongAxis:0 withUpdatesTensor:flat_weights indicesTensor:order name:nil];
				MPSGraphTensor* const flat_token_grid = [graph reshapeTensor:token_grid withShape:@[@(pair_count)] name:nil];
				ordered_tokens = [graph gatherAlongAxis:0 withUpdatesTensor:flat_token_grid indicesTensor:order name:nil];
				MPSGraphTensor* const activation_indices = [graph broadcastTensor:[graph reshapeTensor:ordered_tokens withShape:@[@(pair_count), @1] name:nil] toShape:@[@(pair_count), @(hidden)] name:nil];
				ordered_activations = [graph gatherAlongAxis:0 withUpdatesTensor:mps_activation indicesTensor:activation_indices name:nil];
				_ccv_nnc_moe_routing_unique_segments(graph, ordered_experts, pair_count, group_count, &grouped_experts, &grouped_counts);
			}
			if (ordered_weights.dataType != MPSDataTypeFloat32)
				ordered_weights = [graph castTensor:ordered_weights toType:MPSDataTypeFloat32 name:nil];
			[result_tensors addObject:ordered_activations];
			[result_tensors addObject:ordered_weights];
			[result_tensors addObject:ordered_tokens];
			[result_tensors addObject:grouped_experts];
			[result_tensors addObject:grouped_counts];
		});
		MPSGraphTensorData* const logits_data = ccv_nnc_mps_graph_tensor_data(logits, logits->info.dim, logits->stride);
		MPSGraphTensorData* const route_data = ccv_nnc_mps_graph_tensor_data(route, route->info.dim, route->stride);
		MPSGraphTensorData* const activation_data = ccv_nnc_mps_graph_tensor_data(activation, activation->info.dim, activation->stride);
		MPSGraphTensorData* const input_data[] = { logits_data, route_data, activation_data };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer,
			@[input_data[input_indices[0]], input_data[input_indices[1]], input_data[input_indices[2]]],
			(ccv_nnc_tensor_view_t*[]){ gathered, route_weights, token_indices, expert_indices, expert_counts },
			(int*[]){ gathered->info.dim, route_weights->info.dim, token_indices->info.dim, expert_indices->info.dim, expert_counts->info.dim },
			(int*[]){ gathered->stride, route_weights->stride, token_indices->stride, expert_indices->stride, expert_counts->stride }, 5, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MOE_ROUTING_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_moe_routing_forw;
}
