#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif

static int _ccv_nnc_scaled_dot_product_arg_partition_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 1);
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
	assert(k_nd == 2);
	assert(head_w_nd == 2);
	assert(selected_nd == 2);
	const int T = q->info.dim[0];
	const int H = q->info.dim[1];
	const int D = q->info.dim[2];
	const int C = k->info.dim[0];
	const int kth = cmd.info.scaled_dot_product_arg_partition.kth;
	const int compression_ratio = cmd.info.scaled_dot_product_arg_partition.compression_ratio;
	assert(k->info.dim[1] == D);
	assert(head_w->info.dim[0] == T);
	assert(head_w->info.dim[1] == H);
	assert(selected->info.dim[0] == T);
	assert(selected->info.dim[1] == kth);
	assert(kth > 0);
	assert(compression_ratio > 0);
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
		if (use_mfa && (H != 64 || D != 128 || kth > 1024))
			use_mfa = false;
		if (use_mfa)
		{
			const ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params = {
				.data_type = mtl_data_type,
				.T = (uint32_t)T,
				.C = (uint32_t)C,
				.H = (uint32_t)H,
				.D = (uint32_t)D,
				.kth = (uint32_t)kth,
				.compression_ratio = (uint32_t)compression_ratio,
				.scale = cmd.info.scaled_dot_product_arg_partition.scale,
				.is_causal = (uint8_t)(cmd.info.scaled_dot_product_arg_partition.is_causal != 0),
				.use_neural_accelerators = (uint8_t)use_neural_accelerators,
			};
			ccv_nnc_mfa_prepare_scaled_dot_product_arg_partition(context, params);
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[5] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[1]),
				mpgetbuffer(inputs[2]),
				mpgetbuffer(outputs[0]),
				NULL,
			};
			size_t tensor_offsets[4] = {
				q->dataof,
				k->dataof,
				head_w->dataof,
				selected->dataof,
			};
			ccv_nnc_mfa_encode_scaled_dot_product_arg_partition(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}
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
					MPSGraphTensor* mps_q_start = [graph constantWithScalar:(float)(C * compression_ratio - T + 1) dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_ratio = [graph constantWithScalar:(float)compression_ratio dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph additionWithPrimaryTensor:mps_t_f secondaryTensor:mps_q_start name:nil] secondaryTensor:mps_ratio name:nil] name:nil];
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
					MPSGraphTensor* mps_q_start = [graph constantWithScalar:(float)(C * compression_ratio - T + 1) dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_ratio = [graph constantWithScalar:(float)compression_ratio dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_visible = [graph floorWithTensor:[graph divisionWithPrimaryTensor:[graph additionWithPrimaryTensor:mps_t_f secondaryTensor:mps_q_start name:nil] secondaryTensor:mps_ratio name:nil] name:nil];
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
