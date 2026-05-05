#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_rmsnorm_gated_mfa_datatype(const int datatype)
{
	switch (datatype) {
		case CCV_16F:
			return 16;
		case CCV_16BF:
			return 121;
		case CCV_32F:
			return 3;
		default:
			return -1;
	}
}

static int _ccv_nnc_rmsnorm_gated_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int elementwise_affine = cmd.info.rmsnorm_gated.elementwise_affine;
	assert(input_size == (elementwise_affine ? 3 : 2));
	assert(output_size == 1);
	const float epsilon = cmd.info.rmsnorm_gated.epsilon;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const gate = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const scale = elementwise_affine ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(b->info.datatype == a->info.datatype);
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd >= 1);
	int i;
	for (i = 0; i < cmd.info.rmsnorm_gated.count; i++)
		assert(cmd.info.rmsnorm_gated.axis[i] >= 0 && cmd.info.rmsnorm_gated.axis[i] < a_nd);
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == gate->info.dim[i]);
		assert(a->info.dim[i] == b->info.dim[i]);
	}
	const int column_count = a->info.dim[a_nd - 1];
	const int row_count = ccv_nnc_tensor_count(a->info) / column_count;
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;

		const int a_data_type = _ccv_nnc_rmsnorm_gated_mfa_datatype(a->info.datatype);
		const int gate_data_type = _ccv_nnc_rmsnorm_gated_mfa_datatype(gate->info.datatype);
		const int scale_data_type = elementwise_affine ? _ccv_nnc_rmsnorm_gated_mfa_datatype(scale->info.datatype) : -1;
		if (use_mfa && (a_data_type < 0 || gate_data_type < 0 || scale_data_type < 0))
			use_mfa = false;

		if (use_mfa && (!elementwise_affine || cmd.info.rmsnorm_gated.count != 1 || cmd.info.rmsnorm_gated.axis[0] != a_nd - 1 || ccv_nnc_tensor_count(scale->info) != column_count))
			use_mfa = false;

		if (use_mfa && (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(gate) || !CCV_IS_TENSOR_CONTIGUOUS(scale) || !CCV_IS_TENSOR_CONTIGUOUS(b)))
			use_mfa = false;

		if (use_mfa) {
			ccv_nnc_mfa_rmsnorm_gated_params_t params = {
				.epsilon = epsilon,
				.a_data_type = (uint64_t)a_data_type,
				.gate_data_type = (uint64_t)gate_data_type,
				.scale_data_type = (uint64_t)scale_data_type,
				.row_count = (uint32_t)row_count,
				.column_count = (uint32_t)column_count
			};
			ccv_nnc_mfa_prepare_rmsnorm_gated(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[5] = {
				mpgetbuffer(inputs[0]), // source
				mpgetbuffer(inputs[1]), // gate
				mpgetbuffer(inputs[2]), // scale
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[4] = {
				a->dataof,
				gate->dataof,
				scale->dataof,
				b->dataof
			};
			ccv_nnc_mfa_encode_rmsnorm_gated(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[3];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_input_gate;
				MPSGraphTensor* mps_gate = ccv_nnc_mps_graph_tensor_input(graph, gate, gate->info.dim, gate->stride, &mps_input_gate);
				[inputTensors addObject:mps_input_gate];
				MPSGraphShapedType* mps_gate_shape = ccv_nnc_mps_graph_tensor_input_shape(gate, gate->info.dim, gate->stride);
				[inputShapedTypes addObject:mps_gate_shape];
				MPSGraphTensor* mps_scale;
				if (elementwise_affine)
				{
					MPSGraphTensor* mps_input_scale;
					mps_scale = ccv_nnc_mps_graph_tensor_input(graph, scale, scale->info.dim, scale->stride, &mps_input_scale);
					[inputTensors addObject:mps_input_scale];
					MPSGraphShapedType* mps_scale_shape = ccv_nnc_mps_graph_tensor_input_shape(scale, scale->info.dim, scale->stride);
					[inputShapedTypes addObject:mps_scale_shape];
				}

				MPSGraphTensor* mps_a_f32 = a->info.datatype == CCV_32F ? mps_a : [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
				MPSGraphTensor* mps_gate_f32 = gate->info.datatype == CCV_32F ? mps_gate : [graph castTensor:mps_gate toType:MPSDataTypeFloat32 name:@"mps_gate_float"];
				MPSGraphTensor* mps_square = [graph squareWithTensor:mps_a_f32 name:nil];
				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				int i;
				for (i = 0; i < cmd.info.rmsnorm_gated.count; i++)
					[axes addObject:@(cmd.info.rmsnorm_gated.axis[i])];
				MPSGraphTensor* mps_variance = [graph meanOfTensor:mps_square axes:axes name:nil];
				[axes release];
				MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_inv_std = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_variance secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
				MPSGraphTensor* mps_norm = [graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_inv_std name:nil];
				if (elementwise_affine)
				{
					MPSGraphTensor* mps_scale_f32 = scale->info.datatype == CCV_32F ? mps_scale : [graph castTensor:mps_scale toType:MPSDataTypeFloat32 name:@"mps_scale_float"];
					mps_norm = [graph multiplicationWithPrimaryTensor:mps_norm secondaryTensor:mps_scale_f32 name:nil];
				}
				MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_gate_f32 name:nil];
				MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_sigmoid = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
				MPSGraphTensor* mps_swish = [graph multiplicationWithPrimaryTensor:mps_gate_f32 secondaryTensor:mps_sigmoid name:nil];
				MPSGraphTensor* mps_b = [graph multiplicationWithPrimaryTensor:mps_norm secondaryTensor:mps_swish name:nil];
				if (b->info.datatype != CCV_32F)
					mps_b = [graph castTensor:mps_b toType:ccv_nnc_mps_datatype(b->info.datatype) name:@"mps_b"];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_gate = ccv_nnc_mps_graph_tensor_data(gate, gate->info.dim, gate->stride);
			if (elementwise_affine)
			{
				MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(scale, scale->info.dim, scale->stride);
				MPSGraphTensorData* data[] = {data_a, data_gate, data_scale};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
			} else {
				MPSGraphTensorData* data[] = {data_a, data_gate};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_GATED_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_gated_forw;
}
