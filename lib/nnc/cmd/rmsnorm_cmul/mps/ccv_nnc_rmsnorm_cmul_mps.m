#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_rmsnorm_cmul_mfa_datatype(const int datatype)
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

static int _ccv_nnc_rmsnorm_cmul_fallback(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const rotation, ccv_nnc_tensor_view_t* const b, ccv_nnc_stream_context_t* const stream_context)
{
	MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
	ccv_nnc_tensor_t* const inputs[] = { (ccv_nnc_tensor_t*)a, (ccv_nnc_tensor_t*)rotation };
	ccv_nnc_tensor_t* const outputs[] = { (ccv_nnc_tensor_t*)b };
	ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, 2, outputs, 1);
	int indices[2];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int rotation_nd = ccv_nnc_tensor_nd(rotation->info.dim);
	MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
		[inputTensors addObject:mps_input_a];
		MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
		[inputShapedTypes addObject:mps_a_shape];
		MPSGraphTensor* mps_input_rotation;
		MPSGraphTensor* mps_rotation = ccv_nnc_mps_graph_tensor_input(graph, rotation, rotation->info.dim, rotation->stride, &mps_input_rotation);
		[inputTensors addObject:mps_input_rotation];
		MPSGraphShapedType* mps_rotation_shape = ccv_nnc_mps_graph_tensor_input_shape(rotation, rotation->info.dim, rotation->stride);
		[inputShapedTypes addObject:mps_rotation_shape];
		if (a->info.datatype != CCV_32F)
			mps_a = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
		if (rotation->info.datatype != CCV_32F)
			mps_rotation = [graph castTensor:mps_rotation toType:MPSDataTypeFloat32 name:@"mps_rotation_float"];
		NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
		int i;
		for (i = 0; i < cmd.info.rmsnorm_cmul.count; i++)
			[axes addObject:@(cmd.info.rmsnorm_cmul.axis[i])];
		MPSGraphTensor* mps_square = [graph squareWithTensor:mps_a name:nil];
		MPSGraphTensor* mps_mean = [graph meanOfTensor:mps_square axes:axes name:nil];
		[axes release];
		MPSGraphTensor* mps_epsilon = [graph constantWithScalar:cmd.info.rmsnorm_cmul.epsilon dataType:MPSDataTypeFloat32];
		MPSGraphTensor* mps_inv_rms = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_mean secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
		MPSGraphTensor* mps_normalized = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_inv_rms name:nil];
		NSMutableArray<NSNumber*>* a_shape = [NSMutableArray new];
		for (i = 0; i < a_nd - 1; i++)
			[a_shape addObject:@(a->info.dim[i])];
		[a_shape addObject:@(a->info.dim[a_nd - 1] / 2)];
		[a_shape addObject:@2];
		mps_normalized = [graph reshapeTensor:mps_normalized withShape:a_shape name:nil];
		[a_shape release];
		NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_normalized numSplits:2 axis:a_nd name:nil];
		NSMutableArray<NSNumber*>* rotation_shape = [NSMutableArray new];
		for (i = 0; i < rotation_nd - 1; i++)
			[rotation_shape addObject:@(rotation->info.dim[i])];
		[rotation_shape addObject:@(rotation->info.dim[rotation_nd - 1] / 2)];
		[rotation_shape addObject:@2];
		mps_rotation = [graph reshapeTensor:mps_rotation withShape:rotation_shape name:nil];
		[rotation_shape release];
		NSArray<MPSGraphTensor*>* mps_rotation_splits = [graph splitTensor:mps_rotation numSplits:2 axis:rotation_nd name:nil];
		MPSGraphTensor* mps_real = [graph subtractionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_rotation_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_rotation_splits[1] name:nil] name:nil];
		MPSGraphTensor* mps_imag = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_rotation_splits[1] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_rotation_splits[0] name:nil] name:nil];
		NSMutableArray<NSNumber*>* b_shape = [NSMutableArray new];
		for (i = 0; i < a_nd; i++)
			[b_shape addObject:@(b->info.dim[i])];
		MPSGraphTensor* mps_b = [graph reshapeTensor:[graph concatTensor:mps_real withTensor:mps_imag dimension:a_nd name:nil] withShape:b_shape name:nil];
		[b_shape release];
		if (b->info.datatype != CCV_32F)
			mps_b = [graph castTensor:mps_b toType:ccv_nnc_mps_datatype(b->info.datatype) name:@"mps_b"];
		[resultTensors addObject:mps_b];
	});
	MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
	MPSGraphTensorData* data_rotation = ccv_nnc_mps_graph_tensor_data(rotation, rotation->info.dim, rotation->stride);
	MPSGraphTensorData* data[] = { data_a, data_rotation };
	ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
	ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rmsnorm_cmul_broadcastable(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const rotation)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int rotation_nd = ccv_nnc_tensor_nd(rotation->info.dim);
	if (a_nd < 1 || rotation_nd < 1 || a->info.dim[a_nd - 1] != rotation->info.dim[rotation_nd - 1])
		return 0;
	const int rotation_axis_offset = a_nd - rotation_nd;
	int i;
	for (i = 0; i < rotation_nd - 1; i++)
	{
		const int a_axis = i + rotation_axis_offset;
		if (a_axis < 0)
		{
			if (rotation->info.dim[i] != 1)
				return 0;
		} else if (rotation->info.dim[i] != 1 && rotation->info.dim[i] != a->info.dim[a_axis]) {
			return 0;
		}
	}
	return 1;
}

static int _ccv_nnc_rmsnorm_cmul_broadcast_ratio(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const rotation)
{
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	if (a_nd != ccv_nnc_tensor_nd(rotation->info.dim))
		return 0;
	int i;
	int equal = 1;
	for (i = 0; i < a_nd; i++)
		equal = equal && a->info.dim[i] == rotation->info.dim[i];
	if (equal)
		return 1;
	if (a_nd == 3 && a->info.dim[0] == rotation->info.dim[0] && rotation->info.dim[1] == 1 && a->info.dim[2] == rotation->info.dim[2])
		return a->info.dim[1];
	return 0;
}

static int _ccv_nnc_rmsnorm_cmul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const rotation = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd >= 1);
	assert(cmd.info.rmsnorm_cmul.count == 1);
	assert(cmd.info.rmsnorm_cmul.axis[0] >= 0 && cmd.info.rmsnorm_cmul.axis[0] < a_nd);
	assert(a->info.datatype == b->info.datatype);
	assert(ccv_nnc_tensor_count(a->info) == ccv_nnc_tensor_count(b->info));
	assert(_ccv_nnc_rmsnorm_cmul_broadcastable(a, rotation));
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == a_nd);
	int i;
	for (i = 0; i < a_nd; i++)
		assert(b->info.dim[i] == a->info.dim[i]);
	const int column_count = a->info.dim[a_nd - 1];
	assert(column_count > 0 && column_count % 2 == 0);
	const size_t row_count = ccv_nnc_tensor_count(a->info) / column_count;
	const int broadcast_ratio = _ccv_nnc_rmsnorm_cmul_broadcast_ratio(a, rotation);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		const int a_data_type = _ccv_nnc_rmsnorm_cmul_mfa_datatype(a->info.datatype);
		const int rotation_data_type = _ccv_nnc_rmsnorm_cmul_mfa_datatype(rotation->info.datatype);
		const int use_mfa = ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA) &&
			a_data_type >= 0 && rotation_data_type >= 0 && row_count <= UINT32_MAX &&
			cmd.info.rmsnorm_cmul.axis[0] == a_nd - 1 && column_count <= 512 && broadcast_ratio > 0 &&
			CCV_IS_TENSOR_CONTIGUOUS(a) && CCV_IS_TENSOR_CONTIGUOUS(rotation) && CCV_IS_TENSOR_CONTIGUOUS(b);
		if (!use_mfa)
			return _ccv_nnc_rmsnorm_cmul_fallback(cmd, hint, flags, a, rotation, b, stream_context);
		const ccv_nnc_mfa_rmsnorm_cmul_params_t params = {
			.epsilon = cmd.info.rmsnorm_cmul.epsilon,
			.a_data_type = (uint64_t)a_data_type,
			.rotation_data_type = (uint64_t)rotation_data_type,
			.row_count = (uint32_t)row_count,
			.column_count = (uint32_t)column_count,
			.broadcast_ratio = (uint32_t)broadcast_ratio,
		};
		ccv_nnc_mfa_prepare_rmsnorm_cmul(context, params);
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
		mtl_buffer_t* tensors[4] = {
			mpgetbuffer(inputs[0]), mpgetbuffer(inputs[1]), mpgetbuffer(outputs[0]), NULL,
		};
		size_t tensor_offsets[3] = { a->dataof, rotation->dataof, b->dataof };
		ccv_nnc_mfa_encode_rmsnorm_cmul(context, params, command_batch, tensors, tensor_offsets);
		ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_CMUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_cmul_forw;
}
