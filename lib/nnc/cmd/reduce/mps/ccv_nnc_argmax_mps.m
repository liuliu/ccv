#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

// Keep this state expansion identical to the production MPS random commands.
#define CCV_NNC_SRAND48_A 0x5DEECE66DULL
#define CCV_NNC_SRAND48_C 0xBULL
#define CCV_NNC_SRAND48_M (1ULL << 48)

static uint64_t _ccv_nnc_argmax_stateless_srand48(const long seed)
{
	return (((uint64_t)seed) << 16) | 0x330EULL;
}

static uint32_t _ccv_nnc_argmax_stateless_mrand48(uint64_t* const seed)
{
	seed[0] = (CCV_NNC_SRAND48_A * seed[0] + CCV_NNC_SRAND48_C) & (CCV_NNC_SRAND48_M - 1);
	return (uint32_t)(seed[0] >> 16);
}

static void _ccv_nnc_gumbel_argmax_random_state(const uint32_t seed, uint32_t states[7])
{
	states[0] = 1;
	uint64_t rand48_seed = _ccv_nnc_argmax_stateless_srand48((long)seed);
	states[2] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed); // counterLow
	states[1] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed);
	states[4] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed); // counterHigh
	states[3] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed);
	states[6] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed); // key
	states[5] = _ccv_nnc_argmax_stateless_mrand48(&rand48_seed);
}

static int _ccv_nnc_argmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_view_t atv = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t btv = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t* tvs[] = {
		&atv, &btv
	};
	ccv_nnc_tensor_view_alignment(tvs, 2);
	assert(atv.info.datatype == CCV_32F || atv.info.datatype == CCV_16F || atv.info.datatype == CCV_16BF);
	const int a_nd = ccv_nnc_tensor_nd(atv.info.dim);
	int noop = 1;
	int i;
	for (i = 0; noop && i < a_nd; i++)
		noop = btv.info.dim[i] != atv.info.dim[i];
	const int axis = cmd.info.reduce.axis[0];
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (cmd.info.reduce.count == 1 && a_nd >= 1 && axis == a_nd - 1 &&
			CCV_IS_TENSOR_CONTIGUOUS(inputs[0]) && CCV_IS_TENSOR_CONTIGUOUS(outputs[0]) &&
			btv.info.datatype == CCV_32S && ccv_nnc_mfa_context_supported(context) &&
			!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const size_t column_count = atv.info.dim[axis];
			const size_t a_count = ccv_nnc_tensor_count(atv.info);
			const size_t row_count = column_count > 0 ? a_count / column_count : 0;
			if (column_count > 0 && a_count % column_count == 0 && row_count > 0 &&
				row_count <= UINT32_MAX && column_count <= UINT32_MAX &&
				ccv_nnc_tensor_count(btv.info) == row_count)
			{
				const ccv_nnc_mfa_argmax_params_t params = {
					.data_type = atv.info.datatype == CCV_16F ? 16 : (atv.info.datatype == CCV_16BF ? 121 : 3),
					.row_count = (uint32_t)row_count,
					.column_count = (uint32_t)column_count,
				};
				ccv_nnc_mfa_prepare_argmax(context, params);
				mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer(inputs[0]), mpgetbuffer(outputs[0]), NULL,
				};
				size_t tensor_offsets[2] = {
					(size_t)mpgetoffset(inputs[0]), (size_t)mpgetoffset(outputs[0]),
				};
				ccv_nnc_mfa_encode_argmax(context, params, command_batch, tensors, tensor_offsets);
				ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
				return CCV_NNC_EXEC_SUCCESS;
			}
		}
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		if (noop)
		{
			MPSGraph* graph = [MPSGraph new];
			graph.options = MPSGraphOptionsSynchronizeResults;
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &atv, atv.info.dim, atv.stride, &mps_input_a);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&atv, atv.info.dim, atv.stride);
			if (mps_a != mps_input_a)
				ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, &btv, btv.info.dim, btv.stride);
			else
				ccv_nnc_mps_export_data(data_a, command_buffer, &btv, btv.info.dim, btv.stride);
			[graph release];
		} else {
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &atv, atv.info.dim, atv.stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(&atv, atv.info.dim, atv.stride);
				[inputShapedTypes addObject:mps_a_shape];
				// On macOS 26.4.1 (25E253), MPSGraph's native BF16
				// reductionArgMaximumWithTensor can return an incorrect index for
				// larger reduction axes. Upcast BF16 to FP32 before argmax.
				if (atv.info.datatype == CCV_16BF)
					mps_a = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
				MPSGraphTensor* mps_b = [graph reductionArgMaximumWithTensor:mps_a axis:axis name:nil];
				[resultTensors addObject:mps_b];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&atv, atv.info.dim, atv.stride);
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &tvs[1], (int*[]){ btv.info.dim }, (int*[]){ btv.stride }, 1, 0);
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gumbel_argmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	ccv_nnc_tensor_view_t atv = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t btv = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t* tvs[] = {
		&atv, &btv
	};
	ccv_nnc_tensor_view_alignment(tvs, 2);
	assert(atv.info.datatype == CCV_32F || atv.info.datatype == CCV_16F || atv.info.datatype == CCV_16BF);
	assert(cmd.info.reduce.count == 1);
	const int a_nd = ccv_nnc_tensor_nd(atv.info.dim);
	const int axis = cmd.info.reduce.axis[0];
	const float scale = cmd.info.reduce.scale;
	const uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
	uint32_t states[7];
	_ccv_nnc_gumbel_argmax_random_state(seed, states);
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (a_nd >= 1 && axis == a_nd - 1 && CCV_IS_TENSOR_CONTIGUOUS(inputs[0]) &&
			CCV_IS_TENSOR_CONTIGUOUS(outputs[0]) && btv.info.datatype == CCV_32S &&
			ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const size_t column_count = atv.info.dim[axis];
			const size_t a_count = ccv_nnc_tensor_count(atv.info);
			const size_t row_count = column_count > 0 ? a_count / column_count : 0;
			if (column_count > 0 && a_count % column_count == 0 && row_count > 0 &&
				row_count <= UINT32_MAX && column_count <= UINT32_MAX &&
				ccv_nnc_tensor_count(btv.info) == row_count)
			{
				ccv_nnc_mfa_argmax_params_t params = {
					.data_type = atv.info.datatype == CCV_16F ? 16 : (atv.info.datatype == CCV_16BF ? 121 : 3),
					.row_count = (uint32_t)row_count,
					.column_count = (uint32_t)column_count,
					.scale = scale,
					.gumbel = 1,
				};
				memcpy(params.state, states, sizeof(params.state));
				ccv_nnc_mfa_prepare_argmax(context, params);
				mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
				mtl_buffer_t* tensors[3] = {
					mpgetbuffer(inputs[0]), mpgetbuffer(outputs[0]), NULL,
				};
				size_t tensor_offsets[2] = {
					(size_t)mpgetoffset(inputs[0]), (size_t)mpgetoffset(outputs[0]),
				};
				ccv_nnc_mfa_encode_argmax(context, params, command_batch, tensors, tensor_offsets);
				ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
				return CCV_NNC_EXEC_SUCCESS;
			}
		}
		NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
		int i;
		for (i = 0; i < a_nd; i++)
			[shape addObject:@(atv.info.dim[i])];
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[2];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &atv, atv.info.dim, atv.stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(&atv, atv.info.dim, atv.stride);
			[inputShapedTypes addObject:mps_a_shape];
			if (atv.info.datatype != CCV_32F)
				mps_a = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
			MPSGraphTensor* mps_state = [graph placeholderWithShape:@[@7] dataType:MPSDataTypeInt32 name:nil];
			[inputTensors addObject:mps_state];
			MPSGraphShapedType* mps_state_shape = [[MPSGraphShapedType alloc] initWithShape:@[@7] dataType:MPSDataTypeInt32];
			[inputShapedTypes addObject:mps_state_shape];
			[mps_state_shape release];
			MPSGraphRandomOpDescriptor* descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform dataType:MPSDataTypeFloat32];
			descriptor.min = 0;
			descriptor.max = 1;
			MPSGraphTensor* mps_uniform = [graph randomTensorWithShape:shape descriptor:descriptor stateTensor:mps_state name:nil][0];
			MPSGraphTensor* mps_log_uniform = [graph logarithmWithTensor:mps_uniform name:nil];
			MPSGraphTensor* mps_neg_log_uniform = [graph negativeWithTensor:mps_log_uniform name:nil];
			MPSGraphTensor* mps_log_neg_log_uniform = [graph logarithmWithTensor:mps_neg_log_uniform name:nil];
			MPSGraphTensor* mps_gumbel = [graph negativeWithTensor:mps_log_neg_log_uniform name:nil];
			mps_gumbel = [graph multiplicationWithPrimaryTensor:mps_gumbel secondaryTensor:[graph constantWithScalar:scale dataType:MPSDataTypeFloat32] name:nil];
			mps_a = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_gumbel name:nil];
			MPSGraphTensor* mps_b = [graph reductionArgMaximumWithTensor:mps_a axis:axis name:nil];
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&atv, atv.info.dim, atv.stride);
		NSData* state = [[NSData alloc] initWithBytesNoCopy:states length:sizeof(states) freeWhenDone:NO];
		MPSGraphTensorData* data_state = [[MPSGraphTensorData alloc] initWithDevice:ccv_nnc_default_mps_device() data:state shape:@[@7] dataType:MPSDataTypeInt32];
		MPSGraphTensorData* data[] = { data_a, data_state };
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &tvs[1], (int*[]){ btv.info.dim }, (int*[]){ btv.stride }, 1, 0);
		[shape release];
		[data_state release];
		[state release];
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argmax_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_32S | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gumbel_argmax_forw;
}
