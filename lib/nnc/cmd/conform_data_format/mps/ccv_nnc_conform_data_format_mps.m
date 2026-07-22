#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

#define CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE (64)

static int _ccv_nnc_conform_data_format_validate(const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_view_t* const b, int* const head_dim)
{
	if (cmd.info.conform_data_format.datatype != CCV_NNC_FP8_E4M3 || !a || !b)
		return 0;
	if (a->info.datatype != CCV_32F || b->info.datatype != CCV_32F || !CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b))
		return 0;
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	if (a_nd <= 0 || a_nd != b_nd)
		return 0;
	int i;
	for (i = 0; i < a_nd; i++)
		if (a->info.dim[i] != b->info.dim[i])
			return 0;
	*head_dim = a->info.dim[a_nd - 1];
	const int preserved_tail = cmd.info.conform_data_format.preserved_tail;
	if (*head_dim <= 0 || preserved_tail < 0 || preserved_tail > *head_dim || ((*head_dim - preserved_tail) % CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE) != 0)
		return 0;
	const size_t count = ccv_nnc_tensor_count(a->info);
	return count % *head_dim == 0;
}

static int _ccv_nnc_conform_data_format_copy(const ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_stream_context_t* const stream_context)
{
	@autoreleasepool {
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		id<MTLCommandBuffer> const mtl_command_buffer = command_buffer.commandBuffer;
		id<MTLBlitCommandEncoder> const encoder = [mtl_command_buffer blitCommandEncoder];
		if (mpgetbuffer((ccv_nnc_tensor_t*)a) != mpgetbuffer((ccv_nnc_tensor_t*)b) || mpgetoffset((ccv_nnc_tensor_t*)a) != mpgetoffset((ccv_nnc_tensor_t*)b))
			[encoder copyFromBuffer:mpgetbuffer((ccv_nnc_tensor_t*)a) sourceOffset:mpgetoffset((ccv_nnc_tensor_t*)a) toBuffer:mpgetbuffer((ccv_nnc_tensor_t*)b) destinationOffset:mpgetoffset((ccv_nnc_tensor_t*)b) size:sizeof(float) * ccv_nnc_tensor_count(a->info)];
		[encoder endEncoding];
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static MPSGraphTensor* _ccv_nnc_conform_data_format_abs(MPSGraph* const graph, MPSGraphTensor* const x)
{
	MPSGraphTensor* const zero = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
	return [graph maximumWithPrimaryTensor:x secondaryTensor:[graph subtractionWithPrimaryTensor:zero secondaryTensor:x name:nil] name:nil];
}

static float _ccv_nnc_conform_data_format_e4m3_value(const int i)
{
	static const float exp_scale[16] = {
		0.0f, 0.015625f, 0.03125f, 0.0625f,
		0.125f, 0.25f, 0.5f, 1.0f,
		2.0f, 4.0f, 8.0f, 16.0f,
		32.0f, 64.0f, 128.0f, 256.0f,
	};
	const int exp = (i >> 3) & 0xf;
	const int mant = i & 0x7;
	return exp == 0 ? (float)mant * 0.001953125f : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static MPSGraphTensor* _ccv_nnc_conform_data_format_bucket(MPSGraph* const graph, MPSGraphTensor* const x, MPSGraphTensor* const dequant, const float threshold, const float value, const int equal)
{
	MPSGraphTensor* const t = [graph constantWithScalar:threshold dataType:MPSDataTypeFloat32];
	MPSGraphTensor* const v = [graph constantWithScalar:value dataType:MPSDataTypeFloat32];
	MPSGraphTensor* const predicate = equal ? [graph greaterThanOrEqualToWithPrimaryTensor:x secondaryTensor:t name:nil] : [graph greaterThanWithPrimaryTensor:x secondaryTensor:t name:nil];
	return [graph selectWithPredicateTensor:predicate truePredicateTensor:v falsePredicateTensor:dequant name:nil];
}

static MPSGraphTensor* _ccv_nnc_conform_data_format_graph(MPSGraph* const graph, MPSGraphTensor* const x, NSArray<NSNumber*>* const original_shape, const size_t count, const int head_dim, const int preserved_tail)
{
	const int prefix = head_dim - preserved_tail;
	const int blocks_per_row = prefix / CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE;
	const size_t rows = count / head_dim;
	MPSGraphTensor* const rows_2d = [graph reshapeTensor:x withShape:@[@(rows), @(head_dim)] name:nil];
	MPSGraphTensor* const prefix_tensor = [graph sliceTensor:rows_2d dimension:1 start:0 length:prefix name:nil];
	MPSGraphTensor* const blocks = [graph reshapeTensor:prefix_tensor withShape:@[@(rows * blocks_per_row), @CCV_NNC_CONFORM_DATA_FORMAT_BLOCK_SIZE] name:nil];
	MPSGraphTensor* const abs_blocks = _ccv_nnc_conform_data_format_abs(graph, blocks);
	MPSGraphTensor* amax = [graph reductionMaximumWithTensor:abs_blocks axes:@[@1] name:nil];
	amax = [graph reshapeTensor:amax withShape:@[@(rows * blocks_per_row), @1] name:nil];
	amax = [graph maximumWithPrimaryTensor:amax secondaryTensor:[graph constantWithScalar:1.0e-4f dataType:MPSDataTypeFloat32] name:nil];
	MPSGraphTensor* scale_exp = [graph multiplicationWithPrimaryTensor:amax secondaryTensor:[graph constantWithScalar:(1.0f / 448.0f) dataType:MPSDataTypeFloat32] name:nil];
	scale_exp = [graph logarithmWithTensor:scale_exp name:nil];
	MPSGraphTensor* const log_2 = [graph constantWithScalar:0.69314718055994530942f dataType:MPSDataTypeFloat32];
	scale_exp = [graph ceilWithTensor:[graph divisionWithPrimaryTensor:scale_exp secondaryTensor:log_2 name:nil] name:nil];
	MPSGraphTensor* const scale = [graph exponentWithTensor:[graph multiplicationWithPrimaryTensor:scale_exp secondaryTensor:log_2 name:nil] name:nil];
	MPSGraphTensor* normalized = [graph divisionWithPrimaryTensor:blocks secondaryTensor:scale name:nil];
	normalized = [graph clampWithTensor:normalized minValueTensor:[graph constantWithScalar:-448.0f dataType:MPSDataTypeFloat32] maxValueTensor:[graph constantWithScalar:448.0f dataType:MPSDataTypeFloat32] name:nil];
	MPSGraphTensor* const abs_normalized = _ccv_nnc_conform_data_format_abs(graph, normalized);
	MPSGraphTensor* dequant_abs = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
	int i;
	for (i = 1; i < 127; i++)
	{
		const float previous = _ccv_nnc_conform_data_format_e4m3_value(i - 1);
		const float current = _ccv_nnc_conform_data_format_e4m3_value(i);
		dequant_abs = _ccv_nnc_conform_data_format_bucket(graph, abs_normalized, dequant_abs, (previous + current) * 0.5f, current, (i & 1) == 0);
	}
	MPSGraphTensor* const zero = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
	MPSGraphTensor* const sign = [graph selectWithPredicateTensor:[graph lessThanWithPrimaryTensor:normalized secondaryTensor:zero name:nil] truePredicateTensor:[graph constantWithScalar:-1.0f dataType:MPSDataTypeFloat32] falsePredicateTensor:[graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32] name:nil];
	MPSGraphTensor* quantized = [graph multiplicationWithPrimaryTensor:dequant_abs secondaryTensor:sign name:nil];
	quantized = [graph multiplicationWithPrimaryTensor:quantized secondaryTensor:scale name:nil];
	MPSGraphTensor* output = [graph reshapeTensor:quantized withShape:@[@(rows), @(prefix)] name:nil];
	if (preserved_tail > 0)
	{
		MPSGraphTensor* const tail = [graph sliceTensor:rows_2d dimension:1 start:prefix length:preserved_tail name:nil];
		output = [graph concatTensor:output withTensor:tail dimension:1 name:nil];
	}
	return [graph reshapeTensor:output withShape:original_shape name:nil];
}

static int _ccv_nnc_conform_data_format_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size != 1 || output_size != 1 || !inputs || !outputs)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	int head_dim;
	if (!_ccv_nnc_conform_data_format_validate(cmd, a, b, &head_dim))
		return CCV_NNC_EXEC_INVALID;
	const int preserved_tail = cmd.info.conform_data_format.preserved_tail;
	const int prefix = head_dim - preserved_tail;
	if (prefix == 0)
		return _ccv_nnc_conform_data_format_copy(a, b, stream_context);
	const size_t count = ccv_nnc_tensor_count(a->info);
	const size_t rows = count / head_dim;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (count <= UINT32_MAX && rows <= UINT32_MAX && ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const ccv_nnc_mfa_conform_data_format_params_t params = {
				.row_count = (uint32_t)rows,
				.head_dim = (uint32_t)head_dim,
				.preserved_tail = (uint32_t)preserved_tail,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_conform_data_format(context, params);
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[3] = { mpgetbuffer(inputs[0]), mpgetbuffer(outputs[0]), NULL };
			size_t tensor_offsets[2] = { (size_t)mpgetoffset(inputs[0]), (size_t)mpgetoffset(outputs[0]) };
			ccv_nnc_mfa_encode_conform_data_format(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
			return CCV_NNC_EXEC_SUCCESS;
		}

		NSMutableArray<NSNumber*>* const shape = [NSMutableArray arrayWithCapacity:CCV_NNC_MAX_DIM_ALLOC];
		const int nd = ccv_nnc_tensor_nd(a->info.dim);
		int i;
		for (i = 0; i < nd; i++)
			[shape addObject:@(a->info.dim[i])];
		MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* input_tensors, NSMutableArray<MPSGraphShapedType*>* input_shapes, NSMutableArray<MPSGraphTensor*>* result_tensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* const mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[input_tensors addObject:mps_input_a];
			[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride)];
			[result_tensors addObject:_ccv_nnc_conform_data_format_graph(graph, mps_a, shape, count, head_dim, preserved_tail)];
		});
		MPSGraphTensorData* const data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conform_data_format_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	if (input_size < 1 || output_size != 1 || !inputs || !outputs)
		return CCV_NNC_EXEC_INVALID;
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	int head_dim;
	if (!_ccv_nnc_conform_data_format_validate(cmd, g, h, &head_dim))
		return CCV_NNC_EXEC_INVALID;
	return _ccv_nnc_conform_data_format_copy(g, h, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONFORM_DATA_FORMAT_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conform_data_format_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONFORM_DATA_FORMAT_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conform_data_format_back;
}
