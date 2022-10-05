#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_ewdiv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
		{ assert(b->info.dim[i] == c->info.dim[i]); }
	if (a)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(a));
		assert(a->info.datatype == b->info.datatype);
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
			{ assert(a->info.dim[i] == b->info.dim[i]); }
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			MPSGraphTensor* mps_c = ccv_nnc_mps_graph_tensor_result(graph, [graph divisionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil], c);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data_c = ccv_nnc_mps_graph_tensor_data(c, c->info.dim, c->stride);
			[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a, mps_input_b: data_b} targetOperations:nil resultsDictionary:@{mps_c: data_c} executionDescriptor:nil];
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	} else {
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			MPSGraphTensor* mps_c = ccv_nnc_mps_graph_tensor_result(graph, [graph reciprocalWithTensor:mps_b name:nil], c);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data_c = ccv_nnc_mps_graph_tensor_data(c, c->info.dim, c->stride);
			[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_b: data_b} targetOperations:nil resultsDictionary:@{mps_c: data_c} executionDescriptor:nil];
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewdiv_forw;
}

static int _ccv_nnc_ewexp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == c->info.dim[i]); }
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
		MPSGraphTensor* mps_c = ccv_nnc_mps_graph_tensor_result(graph, [graph exponentWithTensor:mps_a name:nil], c);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_c = ccv_nnc_mps_graph_tensor_data(c, c->info.dim, c->stride);
		[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_c: data_c} executionDescriptor:nil];
		[graph release];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewexp_forw;
}

static int _ccv_nnc_ewlog_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == c->info.dim[i]); }
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
		MPSGraphTensor* mps_c = ccv_nnc_mps_graph_tensor_result(graph, [graph logarithmWithTensor:mps_a name:nil], c);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_c = ccv_nnc_mps_graph_tensor_data(c, c->info.dim, c->stride);
		[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_c: data_c} executionDescriptor:nil];
		[graph release];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewlog_forw;
}

static int _ccv_nnc_ewsqrt_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == c->info.dim[i]); }
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		MPSGraph *graph = [MPSGraph new];
		MPSGraphTensor* mps_input_a;
		MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
		MPSGraphTensor* mps_c = ccv_nnc_mps_graph_tensor_result(graph, [graph squareRootWithTensor:mps_a name:nil], c);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_c = ccv_nnc_mps_graph_tensor_data(c, c->info.dim, c->stride);
		[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_c: data_c} executionDescriptor:nil];
		[graph release];
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewsqrt_forw;
}

static int _ccv_nnc_clamp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	const float minv = cmd.info.clamp.min;
	const float maxv = cmd.info.clamp.max;
	assert(!isnan(minv) || !isnan(maxv));
	if (isnan(minv))
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensor* mps_c = [graph constantWithScalar:maxv dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_result(graph, [graph minimumWithPrimaryTensor:mps_a secondaryTensor:mps_c name:nil], b);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_b: data_b} executionDescriptor:nil];
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	} else if (isnan(maxv)) {
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensor* mps_c = [graph constantWithScalar:minv dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_result(graph, [graph maximumWithPrimaryTensor:mps_a secondaryTensor:mps_c name:nil], b);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_b: data_b} executionDescriptor:nil];
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	} else {
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
			MPSGraph *graph = [MPSGraph new];
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			MPSGraphTensor* mps_min = [graph constantWithScalar:minv dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_max = [graph constantWithScalar:maxv dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_result(graph, [graph clampWithTensor:mps_a minValueTensor:mps_min maxValueTensor:mps_max name:nil], b);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			[graph encodeToCommandBuffer:command_buffer feeds:@{mps_input_a: data_a} targetOperations:nil resultsDictionary:@{mps_b: data_b} executionDescriptor:nil];
			[graph release];
			[command_buffer commit];
			[command_buffer waitUntilCompleted];
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_clamp_forw;
}
