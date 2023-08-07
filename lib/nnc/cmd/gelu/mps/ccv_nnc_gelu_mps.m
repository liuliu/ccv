#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_gelu_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_b;
			if (cmd.info.gelu.tanh)
			{
				MPSGraphTensor* mps_x_3 = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:[graph squareWithTensor:mps_a name:nil] name:nil];
				MPSGraphTensor* mps_c0 = [graph constantWithScalar:0.044715 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_mul0 = [graph multiplicationWithPrimaryTensor:mps_x_3 secondaryTensor:mps_c0 name:nil];
				MPSGraphTensor* mps_x_sum = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_mul0 name:nil];
				MPSGraphTensor* mps_c1 = [graph constantWithScalar:0.797884560802865355 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_mul1 = [graph multiplicationWithPrimaryTensor:mps_x_sum secondaryTensor:mps_c1 name:nil];
				MPSGraphTensor* mps_tanh = [graph tanhWithTensor:mps_mul1 name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_tanh secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_half = [graph constantWithScalar:0.5 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				mps_b = [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_sum secondaryTensor:mps_a name:nil] secondaryTensor:mps_half name:nil];
			} else {
				MPSGraphTensor* mps_c = [graph constantWithScalar:0.70710678118654752440 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_x = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_c name:nil];
				MPSGraphTensor* mps_erf = [graph erfWithTensor:mps_x name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				MPSGraphTensor* mps_sum = [graph additionWithPrimaryTensor:mps_erf secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_half = [graph constantWithScalar:0.5 dataType:ccv_nnc_mps_datatype(a->info.datatype)];
				mps_b = [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_sum secondaryTensor:mps_a name:nil] secondaryTensor:mps_half name:nil];
			}
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

MPSGraphTensor* normcdf(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // (1.0f + erf(x*SQRT1_2)) * 0.5f * x;
  MPSDataType dataType = [inputTensor dataType];
  const float SQRT1_2 = 0.70710678118654752440;
  MPSGraphTensor* sqrt1_2 = [mpsGraph constantWithScalar:SQRT1_2 shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* onef = [mpsGraph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
  MPSGraphTensor* halff = [mpsGraph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];

  MPSGraphTensor* erf_tensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:sqrt1_2 name:nil];
  erf_tensor = [mpsGraph erfWithTensor:erf_tensor name:nil];
  erf_tensor = [mpsGraph additionWithPrimaryTensor:erf_tensor secondaryTensor:onef name:nil];
  erf_tensor = [mpsGraph multiplicationWithPrimaryTensor:erf_tensor secondaryTensor:halff name:nil];

  return erf_tensor;
}

static int _ccv_nnc_gelu_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
			[inputShapedTypes addObject:mps_g_shape];

			MPSGraphTensor* mps_input_b;
			MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
			[inputTensors addObject:mps_input_b];
			MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
			[inputShapedTypes addObject:mps_b_shape];
			MPSGraphTensor* inputTensor = mps_b;
			MPSGraphTensor* gradTensor = mps_g;
			MPSDataType dataType = mps_b.dataType;
			MPSGraphTensor* mps_h;
			if (cmd.info.gelu.tanh) {
				float kBeta = 0.797884560802865355 * (0.5f);
				float kKappa = 0.044715f;
				MPSGraphTensor* betaf = [graph constantWithScalar:kBeta shape:@[ @1 ] dataType:dataType];
				MPSGraphTensor* kappaf = [graph constantWithScalar:kKappa shape:@[ @1 ] dataType:dataType];
				MPSGraphTensor* halff = [graph constantWithScalar:0.5f shape:@[ @1 ] dataType:dataType];
				MPSGraphTensor* onef = [graph constantWithScalar:1.0f shape:@[ @1 ] dataType:dataType];
				MPSGraphTensor* threef = [graph constantWithScalar:3.0f shape:@[ @1 ] dataType:dataType];
				MPSGraphTensor* x_sq = [graph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:inputTensor name:nil];
				MPSGraphTensor* x_cube = [graph multiplicationWithPrimaryTensor:x_sq secondaryTensor:inputTensor name:nil];
				MPSGraphTensor* inner = [graph multiplicationWithPrimaryTensor:kappaf secondaryTensor:x_cube name:nil];
				inner = [graph additionWithPrimaryTensor:inner secondaryTensor:inputTensor name:nil];
				inner = [graph multiplicationWithPrimaryTensor:betaf secondaryTensor:inner name:nil];
				MPSGraphTensor* tanhInner = [graph tanhWithTensor:inner name:nil];
				MPSGraphTensor* left = [graph multiplicationWithPrimaryTensor:halff secondaryTensor:inputTensor name:nil];
				MPSGraphTensor* right = [graph additionWithPrimaryTensor:onef secondaryTensor:tanhInner name:nil];
				MPSGraphTensor* left_derivative = [graph multiplicationWithPrimaryTensor:halff secondaryTensor:right name:nil];
				MPSGraphTensor* tanh_derivative = [graph multiplicationWithPrimaryTensor:tanhInner secondaryTensor:tanhInner name:nil];
				tanh_derivative = [graph subtractionWithPrimaryTensor:onef secondaryTensor:tanh_derivative name:nil];
				MPSGraphTensor* inner_derivative = [graph multiplicationWithPrimaryTensor:threef secondaryTensor:kappaf name:nil];
				inner_derivative = [graph multiplicationWithPrimaryTensor:inner_derivative secondaryTensor:x_sq name:nil];
				inner_derivative = [graph additionWithPrimaryTensor:inner_derivative secondaryTensor:onef name:nil];
				inner_derivative = [graph multiplicationWithPrimaryTensor:betaf secondaryTensor:inner_derivative name:nil];
				MPSGraphTensor* right_derivative = [graph multiplicationWithPrimaryTensor:left secondaryTensor:tanh_derivative name:nil];
				right_derivative = [graph multiplicationWithPrimaryTensor:right_derivative secondaryTensor:inner_derivative name:nil];
				mps_h = [graph additionWithPrimaryTensor:left_derivative secondaryTensor:right_derivative name:nil];
				mps_h = [graph multiplicationWithPrimaryTensor:gradTensor secondaryTensor:mps_h name:nil];
			} else {
				float kBeta = 0.797884560802865355;
				MPSGraphTensor* halff = [graph constantWithScalar:-0.5f dataType:dataType];
				MPSGraphTensor* betaf = [graph constantWithScalar:kBeta dataType:dataType];
				MPSGraphTensor* cdf = normcdf(graph, inputTensor);
				MPSGraphTensor* pdfMul = [graph squareWithTensor:inputTensor name:nil];
				pdfMul = [graph multiplicationWithPrimaryTensor:pdfMul secondaryTensor:halff name:nil];
				pdfMul = [graph exponentWithTensor:pdfMul name:nil];
				MPSGraphTensor* pdf = [graph multiplicationWithPrimaryTensor:pdfMul secondaryTensor:betaf name:nil];
				pdf = [graph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:pdf name:nil];
				pdf = [graph additionWithPrimaryTensor:pdf secondaryTensor:cdf name:nil];
				mps_h = [graph multiplicationWithPrimaryTensor:gradTensor secondaryTensor:pdf name:nil];
			}

			[resultTensors addObject:mps_h];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
		MPSGraphTensorData* data[] = {data_g, data_b};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &h, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GELU_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gelu_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GELU_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gelu_back;
}
