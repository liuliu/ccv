#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_swish_mul_mfa_datatype(const int datatype)
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

static int _ccv_nnc_swish_mul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const float beta = cmd.info.swish_mul.beta;
	const float scale = cmd.info.swish_mul.scale;
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(c->info.datatype == a->info.datatype);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
	{
		assert(a->info.dim[i] == b->info.dim[i]);
		assert(a->info.dim[i] == c->info.dim[i]);
	}
	@autoreleasepool {
		bool use_mfa = true;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
			use_mfa = false;

		const int a_data_type = _ccv_nnc_swish_mul_mfa_datatype(a->info.datatype);
		const int b_data_type = _ccv_nnc_swish_mul_mfa_datatype(b->info.datatype);
		if (use_mfa && (a_data_type < 0 || b_data_type < 0))
			use_mfa = false;

		if (use_mfa && (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b) || !CCV_IS_TENSOR_CONTIGUOUS(c)))
			use_mfa = false;

		if (use_mfa) {
			ccv_nnc_mfa_swish_mul_params_t params = {
				.beta = beta,
				.scale = scale,
				.a_data_type = (uint64_t)a_data_type,
				.b_data_type = (uint64_t)b_data_type,
				.length = (uint32_t)ccv_nnc_tensor_count(a->info)
			};
			ccv_nnc_mfa_prepare_swish_mul(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]), // value
				mpgetbuffer(inputs[1]), // gate
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[3] = {
				a->dataof,
				b->dataof,
				c->dataof
			};
			ccv_nnc_mfa_encode_swish_mul(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				[inputTensors addObject:mps_input_a];
				MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
				[inputShapedTypes addObject:mps_a_shape];
				MPSGraphTensor* mps_input_b;
				MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
				[inputTensors addObject:mps_input_b];
				MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
				[inputShapedTypes addObject:mps_b_shape];
				MPSGraphTensor* mps_a_f32 = mps_a;
				if (a->info.datatype != CCV_32F)
					mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];
				MPSGraphTensor* mps_b_f32 = mps_b;
				if (b->info.datatype != CCV_32F)
					mps_b_f32 = [graph castTensor:mps_b toType:MPSDataTypeFloat32 name:@"mps_b_float"];
				MPSGraphTensor* mps_beta = [graph constantWithScalar:beta dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_beta_b = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_beta name:nil];
				MPSGraphTensor* mps_neg = [graph negativeWithTensor:mps_beta_b name:nil];
				MPSGraphTensor* mps_exp = [graph exponentWithTensor:mps_neg name:nil];
				MPSGraphTensor* mps_one = [graph constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_denom = [graph additionWithPrimaryTensor:mps_exp secondaryTensor:mps_one name:nil];
				MPSGraphTensor* mps_sigmoid = [graph divisionWithPrimaryTensor:mps_one secondaryTensor:mps_denom name:nil];
				MPSGraphTensor* mps_swish = [graph multiplicationWithPrimaryTensor:mps_b_f32 secondaryTensor:mps_sigmoid name:nil];
				MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_swish name:nil];
				if (scale != 1)
				{
					MPSGraphTensor* mps_scale = [graph constantWithScalar:scale dataType:MPSDataTypeFloat32];
					mps_c = [graph multiplicationWithPrimaryTensor:mps_c secondaryTensor:mps_scale name:nil];
				}
				if (c->info.datatype != CCV_32F)
					mps_c = [graph castTensor:mps_c toType:ccv_nnc_mps_datatype(c->info.datatype) name:@"mps_c"];
				[resultTensors addObject:mps_c];
			});
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
			MPSGraphTensorData* data[] = {data_a, data_b};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SWISH_MUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_swish_mul_forw;
}
