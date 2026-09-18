#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>

static int _ccv_nnc_signed_sqrt_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const int backward = cmd.cmd == CCV_NNC_SIGNED_SQRT_BACKWARD;
	const int used_inputs = backward ? 2 : 1;
	const float minimum = cmd.info.signed_sqrt.minimum_magnitude;
	if (input_size < used_inputs || output_size != 1 || !outputs[0] || !(minimum > 0) || !isfinite(minimum))
		return CCV_NNC_EXEC_INVALID;
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	if (b->info.datatype != CCV_32F && b->info.datatype != CCV_16F && b->info.datatype != CCV_16BF)
		return CCV_NNC_EXEC_INVALID;
	int i;
	int contiguous = CCV_IS_TENSOR_CONTIGUOUS(b);
	for (i = 0; i < used_inputs; i++)
	{
		if (!inputs[i] || inputs[i]->info.datatype != b->info.datatype || memcmp(inputs[i]->info.dim, b->info.dim, sizeof(b->info.dim)) != 0)
			return CCV_NNC_EXEC_INVALID;
		contiguous = contiguous && CCV_IS_TENSOR_CONTIGUOUS(inputs[i]);
	}
	const size_t count = ccv_nnc_tensor_count(b->info);
	if (count == 0)
		return CCV_NNC_EXEC_SUCCESS;
	@autoreleasepool {
		ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
		if (contiguous && count <= UINT32_MAX && ccv_nnc_mfa_context_supported(context) && !(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		{
			const ccv_nnc_mfa_signed_sqrt_params_t params = {
				.gradient = backward,
				.minimum_magnitude = minimum,
				.data_type = b->info.datatype == CCV_32F ? 3 : (b->info.datatype == CCV_16BF ? 121 : 16),
				.length = (uint32_t)count,
				.loadM = !!(ccv_nnc_flags() & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M),
			};
			ccv_nnc_mfa_prepare_signed_sqrt(context, params);
			mtl_buffer_t* tensors[4] = {};
			size_t offsets[3];
			for (i = 0; i < used_inputs; i++)
			{
				tensors[i] = mpgetbuffer(inputs[i]);
				offsets[i] = inputs[i]->dataof;
			}
			tensors[used_inputs] = mpgetbuffer(outputs[0]);
			offsets[used_inputs] = b->dataof;
			mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			ccv_nnc_mfa_encode_signed_sqrt(context, params, command_batch, tensors, offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			ccv_nnc_tensor_view_t graph_input_storage[2];
			const ccv_nnc_tensor_view_t* const graph_inputs = graph_input_storage;
			for (i = 0; i < used_inputs; i++)
			{
				memcpy(&graph_input_storage[i], inputs[i], CCV_IS_TENSOR_VIEW(inputs[i]) ? sizeof(ccv_nnc_tensor_view_t) : sizeof(ccv_nnc_tensor_t));
				if (CCV_IS_TENSOR_VIEW(inputs[i]))
				{
					const int a_nd = ccv_nnc_tensor_nd(inputs[i]->info.dim);
					int j, minimum_stride = graph_input_storage[i].stride[0];
					for (j = 1; j < a_nd; j++)
						minimum_stride = ccv_min(minimum_stride, graph_input_storage[i].stride[j]);
					if (minimum_stride > 1)
					{
						// The graph input helper needs a unit-stride innermost axis.
						// A singleton axis expresses stepped views without a copy.
						if (a_nd + 1 >= CCV_NNC_MAX_DIM_ALLOC)
							return CCV_NNC_EXEC_INVALID;
						graph_input_storage[i].info.dim[a_nd] = 1;
						graph_input_storage[i].info.dim[a_nd + 1] = 0;
						graph_input_storage[i].stride[a_nd] = 1;
						graph_input_storage[i].stride[a_nd + 1] = 0;
					}
				}
			}
			MPSCommandBuffer* const command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			const ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, used_inputs, outputs, output_size);
			int indices[2];
			MPSGraphExecutable* const executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* input_tensors, NSMutableArray<MPSGraphShapedType*>* input_shapes, NSMutableArray<MPSGraphTensor*>* result_tensors) {
				MPSGraphTensor* tensors[2];
				int j;
				for (j = 0; j < used_inputs; j++)
				{
					const ccv_nnc_tensor_view_t* const a = &graph_inputs[j];
					MPSGraphTensor* placeholder;
					tensors[j] = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &placeholder);
					[input_tensors addObject:placeholder];
					[input_shapes addObject:ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride)];
					if (memcmp(a->info.dim, b->info.dim, sizeof(b->info.dim)) != 0)
					{
						NSMutableArray<NSNumber*>* const shape = [NSMutableArray array];
						const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
						int k;
						for (k = 0; k < b_nd; k++)
							[shape addObject:@(b->info.dim[k])];
						tensors[j] = [graph reshapeTensor:tensors[j] withShape:shape name:nil];
					}
					if (a->info.datatype != CCV_32F)
						tensors[j] = [graph castTensor:tensors[j] toType:MPSDataTypeFloat32 name:nil];
				}
				MPSGraphTensor* const x = tensors[backward ? 1 : 0];
				MPSGraphTensor* const absolute = [graph absoluteWithTensor:x name:nil];
				MPSGraphTensor* const floor = [graph constantWithScalar:minimum dataType:MPSDataTypeFloat32];
				// Match fmax: a NaN magnitude selects the finite floor.
				MPSGraphTensor* const magnitude = [graph selectWithPredicateTensor:[graph isNaNWithTensor:absolute name:nil] truePredicateTensor:floor falsePredicateTensor:absolute name:nil];
				MPSGraphTensor* const root = [graph squareRootWithTensor:[graph maximumWithPrimaryTensor:magnitude secondaryTensor:floor name:nil] name:nil];
				MPSGraphTensor* output;
				if (backward)
				{
					MPSGraphTensor* const half_gradient = [graph multiplicationWithPrimaryTensor:tensors[0] secondaryTensor:[graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32] name:nil];
					MPSGraphTensor* const gradient = [graph divisionWithPrimaryTensor:half_gradient secondaryTensor:root name:nil];
					output = [graph selectWithPredicateTensor:[graph greaterThanOrEqualToWithPrimaryTensor:absolute secondaryTensor:floor name:nil] truePredicateTensor:gradient falsePredicateTensor:[graph constantWithScalar:0 dataType:MPSDataTypeFloat32] name:nil];
				} else {
					// Comparing x < 0 would lose the sign of negative zero.
					MPSGraphTensor* const bits = [graph reinterpretCastTensor:x toType:MPSDataTypeInt32 name:nil];
					MPSGraphTensor* const negative = [graph lessThanWithPrimaryTensor:bits secondaryTensor:[graph constantWithScalar:0 dataType:MPSDataTypeInt32] name:nil];
					output = [graph selectWithPredicateTensor:negative truePredicateTensor:[graph negativeWithTensor:root name:nil] falsePredicateTensor:root name:nil];
				}
				if (b->info.datatype != CCV_32F)
					output = [graph castTensor:output toType:ccv_nnc_mps_datatype(b->info.datatype) name:nil];
				[result_tensors addObject:output];
			});
			NSMutableArray<MPSGraphTensorData*>* const data = [NSMutableArray arrayWithCapacity:used_inputs];
			for (i = 0; i < used_inputs; i++)
			{
				const ccv_nnc_tensor_view_t* const a = &graph_inputs[indices[i]];
				[data addObject:ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride)];
			}
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, data, &b, (int*[]){b->info.dim}, (int*[]){b->stride}, 1, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGNED_SQRT_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_signed_sqrt_exec;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SIGNED_SQRT_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F | CCV_16BF;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_signed_sqrt_exec;
}
