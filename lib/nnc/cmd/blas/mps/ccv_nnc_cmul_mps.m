#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_cmul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const ccv_nnc_tensor_t* const b = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	const size_t count = ccv_nnc_tensor_count(c->info) / 2;
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}

		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
			const int is_same_dtype =
				(a->info.datatype == b->info.datatype) &&
				(a->info.datatype == c->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (a->info.datatype) {
				case CCV_16F: {
					mtl_data_type = 16;
					break;
				}
				case CCV_32F: {
					mtl_data_type = 3;
					break;
				}
				default: {
					use_mfa = false;
					fallback_reason = "Unsupported data type.";
					break;
				}
			}
		}

		if (use_mfa) {
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) ||
					!CCV_IS_TENSOR_CONTIGUOUS(b) ||
					!CCV_IS_TENSOR_CONTIGUOUS(c))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		if (use_mfa) {
			ccv_nnc_mfa_cmul_params_t params = {
				.data_type = mtl_data_type,
				.astride = {0, 0, 0},
				.bstride = {0, 0, 0},
				.cstride = {0, 0, 0},
				.dim = {0, 0, 0, 0}
			};
			const size_t count = ccv_nnc_tensor_count(c->info);
			if (ccv_nnc_tensor_count(a->info) == count && ccv_nnc_tensor_count(b->info) == count) {
				params.dim[0] = count;
			} else {
				int i;
				int nd = ccv_nnc_tensor_nd(a->info.dim);
				assert(nd = ccv_nnc_tensor_nd(b->info.dim));
				assert(nd = ccv_nnc_tensor_nd(c->info.dim));
				int adim[CCV_NNC_MAX_DIM_ALLOC];
				int bdim[CCV_NNC_MAX_DIM_ALLOC];
				int cdim[CCV_NNC_MAX_DIM_ALLOC];
				int squeezed_dims = 0;
				for (i = nd - 1; i >= 0; i--)
				{
					if (c->info.dim[i] == 1)
						continue;
					adim[squeezed_dims] = a->info.dim[i];
					bdim[squeezed_dims] = b->info.dim[i];
					cdim[squeezed_dims] = c->info.dim[i];
					squeezed_dims += 1;
				}
				nd = squeezed_dims;
				int astride[CCV_NNC_MAX_DIM_ALLOC];
				int bstride[CCV_NNC_MAX_DIM_ALLOC];
				int cstride[CCV_NNC_MAX_DIM_ALLOC];
				astride[0] = 1;
				bstride[0] = 1;
				cstride[0] = 1;
				for (i = 1; i < nd; i++)
				{
					astride[i] = adim[i - 1] * astride[i - 1];
					bstride[i] = bdim[i - 1] * bstride[i - 1];
					cstride[i] = cdim[i - 1] * cstride[i - 1];
				}
				for (i = 0; i < nd; i++)
				{
					if (cdim[i] == adim[i] && cdim[i] == bdim[i])
						continue;
					if (cdim[i] == adim[i])
					{
						assert(bdim[i] == 1);
						bstride[i] = 0;
					} else {
						assert(cdim[i] == bdim[i]);
						assert(adim[i] == 1);
						astride[i] = 0;
					}
				}
				assert(nd <= 4);
				params.dim[0] = cdim[0];
				params.dim[1] = cdim[1];
				params.dim[2] = cdim[2];
				params.dim[3] = cdim[3];
				for (i = nd; i < 4; i++)
					params.dim[i] = 0;
				params.astride[0] = astride[1];
				params.astride[1] = astride[2];
				params.astride[2] = astride[3];
				params.bstride[0] = bstride[1];
				params.bstride[1] = bstride[2];
				params.bstride[2] = bstride[3];
				params.cstride[0] = cstride[1];
				params.cstride[1] = cstride[2];
				params.cstride[2] = cstride[3];
			}
			ccv_nnc_mfa_prepare_cmul(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]), // gradient
				mpgetbuffer(inputs[1]), // source
				mpgetbuffer(outputs[0]), // destination
				NULL,
			};
			size_t tensor_offsets[3] = {
				a->dataof,
				b->dataof,
				c->dataof
			};
			ccv_nnc_mfa_encode_cmul(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
			const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
			ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
			int nd = ccv_nnc_tensor_nd(a->info.dim);
			assert(nd = ccv_nnc_tensor_nd(b->info.dim));
			assert(nd = ccv_nnc_tensor_nd(c->info.dim));
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
				int i;
				// Reshape to [..., n / 2, 2]
				NSMutableArray<NSNumber*>* a_shape = [NSMutableArray new];
				for (i = 0; i < nd - 1; i++)
					[a_shape addObject:@(a->info.dim[i])];
				[a_shape addObject: @(a->info.dim[nd - 1] / 2)];
				[a_shape addObject: @2];
				mps_a = [graph reshapeTensor:mps_a withShape:a_shape name:nil];
				[a_shape release];
				NSArray<MPSGraphTensor*>* mps_a_splits = [graph splitTensor:mps_a numSplits:2 axis:nd name:nil];
				NSMutableArray<NSNumber*>* b_shape = [NSMutableArray new];
				for (i = 0; i < nd - 1; i++)
					[b_shape addObject:@(b->info.dim[i])];
				[b_shape addObject: @(b->info.dim[nd - 1] / 2)];
				[b_shape addObject: @2];
				mps_b = [graph reshapeTensor:mps_b withShape:b_shape name:nil];
				[b_shape release];
				NSArray<MPSGraphTensor*>* mps_b_splits = [graph splitTensor:mps_b numSplits:2 axis:nd name:nil];
				MPSGraphTensor* mps_c_0 = [graph subtractionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[0] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[1] name:nil] name:nil];
				MPSGraphTensor* mps_c_1 = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[0] secondaryTensor:mps_b_splits[1] name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_splits[1] secondaryTensor:mps_b_splits[0] name:nil] name:nil];
				NSMutableArray<NSNumber*>* c_shape = [NSMutableArray new];
				for (i = 0; i < nd; i++)
					[c_shape addObject:@(c->info.dim[i])];
				MPSGraphTensor* mps_c = [graph reshapeTensor:[graph concatTensor:mps_c_0 withTensor:mps_c_1 dimension:nd name:nil] withShape:c_shape name:nil];
				[resultTensors addObject:mps_c];
				[c_shape release];
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

REGISTER_COMMAND_BACKEND(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_cmul_forw;
}
