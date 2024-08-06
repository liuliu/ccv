#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef HAVE_MPS
#include "nnc/mps/ccv_nnc_mps.h"
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

static int _ccv_nnc_add_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)outputs[0];
	if (inputs[1] == 0)
	{
		@autoreleasepool {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			if (p == 1)
			{
				MPSGraph* graph = [MPSGraph new];
				graph.options = MPSGraphOptionsSynchronizeResults;
				MPSGraphTensor* mps_input_a;
				MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				if (mps_a != mps_input_a)
					ccv_nnc_mps_graph_result(graph, command_buffer, @{mps_input_a: data_a}, mps_a, c, c->info.dim, c->stride);
				else
					ccv_nnc_mps_export_data(data_a, command_buffer, c, c->info.dim, c->stride);
				[graph release];
			} else if (p == 0.5 || p == 2 || p == 1.0 / 3 || p == 3 || p == 1.0 / 10 || p == 10) { // Only create specialized kernels for special p values.
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[1];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			} else {
				ccv_nnc_cmd_t cmd_without_p = cmd;
				cmd_without_p.info.blas.a[0] = 0;
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd_without_p, 1, hint, flags, inputs, input_size, outputs, output_size);
				int indices[2];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					MPSGraphTensor* mps_p = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype) name:nil];
					[inputTensors addObject:mps_p];
					MPSGraphShapedType* mps_p_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					[inputShapedTypes addObject:mps_p_shape];
					[mps_p_shape release];
					MPSGraphTensor* mps_c = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_p = ccv_nnc_mps_graph_constant_data(p, a->info.datatype);
				MPSGraphTensorData* data[] = {data_a, data_p};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
		return CCV_NNC_EXEC_SUCCESS;
	}
	const ccv_nnc_tensor_view_t* const b = (const ccv_nnc_tensor_view_t*)inputs[1];
	@autoreleasepool {
		bool use_mfa = true;
		const char *fallback_reason = NULL;
		ccv_nnc_mfa_context_t* context = ccv_nnc_default_mfa_context();

		if (!ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_METAL_FLASH_ATTENTION)) {
			use_mfa = false;
			fallback_reason = "Disabled.";
		}
		if (use_mfa) {
			if (!(p == 1 && q == 1)) {
				use_mfa = false;
				fallback_reason = "Unsupported coefficients.";
			}
		}

		if (use_mfa) {
			if (a->info.datatype != CCV_16F && a->info.datatype != CCV_32F) {
				use_mfa = false;
				fallback_reason = "Unsupported data type.";
			}
			if (b->info.datatype != CCV_16F && b->info.datatype != CCV_32F) {
				use_mfa = false;
				fallback_reason = "Unsupported data type.";
			}
			if (c->info.datatype != CCV_16F && c->info.datatype != CCV_32F) {
				use_mfa = false;
				fallback_reason = "Unsupported data type.";
			}
		}
		if (use_mfa) {
			if (!(a->info.datatype == b->info.datatype && b->info.datatype == c->info.datatype)) {
				use_mfa = false;
				fallback_reason = "Mismatched data type.";
			}
		}
		const size_t length = ccv_nnc_tensor_count(c->info);
		if (use_mfa) {
			if (!(ccv_nnc_tensor_count(a->info) == length && ccv_nnc_tensor_count(b->info) == length)) {
				use_mfa = false;
				fallback_reason = "Broadcast semantics unsupported.";
			}
		}
		if (use_mfa) {
			if (!CCV_IS_TENSOR_CONTIGUOUS(a) || !CCV_IS_TENSOR_CONTIGUOUS(b) || !CCV_IS_TENSOR_CONTIGUOUS(c)) {
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}
		uint32_t mtl_data_type = UINT32_MAX;
		if (use_mfa) {
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
			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			ccv_nnc_mfa_add_params_t params = {
				.data_type = mtl_data_type,
				.length = (uint32_t)length,
			};
			ccv_nnc_mfa_prepare_add(context, params);

			mtl_buffer_t* tensors[4] = {
				mpgetbuffer(inputs[0]),
				mpgetbuffer(inputs[1]),
				mpgetbuffer(outputs[0]),
				NULL
			};
			size_t tensor_offsets[3] = {
				a->dataof,
				b->dataof,
				c->dataof,
			};
			ccv_nnc_mfa_encode_add(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			if (p == 1 && q == 1)
			{
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
					MPSGraphTensor* mps_c = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
				MPSGraphTensorData* data[] = {data_a, data_b};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			} else if ((p == 1 || p == 0.5 || p == 2 || p == 1.0 / 3 || p == 3 || p == 1.0 / 10 || p == 10) && (q == 1 || q == 0.5 || q == 2 || q == 1.0 / 3 || q == 3 || q == 1.0 / 10 || q == 10)) { // Only create specialized kernels for special p / q values.
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[2];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					if (p != 1)
					{
						MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(a->info.datatype)];
						mps_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
					}
					MPSGraphTensor* mps_input_b;
					MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
					[inputTensors addObject:mps_input_b];
					MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
					[inputShapedTypes addObject:mps_b_shape];
					if (q != 1)
					{
						MPSGraphTensor* mps_q = [graph constantWithScalar:q dataType:ccv_nnc_mps_datatype(b->info.datatype)];
						mps_b = [graph multiplicationWithPrimaryTensor:mps_b secondaryTensor:mps_q name:nil];
					}
					MPSGraphTensor* mps_c = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
				MPSGraphTensorData* data[] = {data_a, data_b};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			} else {
				ccv_nnc_cmd_t cmd_without_p_q = cmd;
				cmd_without_p_q.info.blas.a[0] = 0;
				cmd_without_p_q.info.blas.a[1] = 0;
				ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd_without_p_q, 1, hint, flags, inputs, input_size, outputs, output_size);
				int indices[4];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_a;
					MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
					[inputTensors addObject:mps_input_a];
					MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
					[inputShapedTypes addObject:mps_a_shape];
					MPSGraphTensor* mps_p = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype) name:nil];
					[inputTensors addObject:mps_p];
					MPSGraphShapedType* mps_p_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(a->info.datatype)];
					[inputShapedTypes addObject:mps_p_shape];
					mps_a = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_p name:nil];
					MPSGraphTensor* mps_input_b;
					MPSGraphTensor* mps_b = ccv_nnc_mps_graph_tensor_input(graph, b, b->info.dim, b->stride, &mps_input_b);
					[inputTensors addObject:mps_input_b];
					MPSGraphShapedType* mps_b_shape = ccv_nnc_mps_graph_tensor_input_shape(b, b->info.dim, b->stride);
					[inputShapedTypes addObject:mps_b_shape];
					MPSGraphTensor* mps_q = [graph placeholderWithShape:@[@1] dataType:ccv_nnc_mps_datatype(b->info.datatype) name:nil];
					[inputTensors addObject:mps_q];
					MPSGraphShapedType* mps_q_shape = [[MPSGraphShapedType alloc] initWithShape:@[@1] dataType:ccv_nnc_mps_datatype(b->info.datatype)];
					[inputShapedTypes addObject:mps_q_shape];
					mps_b = [graph multiplicationWithPrimaryTensor:mps_b secondaryTensor:mps_q name:nil];
					MPSGraphTensor* mps_c = [graph additionWithPrimaryTensor:mps_a secondaryTensor:mps_b name:nil];
					[resultTensors addObject:mps_c];
				});
				MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
				MPSGraphTensorData* data_p = ccv_nnc_mps_graph_constant_data(p, a->info.datatype);
				MPSGraphTensorData* data_b = ccv_nnc_mps_graph_tensor_data(b, b->info.dim, b->stride);
				MPSGraphTensorData* data_q = ccv_nnc_mps_graph_constant_data(q, a->info.datatype);
				MPSGraphTensorData* data[] = {data_a, data_p, data_b, data_q};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]]], &c, (int*[]){ c->info.dim }, (int*[]){ c->stride }, 1, 0);
			}
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_add_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 1);
	assert(output_size >= 1);
	const float p = cmd.info.blas.a[0];
	const float q = cmd.info.blas.a[1];
	const ccv_nnc_tensor_view_t* const g = (const ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const b = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);

		if (a && b)
		{
				ccv_nnc_mps_graph_key_t a_key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
				int indices[1];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(a_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphTensor* mps_a = mps_g;
					if (p != 1)
					{
						MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(g->info.datatype)];
						mps_a = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_p name:nil];
					}
					const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
					NSMutableArray<NSNumber*>* da_axes = [NSMutableArray new];
					for (int i = 0; i < a_nd; i++) {
						if (a->info.dim[i] != g->info.dim[i])
							[da_axes addObject:@(i)];
					}
					if (da_axes.count > 0)
						mps_a = [graph reductionSumWithTensor:mps_a axes:da_axes name:nil];
					[da_axes release];
					[resultTensors addObject:mps_a];
					MPSGraphTensor* mps_b = mps_g;
					if (q != 1)
					{
						MPSGraphTensor* mps_q = [graph constantWithScalar:q dataType:ccv_nnc_mps_datatype(g->info.datatype)];
						mps_b = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_q name:nil];
					}
					NSMutableArray<NSNumber*>* db_axes = [NSMutableArray new];
					const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
					for (int i = 0; i < b_nd; i++) {
						if (b->info.dim[i] != g->info.dim[i])
							[db_axes addObject:@(i)];
					}
					if (db_axes.count > 0)
						mps_b = [graph reductionSumWithTensor:mps_b axes:db_axes name:nil];
					[db_axes release];
					[resultTensors addObject:mps_b];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data[] = {data_g};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], (ccv_nnc_tensor_view_t* []){ a, b }, (int*[]){ a->info.dim, b->info.dim }, (int*[]){ a->stride, b->stride }, 2, 0);
		} else {
			if (a) {
				ccv_nnc_mps_graph_key_t a_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
				int indices[1];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(a_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphTensor* mps_a = mps_g;
					if (p != 1)
					{
						MPSGraphTensor* mps_p = [graph constantWithScalar:p dataType:ccv_nnc_mps_datatype(g->info.datatype)];
						mps_a = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_p name:nil];
					}
					const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
					NSMutableArray<NSNumber*>* da_axes = [NSMutableArray new];
					for (int i = 0; i < a_nd; i++) {
						if (a->info.dim[i] != g->info.dim[i])
							[da_axes addObject:@(i)];
					}
					if (da_axes.count > 0)
						mps_a = [graph reductionSumWithTensor:mps_a axes:da_axes name:nil];
					[da_axes release];
					[resultTensors addObject:mps_a];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data[] = {data_g};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], (ccv_nnc_tensor_view_t* []){ a }, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1, 0);
			}
			if (b) {
				ccv_nnc_mps_graph_key_t b_key = ccv_nnc_mps_graph_key_new(cmd, 2, hint, flags, inputs, input_size, outputs, output_size);
				int indices[1];
				MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(b_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
					MPSGraphTensor* mps_input_g;
					MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
					[inputTensors addObject:mps_input_g];
					MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
					[inputShapedTypes addObject:mps_g_shape];

					MPSGraphTensor* mps_b = mps_g;
					if (q != 1)
					{
						MPSGraphTensor* mps_q = [graph constantWithScalar:q dataType:ccv_nnc_mps_datatype(g->info.datatype)];
						mps_b = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_q name:nil];
					}
					NSMutableArray<NSNumber*>* db_axes = [NSMutableArray new];
					const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
					for (int i = 0; i < b_nd; i++) {
						if (b->info.dim[i] != g->info.dim[i])
							[db_axes addObject:@(i)];
					}
					if (db_axes.count > 0)
						mps_b = [graph reductionSumWithTensor:mps_b axes:db_axes name:nil];
					[db_axes release];
					[resultTensors addObject:mps_b];
				});
				MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
				MPSGraphTensorData* data[] = {data_g};
				ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]]], (ccv_nnc_tensor_view_t* []){ b }, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
			}
		}
		
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ADD_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_add_back;
}
