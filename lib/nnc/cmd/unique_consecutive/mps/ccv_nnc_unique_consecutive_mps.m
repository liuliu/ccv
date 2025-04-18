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

static int _ccv_nnc_unique_consecutive_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	assert(a->info.datatype == b->info.datatype);
	const int count = ccv_nnc_tensor_count(a->info);
	assert(a_nd == 1); // Can only handle 1d tensor for this.
	const int bincount = b->info.dim[0];
	assert(bincount > 0);
	assert(bincount == indices->info.dim[0]);
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int idx[1];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, idx, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* front_n_minus_one = [graph sliceTensor:mps_a dimension:0 start:0 length:count - 1 name:nil];
			MPSGraphTensor* back_n_minus_one = [graph sliceTensor:mps_a dimension:0 start:1 length:count - 1 name:nil];
			MPSGraphTensor* not_equal_to_prev = [graph notEqualWithPrimaryTensor:back_n_minus_one secondaryTensor:front_n_minus_one name:nil];
			MPSGraphTensor* mask = [graph castTensor:not_equal_to_prev toType:MPSDataTypeInt32 name:nil];
			MPSGraphTensor* scanned_indices = [graph cumulativeSumWithTensor:mask axis:0 name:nil];
			MPSGraphTensor* minus_one = [graph constantWithScalar:-1.0f dataType:MPSDataTypeInt32];
			MPSGraphTensor* masked_indices = [graph selectWithPredicateTensor:mask truePredicateTensor:scanned_indices falsePredicateTensor:minus_one name:nil];
			MPSGraphTensor* zero = [graph constantWithScalar:0.0f shape:@[@1] dataType:MPSDataTypeInt32];
			MPSGraphTensor* masked_indices_with_head = [graph concatTensors:@[ zero, masked_indices ] dimension:0 name:nil];
			MPSGraphTensor* scanned_indices_with_head = [graph concatTensors:@[ zero, scanned_indices ] dimension:0 name:nil];
			MPSGraphTensor* initial_b = [graph constantWithScalar:-1.0f shape:@[@(bincount)] dataType:MPSDataTypeInt32];
			MPSGraphTensor* mps_b = [graph scatterWithDataTensor:initial_b updatesTensor:mps_a indicesTensor:masked_indices_with_head axis:0 mode:MPSGraphScatterModeSet name:nil];
			[resultTensors addObject:mps_b];
			MPSGraphTensor* unit = [graph constantWithScalar:1.0f shape:@[@(count)] dataType:MPSDataTypeInt32];
			MPSGraphTensor* mps_counts = [graph scatterWithUpdatesTensor:unit indicesTensor:scanned_indices_with_head shape:@[@(bincount)] axis:0 mode:MPSGraphScatterModeAdd name:nil];
			[resultTensors addObject:mps_counts];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_a], (ccv_nnc_tensor_view_t*[]){ b, indices }, (int*[]){ b->info.dim, indices->info.dim }, (int*[]){ b->stride, indices->stride }, 2, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_UNIQUE_CONSECUTIVE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_unique_consecutive_forw;
}
