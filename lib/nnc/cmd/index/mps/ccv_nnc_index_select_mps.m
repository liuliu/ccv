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

static int _ccv_nnc_index_select_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* const indices = (const ccv_nnc_tensor_view_t*)inputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == 1);
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) <= 2);
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int idx[2];
		int indices_dim[CCV_NNC_MAX_DIM_ALLOC] = {0};
		int indices_stride[CCV_NNC_MAX_DIM_ALLOC] = {0};
		const int nd = ccv_nnc_tensor_nd(b->info.dim);
		if (nd == 2)
		{
			indices_dim[0] = indices->info.dim[0];
			indices_dim[1] = 1;
			indices_stride[0] = CCV_IS_TENSOR_VIEW(indices) ? indices->stride[0] : 1;
			indices_stride[1] = indices_stride[0];
		} else {
			indices_dim[0] = indices->info.dim[0];
			indices_stride[0] = CCV_IS_TENSOR_VIEW(indices) ? indices->stride[0] : 1;
		}
		int* indices_dim_r = indices_dim;
		int* indices_stride_r = indices_stride;
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, idx, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			if (indices->info.datatype == CCV_32S)
			{
				MPSGraphTensor* mps_input_indices;
				MPSGraphTensor* mps_indices = ccv_nnc_mps_graph_tensor_input(graph, indices, indices_dim_r, indices_stride_r, &mps_input_indices);
				[inputTensors addObject:mps_input_indices];
				MPSGraphShapedType* mps_indices_shape = ccv_nnc_mps_graph_tensor_input_shape(indices, indices_dim_r, indices_stride_r);
				[inputShapedTypes addObject:mps_indices_shape];
				if (nd == 2) // Only need to broadcast when we have 2-d vector.
				{
					int i;
					NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
					for (i = 0; i < nd; i++)
						[shape addObject:@(b->info.dim[i])];
					mps_indices = [graph broadcastTensor:mps_indices toShape:shape name:nil];
					[shape release];
				}
				MPSGraphTensor* mps_b = [graph gatherAlongAxis:0 withUpdatesTensor:mps_a indicesTensor:mps_indices name:nil];
				[resultTensors addObject:mps_b];
			} else {
				assert(indices->info.datatype == CCV_32F);
				MPSGraphTensor* mps_input_indices;
				MPSGraphTensor* mps_indices = ccv_nnc_mps_graph_tensor_input(graph, indices, indices_dim_r, indices_stride_r, &mps_input_indices);
				[inputTensors addObject:mps_input_indices];
				MPSGraphShapedType* mps_indices_shape = ccv_nnc_mps_graph_tensor_input_shape(indices, indices_dim_r, indices_stride_r);
				[inputShapedTypes addObject:mps_indices_shape];
				MPSGraphTensor* mps_indices_0 = [graph castTensor:mps_indices toType:MPSDataTypeInt32 name:nil];
				MPSGraphTensor* mps_1 = [graph constantWithScalar:1 dataType:MPSDataTypeInt32];
				MPSGraphTensor* mps_a_rows_1 = [graph constantWithScalar:(a->info.dim[0] - 1) dataType:MPSDataTypeInt32];
				MPSGraphTensor* mps_indices_1 = [graph minimumWithPrimaryTensor:[graph additionWithPrimaryTensor:mps_indices_0 secondaryTensor:mps_1 name:nil] secondaryTensor:mps_a_rows_1 name:nil];
				MPSGraphTensor* mps_indices_f0 = [graph castTensor:mps_indices_0 toType:MPSDataTypeFloat32 name:nil];
				MPSGraphTensor* mps_w1 = [graph subtractionWithPrimaryTensor:mps_indices secondaryTensor:mps_indices_f0 name:nil];
				MPSGraphTensor* mps_f1 = [graph constantWithScalar:1 dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_w0 = [graph subtractionWithPrimaryTensor:mps_f1 secondaryTensor:mps_w1 name:nil];
				mps_w1 = [graph castTensor:mps_w1 toType:ccv_nnc_mps_datatype(a->info.datatype) name:nil];
				mps_w0 = [graph castTensor:mps_w0 toType:ccv_nnc_mps_datatype(a->info.datatype) name:nil];
				if (nd == 2) // Only need to broadcast when we have 2-d vector.
				{
					int i;
					NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
					for (i = 0; i < nd; i++)
						[shape addObject:@(b->info.dim[i])];
					mps_indices_0 = [graph broadcastTensor:mps_indices_0 toShape:shape name:nil];
					mps_indices_1 = [graph broadcastTensor:mps_indices_1 toShape:shape name:nil];
					[shape release];
				}
				MPSGraphTensor* mps_b_0 = [graph gatherAlongAxis:0 withUpdatesTensor:mps_a indicesTensor:mps_indices_0 name:nil];
				MPSGraphTensor* mps_b_1 = [graph gatherAlongAxis:0 withUpdatesTensor:mps_a indicesTensor:mps_indices_1 name:nil];
				MPSGraphTensor* mps_b = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_b_0 secondaryTensor:mps_w0 name:nil] secondaryTensor:[graph multiplicationWithPrimaryTensor:mps_b_1 secondaryTensor:mps_w1 name:nil] name:nil];
				[resultTensors addObject:mps_b];
			}
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_indices = ccv_nnc_mps_graph_tensor_data(indices, indices_dim, indices_stride);
		MPSGraphTensorData* data[] = {data_a, data_indices};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[idx[0]], data[idx[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_index_select_forw;
}
