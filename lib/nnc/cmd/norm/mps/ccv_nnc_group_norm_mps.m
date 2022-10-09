#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static void _ccv_break_axis_to_groups(ccv_nnc_tensor_view_t* const tv, const int axis, const int dim1, const int dim2)
{
	if (CCV_IS_TENSOR_VIEW(tv))
	{
		if (CCV_NNC_MAX_DIM_ALLOC - axis - 2 > 0)
		{
			// Need to handle ofs and stride.
			memmove(tv->info.dim + axis + 2, tv->info.dim + axis + 1, sizeof(int) * (CCV_NNC_MAX_DIM_ALLOC - axis - 2));
			memmove(tv->stride + axis + 2, tv->stride + axis + 1, sizeof(int) * (CCV_NNC_MAX_DIM_ALLOC - axis - 2));
		}
		tv->info.dim[axis] = dim1;
		tv->info.dim[axis + 1] = dim2;
		tv->stride[axis + 1] = tv->stride[axis];
		tv->stride[axis] = tv->stride[axis] * dim2; // This all worked out because we can still skip that much.
	} else {
		// Non tensor view, straightforward.
		if (CCV_NNC_MAX_DIM_ALLOC - axis - 2 > 0)
			memmove(tv->info.dim + axis + 2, tv->info.dim + axis + 1, sizeof(int) * (CCV_NNC_MAX_DIM_ALLOC - axis - 2));
		tv->info.dim[axis] = dim1;
		tv->info.dim[axis + 1] = dim2;
	}
}

static int _ccv_nnc_group_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 3);
	ccv_nnc_tensor_view_t at = ccv_nnc_get_tensor_view(inputs[0]);
	_ccv_break_axis_to_groups(&at, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, at.info.dim[cmd.info.gnorm.group_axis] / cmd.info.gnorm.groups);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
	ccv_nnc_tensor_view_t scalet = ccv_nnc_get_tensor_view(inputs[1]);
	_ccv_break_axis_to_groups(&scalet, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, scalet.info.dim[cmd.info.gnorm.group_axis] / cmd.info.gnorm.groups);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[2]));
	ccv_nnc_tensor_view_t biast = ccv_nnc_get_tensor_view(inputs[2]);
	_ccv_break_axis_to_groups(&biast, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, biast.info.dim[cmd.info.gnorm.group_axis] / cmd.info.gnorm.groups);
	ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[0]);
	_ccv_break_axis_to_groups(&bt, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, bt.info.dim[cmd.info.gnorm.group_axis] / cmd.info.gnorm.groups);
	ccv_nnc_tensor_view_t saved_meant = ccv_nnc_get_tensor_view(outputs[1]);
	_ccv_break_axis_to_groups(&saved_meant, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, 1);
	ccv_nnc_tensor_view_t saved_inv_stdt = ccv_nnc_get_tensor_view(outputs[2]);
	_ccv_break_axis_to_groups(&saved_inv_stdt, cmd.info.gnorm.group_axis, cmd.info.gnorm.groups, 1);
	ccv_nnc_tensor_view_alignment((ccv_nnc_tensor_view_t*[]){
		&at,
		&saved_meant,
		&saved_inv_stdt,
		&bt
	}, 4);
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, &at, at.info.dim, at.stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(&at, at.info.dim, at.stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_input_scale;
			MPSGraphTensor* mps_scale = ccv_nnc_mps_graph_tensor_input(graph, &scalet, scalet.info.dim, scalet.stride, &mps_input_scale);
			[inputTensors addObject:mps_input_scale];
			MPSGraphShapedType* mps_scale_shape = ccv_nnc_mps_graph_tensor_input_shape(&scalet, scalet.info.dim, scalet.stride);
			[inputShapedTypes addObject:mps_scale_shape];
			MPSGraphTensor* mps_input_bias;
			MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, &biast, biast.info.dim, biast.stride, &mps_input_bias);
			[inputTensors addObject:mps_input_bias];
			MPSGraphShapedType* mps_bias_shape = ccv_nnc_mps_graph_tensor_input_shape(&biast, biast.info.dim, biast.stride);
			[inputShapedTypes addObject:mps_bias_shape];
			int i;
			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
			const int rnd = ccv_nnc_tensor_nd(saved_meant.info.dim);
			for (i = 0; i < rnd; i++)
				if (at.info.dim[i] != saved_meant.info.dim[i])
					[axes addObject:@(i)];
			MPSGraphTensor* mps_saved_mean = [graph meanOfTensor:mps_a axes:axes name:nil];
			MPSGraphTensor* mps_saved_inv_std = [graph varianceOfTensor:mps_a meanTensor:mps_saved_mean axes:axes name:nil];
			[axes release];
			const float epsilon = cmd.info.gnorm.epsilon;
			MPSGraphTensor* mps_b = [graph normalizationWithTensor:mps_a meanTensor:mps_saved_mean varianceTensor:mps_saved_inv_std gammaTensor:mps_scale betaTensor:mps_bias epsilon:epsilon name:nil];
			[resultTensors addObject:mps_b];
		});
		// I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&at, at.info.dim, at.stride);
		MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(&scalet, scalet.info.dim, scalet.stride);
		MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(&biast, biast.info.dim, biast.stride);
		MPSGraphTensorData* data[] = {data_a, data_scale, data_bias};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &bt, bt.info.dim, bt.stride);
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_group_norm_forw;
}
