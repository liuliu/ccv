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
	const int group_axis = cmd.info.gnorm.group_axis;
	_ccv_break_axis_to_groups(&at, group_axis, cmd.info.gnorm.groups, at.info.dim[group_axis] / cmd.info.gnorm.groups);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[1]));
	ccv_nnc_tensor_view_t scalet = ccv_nnc_get_tensor_view(inputs[1]);
	_ccv_break_axis_to_groups(&scalet, group_axis, cmd.info.gnorm.groups, scalet.info.dim[group_axis] / cmd.info.gnorm.groups);
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[2]));
	ccv_nnc_tensor_view_t biast = ccv_nnc_get_tensor_view(inputs[2]);
	_ccv_break_axis_to_groups(&biast, group_axis, cmd.info.gnorm.groups, biast.info.dim[group_axis] / cmd.info.gnorm.groups);
	ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[0]);
	_ccv_break_axis_to_groups(&bt, group_axis, cmd.info.gnorm.groups, bt.info.dim[group_axis] / cmd.info.gnorm.groups);
	ccv_nnc_tensor_view_t saved_meant = ccv_nnc_get_tensor_view(outputs[1]);
	_ccv_break_axis_to_groups(&saved_meant, group_axis, cmd.info.gnorm.groups, 1);
	ccv_nnc_tensor_view_t saved_inv_stdt = ccv_nnc_get_tensor_view(outputs[2]);
	_ccv_break_axis_to_groups(&saved_inv_stdt, group_axis, cmd.info.gnorm.groups, 1);
	ccv_nnc_tensor_view_alignment((ccv_nnc_tensor_view_t*[]){
		&at,
		&saved_meant,
		&saved_inv_stdt,
		&bt
	}, 4);
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[3];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a;
			MPSGraphShapedType* mps_a_shape;
			int i;
			// If we cannot break input as expected, it only means: 1. the input is a tensor view. 2. We need to get mps_a first and then reshape into the break dimensions.
			if (group_axis > 0 && CCV_IS_TENSOR_VIEW(inputs[0]) && (at.stride[group_axis - 1] % at.stride[group_axis]) != 0)
			{
				mps_a = ccv_nnc_mps_graph_tensor_input(graph, (ccv_nnc_tensor_view_t*)inputs[0], inputs[0]->info.dim, ((ccv_nnc_tensor_view_t*)inputs[0])->stride, &mps_input_a);
				NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
				int nd = ccv_nnc_tensor_nd(at.info.dim);
				for (i = 0; i < nd; i++)
					[shape addObject:@(at.info.dim[i])];
				mps_a = [graph reshapeTensor:mps_a withShape:shape name:nil];
				mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape((ccv_nnc_tensor_view_t*)inputs[0], inputs[0]->info.dim, ((ccv_nnc_tensor_view_t*)inputs[0])->stride);
			} else {
				mps_a = ccv_nnc_mps_graph_tensor_input(graph, &at, at.info.dim, at.stride, &mps_input_a);
				mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(&at, at.info.dim, at.stride);
			}
			[inputTensors addObject:mps_input_a];
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
			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
			const int rnd = ccv_nnc_tensor_nd(saved_meant.info.dim);
			for (i = 0; i < rnd; i++)
				if (at.info.dim[i] != saved_meant.info.dim[i])
					[axes addObject:@(i)];
			MPSGraphTensor* mps_saved_mean = [graph meanOfTensor:mps_a axes:axes name:nil];
			MPSGraphTensor* mps_a_subtract_mean = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_saved_mean name:nil];
			MPSGraphTensor* mps_saved_inv_std;
			const double epsilon = cmd.info.gnorm.epsilon;
			if (at.info.datatype == CCV_32F)
			{
				MPSGraphTensor* mps_square = [graph squareWithTensor:mps_a_subtract_mean name:nil];
				MPSGraphTensor* mps_variance = [graph meanOfTensor:mps_square axes:axes name:nil];
				[axes release];
				MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
				mps_saved_inv_std = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_variance secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
			} else {
				// Compute variance at higher resolution.
				MPSGraphTensor* mps_a_subtract_mean_f32 = [graph castTensor:mps_a_subtract_mean toType:MPSDataTypeFloat32 name:@"float"];
				MPSGraphTensor* mps_square_f32 = [graph squareWithTensor:mps_a_subtract_mean_f32 name:nil];
				MPSGraphTensor* mps_variance_f32 = [graph meanOfTensor:mps_square_f32 axes:axes name:nil];
				[axes release];
				MPSGraphTensor* mps_epsilon_f32 = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
				MPSGraphTensor* mps_inv_std_f32 = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_variance_f32 secondaryTensor:mps_epsilon_f32 name:nil] name:nil] name:nil];
				mps_saved_inv_std = [graph castTensor:mps_inv_std_f32 toType:MPSDataTypeFloat16 name:@"inv_std"];
			}
			MPSGraphTensor* mps_b = [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a_subtract_mean secondaryTensor:mps_saved_inv_std name:nil] secondaryTensor:mps_scale name:nil] secondaryTensor:mps_bias name:nil];
			if (group_axis > 0 && CCV_IS_TENSOR_VIEW(outputs[0]) && (bt.stride[group_axis - 1] % bt.stride[group_axis]) != 0)
			{
				NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
				const int nd = ccv_nnc_tensor_nd(outputs[0]->info.dim);
				for (i = 0; i < nd; i++)
					[shape addObject:@(outputs[0]->info.dim[i])];
				mps_b = [graph reshapeTensor:mps_b withShape:shape name:nil];
			}
			[resultTensors addObject:mps_b];
			[resultTensors addObject:mps_saved_mean];
			[resultTensors addObject:mps_saved_inv_std];
		});
		// I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
		MPSGraphTensorData* data_a;
		if (group_axis > 0 && CCV_IS_TENSOR_VIEW(inputs[0]) && (at.stride[group_axis - 1] % at.stride[group_axis]) != 0)
			data_a = ccv_nnc_mps_graph_tensor_data((ccv_nnc_tensor_view_t*)inputs[0], inputs[0]->info.dim, ((ccv_nnc_tensor_view_t*)inputs[0])->stride);
		else
			data_a = ccv_nnc_mps_graph_tensor_data(&at, at.info.dim, at.stride);
		MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(&scalet, scalet.info.dim, scalet.stride);
		MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(&biast, biast.info.dim, biast.stride);
		MPSGraphTensorData* data[] = {data_a, data_scale, data_bias};
		if (group_axis > 0 && CCV_IS_TENSOR_VIEW(outputs[0]) && (bt.stride[group_axis - 1] % bt.stride[group_axis]) != 0)
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){ (ccv_nnc_tensor_view_t*)outputs[0], &saved_meant, &saved_inv_stdt }, (int*[]){ outputs[0]->info.dim, saved_meant.info.dim, saved_inv_stdt.info.dim }, (int*[]){ ((ccv_nnc_tensor_view_t*)outputs[0])->stride, saved_meant.stride, saved_inv_stdt.stride }, 3);
		else
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){ &bt, &saved_meant, &saved_inv_stdt }, (int*[]){ bt.info.dim, saved_meant.info.dim, saved_inv_stdt.info.dim }, (int*[]){ bt.stride, saved_meant.stride, saved_inv_stdt.stride }, 3);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
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
