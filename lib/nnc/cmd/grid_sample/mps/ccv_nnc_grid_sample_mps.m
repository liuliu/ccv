#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_grid_sample_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_view_t* const a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* const grid = (const ccv_nnc_tensor_view_t*)inputs[1];
	ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(a->info.datatype == CCV_32F);
	assert(grid->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(a->info.format == b->info.format);
	assert(a->info.format == CCV_TENSOR_FORMAT_NCHW || a->info.format == CCV_TENSOR_FORMAT_NHWC);
	const int and = ccv_nnc_tensor_nd(a->info.dim);
	const int bnd = ccv_nnc_tensor_nd(b->info.dim);
	const int gnd = ccv_nnc_tensor_nd(grid->info.dim);
	assert(and == 3 || and == 4);
	assert(bnd == 3 || bnd == 4);
	assert(gnd == 3 || gnd == 4);
	const int an_axis = (and == 4) ? 0 : -1;
	const int bn_axis = (bnd == 4) ? 0 : -1;
	const int gn_axis = (gnd == 4) ? 0 : -1;
	int ac_axis = 0;
	int bc_axis = 0, bh_axis = 0, bw_axis = 0;
	if (a->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		ac_axis = (and == 4) ? 1 : 0;
		bc_axis = (bnd == 4) ? 1 : 0;
		bh_axis = (bnd == 4) ? 2 : 1;
		bw_axis = (bnd == 4) ? 3 : 2;
	} else {
		ac_axis = (and == 4) ? 3 : 2;
		bh_axis = (bnd == 4) ? 1 : 0;
		bw_axis = (bnd == 4) ? 2 : 1;
		bc_axis = (bnd == 4) ? 3 : 2;
	}
	const int gh_axis = (gnd == 4) ? 1 : 0;
	const int gw_axis = (gnd == 4) ? 2 : 1;
	const int gc_axis = (gnd == 4) ? 3 : 2;
	const int N = (an_axis >= 0) ? a->info.dim[an_axis] : 1;
	const int C = a->info.dim[ac_axis];
	const int N_out = (bn_axis >= 0) ? b->info.dim[bn_axis] : 1;
	const int C_out = b->info.dim[bc_axis];
	const int H_out = b->info.dim[bh_axis];
	const int W_out = b->info.dim[bw_axis];
	assert(N_out == N);
	assert(C_out == C);
	const int N_grid = (gn_axis >= 0) ? grid->info.dim[gn_axis] : 1;
	assert(grid->info.dim[gc_axis] == 2);
	assert(N_grid == N);
	assert(grid->info.dim[gh_axis] == H_out);
	assert(grid->info.dim[gw_axis] == W_out);
	const int source_add_batch = (and == 3);
	const int grid_add_batch = (gnd == 3);
	const int output_remove_batch = (bnd == 3);
	const MPSGraphTensorNamedDataLayout layout = (a->info.format == CCV_TENSOR_FORMAT_NCHW) ? MPSGraphTensorNamedDataLayoutNCHW : MPSGraphTensorNamedDataLayoutNHWC;
	const int align_corners = cmd.info.grid_sample.align_corners;
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
		int indices[2];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_input_grid;
			MPSGraphTensor* mps_grid = ccv_nnc_mps_graph_tensor_input(graph, grid, grid->info.dim, grid->stride, &mps_input_grid);
			[inputTensors addObject:mps_input_grid];
			MPSGraphShapedType* mps_grid_shape = ccv_nnc_mps_graph_tensor_input_shape(grid, grid->info.dim, grid->stride);
			[inputShapedTypes addObject:mps_grid_shape];
			if (source_add_batch)
				mps_a = [graph expandDimsOfTensor:mps_a axis:0 name:nil];
			if (grid_add_batch)
				mps_grid = [graph expandDimsOfTensor:mps_grid axis:0 name:nil];
			MPSGraphTensor* mps_b = [graph sampleGridWithSourceTensor:mps_a
			                                         coordinateTensor:mps_grid
			                                                   layout:layout
			                                     normalizeCoordinates:YES
			                                      relativeCoordinates:NO
			                                             alignCorners:(align_corners ? YES : NO)
			                                              paddingMode:MPSGraphPaddingModeZero
			                                             samplingMode:MPSGraphResizeBilinear
			                                            constantValue:0
			                                                     name:nil];
			if (output_remove_batch)
				mps_b = [graph squeezeTensor:mps_b axis:0 name:nil];
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_grid = ccv_nnc_mps_graph_tensor_data(grid, grid->info.dim, grid->stride);
		MPSGraphTensorData* data[] = {data_a, data_grid};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ b->info.dim }, (int*[]){ b->stride }, 1, 0);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_grid_sample_forw;
}
