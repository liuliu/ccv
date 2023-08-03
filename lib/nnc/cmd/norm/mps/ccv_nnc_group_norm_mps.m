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

static int _ccv_nnc_group_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 9);
	assert(output_size >= 1);
	
	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[4];
	ccv_nnc_tensor_view_t* const saved_mean = (ccv_nnc_tensor_view_t*)inputs[7];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)inputs[8];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dscale = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	ccv_nnc_tensor_view_t* const dbias = output_size > 2 ? (ccv_nnc_tensor_view_t*)outputs[2] : 0;
	assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));

	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(saved_mean, rdim);
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= adim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];

	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int indices[4];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, g->info.dim, g->stride, &mps_input_g);
			[inputTensors addObject:mps_input_g];
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, g->info.dim, g->stride);
			[inputShapedTypes addObject:mps_g_shape];

			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, a->info.dim, a->stride, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, a->info.dim, a->stride);
			[inputShapedTypes addObject:mps_a_shape];

			MPSGraphTensor* mps_input_scale;
			MPSGraphTensor* mps_scale = ccv_nnc_mps_graph_tensor_input(graph, scale, scale->info.dim, scale->stride, &mps_input_scale);
			[inputTensors addObject:mps_input_scale];
			MPSGraphShapedType* mps_scale_shape = ccv_nnc_mps_graph_tensor_input_shape(scale, scale->info.dim, scale->stride);
			[inputShapedTypes addObject:mps_scale_shape];

			MPSGraphTensor* mps_input_saved_mean;
			MPSGraphTensor* mps_saved_mean = ccv_nnc_mps_graph_tensor_input(graph, saved_mean, saved_mean->info.dim, saved_mean->stride, &mps_input_saved_mean);
			[inputTensors addObject:mps_input_saved_mean];
			MPSGraphShapedType* mps_saved_mean_shape = ccv_nnc_mps_graph_tensor_input_shape(saved_mean, saved_mean->info.dim, saved_mean->stride);
			[inputShapedTypes addObject:mps_saved_mean_shape];

			MPSGraphTensor* mps_input_saved_inv_std;
			MPSGraphTensor* mps_saved_inv_std = ccv_nnc_mps_graph_tensor_input(graph, saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride, &mps_input_saved_inv_std);
			[inputTensors addObject:mps_saved_inv_std];
			MPSGraphShapedType* mps_saved_inv_std_shape = ccv_nnc_mps_graph_tensor_input_shape(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);
			[inputShapedTypes addObject:mps_saved_inv_std_shape];

			MPSGraphTensor* mps_dscale = nil;
			MPSGraphTensor* mps_dbias = nil;
			MPSGraphTensor* mps_h = nil;
			
			NSMutableArray<NSNumber*>* group_broadcastable_shape = [NSMutableArray new];  // [N,G,1,H,W]
			NSMutableArray<NSNumber*>* group_reducible_shape = [NSMutableArray new];  // [N,G,C/G,H,W]
			int c_divide_g_axis = 0;
			for (int i = 0; i < a_nd; i++) {
				[group_reducible_shape addObject:@(saved_mean->info.dim[i])];
				[group_broadcastable_shape addObject:@(saved_mean->info.dim[i])];
				if (a->info.dim[i] != saved_mean->info.dim[i]) {
					c_divide_g_axis = i+1; // axis of C/G in [N,G,C/G,H,W]
					[group_broadcastable_shape addObject:@(1)];
					[group_reducible_shape addObject:@(a->info.dim[i]/saved_mean->info.dim[i])];
				}
			}

			// [N,G,H,W] --> [N,G,1,H,W] --> [N,G,C/G,H,W] --> [N,C,H,W]
			mps_saved_mean = [graph reshapeTensor:mps_saved_mean withShape:group_broadcastable_shape name:nil];
			mps_saved_mean = [graph broadcastTensor:mps_saved_mean toShape:group_reducible_shape name:nil];
			mps_saved_mean = [graph reshapeTensor:mps_saved_mean withShape:mps_a.shape name:nil];

			// [N,G,H,W] --> [N,G,1,H,W] --> [N,G,C/G,H,W] --> [N,C,H,W]
			mps_saved_inv_std = [graph reshapeTensor:mps_saved_inv_std withShape:group_broadcastable_shape name:nil];
			mps_saved_inv_std = [graph broadcastTensor:mps_saved_inv_std toShape:group_reducible_shape name:nil];
			mps_saved_inv_std = [graph reshapeTensor:mps_saved_inv_std withShape:mps_a.shape name:nil];


			// ap1[x] - meanp2[0]
			MPSGraphTensor* x_minus_mean = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_saved_mean name:nil];

			// ahp[x] = (ap1[x] - meanp2[0]) * inv_stdp2[0];
			MPSGraphTensor* ah = [graph multiplicationWithPrimaryTensor:x_minus_mean secondaryTensor:mps_saved_inv_std name:nil];
			
			if (dscale) {
				NSMutableArray<NSNumber*>* dscale_axes = [NSMutableArray new];	
				for (int i = 0; i < a_nd; i++) {
					if (a->info.dim[i] != dscale->info.dim[i])
						[dscale_axes addObject:@(i)];
				}
				//  dscalep2[x] = ahp[x] * gp1[x]; no reduce
				MPSGraphTensor* mps_dscale_original = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:mps_g name:nil];

				//  dscalep2[x] += ahp[x] * gp1[x]; reduce
				mps_dscale = [graph reductionSumWithTensor:mps_dscale_original axes:dscale_axes name:nil];
			}

			if (dbias) {
				NSMutableArray<NSNumber*>* dbias_axes = [NSMutableArray new];	
				for (int i = 0; i < a_nd; i++) {
					if (a->info.dim[i] != dbias->info.dim[i])
						[dbias_axes addObject:@(i)];
				}
				mps_dbias = [graph reductionSumWithTensor:mps_g axes:dbias_axes name:nil];
			}

			if (h) {							
				// gp1[x] * scalep2[x]
				mps_g = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_scale name:nil];

				// inv_n 
				MPSGraphTensor* sizeReciprocalTensor = [graph reciprocalWithTensor:[graph constantWithScalar:n dataType:mps_a.dataType] name:nil];          

				// gssp = gp1[x] * scalep2[x] * inv_stdp2[x]
				MPSGraphTensor* gss = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_saved_inv_std name:nil];

				// gssr = gss reduce by group
				//  [N,C,H,W] -->  [N,G,C/G,H,W] 
				MPSGraphTensor* gssr = [graph reshapeTensor:gss withShape:group_reducible_shape name:nil];

				// [N,G,C/G,H,W] --> [N,G,1,H,W] 
				gssr = [graph reductionSumWithTensor:gssr axes:@[@(c_divide_g_axis)] name:nil];

				// [N,G,1,H,W] --> [N,G,C/G,H,W] broadcast
				gssr = [graph broadcastTensor:gssr toShape:group_reducible_shape name:nil];;

				// [N,G,C/G,H,W] --> [N,C,H,W]
				gssr = [graph reshapeTensor:gssr withShape:mps_a.shape name:nil];;

				// ah[x] * gss[x]
				MPSGraphTensor* ahgss = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:gss name:nil];
						
				// ah[x] * gssp[x]; ahgssr reduce by group
				// [N,C,H,W] -->  [N,G,C/G,H,W] 
				MPSGraphTensor* ahgssr = [graph reshapeTensor:ahgss withShape:group_reducible_shape name:nil];

				// [N,G,C/G,H,W] --> [N,G,1,H,W] 
				ahgssr = [graph reductionSumWithTensor:ahgssr axes:@[@(c_divide_g_axis)] name:nil];

				// [N,G,1,H,W] --> [N,G,C/G,H,W] broadcast
				ahgssr = [graph broadcastTensor:ahgssr toShape:group_reducible_shape name:nil];;

				// [N,G,C/G,H,W] --> [N,C,H,W]
				ahgssr = [graph reshapeTensor:ahgssr withShape:mps_a.shape name:nil];;

				// ahp[x] * ahgssrp2[x]
				MPSGraphTensor* gssrp_ahp_ahgssrp = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:ahgssr name:nil];

				// gssrp2[x] + ahp[x] * ahgssrp2[x]
				gssrp_ahp_ahgssrp = [graph additionWithPrimaryTensor:gssrp_ahp_ahgssrp secondaryTensor:gssr name:nil];

				// inv_n * (gssrp2[x] + ahp[x] * ahgssrp2[x])
				gssrp_ahp_ahgssrp = [graph multiplicationWithPrimaryTensor:gssrp_ahp_ahgssrp secondaryTensor:sizeReciprocalTensor name:nil]; 

				// h = gssp[x] - inv_n * (gssrp2[x] + ahp[x] * ahgssrp2[x])
				mps_h = [graph subtractionWithPrimaryTensor:gss secondaryTensor:gssrp_ahp_ahgssrp name:nil];
			}

			if (mps_h) {
				[resultTensors addObject:mps_h];
			} 

			if (mps_dscale) {
				[resultTensors addObject:mps_dscale];
			}

			if (mps_dbias) {
				[resultTensors addObject:mps_dbias];
			}
			[graph dump];
		});
		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
		MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(scale, scale->info.dim, scale->stride);
		MPSGraphTensorData* data_saved_mean = ccv_nnc_mps_graph_tensor_data(saved_mean, saved_mean->info.dim, saved_mean->stride);
		MPSGraphTensorData* data_saved_inv_std = ccv_nnc_mps_graph_tensor_data(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);

		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_g, data_a, data_scale, data_saved_mean, data_saved_inv_std], (ccv_nnc_tensor_view_t* []){ h, dscale, dbias }, (int*[]){ h->info.dim, dscale->info.dim, dbias->info.dim }, (int*[]){ h->stride, dscale->stride, dbias->stride }, 3);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GROUP_NORM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_group_norm_back;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GROUP_NORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_group_norm_forw;
}
