#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_rmsnorm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 2);
	ccv_nnc_tensor_view_t at = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t scalet = ccv_nnc_get_tensor_view(inputs[1]);
	ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t saved_inv_stdt = ccv_nnc_get_tensor_view(outputs[1]);
	ccv_nnc_tensor_view_alignment((ccv_nnc_tensor_view_t*[]){
		&at,
		&saved_inv_stdt,
		&bt
	}, 3);
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
				(inputs[0]->info.datatype == outputs[0]->info.datatype) &&
				(inputs[0]->info.datatype == outputs[1]->info.datatype) &&
				(inputs[0]->info.datatype == inputs[1]->info.datatype);
			if (!is_same_dtype) {
				use_mfa = false;
				fallback_reason = "Mixed precision.";
			}

			switch (at.info.datatype) {
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
			if (!CCV_IS_TENSOR_CONTIGUOUS(inputs[0]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(outputs[0]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(outputs[1]) ||
					!CCV_IS_TENSOR_CONTIGUOUS(inputs[1]))
			{
				use_mfa = false;
				fallback_reason = "Strided.";
			}
		}

		int channel_count;
		const int channel_groups = 1;
		int sequence_count = 1;
		int data_batch_dim = 0;
		int scale_translation_batch_dim = 0;
		uint8_t data_batched = 0;
		uint8_t scale_translation_batched = 0;

		if (use_mfa) {
			const int rnd = ccv_nnc_tensor_nd(saved_inv_stdt.info.dim);
			channel_count = 1;
			int i;
			for (i = rnd - 1; i >= 0; i--)
				if (at.info.dim[i] != saved_inv_stdt.info.dim[i])
					channel_count *= at.info.dim[i];
				else
					break;
			for (i = 0; i < rnd; i++)
				if (at.info.dim[i] == saved_inv_stdt.info.dim[i])
					sequence_count *= at.info.dim[i];
			if (ccv_nnc_tensor_count(at.info) != sequence_count * channel_count)
				use_mfa = false;
			else {
				if (sequence_count > at.info.dim[0] && at.info.dim[0] > 1)
				{
					data_batched = true;
					data_batch_dim = at.info.dim[0];
					sequence_count = sequence_count / data_batch_dim;
					if (scalet.info.dim[0] == at.info.dim[0])
					{
						scale_translation_batched = true;
						scale_translation_batch_dim = at.info.dim[0];
					}
				}
			}
		}

		if (METAL_LOG_LEVEL(context) >= 3) {
			if (use_mfa) {
				ccv_nnc_mfa_log_message("Compatible normalization found.");
			} else {
				ccv_nnc_mfa_log_message("Incompatible normalization found. Incompatible because:");
				ccv_nnc_mfa_log_message(fallback_reason);
			}
		}

		if (use_mfa) {
			ccv_nnc_mfa_normalization_params_t params = {
				.data_type = mtl_data_type,
				.channel_count = (uint32_t)channel_count,
				.channel_groups = (uint32_t)channel_groups,
				.sequence_count = (uint32_t)sequence_count,
				.epsilon = cmd.info.rmsnorm.epsilon,
				.elementwise_affine = 1,
				.scale_translation_batched = scale_translation_batched,
				.normalization_type = 2,
				.reuse_saved_statistics = false,

				.batch_dims_data = { 0 },
				.batch_dims_scale_translation = { 0 },
			};

			// Create a null-terminated list of batch dimensions.
			if (data_batched) {
				params.batch_dims_data[0] = data_batch_dim;
				params.batch_dims_data[1] = 0;

				if (scale_translation_batched) {
					params.batch_dims_scale_translation[0] = scale_translation_batch_dim;
					params.batch_dims_scale_translation[1] = 0;
				}
			}
			ccv_nnc_mfa_prepare_normalization(context, params);

			mtl_command_batch_t* command_batch = ccv_nnc_stream_context_start_command_batch(stream_context);
			mtl_buffer_t* tensors[5] = {
				mpgetbuffer(inputs[0]), // source
				mpgetbuffer(outputs[0]), // destination
				mpgetbuffer(outputs[1]), // saved_standard_deviation_reciprocal
				mpgetbuffer(inputs[1]), // channel_scales
				NULL,
			};
			size_t tensor_offsets[4] = {
				at.dataof, // source offset
				bt.dataof, // destination offset
				saved_inv_stdt.dataof, // saved_standard_deviation_reciprocal offset
				scalet.dataof, // channel_scales offset
			};
			ccv_nnc_mfa_encode_normalization(context, params, command_batch, tensors, tensor_offsets);
			ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
		} else {
			MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[2];
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
				int i;
				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				const int rnd = ccv_nnc_tensor_nd(saved_inv_stdt.info.dim);
				for (i = 0; i < rnd; i++)
					if (at.info.dim[i] != saved_inv_stdt.info.dim[i])
						[axes addObject:@(i)];
				MPSGraphTensor* mps_saved_inv_std;
				const double epsilon = cmd.info.rmsnorm.epsilon;
				MPSGraphTensor* mps_b;
				if (at.info.datatype == CCV_32F)
				{
					MPSGraphTensor* mps_square = [graph squareWithTensor:mps_a name:nil];
					MPSGraphTensor* mps_variance = [graph meanOfTensor:mps_square axes:axes name:nil];
					[axes release];
					MPSGraphTensor* mps_epsilon = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
					mps_saved_inv_std = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_variance secondaryTensor:mps_epsilon name:nil] name:nil] name:nil];
					mps_b = [graph multiplicationWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_saved_inv_std name:nil] secondaryTensor:mps_scale name:nil];
				} else {
					// Compute variance at higher resolution.
					MPSGraphTensor* mps_a_f32 = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"float"];
					MPSGraphTensor* mps_square_f32 = [graph squareWithTensor:mps_a_f32 name:nil];
					MPSGraphTensor* mps_variance_f32 = [graph meanOfTensor:mps_square_f32 axes:axes name:nil];
					[axes release];
					MPSGraphTensor* mps_epsilon_f32 = [graph constantWithScalar:epsilon dataType:MPSDataTypeFloat32];
					MPSGraphTensor* mps_inv_std_f32 = [graph reciprocalWithTensor:[graph squareRootWithTensor:[graph additionWithPrimaryTensor:mps_variance_f32 secondaryTensor:mps_epsilon_f32 name:nil] name:nil] name:nil];
					mps_saved_inv_std = [graph castTensor:mps_inv_std_f32 toType:MPSDataTypeFloat16 name:@"inv_std"];
					mps_b = [graph multiplicationWithPrimaryTensor:[graph castTensor:[graph multiplicationWithPrimaryTensor:mps_a_f32 secondaryTensor:mps_inv_std_f32 name:nil] toType:MPSDataTypeFloat16 name:@"b"] secondaryTensor:mps_scale name:nil];
				}
				[resultTensors addObject:mps_b];
				[resultTensors addObject:mps_saved_inv_std];
			});
			// I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&at, at.info.dim, at.stride);
			MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(&scalet, scalet.info.dim, scalet.stride);
			MPSGraphTensorData* data[] = {data_a, data_scale};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], (ccv_nnc_tensor_view_t* []){ &bt, &saved_inv_stdt }, (int*[]){ bt.info.dim, saved_inv_stdt.info.dim }, (int*[]){ bt.stride, saved_inv_stdt.stride }, 2, 0);
			ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_rmsnorm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 6);
	assert(output_size >= 1);

	const ccv_nnc_tensor_view_t* g = (ccv_nnc_tensor_view_t*)inputs[0];
	ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const scale = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const saved_inv_std = (ccv_nnc_tensor_view_t*)inputs[5];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const dscale = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	assert(ccv_nnc_tensor_nd(g->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= CCV_NNC_MAX_DIM + 2);
	assert(ccv_nnc_tensor_nd(h->info.dim) <= CCV_NNC_MAX_DIM + 2);

	// Assuming this is float 32.
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	int rdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	ccv_nnc_tensor_view_get_dim(saved_inv_std, rdim);
	int x;
	int n = 1;
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n *= adim[x];
	for (x = 0; x < CCV_NNC_MAX_DIM + 2; x++)
		n /= rdim[x];

	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);

	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);

		if (h) {
			ccv_nnc_mps_graph_key_t h_key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, inputs, input_size, outputs, output_size);
			int indices[4];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(h_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
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

				MPSGraphTensor* mps_input_saved_inv_std;
				MPSGraphTensor* mps_saved_inv_std = ccv_nnc_mps_graph_tensor_input(graph, saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride, &mps_input_saved_inv_std);
				[inputTensors addObject:mps_saved_inv_std];
				MPSGraphShapedType* mps_saved_inv_std_shape = ccv_nnc_mps_graph_tensor_input_shape(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);
				[inputShapedTypes addObject:mps_saved_inv_std_shape];

				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				for (int k = 0; k < cmd.info.rmsnorm.count; k++) {
					[axes addObject:@(cmd.info.rmsnorm.axis[k])];
				}

				MPSGraphTensor* mps_h = nil;

				if (a->info.datatype == CCV_16F)
					mps_a = [graph castTensor:mps_a toType:MPSDataTypeFloat32 name:@"mps_a_float"];

				if (saved_inv_std->info.datatype == CCV_16F)
					mps_saved_inv_std = [graph castTensor:mps_saved_inv_std toType:MPSDataTypeFloat32 name:@"mps_saved_inv_std_float"];

				// ahp[x] = ap1[x] * inv_stdp2[0];
				MPSGraphTensor* ah = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_saved_inv_std name:nil];

				if (g->info.datatype == CCV_16F)
					mps_g = [graph castTensor:mps_g toType:MPSDataTypeFloat32 name:@"mps_g_float"];
				if (scale->info.datatype == CCV_16F)
					mps_scale = [graph castTensor:mps_scale toType:MPSDataTypeFloat32 name:@"mps_scale_float"];

				// gp1[x] * scalep2[x]
				mps_g = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_scale name:nil];

				// inv_n
				MPSGraphTensor* inv_n = [graph constantWithScalar:1.0 / (float)n dataType:mps_a.dataType];

				// gssp = gp1[x] * scalep2[x] * inv_stdp2[x]
				MPSGraphTensor* gss = [graph multiplicationWithPrimaryTensor:mps_g secondaryTensor:mps_saved_inv_std name:nil];

				// ah[x] * gss[x]
				MPSGraphTensor* ahgss = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:gss name:nil];

				// ah[x] * gssp[x]; reduce
				MPSGraphTensor* ahgssr = [graph reductionSumWithTensor:ahgss axes:axes name:nil];

				// ahp[x] * ahgssrp2[x]
				MPSGraphTensor* gssrp_ahp_ahgssrp = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:ahgssr name:nil];

				// inv_n * ahp[x] * ahgssrp2[x]
				gssrp_ahp_ahgssrp = [graph multiplicationWithPrimaryTensor:gssrp_ahp_ahgssrp secondaryTensor:inv_n name:nil];

				// h = gssp[x] - inv_n * ahp[x] * ahgssrp2[x]
				mps_h = [graph subtractionWithPrimaryTensor:gss secondaryTensor:gssrp_ahp_ahgssrp name:nil];

				if (h->info.datatype == CCV_16F)
					mps_h = [graph castTensor:mps_h toType:MPSDataTypeFloat16 name:@"mps_h_half"];

				[resultTensors addObject:mps_h];
			
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(scale, scale->info.dim, scale->stride);
			MPSGraphTensorData* data_saved_inv_std = ccv_nnc_mps_graph_tensor_data(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);
			MPSGraphTensorData* data[] = {data_g, data_a, data_scale, data_saved_inv_std};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]], data[indices[3]]], (ccv_nnc_tensor_view_t* []){ h }, (int*[]){ h->info.dim }, (int*[]){ h->stride }, 1, 0);
			
		}

		if (dscale) {
			ccv_nnc_mps_graph_key_t dscale_key = ccv_nnc_mps_graph_key_new(cmd, 1, hint, flags, inputs, input_size, outputs, output_size);
			int indices[3];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(dscale_key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
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

				MPSGraphTensor* mps_input_saved_inv_std;
				MPSGraphTensor* mps_saved_inv_std = ccv_nnc_mps_graph_tensor_input(graph, saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride, &mps_input_saved_inv_std);
				[inputTensors addObject:mps_saved_inv_std];
				MPSGraphShapedType* mps_saved_inv_std_shape = ccv_nnc_mps_graph_tensor_input_shape(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);
				[inputShapedTypes addObject:mps_saved_inv_std_shape];

				NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
				for (int k = 0; k < cmd.info.rmsnorm.count; k++) {
					[axes addObject:@(cmd.info.rmsnorm.axis[k])];
				}

				MPSGraphTensor* mps_dscale = nil;

				// ahp[x] = ap1[x] * inv_stdp2[0];
				MPSGraphTensor* ah = [graph multiplicationWithPrimaryTensor:mps_a secondaryTensor:mps_saved_inv_std name:nil];

				NSMutableArray<NSNumber*>* dscale_axes = [NSMutableArray new];
				for (int i = 0; i < a_nd; i++) {
					if (g->info.dim[i] != dscale->info.dim[i])
						[dscale_axes addObject:@(i)];
				}
				//	dscalep2[x] = ahp[x] * gp1[x]; no reduce
				MPSGraphTensor* mps_dscale_original = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:mps_g name:nil];

				//	dscalep2[x] += ahp[x] * gp1[x]; reduce
				mps_dscale = [graph reductionSumWithTensor:mps_dscale_original axes:dscale_axes name:nil];
				[dscale_axes release];

				[resultTensors addObject:mps_dscale];
			});
			MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, g->info.dim, g->stride);
			MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);
			MPSGraphTensorData* data_saved_inv_std = ccv_nnc_mps_graph_tensor_data(saved_inv_std, saved_inv_std->info.dim, saved_inv_std->stride);
			MPSGraphTensorData* data[] = {data_g, data_a, data_saved_inv_std};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){  dscale }, (int*[]){ dscale->info.dim }, (int*[]){ dscale->stride }, 1, 0);
		}

		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RMSNORM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_rmsnorm_back;
}
