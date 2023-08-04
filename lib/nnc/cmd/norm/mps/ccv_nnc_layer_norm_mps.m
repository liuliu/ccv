#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_layer_norm_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	assert(output_size == 3);
	ccv_nnc_tensor_view_t at = ccv_nnc_get_tensor_view(inputs[0]);
	ccv_nnc_tensor_view_t scalet = ccv_nnc_get_tensor_view(inputs[1]);
	ccv_nnc_tensor_view_t biast = ccv_nnc_get_tensor_view(inputs[2]);
	ccv_nnc_tensor_view_t bt = ccv_nnc_get_tensor_view(outputs[0]);
	ccv_nnc_tensor_view_t saved_meant = ccv_nnc_get_tensor_view(outputs[1]);
	ccv_nnc_tensor_view_t saved_inv_stdt = ccv_nnc_get_tensor_view(outputs[2]);
	ccv_nnc_tensor_view_alignment((ccv_nnc_tensor_view_t*[]){
		&at,
		&saved_meant,
		&saved_inv_stdt,
		&bt
	}, 4);
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
        (at.info.datatype == bt.info.datatype) &&
        (at.info.datatype == saved_meant.info.datatype) &&
        (at.info.datatype == saved_inv_stdt.info.datatype) &&
        (at.info.datatype == scalet.info.datatype) &&
        (at.info.datatype == biast.info.datatype);
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
    
    int tensor_dim_count = -1;
    
    if (use_mfa) {
      const int source_nd = ccv_nnc_tensor_nd(at.info.dim);
      const int destination_nd = ccv_nnc_tensor_nd(bt.info.dim);
      const int saved_mean_nd = ccv_nnc_tensor_nd(saved_meant.info.dim);
      const int saved_standard_deviation_reciprocal_nd = ccv_nnc_tensor_nd(saved_inv_stdt.info.dim);
      const int scale_nd = ccv_nnc_tensor_nd(scalet.info.dim);
      const int bias_nd = ccv_nnc_tensor_nd(biast.info.dim);
      
      if (source_nd == 3 &&
          destination_nd == 3 &&
          saved_mean_nd == 3 &&
          saved_standard_deviation_reciprocal_nd == 3 &&
          scale_nd == 3 &&
          bias_nd == 3)
      {
        tensor_dim_count = 3;
      }
      else if (source_nd == 4 &&
               destination_nd == 4 &&
               saved_mean_nd == 4 &&
               saved_standard_deviation_reciprocal_nd == 4 &&
               scale_nd == 4 &&
               bias_nd == 4)
      {
        tensor_dim_count = 4;
      }
      else
      {
        use_mfa = false;
        fallback_reason = "Unsupported input shape.";
      }
      
      if (!CCV_IS_TENSOR_CONTIGUOUS(&at) ||
          !CCV_IS_TENSOR_CONTIGUOUS(&bt) ||
          !CCV_IS_TENSOR_CONTIGUOUS(&saved_meant) ||
          !CCV_IS_TENSOR_CONTIGUOUS(&saved_inv_stdt) ||
          !CCV_IS_TENSOR_CONTIGUOUS(&scalet) ||
          !CCV_IS_TENSOR_CONTIGUOUS(&biast))
      {
        use_mfa = false;
        fallback_reason = "Strided.";
      }
    }
    
    // Start off encoding, assuming there is no batch.
    int channel_count = -1;
    int channel_groups = -1;
    int sequence_count = -1;
    int data_batch_dim = -1;
    int scale_translation_batch_dim = -1;
    uint8_t data_batched = 0;
    uint8_t scale_translation_batched = 0;
    
    if (use_mfa) {
      int source_dim[CCV_NNC_MAX_DIM_ALLOC];
      int destination_dim[CCV_NNC_MAX_DIM_ALLOC];
      int saved_mean_dim[CCV_NNC_MAX_DIM_ALLOC];
      int saved_standard_deviation_reciprocal_dim[CCV_NNC_MAX_DIM_ALLOC];
      int scale_dim[CCV_NNC_MAX_DIM_ALLOC];
      int bias_dim[CCV_NNC_MAX_DIM_ALLOC];
      ccv_nnc_tensor_view_get_dim(&at, source_dim);
      ccv_nnc_tensor_view_get_dim(&bt, destination_dim);
      ccv_nnc_tensor_view_get_dim(&saved_meant, saved_mean_dim);
      ccv_nnc_tensor_view_get_dim(&saved_inv_stdt, saved_standard_deviation_reciprocal_dim);
      ccv_nnc_tensor_view_get_dim(&scalet, scale_dim);
      ccv_nnc_tensor_view_get_dim(&biast, bias_dim);
      
      if (source_dim[0] != destination_dim[0] ||
          source_dim[0] != saved_mean_dim[0] ||
          source_dim[0] != saved_standard_deviation_reciprocal_dim[0])
      {
        use_mfa = false;
        fallback_reason = "Shape not handled yet (1).";
        printf("(1)\n");
        exit(8);
      }
      
      if (tensor_dim_count == 3) {
        // Example from Stable Diffusion.
        if (source_dim[1] != destination_dim[1] ||
            source_dim[1] != saved_mean_dim[1] ||
            source_dim[1] != saved_standard_deviation_reciprocal_dim[1])
        {
          use_mfa = false;
          fallback_reason = "Shape not handled yet (2).";
          printf("(2)\n");
          exit(8);
        }
        
        if (scale_dim[0] != 1 ||
            scale_dim[1] != 1 ||
            bias_dim[0] != 1 ||
            bias_dim[1] != 1)
        {
          use_mfa = false;
          fallback_reason = "Shape not handled yet (3).";
          printf("(3)\n");
          exit(8);
        }
        
        if (source_dim[2] != destination_dim[2] ||
            source_dim[2] != scale_dim[2] ||
            source_dim[2] != bias_dim[2])
        {
          use_mfa = false;
          fallback_reason = "Shape not handled yet (4).";
          printf("(4)\n");
          exit(8);
        }
        
        if (saved_mean_dim[2] != 1 ||
            saved_standard_deviation_reciprocal_dim[2] != 1)
        {
          use_mfa = false;
          printf("%d %d %d\n", source_dim[0], source_dim[1], source_dim[2]);
          printf("%d %d %d\n", destination_dim[0], destination_dim[1], destination_dim[2]);
          printf("%d %d %d\n", saved_mean_dim[0], saved_mean_dim[1], saved_mean_dim[2]);
          printf("%d %d %d\n", saved_standard_deviation_reciprocal_dim[0], saved_standard_deviation_reciprocal_dim[1], saved_standard_deviation_reciprocal_dim[2]);
          printf("%d %d %d\n", scale_dim[0], scale_dim[1], scale_dim[2]);
          printf("%d %d %d\n", bias_dim[0], bias_dim[1], bias_dim[2]);
          fallback_reason = "Shape not handled yet (5).";
          exit(8);
        }
        
        channel_count = source_dim[2];
        channel_groups = 1;
        sequence_count = source_dim[1];
        data_batch_dim = source_dim[0];
        data_batched = true;
        scale_translation_batched = false;
      } else if (tensor_dim_count == 4) {
        // Example from the unit tests.
        for (int i = 1; i < 4; ++i) {
          if (source_dim[i] != destination_dim[i] ||
              source_dim[i] != scale_dim[i] ||
              source_dim[i] != bias_dim[i])
          {
            use_mfa = false;
            fallback_reason = "Shape not handled yet (2).";
          }
        }
        
        for (int i = 1; i < 4; ++i) {
          if (saved_mean_dim[i] != 1 ||
              saved_standard_deviation_reciprocal_dim[i] != 1)
          {
            use_mfa = false;
            fallback_reason = "Shape not handled yet (3).";
          }
        }
      }
      
      channel_count = source_dim[1] * source_dim[2] * source_dim[3];
      channel_groups = 1;
      sequence_count = 1;
      data_batch_dim = source_dim[0];
      data_batched = true;
      scale_translation_batched = false;
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
        .scale_translation_batched = scale_translation_batched,
        .layer_normalization = true,
        
        .batch_dims_data = { 0 },
        .batch_dims_scale_translation = { 0 },
        .reuse_saved_statistics = 0,
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
      mtl_buffer_t* tensors[7] = {
        mpgetbuffer(inputs[0]), // source
        mpgetbuffer(outputs[0]), // destination
        mpgetbuffer(outputs[1]), // saved_mean
        mpgetbuffer(outputs[2]), // saved_standard_deviation_reciprocal
        mpgetbuffer(inputs[1]), // channel_scales
        mpgetbuffer(inputs[2]), // channel_translations
        NULL,
      };
      size_t tensor_offsets[6] = {
        at.dataof, // source offset
        bt.dataof, // destination offset
        saved_meant.dataof, // saved_mean offset
        saved_inv_stdt.dataof, // saved_standard_deviation_reciprocal offset
        scalet.dataof, // channel_scales offset
        biast.dataof, // channel_translations offset
      };
      ccv_nnc_mfa_encode_normalization(context, params, command_batch, tensors, tensor_offsets);
      ccv_nnc_stream_context_finish_command_batch(stream_context, command_batch);
    } else {
      MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
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
        MPSGraphTensor* mps_a_subtract_mean = [graph subtractionWithPrimaryTensor:mps_a secondaryTensor:mps_saved_mean name:nil];
        MPSGraphTensor* mps_saved_inv_std;
        const double epsilon = cmd.info.lnorm.epsilon;
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
        [resultTensors addObject:mps_b];
        [resultTensors addObject:mps_saved_mean];
        [resultTensors addObject:mps_saved_inv_std];
      });
      // I don't think that I want to implement saved_mean / saved_inv_std properly just yet.
      MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(&at, at.info.dim, at.stride);
      MPSGraphTensorData* data_scale = ccv_nnc_mps_graph_tensor_data(&scalet, scalet.info.dim, scalet.stride);
      MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(&biast, biast.info.dim, biast.stride);
      MPSGraphTensorData* data[] = {data_a, data_scale, data_bias};
      ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], (ccv_nnc_tensor_view_t* []){ &bt, &saved_meant, &saved_inv_stdt }, (int*[]){ bt.info.dim, saved_meant.info.dim, saved_inv_stdt.info.dim }, (int*[]){ bt.stride, saved_meant.stride, saved_inv_stdt.stride }, 3);
      ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
    }
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_layer_norm_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
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

			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];			
			for (int k = 0; k < cmd.info.lnorm.count; k++) {
				[axes addObject:@(cmd.info.lnorm.axis[k])];
			}

			MPSGraphTensor* mps_dscale = nil;
			MPSGraphTensor* mps_dbias = nil;
			MPSGraphTensor* mps_h = nil;

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
				
				// ah[x] * gss[x]
				MPSGraphTensor* ahgss = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:gss name:nil];
										
				// ah[x] * gssp[x]; reduce
				MPSGraphTensor* ahgssr = [graph reductionSumWithTensor:ahgss axes:axes name:nil];

				// ahp[x] * ahgssrp2[x]
				MPSGraphTensor* gssrp_ahp_ahgssrp = [graph multiplicationWithPrimaryTensor:ah secondaryTensor:ahgssr name:nil];

				// gssr = gss reduce
				MPSGraphTensor* gssr = [graph reductionSumWithTensor:gss axes:axes name:nil];

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

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_LAYER_NORM_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_layer_norm_back;
}
