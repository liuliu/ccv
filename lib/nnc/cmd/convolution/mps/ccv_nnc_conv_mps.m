#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_conv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 2);
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[0];
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* bias = input_size > 2 ? (const ccv_nnc_tensor_view_t*)inputs[2] : 0;
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[0];
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int wdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(w, wdim);
	int wstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(w, wstride);
	int bdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(b, bdim);
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(b, bstride);
	assert(w->info.format == CCV_TENSOR_FORMAT_NCHW);
	int biasdim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int biasstride[CCV_NNC_MAX_DIM_ALLOC] = {0};
	if (bias)
	{
		assert(ccv_nnc_tensor_nd(bias->info.dim) == 1);
		int i;
		for (i = 0; i < CCV_NNC_MAX_DIM + 2; i++)
			biasdim[i] = 1;
		int c;
		if (b->info.format == CCV_TENSOR_FORMAT_NCHW)
			c = 1;
		else if (b->info.format == CCV_TENSOR_FORMAT_NHWC)
			c = CCV_NNC_MAX_DIM + 1;
		else
			c = 0;
		biasdim[c] = bias->info.dim[0];
		if (CCV_IS_TENSOR_VIEW(bias))
		{
			for (i = 0; i < c; i++)
				biasstride[i] = bias->info.dim[0] * bias->stride[0];
			for (i = c; i < CCV_NNC_MAX_DIM + 2; i++)
				biasstride[i] = bias->stride[0];
		}
	}
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int* adim_r = adim;
		int* astride_r = astride;
		int* wdim_r = wdim;
		int* wstride_r = wstride;
		int* biasdim_r = biasdim;
		int* biasstride_r = biasstride;
		int indices[3];
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
			[inputTensors addObject:mps_input_a];
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
			[inputShapedTypes addObject:mps_a_shape];
			MPSGraphTensor* mps_input_w;
			MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, wdim_r, wstride_r, &mps_input_w);
			[inputTensors addObject:mps_input_w];
			MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, wdim_r, wstride_r);
			[inputShapedTypes addObject:mps_w_shape];
			MPSGraphConvolution2DOpDescriptor* descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:1 dilationRateInY:1 groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:ccv_nnc_mps_tensor_data_layout(a->info.format) weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
			MPSGraphTensor* mps_b = [graph convolution2DWithSourceTensor:mps_a weightsTensor:mps_w descriptor:descriptor name:nil];
			if (bias)
			{
				MPSGraphTensor* mps_input_bias;
				MPSGraphTensor* mps_bias = ccv_nnc_mps_graph_tensor_input(graph, bias, biasdim_r, biasstride_r, &mps_input_bias);
				[inputTensors addObject:mps_input_bias];
				MPSGraphShapedType* mps_bias_shape = ccv_nnc_mps_graph_tensor_input_shape(bias, biasdim_r, biasstride_r);
				[inputShapedTypes addObject:mps_bias_shape];
				// Add support broadcast directly.
				mps_b = [graph additionWithPrimaryTensor:mps_b secondaryTensor:mps_bias name:nil];
			}
			[resultTensors addObject:mps_b];
		});
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, adim, astride);
		MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data(w, wdim, wstride);
		if (bias)
		{
			MPSGraphTensorData* data_bias = ccv_nnc_mps_graph_tensor_data(bias, biasdim, biasstride);
			MPSGraphTensorData* data[] = {data_a, data_w, data_bias};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]], data[indices[2]]], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
		} else {
			MPSGraphTensorData* data[] = {data_a, data_w};
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]]], &b, (int*[]){ bdim }, (int*[]){ bstride }, 1);
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_conv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	printf("_ccv_nnc_conv_back\n");

	// inputs: gradient, forw prop input, [w]
	// outputs: [output gradient], weight updates, (no bias updates yet)
	assert(input_size >= 2 && output_size >= 2);
	const ccv_nnc_tensor_view_t* g = (const ccv_nnc_tensor_view_t*)inputs[0]; // gradients input
	const ccv_nnc_tensor_view_t* a = (const ccv_nnc_tensor_view_t*)inputs[1]; // forward input
	const ccv_nnc_tensor_view_t* w = (const ccv_nnc_tensor_view_t*)inputs[2]; // weights input

	ccv_nnc_tensor_view_t* dw = output_size > 1 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0; // weight_update
	assert(CCV_IS_TENSOR_CONTIGUOUS(dw));
	ccv_nnc_tensor_view_t* h = (ccv_nnc_tensor_view_t*)outputs[0]; // output gradients
	if (!(flags & CCV_NNC_ACCUMULATE_OUTPUT)) // reset the gradients to 0
	{
		if (dw)
			memset(dw->data.u8, 0, sizeof(float) * ccv_nnc_tensor_count(dw->info));
	}
	
	int gdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(g, gdim);
	int gstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(g, gstride);
	int hdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(h, hdim);
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(h, hstride);
	int adim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, adim);
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(a, astride);
	int wdim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(w, wdim);
	int wstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(w, wstride);
	int dw_dim[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(dw, dw_dim);
	int dw_stride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_stride(dw, dw_stride);
	assert(w->info.format == CCV_TENSOR_FORMAT_NCHW);
	
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, hint, flags, inputs, input_size, outputs, output_size);
		int* gdim_r = gdim;
		int* gstride_r = gstride;
		int* hdim_r = hdim;
		int* hstride_r = hstride;
		int* adim_r = adim;
		int* astride_r = astride;
		int* wdim_r = wdim;
		int* wstride_r = wstride;
		int* dw_dim_r = dw_dim;
		int* dw_stride_r = dw_stride;
		int indices[3];

		// [output gradient]
		MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
			MPSGraphTensor* mps_input_g;
			MPSGraphTensor* mps_g = ccv_nnc_mps_graph_tensor_input(graph, g, gdim_r, gstride_r, &mps_input_g);
			MPSGraphShapedType* mps_g_shape = ccv_nnc_mps_graph_tensor_input_shape(g, hdim_r, gstride_r);
			[inputTensors addObject:mps_g];
			[inputShapedTypes addObject:mps_g_shape];

			MPSGraphTensor* mps_input_w;
			MPSGraphTensor* mps_w = ccv_nnc_mps_graph_tensor_input(graph, w, wdim_r, wstride_r, &mps_input_w);
			MPSGraphShapedType* mps_w_shape = ccv_nnc_mps_graph_tensor_input_shape(w, wdim_r, wstride_r);
			[inputTensors addObject:mps_w];
			[inputShapedTypes addObject:mps_w_shape];

			MPSGraphShapedType* mps_h_shape = ccv_nnc_mps_graph_tensor_input_shape(h, hdim_r, hstride_r);
			MPSGraphConvolution2DOpDescriptor* descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:1 dilationRateInY:1 groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
			printf("// gradient updates		\n");
			// gradient updates		
			MPSGraphTensor* mps_h = [graph convolution2DDataGradientWithIncomingGradientTensor:mps_g
                                                                          weightsTensor:mps_w
                                                                            outputShape:mps_h_shape.shape
                                                           forwardConvolutionDescriptor:descriptor
                                                                                   name:nil];
			[resultTensors addObject:mps_h];

			MPSGraphTensor* mps_input_a;
			MPSGraphTensor* mps_a = ccv_nnc_mps_graph_tensor_input(graph, a, adim_r, astride_r, &mps_input_a);
			MPSGraphShapedType* mps_a_shape = ccv_nnc_mps_graph_tensor_input_shape(a, adim_r, astride_r);
			[inputTensors addObject:mps_a];
			[inputShapedTypes addObject:mps_a_shape];

			MPSGraphShapedType* mps_dw_shape = ccv_nnc_mps_graph_tensor_input_shape(dw, dw_dim_r, dw_stride_r);

			MPSGraphConvolution2DOpDescriptor* dw_descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:hint.stride.dim[1] strideInY:hint.stride.dim[0] dilationRateInX:1 dilationRateInY:1 groups:cmd.info.convolution.groups paddingLeft:hint.border.begin[1] paddingRight:hint.border.end[1] paddingTop:hint.border.begin[0] paddingBottom:hint.border.end[0] paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNCHW weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];			
			// weight updates
			printf("// weight updates		\n");

			MPSGraphTensor* mps_dw = [graph convolution2DWeightsGradientWithIncomingGradientTensor:mps_g
                                                                          	  sourceTensor:mps_a
                                                                            outputShape:mps_dw_shape.shape
                                                           forwardConvolutionDescriptor:dw_descriptor
                                                                                   name:nil];

			[resultTensors addObject:mps_dw];

		});
		printf("// ccv_nnc_mps_graph_tensor_data		\n");

		MPSGraphTensorData* data_g = ccv_nnc_mps_graph_tensor_data(g, gdim, gstride);
		MPSGraphTensorData* data_w = ccv_nnc_mps_graph_tensor_data(w, w->info.dim, w->stride);
		MPSGraphTensorData* data_a = ccv_nnc_mps_graph_tensor_data(a, a->info.dim, a->stride);

		MPSGraphTensorData* data[] = {data_g, data_w, data_a};
		ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data[indices[0]], data[indices[1]],  data[indices[2]]], (ccv_nnc_tensor_view_t* []){ h, dw }, (int*[]){ hdim, dw_dim }, (int*[]){ hstride, dw_stride }, 2);
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CONVOLUTION_BACKWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_conv_back;
}