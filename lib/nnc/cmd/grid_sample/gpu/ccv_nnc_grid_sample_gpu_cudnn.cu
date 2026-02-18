extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDNN

__global__ static void _ccv_nnc_grid_sample_transform_grid_kernel(const size_t count, const float x_scale, const float y_scale, const int x_zero, const int y_zero, const float* const input, float* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float gx = input[i * 2];
		const float gy = input[i * 2 + 1];
		output[i * 2] = x_zero ? 0 : gx * x_scale;
		output[i * 2 + 1] = y_zero ? 0 : gy * y_scale;
	}
}

static int _ccv_nnc_grid_sample_forw_cudnn(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const grid = inputs[1];
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(a->info.format == b->info.format);
	assert(a->info.format == CCV_TENSOR_FORMAT_NCHW || a->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(grid->info.format == CCV_TENSOR_FORMAT_NHWC);
	assert(a->info.datatype == CCV_32F);
	assert(grid->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(grid));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));

	const int format = a->info.format;
	const cudnnTensorFormat_t cudnn_format = (format == CCV_TENSOR_FORMAT_NHWC) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
	const int and_nd = ccv_nnc_tensor_nd(a->info.dim);
	const int bnd = ccv_nnc_tensor_nd(b->info.dim);
	const int gnd = ccv_nnc_tensor_nd(grid->info.dim);
	assert(and_nd == 3 || and_nd == 4);
	assert(bnd == 3 || bnd == 4);
	assert(gnd == 3 || gnd == 4);
	const int ahw = ccv_nnc_tensor_hw(a->info, and_nd, CCV_NNC_MAX_DIM);
	const int bhw = ccv_nnc_tensor_hw(b->info, bnd, CCV_NNC_MAX_DIM);
	const int ghw = ccv_nnc_tensor_hw(grid->info, gnd, CCV_NNC_MAX_DIM);
	assert(ahw >= 0);
	assert(bhw >= 0);
	assert(ghw >= 0);
	const int n = ccv_nnc_tensor_get_n(a->info);
	const int c = ccv_nnc_tensor_get_c(a->info);
	const int h = a->info.dim[ahw];
	const int w = a->info.dim[ahw + 1];
	const int bn = ccv_nnc_tensor_get_n(b->info);
	const int bc = ccv_nnc_tensor_get_c(b->info);
	const int bh = b->info.dim[bhw];
	const int bw = b->info.dim[bhw + 1];
	const int gn = ccv_nnc_tensor_get_n(grid->info);
	const int gh = grid->info.dim[ghw];
	const int gw = grid->info.dim[ghw + 1];
	assert(grid->info.dim[ghw + 2] == 2);
	assert(bn == n);
	assert(bc == c);
	assert(gn == n);
	assert(gh == bh);
	assert(gw == bw);
	assert(cmd.info.grid_sample.align_corners == 0 || cmd.info.grid_sample.align_corners == 1);

	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	cudnnTensorDescriptor_t a_desc = 0;
	cudnnTensorDescriptor_t b_desc = 0;
	cudnnSpatialTransformerDescriptor_t st_desc = 0;
	CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&a_desc));
	CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&b_desc));
	CUDNN_ENFORCE(cudnnCreateSpatialTransformerDescriptor(&st_desc));
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(a_desc, cudnn_format, CUDNN_DATA_FLOAT, n, c, h, w));
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(b_desc, cudnn_format, CUDNN_DATA_FLOAT, bn, bc, bh, bw));
	const int st_dim[4] = {bn, bc, bh, bw};
	CUDNN_ENFORCE(cudnnSetSpatialTransformerNdDescriptor(st_desc, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, st_dim));
	static const float one = 1;
	static const float zero = 0;
	const void* grid_data = grid->data.u8;
	if (!cmd.info.grid_sample.align_corners)
	{
		const size_t count = (size_t)n * gh * gw;
		float* const transformed_grid = (float*)ccv_nnc_stream_context_get_workspace(stream_context, count * 2 * sizeof(float), CCV_TENSOR_GPU_MEMORY);
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		const int x_zero = (w <= 1);
		const int y_zero = (h <= 1);
		const float x_scale = x_zero ? 1 : (float)w / (w - 1);
		const float y_scale = y_zero ? 1 : (float)h / (h - 1);
		_ccv_nnc_grid_sample_transform_grid_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, x_scale, y_scale, x_zero, y_zero, grid->data.f32, transformed_grid);
		grid_data = transformed_grid;
	}
	CUDNN_ENFORCE(cudnnSpatialTfSamplerForward(cudnn, st_desc, &one, a_desc, a->data.u8, grid_data, &zero, b_desc, b->data.u8));
	CUDNN_ENFORCE(cudnnDestroySpatialTransformerDescriptor(st_desc));
	CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(b_desc));
	CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(a_desc));
	return CCV_NNC_EXEC_SUCCESS;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_GRID_SAMPLE_FORWARD, CCV_NNC_BACKEND_GPU_CUDNN)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDNN
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_grid_sample_forw_cudnn;
#endif
}
