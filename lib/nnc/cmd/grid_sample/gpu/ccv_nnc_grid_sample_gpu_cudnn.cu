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

static void _ccv_nnc_grid_sample_nchw_dim(const ccv_nnc_tensor_t* const tensor, int* const n, int* const c, int* const h, int* const w)
{
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	assert(nd == 3 || nd == 4);
	assert(tensor->info.format == CCV_TENSOR_FORMAT_NCHW);
	if (nd == 4)
	{
		*n = tensor->info.dim[0];
		*c = tensor->info.dim[1];
		*h = tensor->info.dim[2];
		*w = tensor->info.dim[3];
	} else {
		*n = 1;
		*c = tensor->info.dim[0];
		*h = tensor->info.dim[1];
		*w = tensor->info.dim[2];
	}
}

static void _ccv_nnc_grid_sample_grid_dim(const ccv_nnc_tensor_t* const grid, int* const n, int* const h, int* const w)
{
	const int nd = ccv_nnc_tensor_nd(grid->info.dim);
	assert(nd == 3 || nd == 4);
	if (nd == 4)
	{
		*n = grid->info.dim[0];
		*h = grid->info.dim[1];
		*w = grid->info.dim[2];
		assert(grid->info.dim[3] == 2);
	} else {
		*n = 1;
		*h = grid->info.dim[0];
		*w = grid->info.dim[1];
		assert(grid->info.dim[2] == 2);
	}
}

static int _ccv_nnc_grid_sample_forw_cudnn(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const grid = inputs[1];
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(a->info.format == CCV_TENSOR_FORMAT_NCHW);
	assert(b->info.format == CCV_TENSOR_FORMAT_NCHW);
	assert(a->info.datatype == CCV_32F);
	assert(grid->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(grid));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));

	int n = 0, c = 0, h = 0, w = 0;
	int bn = 0, bc = 0, bh = 0, bw = 0;
	int gn = 0, gh = 0, gw = 0;
	_ccv_nnc_grid_sample_nchw_dim(a, &n, &c, &h, &w);
	_ccv_nnc_grid_sample_nchw_dim(b, &bn, &bc, &bh, &bw);
	_ccv_nnc_grid_sample_grid_dim(grid, &gn, &gh, &gw);
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
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(a_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
	CUDNN_ENFORCE(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, bn, bc, bh, bw));
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
