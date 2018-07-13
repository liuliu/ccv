#include "ccv_nnc_compat.h"
extern "C" {
#include <nnc/ccv_nnc_easy.h>
}

void* cumalloc(int device, size_t size)
{
	void* ptr = 0;
	cudaSetDevice(device);
	cudaMalloc(&ptr, size);
	return ptr;
}

void cufree(int device, void* ptr)
{
	cudaSetDevice(device);
	cudaFree(ptr);
}

typedef struct {
	int type; // Kept the type specifier.
	cudaStream_t stream;
	cublasHandle_t cublas;
	struct {
		int n;
		float* data;
	} ones;
#ifdef HAVE_CUDNN
	cudnnHandle_t cudnn;
	void* rngs; // user-allocated GPU memory that will hold random number generator states.
	unsigned long long seed;
#endif
} ccv_nnc_stream_context_compat_t;

static __thread ccv_nnc_stream_context_compat_t ccv_nnc_per_thread_gpu_stream_context = {
	.type = CCV_STREAM_CONTEXT_GPU,
#ifdef cudaStreamPerThread
	.stream = cudaStreamPerThread
#endif
};

ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)stream_context)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)ccrealloc(stream_context, sizeof(ccv_nnc_stream_context_compat_t));
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamCreate(&stream_compat->stream);
	stream_compat->cublas = 0;
	stream_compat->ones.data = 0;
#ifdef HAVE_CUDNN
	stream_compat->cudnn = 0;
	stream_compat->rngs = 0;
#endif
	return (ccv_nnc_stream_context_t*)stream_compat;
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamSynchronize(stream_compat->stream);
}

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamDestroy(stream_compat->stream);
	if (stream_compat->cublas)
		cublasDestroy(stream_compat->cublas);
	if (stream_compat->ones.data)
		cudaFreeAsync(stream_compat->ones.data, stream_compat->stream);
#ifdef HAVE_CUDNN
	if (stream_compat->cudnn)
		cudnnDestroy(stream_compat->cudnn);
	if (stream_compat->rngs)
		cudaFreeAsync(stream_compat->rngs, stream_compat->stream);
#endif
}

int ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
		return 0;
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	return CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
}

cudaStream_t ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = &ccv_nnc_per_thread_gpu_stream_context;
	return stream_compat->stream;
}

cublasHandle_t ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = &ccv_nnc_per_thread_gpu_stream_context;
	if (!stream_compat->cublas)
	{
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		cublasCreate(&stream_compat->cublas);
		cublasSetStream(stream_compat->cublas, stream_compat->stream);
	}
	return stream_compat->cublas;
}

// A simple kernel to set all values to 1.
__global__ static void ones(float* x, int n)
{
	const int thidx = blockIdx.x * blockDim.x + threadIdx.x;
	if (thidx < n)
		x[thidx] = 1.;
}

float* ccv_nnc_stream_context_get_ones(const ccv_nnc_stream_context_t* const stream_context, const int n)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = &ccv_nnc_per_thread_gpu_stream_context;
	if (!stream_compat->ones.data || n > stream_compat->ones.n)
	{
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		if (stream_compat->ones.data)
			cudaFreeAsync(stream_compat->ones.data, stream);
		stream_compat->ones.n = n;
		stream_compat->ones.data = (float*)cumalloc(device, sizeof(float) * n);
		const int block_x = (n + 255) >> 8;
		ones<<<block_x, 256, 0, stream>>>(stream_compat->ones.data, n);
	}
	return stream_compat->ones.data;
}

#ifdef HAVE_CUDNN
cudnnHandle_t ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = &ccv_nnc_per_thread_gpu_stream_context;
	if (!stream_compat->cudnn)
	{
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		CUDNN_ENFORCE(cudnnCreate(&stream_compat->cudnn));
		CUDNN_ENFORCE(cudnnSetStream(stream_compat->cudnn, stream_compat->stream));
	}
	return stream_compat->cudnn;
}

cudnnActivationDescriptor_t ccv_nnc_stream_context_get_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnActivationDescriptor_t desc;
	cudnnCreateActivationDescriptor(&desc);
	return desc;
}

cudnnConvolutionDescriptor_t ccv_nnc_stream_context_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnConvolutionDescriptor_t desc;
	cudnnCreateConvolutionDescriptor(&desc);
	return desc;
}

cudnnDropoutDescriptor_t ccv_nnc_stream_context_get_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, const float p)
{
	cudnnDropoutDescriptor_t desc;
	cudnnCreateDropoutDescriptor(&desc);
	cudnnHandle_t cudnn = ccv_nnc_stream_context_get_cudnn(stream_context);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = &ccv_nnc_per_thread_gpu_stream_context;
	size_t state_size;
	cudnnDropoutGetStatesSize(cudnn, &state_size);
	if (stream_compat->rngs)
	{
#if CUDNN_VERSION >= 7100
		cudnnRestoreDropoutDescriptor(desc, cudnn, p, stream_compat->rngs, state_size, stream_compat->seed);
#else
		++stream_compat->seed;
		cudnnSetDropoutDescriptor(desc, cudnn, p, stream_compat->rngs, state_size, stream_compat->seed);
#endif
	} else {
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		cudaMalloc(&stream_compat->rngs, state_size);
		stream_compat->seed = (unsigned long long)stream_compat;
		cudnnSetDropoutDescriptor(desc, cudnn, p, stream_compat->rngs, state_size, stream_compat->seed);
	}
	return desc;
}

cudnnFilterDescriptor_t ccv_nnc_stream_context_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnFilterDescriptor_t desc;
	cudnnCreateFilterDescriptor(&desc);
	return desc;
}

cudnnOpTensorDescriptor_t ccv_nnc_stream_context_get_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnOpTensorDescriptor_t desc;
	cudnnCreateOpTensorDescriptor(&desc);
	return desc;
}

cudnnPoolingDescriptor_t ccv_nnc_stream_context_get_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnPoolingDescriptor_t desc;
	cudnnCreatePoolingDescriptor(&desc);
	return desc;
}

cudnnReduceTensorDescriptor_t ccv_nnc_stream_context_get_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnReduceTensorDescriptor_t desc;
	cudnnCreateReduceTensorDescriptor(&desc);
	return desc;
}

cudnnTensorDescriptor_t ccv_nnc_stream_context_get_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context)
{
	cudnnTensorDescriptor_t desc;
	cudnnCreateTensorDescriptor(&desc);
	return desc;
}

void ccv_nnc_stream_context_return_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnActivationDescriptor_t activation_desc)
{
	cudnnDestroyActivationDescriptor(activation_desc);
}

void ccv_nnc_stream_context_return_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnConvolutionDescriptor_t convolution_desc)
{
	cudnnDestroyConvolutionDescriptor(convolution_desc);
}

void ccv_nnc_stream_context_return_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnDropoutDescriptor_t dropout_desc)
{
	cudnnDestroyDropoutDescriptor(dropout_desc);
}

void ccv_nnc_stream_context_return_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnFilterDescriptor_t filter_desc)
{
	cudnnDestroyFilterDescriptor(filter_desc);
}

void ccv_nnc_stream_context_return_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnOpTensorDescriptor_t op_tensor_desc)
{
	cudnnDestroyOpTensorDescriptor(op_tensor_desc);
}

void ccv_nnc_stream_context_return_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnPoolingDescriptor_t pooling_desc)
{
	cudnnDestroyPoolingDescriptor(pooling_desc);
}

void ccv_nnc_stream_context_return_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnReduceTensorDescriptor_t reduce_tensor_desc)
{
	cudnnDestroyReduceTensorDescriptor(reduce_tensor_desc);
}

void ccv_nnc_stream_context_return_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnTensorDescriptor_t tensor_desc)
{
	cudnnDestroyTensorDescriptor(tensor_desc);
}

ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor)
{
	ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc = {
		stream_context,
		ccv_nnc_stream_context_get_tensor_descriptor(stream_context),
		tensor->data,
	};
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	const int axis_count = ccv_nnc_tensor_nd(tensor->info.dim);
	const int* const inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int i;
	const int offset = CCV_NNC_MAX_DIM + 2 - axis_count;
	if (offset > 0)
	{
		dim[CCV_NNC_MAX_DIM + 1] = tensor->info.dim[axis_count - 1];
		stride[CCV_NNC_MAX_DIM + 1] = 1;
		for (i = CCV_NNC_MAX_DIM; i >= offset; i--)
		{
			dim[i] = tensor->info.dim[i - offset];
			stride[i] = stride[i + 1] * inc[i - offset + 1];
		}
		for (i = 0; i < offset; i++)
		{
			dim[i] = 1;
			stride[i] = stride[offset];
		}
	} else {
		dim[axis_count - 1] = tensor->info.dim[axis_count - 1];
		stride[axis_count - 1] = 1;
		for (i = axis_count - 2; i >= 0; i--)
		{
			dim[i] = tensor->info.dim[i];
			stride[i] = stride[i + 1] * inc[i + 1];
		}
	}
	if (axis_count <= 4)
	{
		CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(tensor_desc.descriptor, CUDNN_DATA_FLOAT, dim[0], dim[1], dim[2], dim[3], stride[0], stride[1], stride[2], stride[3]));
	} else {
		CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(tensor_desc.descriptor, CUDNN_DATA_FLOAT, axis_count, dim, stride));
	}
	return tensor_desc;
}

ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor)
{
	ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc = {
		stream_context,
		ccv_nnc_stream_context_get_tensor_descriptor(stream_context),
		tensor->data,
	};
	// Fill up dimensions with 1s.
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int stride[CCV_NNC_MAX_DIM_ALLOC] = {};
	const int axis_count = ccv_nnc_tensor_nd(tensor->info.dim);
	const int* const inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int i;
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[1];
				dim[2] = dim[3] = 1;
				stride[0] = inc[1];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[0];
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 1] = stride[i + 2] * inc[i + 1];
				}
				stride[0] = stride[1] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				dim[CCV_NNC_MAX_DIM + 1] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				for (i = CCV_NNC_MAX_DIM; i >= 0; i--)
				{
					dim[i] = tensor->info.dim[i];
					stride[i] = stride[i + 1] * inc[i + 1];
				}
				break;
			default:
				assert(0);
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1; // Even though technically this should be inc[1] (because hw is after c), however, make it 1 doesn't have any differences and more versatile.
				break;
			case 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[1];
				dim[2] = dim[3] = 1;
				stride[0] = inc[1];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1; // Even though technically this should be inc[1] (because hw is after c), however, make it 1 doesn't have any differences and more versatile.
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[CCV_NNC_MAX_DIM];
				stride[1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i];
					stride[i + 2] = (i == CCV_NNC_MAX_DIM - 1) ? inc[i + 1] : stride[i + 3] * inc[i + 1];
				}
				stride[0] = stride[2] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				dim[0] = tensor->info.dim[0];
				dim[1] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				stride[1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 2] = (i == CCV_NNC_MAX_DIM - 1) ? inc[i + 2] : stride[i + 3] * inc[i + 2];
				}
				stride[0] = stride[2] * inc[1];
				break;
			default:
				assert(0);
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_CHWN) {
		switch (axis_count)
		{
			case 1:
				dim[0] = dim[2] = dim[3] = 1;
				dim[1] = tensor->info.dim[0];
				stride[0] = inc[0];
				stride[1] = 1;
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = 1;
				break;
			case 2:
				dim[0] = tensor->info.dim[1];
				dim[1] = tensor->info.dim[0];
				dim[2] = dim[3] = 1;
				stride[0] = 1;
				stride[1] = inc[1];
				for (i = 2; i < CCV_NNC_MAX_DIM + 2; i++)
					stride[i] = inc[1];
				break;
			case CCV_NNC_MAX_DIM + 1:
				dim[0] = 1;
				dim[1] = tensor->info.dim[0];
				stride[CCV_NNC_MAX_DIM + 1] = 1;
				for (i = CCV_NNC_MAX_DIM - 1; i >= 0; i--)
				{
					dim[i + 2] = tensor->info.dim[i + 1];
					stride[i + 1] = stride[i + 2] * inc[i + 1];
				}
				stride[0] = stride[1] * inc[0];
				break;
			case CCV_NNC_MAX_DIM + 2:
				dim[0] = tensor->info.dim[CCV_NNC_MAX_DIM + 1];
				stride[0] = 1;
				dim[CCV_NNC_MAX_DIM + 1] = tensor->info.dim[CCV_NNC_MAX_DIM];
				stride[CCV_NNC_MAX_DIM + 1] = inc[CCV_NNC_MAX_DIM + 1];
				for (i = CCV_NNC_MAX_DIM; i > 0; i--)
				{
					dim[i] = tensor->info.dim[i - 1];
					stride[i] = stride[i + 1] * inc[i]; // inc[i] is actually the one before.
				}
				break;
			default:
				assert(0);
		}
	}
	if (CCV_NNC_MAX_DIM == 2)
	{
		CUDNN_ENFORCE(cudnnSetTensor4dDescriptorEx(tensor_desc.descriptor, CUDNN_DATA_FLOAT, dim[0], dim[1], dim[2], dim[3], stride[0], stride[1], stride[2], stride[3]));
	} else {
		CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(tensor_desc.descriptor, CUDNN_DATA_FLOAT, CCV_NNC_MAX_DIM + 2, dim, stride));
	}
	return tensor_desc;
}

void ccv_nnc_cudnn_deinit_tensor_view_descriptor(const ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc)
{
	ccv_nnc_stream_context_return_tensor_descriptor(tensor_desc.stream_context, tensor_desc.descriptor);
}

ccv_nnc_cudnn_filter_descriptor_t ccv_nnc_cudnn_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_t* const tensor)
{
	ccv_nnc_cudnn_filter_descriptor_t filter_desc = {
		stream_context,
		ccv_nnc_stream_context_get_filter_descriptor(stream_context),
		tensor->data,
	};
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	const int nd = ccv_nnc_tensor_nd(tensor->info.dim);
	assert(nd == CCV_NNC_MAX_DIM + 2);
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {};
	int i;
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		for (i = 0; i < nd; i++)
			dim[i] = tensor->info.dim[i];
		if (nd == 4)
		{
			CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, dim[0], dim[1], dim[2], dim[3]));
		} else {
			CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nd, dim));
		}
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = tensor->info.dim[0];
		dim[1] = tensor->info.dim[nd - 1];
		for (i = 2; i < nd; i++)
			dim[i] = tensor->info.dim[i - 1];
		if (nd == 4)
		{
			CUDNN_ENFORCE(cudnnSetFilter4dDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, dim[0], dim[1], dim[2], dim[3]));
		} else {
			CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, nd, dim));
		}
	}
	return filter_desc;
}

void ccv_nnc_cudnn_deinit_filter_descriptor(const ccv_nnc_cudnn_filter_descriptor_t filter_desc)
{
	ccv_nnc_stream_context_return_filter_descriptor(filter_desc.stream_context, filter_desc.descriptor);
}

ccv_nnc_cudnn_convolution_descriptor_t ccv_nnc_cudnn_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_hint_t hint)
{
	ccv_nnc_cudnn_convolution_descriptor_t convolution_desc = {
		stream_context,
		ccv_nnc_stream_context_get_convolution_descriptor(stream_context),
	};
	int i;
	int p[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		p[i] = ccv_max(hint.border.begin[i], hint.border.end[i]);
	int v[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		v[i] = hint.stride.dim[i];
	if (CCV_NNC_MAX_DIM == 2)
	{
		CUDNN_ENFORCE(cudnnSetConvolution2dDescriptor(convolution_desc.descriptor, p[0], p[1], v[0], v[1], 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	} else {
		int u[CCV_NNC_MAX_DIM];
		for (i = 0; i < CCV_NNC_MAX_DIM; i++)
			u[i] = 1;
		CUDNN_ENFORCE(cudnnSetConvolutionNdDescriptor(convolution_desc.descriptor, CCV_NNC_MAX_DIM, p, v, u, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
	}
	return convolution_desc;
}

void ccv_nnc_cudnn_deinit_convolution_descriptor(const ccv_nnc_cudnn_convolution_descriptor_t convolution_desc)
{
	ccv_nnc_stream_context_return_convolution_descriptor(convolution_desc.stream_context, convolution_desc.descriptor);
}
#endif

static void _ccv_nnc_cufree_stream_callback(cudaStream_t stream, cudaError_t status, void* ptr)
{
	cudaFree(ptr);
}

void cudaFreeAsync(void* ptr, cudaStream_t stream)
{
	cudaStreamAddCallback(stream, _ccv_nnc_cufree_stream_callback, ptr, 0);
}
