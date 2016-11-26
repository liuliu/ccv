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
#ifdef HAVE_CUDNN
	cudnnHandle_t cudnn;
#endif
} ccv_nnc_stream_context_compat_t;

ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* stream_context)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)stream_context)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)ccrealloc(stream_context, sizeof(ccv_nnc_stream_context_compat_t));
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamCreate(&stream_compat->stream);
	stream_compat->cublas = 0;
#ifdef HAVE_CUDNN
	stream_compat->cudnn = 0;
#endif
	return (ccv_nnc_stream_context_t*)stream_compat;
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamSynchronize(stream_compat->stream);
}

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
	cudaSetDevice(device);
	cudaStreamDestroy(stream_compat->stream);
	if (stream_compat->cublas)
		cublasDestroy(stream_compat->cublas);
#ifdef HAVE_CUDNN
	if (stream_compat->cudnn)
		cudnnDestroy(stream_compat->cudnn);
#endif
}

int ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	return CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
}

cudaStream_t ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	return stream_compat->stream;
}

cublasHandle_t ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat->cublas)
	{
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		cublasCreate(&stream_compat->cublas);
		cublasSetStream(stream_compat->cublas, stream_compat->stream);
	}
	return stream_compat->cublas;
}

#ifdef HAVE_CUDNN
cudnnHandle_t ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat->cudnn)
	{
		int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->type);
		cudaSetDevice(device);
		cudnnCreate(&stream_compat->cudnn);
		cudnnSetStream(stream_compat->cudnn, stream_compat->stream);
	}
	return stream_compat->cudnn;
}

cudnnConvolutionDescriptor_t ccv_nnc_stream_context_get_convolution_descriptor(const ccv_nnc_stream_context_t* stream_context)
{
	cudnnConvolutionDescriptor_t desc;
	cudnnCreateConvolutionDescriptor(&desc);
	return desc;
}

cudnnTensorDescriptor_t ccv_nnc_stream_context_get_tensor_descriptor(const ccv_nnc_stream_context_t* stream_context)
{
	cudnnTensorDescriptor_t desc;
	cudnnCreateTensorDescriptor(&desc);
	return desc;
}

cudnnFilterDescriptor_t ccv_nnc_stream_context_get_filter_descriptor(const ccv_nnc_stream_context_t* stream_context)
{
	cudnnFilterDescriptor_t desc;
	cudnnCreateFilterDescriptor(&desc);
	return desc;
}

void ccv_nnc_stream_context_return_convolution_descriptor(const ccv_nnc_stream_context_t* stream_context, cudnnConvolutionDescriptor_t convolution_descriptor)
{
	cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}

void ccv_nnc_stream_context_return_tensor_descriptor(const ccv_nnc_stream_context_t* stream_context, cudnnTensorDescriptor_t tensor_descriptor)
{
	cudnnDestroyTensorDescriptor(tensor_descriptor);
}

void ccv_nnc_stream_context_return_filter_descriptor(const ccv_nnc_stream_context_t* stream_context, cudnnFilterDescriptor_t filter_descriptor)
{
	cudnnDestroyFilterDescriptor(filter_descriptor);
}

ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor(const ccv_nnc_stream_context_t* stream_context, const ccv_nnc_tensor_view_t* tensor)
{
	ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc = {
		stream_context,
		ccv_nnc_stream_context_get_tensor_descriptor(stream_context),
		tensor->data,
	};
	// N is the outer one nevertheless.
	assert(tensor->info.format == CCV_TENSOR_FORMAT_NCHW || tensor->info.format == CCV_TENSOR_FORMAT_NHWC);
	// Fill up dimensions with 1s.
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int i;
	const int nd = CCV_NNC_MAX_DIM + 2;
	// This has to follow NCHW
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
		for (i = 0; i < nd; i++)
			dim[i] = ccv_max(1, tensor->info.dim[nd - 1 - i]);
	else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = ccv_max(1, tensor->info.dim[nd - 1]);
		dim[1] = ccv_max(1, tensor->info.dim[0]);
		for (i = 2; i < nd; i++)
			dim[i] = ccv_max(1, tensor->info.dim[nd - i]);
	}
	const int* inc = CCV_IS_TENSOR_VIEW(tensor) ? tensor->inc : tensor->info.dim;
	int stride[CCV_NNC_MAX_DIM_ALLOC];
	stride[nd - 1] = 1;
	// Compute the stride from inc, so it will fit the tensor view.
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
		for (i = 1; i < CCV_NNC_MAX_DIM_ALLOC && dim[i] > 0; i++)
			stride[nd - 1 - i] = stride[nd - i] * ccv_max(1, inc[i - 1]);
	else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		stride[1] = 1;
		stride[nd - 1] = ccv_max(1, inc[0]);
		for (i = 0; i < nd - 3; i++)
			stride[nd - 2 - i] = stride[nd - 1 - i] * ccv_max(1, inc[i + 1]);
		stride[0] = stride[2] * ccv_max(1, inc[nd - 2]);
	}
	assert_cudnn(cudnnSetTensorNdDescriptor(tensor_desc.descriptor, CUDNN_DATA_FLOAT, nd, dim, stride));
	return tensor_desc;
}

void ccv_nnc_cudnn_deinit_tensor_view_descriptor(const ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc)
{
	ccv_nnc_stream_context_return_tensor_descriptor(tensor_desc.stream_context, tensor_desc.descriptor);
}

ccv_nnc_cudnn_filter_descriptor_t ccv_nnc_cudnn_get_filter_descriptor(const ccv_nnc_stream_context_t* stream_context, const ccv_nnc_tensor_t* tensor)
{
	ccv_nnc_cudnn_filter_descriptor_t filter_desc = {
		stream_context,
		ccv_nnc_stream_context_get_filter_descriptor(stream_context),
		tensor->data,
	};
	assert(!CCV_IS_TENSOR_VIEW(tensor));
	int nd = ccv_nnc_tensor_nd(tensor->info);
	int dim[CCV_NNC_MAX_DIM_ALLOC] = {0};
	int i;
	// Reorder since nnc have different idea about NCHW and NHWC (N is in 3, C is in 0).
	if (tensor->info.format == CCV_TENSOR_FORMAT_NCHW)
	{
		for (i = 0; i < nd; i++)
			dim[i] = tensor->info.dim[nd - 1 - i];
		assert_cudnn(cudnnSetFilterNdDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, nd, dim));
	} else if (tensor->info.format == CCV_TENSOR_FORMAT_NHWC) {
		dim[0] = tensor->info.dim[nd - 1];
		dim[1] = tensor->info.dim[0];
		for (i = 2; i < nd; i++)
			dim[i] = tensor->info.dim[nd - i];
		assert_cudnn(cudnnSetFilterNdDescriptor(filter_desc.descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, nd, dim));
	}
	return filter_desc;
}

void ccv_nnc_cudnn_deinit_filter_descriptor(const ccv_nnc_cudnn_filter_descriptor_t filter_desc)
{
	ccv_nnc_stream_context_return_filter_descriptor(filter_desc.stream_context, filter_desc.descriptor);
}

ccv_nnc_cudnn_convolution_descriptor_t ccv_nnc_cudnn_get_convolution_descriptor(const ccv_nnc_stream_context_t* stream_context, const ccv_nnc_hint_t hint)
{
	ccv_nnc_cudnn_convolution_descriptor_t convolution_desc = {
		stream_context,
		ccv_nnc_stream_context_get_convolution_descriptor(stream_context),
	};
	int i;
	int p[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		p[i] = ccv_max(hint.border.begin[CCV_NNC_MAX_DIM - i], hint.border.end[CCV_NNC_MAX_DIM - i]);
	int v[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		v[i] = hint.stride.dim[CCV_NNC_MAX_DIM - i];
	int u[CCV_NNC_MAX_DIM];
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
		u[i] = 1;
	assert_cudnn(cudnnSetConvolutionNdDescriptor(convolution_desc.descriptor, CCV_NNC_MAX_DIM, p, v, u, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
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
