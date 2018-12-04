/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_compat_h
#define GUARD_ccv_nnc_compat_h

#ifndef HAVE_CUDA
#error "This file requires to be compiled with CUDA support."
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "../ccv_nnc.h"
#include "../_ccv_nnc_stream.h"

// Simple counterparts of ccmalloc / ccfree.
void* cumalloc(int device, size_t size);
void cufree(int device, void* ptr);
void cudevice(int device);
void cumemcpy(void* dest, const int dest_type, const void* src, const int src_type, size_t n);
void* cuhostalloc(size_t size);
void cuhostfree(void* ptr);
int curegister(void* ptr, size_t size);
void cuunregister(void* ptr);

// Stream context
CCV_WARN_UNUSED(ccv_nnc_stream_context_t*) ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_compat_task_synchronize(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_context_t* const stream);
void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_deinit_tensor(ccv_nnc_tensor_t* const tensor);
CCV_WARN_UNUSED(void*) ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem);
void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(ccv_nnc_stream_signal_t*) ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal);
CCV_WARN_UNUSED(int) ccv_nnc_gpu_device_count(void);
#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__
#ifndef __cplusplus
#error "A C++ compiler is required for this header."
#endif

// CUDA objects are C++ objects, this is a C++ header.
#include <cuda.h>
#include <cublas_v2.h>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#if CUDNN_VERSION < 5000 // Doesn't support CUDNN with version lower than 5000 (major version 5).
#undef HAVE_CUDNN
#endif
#endif
#ifdef HAVE_NCCL
#include <nccl.h>
#if NCCL_VERSION_CODE < 2200 // Doesn't support NCCL with version lower than 2.2
#undef HAVE_NCCL
#endif
#endif

#define CUDA_NUM_THREADS (512)
#define CUDA_1D_KERNEL_LOOP(i, n) \
	for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define CUDA_GET_BLOCKS(n) ccv_min(((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS, 4096)

extern "C" {
// Stream context methods to get the underlying objects, note that none of these methods are thread-safe.
CCV_WARN_UNUSED(int) ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudaStream_t) ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cublasHandle_t) ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* const stream_context);

#ifdef NDEBUG
#define CUBLAS_ENFORCE(status) status
#else
#define CUBLAS_ENFORCE(status) do {                               \
	if (status != CUBLAS_STATUS_SUCCESS) {                        \
		printf("[%s:%d]:CUBLAS - Error: %d\n",                    \
				__FILE__, __LINE__, (int)status);                 \
		cudaDeviceReset();                                        \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
} while (0)
#endif

// Return floating point one on device memory.
CCV_WARN_UNUSED(float*) ccv_nnc_stream_context_get_ones(const ccv_nnc_stream_context_t* const stream_context, const int n);

#ifdef HAVE_CUDNN
CCV_WARN_UNUSED(cudnnHandle_t) ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* const stream_context);
// CUDNN related descriptors.
CCV_WARN_UNUSED(cudnnActivationDescriptor_t) ccv_nnc_stream_context_get_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnConvolutionDescriptor_t) ccv_nnc_stream_context_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnDropoutDescriptor_t) ccv_nnc_stream_context_get_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, const float p);
CCV_WARN_UNUSED(cudnnFilterDescriptor_t) ccv_nnc_stream_context_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnOpTensorDescriptor_t) ccv_nnc_stream_context_get_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnPoolingDescriptor_t) ccv_nnc_stream_context_get_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnReduceTensorDescriptor_t) ccv_nnc_stream_context_get_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(cudnnTensorDescriptor_t) ccv_nnc_stream_context_get_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_return_activation_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnActivationDescriptor_t activation_desc);
void ccv_nnc_stream_context_return_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnConvolutionDescriptor_t convolution_desc);
void ccv_nnc_stream_context_return_dropout_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnDropoutDescriptor_t dropout_desc);
void ccv_nnc_stream_context_return_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnFilterDescriptor_t filter_desc);
void ccv_nnc_stream_context_return_op_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnOpTensorDescriptor_t op_tensor_desc);
void ccv_nnc_stream_context_return_pooling_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnPoolingDescriptor_t pooling_desc);
void ccv_nnc_stream_context_return_reduce_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnReduceTensorDescriptor_t reduce_tensor_desc);
void ccv_nnc_stream_context_return_tensor_descriptor(const ccv_nnc_stream_context_t* const stream_context, cudnnTensorDescriptor_t tensor_desc);

#ifdef NDEBUG
#define CUDNN_ENFORCE(status) status
#else
#define CUDNN_ENFORCE(status) do {                                \
	if (status != CUDNN_STATUS_SUCCESS) {                         \
		printf("[%s:%d]:CUDNN - Error: %s\n",                     \
				__FILE__, __LINE__, cudnnGetErrorString(status)); \
		cudaDeviceReset();                                        \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
} while (0)
#endif

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnTensorDescriptor_t descriptor;
	ccv_numeric_data_t data;
} ccv_nnc_cudnn_tensor_view_descriptor_t;
ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor_for_op(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* const tensor);
ccv_nnc_cudnn_tensor_view_descriptor_t ccv_nnc_cudnn_get_tensor_view_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_view_t* tensor);
void ccv_nnc_cudnn_deinit_tensor_view_descriptor(const ccv_nnc_cudnn_tensor_view_descriptor_t tensor_desc);

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnFilterDescriptor_t descriptor;
	ccv_numeric_data_t data;
} ccv_nnc_cudnn_filter_descriptor_t;
ccv_nnc_cudnn_filter_descriptor_t ccv_nnc_cudnn_get_filter_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_tensor_t* tensor);
void ccv_nnc_cudnn_deinit_filter_descriptor(const ccv_nnc_cudnn_filter_descriptor_t filter_desc);

typedef struct {
	const ccv_nnc_stream_context_t* stream_context;
	cudnnConvolutionDescriptor_t descriptor;
} ccv_nnc_cudnn_convolution_descriptor_t;
ccv_nnc_cudnn_convolution_descriptor_t ccv_nnc_cudnn_get_convolution_descriptor(const ccv_nnc_stream_context_t* const stream_context, const ccv_nnc_hint_t hint);
void ccv_nnc_cudnn_deinit_convolution_descriptor(const ccv_nnc_cudnn_convolution_descriptor_t convolution_desc);
#endif

#ifdef HAVE_NCCL
CCV_WARN_UNUSED(ncclComm_t) ccv_nnc_nccl_get_comm(ccv_nnc_stream_context_t* const stream, const int device_count, const int device_id);

#ifdef NDEBUG
#define NCCL_ENFORCE(status) status
#else
#define NCCL_ENFORCE(status) do {                                 \
	if (status != ncclSuccess) {                                  \
		printf("[%s:%d]:NCCL - Error: %s\n",                      \
				__FILE__, __LINE__, ncclGetErrorString(status));  \
		cudaDeviceReset();                                        \
		exit(EXIT_FAILURE);                                       \
	}                                                             \
} while (0)
#endif

#endif
}
#endif

#endif
