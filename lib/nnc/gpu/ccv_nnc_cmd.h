/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_cmd_h
#define GUARD_ccv_nnc_cmd_h

#ifndef HAVE_CUDA
#error "This file requires to be compiled with CUDA support."
#endif

#ifndef __cplusplus
#error "A C++ compiler is required for this header."
#endif

// CUDA objects are C++ objects, this is a C++ header.
#include <cuda.h>
#include <cublas_v2.h>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif

extern "C" {
#include "../ccv_nnc.h"
// Stream context methods to get the underlying objects, note that none of these methods are thread-safe.
CCV_WARN_UNUSED(int) ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* stream_context);
CCV_WARN_UNUSED(cudaStream_t) ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* stream_context);
CCV_WARN_UNUSED(cublasHandle_t) ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* stream_context);
#ifdef HAVE_CUDNN
CCV_WARN_UNUSED(cudnnHandle_t) ccv_nnc_stream_context_get_cudnn(const ccv_nnc_stream_context_t* stream_context);
// CUDNN related descriptors.
CCV_WARN_UNUSED(cudnnConvolutionDescriptor_t) ccv_nnc_stream_context_deq_convolution_desc(const ccv_nnc_stream_context_t* stream_context);
CCV_WARN_UNUSED(cudnnTensorDescriptor_t) ccv_nnc_stream_context_deq_tensor_desc(const ccv_nnc_stream_context_t* stream_context);
CCV_WARN_UNUSED(cudnnFilterDescriptor_t) ccv_nnc_stream_context_deq_filter_desc(const ccv_nnc_stream_context_t* stream_context);
void ccv_nnc_stream_context_enq_convolution_desc(const ccv_nnc_stream_context_t* stream_context, cudnnConvolutionDescriptor_t convolution_desc);
void ccv_nnc_stream_context_enq_tensor_desc(const ccv_nnc_stream_context_t* stream_context, cudnnTensorDescriptor_t tensor_desc);
void ccv_nnc_stream_context_enq_filter_desc(const ccv_nnc_stream_context_t* stream_context, cudnnFilterDescriptor_t filter_desc);
#endif
// Extended memory managements.
void cudaFreeAsync(void* ptr, cudaStream_t stream);
}

#endif
