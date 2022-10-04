#ifndef GUARD_ccv_nnc_mps_h
#define GUARD_ccv_nnc_mps_h

#include "nnc/ccv_nnc.h"
#include "nnc/_ccv_nnc_stream.h"

void* mpmalloc(int device, size_t size);
void mpfree(int device, void* ptr);
void mpsetoffset(void* ptr, off_t off);
off_t mpgetoffset(void* ptr);

// Stream context
CCV_WARN_UNUSED(ccv_nnc_stream_context_t*) ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_compat_add_callback(ccv_nnc_stream_context_t* const stream, const ccv_nnc_callback_f callback, const ccv_nnc_async_callback_f async_callback, void* const callback_context);
int co_stream_compat_await(co_routine_t* const self, ccv_nnc_stream_context_t* const stream);
void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_deinit_tensor(ccv_nnc_tensor_t* const tensor);
CCV_WARN_UNUSED(void*) ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem);
void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(ccv_nnc_stream_signal_t*) ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal);
void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal);
CCV_WARN_UNUSED(int) ccv_nnc_gpu_device_count(void);

#ifdef __OBJC__

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

id<MTLDevice> ccv_nnc_default_device(void);
id<MTLCommandQueue> ccv_nnc_default_queue(void);
id<MTLCommandBuffer> ccv_nnc_stream_context_get_command_buffer(ccv_nnc_stream_context_t* const stream_context);
CCV_WARN_UNUSED(MPSDataType) ccv_nnc_mps_datatype(const int datatype); // Get the datatype corresponding to MPS datatype.
CCV_WARN_UNUSED(MPSGraphTensor*) ccv_nnc_mps_graph_tensor_input(MPSGraph* graph, const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], MPSGraphTensor** input);
CCV_WARN_UNUSED(MPSGraphTensor*) ccv_nnc_mps_graph_tensor_result(MPSGraph* graph, MPSGraphTensor* tensor, ccv_nnc_tensor_view_t* tensor_view);
CCV_WARN_UNUSED(MPSGraphTensorData*) ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC]);

#endif

#endif
