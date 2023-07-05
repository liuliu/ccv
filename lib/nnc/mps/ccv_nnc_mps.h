#ifndef GUARD_ccv_nnc_mps_h
#define GUARD_ccv_nnc_mps_h

#include "nnc/ccv_nnc.h"
#include "nnc/_ccv_nnc_stream.h"
#include "nnc/mfa/ccv_nnc_mfa.hpp"

void* mpheapalloc(int device, size_t size);
void mpheapfree(int device, void* ptr);
void* mpobjmalloc(int device, size_t size);
void* mpobjcreate(void* ptr, off_t offset, size_t size);
void mpobjfree(int device, void* ptr);
typedef void(*mpmp_f)(int device_id, void* const context);
int mpregmp(int device_id, mpmp_f func, void* const context); // register memory pressure handler
void mpunregmp(const int id); // un-register memory pressure handler.
void* mpmemmap(void* dest, const void* src, size_t n, size_t expected_n, const char* dir, const char* name);
void mpmemcpy(void* dest, const off_t dest_off, const int dest_type, const void* src, const off_t src_off, const int src_type, size_t n);

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
void ccv_nnc_mps_unbounded_command_buffers(int state);
void ccv_nnc_mps_clear_graph_executable_cache(void);

#ifdef __OBJC__

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

typedef struct {
	int format;
	int datatype;
	int nd;
	off_t dataof;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int stride[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_mps_graph_tensor_shape_t;

typedef struct {
	uint32_t cmd;
	ccv_nnc_cmd_param_t params;
	ccv_nnc_hint_t hint;
	int flags;
	int input_size;
	int output_size;
	ccv_nnc_mps_graph_tensor_shape_t* inputs;
	ccv_nnc_mps_graph_tensor_shape_t* outputs;
} ccv_nnc_mps_graph_key_t;

off_t mpgetoffset(const ccv_nnc_tensor_t* const tensor);
id<MTLBuffer> mpgetbuffer(const ccv_nnc_tensor_t* const tensor);
id<MTLDevice> ccv_nnc_default_device(void);
ccv_nnc_mfa_context_t* ccv_nnc_default_mfa_context(void);
CCV_WARN_UNUSED(MPSCommandBuffer*) ccv_nnc_stream_context_start_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context);
void ccv_nnc_stream_context_finish_mps_command_buffer(ccv_nnc_stream_context_t* const stream_context, MPSCommandBuffer* command_buffer);
CCV_WARN_UNUSED(MPSGraphExecutable*) ccv_nnc_mps_graph_executable_cache(const ccv_nnc_mps_graph_key_t key, int* indices, void(NS_NOESCAPE ^block)(MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors));
CCV_WARN_UNUSED(ccv_nnc_mps_graph_key_t) ccv_nnc_mps_graph_key_new(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size);
CCV_WARN_UNUSED(MPSDataType) ccv_nnc_mps_datatype(const int datatype); // Get the datatype corresponding to MPS datatype.
CCV_WARN_UNUSED(MPSGraphTensorNamedDataLayout) ccv_nnc_mps_tensor_data_layout(const int format); // Get the format corresponding to MPS data layout.
CCV_WARN_UNUSED(MPSGraphTensor*) ccv_nnc_mps_graph_tensor_input(MPSGraph* graph, const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC], MPSGraphTensor** input);
CCV_WARN_UNUSED(MPSGraphShapedType*) ccv_nnc_mps_graph_tensor_input_shape(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC]);
CCV_WARN_UNUSED(MPSGraphTensorData*) ccv_nnc_mps_graph_tensor_data(const ccv_nnc_tensor_view_t* tensor_view, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC]);
void ccv_nnc_mps_export_data(MPSGraphTensorData* data, MPSCommandBuffer* command_buffer, ccv_nnc_tensor_view_t* const tensor, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC]);
void ccv_nnc_mps_graph_result(MPSGraph* graph, MPSCommandBuffer* command_buffer, MPSGraphTensorDataDictionary* feeds, MPSGraphTensor* output, ccv_nnc_tensor_view_t* const data, const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC]);
void ccv_nnc_mps_graph_executable_result(MPSGraphExecutable* executable, MPSCommandBuffer* command_buffer, NSArray<MPSGraphTensorData*>* inputsArray, ccv_nnc_tensor_view_t* const* const data, int* dim[CCV_NNC_MAX_DIM_ALLOC], int* stride[CCV_NNC_MAX_DIM_ALLOC], const int size);

#endif

#endif
