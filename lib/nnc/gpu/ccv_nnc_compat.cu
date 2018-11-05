#include "ccv_nnc_compat.h"
extern "C" {
#include <nnc/ccv_nnc_easy.h>
#include <nnc/_ccv_nnc_stream.h>
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

void cudevice(int device)
{
	if (device >= 0)
		cudaSetDevice(device);
}

void cumemcpy(void* dest, const int dest_type, const void* src, const int src_type, size_t n)
{
	if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY)
		cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice);
	else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_CPU_MEMORY)
		cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost);
	else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_CPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_CPU_MEMORY)
		cudaMemcpy(dest, src, n, cudaMemcpyHostToHost);
	else if (CCV_TENSOR_GET_MEMORY(src_type) == CCV_TENSOR_GPU_MEMORY && CCV_TENSOR_GET_MEMORY(dest_type) == CCV_TENSOR_GPU_MEMORY) {
		int device_a = CCV_TENSOR_GET_DEVICE_ID(src_type);
		int device_b = CCV_TENSOR_GET_DEVICE_ID(dest_type);
		if (device_a == device_b)
			cudaMemcpy(dest, src, n, cudaMemcpyDeviceToDevice);
		else
			cudaMemcpyPeer(dest, device_b, src, device_a, n);
	}
}

void* cuhostalloc(size_t size)
{
	void* ptr = 0;
	cudaHostAlloc(&ptr, size, cudaHostAllocPortable);
	return ptr;
}

void cuhostfree(void* ptr)
{
	cudaFreeHost(ptr);
}

int curegister(void* ptr, size_t size)
{
	return cudaSuccess == cudaHostRegister(ptr, size, cudaHostRegisterPortable);
}

void cuunregister(void* ptr)
{
	cudaHostUnregister(ptr);
}

typedef struct {
	ccv_nnc_stream_context_t super;
	cudaStream_t stream;
	cublasHandle_t cublas;
	struct {
		int n;
		float* data;
	} ones;
	struct {
		size_t workspace_size;
		void* workspace;
	} cpu;
	struct {
		size_t workspace_size;
		void* workspace;
	} gpu;
#ifdef HAVE_CUDNN
	cudnnHandle_t cudnn;
	void* rngs; // user-allocated GPU memory that will hold random number generator states.
	unsigned long long seed;
#endif
} ccv_nnc_stream_context_compat_t;

static int cpu_stream_compat_init_per_thread;
static int gpu_stream_compat_init_per_thread;

static ccv_nnc_stream_context_compat_t* _ccv_nnc_default_stream_compat(const int type)
{
	if (type == CCV_STREAM_CONTEXT_GPU)
	{
		int device = 0;
		cudaGetDevice(&device);
		static __thread ccv_nnc_stream_context_compat_t ccv_nnc_per_thread_gpu_stream_context[CCV_STREAM_GET_DEVICE_ID(CCV_COMPUTE_DEVICE_ANY)];
		ccv_nnc_stream_context_compat_t* stream_compat = ccv_nnc_per_thread_gpu_stream_context + device;
		if (!stream_compat->super.type)
		{
			stream_compat->super.type = CCV_STREAM_CONTEXT_GPU;
			CCV_STREAM_SET_DEVICE_ID(stream_compat->super.type, device);
#ifdef cudaStreamPerThread
			stream_compat->stream = cudaStreamPerThread;
#endif
		}
		gpu_stream_compat_init_per_thread = 1;
		return stream_compat;
	} else {
		assert(type == CCV_STREAM_CONTEXT_CPU);
		static __thread ccv_nnc_stream_context_compat_t ccv_nnc_per_thread_cpu_stream_context = {
			.super = {
				.type = CCV_STREAM_CONTEXT_CPU,
			},
		};
		cpu_stream_compat_init_per_thread = 1;
		return &ccv_nnc_per_thread_cpu_stream_context;
	}
}

typedef struct {
	ccv_nnc_stream_signal_t super;
	cudaEvent_t event;
} ccv_nnc_stream_compat_signal_t;

ccv_nnc_stream_signal_t* ccv_nnc_init_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)signal)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)ccrealloc(signal, sizeof(ccv_nnc_stream_compat_signal_t));
	const int device = CCV_STREAM_GET_DEVICE_ID(compat_signal->super.type);
	cudaSetDevice(device);
	cudaEventCreateWithFlags(&compat_signal->event, cudaEventDisableTiming);
	return (ccv_nnc_stream_signal_t*)compat_signal;
}

void ccv_nnc_stream_compat_emit_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	cudaSetDevice(device);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	cudaEventRecord(compat_signal->event, stream_compat->stream);
}

void ccv_nnc_stream_compat_wait_signal(const ccv_nnc_stream_context_t* const stream, const ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	cudaSetDevice(device);
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	cudaStreamWaitEvent(stream_compat->stream, compat_signal->event, 0);
}

void ccv_nnc_deinit_stream_signal(ccv_nnc_stream_signal_t* const signal)
{
	ccv_nnc_stream_compat_signal_t* compat_signal = (ccv_nnc_stream_compat_signal_t*)signal;
	const int device = CCV_STREAM_GET_DEVICE_ID(compat_signal->super.type);
	cudaSetDevice(device);
	cudaEventDestroy(compat_signal->event);
}

int ccv_nnc_gpu_device_count(void)
{
	int count = 0;
	cudaGetDeviceCount(&count);
	return count;
}

ccv_nnc_stream_context_t* ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	assert(CCV_STREAM_GET_CONTEXT(((int*)stream_context)[0]) == CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)ccrealloc(stream_context, sizeof(ccv_nnc_stream_context_compat_t));
	stream_compat->cpu.workspace = 0;
	stream_compat->cpu.workspace_size = 0;
	stream_compat->gpu.workspace = 0;
	stream_compat->gpu.workspace_size = 0;
	int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
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

void* ccv_nnc_stream_compat_get_workspace(const ccv_nnc_stream_context_t* const stream_context, const size_t workspace_size, const int mem)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (mem == CCV_TENSOR_CPU_MEMORY)
	{
		if (!stream_compat)
			stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_CPU);
		if (stream_compat->cpu.workspace_size >= workspace_size)
			return stream_compat->cpu.workspace;
		stream_compat->cpu.workspace_size = workspace_size;
		if (stream_compat->cpu.workspace)
			ccfree(stream_compat->cpu.workspace);
		stream_compat->cpu.workspace = 0;
		const int success = ccmemalign(&stream_compat->cpu.workspace, 16, workspace_size);
		return success != 0 ? 0 : stream_compat->cpu.workspace;
	} else if (mem == CCV_TENSOR_GPU_MEMORY) {
		if (!stream_compat)
			stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
		if (stream_compat->gpu.workspace_size >= workspace_size)
			return stream_compat->gpu.workspace;
		stream_compat->gpu.workspace_size = workspace_size;
		const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
		cudaSetDevice(device);
		if (stream_compat->gpu.workspace)
			cudaFree(stream_compat->gpu.workspace);
		stream_compat->gpu.workspace = 0;
		cudaMalloc(&stream_compat->gpu.workspace, workspace_size);
		return stream_compat->gpu.workspace;
	}
	return 0;
}

static void _ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_compat_t* const stream_compat)
{
	if (stream_compat->cpu.workspace)
	{
		ccfree(stream_compat->cpu.workspace);
		stream_compat->cpu.workspace = 0;
		stream_compat->cpu.workspace_size = 0;
	}
	if (stream_compat->gpu.workspace)
	{
		cudaFree(stream_compat->gpu.workspace);
		stream_compat->gpu.workspace = 0;
		stream_compat->gpu.workspace_size = 0;
	}
}

void ccv_nnc_stream_compat_drain(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (stream_compat)
		_ccv_nnc_stream_compat_drain(stream_compat);
	else {
		if (gpu_stream_compat_init_per_thread)
			_ccv_nnc_stream_compat_drain(_ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU));
		if (cpu_stream_compat_init_per_thread)
			_ccv_nnc_stream_compat_drain(_ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_CPU));
	}
}

void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	cudaSetDevice(device);
	cudaStreamSynchronize(stream_compat->stream);
}

#if CUDA_VERSION >= 10000
static void _ccv_nnc_stream_compat_task_resume(void* userdata)
#else
static void _ccv_nnc_stream_compat_task_resume(cudaStream_t stream, cudaError_t status, void* userdata)
#endif
{
	ccv_nnc_stream_task_t* const task = (ccv_nnc_stream_task_t*)userdata;
	ccv_nnc_stream_scheduler_t* const scheduler = task->super;
	pthread_mutex_lock(&scheduler->mutex);
	ccv_nnc_stream_scheduler_add_task(scheduler, task);
	--scheduler->stream_wait_task_count;
	pthread_cond_signal(&scheduler->wait);
	pthread_mutex_unlock(&scheduler->mutex);
}

void ccv_nnc_stream_compat_task_synchronize(ccv_nnc_stream_task_t* const self, ccv_nnc_stream_context_t* const stream)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream;
	// If the stream is completed, no need to wait.
	if (cudaStreamQuery(stream_compat->stream) == cudaSuccess)
		return;
	ccv_nnc_stream_scheduler_t* const scheduler = self->super;
	pthread_mutex_lock(&scheduler->mutex);
	++scheduler->stream_wait_task_count;
#if CUDA_VERSION >= 10000
	cudaLaunchHostFunc(stream_compat->stream, _ccv_nnc_stream_compat_task_resume, self);
#else
	cudaStreamAddCallback(stream_compat->stream, _ccv_nnc_stream_compat_task_resume, self, 0);
#endif
	pthread_mutex_unlock(&scheduler->mutex);
	swapcontext(&scheduler->callee, &scheduler->caller);
}

void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
	cudaSetDevice(device);
	if (stream_compat->cpu.workspace)
		ccfree(stream_compat->cpu.workspace);
	if (stream_compat->gpu.workspace)
		cudaFree(stream_compat->gpu.workspace);
	cudaStreamDestroy(stream_compat->stream);
	if (stream_compat->cublas)
		cublasDestroy(stream_compat->cublas);
	if (stream_compat->ones.data)
		cudaFree(stream_compat->ones.data);
#ifdef HAVE_CUDNN
	if (stream_compat->cudnn)
		cudnnDestroy(stream_compat->cudnn);
	if (stream_compat->rngs)
		cudaFree(stream_compat->rngs);
#endif
}

int ccv_nnc_stream_context_get_device(const ccv_nnc_stream_context_t* const stream_context)
{
	if (!stream_context)
	{
		int device = 0;
		cudaGetDevice(&device);
		return device;
	}
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	return CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
}

cudaStream_t ccv_nnc_stream_context_get_stream(const ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_nnc_stream_context_compat_t* stream_compat = (const ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	return stream_compat->stream;
}

cublasHandle_t ccv_nnc_stream_context_get_cublas(const ccv_nnc_stream_context_t* const stream_context)
{
	ccv_nnc_stream_context_compat_t* stream_compat = (ccv_nnc_stream_context_compat_t*)stream_context;
	if (!stream_compat)
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	if (!stream_compat->cublas)
	{
		const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
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
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	if (!stream_compat->ones.data || n > stream_compat->ones.n)
	{
		const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
		cudaSetDevice(device);
		cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
		if (stream_compat->ones.data)
			cudaFree(stream_compat->ones.data);
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
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
	if (!stream_compat->cudnn)
	{
		const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
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
		stream_compat = _ccv_nnc_default_stream_compat(CCV_STREAM_CONTEXT_GPU);
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
		const int device = CCV_STREAM_GET_DEVICE_ID(stream_compat->super.type);
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
	for (i = axis_count; i < CCV_NNC_MAX_DIM + 2; i++)
		dim[i] = stride[i] = 1;
	dim[axis_count - 1] = tensor->info.dim[axis_count - 1];
	stride[axis_count - 1] = 1;
	for (i = axis_count - 2; i >= 0; i--)
	{
		dim[i] = tensor->info.dim[i];
		stride[i] = stride[i + 1] * inc[i + 1];
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
