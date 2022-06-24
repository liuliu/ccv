extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_max_kernel_forw(const size_t count, const NUM1* const a, const NUM1* const b, NUM2* const c)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		if (a[i] > b[i])
			c[i] = (NUM2)a[i];
		else
			c[i] = (NUM2)b[i];
	}
}

static int _ccv_nnc_max_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const ccv_nnc_tensor_t* const b = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	const size_t count = ccv_nnc_tensor_count(b->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
		{ assert(b->info.dim[i] == c->info.dim[i]); }
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == b->info.datatype);
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
	{
		_ccv_nnc_max_kernel_forw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, c->data.f32);
	} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
		_ccv_nnc_max_kernel_forw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, (__half*)c->data.f16);
	} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
		_ccv_nnc_max_kernel_forw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
	} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
		_ccv_nnc_max_kernel_forw<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_max_kernel_back(const size_t count, const NUM1* const g, const NUM2* const a, const NUM2* const b, NUM3* const ha, NUM3* const hb)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		if (a[i] > b[i]) {
			ha[i] = (NUM3)g[i];
			hb[i] = 0;
		} else if (a[i] < b[i]) {
			hb[i] = (NUM3)g[i];
			ha[i] = 0;
		} else
			ha[i] = hb[i] = (NUM3)g[i];
	}
}

template<typename NUM2, typename NUM3>
__global__ void _ccv_nnc_max_kernel_back(const size_t count, const NUM2* const a, const NUM2* const b, NUM3* const ha, NUM3* const hb)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		if (a[i] > b[i]) {
			ha[i] = 1;
			hb[i] = 0;
		} else if (a[i] < b[i]) {
			ha[i] = 0;
			hb[i] = 1;
		} else
			ha[i] = hb[i] = 1;
	}
}

static int _ccv_nnc_max_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(input_size >= 3);
	assert(output_size >= 2);
	const ccv_nnc_tensor_t* const a = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const ccv_nnc_tensor_t* const b = inputs[2];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	const size_t count = ccv_nnc_tensor_count(a->info);
	assert(ccv_nnc_tensor_count(b->info) == count);
	ccv_nnc_tensor_t* const ha = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(ha));
	assert(ccv_nnc_tensor_count(ha->info) == count);
	ccv_nnc_tensor_t* const hb = outputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(hb));
	assert(ccv_nnc_tensor_count(hb->info) == count);
	if (inputs[0])
	{
		const ccv_nnc_tensor_t* const g = inputs[0]; // gradient
		assert(CCV_IS_TENSOR_CONTIGUOUS(g));
		assert(ccv_nnc_tensor_count(g->info) == count);
		assert(a->info.datatype == b->info.datatype);
		assert(ha->info.datatype == hb->info.datatype);
		if (g->info.datatype == CCV_32F)
		{
			if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_32F)
			{
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, a->data.f32, b->data.f32, ha->data.f32, hb->data.f32);
			} else if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, a->data.f32, b->data.f32, (__half*)ha->data.f16, (__half*)hb->data.f16);
			} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_32F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, (__half*)a->data.f16, (__half*)b->data.f16, ha->data.f32, hb->data.f32);
			} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)ha->data.f16, (__half*)hb->data.f16);
			}
		} else {
			if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_32F)
			{
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, a->data.f32, b->data.f32, ha->data.f32, hb->data.f32);
			} else if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, a->data.f32, b->data.f32, (__half*)ha->data.f16, (__half*)hb->data.f16);
			} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_32F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)b->data.f16, ha->data.f32, hb->data.f32);
			} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)ha->data.f16, (__half*)hb->data.f16);
			}
		}
	} else {
		if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_32F)
		{
			_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, ha->data.f32, hb->data.f32);
		} else if (a->info.datatype == CCV_32F && ha->info.datatype == CCV_16F) {
			_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, (__half*)ha->data.f16, (__half*)hb->data.f16);
		} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_32F) {
			_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, ha->data.f32, hb->data.f32);
		} else if (a->info.datatype == CCV_16F && ha->info.datatype == CCV_16F) {
			_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)ha->data.f16, (__half*)hb->data.f16);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MAX_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_max_back;
}
