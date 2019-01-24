extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

static inline __device__ __half log(const half v)
{
	return hlog(v);
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_categorical_crossentropy_forw_kernel(const int batch_size, const int count, const NUM1* const label, const NUM2* const a, NUM2* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		c[i] = -log(a[i * count + idx]);
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_one_hot_forw_kernel(const int batch_size, const int count, const NUM* const label, const NUM* const a, NUM* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		NUM p = label[i * count] * log(a[i * count]);
		for (int j = 1; j < count; j++)
			p += label[i * count + j] * log(a[i * count + j]);
		c[i] = -p;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_forw_kernel(const int batch_size, const int count, const int* const label, const NUM* const a, NUM* const c)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		c[i] = -log(a[i * count + label[i]]);
	}
}

static int _ccv_nnc_categorical_crossentropy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const ccv_nnc_tensor_t* b = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* c = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(c));
	const int axis_count = ccv_nnc_tensor_nd(a->info.dim);
	const int batch_size = axis_count < 2 ? 1 : a->info.dim[0];
	const int count = ccv_nnc_tensor_count(a->info) / batch_size;
	int i;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == c->info.datatype);
	if (b->info.datatype == CCV_32F || b->info.datatype == CCV_16F)
	{
		// If has more than 1 axis, then the range is the channel count. Otherwise, if our batch size is 1, then the range is
		// the channel count. Otherwise, the range is 1 (and the only axis is the batch size).
		const int range = ccv_nnc_tensor_nd(b->info.dim) > 1 ? ccv_nnc_tensor_get_c(b->info) : (batch_size == 1 ? b->info.dim[0] : 1);
		if (range == 1)
		{
			for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
				{ assert(b->info.dim[i] == c->info.dim[i]); }
			if (b->info.datatype == CCV_32F)
			{
				if (a->info.datatype == CCV_16F)
					_ccv_nnc_categorical_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, (__half*)a->data.f16, (__half*)c->data.f16);
				else
					_ccv_nnc_categorical_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, a->data.f32, c->data.f32);
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(a->info.datatype == CCV_16F);
				_ccv_nnc_categorical_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)c->data.f16);
			}
		} else {
			assert(range == count);
			assert(a->info.datatype == b->info.datatype);
			if (a->info.datatype == CCV_16F)
				_ccv_nnc_categorical_crossentropy_one_hot_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)c->data.f16);
			else
				_ccv_nnc_categorical_crossentropy_one_hot_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, a->data.f32, c->data.f32);
		}
	} else if (b->info.datatype == CCV_32S) {
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
			{ assert(b->info.dim[i] == c->info.dim[i]); }
		if (a->info.datatype == CCV_16F)
			_ccv_nnc_categorical_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, (__half*)a->data.f16, (__half*)c->data.f16);
		else
			_ccv_nnc_categorical_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, a->data.f32, c->data.f32);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM>
__global__ void _ccv_nnc_set_zero_kernel(const int n, NUM* const a)
{
	CUDA_1D_KERNEL_LOOP(i, n) {
		a[i] = 0;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_categorical_crossentropy_back_kernel(const int batch_size, const int count, const NUM2* const g, const NUM1* const label, const NUM2* const a, NUM2* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		h[i * count + idx] = -g[i] / a[i * count + idx];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_one_hot_back_kernel(const int batch_size_count, const int count, const NUM* const g, const NUM* const label, const NUM* const a, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size_count) {
		const int idx = i / count;
		h[i] = -g[idx] * label[i] / a[i];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_back_kernel(const int batch_size, const int count, const NUM* const g, const int* const label, const NUM* const a, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = label[i];
		h[i * count + idx] = -g[i] / a[i * count + idx];
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_categorical_crossentropy_back_kernel(const int batch_size, const int count, const NUM1* const label, const NUM2* const a, NUM2* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = (int)((float)label[i] + 0.5);
		h[i * count + idx] = (NUM2)-1. / a[i * count + idx];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_one_hot_back_kernel(const int batch_size_count, const int count, const NUM* const label, const NUM* const a, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size_count) {
		h[i] = -label[i] / a[i];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_categorical_crossentropy_back_kernel(const int batch_size, const int count, const int* const label, const NUM* const a, NUM* const h)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const int idx = label[i];
		h[i * count + idx] = (NUM)-1. / a[i * count + idx];
	}
}

static int _ccv_nnc_categorical_crossentropy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_t* g = inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_t* a = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const ccv_nnc_tensor_t* b = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(b));
	ccv_nnc_tensor_t* h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	const int axis_count = ccv_nnc_tensor_nd(a->info.dim);
	const int batch_size = axis_count < 2 ? 1 : a->info.dim[0];
	const int bcount = ccv_nnc_tensor_count(a->info);
	const int count = bcount / batch_size;
	int i;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == h->info.datatype);
	const int datatype = a->info.datatype;
	if (datatype == CCV_16F)
		_ccv_nnc_set_zero_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, (__half *)h->data.f16);
	else
		_ccv_nnc_set_zero_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, h->data.f32);
	if (g)
	{
		assert(g->info.datatype == datatype);
		if (b->info.datatype == CCV_32F || b->info.datatype == CCV_16F)
		{
			// If has more than 1 axis, then the range is the channel count. Otherwise, if our batch size is 1, then the range is
			// the channel count. Otherwise, the range is 1 (and the only axis is the batch size).
			const int range = ccv_nnc_tensor_nd(b->info.dim) > 1 ? ccv_nnc_tensor_get_c(b->info) : (batch_size == 1 ? b->info.dim[0] : 1);
			if (range == 1)
			{
				for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
					{ assert(a->info.dim[i] == h->info.dim[i]); }
				if (b->info.datatype == CCV_32F)
				{
					if (datatype == CCV_16F)
						_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, b->data.f32, (__half*)a->data.f16, (__half*)h->data.f16);
					else
						_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, b->data.f32, a->data.f32, h->data.f32);
				} else {
					assert(b->info.datatype == CCV_16F);
					assert(datatype == CCV_16F);
					_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)h->data.f16);
				}
			} else {
				assert(range == count);
				assert(b->info.datatype == datatype);
				if (datatype == CCV_16F)
					_ccv_nnc_categorical_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, (__half*)g->data.f16, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)h->data.f16);
				else
					_ccv_nnc_categorical_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, g->data.f32, b->data.f32, a->data.f32, h->data.f32);
			}
		} else if (b->info.datatype == CCV_32S) {
			for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
				{ assert(a->info.dim[i] == h->info.dim[i]); }
			if (datatype == CCV_16F)
				_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, b->data.i32, (__half*)a->data.f16, (__half*)h->data.f16);
			else
				_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, b->data.i32, a->data.f32, h->data.f32);
		}
	} else {
		if (b->info.datatype == CCV_32F || b->info.datatype == CCV_16F)
		{
			// If has more than 1 axis, then the range is the channel count. Otherwise, if our batch size is 1, then the range is
			// the channel count. Otherwise, the range is 1 (and the only axis is the batch size).
			const int range = ccv_nnc_tensor_nd(b->info.dim) > 1 ? ccv_nnc_tensor_get_c(b->info) : (batch_size == 1 ? b->info.dim[0] : 1);
			if (range == 1)
			{
				for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
					{ assert(a->info.dim[i] == h->info.dim[i]); }
				if (b->info.datatype == CCV_32F)
				{
					if (datatype == CCV_16F)
						_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, (__half*)a->data.f16, (__half*)h->data.f16);
					else
						_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, a->data.f32, h->data.f32);
				} else {
					assert(b->info.datatype == CCV_16F);
					assert(datatype == CCV_16F);
					_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)h->data.f16);
				}
			} else {
				assert(range == count);
				assert(b->info.datatype == datatype);
				if (datatype == CCV_16F)
					_ccv_nnc_categorical_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, (__half*)b->data.f16, (__half*)a->data.f16, (__half*)h->data.f16);
				else
					_ccv_nnc_categorical_crossentropy_one_hot_back_kernel<<<CUDA_GET_BLOCKS(bcount), CUDA_NUM_THREADS, 0, stream>>>(bcount, count, b->data.f32, a->data.f32, h->data.f32);
			}
		} else if (b->info.datatype == CCV_32S) {
			for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
				{ assert(a->info.dim[i] == h->info.dim[i]); }
			if (datatype == CCV_16F)
				_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, (__half*)a->data.f16, (__half*)h->data.f16);
			else
				_ccv_nnc_categorical_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.i32, a->data.f32, h->data.f32);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CATEGORICAL_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_categorical_crossentropy_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CATEGORICAL_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_categorical_crossentropy_back;
}
