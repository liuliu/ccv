extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_reciprocal_kernel(const size_t count, const NUM1* const a, NUM2* const b)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM2)((NUM1)1. / a[i]);
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_ewdiv_kernel(const size_t count, const NUM1* const a, const NUM2* const b, NUM3* const c)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		c[i] = (NUM3)(a[i] / (NUM1)b[i]);
	}
}

static int _ccv_nnc_ewdiv_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* const a = inputs[0];
	const ccv_nnc_tensor_t* const b = inputs[1];
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(c));
	const size_t count = ccv_nnc_tensor_count(b->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
		{ assert(b->info.dim[i] == c->info.dim[i]); }
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (a)
	{
		assert(!CCV_IS_TENSOR_VIEW(a));
		assert(a->info.datatype == b->info.datatype);
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
			{ assert(a->info.dim[i] == b->info.dim[i]); }
		if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
		{
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, c->data.f32);
		} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, (__half*)c->data.f16);
		} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
		} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
		}
	} else {
		if (b->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
		{
			_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, c->data.f32);
		} else if (b->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
			_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, (__half*)c->data.f16);
		} else if (b->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
			_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, c->data.f32);
		} else if (b->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
			_ccv_nnc_reciprocal_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)c->data.f16);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_ewnegmuldiv_kernel(const size_t count, const NUM1* const g, const NUM2* const b, const NUM1* const c, NUM2* const h)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)(-g[i] * c[i] / (NUM1)b[i]);
	}
}

static int _ccv_nnc_ewdiv_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	const ccv_nnc_tensor_t* const g = inputs[0]; // gradient
	assert(!CCV_IS_TENSOR_VIEW(g));
	const size_t count = ccv_nnc_tensor_count(g->info);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (outputs[0])
	{
		const ccv_nnc_tensor_t* const b = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(b));
		ccv_nnc_tensor_t* const h = outputs[0];
		assert(!CCV_IS_TENSOR_VIEW(h));
		assert(ccv_nnc_tensor_count(h->info) == count);
		assert(b->info.datatype == h->info.datatype);
		if (b->info.datatype == CCV_32F && g->info.datatype == CCV_32F)
		{
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, b->data.f32, h->data.f32);
		} else if (b->info.datatype == CCV_32F && g->info.datatype == CCV_16F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, b->data.f32, h->data.f32);
		} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_32F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, (__half*)b->data.f16, (__half*)h->data.f16);
		} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_16F) {
			_ccv_nnc_ewdiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, (__half*)b->data.f16, (__half*)h->data.f16);
		}
	}
	if (output_size >= 2 && outputs[1])
	{
		const ccv_nnc_tensor_t* const b = inputs[2];
		assert(!CCV_IS_TENSOR_VIEW(b));
		const ccv_nnc_tensor_t* const c = inputs[3];
		assert(!CCV_IS_TENSOR_VIEW(c));
		ccv_nnc_tensor_t* const h = outputs[1];
		assert(!CCV_IS_TENSOR_VIEW(h));
		assert(ccv_nnc_tensor_count(h->info) == count);
		assert(b->info.datatype == h->info.datatype);
		assert(c->info.datatype == g->info.datatype);
		if (b->info.datatype == CCV_32F && g->info.datatype == CCV_32F)
		{
			_ccv_nnc_ewnegmuldiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, b->data.f32, c->data.f32, h->data.f32);
		} else if (b->info.datatype == CCV_32F && g->info.datatype == CCV_16F) {
			_ccv_nnc_ewnegmuldiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, b->data.f32, (__half*)c->data.f16, h->data.f32);
		} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_32F) {
			_ccv_nnc_ewnegmuldiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, g->data.f32, (__half*)b->data.f16, c->data.f32, (__half*)h->data.f16);
		} else if (b->info.datatype == CCV_16F && g->info.datatype == CCV_16F) {
			_ccv_nnc_ewnegmuldiv_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)g->data.f16, (__half*)b->data.f16, (__half*)c->data.f16, (__half*)h->data.f16);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewdiv_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_EWDIV_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_ewdiv_back;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_clamp_kernel(const size_t count, const NUM1* const a, NUM2* const b, const float minv, const float maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM2)min(max(a[i], minv), maxv);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_min_kernel(const size_t count, const NUM1* const a, NUM2* const b, const float maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM2)min(a[i], maxv);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_max_kernel(const size_t count, const NUM1* const a, NUM2* const b, const float minv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		b[i] = (NUM2)max(a[i], minv);
	}
}

static int _ccv_nnc_clamp_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const size_t count = ccv_nnc_tensor_count(a->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
		{ assert(a->info.dim[i] == b->info.dim[i]); }
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float minv = cmd.info.clamp.min;
	const float maxv = cmd.info.clamp.max;
	assert(!isnan(minv) || !isnan(maxv));
	if (isnan(minv))
	{
		if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
		{
			_ccv_nnc_min_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, maxv);
		} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
			_ccv_nnc_min_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, (__half*)b->data.f16, maxv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
			_ccv_nnc_min_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, b->data.f32, maxv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
			_ccv_nnc_min_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, maxv);
		}
	} else if (isnan(maxv)) {
		if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
		{
			_ccv_nnc_max_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, minv);
		} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
			_ccv_nnc_max_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, (__half*)b->data.f16, minv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
			_ccv_nnc_max_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, b->data.f32, minv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
			_ccv_nnc_max_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, minv);
		}
	} else {
		if (a->info.datatype == CCV_32F && b->info.datatype == CCV_32F)
		{
			_ccv_nnc_clamp_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, minv, maxv);
		} else if (a->info.datatype == CCV_32F && b->info.datatype == CCV_16F) {
			_ccv_nnc_clamp_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, (__half*)b->data.f16, minv, maxv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_32F) {
			_ccv_nnc_clamp_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, b->data.f32, minv, maxv);
		} else if (a->info.datatype == CCV_16F && b->info.datatype == CCV_16F) {
			_ccv_nnc_clamp_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, minv, maxv);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_clamp_kernel_back(const size_t count, const NUM1* const b, const NUM1* const g, NUM2* const h, const NUM1 minv, const NUM1 maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] <= minv || b[i] >= maxv) ? (NUM1)0.0 : g[i]);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_clamp_kernel_back(const size_t count, const NUM1* const b, NUM2* const h, const NUM1 minv, const NUM1 maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] <= minv || b[i] >= maxv) ? 0.0 : 1.0);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_min_kernel_back(const size_t count, const NUM1* const b, const NUM1* const g, NUM2* const h, const NUM1 maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] >= maxv) ? (NUM1)0.0 : g[i]);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_min_kernel_back(const size_t count, const NUM1* const b, NUM2* const h, const NUM1 maxv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] >= maxv) ? 0.0 : 1.0);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_max_kernel_back(const size_t count, const NUM1* const b, const NUM1* const g, NUM2* const h, const NUM1 minv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] <= minv) ? (NUM1)0.0 : g[i]);
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_max_kernel_back(const size_t count, const NUM1* const b, NUM2* const h, const NUM1 minv)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		h[i] = (NUM2)((b[i] <= minv) ? 0.0 : 1.0);
	}
}

static int _ccv_nnc_clamp_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 3);
	const ccv_nnc_tensor_t* const g = inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_t* const b = inputs[2];
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const h = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(h));
	const size_t count = ccv_nnc_tensor_count(b->info);
	int i;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
		{ assert(b->info.dim[i] == h->info.dim[i]); }
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const float minv = cmd.info.clamp.min;
	const float maxv = cmd.info.clamp.max;
	assert(!isnan(minv) || !isnan(maxv));
	if (g)
	{
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
			{ assert(b->info.dim[i] == g->info.dim[i]); }
		assert(g->info.datatype == b->info.datatype);
		if (isnan(minv))
		{
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, h->data.f32, maxv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, (__half*)h->data.f16, maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, h->data.f32, (__half)maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, (__half*)h->data.f16, (__half)maxv);
			}
		} else if (isnan(maxv)) {
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, h->data.f32, minv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, (__half*)h->data.f16, minv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, h->data.f32, (__half)minv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, (__half*)h->data.f16, (__half)minv);
			}
		} else {
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, h->data.f32, minv, maxv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, g->data.f32, (__half*)h->data.f16, minv, maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, h->data.f32, (__half)minv, (__half)maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)g->data.f16, (__half*)h->data.f16, (__half)minv, (__half)maxv);
			}
		}
	} else {
		if (isnan(minv))
		{
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, h->data.f32, maxv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, (__half*)h->data.f16, maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, h->data.f32, (__half)maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_min_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)h->data.f16, (__half)maxv);
			}
		} else if (isnan(maxv)) {
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, h->data.f32, minv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, (__half*)h->data.f16, minv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, h->data.f32, (__half)minv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_max_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)h->data.f16, (__half)minv);
			}
		} else {
			if (b->info.datatype == CCV_32F && h->info.datatype == CCV_32F)
			{
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, h->data.f32, minv, maxv);
			} else if (b->info.datatype == CCV_32F && h->info.datatype == CCV_16F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, b->data.f32, (__half*)h->data.f16, minv, maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_32F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, h->data.f32, (__half)minv, (__half)maxv);
			} else if (b->info.datatype == CCV_16F && h->info.datatype == CCV_16F) {
				_ccv_nnc_clamp_kernel_back<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)b->data.f16, (__half*)h->data.f16, (__half)minv, (__half)maxv);
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_clamp_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CLAMP_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_clamp_back;
}
