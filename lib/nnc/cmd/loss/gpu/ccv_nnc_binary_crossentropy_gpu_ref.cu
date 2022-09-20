extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_forw_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM1* const c, const int cstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		float p = 0;
		for (int j = 0; j < count; j++)
			p += ((float)bp[j] - 1) * log(1 - (float)ap[j]) - (float)bp[j] * log((float)ap[j]);
		c[i * cstep] = (NUM1)p;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_forw_kernel(const int batch_size, const int count, const float pos_weight, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM1* const c, const int cstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		float p1 = 0, p2 = 0;
		for (int j = 0; j < count; j++)
			p1 += ((float)bp[j] - 1) * log(1 - (float)ap[j]), p2 += (float)bp[j] * log((float)ap[j]);
		c[i * cstep] = (NUM1)(p1 - p2 * pos_weight);
	}
}

static int _ccv_nnc_binary_crossentropy_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int cstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(c, cstride);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = astride[CCV_NNC_MAX_DIM];
	const int bstep = bstride[CCV_NNC_MAX_DIM];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cstride[CCV_NNC_MAX_DIM];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == c->info.datatype);
	const float pos_weight = cmd.info.binary_crossentropy.pos_weight;
	if (pos_weight == 1)
	{
		if (b->info.datatype == CCV_32F)
		{
			if (a->info.datatype == CCV_16F)
				_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep);
			else
				_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(a->info.datatype == CCV_16F);
			_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep);
		}
	} else {
		if (b->info.datatype == CCV_32F)
		{
			if (a->info.datatype == CCV_16F)
				_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep);
			else
				_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(a->info.datatype == CCV_16F);
			_ccv_nnc_binary_crossentropy_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_back_kernel(const int batch_size, const int count, const NUM2* const g, const int gstep, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		const float gp = (float)g[i * gstep];
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM2)(gp * (av - bv) / max((1 - av) * av, 1e-12));
		}
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_back_kernel(const int batch_size, const int count, const float pos_weight, const NUM2* const g, const int gstep, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		const float gp = (float)g[i * gstep];
		const float pos_weight_1 = pos_weight - 1;
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM2)(gp * (av * bv * pos_weight_1 + av - pos_weight * bv) / max((1 - av) * av, 1e-12));
		}
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_back_kernel(const int batch_size, const int count, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM2)((av - bv) / max((1 - av) * av, 1e-12));
		}
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_binary_crossentropy_back_kernel(const int batch_size, const int count, const float pos_weight, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		const float pos_weight_1 = pos_weight - 1;
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM2)((av * bv * pos_weight_1 + av - pos_weight * bv) / max((1 - av) * av, 1e-12));
		}
	}
}

static int _ccv_nnc_binary_crossentropy_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int astride[CCV_NNC_MAX_DIM_ALLOC];
	int bstride[CCV_NNC_MAX_DIM_ALLOC];
	int hstride[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(h, dim));
	ccv_nnc_tensor_view_get_stride(a, astride);
	ccv_nnc_tensor_view_get_stride(b, bstride);
	ccv_nnc_tensor_view_get_stride(h, hstride);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = astride[CCV_NNC_MAX_DIM];
	const int bstep = bstride[CCV_NNC_MAX_DIM];
	const int hstep = hstride[CCV_NNC_MAX_DIM];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == h->info.datatype);
	const int datatype = a->info.datatype;
	const float pos_weight = cmd.info.binary_crossentropy.pos_weight;
	if (pos_weight == 1)
	{
		if (g)
		{
			int gstride[CCV_NNC_MAX_DIM_ALLOC];
			ccv_nnc_tensor_view_get_stride(g, gstride);
			assert(ccv_nnc_tensor_count(g->info) == batch_size);
			const int gstep = ccv_nnc_tensor_nd(g->info.dim) == 1 ? 1 : gstride[CCV_NNC_MAX_DIM];
			assert(g->info.datatype == datatype);
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)h->data.f16, hstep);
				else
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, a->data.f32, astep, b->data.f32, bstep, h->data.f32, hstep);
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)h->data.f16, hstep);
			}
		} else {
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)h->data.f16, hstep);
				else
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, h->data.f32, hstep);
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)h->data.f16, hstep);
			}
		}
	} else {
		if (g)
		{
			int gstride[CCV_NNC_MAX_DIM_ALLOC];
			ccv_nnc_tensor_view_get_stride(g, gstride);
			assert(ccv_nnc_tensor_count(g->info) == batch_size);
			const int gstep = ccv_nnc_tensor_nd(g->info.dim) == 1 ? 1 : gstride[CCV_NNC_MAX_DIM];
			assert(g->info.datatype == datatype);
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)h->data.f16, hstep);
				else
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, g->data.f32, gstep, a->data.f32, astep, b->data.f32, bstep, h->data.f32, hstep);
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)h->data.f16, hstep);
			}
		} else {
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)h->data.f16, hstep);
				else
					_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, a->data.f32, astep, b->data.f32, bstep, h->data.f32, hstep);
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				_ccv_nnc_binary_crossentropy_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, pos_weight, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)h->data.f16, hstep);
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_BINARY_CROSSENTROPY_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_binary_crossentropy_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_BINARY_CROSSENTROPY_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_binary_crossentropy_back;
}
