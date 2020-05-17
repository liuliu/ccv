extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_smooth_l1_forw_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM1* const c, const int cstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		float p = 0;
		for (int j = 0; j < count; j++)
			p += fabs((float)bp[j] - (float)ap[j]);
		if (p < 1)
		{
			p = 0;
			for (int j = 0; j < count; j++)
				p += ((float)bp[j] - (float)ap[j]) * ((float)bp[j] - (float)ap[j]);
			p *= 0.5;
		} else
			p -= 0.5;
		c[i * cstep] = (NUM1)p;
	}
}

static int _ccv_nnc_smooth_l1_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)inputs[1];
	assert(output_size == 1);
	ccv_nnc_tensor_view_t* c = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int cinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(c, cinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cinc[CCV_NNC_MAX_DIM + 1];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == c->info.datatype);
	if (b->info.datatype == CCV_32F)
	{
		if (a->info.datatype == CCV_16F)
			_ccv_nnc_smooth_l1_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep);
		else
			_ccv_nnc_smooth_l1_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep);
	} else {
		assert(b->info.datatype == CCV_16F);
		assert(a->info.datatype == CCV_16F);
		_ccv_nnc_smooth_l1_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_smooth_l1_back_kernel(const int batch_size, const int count, const NUM2* const g, const int gstep, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, const NUM2* const c, const int cstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		const float gp = (float)g[i * gstep];
		const float cp = (float)c[i * cstep];
		if (cp < 0.5)
			for (int j = 0; j < count; j++)
			{
				const float av = ap[j];
				const float bv = bp[j];
				hp[j] = (NUM2)(gp * (av - bv));
			}
		else
			for (int j = 0; j < count; j++)
			{
				const float av = ap[j];
				const float bv = bp[j];
				hp[j] = (NUM2)(((av - bv) > 0 ? 0.5 : -0.5) * gp);
			}
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_smooth_l1_back_kernel(const int batch_size, const int count, const NUM2* const a, const int astep, const NUM1* const b, const int bstep, const NUM2* const c, const int cstep, NUM2* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM1* const bp = b + i * bstep;
		NUM2* const hp = h + i * hstep;
		const float cp = (float)c[i * cstep];
		if (cp < 0.5)
			for (int j = 0; j < count; j++)
			{
				const float av = ap[j];
				const float bv = bp[j];
				hp[j] = (NUM2)(av - bv);
			}
		else
			for (int j = 0; j < count; j++)
			{
				const float av = ap[j];
				const float bv = bp[j];
				hp[j] = (NUM2)((av - bv) > 0 ? 0.5 : -0.5);
			}
	}
}

static int _ccv_nnc_smooth_l1_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	const ccv_nnc_tensor_view_t* const c = (ccv_nnc_tensor_view_t*)inputs[3];
	ccv_nnc_tensor_view_t* const h = (ccv_nnc_tensor_view_t*)outputs[0];
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int cinc[CCV_NNC_MAX_DIM_ALLOC];
	int hinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	assert(ccv_nnc_tensor_view_check_dim(h, dim));
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	ccv_nnc_tensor_view_get_inc(c, cinc);
	ccv_nnc_tensor_view_get_inc(h, hinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	assert(ccv_nnc_tensor_count(c->info) == batch_size);
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int hstep = hinc[CCV_NNC_MAX_DIM + 1];
	const int cstep = ccv_nnc_tensor_nd(c->info.dim) == 1 ? 1 : cinc[CCV_NNC_MAX_DIM + 1];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == h->info.datatype);
	assert(a->info.datatype == c->info.datatype);
	const int datatype = a->info.datatype;
	if (g)
	{
		int ginc[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_inc(g, ginc);
		assert(ccv_nnc_tensor_count(g->info) == batch_size);
		const int gstep = ccv_nnc_tensor_nd(g->info.dim) == 1 ? 1 : ginc[CCV_NNC_MAX_DIM + 1];
		assert(g->info.datatype == datatype);
		if (b->info.datatype == CCV_32F)
		{
			if (datatype == CCV_16F)
				_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep, (__half*)h->data.f16, hstep);
			else
				_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep, h->data.f32, hstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(datatype == CCV_16F);
			_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep, (__half*)h->data.f16, hstep);
		}
	} else {
		if (b->info.datatype == CCV_32F)
		{
			if (datatype == CCV_16F)
				_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep, (__half*)h->data.f16, hstep);
			else
				_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep, h->data.f32, hstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(datatype == CCV_16F);
			_ccv_nnc_smooth_l1_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep, (__half*)h->data.f16, hstep);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SMOOTH_L1_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_smooth_l1_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SMOOTH_L1_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_smooth_l1_back;
}
