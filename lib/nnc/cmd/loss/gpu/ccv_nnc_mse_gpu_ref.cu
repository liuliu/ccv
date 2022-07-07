extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_mse_mean_forw_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM1* const c, const int cstep)
{
	const float inv_mean = 1.0 / (float)count;
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		float p = 0;
		for (int j = 0; j < count; j++)
			p += ((float)bp[j] - (float)ap[j]) * ((float)bp[j] - (float)ap[j]);
		p *= inv_mean;
		c[i * cstep] = (NUM1)p;
	}
}

template<typename NUM1, typename NUM2>
__global__ void _ccv_nnc_mse_sum_forw_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM1* const c, const int cstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		float p = 0;
		for (int j = 0; j < count; j++)
			p += ((float)bp[j] - (float)ap[j]) * ((float)bp[j] - (float)ap[j]);
		c[i * cstep] = (NUM1)p;
	}
}

static int _ccv_nnc_mse_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
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
	if (cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN)
	{
		if (b->info.datatype == CCV_32F)
		{
			if (a->info.datatype == CCV_16F)
				_ccv_nnc_mse_mean_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep);
			else
				_ccv_nnc_mse_mean_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(a->info.datatype == CCV_16F);
			_ccv_nnc_mse_mean_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep);
		}
	} else {
		assert(cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_SUM);
		if (b->info.datatype == CCV_32F)
		{
			if (a->info.datatype == CCV_16F)
				_ccv_nnc_mse_sum_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)c->data.f16, cstep);
			else
				_ccv_nnc_mse_sum_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, c->data.f32, cstep);
		} else {
			assert(b->info.datatype == CCV_16F);
			assert(a->info.datatype == CCV_16F);
			_ccv_nnc_mse_sum_forw_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)c->data.f16, cstep);
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

template<typename NUM1, typename NUM2, typename NUM3, typename NUM4>
__global__ void _ccv_nnc_mse_mean_back_kernel(const int batch_size, const int count, const NUM1* const g, const int gstep, const NUM2* const a, const int astep, const NUM3* const b, const int bstep, NUM4* const h, const int hstep)
{
	const float inv_mean_2 = 2.0 / (float)count;
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM3* const bp = b + i * bstep;
		NUM4* const hp = h + i * hstep;
		const float gp = inv_mean_2 * (float)g[i * gstep];
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM4)(gp * (av - bv));
		}
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_mse_mean_back_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM3* const h, const int hstep)
{
	const float inv_mean_2 = 2.0 / (float)count;
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		NUM3* const hp = h + i * hstep;
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM3)(inv_mean_2 * (av - bv));
		}
	}
}

template<typename NUM1, typename NUM2, typename NUM3, typename NUM4>
__global__ void _ccv_nnc_mse_sum_back_kernel(const int batch_size, const int count, const NUM1* const g, const int gstep, const NUM2* const a, const int astep, const NUM3* const b, const int bstep, NUM4* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM2* const ap = a + i * astep;
		const NUM3* const bp = b + i * bstep;
		NUM4* const hp = h + i * hstep;
		const float gp = 2.0 * (float)g[i * gstep];
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM4)(gp * (av - bv));
		}
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_mse_sum_back_kernel(const int batch_size, const int count, const NUM1* const a, const int astep, const NUM2* const b, const int bstep, NUM3* const h, const int hstep)
{
	CUDA_1D_KERNEL_LOOP(i, batch_size) {
		const NUM1* const ap = a + i * astep;
		const NUM2* const bp = b + i * bstep;
		NUM3* const hp = h + i * hstep;
		for (int j = 0; j < count; j++)
		{
			const float av = ap[j];
			const float bv = bp[j];
			hp[j] = (NUM3)(2.0 * (av - bv));
		}
	}
}

static int _ccv_nnc_mse_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size >= 3);
	assert(output_size >= 1);
	const ccv_nnc_tensor_view_t* const g = (ccv_nnc_tensor_view_t*)inputs[0];
	assert(!g || !CCV_IS_TENSOR_VIEW(g));
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[1];
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)inputs[2];
	ccv_nnc_tensor_view_t* const ha = (ccv_nnc_tensor_view_t*)outputs[0];
	ccv_nnc_tensor_view_t* const hb = output_size >= 2 ? (ccv_nnc_tensor_view_t*)outputs[1] : 0;
	int dim[CCV_NNC_MAX_DIM_ALLOC];
	int ainc[CCV_NNC_MAX_DIM_ALLOC];
	int binc[CCV_NNC_MAX_DIM_ALLOC];
	int hainc[CCV_NNC_MAX_DIM_ALLOC];
	int hbinc[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_tensor_view_get_dim(a, dim);
	assert(ccv_nnc_tensor_view_check_dim(b, dim));
	if (ha)
		{ assert(ccv_nnc_tensor_view_check_dim(ha, dim)); }
	if (hb)
		{ assert(ccv_nnc_tensor_view_check_dim(hb, dim)); }
	ccv_nnc_tensor_view_get_inc(a, ainc);
	ccv_nnc_tensor_view_get_inc(b, binc);
	if (ha)
		ccv_nnc_tensor_view_get_inc(ha, hainc);
	if (hb)
		ccv_nnc_tensor_view_get_inc(hb, hbinc);
	assert(ccv_nnc_tensor_nd(a->info.dim) <= 2);
	const int batch_size = dim[CCV_NNC_MAX_DIM];
	const int count = dim[CCV_NNC_MAX_DIM + 1];
	const int astep = ainc[CCV_NNC_MAX_DIM + 1];
	const int bstep = binc[CCV_NNC_MAX_DIM + 1];
	const int hastep = hainc[CCV_NNC_MAX_DIM + 1];
	const int hbstep = hbinc[CCV_NNC_MAX_DIM + 1];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == ha->info.datatype);
	assert(b->info.datatype == hb->info.datatype);
	const int datatype = a->info.datatype;
	if (g)
	{
		int ginc[CCV_NNC_MAX_DIM_ALLOC];
		ccv_nnc_tensor_view_get_inc(g, ginc);
		assert(ccv_nnc_tensor_count(g->info) == batch_size);
		const int gstep = ccv_nnc_tensor_nd(g->info.dim) == 1 ? 1 : ginc[CCV_NNC_MAX_DIM + 1];
		assert(g->info.datatype == datatype);
		if (cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN)
		{
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
				{
					if (ha)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)ha->data.f16, hastep);
					if (hb)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, b->data.f32, bstep, (__half*)a->data.f16, astep, hb->data.f32, hbstep);
				} else {
					if (ha)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, a->data.f32, astep, b->data.f32, bstep, ha->data.f32, hastep);
					if (hb)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, b->data.f32, bstep, a->data.f32, astep, hb->data.f32, hbstep);
				}
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				if (ha)
					_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)ha->data.f16, hastep);
				if (hb)
					_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)b->data.f16, bstep, (__half*)a->data.f16, astep, (__half*)hb->data.f16, hbstep);
			}
		} else {
			assert(cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_SUM);
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
				{
					if (ha)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)ha->data.f16, hastep);
					if (hb)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, b->data.f32, bstep, (__half*)a->data.f16, astep, hb->data.f32, hbstep);
				} else {
					if (ha)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, a->data.f32, astep, b->data.f32, bstep, ha->data.f32, hastep);
					if (hb)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, g->data.f32, gstep, b->data.f32, bstep, a->data.f32, astep, hb->data.f32, hbstep);
				}
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				if (ha)
					_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)ha->data.f16, hastep);
				if (hb)
					_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)g->data.f16, gstep, (__half*)b->data.f16, bstep, (__half*)a->data.f16, astep, (__half*)hb->data.f16, hbstep);
			}
		}
	} else {
		if (cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_MEAN)
		{
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
				{
					if (ha)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)ha->data.f16, hastep);
					if (hb)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, bstep, (__half*)a->data.f16, astep, hb->data.f32, hbstep);
				} else {
					if (ha)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, ha->data.f32, hastep);
					if (hb)
						_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, bstep, a->data.f32, astep, hb->data.f32, hbstep);
				}
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				if (ha)
					_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)ha->data.f16, hastep);
				if (hb)
					_ccv_nnc_mse_mean_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, bstep, (__half*)a->data.f16, astep, (__half*)hb->data.f16, hbstep);
			}
		} else {
			assert(cmd.info.mse.reduce_op == CCV_NNC_MSE_REDUCE_SUM);
			if (b->info.datatype == CCV_32F)
			{
				if (datatype == CCV_16F)
				{
					if (ha)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, b->data.f32, bstep, (__half*)ha->data.f16, hastep);
					if (hb)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, bstep, (__half*)a->data.f16, astep, hb->data.f32, hbstep);
				} else {
					if (ha)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, a->data.f32, astep, b->data.f32, bstep, ha->data.f32, hastep);
					if (hb)
						_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, b->data.f32, bstep, a->data.f32, astep, hb->data.f32, hbstep);
				}
			} else {
				assert(b->info.datatype == CCV_16F);
				assert(datatype == CCV_16F);
				if (ha)
					_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)a->data.f16, astep, (__half*)b->data.f16, bstep, (__half*)ha->data.f16, hastep);
				if (hb)
					_ccv_nnc_mse_sum_back_kernel<<<CUDA_GET_BLOCKS(batch_size), CUDA_NUM_THREADS, 0, stream>>>(batch_size, count, (__half*)b->data.f16, bstep, (__half*)a->data.f16, astep, (__half*)hb->data.f16, hbstep);
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MSE_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mse_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_MSE_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_mse_back;
}
