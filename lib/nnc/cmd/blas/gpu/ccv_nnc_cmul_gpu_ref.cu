extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

#ifdef HAVE_CUDA

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_cmul_kernel(const size_t count, const NUM1* const a, const NUM2* const b, NUM3* const c)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const NUM1 a0 = a[i * 2];
		const NUM1 a1 = a[i * 2 + 1];
		const NUM1 b0 = b[i * 2];
		const NUM1 b1 = b[i * 2 + 1];
		c[i * 2] = (NUM3)(a0 * b0 - a1 * b1);
		c[i * 2 + 1] = (NUM3)(a0 * b1 + a1 * b0);
	}
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_cmul_kernel_4d_0(const int astride2, const int astride1, const int astride0, const int bstride2, const int bstride1, const int bstride0, const int cstride2, const int cstride1, const int cstride0, const int dim2, const int dim1, const int dim0, const NUM1* const a, const NUM2* const b, NUM3* const c)
{
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= dim1 || x >= dim0)
		return;
	const int u = z % dim2;
	const int v = z / dim2;
	const int ida = v * astride2 + u * astride1 + y * astride0 + x * 2;
	const int idb = v * bstride2 + u * bstride1 + y * bstride0 + x * 2;
	const int idc = v * cstride2 + u * cstride1 + y * cstride0 + x * 2;
	const NUM1 a0 = a[ida];
	const NUM1 a1 = a[ida + 1];
	const NUM1 b0 = b[idb];
	const NUM1 b1 = b[idb + 1];
	c[idc] = (NUM3)(a0 * b0 - a1 * b1);
	c[idc + 1] = (NUM3)(a0 * b1 + a1 * b0);
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_cmul_kernel_3d_0(const int astride1, const int astride0, const int bstride1, const int bstride0, const int cstride1, const int cstride0, const int dim1, const int dim0, const NUM1* const a, const NUM2* const b, NUM3* const c)
{
	const int z = blockIdx.z * blockDim.z + threadIdx.z;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= dim1 || x >= dim0)
		return;
	const int ida = z * astride1 + y * astride0 + x * 2;
	const int idb = z * bstride1 + y * bstride0 + x * 2;
	const int idc = z * cstride1 + y * cstride0 + x * 2;
	const NUM1 a0 = a[ida];
	const NUM1 a1 = a[ida + 1];
	const NUM1 b0 = b[idb];
	const NUM1 b1 = b[idb + 1];
	c[idc] = (NUM3)(a0 * b0 - a1 * b1);
	c[idc + 1] = (NUM3)(a0 * b1 + a1 * b0);
}

template<typename NUM1, typename NUM2, typename NUM3>
__global__ void _ccv_nnc_cmul_kernel_2d_0(const int astride, const int bstride, const int cstride, const int dim1, const int dim0, const NUM1* const a, const NUM2* const b, NUM3* const c)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= dim1 || x >= dim0)
		return;
	const int ida = y * astride + x * 2;
	const int idb = y * bstride + x * 2;
	const int idc = y * cstride + x * 2;
	const NUM1 a0 = a[ida];
	const NUM1 a1 = a[ida + 1];
	const NUM1 b0 = b[idb];
	const NUM1 b1 = b[idb + 1];
	c[idc] = (NUM3)(a0 * b0 - a1 * b1);
	c[idc + 1] = (NUM3)(a0 * b1 + a1 * b0);
}

static int _ccv_nnc_cmul_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 2);
	const ccv_nnc_tensor_t* const a = inputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const ccv_nnc_tensor_t* const b = inputs[1];
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const c = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(c));
	const size_t count = ccv_nnc_tensor_count(c->info) / 2;
	int i;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	assert(a->info.datatype == b->info.datatype);
	// If there is no broadcast, just do the simplest cmul.
	if (ccv_nnc_tensor_count(a->info) == count * 2 && ccv_nnc_tensor_count(b->info) == count * 2)
	{
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && b->info.dim[i] > 0; i++)
			{ assert(b->info.dim[i] == c->info.dim[i]); }
		for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && a->info.dim[i] > 0; i++)
			{ assert(a->info.dim[i] == b->info.dim[i]); }
		if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
		{
			_ccv_nnc_cmul_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, c->data.f32);
		} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
			_ccv_nnc_cmul_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, a->data.f32, b->data.f32, (__half*)c->data.f16);
		} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
			_ccv_nnc_cmul_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
		} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
			_ccv_nnc_cmul_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
		}
	} else {
		// Otherwise, find strides for both to increment.
		int nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(nd = ccv_nnc_tensor_nd(b->info.dim));
		assert(nd = ccv_nnc_tensor_nd(c->info.dim));
		int adim[CCV_NNC_MAX_DIM_ALLOC];
		int bdim[CCV_NNC_MAX_DIM_ALLOC];
		int cdim[CCV_NNC_MAX_DIM_ALLOC];
		int squeezed_dims = 0;
		for (i = nd - 1; i >= 0; i--)
		{
			if (c->info.dim[i] == 1)
				continue;
			adim[squeezed_dims] = a->info.dim[i];
			bdim[squeezed_dims] = b->info.dim[i];
			cdim[squeezed_dims] = c->info.dim[i];
			squeezed_dims += 1;
		}
		nd = squeezed_dims;
		int astride[CCV_NNC_MAX_DIM_ALLOC];
		int bstride[CCV_NNC_MAX_DIM_ALLOC];
		int cstride[CCV_NNC_MAX_DIM_ALLOC];
		astride[0] = 1;
		bstride[0] = 1;
		cstride[0] = 1;
		for (i = 1; i < nd; i++)
		{
			astride[i] = adim[i - 1] * astride[i - 1];
			bstride[i] = bdim[i - 1] * bstride[i - 1];
			cstride[i] = cdim[i - 1] * cstride[i - 1];
		}
		for (i = 0; i < nd; i++)
		{
			if (cdim[i] == adim[i] && cdim[i] == bdim[i])
				continue;
			if (cdim[i] == adim[i])
			{
				assert(bdim[i] == 1);
				bstride[i] = 0;
			} else {
				assert(cdim[i] == bdim[i]);
				assert(adim[i] == 1);
				astride[i] = 0;
			}
		}
		assert(nd <= 4);
		if (nd == 4)
		{
			if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
			{
				_ccv_nnc_cmul_kernel_4d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2] * cdim[3]), dim3(64, 8, 1), 0, stream>>>(astride[3], astride[2], astride[1], bstride[3], bstride[2], bstride[1], cstride[3], cstride[2], cstride[1], cdim[2], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, c->data.f32);
			} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_4d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2] * cdim[3]), dim3(64, 8, 1), 0, stream>>>(astride[3], astride[2], astride[1], bstride[3], bstride[2], bstride[1], cstride[3], cstride[2], cstride[1], cdim[2], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, (__half*)c->data.f16);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
				_ccv_nnc_cmul_kernel_4d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2] * cdim[3]), dim3(64, 8, 1), 0, stream>>>(astride[3], astride[2], astride[1], bstride[3], bstride[2], bstride[1], cstride[3], cstride[2], cstride[1], cdim[2], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_4d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2] * cdim[3]), dim3(64, 8, 1), 0, stream>>>(astride[3], astride[2], astride[1], bstride[3], bstride[2], bstride[1], cstride[3], cstride[2], cstride[1], cdim[2], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
			}
		} else if (nd == 3) {
			if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
			{
				_ccv_nnc_cmul_kernel_3d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2]), dim3(64, 8, 1), 0, stream>>>(astride[2], astride[1], bstride[2], bstride[1], cstride[2], cstride[1], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, c->data.f32);
			} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_3d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2]), dim3(64, 8, 1), 0, stream>>>(astride[2], astride[1], bstride[2], bstride[1], cstride[2], cstride[1], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, (__half*)c->data.f16);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
				_ccv_nnc_cmul_kernel_3d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2]), dim3(64, 8, 1), 0, stream>>>(astride[2], astride[1], bstride[2], bstride[1], cstride[2], cstride[1], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_3d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, cdim[2]), dim3(64, 8, 1), 0, stream>>>(astride[2], astride[1], bstride[2], bstride[1], cstride[2], cstride[1], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
			}
		} else if (nd == 2) {
			assert(adim[0] == bdim[0] && adim[0] == cdim[0]);
			if (a->info.datatype == CCV_32F && c->info.datatype == CCV_32F)
			{
				_ccv_nnc_cmul_kernel_2d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, 1), dim3(64, 8, 1), 0, stream>>>(astride[1], bstride[1], cstride[1], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, c->data.f32);
			} else if (a->info.datatype == CCV_32F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_2d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, 1), dim3(64, 8, 1), 0, stream>>>(astride[1], bstride[1], cstride[1], cdim[1], cdim[0] / 2, a->data.f32, b->data.f32, (__half*)c->data.f16);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_32F) {
				_ccv_nnc_cmul_kernel_2d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, 1), dim3(64, 8, 1), 0, stream>>>(astride[1], bstride[1], cstride[1], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, c->data.f32);
			} else if (a->info.datatype == CCV_16F && c->info.datatype == CCV_16F) {
				_ccv_nnc_cmul_kernel_2d_0<<<dim3((cdim[0] / 2 + 63) / 64, (cdim[1] + 7) / 8, 1), dim3(64, 8, 1), 0, stream>>>(astride[1], bstride[1], cstride[1], cdim[1], cdim[0] / 2, (__half*)a->data.f16, (__half*)b->data.f16, (__half*)c->data.f16);
			}
		}
		nd = squeezed_dims;
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_cmul_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

#endif

REGISTER_COMMAND_BACKEND(CCV_NNC_CMUL_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_cmul_forw;
#endif
}

REGISTER_COMMAND_BACKEND(CCV_NNC_CMUL_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
#ifdef HAVE_CUDA
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_cmul_back;
#endif
}

