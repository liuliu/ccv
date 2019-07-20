extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

__global__ void _ccv_nnc_lssc_nchw_forw_kernel_no_padding(const int X_H, const int X_W, const int Y_H, const int Y_W, const __half* const X, __half* const Y)
{
	const int X_HxW = X_H * X_W;
	const int Y_HxW = Y_H * Y_W;
	const int Y_W4 = Y_W / 4;
	const int Y_HxW4 = Y_H * Y_W4;
	const __half* const Xp = X + X_HxW * blockIdx.x;
	__half* const Yp = Y + Y_HxW * blockIdx.x;
	for (size_t idx = threadIdx.x; idx < Y_HxW4; idx += blockDim.x)
	{
		const int y = idx / Y_W4;
		const int x = idx % Y_W4;
		const __half* const Xpz = Xp + (x + y * X_W) * 4;
		__half X16_0 = 0;
#if __CUDA_ARCH__ >= 350
		X16_0 = __ldg(Xpz);
#else
		X16_0 = Xpz[0];
#endif
		float X16[16];
		float Xmax = (float)X16_0;
		float Xmin = Xmax;
#pragma unroll
		for (int i = 0; i < 16; i++)
			X16[i] = Xmax;
#pragma unroll
		for (int i = 0; i < 4; i++)
#pragma unroll
			for (int j = 0; j < 4; j++)
#if __CUDA_ARCH__ >= 350
				X16[i * 4 + j] = (float)__ldg(Xpz + i * X_W + j);
#else
				X16[i * 4 + j] = (float)Xpz[i * X_W + j];
#endif
#pragma unroll
		for (int i = 1; i < 16; i++)
			Xmax = max(Xmax, X16[i]), Xmin = min(Xmin, X16[i]);
		const float Xbottom = Xmin * 7 / 6 - Xmax / 6;
		const float Xscale = 3 / max(Xmax - Xmin, 1e-6);
		uint16_t m0 = 0, m1 = 0;
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int v = (int)((X16[i] - Xbottom) * Xscale);
			m0 |= (min(max(v, 0), 3) << (i << 1));
		}
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int v = (int)((X16[8 + i] - Xbottom) * Xscale);
			m1 |= (min(max(v, 0), 3) << (i << 1));
		}
		__half* const Ypz = Yp + y * Y_W + x * 4;
		Ypz[0] = Xmin;
		Ypz[1] = Xmax;
		((uint16_t*)Ypz)[2] = m0;
		((uint16_t*)Ypz)[3] = m1;
	}
}

__global__ void _ccv_nnc_lssc_nchw_forw_kernel(const int X_H, const int X_W, const int Y_H, const int Y_W, const __half* const X, __half* const Y)
{
	const int X_HxW = X_H * X_W;
	const int Y_HxW = Y_H * Y_W;
	const int Y_W4 = Y_W / 4;
	const int Y_HxW4 = Y_H * Y_W4;
	const __half* const Xp = X + X_HxW * blockIdx.x;
	__half* const Yp = Y + Y_HxW * blockIdx.x;
	for (size_t idx = threadIdx.x; idx < Y_HxW4; idx += blockDim.x)
	{
		const int y = idx / Y_W4;
		const int x = idx % Y_W4;
		const __half* const Xpz = Xp + (x + y * X_W) * 4;
		__half X16_0 = 0;
#if __CUDA_ARCH__ >= 350
		X16_0 = __ldg(Xpz);
#else
		X16_0 = Xpz[0];
#endif
		float X16[16];
		float Xmax = (float)X16_0;
		float Xmin = Xmax;
#pragma unroll
		for (int i = 0; i < 16; i++)
			X16[i] = Xmax;
		const int xh = min(y * 4 + 4, X_H) - y * 4;
		const int xw = min(x * 4 + 4, X_W) - x * 4;
#pragma unroll
		for (int j = 1; j < xw; j++)
#if __CUDA_ARCH__ >= 350
			X16[j] = (float)__ldg(Xpz + j);
#else
			X16[j] = (float)Xpz[j];
#endif
#pragma unroll
		for (int i = 1; i < xh; i++)
		{
#if __CUDA_ARCH__ >= 350
			X16[i * 4] = (float)__ldg(Xpz + i * X_W);
#else
			X16[i * 4] = (float)Xpz[i * X_W];
#endif
#pragma unroll
			for (int j = 1; j < xw; j++)
#if __CUDA_ARCH__ >= 350
				X16[i * 4 + j] = (float)__ldg(Xpz + i * X_W + j);
#else
				X16[i * 4 + j] = (float)Xpz[i * X_W + j];
#endif
		}
#pragma unroll
		for (int i = 1; i < 16; i++)
			Xmax = max(Xmax, X16[i]), Xmin = min(Xmin, X16[i]);
		const float Xbottom = Xmin * 7 / 6 - Xmax / 6;
		const float Xscale = 3 / max(Xmax - Xmin, 1e-6);
		uint16_t m0 = 0, m1 = 0;
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int v = (int)((X16[i] - Xbottom) * Xscale);
			m0 |= (min(max(v, 0), 3) << (i << 1));
		}
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int v = (int)((X16[8 + i] - Xbottom) * Xscale);
			m1 |= (min(max(v, 0), 3) << (i << 1));
		}
		__half* const Ypz = Yp + y * Y_W + x * 4;
		Ypz[0] = Xmin;
		Ypz[1] = Xmax;
		((uint16_t*)Ypz)[2] = m0;
		((uint16_t*)Ypz)[3] = m1;
	}
}

static int _ccv_nnc_lssc_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	int i;
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		assert(!CCV_IS_TENSOR_VIEW(a));
		assert(!CCV_IS_TENSOR_VIEW(b));
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
		const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
		const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
		const int n = ccv_nnc_tensor_get_n(a->info);
		assert(n == ccv_nnc_tensor_get_n(b->info));
		const int c = ccv_nnc_tensor_get_c(a->info);
		assert(c == ccv_nnc_tensor_get_c(b->info));
		assert(bdim[CCV_NNC_MAX_DIM] % 4 == 0);
		const int threadDim = ccv_min(bdim[1] * bdim[CCV_NNC_MAX_DIM] / 4, CUDA_NUM_THREADS);
		if (adim[1] % 4 == 0 && adim[CCV_NNC_MAX_DIM] % 4 == 0)
			_ccv_nnc_lssc_nchw_forw_kernel_no_padding<<<n * c, threadDim, 0, stream>>>(adim[1], adim[CCV_NNC_MAX_DIM], bdim[1], bdim[CCV_NNC_MAX_DIM], (__half*)a->data.f16, (__half*)b->data.f16);
		else
			_ccv_nnc_lssc_nchw_forw_kernel<<<n * c, threadDim, 0, stream>>>(adim[1], adim[CCV_NNC_MAX_DIM], bdim[1], bdim[CCV_NNC_MAX_DIM], (__half*)a->data.f16, (__half*)b->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

__global__ void _ccv_nnc_lssc_nchw_back_kernel_no_padding(const int Y_H, const int Y_W, const int X_H, const int X_W, const __half* const Y, __half* const X)
{
	const int X_HxW = X_H * X_W;
	const int Y_HxW = Y_H * Y_W;
	const int Y_W4 = Y_W / 4;
	const int Y_HxW4 = Y_H * Y_W4;
	__half* const Xp = X + X_HxW * blockIdx.x;
	const __half* const Yp = Y + Y_HxW * blockIdx.x;
	for (size_t idx = threadIdx.x; idx < Y_HxW4; idx += blockDim.x)
	{
		const int y = idx / Y_W4;
		const int x = idx % Y_W4;
		const __half* const Ypz = Yp + y * Y_W + x * 4;
		__half X4[4];
		uint16_t m0;
		uint16_t m1;
#if __CUDA_ARCH__ >= 350
		X4[0] = __ldg(Ypz);
		X4[3] = __ldg(Ypz + 1);
		m0 = __ldg((uint16_t*)Ypz + 2);
		m1 = __ldg((uint16_t*)Ypz + 3);
#else
		X4[0] = Ypz[0];
		X4[3] = Ypz[1];
		m0 = ((uint16_t*)Ypz)[2];
		m1 = ((uint16_t*)Ypz)[3];
#endif
		const float Xmin = (float)X4[0];
		const float Xmax = (float)X4[3];
		X4[1] = (__half)(Xmax / 3 + Xmin * 2 / 3);
		X4[2] = (__half)(Xmax * 2 / 3 + Xmin / 3);
		__half X16[16];
#pragma unroll
		for (int i = 0; i < 8; i++)
			X16[i] = X4[((m0 >> (i << 1)) & 3)];
#pragma unroll
		for (int i = 0; i < 8; i++)
			X16[8 + i] = X4[((m1 >> (i << 1)) & 3)];
		__half* const Xpz = Xp + (x + y * X_W) * 4;
#pragma unroll
		for (int i = 0; i < 4; i++)
#pragma unroll
			for (int j = 0; j < 4; j++)
				Xpz[i * X_W + j] = X16[i * 4 + j];
	}
}

__global__ void _ccv_nnc_lssc_nchw_back_kernel(const int Y_H, const int Y_W, const int X_H, const int X_W, const __half* const Y, __half* const X)
{
	const int X_HxW = X_H * X_W;
	const int Y_HxW = Y_H * Y_W;
	const int Y_W4 = Y_W / 4;
	const int Y_HxW4 = Y_H * Y_W4;
	__half* const Xp = X + X_HxW * blockIdx.x;
	const __half* const Yp = Y + Y_HxW * blockIdx.x;
	for (size_t idx = threadIdx.x; idx < Y_HxW4; idx += blockDim.x)
	{
		const int y = idx / Y_W4;
		const int x = idx % Y_W4;
		const __half* const Ypz = Yp + y * Y_W + x * 4;
		__half X4[4];
		uint16_t m0;
		uint16_t m1;
#if __CUDA_ARCH__ >= 350
		X4[0] = __ldg(Ypz);
		X4[3] = __ldg(Ypz + 1);
		m0 = __ldg((uint16_t*)Ypz + 2);
		m1 = __ldg((uint16_t*)Ypz + 3);
#else
		X4[0] = Ypz[0];
		X4[3] = Ypz[1];
		m0 = ((uint16_t*)Ypz)[2];
		m1 = ((uint16_t*)Ypz)[3];
#endif
		const float Xmin = (float)X4[0];
		const float Xmax = (float)X4[3];
		X4[1] = (__half)(Xmax / 3 + Xmin * 2 / 3);
		X4[2] = (__half)(Xmax * 2 / 3 + Xmin / 3);
		__half X16[16];
#pragma unroll
		for (int i = 0; i < 8; i++)
			X16[i] = X4[((m0 >> (i << 1)) & 3)];
#pragma unroll
		for (int i = 0; i < 8; i++)
			X16[8 + i] = X4[((m1 >> (i << 1)) & 3)];
		__half* const Xpz = Xp + (x + y * X_W) * 4;
		const int xh = min(y * 4 + 4, X_H) - y * 4;
		const int xw = min(x * 4 + 4, X_W) - x * 4;
		Xpz[0] = X16[0];
#pragma unroll
		for (int j = 1; j < xw; j++)
			Xpz[j] = X16[j];
#pragma unroll
		for (int i = 1; i < xh; i++)
		{
			Xpz[i * X_W] = X16[i * 4];
#pragma unroll
			for (int j = 1; j < xw; j++)
				Xpz[i * X_W + j] = X16[i * 4 + j];
		}
	}
}

static int _ccv_nnc_lssc_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size <= input_size);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	int i;
	for (i = 0; i < output_size; i++)
	{
		const ccv_nnc_tensor_view_t* a = (ccv_nnc_tensor_view_t*)inputs[i];
		ccv_nnc_tensor_view_t* b = (ccv_nnc_tensor_view_t*)outputs[i];
		assert(!CCV_IS_TENSOR_VIEW(a));
		assert(!CCV_IS_TENSOR_VIEW(b));
		const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
		assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
		const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
		const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
		assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
		const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
		const int n = ccv_nnc_tensor_get_n(a->info);
		assert(n == ccv_nnc_tensor_get_n(b->info));
		const int c = ccv_nnc_tensor_get_c(a->info);
		assert(c == ccv_nnc_tensor_get_c(b->info));
		assert(adim[CCV_NNC_MAX_DIM] % 4 == 0);
		const int threadDim = ccv_min(adim[1] * adim[CCV_NNC_MAX_DIM] / 4, CUDA_NUM_THREADS);
		if (bdim[1] % 4 == 0 && bdim[CCV_NNC_MAX_DIM] % 4 == 0)
			_ccv_nnc_lssc_nchw_back_kernel_no_padding<<<n * c, threadDim, 0, stream>>>(adim[1], adim[CCV_NNC_MAX_DIM], bdim[1], bdim[CCV_NNC_MAX_DIM], (__half*)a->data.f16, (__half*)b->data.f16);
		else
			_ccv_nnc_lssc_nchw_back_kernel<<<n * c, threadDim, 0, stream>>>(adim[1], adim[CCV_NNC_MAX_DIM], bdim[1], bdim[CCV_NNC_MAX_DIM], (__half*)a->data.f16, (__half*)b->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMPRESSION_LSSC_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lssc_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMPRESSION_LSSC_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_lssc_back;
}
