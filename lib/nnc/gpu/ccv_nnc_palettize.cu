extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>

template<typename NUM>
__global__ void _ccv_nnc_q4_slow(const size_t count, const size_t length, const int number_in_blocks, const int number_in_blocks_2, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_2;
		const int j = k % number_in_blocks_2;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 16 + number_in_blocks_2) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 16 + j;
		const uint8_t u0 = *ui1;
		const int i0 = (int)(u0 >> 4);
		const int i1 = (int)(u0 & 15);
		const int j2 = j * 2;
		f[j2] = palette[i0];
		if (j2 + 1 < length)
			f[j2 + 1] = palette[i1];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q4_fast(const size_t count, const int number_in_blocks, const int number_in_blocks_2, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_2;
		const int j = k % number_in_blocks_2;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 16 + number_in_blocks_2) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 16 + j;
		const uint8_t u0 = *ui1;
		const int i0 = (int)(u0 >> 4);
		const int i1 = (int)(u0 & 15);
		const int j2 = j * 2;
		f[j2] = palette[i0];
		f[j2 + 1] = palette[i1];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q5_slow(const size_t count, const size_t length, const int number_in_blocks, const int number_in_blocks_8, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_8;
		const int j = k % number_in_blocks_8;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 32 + number_in_blocks_8 * 5) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 32 + j * 5;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const uint8_t u3 = ui1[3];
		const uint8_t u4 = ui1[4];
		const int i0 = (int)(u0 >> 3);
		const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
		const int i2 = (int)((u1 >> 1) & 31);
		const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
		const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
		const int i5 = (int)((u3 >> 2) & 31);
		const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
		const int i7 = (int)(u4 & 31);
		const int j8 = j * 8;
		f[j8] = palette[i0];
		if (j8 + 1 < length)
			f[j8 + 1] = palette[i1];
		if (j8 + 2 < length)
			f[j8 + 2] = palette[i2];
		if (j8 + 3 < length)
			f[j8 + 3] = palette[i3];
		if (j8 + 4 < length)
			f[j8 + 4] = palette[i4];
		if (j8 + 5 < length)
			f[j8 + 5] = palette[i5];
		if (j8 + 6 < length)
			f[j8 + 6] = palette[i6];
		if (j8 + 7 < length)
			f[j8 + 7] = palette[i7];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q5_fast(const size_t count, const int number_in_blocks, const int number_in_blocks_8, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_8;
		const int j = k % number_in_blocks_8;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 32 + number_in_blocks_8 * 5) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 32 + j * 5;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const uint8_t u3 = ui1[3];
		const uint8_t u4 = ui1[4];
		const int i0 = (int)(u0 >> 3);
		const int i1 = (int)(((u0 & 7) << 2) | (u1 >> 6));
		const int i2 = (int)((u1 >> 1) & 31);
		const int i3 = (int)(((u1 & 1) << 4) | (u2 >> 4));
		const int i4 = (int)(((u2 & 15) << 1) | (u3 >> 7));
		const int i5 = (int)((u3 >> 2) & 31);
		const int i6 = (int)(((u3 & 3) << 3) | (u4 >> 5));
		const int i7 = (int)(u4 & 31);
		const int j8 = j * 8;
		f[j8] = palette[i0];
		f[j8 + 1] = palette[i1];
		f[j8 + 2] = palette[i2];
		f[j8 + 3] = palette[i3];
		f[j8 + 4] = palette[i4];
		f[j8 + 5] = palette[i5];
		f[j8 + 6] = palette[i6];
		f[j8 + 7] = palette[i7];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q6_slow(const size_t count, const size_t length, const int number_in_blocks, const int number_in_blocks_4, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_4;
		const int j = k % number_in_blocks_4;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 64 + number_in_blocks_4 * 3) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 64 + j * 3;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const int i0 = (int)(u0 >> 2);
		const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
		const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
		const int i3 = (int)(u2 & 63);
		const int j4 = j * 4;
		f[j4] = palette[i0];
		if (j4 + 1 < length)
			f[j4 + 1] = palette[i1];
		if (j4 + 2 < length)
			f[j4 + 2] = palette[i2];
		if (j4 + 3 < length)
			f[j4 + 3] = palette[i3];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q6_fast(const size_t count, const int number_in_blocks, const int number_in_blocks_4, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_4;
		const int j = k % number_in_blocks_4;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 64 + number_in_blocks_4 * 3) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 64 + j * 3;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const int i0 = (int)(u0 >> 2);
		const int i1 = (int)(((u0 & 3) << 4) | (u1 >> 4));
		const int i2 = (int)(((u1 & 15) << 2) | (u2 >> 6));
		const int i3 = (int)(u2 & 63);
		const int j4 = j * 4;
		f[j4] = palette[i0];
		f[j4 + 1] = palette[i1];
		f[j4 + 2] = palette[i2];
		f[j4 + 3] = palette[i3];
	}
}

template<int REPEAT_4, typename NUM>
__global__ void _ccv_nnc_q6_fast_s4(const int number_in_blocks_4, const uint8_t* const a, NUM* const b)
{
	const int i = blockIdx.y;
	const int j = blockIdx.x;
	const uint8_t* const ui0 = a + (sizeof(NUM) * 64 + number_in_blocks_4 * 3) * i;
	__shared__ NUM palette[64];
	if (threadIdx.x < 64)
		palette[threadIdx.x] = ((NUM*)ui0)[threadIdx.x];
	__syncthreads();
	NUM* const f = b + number_in_blocks_4 * 4 * i + j * blockDim.x * REPEAT_4 * 4;
	const uint8_t* ui1 = (uint8_t*)(ui0 + sizeof(NUM) * 64) + j * blockDim.x * REPEAT_4 * 3;
	#pragma unroll
	for (int k = 0; k < REPEAT_4; k++)
	{
		const uint8_t u0 = ui1[(k * blockDim.x + threadIdx.x) * 3];
		const uint8_t u1 = ui1[(k * blockDim.x + threadIdx.x) * 3 + 1];
		const uint8_t u2 = ui1[(k * blockDim.x + threadIdx.x) * 3 + 2];
		f[(k * blockDim.x + threadIdx.x) * 4] = palette[u0 >> 2];
		f[(k * blockDim.x + threadIdx.x) * 4 + 1] = palette[((u0 & 3) << 4) | (u1 >> 4)];
		f[(k * blockDim.x + threadIdx.x) * 4 + 2] = palette[((u1 & 15) << 2) | (u2 >> 6)];
		f[(k * blockDim.x + threadIdx.x) * 4 + 3] = palette[u2 & 63];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q7_slow(const size_t count, const size_t length, const int number_in_blocks, const int number_in_blocks_8, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_8;
		const int j = k % number_in_blocks_8;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 128 + number_in_blocks_8 * 7) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 128 + j * 7;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const uint8_t u3 = ui1[3];
		const uint8_t u4 = ui1[4];
		const uint8_t u5 = ui1[5];
		const uint8_t u6 = ui1[6];
		const int i0 = (int)(u0 >> 1);
		const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
		const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
		const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
		const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
		const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
		const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
		const int i7 = (int)(u6 & 127);
		const int j8 = j * 8;
		f[j8] = palette[i0];
		if (j8 + 1 < length)
			f[j8 + 1] = palette[i1];
		if (j8 + 2 < length)
			f[j8 + 2] = palette[i2];
		if (j8 + 3 < length)
			f[j8 + 3] = palette[i3];
		if (j8 + 4 < length)
			f[j8 + 4] = palette[i4];
		if (j8 + 5 < length)
			f[j8 + 5] = palette[i5];
		if (j8 + 6 < length)
			f[j8 + 6] = palette[i6];
		if (j8 + 7 < length)
			f[j8 + 7] = palette[i7];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q7_fast(const size_t count, const int number_in_blocks, const int number_in_blocks_8, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks_8;
		const int j = k % number_in_blocks_8;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 128 + number_in_blocks_8 * 7) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 128 + j * 7;
		const uint8_t u0 = ui1[0];
		const uint8_t u1 = ui1[1];
		const uint8_t u2 = ui1[2];
		const uint8_t u3 = ui1[3];
		const uint8_t u4 = ui1[4];
		const uint8_t u5 = ui1[5];
		const uint8_t u6 = ui1[6];
		const int i0 = (int)(u0 >> 1);
		const int i1 = (int)(((u0 & 1) << 6) | (u1 >> 2));
		const int i2 = (int)(((u1 & 3) << 5) | (u2 >> 3));
		const int i3 = (int)(((u2 & 7) << 4) | (u3 >> 4));
		const int i4 = (int)(((u3 & 15) << 3) | (u4 >> 5));
		const int i5 = (int)(((u4 & 31) << 2) | (u5 >> 6));
		const int i6 = (int)(((u5 & 63) << 1) | (u6 >> 7));
		const int i7 = (int)(u6 & 127);
		const int j8 = j * 8;
		f[j8] = palette[i0];
		f[j8 + 1] = palette[i1];
		f[j8 + 2] = palette[i2];
		f[j8 + 3] = palette[i3];
		f[j8 + 4] = palette[i4];
		f[j8 + 5] = palette[i5];
		f[j8 + 6] = palette[i6];
		f[j8 + 7] = palette[i7];
	}
}

template<typename NUM>
__global__ void _ccv_nnc_q8_fast(const size_t count, const int number_in_blocks, const uint8_t* const a, NUM* const b)
{
	CUDA_1D_KERNEL_LOOP(k, count) {
		const int i = k / number_in_blocks;
		const int j = k % number_in_blocks;
		const uint8_t* const ui0 = a + (sizeof(NUM) * 256 + number_in_blocks) * i;
		NUM* const f = b + number_in_blocks * i;
		const NUM* const palette = (NUM*)ui0;
		const uint8_t* ui1 = ui0 + sizeof(NUM) * 256 + j;
		const uint8_t u0 = *ui1;
		const int i0 = (int)u0;
		f[j] = palette[i0];
	}
}

template<int REPEAT_4, typename NUM>
__global__ void _ccv_nnc_q8_fast_s4(const int number_in_blocks, const uint8_t* const a, NUM* const b)
{
	const int i = blockIdx.y;
	const int j = blockIdx.x;
	const uint8_t* const ui0 = a + (sizeof(NUM) * 256 + number_in_blocks) * i;
	__shared__ NUM palette[256];
	if (threadIdx.x < 256)
		palette[threadIdx.x] = ((NUM*)ui0)[threadIdx.x];
	__syncthreads();
	NUM* const f = b + number_in_blocks * i + j * blockDim.x * REPEAT_4 * 4;
	const uint32_t* ui1 = (uint32_t*)(ui0 + sizeof(NUM) * 256) + j * blockDim.x * REPEAT_4;
	#pragma unroll
	for (int k = 0; k < REPEAT_4; k++)
	{
		const uint32_t u0 = ui1[k * blockDim.x + threadIdx.x];
		f[(k * blockDim.x + threadIdx.x) * 4] = palette[u0 & 0xff];
		f[(k * blockDim.x + threadIdx.x) * 4 + 1] = palette[(u0 >> 8) & 0xff];
		f[(k * blockDim.x + threadIdx.x) * 4 + 2] = palette[(u0 >> 16) & 0xff];
		f[(k * blockDim.x + threadIdx.x) * 4 + 3] = palette[u0 >> 24];
	}
}

void ccv_nnc_compat_depalettize(const void* input, const int datatype, const size_t input_length, const int qbits, const int number_in_blocks, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_32F || datatype == CCV_64F);
	assert(qbits == 4 || qbits == 5 || qbits == 6 || qbits == 7 || qbits == 8);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (datatype == CCV_16F)
	{
		if (qbits == 4)
		{
			const int number_in_blocks_2 = number_in_blocks / 2;
			const size_t count = (output_length + 1) / 2;
			if (output_length % 2 == 0)
				_ccv_nnc_q4_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (__half*)output);
			else
				_ccv_nnc_q4_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (__half*)output);
		} else if (qbits == 5) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q5_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__half*)output);
			else
				_ccv_nnc_q5_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__half*)output);
		} else if (qbits == 6) {
			const int number_in_blocks_4 = number_in_blocks / 4;
			const size_t count = (output_length + 3) / 4;
			if (output_length % 4 == 0)
			{
				if (number_in_blocks % (1024 * 4) == 0 && output_length % number_in_blocks == 0)
				{
					const int num_blocks = output_length / number_in_blocks;
					const int repeat_4 = number_in_blocks / (1024 * 4);
					_ccv_nnc_q6_fast_s4<1, __half><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks_4, (uint8_t*)input, (__half*)output);
				} else
					_ccv_nnc_q6_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (__half*)output);
			} else
				_ccv_nnc_q6_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (__half*)output);
		} else if (qbits == 7) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q7_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__half*)output);
			else
				_ccv_nnc_q7_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__half*)output);
		} else {
			if ((number_in_blocks % (1024 * 4 * 2)) == 0 && (output_length % number_in_blocks) == 0)
			{
				const int num_blocks = output_length / number_in_blocks;
				const int repeat_4 = number_in_blocks / (1024 * 4 * 2);
				_ccv_nnc_q8_fast_s4<2, __half><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks, (uint8_t*)input, (__half*)output);
			} else
				_ccv_nnc_q8_fast<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, number_in_blocks, (uint8_t*)input, (__half*)output);
		}
	} else if (datatype == CCV_32F) {
		if (qbits == 4)
		{
			const int number_in_blocks_2 = number_in_blocks / 2;
			const size_t count = (output_length + 1) / 2;
			if (output_length % 2 == 0)
				_ccv_nnc_q4_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (float*)output);
			else
				_ccv_nnc_q4_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (float*)output);
		} else if (qbits == 5) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q5_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (float*)output);
			else
				_ccv_nnc_q5_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (float*)output);
		} else if (qbits == 6) {
			const int number_in_blocks_4 = number_in_blocks / 4;
			const size_t count = (output_length + 3) / 4;
			if (output_length % 4 == 0)
			{
				if (number_in_blocks % (1024 * 4) == 0 && output_length % number_in_blocks == 0)
				{
					const int num_blocks = output_length / number_in_blocks;
					const int repeat_4 = number_in_blocks / (1024 * 4);
					_ccv_nnc_q6_fast_s4<1, float><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks_4, (uint8_t*)input, (float*)output);
				} else
					_ccv_nnc_q6_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (float*)output);
			} else
				_ccv_nnc_q6_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (float*)output);
		} else if (qbits == 7) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q7_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (float*)output);
			else
				_ccv_nnc_q7_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (float*)output);
		} else {
			if ((number_in_blocks % (1024 * 4 * 2)) == 0 && (output_length % number_in_blocks) == 0)
			{
				const int num_blocks = output_length / number_in_blocks;
				const int repeat_4 = number_in_blocks / (1024 * 4 * 2);
				_ccv_nnc_q8_fast_s4<2, float><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks, (uint8_t*)input, (float*)output);
			} else
				_ccv_nnc_q8_fast<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, number_in_blocks, (uint8_t*)input, (float*)output);
		}
	} else {
		if (qbits == 4)
		{
			const int number_in_blocks_2 = number_in_blocks / 2;
			const size_t count = (output_length + 1) / 2;
			if (output_length % 2 == 0)
				_ccv_nnc_q4_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (double*)output);
			else
				_ccv_nnc_q4_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (double*)output);
		} else if (qbits == 5) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q5_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (double*)output);
			else
				_ccv_nnc_q5_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (double*)output);
		} else if (qbits == 6) {
			const int number_in_blocks_4 = number_in_blocks / 4;
			const size_t count = (output_length + 3) / 4;
			if (output_length % 4 == 0)
			{
				if (number_in_blocks % (1024 * 4) == 0 && output_length % number_in_blocks == 0)
				{
					const int num_blocks = output_length / number_in_blocks;
					const int repeat_4 = number_in_blocks / (1024 * 4);
					_ccv_nnc_q6_fast_s4<1, double><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks_4, (uint8_t*)input, (double*)output);
				} else
					_ccv_nnc_q6_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (double*)output);
			} else
				_ccv_nnc_q6_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (double*)output);
		} else if (qbits == 7) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q7_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (double*)output);
			else
				_ccv_nnc_q7_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (double*)output);
		} else {
			if ((number_in_blocks % (1024 * 4 * 2)) == 0 && (output_length % number_in_blocks) == 0)
			{
				const int num_blocks = output_length / number_in_blocks;
				const int repeat_4 = number_in_blocks / (1024 * 4 * 2);
				_ccv_nnc_q8_fast_s4<2, double><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks, (uint8_t*)input, (double*)output);
			} else
				_ccv_nnc_q8_fast<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, number_in_blocks, (uint8_t*)input, (double*)output);
		}
	}
}
