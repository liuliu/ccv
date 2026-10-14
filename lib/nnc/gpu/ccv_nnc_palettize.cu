extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <type_traits>

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

// CUDA kernels cannot access the host-only static const grids emitted by default,
// so include the shared tables with device storage in this translation unit.
#define CCV_NNC_8I_ROWWISE_PACKED_GRID_CONST static __device__ const
#include "../ccv_nnc_8i_rowwise_packed_grids.inc"
#undef CCV_NNC_8I_ROWWISE_PACKED_GRID_CONST

static __device__ __forceinline__ uint32_t _ccv_nnc_8i_rowwise_x_read_bits(const uint8_t* const input, const size_t bit_offset, const int bits)
{
	const size_t byte_offset = bit_offset >> 3;
	const int shift = bit_offset & 7;
	const uint32_t value = (uint32_t)input[byte_offset] |
		((uint32_t)input[byte_offset + 1] << 8) |
		((uint32_t)input[byte_offset + 2] << 16);
	return (value >> shift) & ((1u << bits) - 1);
}

typedef struct {
	const uint8_t* input;
	size_t bit_offset;

	__device__ __forceinline__ uint32_t read(const int offset, const int bits) const
	{
		return _ccv_nnc_8i_rowwise_x_read_bits(input, bit_offset + offset, bits);
	}
} _ccv_nnc_8i_rowwise_x_global_bits_t;

template<int FORMAT, typename BITS>
static __device__ __forceinline__ int _ccv_nnc_8i_rowwise_x_decode(const BITS& bits, const int lane)
{
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K)
	{
		const int q = (int)bits.read(lane * 5, 5) - 16;
		const int m = (int)bits.read(80, 3) + 1;
		const int b = (int)bits.read(83, 5) - 16;
		return q * m + b;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K)
	{
		const int uq = (int)bits.read(lane * 6, 6);
		const int q = (uq ^ 32) - 32;
		const int m = (int)bits.read(48, 2) + 1;
		const int ub = (int)bits.read(50, 2);
		const int b = (ub ^ 2) - 2;
		return q * m + b;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K)
	{
		const int q = (int)bits.read(lane * 4, 4) - 8;
		const int m = (int)bits.read(64, 4) + 1;
		const int b = (int)bits.read(68, 4) - 8;
		return q * m + b;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K)
	{
		const int q = (int)bits.read(lane * 3, 3) - 4;
		const int m = (int)bits.read(48, 5) + 1;
		const int b = ((int)bits.read(53, 3) - 4) * 2;
		return q * m + b;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K)
	{
		const int q = (int)bits.read(lane * 2, 2);
		const int m = (int)bits.read(32, 6) + 1;
		const int z = (int)bits.read(38, 4) * 8;
		return q * m - z;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
	{
		const int grid_lane = lane & 7;
		const int grid_index = (int)bits.read(lane >= 8 ? 10 : 0, 10);
		const uint64_t grid = ccv_nnc_8i_rowwise_packed_iq2s_grid[grid_index];
		const int mag0 = (int)((grid >> (grid_lane * 8)) & 0xff) >> 3;
		const int scale = (int)bits.read(36, 6) + 1;
		const int mag = ccv_min(mag0 * scale, 127);
		const int negative = (int)bits.read(20 + lane, 1);
		return negative ? -mag : mag;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS)
	{
		const int grid_index = (int)bits.read(0, 9);
		const uint64_t grid = ccv_nnc_8i_rowwise_packed_iq2xs_grid[grid_index];
		const int mag0 = (int)((grid >> (lane * 8)) & 0xff) >> 3;
		const int scale_code = (int)bits.read(17, 4);
		const int scale = scale_code < 8 ? scale_code + 1 : (scale_code < 12 ? (scale_code - 3) * 2 : (scale_code - 7) * 4);
		const int mag = ccv_min(mag0 * scale, 127);
		const int negative = (int)bits.read(9 + lane, 1);
		return negative ? -mag : mag;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)
	{
		const int subgroup = lane >> 3;
		const int grid_lane = lane & 7;
		const int grid_index = (int)bits.read(subgroup * 8, 8);
		const uint16_t grid = ccv_nnc_8i_rowwise_packed_iq2xxs_grid[grid_index];
		const int mag0 = 1 + (int)((grid >> (grid_lane * 2)) & 3) * 2;
		const int scale_code = (int)bits.read(60, 4);
		const int scale = scale_code < 8 ? scale_code + 1 : (scale_code < 12 ? (scale_code - 3) * 2 : (scale_code - 7) * 4);
		const int mag = ccv_min(mag0 * scale, 127);
		const int sign_code = (int)bits.read(32 + subgroup * 7, 7);
		const uint8_t signs = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[sign_code];
		return (signs & (1u << grid_lane)) ? -mag : mag;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S)
	{
		const int subgroup = lane >> 2;
		const int grid_lane = lane & 3;
		const int grid_index = (int)bits.read(subgroup * 9, 9);
		const uint32_t grid = ccv_nnc_8i_rowwise_packed_iq3s_grid[grid_index];
		const int mag0 = (int)((grid >> (grid_lane * 8)) & 0xff);
		const int scale = (int)bits.read(52, 4) + 1;
		const int mag = ccv_min(mag0 * scale, 127);
		const int negative = (int)bits.read(36 + lane, 1);
		return negative ? -mag : mag;
	}
	const int subgroup = lane >> 2;
	const int grid_lane = lane & 3;
	const int grid_index = (int)bits.read(subgroup * 8, 8);
	const uint32_t grid = ccv_nnc_8i_rowwise_packed_iq3xxs_grid[grid_index];
	const int mag0 = (int)((grid >> (grid_lane * 8)) & 0xff) >> 2;
	const int scale = (int)bits.read(24, 4) + 1;
	const int mag = ccv_min(mag0 * scale, 127);
	const int negative = (int)bits.read(16 + lane, 1);
	return negative ? -mag : mag;
}

template<int FORMAT, typename BITS>
static __device__ __forceinline__ int _ccv_nnc_8i_rowwise_x_decode_q_k(const BITS& bits, const int lane, const unsigned int mask, const int first_lane)
{
	int params = 0;
	if (lane == 0)
	{
		if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K)
			params = (int)bits.read(80, 3) | ((int)bits.read(83, 5) << 8);
		else if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K)
			params = (int)bits.read(48, 2) | ((int)bits.read(50, 2) << 8);
		else if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K)
			params = (int)bits.read(64, 4) | ((int)bits.read(68, 4) << 8);
		else if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K)
			params = (int)bits.read(48, 5) | ((int)bits.read(53, 3) << 8);
		else
			params = (int)bits.read(32, 6) | ((int)bits.read(38, 4) << 8);
	}
	params = __shfl_sync(mask, params, first_lane);
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K)
		return ((int)bits.read(lane * 5, 5) - 16) * ((params & 0xff) + 1) + ((params >> 8) - 16);
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K)
	{
		const int uq = (int)bits.read(lane * 6, 6);
		return ((uq ^ 32) - 32) * ((params & 0xff) + 1) + (((params >> 8) ^ 2) - 2);
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K)
		return ((int)bits.read(lane * 4, 4) - 8) * ((params & 0xff) + 1) + ((params >> 8) - 8);
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K)
		return ((int)bits.read(lane * 3, 3) - 4) * ((params & 0xff) + 1) + ((params >> 8) - 4) * 2;
	return (int)bits.read(lane * 2, 2) * ((params & 0xff) + 1) - (params >> 8) * 8;
}

// IQ grid lookups, from global memory by default. The H256 decode reads a
// shared-memory copy instead, where scattered lookups cost fewer wavefronts.
typedef struct {
	__device__ __forceinline__ uint64_t iq2s(const int i) const { return ccv_nnc_8i_rowwise_packed_iq2s_grid[i]; }
	__device__ __forceinline__ uint64_t iq2xs(const int i) const { return ccv_nnc_8i_rowwise_packed_iq2xs_grid[i]; }
	__device__ __forceinline__ uint16_t iq2xxs(const int i) const { return ccv_nnc_8i_rowwise_packed_iq2xxs_grid[i]; }
	__device__ __forceinline__ uint8_t iq2xxs_ksigns(const int i) const { return ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[i]; }
	__device__ __forceinline__ uint32_t iq3s(const int i) const { return ccv_nnc_8i_rowwise_packed_iq3s_grid[i]; }
	__device__ __forceinline__ uint32_t iq3xxs(const int i) const { return ccv_nnc_8i_rowwise_packed_iq3xxs_grid[i]; }
} _ccv_nnc_8i_rowwise_x_global_grids_t;

template<int FORMAT, typename BITS, typename GRIDS = _ccv_nnc_8i_rowwise_x_global_grids_t>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_decode_iq_oct(const BITS& bits, const int lane, int* const values, const GRIDS& grids = GRIDS())
{
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
	{
		const int grid_index = (int)bits.read(lane >= 8 ? 10 : 0, 10);
		const uint64_t grid = grids.iq2s(grid_index);
		const int scale = (int)bits.read(36, 6) + 1;
		const int signs = (int)bits.read(20 + lane, 8);
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int mag0 = (int)((grid >> (i * 8)) & 0xff) >> 3;
			const int mag = ccv_min(mag0 * scale, 127);
			values[i] = signs & (1 << i) ? -mag : mag;
		}
		return;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS)
	{
		const int grid_index = (int)bits.read(0, 9);
		const uint64_t grid = grids.iq2xs(grid_index);
		const int scale_code = (int)bits.read(17, 4);
		const int scale = scale_code < 8 ? scale_code + 1 : (scale_code < 12 ? (scale_code - 3) * 2 : (scale_code - 7) * 4);
		const int signs = (int)bits.read(9, 8);
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int mag0 = (int)((grid >> (i * 8)) & 0xff) >> 3;
			const int mag = ccv_min(mag0 * scale, 127);
			values[i] = signs & (1 << i) ? -mag : mag;
		}
		return;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)
	{
		const int subgroup = lane >> 3;
		const int grid_index = (int)bits.read(subgroup * 8, 8);
		const uint16_t grid = grids.iq2xxs(grid_index);
		const int scale_code = (int)bits.read(60, 4);
		const int scale = scale_code < 8 ? scale_code + 1 : (scale_code < 12 ? (scale_code - 3) * 2 : (scale_code - 7) * 4);
		const int sign_code = (int)bits.read(32 + subgroup * 7, 7);
		const int signs = grids.iq2xxs_ksigns(sign_code);
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const int mag0 = 1 + (int)((grid >> (i * 2)) & 3) * 2;
			const int mag = ccv_min(mag0 * scale, 127);
			values[i] = signs & (1 << i) ? -mag : mag;
		}
		return;
	}
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S)
	{
		const int subgroup = lane >> 2;
		const uint32_t grid0 = grids.iq3s((int)bits.read(subgroup * 9, 9));
		const uint32_t grid1 = grids.iq3s((int)bits.read((subgroup + 1) * 9, 9));
		const int scale = (int)bits.read(52, 4) + 1;
		const int signs = (int)bits.read(36 + lane, 8);
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const uint32_t grid = i < 4 ? grid0 : grid1;
			const int mag0 = (int)((grid >> ((i & 3) * 8)) & 0xff);
			const int mag = ccv_min(mag0 * scale, 127);
			values[i] = signs & (1 << i) ? -mag : mag;
		}
		return;
	}
	const uint32_t grid0 = grids.iq3xxs((int)bits.read(0, 8));
	const uint32_t grid1 = grids.iq3xxs((int)bits.read(8, 8));
	const int scale = (int)bits.read(24, 4) + 1;
	const int signs = (int)bits.read(16, 8);
#pragma unroll
	for (int i = 0; i < 8; i++)
	{
		const uint32_t grid = i < 4 ? grid0 : grid1;
		const int mag0 = (int)((grid >> ((i & 3) * 8)) & 0xff) >> 2;
		const int mag = ccv_min(mag0 * scale, 127);
		values[i] = signs & (1 << i) ? -mag : mag;
	}
}

template<int FORMAT, typename BITS, typename GRIDS = _ccv_nnc_8i_rowwise_x_global_grids_t>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_decode_oct(const BITS& bits, const int lane, int* const values, const GRIDS& grids = GRIDS())
{
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ||
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ||
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)
		_ccv_nnc_8i_rowwise_x_decode_iq_oct<FORMAT>(bits, lane, values, grids);
	else {
#pragma unroll
		for (int i = 0; i < 8; i++)
			values[i] = _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, lane + i);
	}
}

static __device__ __forceinline__ __half _ccv_nnc_8i_rowwise_x_mul(const __half scale, const int q)
{
	return __hmul(scale, __int2half_rn(q));
}

static __device__ __forceinline__ __nv_bfloat16 _ccv_nnc_8i_rowwise_x_mul(const __nv_bfloat16 scale, const int q)
{
	return __hmul(scale, __int2bfloat16_rn(q));
}

static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_x_mul(const float scale, const int q)
{
	return q * scale;
}

static __device__ __forceinline__ double _ccv_nnc_8i_rowwise_x_mul(const double scale, const int q)
{
	return q * scale;
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store2(__half* const output, const __half a, const __half b)
{
	*(__half2*)output = __halves2half2(a, b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store2(__nv_bfloat16* const output, const __nv_bfloat16 a, const __nv_bfloat16 b)
{
	*(__nv_bfloat162*)output = __halves2bfloat162(a, b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store2(float* const output, const float a, const float b)
{
	*(float2*)output = make_float2(a, b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store2(double* const output, const double a, const double b)
{
	*(double2*)output = make_double2(a, b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store4(__half* const output, const __half a, const __half b, const __half c, const __half d)
{
	union {
		__half2 h;
		uint32_t u;
	} x, y;
	x.h = __halves2half2(a, b);
	y.h = __halves2half2(c, d);
	*(uint2*)output = make_uint2(x.u, y.u);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store4(__nv_bfloat16* const output, const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c, const __nv_bfloat16 d)
{
	union {
		__nv_bfloat162 h;
		uint32_t u;
	} x, y;
	x.h = __halves2bfloat162(a, b);
	y.h = __halves2bfloat162(c, d);
	*(uint2*)output = make_uint2(x.u, y.u);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store4(float* const output, const float a, const float b, const float c, const float d)
{
	*(float4*)output = make_float4(a, b, c, d);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store4(double* const output, const double a, const double b, const double c, const double d)
{
	*(double4*)output = make_double4(a, b, c, d);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store8(__half* const output, const __half* const values)
{
	union {
		__half2 h;
		uint32_t u;
	} x[4];
#pragma unroll
	for (int i = 0; i < 4; i++)
		x[i].h = __halves2half2(values[i * 2], values[i * 2 + 1]);
	*(uint4*)output = make_uint4(x[0].u, x[1].u, x[2].u, x[3].u);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store8(__nv_bfloat16* const output, const __nv_bfloat16* const values)
{
	union {
		__nv_bfloat162 h;
		uint32_t u;
	} x[4];
#pragma unroll
	for (int i = 0; i < 4; i++)
		x[i].h = __halves2bfloat162(values[i * 2], values[i * 2 + 1]);
	*(uint4*)output = make_uint4(x[0].u, x[1].u, x[2].u, x[3].u);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store8(float* const output, const float* const values)
{
	*(float4*)output = make_float4(values[0], values[1], values[2], values[3]);
	*(float4*)(output + 4) = make_float4(values[4], values[5], values[6], values[7]);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_store8(double* const output, const double* const values)
{
	*(double4*)output = make_double4(values[0], values[1], values[2], values[3]);
	*(double4*)(output + 4) = make_double4(values[4], values[5], values[6], values[7]);
}

template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp(const size_t count, const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / row_length;
		const size_t col = i - row * row_length;
		const size_t group = row * groups_per_row + col / group_size;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			group * group_bits,
		};
		const int q = _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, col % group_size);
		output[i] = _ccv_nnc_8i_rowwise_x_mul(scales[row], q);
	}
}

template<int FORMAT, typename NUM>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_dequantize_oct(const size_t row, const size_t col, const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	const size_t group = row * groups_per_row + col / group_size;
	const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
		input,
		group * group_bits,
	};
	const int group_lane = col % group_size;
	const NUM scale = scales[row];
	int decoded[8];
	_ccv_nnc_8i_rowwise_x_decode_oct<FORMAT>(bits, group_lane, decoded);
	NUM values[8];
#pragma unroll
	for (int i = 0; i < 8; i++)
		values[i] = _ccv_nnc_8i_rowwise_x_mul(scale, decoded[i]);
	_ccv_nnc_8i_rowwise_x_store8(output + row * row_length + col, values);
}

template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d_oct(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const size_t row = blockIdx.y;
	const size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	if (col < row_length)
		_ccv_nnc_8i_rowwise_x_dequantize_oct<FORMAT>(row, col, row_length, groups_per_row, input, scales, output);
}

template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d_quad(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	const size_t row = blockIdx.y;
	const size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
	if (col < row_length)
	{
		const size_t group = row * groups_per_row + col / group_size;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			group * group_bits,
		};
		const int group_lane = col % group_size;
		const NUM scale = scales[row];
		const NUM a = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane));
		const NUM b = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane + 1));
		const NUM c = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane + 2));
		const NUM d = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane + 3));
		_ccv_nnc_8i_rowwise_x_store4(output + row * row_length + col, a, b, c, d);
	}
}

template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d_pair(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	const size_t row = blockIdx.y;
	const size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	if (col < row_length)
	{
		const size_t group = row * groups_per_row + col / group_size;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			group * group_bits,
		};
		const int group_lane = col % group_size;
		const NUM scale = scales[row];
		const NUM a = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane));
		const NUM b = _ccv_nnc_8i_rowwise_x_mul(scale, _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane + 1));
		_ccv_nnc_8i_rowwise_x_store2(output + row * row_length + col, a, b);
	}
}

template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	const size_t row = blockIdx.y;
	const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < row_length)
	{
		const size_t group = row * groups_per_row + col / group_size;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			group * group_bits,
		};
		const int group_lane = col % group_size;
		int q;
		if (FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K)
		{
			const unsigned int mask = __activemask();
			const int warp_lane = threadIdx.x & 31;
			q = _ccv_nnc_8i_rowwise_x_decode_q_k<FORMAT>(bits, group_lane, mask, warp_lane - group_lane);
		} else
			q = _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane);
		output[row * row_length + col] = _ccv_nnc_8i_rowwise_x_mul(scales[row], q);
	}
}

// CCV_NNC_QX_8I_ROWWISE_HADAMARD_256 stores W R with R256 = H4^tensor4 / 16, which is
// its own inverse: the logical-weight decode rotates each block of 256 decoded integers
// once more (exactly, in integers, with the 1/16 folded into the row scale), and GEMM
// can rotate the activations instead. Each H4 stage is y[i] = sum(v) - 2 * v[3 - i].
template<typename T>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_h4(T* const v, const int stride)
{
	const T a = v[0], b = v[stride], c = v[stride * 2], d = v[stride * 3];
	const T s = a + b + c + d;
	v[0] = s - 2 * d;
	v[stride] = s - 2 * c;
	v[stride * 2] = s - 2 * b;
	v[stride * 3] = s - 2 * a;
}

// The 1/16 of R256 is exact in the row scale, so (scale / 16) * q rounds once like the CPU reference.
static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_x_h256_scale(const __half scale)
{
	return __half2float(scale) * 0.0625f;
}

static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_x_h256_scale(const __nv_bfloat16 scale)
{
	return __bfloat162float(scale) * 0.0625f;
}

static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_x_h256_scale(const float scale)
{
	return scale * 0.0625f;
}

static __device__ __forceinline__ double _ccv_nnc_8i_rowwise_x_h256_scale(const double scale)
{
	return scale * 0.0625;
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_h256_store2(__half* const output, const float scale, const int a, const int b)
{
	*(__half2*)output = __floats2half2_rn(scale * a, scale * b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_h256_store2(__nv_bfloat16* const output, const float scale, const int a, const int b)
{
	*(__nv_bfloat162*)output = __floats2bfloat162_rn(scale * a, scale * b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_h256_store2(float* const output, const float scale, const int a, const int b)
{
	*(float2*)output = make_float2(scale * a, scale * b);
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_h256_store2(double* const output, const double scale, const int a, const int b)
{
	*(double2*)output = make_double2(scale * a, scale * b);
}

template<int WORDS>
struct _ccv_nnc_8i_rowwise_x_register_bits_t {
	uint32_t w[WORDS];
	int base;

	__device__ __forceinline__ uint32_t read(const int offset, const int bits) const
	{
		const int o = base + offset;
		return __funnelshift_r(w[o >> 5], w[(o >> 5) + 1], o & 31) & ((1u << bits) - 1);
	}
};

#define CCV_NNC_8I_ROWWISE_X_H256_TEAMS (32)
// One 128-word tile per team, padded so the four teams of a warp land on
// distinct banks in the column pass.
#define CCV_NNC_8I_ROWWISE_X_H256_TILE (128 + 8)

// Shared-memory copy of an IQ codec's grid (IQ2_XXS keeps its ksigns after the grid).
template<int FORMAT>
static __host__ __device__ constexpr int _ccv_nnc_8i_rowwise_x_grid_size()
{
	return FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 1024 * 8 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 512 * 8 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 256 * 2 + 128 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 512 * 4 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS ? 256 * 4 : 0;
}

typedef struct {
	const uint8_t* table;
	__device__ __forceinline__ uint64_t iq2s(const int i) const { return ((const uint64_t*)table)[i]; }
	__device__ __forceinline__ uint64_t iq2xs(const int i) const { return ((const uint64_t*)table)[i]; }
	__device__ __forceinline__ uint16_t iq2xxs(const int i) const { return ((const uint16_t*)table)[i]; }
	__device__ __forceinline__ uint8_t iq2xxs_ksigns(const int i) const { return table[256 * 2 + i]; }
	__device__ __forceinline__ uint32_t iq3s(const int i) const { return ((const uint32_t*)table)[i]; }
	__device__ __forceinline__ uint32_t iq3xxs(const int i) const { return ((const uint32_t*)table)[i]; }
} _ccv_nnc_8i_rowwise_x_shared_grids_t;

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_set_grid_table(_ccv_nnc_8i_rowwise_x_global_grids_t& grids, const uint8_t* const table)
{
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_set_grid_table(_ccv_nnc_8i_rowwise_x_shared_grids_t& grids, const uint8_t* const table)
{
	grids.table = table;
}

template<int FORMAT>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_load_grid(uint8_t* const table)
{
	const void* const grid =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? (const void*)ccv_nnc_8i_rowwise_packed_iq2s_grid :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? (const void*)ccv_nnc_8i_rowwise_packed_iq2xs_grid :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? (const void*)ccv_nnc_8i_rowwise_packed_iq2xxs_grid :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? (const void*)ccv_nnc_8i_rowwise_packed_iq3s_grid : (const void*)ccv_nnc_8i_rowwise_packed_iq3xxs_grid;
	const int grid_words = (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 256 * 2 : _ccv_nnc_8i_rowwise_x_grid_size<FORMAT>()) / 4;
	for (int i = threadIdx.x; i < grid_words; i += blockDim.x)
		((uint32_t*)table)[i] = ((const uint32_t*)grid)[i];
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS)
		for (int i = threadIdx.x; i < 128; i += blockDim.x)
			table[256 * 2 + i] = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[i];
}

// Decode 16 contiguous values starting at col (a multiple of 16) and apply the
// two H4 stages over index bits 0-3.
template<int FORMAT, typename GRIDS>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_decode_h256_16(const uint8_t* const input, const size_t row, const size_t col, const size_t groups_per_row, const GRIDS& grids, int* const v)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	// Load the packed bits once (two groups for 8-value groups, the whole group
	// for 32-value groups) as aligned words, then shift them to start at bit 0.
	// Words past the needed bits are never read, so this stays within the payload.
	const int span = group_size == 8 ? group_bits * 2 : group_bits;
	const int word_count = (span + 31) / 32;
	const size_t bit_offset = (row * groups_per_row + col / group_size) * group_bits;
	const int shift = bit_offset & 31;
	const uint32_t* const words = (const uint32_t*)(input + ((bit_offset >> 5) << 2));
	uint32_t raw[word_count + 1];
#pragma unroll
	for (int i = 0; i < word_count + 1; i++)
		raw[i] = i * 32 < shift + span ? words[i] : 0;
	_ccv_nnc_8i_rowwise_x_register_bits_t<word_count + 1> bits;
#pragma unroll
	for (int i = 0; i < word_count; i++)
		bits.w[i] = __funnelshift_r(raw[i], raw[i + 1], shift);
	bits.w[word_count] = 0;
#pragma unroll
	for (int i = 0; i < 16; i += 8)
	{
		bits.base = group_size == 8 ? (i / 8) * group_bits : 0;
		_ccv_nnc_8i_rowwise_x_decode_oct<FORMAT>(bits, group_size == 8 ? 0 : group_size == 16 ? i : (int)(col % group_size) + i, v + i, grids);
	}
#pragma unroll
	for (int i = 0; i < 16; i += 4)
		_ccv_nnc_8i_rowwise_x_h4(v + i, 1);
#pragma unroll
	for (int i = 0; i < 4; i++)
		_ccv_nnc_8i_rowwise_x_h4(v + i, 4);
}

// A team of 8 threads handles one 256-value block. In the row pass, thread r
// decodes values 32r .. 32r + 31 and applies the H4 stages over index bits
// 0-3. Every codec decodes to |q| <= 192 (Q2_K), so after those stages
// |v| <= 16 * 192 and neighbors (2p, 2p + 1) pack into
// one int16x2 word for the shared-memory transpose. In the column pass, thread
// t holds pairs 8k + t (k = 0..15), i.e. index bits 4-7 for the remaining
// stages, and stores each pair directly; a team covers 32 contiguous bytes
// for every k. IQ codecs run persistent CTAs that copy their grid into shared
// memory once.
template<int FORMAT, typename NUM>
static __global__ void __launch_bounds__(CCV_NNC_8I_ROWWISE_X_H256_TEAMS * 8, 3) _ccv_nnc_dequantize_8i_rowwise_x_fp_h256(const uint32_t block_count, const uint32_t blocks_per_row, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
{
	__shared__ __align__(16) uint32_t shared[CCV_NNC_8I_ROWWISE_X_H256_TEAMS * CCV_NNC_8I_ROWWISE_X_H256_TILE];
	__shared__ __align__(16) uint8_t grid_table[_ccv_nnc_8i_rowwise_x_grid_size<FORMAT>() > 0 ? _ccv_nnc_8i_rowwise_x_grid_size<FORMAT>() : 16];
	typedef typename std::conditional<(_ccv_nnc_8i_rowwise_x_grid_size<FORMAT>() > 0), _ccv_nnc_8i_rowwise_x_shared_grids_t, _ccv_nnc_8i_rowwise_x_global_grids_t>::type grids_t;
	grids_t grids;
	_ccv_nnc_8i_rowwise_x_set_grid_table(grids, grid_table);
	if (_ccv_nnc_8i_rowwise_x_grid_size<FORMAT>() > 0)
	{
		_ccv_nnc_8i_rowwise_x_load_grid<FORMAT>(grid_table);
		__syncthreads();
	}
	const int team = threadIdx.x >> 3;
	const int t = threadIdx.x & 7;
	uint32_t* const tile = shared + team * CCV_NNC_8I_ROWWISE_X_H256_TILE;
	for (uint32_t block_start = blockIdx.x * CCV_NNC_8I_ROWWISE_X_H256_TEAMS; block_start < block_count; block_start += gridDim.x * CCV_NNC_8I_ROWWISE_X_H256_TEAMS)
	{
		const uint32_t block = block_start + team;
		const int active = block < block_count;
		const uint32_t row = active ? block / blocks_per_row : 0;
		const uint32_t col = (block - row * blocks_per_row) * 256;
		if (active)
		{
			int v[32];
			_ccv_nnc_8i_rowwise_x_decode_h256_16<FORMAT>(input, row, col + t * 32, groups_per_row, grids, v);
			_ccv_nnc_8i_rowwise_x_decode_h256_16<FORMAT>(input, row, col + t * 32 + 16, groups_per_row, grids, v + 16);
			// Pair q = 4c + j of thread t sits at word 16t + 4(c ^ (t >> 1)) + j.
#pragma unroll
			for (int c = 0; c < 4; c++)
				*(uint4*)(tile + t * 16 + ((c ^ (t >> 1)) << 2)) = make_uint4(
					__byte_perm(v[c * 8], v[c * 8 + 1], 0x5410),
					__byte_perm(v[c * 8 + 2], v[c * 8 + 3], 0x5410),
					__byte_perm(v[c * 8 + 4], v[c * 8 + 5], 0x5410),
					__byte_perm(v[c * 8 + 6], v[c * 8 + 7], 0x5410));
		}
		__syncwarp();
		if (active)
		{
			int even[16], odd[16];
#pragma unroll
			for (int k = 0; k < 16; k++)
			{
				const uint32_t w = tile[(k >> 1) * 16 + ((((k & 1) * 2 + (t >> 2)) ^ (k >> 2)) << 2) + (t & 3)];
				even[k] = (int)(w << 16) >> 16;
				odd[k] = (int)w >> 16;
			}
#pragma unroll
			for (int i = 0; i < 16; i += 4)
			{
				_ccv_nnc_8i_rowwise_x_h4(even + i, 1);
				_ccv_nnc_8i_rowwise_x_h4(odd + i, 1);
			}
#pragma unroll
			for (int i = 0; i < 4; i++)
			{
				_ccv_nnc_8i_rowwise_x_h4(even + i, 4);
				_ccv_nnc_8i_rowwise_x_h4(odd + i, 4);
			}
			const auto scale = _ccv_nnc_8i_rowwise_x_h256_scale(scales[row]);
			NUM* const out = output + (size_t)block * 256 + t * 2;
#pragma unroll
			for (int k = 0; k < 16; k++)
				_ccv_nnc_8i_rowwise_x_h256_store2(out + k * 16, scale, even[k], odd[k]);
		}
		// The next block reuses the tile.
		__syncwarp();
	}
}

static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8(const __half* const input, float* const values)
{
	const uint4 u = *(const uint4*)input;
	const __half2* const h = (const __half2*)&u;
#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		const float2 f = __half22float2(h[i]);
		values[i * 2] = f.x;
		values[i * 2 + 1] = f.y;
	}
}

static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8(const __nv_bfloat16* const input, float* const values)
{
	const uint4 u = *(const uint4*)input;
	const __nv_bfloat162* const h = (const __nv_bfloat162*)&u;
#pragma unroll
	for (int i = 0; i < 4; i++)
	{
		const float2 f = __bfloat1622float2(h[i]);
		values[i * 2] = f.x;
		values[i * 2 + 1] = f.y;
	}
}

static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8(const float* const input, float* const values)
{
	const float4 a = *(const float4*)input;
	const float4 b = *(const float4*)(input + 4);
	values[0] = a.x, values[1] = a.y, values[2] = a.z, values[3] = a.w;
	values[4] = b.x, values[5] = b.y, values[6] = b.z, values[7] = b.w;
}

// The activations may start at any element offset, so this path reads one element at a time.
static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8_unaligned(const __half* const input, float* const values)
{
#pragma unroll
	for (int i = 0; i < 8; i++)
		values[i] = __half2float(input[i]);
}

static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8_unaligned(const __nv_bfloat16* const input, float* const values)
{
#pragma unroll
	for (int i = 0; i < 8; i++)
		values[i] = __bfloat162float(input[i]);
}

static __device__ __forceinline__ void _ccv_nnc_hadamard_256_load8_unaligned(const float* const input, float* const values)
{
#pragma unroll
	for (int i = 0; i < 8; i++)
		values[i] = input[i];
}

template<typename NUM>
static __device__ __forceinline__ NUM _ccv_nnc_hadamard_256_from_float(const float value);

template<>
__device__ __forceinline__ __half _ccv_nnc_hadamard_256_from_float<__half>(const float value)
{
	return __float2half_rn(value);
}

template<>
__device__ __forceinline__ __nv_bfloat16 _ccv_nnc_hadamard_256_from_float<__nv_bfloat16>(const float value)
{
	return __float2bfloat16_rn(value);
}

template<>
__device__ __forceinline__ float _ccv_nnc_hadamard_256_from_float<float>(const float value)
{
	return value;
}

// Apply R256 to one block of 256 floating-point values with one warp, e.g. GEMM
// activations against H256 weights. Index bits 0-2 are in registers and bits
// 3-7 are lane bits 0-4; reversing an H4 digit flips both of its bits. Blocks are
// 256 elements apart, so every lane shares the input's alignment, given by aligned.
template<typename NUM>
static __device__ __forceinline__ void _ccv_nnc_hadamard_256_warp(const NUM* const input, NUM* const output, const int aligned)
{
	const int lane = threadIdx.x & 31;
	float v[8];
	if (aligned)
		_ccv_nnc_hadamard_256_load8(input + lane * 8, v);
	else
		_ccv_nnc_hadamard_256_load8_unaligned(input + lane * 8, v);
	_ccv_nnc_8i_rowwise_x_h4(v, 1);
	_ccv_nnc_8i_rowwise_x_h4(v + 4, 1);
	// Bits 2 / 3: register bit 2 and lane bit 0.
#pragma unroll
	for (int j = 0; j < 4; j++)
	{
		const float p = v[j] + v[j + 4];
		const float s = p + __shfl_xor_sync(0xffffffff, p, 1);
		const float a = __shfl_xor_sync(0xffffffff, v[j + 4], 1);
		const float b = __shfl_xor_sync(0xffffffff, v[j], 1);
		v[j] = s - 2 * a;
		v[j + 4] = s - 2 * b;
	}
	// Bits 4 / 5 and 6 / 7: lane bits 1 / 2 and 3 / 4.
#pragma unroll
	for (int mask = 2; mask <= 8; mask *= 4)
	{
#pragma unroll
		for (int i = 0; i < 8; i++)
		{
			const float s1 = v[i] + __shfl_xor_sync(0xffffffff, v[i], mask);
			const float s = s1 + __shfl_xor_sync(0xffffffff, s1, mask * 2);
			v[i] = s - 2 * __shfl_xor_sync(0xffffffff, v[i], mask * 3);
		}
	}
	NUM values[8];
#pragma unroll
	for (int i = 0; i < 8; i++)
		values[i] = _ccv_nnc_hadamard_256_from_float<NUM>(v[i] * 0.0625f);
	_ccv_nnc_8i_rowwise_x_store8(output + lane * 8, values);
}

template<typename NUM>
static __global__ void _ccv_nnc_hadamard_256(const size_t block_count, const NUM* const input, NUM* const output, const int aligned)
{
	const size_t block = (size_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
	if (block < block_count)
		_ccv_nnc_hadamard_256_warp(input + block * 256, output + block * 256, aligned);
}

// Decode the stored (rotated) weights and rotate the activations in one launch.
// The first activation_rows grid rows hold the activation warps, so they are
// scheduled first and overlap with the weight decode instead of trailing it.
template<int FORMAT, typename NUM>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d_oct_hadamard_256(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output, const uint32_t activation_rows, const size_t activation_block_count, const NUM* const activation, NUM* const activation_output, const int activation_aligned)
{
	if (blockIdx.y < activation_rows)
	{
		const size_t block = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x / 32) + (threadIdx.x >> 5);
		if (block < activation_block_count)
			_ccv_nnc_hadamard_256_warp(activation + block * 256, activation_output + block * 256, activation_aligned);
		return;
	}
	const size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	if (col < row_length)
		_ccv_nnc_8i_rowwise_x_dequantize_oct<FORMAT>(blockIdx.y - activation_rows, col, row_length, groups_per_row, input, scales, output);
}

template<typename NUM>
static void _ccv_nnc_dequantize_8i_rowwise_x_fp_launch(const int format, const size_t output_length, const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output, cudaStream_t stream)
{
#define DISPATCH_FORMAT(case_format) \
	case case_format: \
		if (format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256) \
			_ccv_nnc_dequantize_8i_rowwise_x_fp_h256<case_format><<<_ccv_nnc_8i_rowwise_x_grid_size<case_format>() > 0 ? ccv_min(h256_ctas, (size_t)ccv_nnc_gpu_device_props().multi_processor_count * 4) : h256_ctas, CCV_NNC_8I_ROWWISE_X_H256_TEAMS * 8, 0, stream>>>(output_length / 256, row_length / 256, groups_per_row, input, scales, output); \
		else if (output_length / row_length <= 65535 && row_length % 8 == 0 && \
			(sizeof(NUM) < 8 || case_format == CCV_NNC_QX_8I_ROWWISE_IQ2_S || case_format == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || case_format == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS || case_format == CCV_NNC_QX_8I_ROWWISE_IQ3_S || case_format == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)) \
			_ccv_nnc_dequantize_8i_rowwise_x_fp_2d_oct<case_format><<<dim3((row_length / 8 + 255) / 256, output_length / row_length), 256, 0, stream>>>(row_length, groups_per_row, input, scales, output); \
		else if (output_length / row_length <= 65535 && row_length % 4 == 0) \
			_ccv_nnc_dequantize_8i_rowwise_x_fp_2d_quad<case_format><<<dim3((row_length / 4 + 255) / 256, output_length / row_length), 256, 0, stream>>>(row_length, groups_per_row, input, scales, output); \
		else if (output_length / row_length <= 65535 && row_length % 2 == 0) \
			_ccv_nnc_dequantize_8i_rowwise_x_fp_2d_pair<case_format><<<dim3((row_length / 2 + 255) / 256, output_length / row_length), 256, 0, stream>>>(row_length, groups_per_row, input, scales, output); \
		else if (output_length / row_length <= 65535) \
			_ccv_nnc_dequantize_8i_rowwise_x_fp_2d<case_format><<<dim3((row_length + 255) / 256, output_length / row_length), 256, 0, stream>>>(row_length, groups_per_row, input, scales, output); \
		else \
			_ccv_nnc_dequantize_8i_rowwise_x_fp<case_format><<<ccv_min((output_length + 255) / 256, 4096), 256, 0, stream>>>(output_length, row_length, groups_per_row, input, scales, output); \
		break
	const size_t h256_ctas = (output_length / 256 + CCV_NNC_8I_ROWWISE_X_H256_TEAMS - 1) / CCV_NNC_8I_ROWWISE_X_H256_TEAMS;
	switch (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK)
	{
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q5_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q6_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q4_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q3_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q2_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
		default:
			assert(0);
	}
#undef DISPATCH_FORMAT
}

// Decode a row-wise-x payload into its int8 codes, 8 codes per thread, without applying the row scales (and
// for H256, without undoing R). The encoders keep every code within [-127, 127], thus the codes are exact; the
// clamp only keeps a malformed payload from wrapping around.
template<int FORMAT>
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_8i(const size_t oct_count, const size_t octs_per_row, const size_t groups_per_row, const uint8_t* const input, uint2* const output)
{
	const int group_size =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 32 :
		(FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS || FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS) ? 8 : 16;
	const int group_bits =
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q5_K ? 88 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q6_K ? 52 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q4_K ? 72 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q3_K ? 56 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_Q2_K ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ? 42 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ? 21 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ? 64 :
		FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ? 56 : 28;
	CUDA_1D_KERNEL_LOOP(i, oct_count) {
		const size_t row = i / octs_per_row;
		const size_t col = (i - row * octs_per_row) * 8;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			(row * groups_per_row + col / group_size) * group_bits,
		};
		int decoded[8];
		_ccv_nnc_8i_rowwise_x_decode_oct<FORMAT>(bits, col % group_size, decoded);
		uint32_t packed[2] = {};
#pragma unroll
		for (int j = 0; j < 8; j++)
			packed[j >> 2] |= (uint32_t)(uint8_t)(int8_t)ccv_clamp(decoded[j], -127, 127) << ((j & 3) * 8);
		output[i] = make_uint2(packed[0], packed[1]);
	}
}

static __global__ void _ccv_nnc_dequantize_8i_rowwise_f16(const size_t count, const size_t row_length, const int8_t* const input, const __half* const scales, __half* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / row_length;
		output[i] = __float2half_rn((float)input[i] * __half2float(scales[row]));
	}
}

static __global__ void _ccv_nnc_dequantize_8i_rowwise_bf16(const size_t count, const size_t row_length, const int8_t* const input, const __nv_bfloat16* const scales, __nv_bfloat16* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / row_length;
		output[i] = __float2bfloat16((float)input[i] * __bfloat162float(scales[row]));
	}
}

static __global__ void _ccv_nnc_dequantize_8i_rowwise_f32(const size_t count, const size_t row_length, const int8_t* const input, const float* const scales, float* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / row_length;
		output[i] = input[i] * scales[row];
	}
}

static __global__ void _ccv_nnc_dequantize_8i_rowwise_f64(const size_t count, const size_t row_length, const int8_t* const input, const double* const scales, double* const output)
{
	CUDA_1D_KERNEL_LOOP(i, count) {
		const size_t row = i / row_length;
		output[i] = input[i] * scales[row];
	}
}

size_t ccv_nnc_compat_qx_dense_data_size(const ccv_nnc_tensor_param_t params)
{
	assert(CCV_GET_DATA_TYPE(params.datatype) == CCV_QX);
	ccv_nnc_tensor_param_t dense_params = params;
	dense_params.datatype = (params.datatype & 0xff) << 12;
	dense_params.reserved = 0;
	return ccv_nnc_tensor_data_size(dense_params);
}

void ccv_nnc_compat_depalettize(const void* input, const int datatype, const size_t input_length, const int qbits, const int number_in_blocks, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
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
	} else if (datatype == CCV_16BF) {
		if (qbits == 4)
		{
			const int number_in_blocks_2 = number_in_blocks / 2;
			const size_t count = (output_length + 1) / 2;
			if (output_length % 2 == 0)
				_ccv_nnc_q4_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (__nv_bfloat16*)output);
			else
				_ccv_nnc_q4_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_2, (uint8_t*)input, (__nv_bfloat16*)output);
		} else if (qbits == 5) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q5_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__nv_bfloat16*)output);
			else
				_ccv_nnc_q5_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__nv_bfloat16*)output);
		} else if (qbits == 6) {
			const int number_in_blocks_4 = number_in_blocks / 4;
			const size_t count = (output_length + 3) / 4;
			if (output_length % 4 == 0)
			{
				if (number_in_blocks % (1024 * 4) == 0 && output_length % number_in_blocks == 0)
				{
					const int num_blocks = output_length / number_in_blocks;
					const int repeat_4 = number_in_blocks / (1024 * 4);
					_ccv_nnc_q6_fast_s4<1, __nv_bfloat16><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks_4, (uint8_t*)input, (__nv_bfloat16*)output);
				} else
					_ccv_nnc_q6_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (__nv_bfloat16*)output);
			} else
				_ccv_nnc_q6_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_4, (uint8_t*)input, (__nv_bfloat16*)output);
		} else if (qbits == 7) {
			const int number_in_blocks_8 = number_in_blocks / 8;
			const size_t count = (output_length + 7) / 8;
			if (output_length % 8 == 0)
				_ccv_nnc_q7_fast<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__nv_bfloat16*)output);
			else
				_ccv_nnc_q7_slow<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, output_length, number_in_blocks, number_in_blocks_8, (uint8_t*)input, (__nv_bfloat16*)output);
		} else {
			if ((number_in_blocks % (1024 * 4 * 2)) == 0 && (output_length % number_in_blocks) == 0)
			{
				const int num_blocks = output_length / number_in_blocks;
				const int repeat_4 = number_in_blocks / (1024 * 4 * 2);
				_ccv_nnc_q8_fast_s4<2, __nv_bfloat16><<<dim3(repeat_4, num_blocks, 1), 1024, 0, stream>>>(number_in_blocks, (uint8_t*)input, (__nv_bfloat16*)output);
			} else
				_ccv_nnc_q8_fast<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, number_in_blocks, (uint8_t*)input, (__nv_bfloat16*)output);
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

void ccv_nnc_compat_dequantize_8i_rowwise(const void* input, const int datatype, const size_t input_length, const size_t row_length, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(row_length > 0);
	assert(output_length % row_length == 0);
	const size_t row_count = output_length / row_length;
	const size_t scale_offset = (output_length + 127) & ~(size_t)127;
	const size_t scale_size = row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	assert(input_length >= scale_offset + scale_size);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int8_t* const q = (const int8_t*)input;
	const uint8_t* const u8 = (const uint8_t*)input;
	if (datatype == CCV_16F)
		_ccv_nnc_dequantize_8i_rowwise_f16<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, row_length, q, (const __half*)(u8 + scale_offset), (__half*)output);
	else if (datatype == CCV_16BF)
		_ccv_nnc_dequantize_8i_rowwise_bf16<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, row_length, q, (const __nv_bfloat16*)(u8 + scale_offset), (__nv_bfloat16*)output);
	else if (datatype == CCV_32F)
		_ccv_nnc_dequantize_8i_rowwise_f32<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, row_length, q, (const float*)(u8 + scale_offset), (float*)output);
	else
		_ccv_nnc_dequantize_8i_rowwise_f64<<<CUDA_GET_BLOCKS(output_length), CUDA_NUM_THREADS, 0, stream>>>(output_length, row_length, q, (const double*)(u8 + scale_offset), (double*)output);
}

void ccv_nnc_compat_dequantize_8i_rowwise_x_fp(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(row_length > 0);
	assert(output_length % row_length == 0);
	if (format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256)
	{
		// The H256 decode loads the packed bits as aligned 32-bit words.
		assert(row_length % 256 == 0 && output_length / 256 <= UINT32_MAX - CCV_NNC_8I_ROWWISE_X_H256_TEAMS);
		assert(((uintptr_t)input & 3) == 0);
	}
	const size_t row_count = output_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t group_bits = ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t payload_size = (row_count * groups_per_row * group_bits + 7) / 8;
	const size_t scale_offset = (payload_size + 127) & ~(size_t)127;
	const size_t scale_size = row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	assert(input_length >= scale_offset + scale_size);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const uint8_t* const u8 = (const uint8_t*)input;
	if (datatype == CCV_16F)
		_ccv_nnc_dequantize_8i_rowwise_x_fp_launch(format, output_length, row_length, groups_per_row, u8, (const __half*)(u8 + scale_offset), (__half*)output, stream);
	else if (datatype == CCV_16BF)
		_ccv_nnc_dequantize_8i_rowwise_x_fp_launch(format, output_length, row_length, groups_per_row, u8, (const __nv_bfloat16*)(u8 + scale_offset), (__nv_bfloat16*)output, stream);
	else if (datatype == CCV_32F)
		_ccv_nnc_dequantize_8i_rowwise_x_fp_launch(format, output_length, row_length, groups_per_row, u8, (const float*)(u8 + scale_offset), (float*)output, stream);
	else
		_ccv_nnc_dequantize_8i_rowwise_x_fp_launch(format, output_length, row_length, groups_per_row, u8, (const double*)(u8 + scale_offset), (double*)output, stream);
}

template<typename NUM>
static void _ccv_nnc_dequantize_8i_rowwise_x_fp_hadamard_256_launch(const int format, const size_t output_length, const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output, const size_t activation_length, const NUM* const activation, NUM* const activation_output, cudaStream_t stream)
{
	const size_t rows = output_length / row_length;
	const size_t activation_block_count = activation_length / 256;
	const size_t grid_x = (row_length / 8 + 255) / 256;
	const size_t activation_rows = (activation_block_count + grid_x * 8 - 1) / (grid_x * 8);
	const int activation_aligned = ((uintptr_t)activation & 15) == 0;
	if (rows + activation_rows > 65535)
	{
		// Decode the stored values without undoing R, as the fused kernel does.
		_ccv_nnc_dequantize_8i_rowwise_x_fp_launch(format & ~CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, output_length, row_length, groups_per_row, input, scales, output, stream);
		if (activation_block_count)
			_ccv_nnc_hadamard_256<<<(activation_block_count + 7) / 8, 256, 0, stream>>>(activation_block_count, activation, activation_output, activation_aligned);
		return;
	}
	const dim3 grid(grid_x, rows + activation_rows);
#define DISPATCH_FORMAT(case_format) \
	case case_format: \
		_ccv_nnc_dequantize_8i_rowwise_x_fp_2d_oct_hadamard_256<case_format><<<grid, 256, 0, stream>>>(row_length, groups_per_row, input, scales, output, (uint32_t)activation_rows, activation_block_count, activation, activation_output, activation_aligned); \
		break
	switch (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK)
	{
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q5_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q6_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q4_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q3_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q2_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
		default:
			assert(0);
	}
#undef DISPATCH_FORMAT
}

void ccv_nnc_compat_dequantize_8i_rowwise_x_fp_hadamard_256(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, const void* activation, const size_t activation_length, void* activation_output, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F);
	assert(row_length > 0 && row_length % 256 == 0);
	assert(output_length % row_length == 0);
	assert(activation_length % 256 == 0);
	// The rotated activations are stored 8 values at a time with 16-byte vector stores.
	assert(((uintptr_t)activation_output & 15) == 0);
	const size_t row_count = output_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t group_bits = ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t payload_size = (row_count * groups_per_row * group_bits + 7) / 8;
	const size_t scale_offset = (payload_size + 127) & ~(size_t)127;
	assert(input_length >= scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype));
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const uint8_t* const u8 = (const uint8_t*)input;
	if (datatype == CCV_16F)
		_ccv_nnc_dequantize_8i_rowwise_x_fp_hadamard_256_launch(format, output_length, row_length, groups_per_row, u8, (const __half*)(u8 + scale_offset), (__half*)output, activation_length, (const __half*)activation, (__half*)activation_output, stream);
	else if (datatype == CCV_16BF)
		_ccv_nnc_dequantize_8i_rowwise_x_fp_hadamard_256_launch(format, output_length, row_length, groups_per_row, u8, (const __nv_bfloat16*)(u8 + scale_offset), (__nv_bfloat16*)output, activation_length, (const __nv_bfloat16*)activation, (__nv_bfloat16*)activation_output, stream);
	else
		_ccv_nnc_dequantize_8i_rowwise_x_fp_hadamard_256_launch(format, output_length, row_length, groups_per_row, u8, (const float*)(u8 + scale_offset), (float*)output, activation_length, (const float*)activation, (float*)activation_output, stream);
}

void ccv_nnc_compat_dequantize_8i_rowwise_x(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	// Every group holds a multiple of 8 values, thus 8 codes never straddle a row.
	assert(row_length > 0 && row_length % 8 == 0);
	assert(output_length % row_length == 0);
	assert(((uintptr_t)output & 7) == 0);
	const size_t row_count = output_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t group_bits = ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t payload_size = (row_count * groups_per_row * group_bits + 7) / 8;
	const size_t scale_offset = (payload_size + 127) & ~(size_t)127;
	const size_t scale_size = row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	assert(input_length >= scale_offset + scale_size);
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const uint8_t* const u8 = (const uint8_t*)input;
	const size_t oct_count = output_length / 8;
#define DISPATCH_FORMAT(case_format) \
	case case_format: \
		_ccv_nnc_dequantize_8i_rowwise_x_8i<case_format><<<CUDA_GET_BLOCKS(oct_count), CUDA_NUM_THREADS, 0, stream>>>(oct_count, row_length / 8, groups_per_row, u8, (uint2*)output); \
		break
	switch (format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK)
	{
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q5_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q6_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q4_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q3_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_Q2_K);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ2_XXS);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_S);
		DISPATCH_FORMAT(CCV_NNC_QX_8I_ROWWISE_IQ3_XXS);
		default:
			assert(0);
	}
#undef DISPATCH_FORMAT
	CUDA_ENFORCE(cudaMemcpyAsync((uint8_t*)output + ((output_length + 127) & ~(size_t)127), u8 + scale_offset, scale_size, cudaMemcpyDeviceToDevice, stream));
}

void ccv_nnc_compat_hadamard_256(const void* input, const int datatype, const size_t length, void* output, ccv_nnc_stream_context_t* const stream_context)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F);
	assert(length % 256 == 0);
	// The rotated values are stored 8 at a time with 16-byte vector stores.
	assert(((uintptr_t)output & 15) == 0);
	const size_t block_count = length / 256;
	if (!block_count)
		return;
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int aligned = ((uintptr_t)input & 15) == 0;
	if (datatype == CCV_16F)
		_ccv_nnc_hadamard_256<<<(block_count + 7) / 8, 256, 0, stream>>>(block_count, (const __half*)input, (__half*)output, aligned);
	else if (datatype == CCV_16BF)
		_ccv_nnc_hadamard_256<<<(block_count + 7) / 8, 256, 0, stream>>>(block_count, (const __nv_bfloat16*)input, (__nv_bfloat16*)output, aligned);
	else
		_ccv_nnc_hadamard_256<<<(block_count + 7) / 8, 256, 0, stream>>>(block_count, (const float*)input, (float*)output, aligned);
}

void ccv_nnc_compat_decode_qx(const void* input, const ccv_nnc_tensor_param_t params, void* output, ccv_nnc_stream_context_t* const stream_context)
{
	assert(CCV_GET_DATA_TYPE(params.datatype) == CCV_QX);
	const size_t count = ccv_nnc_tensor_count(params);
	const int datatype = (params.datatype & 0xff) << 12;
	const int subtype = params.datatype & 0xf00;
	if (subtype >= 0x400 && subtype <= 0x800)
		ccv_nnc_compat_depalettize(input, datatype, ccv_nnc_tensor_data_size_without_padding(params), subtype >> 8, params.reserved, output, count, stream_context);
	else if (subtype == CCV_NNC_QX_8I_ROWWISE) {
		const int nd = ccv_nnc_tensor_nd(params.dim);
		ccv_nnc_compat_dequantize_8i_rowwise(input, datatype, ccv_nnc_tensor_data_size_without_padding(params), params.dim[nd - 1], output, count, stream_context);
	} else if (subtype == CCV_NNC_QX_8I_ROWWISE_X) {
		const int nd = ccv_nnc_tensor_nd(params.dim);
		ccv_nnc_compat_dequantize_8i_rowwise_x_fp(input, datatype, ccv_nnc_tensor_data_size_without_padding(params), params.dim[nd - 1], params.reserved, output, count, stream_context);
	} else {
		assert(0);
	}
}
