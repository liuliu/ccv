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

template<int FORMAT, typename BITS>
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_x_decode_iq_oct(const BITS& bits, const int lane, int* const values)
{
	if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S)
	{
		const int grid_index = (int)bits.read(lane >= 8 ? 10 : 0, 10);
		const uint64_t grid = ccv_nnc_8i_rowwise_packed_iq2s_grid[grid_index];
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
		const uint64_t grid = ccv_nnc_8i_rowwise_packed_iq2xs_grid[grid_index];
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
		const uint16_t grid = ccv_nnc_8i_rowwise_packed_iq2xxs_grid[grid_index];
		const int scale_code = (int)bits.read(60, 4);
		const int scale = scale_code < 8 ? scale_code + 1 : (scale_code < 12 ? (scale_code - 3) * 2 : (scale_code - 7) * 4);
		const int sign_code = (int)bits.read(32 + subgroup * 7, 7);
		const int signs = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[sign_code];
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
		const uint32_t grid0 = ccv_nnc_8i_rowwise_packed_iq3s_grid[(int)bits.read(subgroup * 9, 9)];
		const uint32_t grid1 = ccv_nnc_8i_rowwise_packed_iq3s_grid[(int)bits.read((subgroup + 1) * 9, 9)];
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
	const uint32_t grid0 = ccv_nnc_8i_rowwise_packed_iq3xxs_grid[(int)bits.read(0, 8)];
	const uint32_t grid1 = ccv_nnc_8i_rowwise_packed_iq3xxs_grid[(int)bits.read(8, 8)];
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
static __global__ void _ccv_nnc_dequantize_8i_rowwise_x_fp_2d_oct(const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output)
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
	const size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	if (col < row_length)
	{
		const size_t group = row * groups_per_row + col / group_size;
		const _ccv_nnc_8i_rowwise_x_global_bits_t bits = {
			input,
			group * group_bits,
		};
		const int group_lane = col % group_size;
		const NUM scale = scales[row];
		int decoded[8];
		if (FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_S ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XS ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ2_XXS ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_S ||
			FORMAT == CCV_NNC_QX_8I_ROWWISE_IQ3_XXS)
			_ccv_nnc_8i_rowwise_x_decode_iq_oct<FORMAT>(bits, group_lane, decoded);
		else {
#pragma unroll
			for (int i = 0; i < 8; i++)
				decoded[i] = _ccv_nnc_8i_rowwise_x_decode<FORMAT>(bits, group_lane + i);
		}
		NUM values[8];
#pragma unroll
		for (int i = 0; i < 8; i++)
			values[i] = _ccv_nnc_8i_rowwise_x_mul(scale, decoded[i]);
		_ccv_nnc_8i_rowwise_x_store8(output + row * row_length + col, values);
	}
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

template<typename NUM>
static void _ccv_nnc_dequantize_8i_rowwise_x_fp_launch(const int format, const size_t output_length, const size_t row_length, const size_t groups_per_row, const uint8_t* const input, const NUM* const scales, NUM* const output, cudaStream_t stream)
{
#define DISPATCH_FORMAT(case_format) \
	case case_format: \
		if (output_length / row_length <= 65535 && row_length % 8 == 0 && \
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
	switch (format)
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
