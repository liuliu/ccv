extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <type_traits>

#ifdef HAVE_CUDA_SM80
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#endif

// Row-wise int8 weight GEMM: b[m, n] = sum_k a[m, k] * q[n, k] * w_scale[n] (+ bias[n]), where q / w_scale
// is the CCV_NNC_QX_8I_ROWWISE payload of an [n, k] weight matrix. Like the MFA scaled GEMM, small M
// multiplies the activations as they are, and otherwise quantizes them row-wise to int8 so the GEMM
// runs on int8 tensor cores:
//   a_scale[m] = max_k |a[m, k]| / 127, a_q[m, k] = clamp(rint(a[m, k] / a_scale[m]), -127, 127)
//   b[m, n] = a_scale[m] * w_scale[n] * sum_k a_q[m, k] * q[n, k] (+ bias[n])

// Up to this many activation rows use the GEMV kernel, which reads the int8 weights once and keeps the
// activations in their own precision. Beyond that, the int8 GEMM is faster (measured on RTX 6000 Ada).
#define CCV_NNC_8I_ROWWISE_GEMV_MAX_M (4)

static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_to_f(const float x) { return x; }
static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_to_f(const __half x) { return __half2float(x); }
static __device__ __forceinline__ float _ccv_nnc_8i_rowwise_to_f(const __nv_bfloat16 x) { return __bfloat162float(x); }
template<typename NUM>
static __device__ __forceinline__ NUM _ccv_nnc_8i_rowwise_from_f(const float x);
template<>
__device__ __forceinline__ float _ccv_nnc_8i_rowwise_from_f<float>(const float x) { return x; }
template<>
__device__ __forceinline__ __half _ccv_nnc_8i_rowwise_from_f<__half>(const float x) { return __float2half(x); }
template<>
__device__ __forceinline__ __nv_bfloat16 _ccv_nnc_8i_rowwise_from_f<__nv_bfloat16>(const float x) { return __float2bfloat16(x); }

// Load 8 consecutive elements (16-byte aligned) as float.
static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_load8(const float* const p, float v[8])
{
	const float4 x = ((const float4*)p)[0];
	const float4 y = ((const float4*)p)[1];
	v[0] = x.x;
	v[1] = x.y;
	v[2] = x.z;
	v[3] = x.w;
	v[4] = y.x;
	v[5] = y.y;
	v[6] = y.z;
	v[7] = y.w;
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_load8(const __half* const p, float v[8])
{
	const int4 x = *(const int4*)p;
	const __half2* const h = (const __half2*)&x;
	int i;
#pragma unroll
	for (i = 0; i < 4; i++)
	{
		const float2 f = __half22float2(h[i]);
		v[i * 2] = f.x;
		v[i * 2 + 1] = f.y;
	}
}

static __device__ __forceinline__ void _ccv_nnc_8i_rowwise_load8(const __nv_bfloat16* const p, float v[8])
{
	const int4 x = *(const int4*)p;
	const __nv_bfloat162* const h = (const __nv_bfloat162*)&x;
	int i;
#pragma unroll
	for (i = 0; i < 4; i++)
	{
		const float2 f = __bfloat1622float2(h[i]);
		v[i * 2] = f.x;
		v[i * 2 + 1] = f.y;
	}
}

// One warp per weight row; each lane reads 16 int8 weights per step, UNROLL steps at a time. k % 16 == 0.
template<typename NUM, int MR, int WARPS, int UNROLL>
__global__ void _ccv_nnc_8i_rowwise_gemv(const NUM* const __restrict__ a, const int8_t* const __restrict__ w, const NUM* const __restrict__ w_scale, const NUM* const __restrict__ bias, NUM* const __restrict__ b, const int n, const int k)
{
	const int row = blockIdx.x * WARPS + (threadIdx.x >> 5);
	if (row >= n)
		return;
	const int lane = threadIdx.x & 31;
	const int4* const wr = (const int4*)(w + (size_t)row * k);
	float acc[MR];
	int i, j, u;
#pragma unroll
	for (i = 0; i < MR; i++)
		acc[i] = 0;
	const int k16 = k >> 4;
	int c = lane;
	for (; c + 32 * (UNROLL - 1) < k16; c += 32 * UNROLL)
	{
		int4 wv[UNROLL];
#pragma unroll
		for (u = 0; u < UNROLL; u++)
			wv[u] = __ldcs(wr + c + u * 32);
#pragma unroll
		for (u = 0; u < UNROLL; u++)
		{
			const int8_t* const wc = (const int8_t*)&wv[u];
			float wf[16];
#pragma unroll
			for (j = 0; j < 16; j++)
				wf[j] = (float)wc[j];
#pragma unroll
			for (i = 0; i < MR; i++)
			{
				float av[16];
				const NUM* const ar = a + (size_t)i * k + (c + u * 32) * 16;
				_ccv_nnc_8i_rowwise_load8(ar, av);
				_ccv_nnc_8i_rowwise_load8(ar + 8, av + 8);
#pragma unroll
				for (j = 0; j < 16; j++)
					acc[i] = fmaf(av[j], wf[j], acc[i]);
			}
		}
	}
	for (; c < k16; c += 32)
	{
		const int4 wv = __ldcs(wr + c);
		const int8_t* const wc = (const int8_t*)&wv;
		float wf[16];
#pragma unroll
		for (j = 0; j < 16; j++)
			wf[j] = (float)wc[j];
#pragma unroll
		for (i = 0; i < MR; i++)
		{
			float av[16];
			const NUM* const ar = a + (size_t)i * k + c * 16;
			_ccv_nnc_8i_rowwise_load8(ar, av);
			_ccv_nnc_8i_rowwise_load8(ar + 8, av + 8);
#pragma unroll
			for (j = 0; j < 16; j++)
				acc[i] = fmaf(av[j], wf[j], acc[i]);
		}
	}
#pragma unroll
	for (i = 0; i < MR; i++)
	{
#pragma unroll
		for (j = 16; j > 0; j >>= 1)
			acc[i] += __shfl_xor_sync(0xffffffff, acc[i], j);
	}
	if (lane == 0)
	{
		const float s = _ccv_nnc_8i_rowwise_to_f(w_scale[row]);
		const float bv = bias ? _ccv_nnc_8i_rowwise_to_f(bias[row]) : 0.f;
#pragma unroll
		for (i = 0; i < MR; i++)
			b[(size_t)i * n + row] = _ccv_nnc_8i_rowwise_from_f<NUM>(acc[i] * s + bv);
	}
}

template<typename NUM>
static void _ccv_nnc_8i_rowwise_gemv_launch(const NUM* const a, const int m, const int8_t* const w, const NUM* const w_scale, const NUM* const bias, NUM* const b, const int n, const int k, cudaStream_t stream)
{
	const int warps = 4;
	const int unroll = 4;
	const int num_blocks = (n + warps - 1) / warps;
	switch (m)
	{
		case 1:
			_ccv_nnc_8i_rowwise_gemv<NUM, 1, warps, unroll><<<num_blocks, warps * 32, 0, stream>>>(a, w, w_scale, bias, b, n, k);
			break;
		case 2:
			_ccv_nnc_8i_rowwise_gemv<NUM, 2, warps, unroll><<<num_blocks, warps * 32, 0, stream>>>(a, w, w_scale, bias, b, n, k);
			break;
		case 3:
			_ccv_nnc_8i_rowwise_gemv<NUM, 3, warps, unroll><<<num_blocks, warps * 32, 0, stream>>>(a, w, w_scale, bias, b, n, k);
			break;
		case 4:
			_ccv_nnc_8i_rowwise_gemv<NUM, 4, warps, unroll><<<num_blocks, warps * 32, 0, stream>>>(a, w, w_scale, bias, b, n, k);
			break;
		default:
			assert(0);
	}
}

#ifdef HAVE_CUDA_SM80

// One block per row. Up to VPT vectors of 8 per thread stay in registers between the max reduction
// and the quantization pass. k % 8 == 0.
template<typename NUM, int THREADS, int VPT>
__global__ void __launch_bounds__(THREADS) _ccv_nnc_8i_rowwise_quantize(const NUM* const __restrict__ a, const int k, int8_t* const __restrict__ q, float* const __restrict__ scale)
{
	const size_t row = blockIdx.x;
	const NUM* const ar = a + row * k;
	int8_t* const qr = q + row * k;
	const int k8 = k >> 3;
	float v[VPT][8];
	float local_max = 0;
	int i, j, u;
#pragma unroll
	for (u = 0; u < VPT; u++)
	{
		i = threadIdx.x + u * THREADS;
		if (i < k8)
		{
			_ccv_nnc_8i_rowwise_load8(ar + i * 8, v[u]);
#pragma unroll
			for (j = 0; j < 8; j++)
				local_max = fmaxf(local_max, fabsf(v[u][j]));
		}
	}
	for (i = threadIdx.x + VPT * THREADS; i < k8; i += THREADS)
	{
		float x[8];
		_ccv_nnc_8i_rowwise_load8(ar + i * 8, x);
#pragma unroll
		for (j = 0; j < 8; j++)
			local_max = fmaxf(local_max, fabsf(x[j]));
	}
#pragma unroll
	for (j = 16; j > 0; j >>= 1)
		local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, j));
	__shared__ float warp_max[THREADS / 32];
	if ((threadIdx.x & 31) == 0)
		warp_max[threadIdx.x >> 5] = local_max;
	__syncthreads();
	float max_abs = warp_max[0];
#pragma unroll
	for (j = 1; j < THREADS / 32; j++)
		max_abs = fmaxf(max_abs, warp_max[j]);
	// Round the divisions to nearest even under --use_fast_math, so the codes are reproducible on the CPU.
	const float inv_scale = max_abs > 0 ? __fdiv_rn(127.f, max_abs) : 127.f;
	if (threadIdx.x == 0)
		scale[row] = max_abs > 0 ? __fdiv_rn(max_abs, 127.f) : __fdiv_rn(1.f, 127.f);
#pragma unroll
	for (u = 0; u < VPT; u++)
	{
		i = threadIdx.x + u * THREADS;
		if (i < k8)
		{
			union {
				int2 v;
				int8_t c[8];
			} out;
#pragma unroll
			for (j = 0; j < 8; j++)
				out.c[j] = (int8_t)max(-127, min(127, __float2int_rn(v[u][j] * inv_scale)));
			*(int2*)(qr + i * 8) = out.v;
		}
	}
	for (i = threadIdx.x + VPT * THREADS; i < k8; i += THREADS)
	{
		float x[8];
		_ccv_nnc_8i_rowwise_load8(ar + i * 8, x);
		union {
			int2 v;
			int8_t c[8];
		} out;
#pragma unroll
		for (j = 0; j < 8; j++)
			out.c[j] = (int8_t)max(-127, min(127, __float2int_rn(x[j] * inv_scale)));
		*(int2*)(qr + i * 8) = out.v;
	}
}

template<typename NUM>
static void _ccv_nnc_8i_rowwise_quantize_launch(const NUM* const a, const int m, const int k, int8_t* const q, float* const scale, cudaStream_t stream)
{
	const int k8 = k >> 3;
	// One block per row is latency bound with few rows, thus spread each row over more threads then.
	if (k8 <= 128)
		_ccv_nnc_8i_rowwise_quantize<NUM, 128, 1><<<m, 128, 0, stream>>>(a, k, q, scale);
	else if (k8 <= 256 * 4 && m >= 256)
		_ccv_nnc_8i_rowwise_quantize<NUM, 256, 4><<<m, 256, 0, stream>>>(a, k, q, scale);
	else if (k8 <= 512 * 4)
		_ccv_nnc_8i_rowwise_quantize<NUM, 512, 4><<<m, 512, 0, stream>>>(a, k, q, scale);
	else
		_ccv_nnc_8i_rowwise_quantize<NUM, 1024, 4><<<m, 1024, 0, stream>>>(a, k, q, scale);
}

template<typename NUM>
struct _ccv_nnc_8i_rowwise_cutlass_type_t {
	typedef NUM type;
};

template<>
struct _ccv_nnc_8i_rowwise_cutlass_type_t<__half> {
	typedef cutlass::half_t type;
};

template<>
struct _ccv_nnc_8i_rowwise_cutlass_type_t<__nv_bfloat16> {
	typedef cutlass::bfloat16_t type;
};

// int8 x int8 -> int32 tensor core GEMM with the scales and the bias fused into the epilogue:
// D[m, n] = acc[m, n] * a_scale[m] * w_scale[n] (+ bias[n]). A is row-major [m, k], W is row-major [n, k]
// (thus column-major k x n), D is row-major [m, n].
template<typename ElementOut, typename ThreadblockShape, typename WarpShape, int Stages, bool HasBias>
struct _ccv_nnc_8i_rowwise_cutlass_gemm_t {
	static constexpr int AlignmentAB = 16;
	static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOut>::value;
	static constexpr int EpilogueStages = 1;
	typedef cutlass::epilogue::threadblock::OutputTileThreadLayout<ThreadblockShape, WarpShape, ElementOut, AlignmentC, EpilogueStages> OutputTileThreadMap;
	typedef cutlass::epilogue::threadblock::VisitorAccFetch Accum;
	typedef cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, float, cute::Stride<cute::_1, cute::_0, int32_t>> AScale;
	typedef cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementOut, cute::Stride<cute::_0, cute::_1, int32_t>> RowVector;
	typedef cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest> Mul;
	typedef cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiply_add, float, float, cutlass::FloatRoundStyle::round_to_nearest> MulAdd;
	typedef cutlass::epilogue::threadblock::Sm80EVT<Mul, Accum, AScale> EVTAScale;
	typedef typename std::conditional<HasBias,
		cutlass::epilogue::threadblock::Sm80EVT<MulAdd, EVTAScale, RowVector, RowVector>,
		cutlass::epilogue::threadblock::Sm80EVT<Mul, EVTAScale, RowVector>>::type EVTWScale;
	typedef cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementOut, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>> D;
	typedef cutlass::epilogue::threadblock::Sm80EVT<D, EVTWScale> EVTD;
	// The visitor stores D itself. The base epilogue only provides the accumulator staging, which has no
	// bfloat16 specialization for int32 accumulators, thus instantiate it as half of the same width.
	typedef typename std::conditional<std::is_same<ElementOut, cutlass::bfloat16_t>::value, cutlass::half_t, ElementOut>::type ElementC;
	// The visitor GEMM is only exposed through the Stream-K swizzle. Stream-K itself is disabled
	// (avail_sms = 1, classic data-parallel tiles), which measured no slower and needs no workspace.
	typedef typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
		int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
		int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, AlignmentAB,
		ElementC, cutlass::layout::RowMajor, AlignmentC,
		int32_t, float,
		cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
		ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 32>,
		EVTD, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, Stages,
		cutlass::arch::OpMultiplyAddSaturate,
		EpilogueStages>::GemmKernel GemmKernel;
	typedef cutlass::gemm::device::GemmUniversalAdapter<GemmKernel> Gemm;

	static cutlass::Status run(const int8_t* const a, const float* const a_scale, const int8_t* const w, const ElementOut* const w_scale, const ElementOut* const bias, ElementOut* const d, const int m, const int n, const int k, cudaStream_t stream)
	{
		typename EVTAScale::Arguments a_scale_args = {
			{}, // Accum
			{ a_scale, 0.f, { cute::_1{}, cute::_0{}, int32_t(m) } }, // AScale
			{}, // Mul
		};
		typename RowVector::Arguments w_scale_args = { w_scale, ElementOut(0), { cute::_0{}, cute::_1{}, int32_t(n) } };
		typename EVTWScale::Arguments w_args;
		if constexpr (HasBias)
			w_args = { a_scale_args, w_scale_args, { bias, ElementOut(0), { cute::_0{}, cute::_1{}, int32_t(n) } }, {} };
		else
			w_args = { a_scale_args, w_scale_args, {} };
		typename EVTD::Arguments callback_args = {
			w_args,
			{ d, { int64_t(n), cute::_1{}, int64_t(m) * n } },
		};
		typename Gemm::Arguments args(
			cutlass::gemm::GemmUniversalMode::kGemm,
			{ m, n, k },
			1,
			callback_args,
			a, w, 0, 0,
			(int64_t)m * k, (int64_t)n * k, (int64_t)0, (int64_t)0,
			(int64_t)k, (int64_t)k, (int64_t)0, (int64_t)0,
			1 /* avail_sms */);
		Gemm gemm;
		cutlass::Status status = gemm.can_implement(args);
		if (status != cutlass::Status::kSuccess)
			return status;
		status = gemm.initialize(args, 0, stream);
		if (status != cutlass::Status::kSuccess)
			return status;
		return gemm.run(stream);
	}
};

template<typename ElementOut, typename ThreadblockShape, typename WarpShape, int Stages>
static cutlass::Status _ccv_nnc_8i_rowwise_cutlass_gemm(const int8_t* const a, const float* const a_scale, const int8_t* const w, const ElementOut* const w_scale, const ElementOut* const bias, ElementOut* const d, const int m, const int n, const int k, cudaStream_t stream)
{
	if (bias)
		return _ccv_nnc_8i_rowwise_cutlass_gemm_t<ElementOut, ThreadblockShape, WarpShape, Stages, true>::run(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	return _ccv_nnc_8i_rowwise_cutlass_gemm_t<ElementOut, ThreadblockShape, WarpShape, Stages, false>::run(a, a_scale, w, w_scale, 0, d, m, n, k, stream);
}

// Tile selection measured on RTX 6000 Ada (sm_89): small M favors short tiles (wider ones for wide N),
// larger M favors larger tiles.
template<typename ElementOut>
static cutlass::Status _ccv_nnc_8i_rowwise_cutlass_gemm_dispatch(const int8_t* const a, const float* const a_scale, const int8_t* const w, const ElementOut* const w_scale, const ElementOut* const bias, ElementOut* const d, const int m, const int n, const int k, cudaStream_t stream)
{
	using cutlass::gemm::GemmShape;
	if (m <= 16)
	{
		if (n >= 8192)
			return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<16, 128, 128>, GemmShape<16, 32, 128>, 4>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
		return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<16, 64, 128>, GemmShape<16, 32, 128>, 5>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	}
	if (m <= 32 && n >= 8192)
		return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<32, 128, 128>, GemmShape<32, 32, 128>, 4>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	if (m <= 128 && n < 8192)
		return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<64, 64, 128>, GemmShape<32, 32, 128>, 4>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	if (m <= 256 && !(m > 128 && n >= 8192))
		return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<64, 128, 64>, GemmShape<32, 64, 64>, 4>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	if (m <= 1024)
		return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<128, 128, 128>, GemmShape<64, 64, 128>, 3>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
	return _ccv_nnc_8i_rowwise_cutlass_gemm<ElementOut, GemmShape<256, 128, 64>, GemmShape<64, 64, 64>, 3>(a, a_scale, w, w_scale, bias, d, m, n, k, stream);
}

template<typename NUM>
static int _ccv_nnc_8i_rowwise_gemm(const NUM* const a, const int8_t* const w, const NUM* const w_scale, const NUM* const bias, NUM* const b, const int m, const int n, const int k, void* const workspace, cudaStream_t stream)
{
	typedef typename _ccv_nnc_8i_rowwise_cutlass_type_t<NUM>::type element_t;
	int8_t* const a_q = (int8_t*)workspace;
	float* const a_scale = (float*)((unsigned char*)workspace + (((size_t)m * k + 127) & ~(size_t)127));
	_ccv_nnc_8i_rowwise_quantize_launch(a, m, k, a_q, a_scale, stream);
	return _ccv_nnc_8i_rowwise_cutlass_gemm_dispatch<element_t>(a_q, a_scale, w, (const element_t*)w_scale, (const element_t*)bias, (element_t*)b, m, n, k, stream) == cutlass::Status::kSuccess;
}

#endif

int ccv_nnc_compat_8i_rowwise_gemm_supported(const int datatype, const void* const a, const void* const w, const void* const bias, const void* const b, const int m, const int n, const int k, size_t* const workspace_size)
{
	if (datatype != CCV_16F && datatype != CCV_16BF && datatype != CCV_32F)
		return 0;
	if (m <= 0 || n <= 0 || k <= 0)
		return 0;
	// Both kernels load the weights and the activations 16 bytes at a time.
	if ((k % 16) != 0 || ((uintptr_t)a & 15) || ((uintptr_t)w & 15))
		return 0;
	if (m <= CCV_NNC_8I_ROWWISE_GEMV_MAX_M)
	{
		*workspace_size = 0;
		return 1;
	}
#ifdef HAVE_CUDA_SM80
	// A multi-arch build also compiles the CUTLASS kernel for older archs, where its int8 MMA is an empty stub.
	if (ccv_nnc_gpu_device_props().major < 8)
		return 0;
	// The epilogue loads the scales and the bias, and stores the output 16 bytes at a time.
	const int alignment = 16 / CCV_GET_DATA_TYPE_SIZE(datatype);
	if ((n % alignment) != 0 || ((uintptr_t)b & 15) || ((uintptr_t)bias & 15))
		return 0;
	if ((size_t)m * k > INT_MAX || (size_t)m * n > INT_MAX || (size_t)n * k > INT_MAX)
		return 0;
	*workspace_size = (((size_t)m * k + 127) & ~(size_t)127) + sizeof(float) * m;
	return 1;
#else
	return 0;
#endif
}

int ccv_nnc_compat_8i_rowwise_gemm(const int datatype, const void* const a, const void* const w, const void* const bias, void* const b, const int m, const int n, const int k, void* const workspace, ccv_nnc_stream_context_t* const stream_context)
{
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int8_t* const q = (const int8_t*)w;
	const void* const w_scale = (const unsigned char*)w + (((size_t)n * k + 127) & ~(size_t)127);
	if (m <= CCV_NNC_8I_ROWWISE_GEMV_MAX_M)
	{
		if (datatype == CCV_16F)
			_ccv_nnc_8i_rowwise_gemv_launch((const __half*)a, m, q, (const __half*)w_scale, (const __half*)bias, (__half*)b, n, k, stream);
		else if (datatype == CCV_16BF)
			_ccv_nnc_8i_rowwise_gemv_launch((const __nv_bfloat16*)a, m, q, (const __nv_bfloat16*)w_scale, (const __nv_bfloat16*)bias, (__nv_bfloat16*)b, n, k, stream);
		else
			_ccv_nnc_8i_rowwise_gemv_launch((const float*)a, m, q, (const float*)w_scale, (const float*)bias, (float*)b, n, k, stream);
		return 1;
	}
#ifdef HAVE_CUDA_SM80
	if (datatype == CCV_16F)
		return _ccv_nnc_8i_rowwise_gemm((const __half*)a, q, (const __half*)w_scale, (const __half*)bias, (__half*)b, m, n, k, workspace, stream);
	else if (datatype == CCV_16BF)
		return _ccv_nnc_8i_rowwise_gemm((const __nv_bfloat16*)a, q, (const __nv_bfloat16*)w_scale, (const __nv_bfloat16*)bias, (__nv_bfloat16*)b, m, n, k, workspace, stream);
	return _ccv_nnc_8i_rowwise_gemm((const float*)a, q, (const float*)w_scale, (const float*)bias, (float*)b, m, n, k, workspace, stream);
#else
	return 0;
#endif
}
