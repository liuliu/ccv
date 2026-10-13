#include "Int8MatMulKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <algorithm>

bool Int8MatMulKernelDescriptor::operator==(const Int8MatMulKernelDescriptor& rhs) const
{
	return operation == rhs.operation &&
		ioPrecision == rhs.ioPrecision &&
		simd_all(blockDimensions == rhs.blockDimensions) &&
		simd_all(splits == rhs.splits) &&
		kInterleave == rhs.kInterleave &&
		useBias == rhs.useBias &&
		activationHadamard256 == rhs.activationHadamard256;
}

std::size_t std::hash<Int8MatMulKernelDescriptor>::operator()(const Int8MatMulKernelDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, hash.kInterleave)));
	seed = combine_32(seed, pack_32(hash.splits));
	seed = combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.operation, (uint16_t)hash.ioPrecision.value }));
	seed = combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.useBias, (uint16_t)hash.activationHadamard256 }));
	return seed;
}

Int8MatMulKernelDescriptor::Int8MatMulKernelDescriptor(Int8MatMulOperation operation, GEMMOperandPrecision ioPrecision, simd::ushort3 blockDimensions, simd::ushort2 splits, uint16_t kInterleave, bool useBias, bool activationHadamard256) noexcept
{
	this->operation = operation;
	this->ioPrecision = ioPrecision;
	this->blockDimensions = blockDimensions;
	this->splits = splits;
	this->kInterleave = kInterleave;
	this->useBias = useBias;
	this->activationHadamard256 = activationHadamard256;
}

Int8MatMulKernelDescriptor::Int8MatMulKernelDescriptor(const Int8MatMulDescriptor& descriptor, const DeviceProperties& dprops) noexcept
{
	operation = descriptor.operation;
	// Only these functions read or write IO-precision data; the others (FP32
	// only, or precision independent) share one library.
	const bool usesIOPrecision =
		operation == Int8MatMulQuantizeActivation ||
		operation == Int8MatMulRotateActivation ||
		operation == Int8MatMulDense;
	ioPrecision = usesIOPrecision ? descriptor.ioPrecision : GEMMOperandPrecision::FP32;
	blockDimensions = simd::ushort3 { 0, 0, 0 };
	splits = simd::ushort2 { 1, 1 };
	kInterleave = 1;
	useBias = operation == Int8MatMulDense && descriptor.useBias;
	activationHadamard256 = operation == Int8MatMulQuantizeActivation && descriptor.activationHadamard256;
	if (operation == Int8MatMulSegmented) {
		// Larger routed groups reuse each loaded weight tile across more rows.
		const bool largeGroups = descriptor.binCount > 0 && descriptor.M / descriptor.binCount >= 64;
		blockDimensions[0] = largeGroups ? 32 : 16;
	} else if (operation == Int8MatMulDense) {
		// Measured on M3 Ultra (60 cores) and M4 Pro (20 cores) over FP16 and FP32
		// shapes with 8 <= M <= 9276, 512 <= N <= 129280 and 1024 <= K <= 16384.
		// With many tiles per core this is GEMM's Apple9 tile, and interleaved
		// loads are about 1% faster. With fewer, smaller tiles keep the cores busy
		// to the end, up to 5x faster than 32 x 32 at small M.
		// Splitting a threadgroup across more simdgroups, as GEMM does with async
		// copies, did not help consistently.
		const uint32_t cores = std::max(dprops.coreCount, 1u);
		auto tilesPerCore = [&](uint32_t blockM, uint32_t blockN) {
			return uint64_t((descriptor.M + blockM - 1) / blockM) * ((descriptor.N + blockN - 1) / blockN) / cores;
		};
		if (descriptor.M <= 8 || tilesPerCore(16, 8) < 8) {
			blockDimensions = simd::ushort3 { 8, 8, 64 };
			kInterleave = 4;
		} else if (tilesPerCore(32, 32) < 12) {
			blockDimensions = simd::ushort3 { 16, 8, 64 };
		} else if (tilesPerCore(32, 32) < 200) {
			blockDimensions = simd::ushort3 { 32, 16, 64 };
		} else {
			blockDimensions = simd::ushort3 { 32, 32, 64 };
			kInterleave = 4;
		}
	}
}
