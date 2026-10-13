#ifndef MFA_INT8MATMULKERNELDESCRIPTOR_HPP_
#define MFA_INT8MATMULKERNELDESCRIPTOR_HPP_

#include <simd/simd.h>
#include "Int8MatMulDescriptor.hpp"
#include "GEMMOperandPrecision.hpp"

/// A configuration for one Int8MatMul function, and the key of its library.
///
/// The dense kernel follows GEMMKernel on Apple9 and later: every simdgroup
/// accumulates its register tile in FP32 with direct device loads, without
/// threadgroup memory or async copies. Int8 weights are converted to FP16 as
/// they are loaded, and the row-wise scales are applied while storing. FP16
/// activations are multiplied as they are; activations at any other precision
/// are quantized to FP16 codes with row-wise scales first, so the
/// multiplication stays in FP16.
struct Int8MatMulKernelDescriptor {
	Int8MatMulOperation operation;

	/// The precision of activations, outputs, weight scales and bias.
	GEMMOperandPrecision ioPrecision;

	/// Dense: the M x N output tile of a threadgroup, and the K extent of one
	/// main-loop iteration. Segmented: the rows per register block (M only).
	simd::ushort3 blockDimensions;

	/// Dense: the simdgroups dividing a threadgroup tile along M and N.
	simd::ushort2 splits;

	/// Dense: the number of 8-wide K slices interleaved by each load. A thread
	/// reads this many adjacent int8 weights of a row (and twice as many
	/// activations) at once, instead of single bytes.
	uint16_t kInterleave;

	/// Dense: add a bias to every output row.
	bool useBias;

	/// Quantize: rotate every 256 activations by the normalized Hadamard matrix.
	bool activationHadamard256;

	Int8MatMulKernelDescriptor() = delete;

	/// Choose the configuration for a dispatch on the device.
	Int8MatMulKernelDescriptor(const Int8MatMulDescriptor& descriptor, const DeviceProperties& dprops) noexcept;

	/// Specify the configuration directly, for kernel probes.
	Int8MatMulKernelDescriptor(Int8MatMulOperation operation, GEMMOperandPrecision ioPrecision, simd::ushort3 blockDimensions, simd::ushort2 splits, uint16_t kInterleave, bool useBias, bool activationHadamard256 = false) noexcept;

	bool operator==(const Int8MatMulKernelDescriptor& rhs) const;
};

template<>
struct std::hash<Int8MatMulKernelDescriptor>
{
	std::size_t operator()(const Int8MatMulKernelDescriptor& hash) const noexcept;
};

#endif
