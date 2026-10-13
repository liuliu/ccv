#ifndef MFA_INT8MATMULDESCRIPTOR_HPP_
#define MFA_INT8MATMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

enum Int8MatMulOperation : uint8_t {
	Int8MatMulQuantizeActivation = 0,
	Int8MatMulCastWeights = 1,
	Int8MatMulDequantizeOutput = 2,
	Int8MatMulDequantizeSegmentedOutput = 3,
	Int8MatMulSegmented = 4,
	Int8MatMulDense = 5,
	/// Rotates FP16 activations as Quantize does with activationHadamard256,
	/// without quantizing them (K is a multiple of 256).
	Int8MatMulRotateActivation = 6,
};

struct Int8MatMulKernelDescriptor;
struct Int8MatMulKernel;

struct Int8MatMulDescriptor {
	uint32_t M;
	uint32_t N;
	uint32_t K;
	uint32_t expertCount;
	uint32_t binCount;
	Int8MatMulOperation operation;
	/// The precision of activations, outputs, weight scales and bias.
	GEMMOperandPrecision ioPrecision = GEMMOperandPrecision::FP32;
	/// Dense: add a bias to every output row.
	bool useBias = false;
	/// Quantize: rotate every 256 activations by the normalized Hadamard
	/// matrix the weights were rotated by (K is a multiple of 256, at most 65536).
	bool activationHadamard256 = false;

	bool operator==(const Int8MatMulDescriptor& rhs) const;

	// Specialize an explicit kernel configuration (also used by kernel probes).
	std::pair<Int8MatMulKernelDescriptor, PipelineValue<Int8MatMulKernel>*> findKernel(
		MTL::Device* device,
		const Int8MatMulKernelDescriptor& kernelDescriptor,
		NS::Array* binaryArchivesToRead,
		MTL::BinaryArchive* binaryArchiveToWrite,
		const std::string& pathToWrite,
		std::unordered_map<Int8MatMulKernelDescriptor, std::unique_ptr<Int8MatMulKernel>>* libraryCache) const noexcept;
};

template<>
struct std::hash<Int8MatMulDescriptor>
{
	std::size_t operator()(const Int8MatMulDescriptor& hash) const noexcept;
};

#endif
