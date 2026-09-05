#ifndef MFA_INT8MATMULDESCRIPTOR_HPP_
#define MFA_INT8MATMULDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

enum Int8MatMulOperation : uint8_t {
	Int8MatMulQuantizeActivation = 0,
	Int8MatMulCastWeights = 1,
	Int8MatMulDequantizeOutput = 2,
	Int8MatMulDequantizeSegmentedOutput = 3,
	Int8MatMulSegmented = 4,
};

struct Int8MatMulKernelDescriptor {
	uint32_t blockM = 16;
	constexpr bool operator==(const Int8MatMulKernelDescriptor& rhs) const { return blockM == rhs.blockM; }
};

template<>
struct std::hash<Int8MatMulKernelDescriptor>
{
	std::size_t operator()(const Int8MatMulKernelDescriptor& hash) const noexcept { return hash.blockM; }
};

struct Int8MatMulKernel;

struct Int8MatMulDescriptor {
	uint32_t M;
	uint32_t N;
	uint32_t K;
	uint32_t expertCount;
	uint32_t binCount;
	Int8MatMulOperation operation;

	bool operator==(const Int8MatMulDescriptor& rhs) const;

	std::pair<Int8MatMulKernelDescriptor, PipelineValue<Int8MatMulKernel>*> findKernel(
		MTL::Device* device,
		const DeviceProperties& dprops,
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
