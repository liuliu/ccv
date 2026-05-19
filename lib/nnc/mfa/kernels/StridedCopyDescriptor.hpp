#ifndef MFA_STRIDEDCOPYDESCRIPTOR_HPP_
#define MFA_STRIDEDCOPYDESCRIPTOR_HPP_

#include <functional>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct StridedCopyKernelDescriptor {
	uint8_t vectorized;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const StridedCopyKernelDescriptor& rhs) const { return vectorized == rhs.vectorized && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<StridedCopyKernelDescriptor>
{
	std::size_t operator()(const StridedCopyKernelDescriptor& hash) const noexcept
	{
		return std::hash<int>()((int)hash.vectorized | ((int)hash.memoryPrecision.value << 8));
	}
};

struct StridedCopyKernel;

struct StridedCopyDescriptor {
	uint8_t vectorized;

	GEMMOperandPrecision memoryPrecision;

	uint32_t rows;

	uint32_t cols;

	uint32_t sourceRowStride;

	bool operator==(const StridedCopyDescriptor& rhs) const;

	std::pair<StridedCopyKernelDescriptor, PipelineValue<StridedCopyKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<StridedCopyKernelDescriptor, std::unique_ptr<StridedCopyKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<StridedCopyDescriptor>
{
	std::size_t operator()(const StridedCopyDescriptor& hash) const noexcept;
};

#endif
