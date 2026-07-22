#ifndef MFA_STRIDEDCOPYDESCRIPTOR_HPP_
#define MFA_STRIDEDCOPYDESCRIPTOR_HPP_

#include <functional>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct StridedCopyKernelDescriptor {
	uint8_t vectorized;
	uint8_t destinationStrided;
	uint8_t loadM;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const StridedCopyKernelDescriptor& rhs) const { return vectorized == rhs.vectorized && destinationStrided == rhs.destinationStrided && loadM == rhs.loadM && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<StridedCopyKernelDescriptor>
{
	std::size_t operator()(const StridedCopyKernelDescriptor& hash) const noexcept
	{
		return std::hash<int>()((int)hash.vectorized | ((int)hash.destinationStrided << 8) | ((int)hash.memoryPrecision.value << 16) | ((int)hash.loadM << 24));
	}
};

struct StridedCopyKernel;

struct StridedCopyDescriptor {
	uint8_t vectorized;

	GEMMOperandPrecision memoryPrecision;

	uint32_t rows;

	uint32_t cols;

	uint32_t sourceRowStride;

	uint32_t destinationRowStride;

	uint8_t destinationStrided;

	bool loadM;

	bool operator==(const StridedCopyDescriptor& rhs) const;

	std::pair<StridedCopyKernelDescriptor, PipelineValue<StridedCopyKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<StridedCopyKernelDescriptor, std::unique_ptr<StridedCopyKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<StridedCopyDescriptor>
{
	std::size_t operator()(const StridedCopyDescriptor& hash) const noexcept;
};

#endif
