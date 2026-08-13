#ifndef MFA_INDEXSELECTDESCRIPTOR_HPP_
#define MFA_INDEXSELECTDESCRIPTOR_HPP_

#include <functional>
#include <utility>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

enum class IndexSelectDataType : uint8_t {
	FP16,
	BF16,
	INT32,
};

struct IndexSelectKernelDescriptor {
	uint8_t vectorWidth;
	bool loadM;
	IndexSelectDataType dataType;
	constexpr bool operator==(const IndexSelectKernelDescriptor& rhs) const
	{
		return vectorWidth == rhs.vectorWidth && loadM == rhs.loadM && dataType == rhs.dataType;
	}
};

template<>
struct std::hash<IndexSelectKernelDescriptor>
{
	std::size_t operator()(const IndexSelectKernelDescriptor& value) const noexcept
	{
		return std::hash<int>()((int)value.vectorWidth | ((int)value.dataType << 8) | ((int)value.loadM << 16));
	}
};

struct IndexSelectKernel;

struct IndexSelectDescriptor {
	IndexSelectDataType dataType;
	uint8_t vectorWidth;
	uint16_t threadsPerRow;
	uint32_t outputRows;
	bool loadM;

	bool operator==(const IndexSelectDescriptor& rhs) const;

	std::pair<IndexSelectKernelDescriptor, PipelineValue<IndexSelectKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelectKernelDescriptor, std::unique_ptr<IndexSelectKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<IndexSelectDescriptor>
{
	std::size_t operator()(const IndexSelectDescriptor& value) const noexcept;
};

#endif
