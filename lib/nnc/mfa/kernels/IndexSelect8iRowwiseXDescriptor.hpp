#ifndef MFA_INDEXSELECT8IROWWISEXDESCRIPTOR_HPP_
#define MFA_INDEXSELECT8IROWWISEXDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct IndexSelect8iRowwiseXKernelDescriptor {
	uint32_t format;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const IndexSelect8iRowwiseXKernelDescriptor& rhs) const { return format == rhs.format && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<IndexSelect8iRowwiseXKernelDescriptor>
{
	std::size_t operator()(const IndexSelect8iRowwiseXKernelDescriptor& hash) const noexcept {
		return std::hash<uint64_t>()((uint64_t)hash.format | ((uint64_t)hash.memoryPrecision.value << 32));
	}
};

struct IndexSelect8iRowwiseXKernel;

struct IndexSelect8iRowwiseXDescriptor {
	uint32_t format;
	GEMMOperandPrecision memoryPrecision;
	uint32_t rowLength;
	uint32_t inputLength;
	uint32_t outputLength;

	bool operator==(const IndexSelect8iRowwiseXDescriptor& rhs) const;

	uint32_t inputRowCount() const noexcept;
	uint32_t outputRowCount() const noexcept;
	uint32_t groupSize() const noexcept;
	uint32_t groupsPerRow() const noexcept;
	uint32_t groupBits() const noexcept;
	uint32_t inputGroups() const noexcept;
	uint32_t outputGroups() const noexcept;
		uint64_t inputScaleOffset() const noexcept;

	std::pair<IndexSelect8iRowwiseXKernelDescriptor, PipelineValue<IndexSelect8iRowwiseXKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelect8iRowwiseXKernelDescriptor, std::unique_ptr<IndexSelect8iRowwiseXKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<IndexSelect8iRowwiseXDescriptor>
{
	std::size_t operator()(const IndexSelect8iRowwiseXDescriptor& hash) const noexcept;
};

#endif
