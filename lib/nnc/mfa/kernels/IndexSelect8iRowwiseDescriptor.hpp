#ifndef MFA_INDEXSELECT8IROWWISEDESCRIPTOR_HPP_
#define MFA_INDEXSELECT8IROWWISEDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct IndexSelect8iRowwiseKernelDescriptor {
	uint8_t vectorized;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const IndexSelect8iRowwiseKernelDescriptor& rhs) const { return vectorized == rhs.vectorized && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<IndexSelect8iRowwiseKernelDescriptor>
{
	std::size_t operator()(const IndexSelect8iRowwiseKernelDescriptor& hash) const noexcept {
		return std::hash<int>()((int)hash.vectorized | ((int)hash.memoryPrecision.value << 8));
	}
};

struct IndexSelect8iRowwiseKernel;

struct IndexSelect8iRowwiseDescriptor {
	GEMMOperandPrecision memoryPrecision;
	uint32_t rowLength;
	uint32_t inputLength;
	uint32_t outputLength;

	bool operator==(const IndexSelect8iRowwiseDescriptor& rhs) const;

	bool vectorized() const noexcept;

	std::pair<IndexSelect8iRowwiseKernelDescriptor, PipelineValue<IndexSelect8iRowwiseKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelect8iRowwiseKernelDescriptor, std::unique_ptr<IndexSelect8iRowwiseKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<IndexSelect8iRowwiseDescriptor>
{
	std::size_t operator()(const IndexSelect8iRowwiseDescriptor& hash) const noexcept;
};

#endif
