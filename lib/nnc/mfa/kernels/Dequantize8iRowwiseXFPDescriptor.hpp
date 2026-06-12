#ifndef MFA_DEQUANTIZE8IROWWISEXFPDESCRIPTOR_HPP_
#define MFA_DEQUANTIZE8IROWWISEXFPDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct Dequantize8iRowwiseXFPKernelDescriptor {
	uint32_t format;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const Dequantize8iRowwiseXFPKernelDescriptor& rhs) const { return format == rhs.format && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<Dequantize8iRowwiseXFPKernelDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseXFPKernelDescriptor& hash) const noexcept {
		return std::hash<uint64_t>()((uint64_t)hash.format | ((uint64_t)hash.memoryPrecision.value << 32));
	}
};

struct Dequantize8iRowwiseXFPKernel;

struct Dequantize8iRowwiseXFPDescriptor {
	uint32_t format;
	GEMMOperandPrecision memoryPrecision;
	uint32_t rowLength;
	uint32_t length;

	bool operator==(const Dequantize8iRowwiseXFPDescriptor& rhs) const;

	uint32_t rowCount() const noexcept;
	uint32_t groupSize() const noexcept;
	uint32_t groupsPerRow() const noexcept;
	uint32_t groupBits() const noexcept;
	uint32_t totalGroups() const noexcept;
	uint32_t inputScaleOffset() const noexcept;

	std::pair<Dequantize8iRowwiseXFPKernelDescriptor, PipelineValue<Dequantize8iRowwiseXFPKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXFPKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXFPKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<Dequantize8iRowwiseXFPDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseXFPDescriptor& hash) const noexcept;
};

#endif
