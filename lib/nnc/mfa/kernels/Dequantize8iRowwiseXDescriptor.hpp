#ifndef MFA_DEQUANTIZE8IROWWISEXDESCRIPTOR_HPP_
#define MFA_DEQUANTIZE8IROWWISEXDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct Dequantize8iRowwiseXKernelDescriptor {
	uint32_t format;
	constexpr bool operator==(const Dequantize8iRowwiseXKernelDescriptor& rhs) const { return format == rhs.format; }
};

template<>
struct std::hash<Dequantize8iRowwiseXKernelDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseXKernelDescriptor& hash) const noexcept {
		return std::hash<uint32_t>()(hash.format);
	}
};

struct Dequantize8iRowwiseXKernel;

struct Dequantize8iRowwiseXDescriptor {
	uint32_t format;
	uint32_t scaleSize;
	uint32_t rowLength;
	uint32_t length;

	bool operator==(const Dequantize8iRowwiseXDescriptor& rhs) const;

	uint32_t rowCount() const noexcept;
	uint32_t groupSize() const noexcept;
	uint32_t groupsPerRow() const noexcept;
	uint32_t groupBits() const noexcept;
	uint32_t totalGroups() const noexcept;
		uint64_t inputScaleOffset() const noexcept;
		uint64_t outputScaleOffset() const noexcept;
	uint32_t scaleBytes() const noexcept;
	uint32_t dispatchItems() const noexcept;

	std::pair<Dequantize8iRowwiseXKernelDescriptor, PipelineValue<Dequantize8iRowwiseXKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXKernel>> *const libraryCache) const noexcept;
};

struct Dequantize8iRowwiseXSelectedDescriptor {
	uint32_t format;
	uint32_t scaleSize;
	uint32_t rowLength;
	uint32_t rowsPerExpert;
	uint32_t expertCount;
	uint32_t segmentCount;

	bool operator==(const Dequantize8iRowwiseXSelectedDescriptor& rhs) const;

	uint32_t groupSize() const noexcept;
	uint32_t groupsPerRow() const noexcept;
	uint32_t groupBits() const noexcept;
	uint32_t groupsPerExpert() const noexcept;
		uint64_t inputScaleOffset() const noexcept;
		uint64_t outputScaleOffset() const noexcept;
	uint32_t scaleBytesPerExpert() const noexcept;
	uint32_t dispatchItemsPerExpert() const noexcept;

	std::pair<Dequantize8iRowwiseXKernelDescriptor, PipelineValue<Dequantize8iRowwiseXKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<Dequantize8iRowwiseXDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseXDescriptor& hash) const noexcept;
};

template<>
struct std::hash<Dequantize8iRowwiseXSelectedDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseXSelectedDescriptor& hash) const noexcept;
};

#endif
