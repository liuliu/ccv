#ifndef MFA_DEQUANTIZE8IROWWISEDESCRIPTOR_HPP_
#define MFA_DEQUANTIZE8IROWWISEDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"

struct Dequantize8iRowwiseKernelDescriptor {
	uint8_t vectorized;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const Dequantize8iRowwiseKernelDescriptor& rhs) const { return vectorized == rhs.vectorized && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<Dequantize8iRowwiseKernelDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseKernelDescriptor& hash) const noexcept {
		return std::hash<int>()((int)hash.vectorized | ((int)hash.memoryPrecision.value << 8));
	}
};

struct Dequantize8iRowwiseKernel;

struct Dequantize8iRowwiseDescriptor {
	GEMMOperandPrecision memoryPrecision;
	uint32_t rowLength;
	uint32_t length;

	bool operator==(const Dequantize8iRowwiseDescriptor& rhs) const;

	bool vectorized() const noexcept;

	std::pair<Dequantize8iRowwiseKernelDescriptor, PipelineValue<Dequantize8iRowwiseKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseKernel>> *const libraryCache) const noexcept;
};

template<>
struct std::hash<Dequantize8iRowwiseDescriptor>
{
	std::size_t operator()(const Dequantize8iRowwiseDescriptor& hash) const noexcept;
};

#endif
