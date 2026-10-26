#ifndef MFA_DEPTHWISECONVDESCRIPTOR_HPP_
#define MFA_DEPTHWISECONVDESCRIPTOR_HPP_

#include <functional>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

// Runtime geometry, bound with setBytes so the pipeline does not specialize on shapes.
struct DepthwiseConvArguments {
	uint32_t channel_units;
	uint32_t input_height;
	uint32_t input_width;
	uint32_t output_height;
	uint32_t output_width;
	uint32_t kernel_height;
	uint32_t kernel_width;
	uint32_t stride_height;
	uint32_t stride_width;
	uint32_t dilation_height;
	uint32_t dilation_width;
	uint32_t padding_top;
	uint32_t padding_left;
	uint32_t element_count;
};

struct DepthwiseConvKernelDescriptor {
	uint8_t vectorized;
	uint8_t fusedBias;
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const DepthwiseConvKernelDescriptor& rhs) const { return vectorized == rhs.vectorized && fusedBias == rhs.fusedBias && memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<DepthwiseConvKernelDescriptor>
{
	std::size_t operator()(const DepthwiseConvKernelDescriptor& hash) const noexcept
	{
		return std::hash<int>()((int)hash.vectorized | ((int)hash.fusedBias << 8) | ((int)hash.memoryPrecision.value << 16));
	}
};

struct DepthwiseConvKernel;

struct DepthwiseConvDescriptor {
	uint8_t vectorized;

	uint8_t fusedBias;

	GEMMOperandPrecision memoryPrecision;

	bool operator==(const DepthwiseConvDescriptor& rhs) const;

	std::pair<DepthwiseConvKernelDescriptor, PipelineValue<DepthwiseConvKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<DepthwiseConvKernelDescriptor, std::unique_ptr<DepthwiseConvKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<DepthwiseConvDescriptor>
{
	std::size_t operator()(const DepthwiseConvDescriptor& hash) const noexcept;
};

#endif
