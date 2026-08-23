#ifndef MFA_TRANSPOSEDESCRIPTOR_HPP_
#define MFA_TRANSPOSEDESCRIPTOR_HPP_

#include <functional>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct TransposeKernelDescriptor {
	GEMMOperandPrecision memoryPrecision;
	constexpr bool operator==(const TransposeKernelDescriptor& rhs) const { return memoryPrecision == rhs.memoryPrecision; }
};

template<>
struct std::hash<TransposeKernelDescriptor>
{
	std::size_t operator()(const TransposeKernelDescriptor& hash) const noexcept
	{
		return std::hash<int>()((int)hash.memoryPrecision.value);
	}
};

struct TransposeKernel;

struct TransposeDescriptor {
	GEMMOperandPrecision memoryPrecision;

	bool operator==(const TransposeDescriptor& rhs) const;

	std::pair<TransposeKernelDescriptor, PipelineValue<TransposeKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<TransposeKernelDescriptor, std::unique_ptr<TransposeKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<TransposeDescriptor>
{
	std::size_t operator()(const TransposeDescriptor& hash) const noexcept;
};

#endif
