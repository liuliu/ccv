#ifndef MFA_HYPERCONNECTIONDESCRIPTOR_HPP_
#define MFA_HYPERCONNECTIONDESCRIPTOR_HPP_

#include <simd/simd.h>
#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct HyperConnectionKernelDescriptor {
	uint8_t value;
	uint8_t blockFP16;
	uint8_t loadM;
	constexpr bool operator==(const HyperConnectionKernelDescriptor& rhs) const { return value == rhs.value && blockFP16 == rhs.blockFP16 && loadM == rhs.loadM; }
};

template<>
struct std::hash<HyperConnectionKernelDescriptor>
{
	std::size_t operator()(const HyperConnectionKernelDescriptor& hash) const noexcept { return (size_t)hash.value | ((size_t)hash.blockFP16 << 8) | ((size_t)hash.loadM << 9); }
};

struct HyperConnectionKernel;

struct HyperConnectionDescriptor {
	uint32_t rowCount;
	uint32_t count;
	uint32_t hidden;
	uint32_t sinkhornIterations;
	float epsilon;
	uint32_t operation;
	bool blockFP16;
	bool loadM;

	bool operator==(const HyperConnectionDescriptor& rhs) const;

	std::pair<HyperConnectionKernelDescriptor, PipelineValue<HyperConnectionKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<HyperConnectionKernelDescriptor, std::unique_ptr<HyperConnectionKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<HyperConnectionDescriptor>
{
	std::size_t operator()(const HyperConnectionDescriptor& hash) const noexcept;
};

#endif
