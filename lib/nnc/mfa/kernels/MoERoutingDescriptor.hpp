#ifndef MFA_MOEROUTINGDESCRIPTOR_HPP_
#define MFA_MOEROUTINGDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct MoERoutingKernelDescriptor {
	uint32_t dataType;
	constexpr bool operator==(const MoERoutingKernelDescriptor& rhs) const { return dataType == rhs.dataType; }
};

template<>
struct std::hash<MoERoutingKernelDescriptor>
{
	std::size_t operator()(const MoERoutingKernelDescriptor& hash) const noexcept { return hash.dataType; }
};

struct MoERoutingKernel;

struct MoERoutingDescriptor {
	uint32_t dataType;

	constexpr bool operator==(const MoERoutingDescriptor& rhs) const { return dataType == rhs.dataType; }

	std::pair<MoERoutingKernelDescriptor, PipelineValue<MoERoutingKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MoERoutingKernelDescriptor, std::unique_ptr<MoERoutingKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<MoERoutingDescriptor>
{
	std::size_t operator()(const MoERoutingDescriptor& hash) const noexcept { return hash.dataType; }
};

#endif
