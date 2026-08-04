#ifndef MFA_MOEROUTINGDESCRIPTOR_HPP_
#define MFA_MOEROUTINGDESCRIPTOR_HPP_

#include <utility>
#include "PipelineValue.hpp"
#include "DeviceProperties.hpp"

struct MoERoutingKernelDescriptor {
	uint32_t activationDataType;
	uint32_t routingDataType;
	constexpr bool operator==(const MoERoutingKernelDescriptor& rhs) const { return activationDataType == rhs.activationDataType && routingDataType == rhs.routingDataType; }
};

template<>
struct std::hash<MoERoutingKernelDescriptor>
{
	std::size_t operator()(const MoERoutingKernelDescriptor& hash) const noexcept { return std::hash<uint64_t>()((uint64_t)hash.activationDataType | ((uint64_t)hash.routingDataType << 32)); }
};

struct MoERoutingKernel;

struct MoERoutingDescriptor {
	uint32_t activationDataType;
	uint32_t routingDataType;
	uint32_t expertCount;
	uint32_t kth;
	uint32_t hidden;
	float weightScale;
	bool preselected;
	bool singleInputToken;

	bool operator==(const MoERoutingDescriptor& rhs) const;

	std::pair<MoERoutingKernelDescriptor, PipelineValue<MoERoutingKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MoERoutingKernelDescriptor, std::unique_ptr<MoERoutingKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<MoERoutingDescriptor>
{
	std::size_t operator()(const MoERoutingDescriptor& hash) const noexcept;
};

#endif
