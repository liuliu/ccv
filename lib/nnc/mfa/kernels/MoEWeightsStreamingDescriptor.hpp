#ifndef MFA_MOEWEIGHTSSTREAMINGDESCRIPTOR_HPP_
#define MFA_MOEWEIGHTSSTREAMINGDESCRIPTOR_HPP_

#include <utility>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

struct MoEWeightsStreamingKernelDescriptor {
	uint8_t version;
	constexpr bool operator==(const MoEWeightsStreamingKernelDescriptor& rhs) const
	{
		return version == rhs.version;
	}
};

template<>
struct std::hash<MoEWeightsStreamingKernelDescriptor>
{
	std::size_t operator()(const MoEWeightsStreamingKernelDescriptor& hash) const noexcept
	{
		return std::hash<uint8_t>()(hash.version);
	}
};

struct MoEWeightsStreamingKernel;

struct MoEWeightsStreamingDescriptor {
	uint8_t version;

	constexpr bool operator==(const MoEWeightsStreamingDescriptor& rhs) const
	{
		return version == rhs.version;
	}

	std::pair<MoEWeightsStreamingKernelDescriptor, PipelineValue<MoEWeightsStreamingKernel>*> findKernel(
		MTL::Device* device, const DeviceProperties& dprops, NS::Array* binaryArchivesToRead,
		MTL::BinaryArchive* binaryArchiveToWrite, const std::string& pathToWrite,
		std::unordered_map<MoEWeightsStreamingKernelDescriptor,
			std::unique_ptr<MoEWeightsStreamingKernel>>* libraryCache) const noexcept;
};

template<>
struct std::hash<MoEWeightsStreamingDescriptor>
{
	std::size_t operator()(const MoEWeightsStreamingDescriptor& hash) const noexcept
	{
		return std::hash<uint8_t>()(hash.version);
	}
};

#endif
