#ifndef MFA_SCATTERADDDESCRIPTOR_HPP_
#define MFA_SCATTERADDDESCRIPTOR_HPP_

#include <cstdint>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"
#include "ScatterAddKernel.hpp"

struct ScatterAddDescriptor {
	GEMMOperandPrecision memoryPrecision;

	bool operator==(const ScatterAddDescriptor& rhs) const;
	std::pair<ScatterAddKernelDescriptor, PipelineValue<ScatterAddKernel>*> findKernel(MTL::Device* device, const DeviceProperties& dprops, NS::Array* binaryArchivesToRead, MTL::BinaryArchive* binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ScatterAddKernelDescriptor, std::unique_ptr<ScatterAddKernel>>* libraryCache) const noexcept;
};

template<>
struct std::hash<ScatterAddDescriptor>
{
	std::size_t operator()(const ScatterAddDescriptor& value) const noexcept;
};

#endif
