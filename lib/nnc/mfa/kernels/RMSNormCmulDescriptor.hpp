#ifndef MFA_RMSNORMCMULDESCRIPTOR_HPP_
#define MFA_RMSNORMCMULDESCRIPTOR_HPP_

#include <functional>
#include <simd/simd.h>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

struct RMSNormCmulKernelDescriptor {
	GEMMOperandPrecision aPrecision;
	GEMMOperandPrecision rotationPrecision;

	bool operator==(const RMSNormCmulKernelDescriptor& rhs) const;
};

template<>
struct std::hash<RMSNormCmulKernelDescriptor> {
	std::size_t operator()(const RMSNormCmulKernelDescriptor& value) const noexcept;
};

struct RMSNormCmulKernel;

struct RMSNormCmulDescriptor {
	float epsilon;
	GEMMOperandPrecision aPrecision;
	GEMMOperandPrecision rotationPrecision;
	uint32_t columnCount;
	uint32_t broadcastRatio;
	uint32_t rowsPerThreadgroup;

	bool operator==(const RMSNormCmulDescriptor& rhs) const;
	std::pair<RMSNormCmulKernelDescriptor, PipelineValue<RMSNormCmulKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RMSNormCmulKernelDescriptor, std::unique_ptr<RMSNormCmulKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<RMSNormCmulDescriptor> {
	std::size_t operator()(const RMSNormCmulDescriptor& value) const noexcept;
};

#endif
