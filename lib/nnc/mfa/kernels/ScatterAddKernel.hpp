#ifndef MFA_SCATTERADDKERNEL_HPP_
#define MFA_SCATTERADDKERNEL_HPP_

#include <cstdint>
#include <functional>
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"

struct ScatterAddKernelDescriptor {
	GEMMOperandPrecision memoryPrecision;

	constexpr bool operator==(const ScatterAddKernelDescriptor& rhs) const
	{
		return memoryPrecision == rhs.memoryPrecision;
	}
};

template<>
struct std::hash<ScatterAddKernelDescriptor>
{
	std::size_t operator()(const ScatterAddKernelDescriptor& value) const noexcept
	{
		return std::hash<uint32_t>()((uint32_t)value.memoryPrecision.value);
	}
};

struct ScatterAddKernel {
	NS::SharedPtr<MTL::Library> library;
	std::string source;

	ScatterAddKernel(ScatterAddKernelDescriptor descriptor, MTL::Device* device);
};

#endif
