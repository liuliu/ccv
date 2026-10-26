#ifndef MFA_DEPTHWISECONVKERNEL_HPP_
#define MFA_DEPTHWISECONVKERNEL_HPP_

#include "ShaderCache.hpp"
#include "DepthwiseConvDescriptor.hpp"

struct DepthwiseConvKernel {
	NS::SharedPtr<MTL::Library> library;

	uint8_t vectorized;

	uint8_t fusedBias;

	GEMMOperandPrecision memoryPrecision;

	MTL::Size threadgroupSize;

	DepthwiseConvKernel(DepthwiseConvKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t elementCount) const noexcept;

	std::string createSource() const noexcept;
};

#endif
