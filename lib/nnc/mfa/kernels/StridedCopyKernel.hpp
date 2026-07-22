#ifndef MFA_STRIDEDCOPYKERNEL_HPP_
#define MFA_STRIDEDCOPYKERNEL_HPP_

#include "ShaderCache.hpp"
#include "StridedCopyDescriptor.hpp"

struct StridedCopyKernel {
	NS::SharedPtr<MTL::Library> library;

	uint8_t vectorized;

	uint8_t destinationStrided;

	bool loadM;

	GEMMOperandPrecision memoryPrecision;

	MTL::Size threadgroupSize;

	StridedCopyKernel(StridedCopyKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t rows, uint32_t cols) const noexcept;

	std::string createSource() const noexcept;

	std::string createConstants() const noexcept;
};

#endif
