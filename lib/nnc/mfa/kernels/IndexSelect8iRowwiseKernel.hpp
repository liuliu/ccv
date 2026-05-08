#ifndef IndexSelect8iRowwiseKernel_hpp
#define IndexSelect8iRowwiseKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "IndexSelect8iRowwiseDescriptor.hpp"

struct IndexSelect8iRowwiseKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	unsigned short threadgroupMemoryAllocation;

	MTL::Size threadgroupSize;

	uint8_t vectorized;
	GEMMOperandPrecision memoryPrecision;

	IndexSelect8iRowwiseKernel(IndexSelect8iRowwiseKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t outputLength) const noexcept;

private:
	unsigned short createThreadgroupMemoryAllocation() const noexcept;
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
