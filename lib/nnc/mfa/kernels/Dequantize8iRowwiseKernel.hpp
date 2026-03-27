#ifndef Dequantize8iRowwiseKernel_hpp
#define Dequantize8iRowwiseKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>
#include "Dequantize8iRowwiseDescriptor.hpp"

struct Dequantize8iRowwiseKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	unsigned short threadgroupMemoryAllocation;

	MTL::Size threadgroupSize;

	uint8_t vectorized;
	GEMMOperandPrecision memoryPrecision;

	Dequantize8iRowwiseKernel(Dequantize8iRowwiseKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t length) const noexcept;

private:
	unsigned short createThreadgroupMemoryAllocation() const noexcept;
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
