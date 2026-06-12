#ifndef MFA_DEQUANTIZE8IROWWISEXFPKERNEL_HPP_
#define MFA_DEQUANTIZE8IROWWISEXFPKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "Dequantize8iRowwiseXFPDescriptor.hpp"

struct Dequantize8iRowwiseXFPKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	MTL::Size threadgroupSize;

	uint32_t format;
	GEMMOperandPrecision memoryPrecision;

	Dequantize8iRowwiseXFPKernel(Dequantize8iRowwiseXFPKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t length) const noexcept;

private:
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
