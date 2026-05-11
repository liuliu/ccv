#ifndef MFA_DEQUANTIZE8IROWWISEXKERNEL_HPP_
#define MFA_DEQUANTIZE8IROWWISEXKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "Dequantize8iRowwiseXDescriptor.hpp"

struct Dequantize8iRowwiseXKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	MTL::Size threadgroupSize;

	uint32_t format;

	Dequantize8iRowwiseXKernel(Dequantize8iRowwiseXKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t dispatchItems) const noexcept;

private:
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
