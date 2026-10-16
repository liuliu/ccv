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

	// Metal source defining decode_group(source, group_index, q8) for one packed format, shared with fused kernels.
	static std::string createDecodeGroupSource(uint32_t format) noexcept;

private:
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
