#ifndef MFA_INDEXSELECT8IROWWISEXKERNEL_HPP_
#define MFA_INDEXSELECT8IROWWISEXKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "IndexSelect8iRowwiseXDescriptor.hpp"

struct IndexSelect8iRowwiseXKernel {
	NS::SharedPtr<MTL::Library> library;

	std::string source;

	MTL::Size threadgroupSize;

	uint32_t format;
	GEMMOperandPrecision memoryPrecision;

	IndexSelect8iRowwiseXKernel(IndexSelect8iRowwiseXKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t length) const noexcept;

private:
	std::string createSource() const noexcept;
	std::string createConstants() const noexcept;
};

#endif
