#ifndef IndexSelectKernel_hpp
#define IndexSelectKernel_hpp

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "IndexSelectDescriptor.hpp"

struct IndexSelectKernel {
	NS::SharedPtr<MTL::Library> library;

	IndexSelectKernel(IndexSelectKernelDescriptor descriptor, MTL::Device* const device);

	MTL::Size gridSize(uint32_t outputRows, uint16_t threadsPerRow) const noexcept;
	MTL::Size threadgroupSize(uint32_t outputRows, uint16_t threadsPerRow) const noexcept;
};

#endif
