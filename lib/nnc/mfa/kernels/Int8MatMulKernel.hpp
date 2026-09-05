#ifndef MFA_INT8MATMULKERNEL_HPP_
#define MFA_INT8MATMULKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "Int8MatMulDescriptor.hpp"

struct Int8MatMulKernel {
	NS::SharedPtr<MTL::Library> library;
	std::string source;
	MTL::Size threadgroupSize;

	Int8MatMulKernel(Int8MatMulKernelDescriptor descriptor, MTL::Device* device);

private:
	std::string createSource(uint32_t blockM) const noexcept;
};

#endif
