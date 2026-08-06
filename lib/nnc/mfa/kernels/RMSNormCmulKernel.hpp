#ifndef MFA_RMSNORMCMULKERNEL_HPP_
#define MFA_RMSNORMCMULKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "RMSNormCmulDescriptor.hpp"

struct RMSNormCmulKernel {
	NS::SharedPtr<MTL::Library> library;
	std::string source;
	MTL::Size threadgroupSize;

	RMSNormCmulKernel(RMSNormCmulKernelDescriptor descriptor, MTL::Device* const device);
};

#endif
