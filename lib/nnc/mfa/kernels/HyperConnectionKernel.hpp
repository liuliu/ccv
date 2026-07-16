#ifndef MFA_HYPERCONNECTIONKERNEL_HPP_
#define MFA_HYPERCONNECTIONKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "HyperConnectionDescriptor.hpp"

struct HyperConnectionKernel {
	NS::SharedPtr<MTL::Library> library;
	HyperConnectionKernel(HyperConnectionKernelDescriptor descriptor, MTL::Device* const device);
};

#endif
