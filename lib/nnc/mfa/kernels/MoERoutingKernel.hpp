#ifndef MFA_MOEROUTINGKERNEL_HPP_
#define MFA_MOEROUTINGKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "MoERoutingDescriptor.hpp"

struct MoERoutingKernel {
	NS::SharedPtr<MTL::Library> library;
	MoERoutingKernel(MoERoutingKernelDescriptor descriptor, MTL::Device* const device);
};

#endif
