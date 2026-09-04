#ifndef MFA_MOEWEIGHTSSTREAMINGKERNEL_HPP_
#define MFA_MOEWEIGHTSSTREAMINGKERNEL_HPP_

#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include "MoEWeightsStreamingDescriptor.hpp"

struct MoEWeightsStreamingKernel {
	NS::SharedPtr<MTL::Library> library;

	MoEWeightsStreamingKernel(MoEWeightsStreamingKernelDescriptor descriptor,
		MTL::Device* device);
};

#endif
