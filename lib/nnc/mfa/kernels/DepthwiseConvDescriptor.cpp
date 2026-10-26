#include "DepthwiseConvDescriptor.hpp"
#include "DepthwiseConvKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool DepthwiseConvDescriptor::operator==(const DepthwiseConvDescriptor& rhs) const
{
	return
	memoryPrecision == rhs.memoryPrecision &&
	vectorized == rhs.vectorized &&
	fusedBias == rhs.fusedBias;
}

std::size_t std::hash<DepthwiseConvDescriptor>::operator()(const DepthwiseConvDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.vectorized }));
	seed = combine_32(seed, hash.fusedBias);
	return seed;
}

std::pair<DepthwiseConvKernelDescriptor, PipelineValue<DepthwiseConvKernel>*> DepthwiseConvDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<DepthwiseConvKernelDescriptor, std::unique_ptr<DepthwiseConvKernel>>* const libraryCache) const noexcept
{
	auto createKernel =
	[=](DepthwiseConvKernelDescriptor descriptor) -> DepthwiseConvKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			DepthwiseConvKernel* kernel = new DepthwiseConvKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<DepthwiseConvKernel>(kernel);
			return kernel;
		}
	};

	DepthwiseConvKernelDescriptor kernelDesc;
	kernelDesc.vectorized = vectorized;
	kernelDesc.fusedBias = fusedBias;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		NS::String* swiftName = NS::String::string("depthwise_conv", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName));
		CCV_NNC_MFA_PRECONDITION(function.get() != nullptr);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	DepthwiseConvKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<DepthwiseConvKernel>* output = new PipelineValue<DepthwiseConvKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
