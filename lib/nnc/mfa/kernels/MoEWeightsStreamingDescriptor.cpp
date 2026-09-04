#include "MoEWeightsStreamingDescriptor.hpp"
#include "MoEWeightsStreamingKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::pair<MoEWeightsStreamingKernelDescriptor, PipelineValue<MoEWeightsStreamingKernel>*>
MoEWeightsStreamingDescriptor::findKernel(
	MTL::Device* const device, const DeviceProperties&, NS::Array* const,
	MTL::BinaryArchive* const, const std::string&,
	std::unordered_map<MoEWeightsStreamingKernelDescriptor,
		std::unique_ptr<MoEWeightsStreamingKernel>>* const libraryCache) const noexcept
{
	const MoEWeightsStreamingKernelDescriptor kernel_descriptor { version };
	auto iterator = libraryCache->find(kernel_descriptor);
	if (iterator == libraryCache->end())
		iterator = libraryCache->try_emplace(kernel_descriptor,
			std::make_unique<MoEWeightsStreamingKernel>(kernel_descriptor, device)).first;
	NS::Error* error = nil;
	auto function = NS::TransferPtr(iterator->second->library->newFunction(
		NS::String::string("moe_weights_streaming", NS::UTF8StringEncoding)));
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto output = new PipelineValue<MoEWeightsStreamingKernel> {
		iterator->second.get(), pipeline
	};
	return std::make_pair(kernel_descriptor, output);
}
