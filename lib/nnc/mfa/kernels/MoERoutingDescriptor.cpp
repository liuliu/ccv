#include "MoERoutingDescriptor.hpp"
#include "MoERoutingKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

std::pair<MoERoutingKernelDescriptor, PipelineValue<MoERoutingKernel>*> MoERoutingDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<MoERoutingKernelDescriptor, std::unique_ptr<MoERoutingKernel>>* const libraryCache) const noexcept
{
	const MoERoutingKernelDescriptor kernel_descriptor { dataType };
	auto iterator = libraryCache->find(kernel_descriptor);
	if (iterator == libraryCache->end())
		iterator = libraryCache->try_emplace(kernel_descriptor, std::make_unique<MoERoutingKernel>(kernel_descriptor, device)).first;
	NS::Error* error = nil;
	auto function = NS::TransferPtr(iterator->second->library->newFunction(NS::String::string("moe_routing_t1", NS::UTF8StringEncoding)));
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto output = new PipelineValue<MoERoutingKernel> { iterator->second.get(), pipeline };
	return std::make_pair(kernel_descriptor, output);
}
