#include "MoERoutingDescriptor.hpp"
#include "MoERoutingKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool MoERoutingDescriptor::operator==(const MoERoutingDescriptor& rhs) const
{
	return activationDataType == rhs.activationDataType &&
		routingDataType == rhs.routingDataType &&
		expertCount == rhs.expertCount &&
		kth == rhs.kth &&
		hidden == rhs.hidden &&
		weightScale == rhs.weightScale &&
		preselected == rhs.preselected &&
		singleInputToken == rhs.singleInputToken;
}

std::size_t std::hash<MoERoutingDescriptor>::operator()(const MoERoutingDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_32(seed, hash.activationDataType);
	seed = combine_32(seed, hash.routingDataType);
	seed = combine_32(seed, hash.expertCount);
	seed = combine_32(seed, hash.kth);
	seed = combine_32(seed, hash.hidden);
	seed = combine_32(seed, reinterpret_cast<const uint32_t&>(hash.weightScale));
	seed = combine_32(seed, hash.preselected ? 1 : 0);
	seed = combine_32(seed, hash.singleInputToken ? 1 : 0);
	return seed;
}

std::pair<MoERoutingKernelDescriptor, PipelineValue<MoERoutingKernel>*> MoERoutingDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<MoERoutingKernelDescriptor, std::unique_ptr<MoERoutingKernel>>* const libraryCache) const noexcept
{
	const MoERoutingKernelDescriptor kernel_descriptor { activationDataType, routingDataType };
	auto iterator = libraryCache->find(kernel_descriptor);
	if (iterator == libraryCache->end())
		iterator = libraryCache->try_emplace(kernel_descriptor, std::make_unique<MoERoutingKernel>(kernel_descriptor, device)).first;
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	constants->setConstantValue(&expertCount, MTL::DataTypeUInt, NS::UInteger(0));
	constants->setConstantValue(&kth, MTL::DataTypeUInt, NS::UInteger(1));
	constants->setConstantValue(&hidden, MTL::DataTypeUInt, NS::UInteger(2));
	constants->setConstantValue(&weightScale, MTL::DataTypeFloat, NS::UInteger(3));
	constants->setConstantValue(&preselected, MTL::DataTypeBool, NS::UInteger(4));
	constants->setConstantValue(&singleInputToken, MTL::DataTypeBool, NS::UInteger(5));
	NS::Error* error = nil;
	auto function = NS::TransferPtr(iterator->second->library->newFunction(NS::String::string("moe_routing_t1", NS::UTF8StringEncoding), constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto output = new PipelineValue<MoERoutingKernel> { iterator->second.get(), pipeline };
	return std::make_pair(kernel_descriptor, output);
}
