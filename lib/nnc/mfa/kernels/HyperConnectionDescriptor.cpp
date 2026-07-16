#include "HyperConnectionDescriptor.hpp"
#include "HyperConnectionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include <cstring>

bool HyperConnectionDescriptor::operator==(const HyperConnectionDescriptor& rhs) const {
	return rowCount == rhs.rowCount && count == rhs.count && hidden == rhs.hidden && sinkhornIterations == rhs.sinkhornIterations && epsilon == rhs.epsilon && operation == rhs.operation;
}

std::size_t std::hash<HyperConnectionDescriptor>::operator()(const HyperConnectionDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	uint32_t epsilon_bits;
	static_assert(sizeof(epsilon_bits) == sizeof(hash.epsilon));
	std::memcpy(&epsilon_bits, &hash.epsilon, sizeof(epsilon_bits));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.rowCount, hash.count }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.hidden, hash.sinkhornIterations }));
	seed = combine_64(seed, pack_64(simd::uint2 { epsilon_bits, hash.operation }));
	return seed;
}

std::pair<HyperConnectionKernelDescriptor, PipelineValue<HyperConnectionKernel>*> HyperConnectionDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<HyperConnectionKernelDescriptor, std::unique_ptr<HyperConnectionKernel>>* const libraryCache) const noexcept {
	HyperConnectionKernelDescriptor kernelDesc { 0 };
	auto iterator = libraryCache->find(kernelDesc);
	if (iterator == libraryCache->end())
		iterator = libraryCache->try_emplace(kernelDesc, std::make_unique<HyperConnectionKernel>(kernelDesc, device)).first;
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	constants->setConstantValue(&rowCount, MTL::DataTypeUInt, NS::UInteger(0));
	constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(1));
	constants->setConstantValue(&hidden, MTL::DataTypeUInt, NS::UInteger(2));
	constants->setConstantValue(&sinkhornIterations, MTL::DataTypeUInt, NS::UInteger(3));
	constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(4));
	constants->setConstantValue(&operation, MTL::DataTypeUInt, NS::UInteger(5));
	NS::Error* error = nil;
	auto function = NS::TransferPtr(iterator->second->library->newFunction(NS::String::string("hyper_connection", NS::UTF8StringEncoding), constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto output = new PipelineValue<HyperConnectionKernel> { iterator->second.get(), pipeline };
	return std::make_pair(kernelDesc, output);
}
