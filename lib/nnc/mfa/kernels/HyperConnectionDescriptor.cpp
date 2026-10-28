#include "HyperConnectionDescriptor.hpp"
#include "HyperConnectionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include <cstring>

bool HyperConnectionDescriptor::operator==(const HyperConnectionDescriptor& rhs) const {
	return loadM == rhs.loadM && (loadM || rowCount == rhs.rowCount) && count == rhs.count && hidden == rhs.hidden && sinkhornIterations == rhs.sinkhornIterations && epsilon == rhs.epsilon && operation == rhs.operation && blockFP16 == rhs.blockFP16 && normEpsilon == rhs.normEpsilon && expandAddend == rhs.expandAddend && dotSplits == rhs.dotSplits && collapseSplits == rhs.collapseSplits;
}

std::size_t std::hash<HyperConnectionDescriptor>::operator()(const HyperConnectionDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	uint32_t epsilon_bits;
	static_assert(sizeof(epsilon_bits) == sizeof(hash.epsilon));
	std::memcpy(&epsilon_bits, &hash.epsilon, sizeof(epsilon_bits));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.loadM ? 0 : hash.rowCount, hash.count }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.hidden, hash.sinkhornIterations }));
	seed = combine_64(seed, pack_64(simd::uint2 { epsilon_bits, hash.operation }));
	seed = combine_64(seed, hash.blockFP16 ? 1 : 0);
	seed = combine_64(seed, hash.loadM ? 1 : 0);
	uint32_t norm_epsilon_bits;
	std::memcpy(&norm_epsilon_bits, &hash.normEpsilon, sizeof(norm_epsilon_bits));
	seed = combine_64(seed, pack_64(simd::uint2 { norm_epsilon_bits, (uint32_t)hash.expandAddend }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.dotSplits, hash.collapseSplits }));
	return seed;
}

std::pair<HyperConnectionKernelDescriptor, PipelineValue<HyperConnectionKernel>*> HyperConnectionDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<HyperConnectionKernelDescriptor, std::unique_ptr<HyperConnectionKernel>>* const libraryCache) const noexcept {
	HyperConnectionKernelDescriptor kernelDesc { 0, (uint8_t)blockFP16, (uint8_t)loadM };
	auto iterator = libraryCache->find(kernelDesc);
	if (iterator == libraryCache->end())
		iterator = libraryCache->try_emplace(kernelDesc, std::make_unique<HyperConnectionKernel>(kernelDesc, device)).first;
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	if (!loadM)
		constants->setConstantValue(&rowCount, MTL::DataTypeUInt, NS::UInteger(0));
	constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(1));
	constants->setConstantValue(&hidden, MTL::DataTypeUInt, NS::UInteger(2));
	constants->setConstantValue(&sinkhornIterations, MTL::DataTypeUInt, NS::UInteger(3));
	constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(4));
	constants->setConstantValue(&operation, MTL::DataTypeUInt, NS::UInteger(5));
	constants->setConstantValue(&normEpsilon, MTL::DataTypeFloat, NS::UInteger(6));
	constants->setConstantValue(&dotSplits, MTL::DataTypeUInt, NS::UInteger(7));
	constants->setConstantValue(&collapseSplits, MTL::DataTypeUInt, NS::UInteger(8));
	const uint32_t expandAddendValue = expandAddend;
	constants->setConstantValue(&expandAddendValue, MTL::DataTypeUInt, NS::UInteger(9));
	const char* const functionName =
		operation == 3 ? "hyper_connection_mix_partials" :
		operation == 4 ? "hyper_connection_mix_split" :
		operation == 5 ? "hyper_connection_mix_finish" :
		operation == 6 ? "hyper_connection_collapse_norm_rows" : "hyper_connection";
	NS::Error* error = nil;
	auto function = NS::TransferPtr(iterator->second->library->newFunction(NS::String::string(functionName, NS::UTF8StringEncoding), constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto output = new PipelineValue<HyperConnectionKernel> { iterator->second.get(), pipeline };
	return std::make_pair(kernelDesc, output);
}
