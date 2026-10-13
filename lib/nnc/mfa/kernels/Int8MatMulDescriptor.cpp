#include "Int8MatMulDescriptor.hpp"
#include "Int8MatMulKernelDescriptor.hpp"
#include "Int8MatMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool Int8MatMulDescriptor::operator==(const Int8MatMulDescriptor& rhs) const
{
	return M == rhs.M && N == rhs.N && K == rhs.K &&
		expertCount == rhs.expertCount && binCount == rhs.binCount &&
		operation == rhs.operation && ioPrecision == rhs.ioPrecision && useBias == rhs.useBias &&
		activationHadamard256 == rhs.activationHadamard256;
}

std::size_t std::hash<Int8MatMulDescriptor>::operator()(const Int8MatMulDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { hash.M, hash.N }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.K, hash.expertCount }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.binCount, (uint32_t)hash.operation }));
	seed = combine_32(seed, pack_32(simd::ushort2 { (uint16_t)hash.ioPrecision.value, (uint16_t)hash.useBias }));
	seed = combine_32(seed, hash.activationHadamard256 ? 1 : 0);
	return seed;
}

std::pair<Int8MatMulKernelDescriptor, PipelineValue<Int8MatMulKernel>*> Int8MatMulDescriptor::findKernel(
	MTL::Device* const device,
	const Int8MatMulKernelDescriptor& kernelDescriptor,
	NS::Array* const binaryArchivesToRead,
	MTL::BinaryArchive* const binaryArchiveToWrite,
	const std::string& pathToWrite,
	std::unordered_map<Int8MatMulKernelDescriptor, std::unique_ptr<Int8MatMulKernel>>* const libraryCache) const noexcept
{
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	CCV_NNC_MFA_PRECONDITION(kernelDescriptor.operation == operation);
	CCV_NNC_MFA_PRECONDITION(!kernelDescriptor.activationHadamard256 || (K > 0 && K <= 65536 && K % 256 == 0));
	CCV_NNC_MFA_PRECONDITION(operation != Int8MatMulRotateActivation || (K > 0 && K % 256 == 0));
	auto iterator = libraryCache->find(kernelDescriptor);
	Int8MatMulKernel* kernel;
	if (iterator != libraryCache->end())
		kernel = iterator->second.get();
	else {
		kernel = new Int8MatMulKernel(kernelDescriptor, device);
		(*libraryCache)[kernelDescriptor] = std::unique_ptr<Int8MatMulKernel>(kernel);
	}
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
	constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
	constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
	constants->setConstantValue(&expertCount, MTL::DataTypeUInt, NS::UInteger(3));
	constants->setConstantValue(&binCount, MTL::DataTypeUInt, NS::UInteger(4));
	const uint64_t weightCount64 = (uint64_t)expertCount * N * K;
	CCV_NNC_MFA_PRECONDITION(weightCount64 <= UINT32_MAX);
	const uint32_t weightCount = (uint32_t)weightCount64;
	constants->setConstantValue(&weightCount, MTL::DataTypeUInt, NS::UInteger(5));
	auto functionName = NS::String::string(kernel->functionName(), NS::UTF8StringEncoding);
	NS::Error* error = nil;
	auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	return std::make_pair(kernelDescriptor, new PipelineValue<Int8MatMulKernel> { kernel, pipeline });
}
