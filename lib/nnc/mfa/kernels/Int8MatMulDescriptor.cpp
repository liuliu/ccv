#include "Int8MatMulDescriptor.hpp"
#include "Int8MatMulKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool Int8MatMulDescriptor::operator==(const Int8MatMulDescriptor& rhs) const
{
	return M == rhs.M && N == rhs.N && K == rhs.K &&
		expertCount == rhs.expertCount && binCount == rhs.binCount &&
		operation == rhs.operation;
}

std::size_t std::hash<Int8MatMulDescriptor>::operator()(const Int8MatMulDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { hash.M, hash.N }));
	combine_64(seed, pack_64(simd::uint2 { hash.K, hash.expertCount }));
	combine_64(seed, pack_64(simd::uint2 { hash.binCount, (uint32_t)hash.operation }));
	return seed;
}

std::pair<Int8MatMulKernelDescriptor, PipelineValue<Int8MatMulKernel>*> Int8MatMulDescriptor::findKernel(
	MTL::Device* const device,
	const DeviceProperties& dprops,
	NS::Array* const binaryArchivesToRead,
	MTL::BinaryArchive* const binaryArchiveToWrite,
	const std::string& pathToWrite,
	std::unordered_map<Int8MatMulKernelDescriptor, std::unique_ptr<Int8MatMulKernel>>* const libraryCache) const noexcept
{
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	const Int8MatMulKernelDescriptor kernelDesc;
	auto iterator = libraryCache->find(kernelDesc);
	Int8MatMulKernel* kernel;
	if (iterator != libraryCache->end())
		kernel = iterator->second.get();
	else {
		kernel = new Int8MatMulKernel(kernelDesc, device);
		(*libraryCache)[kernelDesc] = std::unique_ptr<Int8MatMulKernel>(kernel);
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
	const char* functionName = nullptr;
	switch (operation) {
		case Int8MatMulQuantizeActivation:
			functionName = "int8_matmul_quantize_activation";
			break;
		case Int8MatMulCastWeights:
			functionName = "int8_matmul_cast_weights";
			break;
		case Int8MatMulDequantizeOutput:
			functionName = "int8_matmul_dequantize_output";
			break;
		case Int8MatMulDequantizeSegmentedOutput:
			functionName = "int8_matmul_dequantize_segmented_output";
			break;
		case Int8MatMulSegmented:
			functionName = "int8_matmul_segmented";
			break;
	}
	CCV_NNC_MFA_PRECONDITION(functionName);
	auto functionNameString = NS::String::string(functionName, NS::UTF8StringEncoding);
	NS::Error* error = nil;
	auto function = NS::TransferPtr(kernel->library->newFunction(functionNameString, constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	return std::make_pair(kernelDesc, new PipelineValue<Int8MatMulKernel> { kernel, pipeline });
}
