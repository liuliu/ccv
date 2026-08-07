#include "RMSNormCmulDescriptor.hpp"
#include "RMSNormCmulKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool RMSNormCmulKernelDescriptor::operator==(const RMSNormCmulKernelDescriptor& rhs) const
{
	return aPrecision == rhs.aPrecision && rotationPrecision == rhs.rotationPrecision && scalePrecision == rhs.scalePrecision && elementwiseAffine == rhs.elementwiseAffine;
}

bool RMSNormCmulDescriptor::operator==(const RMSNormCmulDescriptor& rhs) const
{
	return epsilon == rhs.epsilon && aPrecision == rhs.aPrecision && rotationPrecision == rhs.rotationPrecision && scalePrecision == rhs.scalePrecision && columnCount == rhs.columnCount && broadcastRatio == rhs.broadcastRatio && rowsPerThreadgroup == rhs.rowsPerThreadgroup && elementwiseAffine == rhs.elementwiseAffine;
}

std::size_t std::hash<RMSNormCmulKernelDescriptor>::operator()(const RMSNormCmulKernelDescriptor& value) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)value.aPrecision.value, (unsigned int)value.rotationPrecision.value }));
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)value.scalePrecision.value, (unsigned int)value.elementwiseAffine }));
	return seed;
}

std::size_t std::hash<RMSNormCmulDescriptor>::operator()(const RMSNormCmulDescriptor& value) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)value.aPrecision.value, (unsigned int)value.rotationPrecision.value }));
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)value.scalePrecision.value, (unsigned int)value.elementwiseAffine }));
	combine_64(seed, pack_64(simd::uint2 { value.columnCount, value.broadcastRatio }));
	combine_32(seed, value.rowsPerThreadgroup);
	combine_32(seed, *reinterpret_cast<const uint32_t*>(&value.epsilon));
	return seed;
}

std::pair<RMSNormCmulKernelDescriptor, PipelineValue<RMSNormCmulKernel>*> RMSNormCmulDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RMSNormCmulKernelDescriptor, std::unique_ptr<RMSNormCmulKernel>>* const libraryCache) const noexcept
{
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	const RMSNormCmulKernelDescriptor kernel_desc = {
		.aPrecision = aPrecision,
		.rotationPrecision = rotationPrecision,
		.scalePrecision = scalePrecision,
		.elementwiseAffine = elementwiseAffine,
	};
	RMSNormCmulKernel* kernel;
	const auto iterator = libraryCache->find(kernel_desc);
	if (iterator != libraryCache->end())
		kernel = iterator->second.get();
	else {
		kernel = new RMSNormCmulKernel(kernel_desc, device);
		(*libraryCache)[kernel_desc] = std::unique_ptr<RMSNormCmulKernel>(kernel);
	}
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	constants->setConstantValue(&columnCount, MTL::DataTypeUInt, NS::UInteger(0));
	constants->setConstantValue(&broadcastRatio, MTL::DataTypeUInt, NS::UInteger(1));
	constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(2));
	constants->setConstantValue(&rowsPerThreadgroup, MTL::DataTypeUInt, NS::UInteger(3));
	NS::Error* error = nil;
	auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("rmsnorm_cmul", NS::UTF8StringEncoding), constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	return std::make_pair(kernel_desc, new PipelineValue<RMSNormCmulKernel> { kernel, pipeline });
}
