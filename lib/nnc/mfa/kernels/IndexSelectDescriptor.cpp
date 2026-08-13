#include "IndexSelectDescriptor.hpp"
#include "IndexSelectKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

bool IndexSelectDescriptor::operator==(const IndexSelectDescriptor& rhs) const
{
	return dataType == rhs.dataType &&
		vectorWidth == rhs.vectorWidth &&
		threadsPerRow == rhs.threadsPerRow &&
		loadM == rhs.loadM &&
		(loadM || outputRows == rhs.outputRows);
}

std::size_t std::hash<IndexSelectDescriptor>::operator()(const IndexSelectDescriptor& value) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_32(seed, (uint32_t)value.dataType);
	combine_32(seed, (uint32_t)value.vectorWidth);
	combine_32(seed, (uint32_t)value.threadsPerRow);
	combine_32(seed, value.loadM ? 0 : value.outputRows);
	combine_32(seed, value.loadM ? 1 : 0);
	return seed;
}

std::pair<IndexSelectKernelDescriptor, PipelineValue<IndexSelectKernel>*> IndexSelectDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelectKernelDescriptor, std::unique_ptr<IndexSelectKernel>>* const libraryCache) const noexcept
{
	const IndexSelectKernelDescriptor kernel_descriptor = {
		.vectorWidth = vectorWidth,
		.loadM = loadM,
		.dataType = dataType,
	};
	auto iterator = libraryCache->find(kernel_descriptor);
	IndexSelectKernel* kernel;
	if (iterator != libraryCache->end())
		kernel = iterator->second.get();
	else {
		kernel = new IndexSelectKernel(kernel_descriptor, device);
		(*libraryCache)[kernel_descriptor] = std::unique_ptr<IndexSelectKernel>(kernel);
	}

	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	const uint16_t rows_per_threadgroup = 256 / threadsPerRow;
	constants->setConstantValue(&threadsPerRow, MTL::DataTypeUShort, NS::UInteger(0));
	constants->setConstantValue(&rows_per_threadgroup, MTL::DataTypeUShort, NS::UInteger(1));
	if (!loadM)
		constants->setConstantValue(&outputRows, MTL::DataTypeUInt, NS::UInteger(2));
	NS::Error* error = nil;
	auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string("index_select", NS::UTF8StringEncoding), constants.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	return std::make_pair(kernel_descriptor, new PipelineValue<IndexSelectKernel> { kernel, pipeline });
}
