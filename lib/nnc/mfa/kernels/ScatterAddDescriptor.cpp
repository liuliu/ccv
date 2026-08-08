#include "ScatterAddDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool ScatterAddDescriptor::operator==(const ScatterAddDescriptor& rhs) const
{
	return memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<ScatterAddDescriptor>::operator()(const ScatterAddDescriptor& value) const noexcept
{
	return std::hash<uint32_t>()((uint32_t)value.memoryPrecision.value);
}

std::pair<ScatterAddKernelDescriptor, PipelineValue<ScatterAddKernel>*> ScatterAddDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<ScatterAddKernelDescriptor, std::unique_ptr<ScatterAddKernel>>* const libraryCache) const noexcept
{
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	const ScatterAddKernelDescriptor kernel_desc = {
		.memoryPrecision = memoryPrecision,
	};
	ScatterAddKernel* kernel;
	const auto iterator = libraryCache->find(kernel_desc);
	if (iterator != libraryCache->end())
		kernel = iterator->second.get();
	else {
		kernel = new ScatterAddKernel(kernel_desc, device);
		(*libraryCache)[kernel_desc] = std::unique_ptr<ScatterAddKernel>(kernel);
	}
	auto pipeline = [&](const char* const name) {
		NS::Error* error = nil;
		auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
		auto state = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);
		return state;
	};
	PipelineValue<ScatterAddKernel>* const output = new PipelineValue<ScatterAddKernel> {
		kernel, pipeline("scatter_add_single_output")
	};
	output->second = pipeline("scatter_add_clear");
	output->third = pipeline("scatter_add_build_inverse");
	output->fourth = pipeline("scatter_add_sort_inverse");
	output->fifth = pipeline("scatter_add_reduce");
	return std::make_pair(kernel_desc, output);
}
