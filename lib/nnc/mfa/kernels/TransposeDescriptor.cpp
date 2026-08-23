#include "TransposeDescriptor.hpp"
#include "TransposeKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool TransposeDescriptor::operator==(const TransposeDescriptor& rhs) const
{
	return memoryPrecision == rhs.memoryPrecision;
}

std::size_t std::hash<TransposeDescriptor>::operator()(const TransposeDescriptor& hash) const noexcept
{
	return std::hash<int>()((int)hash.memoryPrecision.value);
}

std::pair<TransposeKernelDescriptor, PipelineValue<TransposeKernel>*> TransposeDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<TransposeKernelDescriptor, std::unique_ptr<TransposeKernel>>* const libraryCache) const noexcept
{
	auto createKernel =
	[=](TransposeKernelDescriptor descriptor) -> TransposeKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			TransposeKernel* kernel = new TransposeKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<TransposeKernel>(kernel);
			return kernel;
		}
	};

	const TransposeKernelDescriptor kernelDesc = {
		.memoryPrecision = memoryPrecision,
	};
	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		NS::String* functionName = NS::String::string("transpose", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(functionName));
		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	TransposeKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<TransposeKernel>* output = new PipelineValue<TransposeKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
