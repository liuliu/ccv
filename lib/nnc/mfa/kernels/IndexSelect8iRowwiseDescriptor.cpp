#include "IndexSelect8iRowwiseDescriptor.hpp"
#include "IndexSelect8iRowwiseKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool IndexSelect8iRowwiseDescriptor::operator==(const IndexSelect8iRowwiseDescriptor& rhs) const {
	return
	memoryPrecision == rhs.memoryPrecision &&
	rowLength == rhs.rowLength &&
	inputLength == rhs.inputLength &&
	outputLength == rhs.outputLength;
}

bool IndexSelect8iRowwiseDescriptor::vectorized() const noexcept {
	return (rowLength % 4) == 0 && (outputLength % 4) == 0;
}

std::size_t std::hash<IndexSelect8iRowwiseDescriptor>::operator()(const IndexSelect8iRowwiseDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.rowLength }));
	combine_64(seed, pack_64(simd::uint2 { hash.inputLength, hash.outputLength }));
	return seed;
}

std::pair<IndexSelect8iRowwiseKernelDescriptor, PipelineValue<IndexSelect8iRowwiseKernel>*> IndexSelect8iRowwiseDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelect8iRowwiseKernelDescriptor, std::unique_ptr<IndexSelect8iRowwiseKernel>> *const libraryCache) const noexcept {
	auto createKernel =
	[=](IndexSelect8iRowwiseKernelDescriptor descriptor) -> IndexSelect8iRowwiseKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			IndexSelect8iRowwiseKernel* kernel = new IndexSelect8iRowwiseKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<IndexSelect8iRowwiseKernel>(kernel);
			return kernel;
		}
	};

	IndexSelect8iRowwiseKernelDescriptor kernelDesc;
	kernelDesc.vectorized = vectorized() ? 1 : 0;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowUnits = vectorized() ? (rowLength / 4) : rowLength;
		const uint32_t elementCount = vectorized() ? (outputLength / 4) : outputLength;
			const uint64_t scaleOffset = ((uint64_t)inputLength + 127) & ~UINT64_C(127);
			constants->setConstantValue(&rowUnits, MTL::DataTypeUInt, NS::UInteger(0));
			constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(1));
			constants->setConstantValue(&scaleOffset, MTL::DataTypeULong, NS::UInteger(2));

		NS::String* swiftName = NS::String::string("index_select_8i_rowwise", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	IndexSelect8iRowwiseKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<IndexSelect8iRowwiseKernel>* output = new PipelineValue<IndexSelect8iRowwiseKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
