#include "Dequantize8iRowwiseDescriptor.hpp"
#include "Dequantize8iRowwiseKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool Dequantize8iRowwiseDescriptor::operator==(const Dequantize8iRowwiseDescriptor& rhs) const {
	return
	memoryPrecision == rhs.memoryPrecision &&
	rowLength == rhs.rowLength &&
	length == rhs.length;
}

bool Dequantize8iRowwiseDescriptor::vectorized() const noexcept {
	return (rowLength % 4) == 0 && (length % 4) == 0;
}

std::size_t std::hash<Dequantize8iRowwiseDescriptor>::operator()(const Dequantize8iRowwiseDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.rowLength }));
	combine_32(seed, hash.length);
	return seed;
}

std::pair<Dequantize8iRowwiseKernelDescriptor, PipelineValue<Dequantize8iRowwiseKernel>*> Dequantize8iRowwiseDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseKernel>> *const libraryCache) const noexcept {
	auto createKernel =
	[=](Dequantize8iRowwiseKernelDescriptor descriptor) -> Dequantize8iRowwiseKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			Dequantize8iRowwiseKernel* kernel = new Dequantize8iRowwiseKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<Dequantize8iRowwiseKernel>(kernel);
			return kernel;
		}
	};

	Dequantize8iRowwiseKernelDescriptor kernelDesc;
	kernelDesc.vectorized = vectorized() ? 1 : 0;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowUnits = vectorized() ? (rowLength / 4) : rowLength;
		const uint32_t elementCount = vectorized() ? (length / 4) : length;
		constants->setConstantValue(&rowUnits, MTL::DataTypeUInt, NS::UInteger(0));
		constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(1));

		NS::String* swiftName = NS::String::string("dequantize_8i_rowwise", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	Dequantize8iRowwiseKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<Dequantize8iRowwiseKernel>* output = new PipelineValue<Dequantize8iRowwiseKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
