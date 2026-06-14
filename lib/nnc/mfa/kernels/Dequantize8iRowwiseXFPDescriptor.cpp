#include "Dequantize8iRowwiseXFPDescriptor.hpp"
#include "Dequantize8iRowwiseXFPKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static uint64_t align_up_64(const uint64_t value, const uint64_t alignment) noexcept
{
	return (value + alignment - 1) & ~(alignment - 1);
}

}

bool Dequantize8iRowwiseXFPDescriptor::operator==(const Dequantize8iRowwiseXFPDescriptor& rhs) const {
	return
	format == rhs.format &&
	memoryPrecision == rhs.memoryPrecision &&
	rowLength == rhs.rowLength &&
	length == rhs.length;
}

uint32_t Dequantize8iRowwiseXFPDescriptor::rowCount() const noexcept {
	return rowLength > 0 ? length / rowLength : 0;
}

uint32_t Dequantize8iRowwiseXFPDescriptor::groupSize() const noexcept {
	switch (format) {
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			return 16;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			return 32;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			return 8;
		default:
			CCV_NNC_MFA_PRECONDITION(false);
			return 0;
	}
}

uint32_t Dequantize8iRowwiseXFPDescriptor::groupsPerRow() const noexcept {
	const uint32_t size = groupSize();
	return size > 0 ? (rowLength + size - 1) / size : 0;
}

uint32_t Dequantize8iRowwiseXFPDescriptor::groupBits() const noexcept {
	switch (format) {
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
			return 88;
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			return 72;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			return 56;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			return 42;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			return 21;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			return 28;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			return 64;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
			return 52;
		default:
			CCV_NNC_MFA_PRECONDITION(false);
			return 0;
	}
}

uint32_t Dequantize8iRowwiseXFPDescriptor::totalGroups() const noexcept {
	return rowCount() * groupsPerRow();
}

uint64_t Dequantize8iRowwiseXFPDescriptor::inputScaleOffset() const noexcept {
	const uint64_t payloadBits = (uint64_t)totalGroups() * groupBits();
	const uint64_t payloadBytes = (payloadBits + 7) / 8;
	return align_up_64(payloadBytes, 128);
}

std::size_t std::hash<Dequantize8iRowwiseXFPDescriptor>::operator()(const Dequantize8iRowwiseXFPDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { hash.format, (uint32_t)hash.memoryPrecision.value }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.rowLength, hash.length }));
	return seed;
}

std::pair<Dequantize8iRowwiseXFPKernelDescriptor, PipelineValue<Dequantize8iRowwiseXFPKernel>*> Dequantize8iRowwiseXFPDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXFPKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXFPKernel>> *const libraryCache) const noexcept {
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	auto createKernel =
	[=](Dequantize8iRowwiseXFPKernelDescriptor descriptor) -> Dequantize8iRowwiseXFPKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			Dequantize8iRowwiseXFPKernel* kernel = new Dequantize8iRowwiseXFPKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<Dequantize8iRowwiseXFPKernel>(kernel);
			return kernel;
		}
	};

	Dequantize8iRowwiseXFPKernelDescriptor kernelDesc;
	kernelDesc.format = format;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowLength = this->rowLength;
		const uint32_t groupSize = this->groupSize();
		const uint32_t groupsPerRow = this->groupsPerRow();
		const uint32_t totalGroups = this->totalGroups();
		constants->setConstantValue(&rowLength, MTL::DataTypeUInt, NS::UInteger(0));
		constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(1));
		constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(2));
		constants->setConstantValue(&totalGroups, MTL::DataTypeUInt, NS::UInteger(6));

		NS::String* swiftName = NS::String::string("dequantize_8i_rowwise_x_fp", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	Dequantize8iRowwiseXFPKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<Dequantize8iRowwiseXFPKernel>* output = new PipelineValue<Dequantize8iRowwiseXFPKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
