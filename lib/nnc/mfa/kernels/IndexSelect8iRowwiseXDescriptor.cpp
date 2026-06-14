#include "IndexSelect8iRowwiseXDescriptor.hpp"
#include "IndexSelect8iRowwiseXKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static uint64_t align_up_64(const uint64_t value, const uint64_t alignment) noexcept
{
	return (value + alignment - 1) & ~(alignment - 1);
}

}

bool IndexSelect8iRowwiseXDescriptor::operator==(const IndexSelect8iRowwiseXDescriptor& rhs) const {
	return
	format == rhs.format &&
	memoryPrecision == rhs.memoryPrecision &&
	rowLength == rhs.rowLength &&
	inputLength == rhs.inputLength &&
	outputLength == rhs.outputLength;
}

uint32_t IndexSelect8iRowwiseXDescriptor::inputRowCount() const noexcept {
	return rowLength > 0 ? inputLength / rowLength : 0;
}

uint32_t IndexSelect8iRowwiseXDescriptor::outputRowCount() const noexcept {
	return rowLength > 0 ? outputLength / rowLength : 0;
}

uint32_t IndexSelect8iRowwiseXDescriptor::groupSize() const noexcept {
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

uint32_t IndexSelect8iRowwiseXDescriptor::groupsPerRow() const noexcept {
	const uint32_t size = groupSize();
	return size > 0 ? (rowLength + size - 1) / size : 0;
}

uint32_t IndexSelect8iRowwiseXDescriptor::groupBits() const noexcept {
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

uint32_t IndexSelect8iRowwiseXDescriptor::inputGroups() const noexcept {
	return inputRowCount() * groupsPerRow();
}

uint32_t IndexSelect8iRowwiseXDescriptor::outputGroups() const noexcept {
	return outputRowCount() * groupsPerRow();
}

uint64_t IndexSelect8iRowwiseXDescriptor::inputScaleOffset() const noexcept {
	const uint64_t payloadBits = (uint64_t)inputGroups() * groupBits();
	const uint64_t payloadBytes = (payloadBits + 7) / 8;
	return align_up_64(payloadBytes, 128);
}

std::size_t std::hash<IndexSelect8iRowwiseXDescriptor>::operator()(const IndexSelect8iRowwiseXDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { hash.format, (uint32_t)hash.memoryPrecision.value }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.rowLength, hash.inputLength }));
	seed = combine_64(seed, hash.outputLength);
	return seed;
}

std::pair<IndexSelect8iRowwiseXKernelDescriptor, PipelineValue<IndexSelect8iRowwiseXKernel>*> IndexSelect8iRowwiseXDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<IndexSelect8iRowwiseXKernelDescriptor, std::unique_ptr<IndexSelect8iRowwiseXKernel>> *const libraryCache) const noexcept {
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	auto createKernel =
	[=](IndexSelect8iRowwiseXKernelDescriptor descriptor) -> IndexSelect8iRowwiseXKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			IndexSelect8iRowwiseXKernel* kernel = new IndexSelect8iRowwiseXKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<IndexSelect8iRowwiseXKernel>(kernel);
			return kernel;
		}
	};

	IndexSelect8iRowwiseXKernelDescriptor kernelDesc;
	kernelDesc.format = format;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowLength = this->rowLength;
		const uint32_t groupSize = this->groupSize();
		const uint32_t groupsPerRow = this->groupsPerRow();
		const uint32_t outputGroups = this->outputGroups();
		constants->setConstantValue(&rowLength, MTL::DataTypeUInt, NS::UInteger(0));
		constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(1));
		constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(2));
		constants->setConstantValue(&outputGroups, MTL::DataTypeUInt, NS::UInteger(6));

		NS::String* swiftName = NS::String::string("index_select_8i_rowwise_x", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	IndexSelect8iRowwiseXKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<IndexSelect8iRowwiseXKernel>* output = new PipelineValue<IndexSelect8iRowwiseXKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
