#include "Dequantize8iRowwiseXDescriptor.hpp"
#include "Dequantize8iRowwiseXKernel.hpp"
#include "../ccv_nnc_mfa.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

#include <algorithm>

namespace {

static uint32_t align_up_32(const uint32_t value, const uint32_t alignment) noexcept
{
	return (value + alignment - 1) & ~(alignment - 1);
}

}

bool Dequantize8iRowwiseXDescriptor::operator==(const Dequantize8iRowwiseXDescriptor& rhs) const {
	return
	format == rhs.format &&
	scaleSize == rhs.scaleSize &&
	rowLength == rhs.rowLength &&
	length == rhs.length;
}

uint32_t Dequantize8iRowwiseXDescriptor::rowCount() const noexcept {
	return rowLength > 0 ? length / rowLength : 0;
}

uint32_t Dequantize8iRowwiseXDescriptor::groupSize() const noexcept {
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

uint32_t Dequantize8iRowwiseXDescriptor::groupsPerRow() const noexcept {
	const uint32_t size = groupSize();
	return size > 0 ? (rowLength + size - 1) / size : 0;
}

uint32_t Dequantize8iRowwiseXDescriptor::groupBits() const noexcept {
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

uint32_t Dequantize8iRowwiseXDescriptor::totalGroups() const noexcept {
	return rowCount() * groupsPerRow();
}

uint32_t Dequantize8iRowwiseXDescriptor::inputScaleOffset() const noexcept {
	const uint64_t payloadBits = (uint64_t)totalGroups() * groupBits();
	const uint32_t payloadBytes = (uint32_t)((payloadBits + 7) / 8);
	return align_up_32(payloadBytes, 128);
}

uint32_t Dequantize8iRowwiseXDescriptor::outputScaleOffset() const noexcept {
	return align_up_32(length, 128);
}

uint32_t Dequantize8iRowwiseXDescriptor::scaleBytes() const noexcept {
	return rowCount() * scaleSize;
}

uint32_t Dequantize8iRowwiseXDescriptor::dispatchItems() const noexcept {
	return std::max(totalGroups(), scaleBytes());
}

std::size_t std::hash<Dequantize8iRowwiseXDescriptor>::operator()(const Dequantize8iRowwiseXDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { hash.format, hash.scaleSize }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.rowLength, hash.length }));
	return seed;
}

std::pair<Dequantize8iRowwiseXKernelDescriptor, PipelineValue<Dequantize8iRowwiseXKernel>*> Dequantize8iRowwiseXDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXKernel>> *const libraryCache) const noexcept {
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	auto createKernel =
	[=](Dequantize8iRowwiseXKernelDescriptor descriptor) -> Dequantize8iRowwiseXKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			Dequantize8iRowwiseXKernel* kernel = new Dequantize8iRowwiseXKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<Dequantize8iRowwiseXKernel>(kernel);
			return kernel;
		}
	};

	Dequantize8iRowwiseXKernelDescriptor kernelDesc;
	kernelDesc.format = format;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowLength = this->rowLength;
		const uint32_t groupSize = this->groupSize();
		const uint32_t groupsPerRow = this->groupsPerRow();
		const uint32_t inputScaleOffset = this->inputScaleOffset();
		const uint32_t outputScaleOffset = this->outputScaleOffset();
		const uint32_t totalGroups = this->totalGroups();
		const uint32_t scaleBytes = this->scaleBytes();
		const uint32_t dispatchItems = this->dispatchItems();
		constants->setConstantValue(&rowLength, MTL::DataTypeUInt, NS::UInteger(0));
		constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(1));
		constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(2));
		constants->setConstantValue(&inputScaleOffset, MTL::DataTypeUInt, NS::UInteger(4));
		constants->setConstantValue(&outputScaleOffset, MTL::DataTypeUInt, NS::UInteger(5));
		constants->setConstantValue(&totalGroups, MTL::DataTypeUInt, NS::UInteger(6));
		constants->setConstantValue(&scaleBytes, MTL::DataTypeUInt, NS::UInteger(7));
		constants->setConstantValue(&dispatchItems, MTL::DataTypeUInt, NS::UInteger(8));

		NS::String* swiftName = NS::String::string("dequantize_8i_rowwise_x", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	Dequantize8iRowwiseXKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<Dequantize8iRowwiseXKernel>* output = new PipelineValue<Dequantize8iRowwiseXKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}

bool Dequantize8iRowwiseXSelectedDescriptor::operator==(const Dequantize8iRowwiseXSelectedDescriptor& rhs) const {
	return
	format == rhs.format &&
	scaleSize == rhs.scaleSize &&
	rowLength == rhs.rowLength &&
	rowsPerExpert == rhs.rowsPerExpert &&
	expertCount == rhs.expertCount &&
	segmentCount == rhs.segmentCount;
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::groupSize() const noexcept {
	Dequantize8iRowwiseXDescriptor descriptor = {
		.format = format,
		.scaleSize = scaleSize,
		.rowLength = rowLength,
		.length = rowsPerExpert * rowLength,
	};
	return descriptor.groupSize();
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::groupsPerRow() const noexcept {
	Dequantize8iRowwiseXDescriptor descriptor = {
		.format = format,
		.scaleSize = scaleSize,
		.rowLength = rowLength,
		.length = rowsPerExpert * rowLength,
	};
	return descriptor.groupsPerRow();
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::groupBits() const noexcept {
	Dequantize8iRowwiseXDescriptor descriptor = {
		.format = format,
		.scaleSize = scaleSize,
		.rowLength = rowLength,
		.length = rowsPerExpert * rowLength,
	};
	return descriptor.groupBits();
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::groupsPerExpert() const noexcept {
	return rowsPerExpert * groupsPerRow();
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::inputScaleOffset() const noexcept {
	const uint64_t payloadBits = (uint64_t)expertCount * groupsPerExpert() * groupBits();
	const uint32_t payloadBytes = (uint32_t)((payloadBits + 7) / 8);
	return align_up_32(payloadBytes, 128);
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::outputScaleOffset() const noexcept {
	return align_up_32((uint32_t)((uint64_t)expertCount * rowsPerExpert * rowLength), 128);
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::scaleBytesPerExpert() const noexcept {
	return rowsPerExpert * scaleSize;
}

uint32_t Dequantize8iRowwiseXSelectedDescriptor::dispatchItemsPerExpert() const noexcept {
	return std::max(groupsPerExpert(), scaleBytesPerExpert());
}

std::size_t std::hash<Dequantize8iRowwiseXSelectedDescriptor>::operator()(const Dequantize8iRowwiseXSelectedDescriptor& hash) const noexcept {
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	seed = combine_64(seed, pack_64(simd::uint2 { hash.format, hash.scaleSize }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.rowLength, hash.rowsPerExpert }));
	seed = combine_64(seed, pack_64(simd::uint2 { hash.expertCount, hash.segmentCount }));
	return seed;
}

std::pair<Dequantize8iRowwiseXKernelDescriptor, PipelineValue<Dequantize8iRowwiseXKernel>*> Dequantize8iRowwiseXSelectedDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<Dequantize8iRowwiseXKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseXKernel>> *const libraryCache) const noexcept {
	(void)dprops;
	(void)binaryArchivesToRead;
	(void)binaryArchiveToWrite;
	(void)pathToWrite;
	auto createKernel =
	[=](Dequantize8iRowwiseXKernelDescriptor descriptor) -> Dequantize8iRowwiseXKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			Dequantize8iRowwiseXKernel* kernel = new Dequantize8iRowwiseXKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<Dequantize8iRowwiseXKernel>(kernel);
			return kernel;
		}
	};

	Dequantize8iRowwiseXKernelDescriptor kernelDesc;
	kernelDesc.format = format;

	auto createPipeline =
	[=](MTL::Library* library, const char* const functionName) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		const uint32_t rowLength = this->rowLength;
		const uint32_t groupSize = this->groupSize();
		const uint32_t groupsPerRow = this->groupsPerRow();
		const uint32_t inputScaleOffset = this->inputScaleOffset();
		const uint32_t outputScaleOffset = this->outputScaleOffset();
		const uint32_t groupsPerExpert = this->groupsPerExpert();
		const uint32_t scaleBytesPerExpert = this->scaleBytesPerExpert();
		const uint32_t dispatchItemsPerExpert = this->dispatchItemsPerExpert();
		const uint32_t rowsPerExpert = this->rowsPerExpert;
		const uint32_t expertCount = this->expertCount;
		const uint32_t segmentCount = this->segmentCount;
		constants->setConstantValue(&rowLength, MTL::DataTypeUInt, NS::UInteger(0));
		constants->setConstantValue(&groupSize, MTL::DataTypeUInt, NS::UInteger(1));
		constants->setConstantValue(&groupsPerRow, MTL::DataTypeUInt, NS::UInteger(2));
		constants->setConstantValue(&inputScaleOffset, MTL::DataTypeUInt, NS::UInteger(4));
		constants->setConstantValue(&outputScaleOffset, MTL::DataTypeUInt, NS::UInteger(5));
		constants->setConstantValue(&groupsPerExpert, MTL::DataTypeUInt, NS::UInteger(9));
		constants->setConstantValue(&scaleBytesPerExpert, MTL::DataTypeUInt, NS::UInteger(10));
		constants->setConstantValue(&dispatchItemsPerExpert, MTL::DataTypeUInt, NS::UInteger(11));
		constants->setConstantValue(&rowsPerExpert, MTL::DataTypeUInt, NS::UInteger(12));
		constants->setConstantValue(&expertCount, MTL::DataTypeUInt, NS::UInteger(13));
		constants->setConstantValue(&segmentCount, MTL::DataTypeUInt, NS::UInteger(14));

		NS::String* swiftName = NS::String::string(functionName, NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	Dequantize8iRowwiseXKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get(), "dequantize_8i_rowwise_x_selected"));
	auto plan = NS::TransferPtr(createPipeline(kernel->library.get(), "dequantize_8i_rowwise_x_selected_plan"));
	PipelineValue<Dequantize8iRowwiseXKernel>* output = new PipelineValue<Dequantize8iRowwiseXKernel> { kernel, pipeline };
	output->second = plan;
	return std::make_pair(kernelDesc, output);
}
