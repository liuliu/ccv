#include "StridedCopyDescriptor.hpp"
#include "StridedCopyKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>

bool StridedCopyDescriptor::operator==(const StridedCopyDescriptor& rhs) const
{
	return
	memoryPrecision == rhs.memoryPrecision &&
	vectorized == rhs.vectorized &&
	rows == rhs.rows &&
	cols == rhs.cols &&
	sourceRowStride == rhs.sourceRowStride &&
	destinationStrided == rhs.destinationStrided &&
	(!destinationStrided || destinationRowStride == rhs.destinationRowStride);
}

std::size_t std::hash<StridedCopyDescriptor>::operator()(const StridedCopyDescriptor& hash) const noexcept
{
	using namespace ccv::nnc::mfa::hash;
	std::size_t seed = 0;
	combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, (unsigned int)hash.vectorized }));
	combine_64(seed, pack_64(simd::uint2 { hash.rows, hash.cols }));
	combine_32(seed, hash.sourceRowStride);
	combine_32(seed, hash.destinationStrided);
	if (hash.destinationStrided)
		combine_32(seed, hash.destinationRowStride);
	return seed;
}

std::pair<StridedCopyKernelDescriptor, PipelineValue<StridedCopyKernel>*> StridedCopyDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<StridedCopyKernelDescriptor, std::unique_ptr<StridedCopyKernel>>* const libraryCache) const noexcept
{
	auto createKernel =
	[=](StridedCopyKernelDescriptor descriptor) -> StridedCopyKernel* {
		auto iterator = libraryCache->find(descriptor);
		if (iterator != libraryCache->end()) {
			return iterator->second.get();
		} else {
			StridedCopyKernel* kernel = new StridedCopyKernel(descriptor, device);
			(*libraryCache)[descriptor] = std::unique_ptr<StridedCopyKernel>(kernel);
			return kernel;
		}
	};

	StridedCopyKernelDescriptor kernelDesc;
	kernelDesc.vectorized = vectorized;
	kernelDesc.destinationStrided = destinationStrided;
	kernelDesc.memoryPrecision = memoryPrecision;

	auto createPipeline =
	[=](MTL::Library* library) -> MTL::ComputePipelineState* {
		auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
		uint32_t elementCount;
		if (vectorized)
		{
			const uint32_t colUnits = cols / 4;
			const uint32_t sourceRowStrideUnits = sourceRowStride / 4;
			elementCount = rows * colUnits;
			constants->setConstantValue(&colUnits, MTL::DataTypeUInt, NS::UInteger(0));
			constants->setConstantValue(&sourceRowStrideUnits, MTL::DataTypeUInt, NS::UInteger(1));
			if (destinationStrided)
			{
				const uint32_t destinationRowStrideUnits = destinationRowStride / 4;
				constants->setConstantValue(&destinationRowStrideUnits, MTL::DataTypeUInt, NS::UInteger(2));
				constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(3));
			} else {
				constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(2));
			}
		} else {
			elementCount = rows * cols;
			constants->setConstantValue(&cols, MTL::DataTypeUInt, NS::UInteger(0));
			constants->setConstantValue(&sourceRowStride, MTL::DataTypeUInt, NS::UInteger(1));
			if (destinationStrided)
			{
				constants->setConstantValue(&destinationRowStride, MTL::DataTypeUInt, NS::UInteger(2));
				constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(3));
			} else {
				constants->setConstantValue(&elementCount, MTL::DataTypeUInt, NS::UInteger(2));
			}
		}

		NS::String* swiftName = NS::String::string("strided_copy", NS::UTF8StringEncoding);
		NS::Error* error = nil;
		auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
		CCV_NNC_MFA_CHECK_ERROR(error);

		auto pipeline = device->newComputePipelineState(function.get(), &error);
		CCV_NNC_MFA_CHECK_ERROR(error);
		return pipeline;
	};

	StridedCopyKernel* kernel = createKernel(kernelDesc);
	auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));
	PipelineValue<StridedCopyKernel>* output = new PipelineValue<StridedCopyKernel> { kernel, pipeline };
	return std::make_pair(kernelDesc, output);
}
