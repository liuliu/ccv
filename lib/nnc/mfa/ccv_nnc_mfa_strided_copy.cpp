#include "ccv_nnc_mfa.hpp"
#include "kernels/StridedCopyDescriptor.hpp"
#include "kernels/StridedCopyKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_encode_strided_copy(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_strided_copy_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr) {
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors += 1;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 2);
	CCV_NNC_MFA_PRECONDITION(tensors[0] != tensors[1]);
	CCV_NNC_MFA_PRECONDITION(params.rows > 0 && params.cols > 0);
	CCV_NNC_MFA_PRECONDITION(params.source_row_stride >= params.cols && params.destination_row_stride >= params.cols);

	StridedCopyDescriptor descriptor;
	if (params.data_type == MTL::DataTypeHalf) {
		descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
	} else if (params.data_type == MTL::DataTypeBFloat) {
		descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
	} else {
		CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat);
		descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
	}
	descriptor.rows = params.rows;
	descriptor.cols = params.cols;
	descriptor.sourceRowStride = params.source_row_stride;
	descriptor.destinationRowStride = params.destination_row_stride;
	// With loadC one kernel serves every shape: it indexes the destination by its row stride
	// whether or not that is the width, and reads its rows rather than an element count.
	descriptor.loadC = params.loadC;
	descriptor.destinationStrided = params.loadC || params.destination_row_stride != params.cols;
	descriptor.loadM = params.loadC ? false : params.loadM;
	const size_t dataTypeSize = params.data_type == MTL::DataTypeFloat ? 4 : 2;
	descriptor.vectorized = (params.cols % 4 == 0 &&
		params.source_row_stride % 4 == 0 &&
		params.destination_row_stride % 4 == 0 &&
		tensor_offsets[0] % (dataTypeSize * 4) == 0 &&
		tensor_offsets[1] % (dataTypeSize * 4) == 0) ? 1 : 0;

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<StridedCopyKernel, StridedCopyDescriptor, StridedCopyKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	encoder->setComputePipelineState(pipeline.get());

	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
	if (params.loadC) {
		const uint32_t unit = descriptor.vectorized ? 4 : 1;
		const uint32_t colUnits = params.cols / unit;
		const uint32_t sizes[4] = { colUnits, params.rows, params.source_row_stride / unit, params.destination_row_stride / unit };
		encoder->setBytes(sizes, sizeof(sizes), 2);
		// A threadgroup keeps the kernel's thread count, as wide as a row up to that and as tall as
		// the rest allows, so neither a wide copy nor a tall, narrow one leaves most of a group idle.
		const uint32_t threads = (uint32_t)kernel->threadgroupSize.width;
		uint32_t width = 1;
		while (width < colUnits && width < threads)
			width <<= 1;
		const uint32_t height = threads / width;
		encoder->dispatchThreadgroups(MTL::Size((colUnits + width - 1) / width, (params.rows + height - 1) / height, 1), MTL::Size(width, height, 1));
	} else {
		if (params.loadM) {
			const uint32_t elementCount = descriptor.vectorized ? params.rows * params.cols / 4 : params.rows * params.cols;
			encoder->setBytes(&elementCount, sizeof(elementCount), 2);
		}

		const MTL::Size gridSize = kernel->gridSize(params.rows, params.cols);
		CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
		encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);
	}

	command_batch->finishCommand(encoder);
}
