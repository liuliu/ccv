#include "ccv_nnc_mfa.hpp"
#include "kernels/TransposeDescriptor.hpp"
#include "kernels/TransposeKernel.hpp"
using namespace ccv::nnc;

void ccv_nnc_mfa_encode_transpose(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_transpose_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	auto encoder = command_batch->startCommand();

	int num_tensors = 0;
	while (tensors[num_tensors] != nullptr) {
		encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
		num_tensors += 1;
	}
	CCV_NNC_MFA_PRECONDITION(num_tensors == 2);
	CCV_NNC_MFA_PRECONDITION(tensors[0] != tensors[1]);
	CCV_NNC_MFA_PRECONDITION(params.batch_size > 0 && params.rows > 0 && params.cols > 0);
	CCV_NNC_MFA_PRECONDITION(params.source_row_stride >= params.cols && params.destination_row_stride >= params.rows);
	CCV_NNC_MFA_PRECONDITION((uint64_t)params.source_batch_stride >= (uint64_t)params.source_row_stride * params.rows);
	CCV_NNC_MFA_PRECONDITION((uint64_t)params.destination_batch_stride >= (uint64_t)params.destination_row_stride * params.cols);

	TransposeDescriptor descriptor;
	if (params.data_type == MTL::DataTypeHalf) {
		descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
	} else if (params.data_type == MTL::DataTypeBFloat) {
		descriptor.memoryPrecision = GEMMOperandPrecision::BF16;
	} else {
		CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat);
		descriptor.memoryPrecision = GEMMOperandPrecision::FP32;
	}

	auto pool = NS::AutoreleasePool::alloc()->init();
	auto& shaderCache = context->kernel_cache;
	DeviceProperties dprops = DeviceProperties();
	auto pipelineValue = shaderCache.findKernel<TransposeKernel, TransposeDescriptor, TransposeKernelDescriptor>(descriptor, context->device.get(), dprops);
	pool->drain();
	auto kernel = pipelineValue->kernel;
	auto pipeline = pipelineValue->pipeline;

	encoder->setComputePipelineState(pipeline.get());
	encoder->useResource(tensors[0], MTL::ResourceUsageRead);
	encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
	const TransposeKernelParams kernelParams = {
		.rows = params.rows,
		.cols = params.cols,
		.sourceBatchStride = params.source_batch_stride,
		.sourceRowStride = params.source_row_stride,
		.destinationBatchStride = params.destination_batch_stride,
		.destinationRowStride = params.destination_row_stride,
	};
	encoder->setBytes(&kernelParams, sizeof(kernelParams), 2);

	const MTL::Size gridSize = kernel->gridSize(params.batch_size, params.rows, params.cols);
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

	command_batch->finishCommand(encoder);
}
