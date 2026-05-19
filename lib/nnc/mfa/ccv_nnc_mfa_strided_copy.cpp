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
	const size_t dataTypeSize = params.data_type == MTL::DataTypeFloat ? 4 : 2;
	descriptor.vectorized = (params.cols % 4 == 0 &&
		params.source_row_stride % 4 == 0 &&
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

	const MTL::Size gridSize = kernel->gridSize(params.rows, params.cols);
	CCV_NNC_MFA_PRECONDITION(gridSize.width > 0 && gridSize.height > 0 && gridSize.depth > 0);
	encoder->dispatchThreadgroups(gridSize, kernel->threadgroupSize);

	command_batch->finishCommand(encoder);
}
