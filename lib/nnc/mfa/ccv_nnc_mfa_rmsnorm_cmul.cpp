#include "ccv_nnc_mfa.hpp"
#include "kernels/RMSNormCmulDescriptor.hpp"
#include "kernels/RMSNormCmulKernel.hpp"

using namespace ccv::nnc;

static GEMMOperandPrecision _ccv_nnc_mfa_rmsnorm_cmul_precision(const uint64_t data_type)
{
	if (data_type == MTL::DataTypeFloat)
		return GEMMOperandPrecision::FP32;
	if (data_type == MTL::DataTypeBFloat)
		return GEMMOperandPrecision::BF16;
	return GEMMOperandPrecision::FP16;
}

static uint32_t _ccv_nnc_mfa_rmsnorm_cmul_rows_per_threadgroup(const ccv_nnc_mfa_rmsnorm_cmul_params_t params)
{
	if (params.broadcast_ratio == 1 || params.row_count < 512)
		return 1;
	if (params.broadcast_ratio >= 4 && params.row_count < 8192)
		return 4;
	return 2;
}

static RMSNormCmulDescriptor _ccv_nnc_mfa_rmsnorm_cmul_descriptor(const ccv_nnc_mfa_rmsnorm_cmul_params_t params)
{
	return RMSNormCmulDescriptor {
		.epsilon = params.epsilon,
		.aPrecision = _ccv_nnc_mfa_rmsnorm_cmul_precision(params.a_data_type),
		.rotationPrecision = _ccv_nnc_mfa_rmsnorm_cmul_precision(params.rotation_data_type),
		.scalePrecision = _ccv_nnc_mfa_rmsnorm_cmul_precision(params.scale_data_type),
		.columnCount = params.column_count,
		.broadcastRatio = params.broadcast_ratio,
		.rowsPerThreadgroup = _ccv_nnc_mfa_rmsnorm_cmul_rows_per_threadgroup(params),
		.elementwiseAffine = params.elementwise_affine != 0,
	};
}

void ccv_nnc_mfa_prepare_rmsnorm_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_cmul_params_t params)
{
	(void)context;
	(void)params;
}

void ccv_nnc_mfa_encode_rmsnorm_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_rmsnorm_cmul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
	CCV_NNC_MFA_PRECONDITION(params.row_count > 0 && params.column_count > 0 && params.column_count <= 512 && params.column_count % 2 == 0 && params.broadcast_ratio > 0);
	CCV_NNC_MFA_PRECONDITION(params.row_count % params.broadcast_ratio == 0);
	CCV_NNC_MFA_PRECONDITION(tensors[0] && tensors[1] && tensors[2] && tensors[3] && !tensors[4]);
	auto pool = NS::AutoreleasePool::alloc()->init();
	auto pipelineValue = context->kernel_cache.findKernel<RMSNormCmulKernel, RMSNormCmulDescriptor, RMSNormCmulKernelDescriptor>(_ccv_nnc_mfa_rmsnorm_cmul_descriptor(params), context->device.get(), DeviceProperties());
	pool->drain();
	auto encoder = command_batch->startCommand();
	encoder->setComputePipelineState(pipelineValue->pipeline.get());
	for (int i = 0; i < 4; i++)
		encoder->setBuffer(tensors[i], tensor_offsets[i], i);
	if (tensors[0] == tensors[3])
		encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
	else {
		encoder->useResource(tensors[0], MTL::ResourceUsageRead);
		encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
	}
	encoder->useResource(tensors[1], MTL::ResourceUsageRead);
	if (params.elementwise_affine)
		encoder->useResource(tensors[2], MTL::ResourceUsageRead);
	const uint32_t rows_per_threadgroup = _ccv_nnc_mfa_rmsnorm_cmul_rows_per_threadgroup(params);
	encoder->dispatchThreadgroups(MTL::Size((params.broadcast_ratio + rows_per_threadgroup - 1) / rows_per_threadgroup, params.row_count / params.broadcast_ratio, 1), pipelineValue->kernel->threadgroupSize);
	command_batch->finishCommand(encoder);
}
