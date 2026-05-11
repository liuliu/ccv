#include "ccv_nnc_mfa.hpp"
#include "kernels/Int8GemvDescriptor.hpp"
#include "kernels/Int8GemvKernel.hpp"
using namespace ccv::nnc;

namespace {

static GEMMOperandPrecision io_precision(const uint64_t data_type) noexcept
{
  switch (data_type) {
    case MTL::DataTypeHalf:
      return GEMMOperandPrecision::FP16;
    case MTL::DataTypeBFloat:
      return GEMMOperandPrecision::BF16;
    case MTL::DataTypeFloat:
      return GEMMOperandPrecision::FP32;
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      return GEMMOperandPrecision::FP16;
  }
}

}

void ccv_nnc_mfa_prepare_scaled_gemv(mfa::context* context, ccv_nnc_mfa_scaled_gemv_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_scaled_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_scaled_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  auto encoder = command_batch->startCommand();

  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3 || num_tensors == 4);
  CCV_NNC_MFA_PRECONDITION((params.fused_bias && num_tensors == 4) || (!params.fused_bias && num_tensors == 3));
  CCV_NNC_MFA_PRECONDITION(params.mrows == 1 || params.mrows == 2);
  CCV_NNC_MFA_PRECONDITION(params.format == 0 || ((params.ncols % 256) == 0 && (params.nrows % 256) == 0));
  CCV_NNC_MFA_PRECONDITION(params.format != 0 || (params.ncols % 4) == 0);

  Int8GemvDescriptor descriptor;
  descriptor.fusedBias = params.fused_bias ? 1 : 0;
  descriptor.mrows = (uint8_t)params.mrows;
  descriptor.format = params.format;
  descriptor.memoryPrecision = io_precision(params.data_type);
  descriptor.nrows = params.nrows;
  descriptor.ncols = params.ncols;

  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue = shaderCache.findKernel<Int8GemvKernel, Int8GemvDescriptor, Int8GemvKernelDescriptor>(descriptor, context->device.get(), dprops);
  pool->drain();
  auto pipeline = pipelineValue->pipeline;

  encoder->setComputePipelineState(pipeline.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (num_tensors == 4) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  }

  MTL::Size gridSize = MTL::Size((params.nrows + kInt8GemvRowsPerThreadgroup - 1) / kInt8GemvRowsPerThreadgroup, 1, 1);
  CCV_NNC_MFA_PRECONDITION(gridSize.width > 0);
  encoder->dispatchThreadgroups(gridSize, MTL::Size(kInt8GemvSIMDGroupsPerThreadgroup * 32, 1, 1));
  command_batch->finishCommand(encoder);
}
