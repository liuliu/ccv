#include "ccv_nnc_mfa.hpp"
#include "kernels/ArgmaxDescriptor.hpp"
#include "kernels/ArgmaxKernel.hpp"
#include <limits>
using namespace ccv::nnc;

namespace {

constexpr uint32_t kArgmaxPartitionSize = 4096;

struct ArgmaxShaderParams {
  uint32_t counter_0;
  uint32_t counter_1;
  uint32_t counter_2;
  uint32_t counter_3;
  uint32_t key_0;
  uint32_t key_1;
};

struct ArgmaxScratchPair {
  float value;
  uint32_t index;
};

static_assert(sizeof(ArgmaxShaderParams) == 24);
static_assert(sizeof(ArgmaxScratchPair) == 8);

GEMMOperandPrecision _ccv_nnc_mfa_argmax_precision(const uint64_t data_type)
{
  if (data_type == MTL::DataTypeFloat) {
    return GEMMOperandPrecision::FP32;
  } else if (data_type == MTL::DataTypeBFloat) {
    return GEMMOperandPrecision::BF16;
  }
  CCV_NNC_MFA_PRECONDITION(data_type == MTL::DataTypeHalf);
  return GEMMOperandPrecision::FP16;
}

uint32_t _ccv_nnc_mfa_argmax_partition_count(const uint32_t column_count)
{
  return static_cast<uint32_t>((static_cast<uint64_t>(column_count) + kArgmaxPartitionSize - 1) / kArgmaxPartitionSize);
}

}

void ccv_nnc_mfa_prepare_argmax(mfa::context* context, ccv_nnc_mfa_argmax_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_argmax(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_argmax_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.row_count > 0);
  CCV_NNC_MFA_PRECONDITION(params.column_count > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[2] == nullptr);
  if (params.gumbel) {
    CCV_NNC_MFA_PRECONDITION(params.state[0] == 1);
  }

  const uint32_t partitionCount = _ccv_nnc_mfa_argmax_partition_count(params.column_count);
  const ArgmaxDescriptor descriptor = {
    .memoryPrecision = _ccv_nnc_mfa_argmax_precision(params.data_type),
    .columnCount = params.column_count,
    .partitionSize = kArgmaxPartitionSize,
    .partitionCount = partitionCount,
    .scale = params.scale,
    .gumbel = params.gumbel != 0,
    .partitioned = partitionCount > 1,
  };
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  auto* const pipelines = shaderCache.findKernel<ArgmaxKernel, ArgmaxDescriptor, ArgmaxKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();

  const ArgmaxShaderParams shaderParams = {
    // The production MPS random state stores each 64-bit value high word
    // first. Philox consumes the low word first.
    .counter_0 = params.state[2],
    .counter_1 = params.state[1],
    .counter_2 = params.state[4],
    .counter_3 = params.state[3],
    .key_0 = params.state[6],
    .key_1 = params.state[5],
  };

  if (partitionCount == 1) {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->pipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBytes(&shaderParams, sizeof(shaderParams), 2);
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), pipelines->kernel->groupSize);
    command_batch->finishCommand(encoder);
    return;
  }

  CCV_NNC_MFA_PRECONDITION(static_cast<size_t>(params.row_count) <= std::numeric_limits<size_t>::max() / partitionCount / sizeof(ArgmaxScratchPair));
  const size_t scratchSize = static_cast<size_t>(params.row_count) * partitionCount * sizeof(ArgmaxScratchPair);
  MTL::Buffer* const scratch = context->request_scratch(scratchSize);
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->pipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->setBytes(&shaderParams, sizeof(shaderParams), 2);
    encoder->dispatchThreadgroups(MTL::Size(static_cast<size_t>(params.row_count) * partitionCount, 1, 1), pipelines->kernel->groupSize);
    command_batch->finishCommand(encoder);
  }
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->second.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->setBytes(&shaderParams, sizeof(shaderParams), 2);
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), pipelines->kernel->groupSize);
    command_batch->finishCommand(encoder);
  }
}
