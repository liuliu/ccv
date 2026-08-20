#include "ccv_nnc_mfa.hpp"
#include "kernels/ReduceMaxDescriptor.hpp"
#include "kernels/ReduceMaxKernel.hpp"
#include <algorithm>
#include <limits>
using namespace ccv::nnc;

namespace {

constexpr uint32_t kReduceMaxPartitionSize = 8192;

GEMMOperandPrecision _ccv_nnc_mfa_reduce_max_precision(const uint64_t data_type)
{
  if (data_type == MTL::DataTypeFloat) {
    return GEMMOperandPrecision::FP32;
  } else if (data_type == MTL::DataTypeBFloat) {
    return GEMMOperandPrecision::BF16;
  }
  CCV_NNC_MFA_PRECONDITION(data_type == MTL::DataTypeHalf);
  return GEMMOperandPrecision::FP16;
}

uint32_t _ccv_nnc_mfa_reduce_max_partition_count(const uint32_t column_count)
{
  return static_cast<uint32_t>((static_cast<uint64_t>(column_count) + kReduceMaxPartitionSize - 1) / kReduceMaxPartitionSize);
}

MTL::Size _ccv_nnc_mfa_reduce_max_group_size(const uint32_t element_count)
{
  const uint32_t work_items = static_cast<uint32_t>((static_cast<uint64_t>(element_count) + 3) / 4);
  const uint32_t thread_count = std::min(256u, std::max(32u, (work_items + 31) & ~31u));
  return MTL::Size(thread_count, 1, 1);
}

}

void ccv_nnc_mfa_prepare_reduce_max(mfa::context* context, ccv_nnc_mfa_reduce_max_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_reduce_max(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_reduce_max_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.row_count > 0);
  CCV_NNC_MFA_PRECONDITION(params.column_count > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[2] == nullptr);

  const uint32_t partitionCount = _ccv_nnc_mfa_reduce_max_partition_count(params.column_count);
  const ReduceMaxDescriptor descriptor = {
    .memoryPrecision = _ccv_nnc_mfa_reduce_max_precision(params.data_type),
    .columnCount = params.column_count,
    .partitionSize = kReduceMaxPartitionSize,
    .partitionCount = partitionCount,
    .partitioned = partitionCount > 1,
  };
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  auto* const pipelines = shaderCache.findKernel<ReduceMaxKernel, ReduceMaxDescriptor, ReduceMaxKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();

  const uint32_t partitionElements = std::min(params.column_count, kReduceMaxPartitionSize);
  if (partitionCount == 1) {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->pipeline.get());
    if (tensors[0] == tensors[1]) {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[0], MTL::ResourceUsageRead);
      encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    }
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), _ccv_nnc_mfa_reduce_max_group_size(partitionElements));
    command_batch->finishCommand(encoder);
    return;
  }

  CCV_NNC_MFA_PRECONDITION(static_cast<size_t>(params.row_count) <= std::numeric_limits<size_t>::max() / partitionCount / sizeof(float));
  const size_t scratchSize = static_cast<size_t>(params.row_count) * partitionCount * sizeof(float);
  MTL::Buffer* const scratch = context->request_scratch(scratchSize);
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->pipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->dispatchThreadgroups(MTL::Size(static_cast<size_t>(params.row_count) * partitionCount, 1, 1), _ccv_nnc_mfa_reduce_max_group_size(partitionElements));
    command_batch->finishCommand(encoder);
  }
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->second.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), _ccv_nnc_mfa_reduce_max_group_size(partitionCount));
    command_batch->finishCommand(encoder);
  }
}
