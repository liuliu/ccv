#include "ccv_nnc_mfa.hpp"
#include "kernels/ReduceLogSumExpDescriptor.hpp"
#include "kernels/ReduceLogSumExpKernel.hpp"
#include <algorithm>
#include <limits>
using namespace ccv::nnc;

namespace {

constexpr uint32_t kReduceLogSumExpPartitionSize = 4096;

}

void ccv_nnc_mfa_prepare_reduce_logsumexp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_reduce_logsumexp_params_t params)
{
  (void)context;
  (void)params;
}

void ccv_nnc_mfa_encode_reduce_logsumexp(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_reduce_logsumexp_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.row_count > 0);
  CCV_NNC_MFA_PRECONDITION(params.column_count > 0);
  CCV_NNC_MFA_PRECONDITION(tensors[0] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[1] != nullptr);
  CCV_NNC_MFA_PRECONDITION(tensors[2] == nullptr);
  CCV_NNC_MFA_PRECONDITION(params.data_type == MTL::DataTypeFloat || params.data_type == MTL::DataTypeHalf || params.data_type == MTL::DataTypeBFloat);

  const uint32_t partition_count = static_cast<uint32_t>((static_cast<uint64_t>(params.column_count) + kReduceLogSumExpPartitionSize - 1) / kReduceLogSumExpPartitionSize);
  const ReduceLogSumExpDescriptor descriptor = {
    .memoryPrecision = params.data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 : (params.data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16),
    .columnCount = params.column_count,
    .partitionSize = kReduceLogSumExpPartitionSize,
    .partitionCount = partition_count,
    .scale = params.scale,
    .partitioned = partition_count > 1,
  };
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shader_cache = context->kernel_cache;
  auto* const pipelines = shader_cache.findKernel<ReduceLogSumExpKernel, ReduceLogSumExpDescriptor, ReduceLogSumExpKernelDescriptor>(descriptor, context->device.get(), DeviceProperties());
  pool->drain();

  const uint32_t partition_elements = std::min(params.column_count, kReduceLogSumExpPartitionSize);
  const uint32_t partition_work_items = static_cast<uint32_t>((static_cast<uint64_t>(partition_elements) + 3) / 4);
  const MTL::Size partition_group_size(std::min(256u, std::max(32u, (partition_work_items + 31) & ~31u)), 1, 1);
  if (partition_count == 1) {
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
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), partition_group_size);
    command_batch->finishCommand(encoder);
    return;
  }

  CCV_NNC_MFA_PRECONDITION(static_cast<size_t>(params.row_count) <= std::numeric_limits<size_t>::max() / partition_count / (2 * sizeof(float)));
  const size_t scratch_size = static_cast<size_t>(params.row_count) * partition_count * 2 * sizeof(float);
  MTL::Buffer* const scratch = context->request_scratch(scratch_size);
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->pipeline.get());
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(scratch, MTL::ResourceUsageWrite);
    encoder->setBuffer(tensors[0], tensor_offsets[0], 0);
    encoder->setBuffer(scratch, 0, 1);
    encoder->dispatchThreadgroups(MTL::Size(static_cast<size_t>(params.row_count) * partition_count, 1, 1), partition_group_size);
    command_batch->finishCommand(encoder);
  }
  {
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipelines->second.get());
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(tensors[1], tensor_offsets[1], 1);
    const uint32_t merge_work_items = (partition_count + 3) / 4;
    const MTL::Size merge_group_size(std::min(256u, std::max(32u, (merge_work_items + 31) & ~31u)), 1, 1);
    encoder->dispatchThreadgroups(MTL::Size(params.row_count, 1, 1), merge_group_size);
    command_batch->finishCommand(encoder);
  }
}
