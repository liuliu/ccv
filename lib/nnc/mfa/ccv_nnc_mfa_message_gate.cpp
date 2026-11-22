#include "ccv_nnc_mfa.hpp"
#include "kernels/MessageGateDescriptor.hpp"
#include "kernels/MessageGateKernel.hpp"

static GEMMOperandPrecision _ccv_nnc_mfa_message_gate_precision(const uint64_t data_type)
{
  return data_type == MTL::DataTypeFloat ? GEMMOperandPrecision::FP32 :
    (data_type == MTL::DataTypeBFloat ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16);
}

static MessageGateDescriptor _ccv_nnc_mfa_message_gate_descriptor(const ccv_nnc_mfa_message_gate_params_t params, const MessageGateDescriptor::Kind kind)
{
  MessageGateDescriptor descriptor;
  descriptor.memoryPrecision = _ccv_nnc_mfa_message_gate_precision(params.data_type);
  descriptor.downPrecision = _ccv_nnc_mfa_message_gate_precision(params.down_data_type);
  descriptor.weightPrecision = _ccv_nnc_mfa_message_gate_precision(params.weight_data_type);
  descriptor.downQuantized = params.down_quantized != 0;
  descriptor.quantized = params.quantized != 0;
  descriptor.rank = params.rank;
  descriptor.channels = params.channels;
  // A threadgroup of the few-rows mix takes up to 8 rows. The gate and the normalization don't depend
  // on it, and take the one-row library whatever the rows.
  descriptor.mixRows = kind != MessageGateDescriptor::mix || params.rows <= 1 ? 1 : (params.rows <= 2 ? 2 : (params.rows <= 4 ? 4 : 8));
  // The gate reads no epsilon.
  descriptor.epsilon = kind == MessageGateDescriptor::gate ? 0 : params.epsilon;
  descriptor.kind = kind;
  return descriptor;
}

void ccv_nnc_mfa_encode_message_gate(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_message_gate_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.rank >= 4 && params.rank % 4 == 0 && params.rank <= CCV_NNC_MFA_MESSAGE_GATE_MAX_RANK);
  CCV_NNC_MFA_PRECONDITION(params.channels % 4 == 0);
  CCV_NNC_MFA_PRECONDITION(params.mix_given || params.rows <= CCV_NNC_MFA_MESSAGE_GATE_FEW_ROWS);
  if (params.rows == 0 || params.channels == 0)
    return;
  auto pool = NS::AutoreleasePool::alloc()->init();
  PipelineValue<MessageGateKernel>* mixPipelineValue = nullptr;
  if (!params.mix_given) {
    mixPipelineValue = context->kernel_cache.findKernel<MessageGateKernel, MessageGateDescriptor, MessageGateKernelDescriptor>(_ccv_nnc_mfa_message_gate_descriptor(params, MessageGateDescriptor::mix), context->device.get(), DeviceProperties());
    CCV_NNC_MFA_PRECONDITION(mixPipelineValue->kernel->threadgroupSize.width <= mixPipelineValue->pipeline->maxTotalThreadsPerThreadgroup());
  }
  auto pipelineValue = context->kernel_cache.findKernel<MessageGateKernel, MessageGateDescriptor, MessageGateKernelDescriptor>(_ccv_nnc_mfa_message_gate_descriptor(params, MessageGateDescriptor::gate), context->device.get(), DeviceProperties());
  pool->drain();
  CCV_NNC_MFA_PRECONDITION(pipelineValue->kernel->threadgroupSize.width <= pipelineValue->pipeline->maxTotalThreadsPerThreadgroup());

  const uint32_t rows = params.rows;
  auto encoder = command_batch->startCommand();
  mtl_buffer_t* const mix = tensors[6];
  const size_t mix_offset = tensor_offsets[6];
  if (mixPipelineValue) {
    encoder->setComputePipelineState(mixPipelineValue->pipeline.get());
    for (int i = 0; i < 3; i++) {
      encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
      encoder->useResource(tensors[i], MTL::ResourceUsageRead);
    }
    encoder->setBuffer(mix, mix_offset, NS::UInteger(3));
    encoder->useResource(mix, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    // A quantized projection's row scales follow its bytes in the same buffer; a dense one binds
    // itself again, unread.
    encoder->setBuffer(tensors[2], tensor_offsets[2] + (params.down_quantized ? params.down_scale_offset : 0), NS::UInteger(4));
    encoder->setBytes(&rows, sizeof(rows), NS::UInteger(5));
    // A threadgroup takes one rank of mixRows rows.
    const uint32_t mixRows = mixPipelineValue->kernel->mixRows;
    encoder->dispatchThreadgroups(MTL::Size(params.rank, (params.rows + mixRows - 1) / mixRows, 1), mixPipelineValue->kernel->threadgroupSize);
  }

  // The gate reads the mix the serial encoder has finished writing.
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  encoder->setBuffer(tensors[0], tensor_offsets[0], NS::UInteger(0));
  encoder->setBuffer(tensors[1], tensor_offsets[1], NS::UInteger(1));
  encoder->setBuffer(mix, mix_offset, NS::UInteger(2));
  for (int i = 0; i < 2; i++)
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  encoder->useResource(mix, MTL::ResourceUsageRead);
  for (int i = 3; i < 5; i++) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  }
  encoder->setBuffer(tensors[5], tensor_offsets[5], NS::UInteger(5));
  encoder->useResource(tensors[5], MTL::ResourceUsageWrite);
  encoder->setBuffer(tensors[3], tensor_offsets[3] + (params.quantized ? params.forget_scale_offset : 0), NS::UInteger(6));
  encoder->setBuffer(tensors[4], tensor_offsets[4] + (params.quantized ? params.gain_scale_offset : 0), NS::UInteger(7));
  encoder->setBytes(&rows, sizeof(rows), NS::UInteger(8));
  // A threadgroup takes a tile of rows and of channels.
  const uint32_t rowsPerThreadgroup = pipelineValue->kernel->rowsPerThreadgroup;
  const uint32_t channelsPerThreadgroup = pipelineValue->kernel->channelsPerThreadgroup;
  encoder->dispatchThreadgroups(MTL::Size((params.rows + rowsPerThreadgroup - 1) / rowsPerThreadgroup, (params.channels + channelsPerThreadgroup - 1) / channelsPerThreadgroup, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}

void ccv_nnc_mfa_encode_message_gate_normalize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_message_gate_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  CCV_NNC_MFA_PRECONDITION(params.channels % 4 == 0);
  if (params.rows == 0 || params.channels == 0)
    return;
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto pipelineValue = context->kernel_cache.findKernel<MessageGateKernel, MessageGateDescriptor, MessageGateKernelDescriptor>(_ccv_nnc_mfa_message_gate_descriptor(params, MessageGateDescriptor::normalize), context->device.get(), DeviceProperties());
  pool->drain();
  CCV_NNC_MFA_PRECONDITION(pipelineValue->kernel->threadgroupSize.width <= pipelineValue->pipeline->maxTotalThreadsPerThreadgroup());
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipelineValue->pipeline.get());
  for (int i = 0; i < 2; i++) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], NS::UInteger(i));
    encoder->useResource(tensors[i], MTL::ResourceUsageRead);
  }
  encoder->setBuffer(tensors[2], tensor_offsets[2], NS::UInteger(2));
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  // A threadgroup to a row.
  encoder->dispatchThreadgroups(MTL::Size(params.rows, 1, 1), pipelineValue->kernel->threadgroupSize);
  command_batch->finishCommand(encoder);
}
