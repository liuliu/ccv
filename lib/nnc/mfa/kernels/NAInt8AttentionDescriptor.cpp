#include "NAInt8AttentionDescriptor.hpp"
#include "NAInt8AttentionKernelDescriptor.hpp"
#include "NAInt8AttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool NAInt8AttentionDescriptor::operator==(const NAInt8AttentionDescriptor& rhs) const {
  return
    batchDimension == rhs.batchDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    ioPrecision == rhs.ioPrecision &&
    scale == rhs.scale &&
    batchStrides == rhs.batchStrides &&
    simd_all(matrixDimensions == rhs.matrixDimensions);
}

std::size_t std::hash<NAInt8AttentionDescriptor>::operator()(const NAInt8AttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.Hq);
  combine_32(seed, hash.Hk);
  combine_32(seed, (uint32_t)hash.ioPrecision.value);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

NAInt8AttentionKernelDescriptor NAInt8AttentionDescriptor::kernelDescriptor() const noexcept {
  const uint16_t blockD = matrixDimensions[2] >= 192 ? 64 : 32;
  const simd::ushort3 blockDimensions { 16, 64, blockD };
  const bool checkCEdge1 = (matrixDimensions[1] % (blockDimensions[1] * 2)) > blockDimensions[1];
  const uint16_t executionSIMDGroups = matrixDimensions[2] > 192 ? 16 : 4;
  const bool mortonOrder = true;
  return NAInt8AttentionKernelDescriptor(
      blockDimensions,
      (unsigned short)matrixDimensions[2],
      Hq,
      Hk,
      executionSIMDGroups,
      checkCEdge1,
      true,
      true,
      true,
      mortonOrder,
      ioPrecision,
      NAInt8AttentionKernelMode::full,
      scale);
}

std::pair<NAInt8AttentionKernelDescriptor, PipelineValue<NAInt8AttentionKernel> *> NAInt8AttentionDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties &dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<NAInt8AttentionKernelDescriptor, std::unique_ptr<NAInt8AttentionKernel>> *const libraryCache) const noexcept
{
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  auto createKernel =
  [=](const NAInt8AttentionKernelDescriptor& descriptor) -> NAInt8AttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    NAInt8AttentionKernel* kernel = new NAInt8AttentionKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<NAInt8AttentionKernel>(kernel);
    return kernel;
  };

  auto createAttentionPipeline =
  [=](NAInt8AttentionKernel* kernel) -> MTL::ComputePipelineState* {
    const auto kernelDesc = kernelDescriptor();
    const uint32_t q_tiles = (matrixDimensions[0] + kernelDesc.blockDimensions[0] - 1) / kernelDesc.blockDimensions[0];
    const uint32_t k_tiles = (matrixDimensions[1] + kernelDesc.blockDimensions[1] - 1) / kernelDesc.blockDimensions[1];
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t rowDimension = matrixDimensions[0];
    const uint32_t columnDimension = matrixDimensions[1];
    const uint32_t qBatchStride = batchStrides[AttentionOperand::Q].value_or(0);
    const uint32_t kBatchStride = batchStrides[AttentionOperand::K].value_or(0);
    const uint32_t vBatchStride = batchStrides[AttentionOperand::V].value_or(0);
    const uint32_t oBatchStride = batchStrides[AttentionOperand::O].value_or(0);
    const uint32_t qScaleBatchStride = batchDimension > 1 ? Hq * q_tiles : 0;
    const uint32_t kScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
    const uint32_t vScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
    constants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&columnDimension, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&qBatchStride, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&kBatchStride, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&vBatchStride, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&oBatchStride, MTL::DataTypeUInt, NS::UInteger(5));
    constants->setConstantValue(&qScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(6));
    constants->setConstantValue(&kScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(7));
    constants->setConstantValue(&vScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(8));

    NS::Error* error = nil;
    auto functionName = NS::String::string("int8_attention", NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    auto pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto createQuantizePipeline =
  [=](NAInt8AttentionKernel* kernel, const char* functionNameString) -> MTL::ComputePipelineState* {
    NS::Error* error = nil;
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t q_sequence = matrixDimensions[0];
    const uint32_t kv_sequence = matrixDimensions[1];
    const uint32_t q_heads = Hq;
    const uint32_t kv_heads = Hk;
    const uint32_t q_tile_size = kernel->blockDimensions[0];
    const uint32_t kv_tile_size = kernel->blockDimensions[1];
    const uint32_t q_tiles = (q_sequence + q_tile_size - 1) / q_tile_size;
    const uint32_t k_tiles = (kv_sequence + kv_tile_size - 1) / kv_tile_size;
    const uint32_t q_batch_stride = batchStrides[AttentionOperand::Q].value_or(0);
    const uint32_t k_batch_stride = batchStrides[AttentionOperand::K].value_or(0);
    const uint32_t v_batch_stride = batchStrides[AttentionOperand::V].value_or(0);
    const uint32_t q_scale_batch_stride = batchDimension > 1 ? Hq * q_tiles : 0;
    const uint32_t kv_scale_batch_stride = batchDimension > 1 ? Hk * k_tiles : 0;
    constants->setConstantValue(&q_sequence, MTL::DataTypeUInt, NS::UInteger(900));
    constants->setConstantValue(&kv_sequence, MTL::DataTypeUInt, NS::UInteger(901));
    constants->setConstantValue(&q_heads, MTL::DataTypeUInt, NS::UInteger(902));
    constants->setConstantValue(&kv_heads, MTL::DataTypeUInt, NS::UInteger(903));
    constants->setConstantValue(&q_tile_size, MTL::DataTypeUInt, NS::UInteger(904));
    constants->setConstantValue(&kv_tile_size, MTL::DataTypeUInt, NS::UInteger(905));
    constants->setConstantValue(&q_tiles, MTL::DataTypeUInt, NS::UInteger(906));
    constants->setConstantValue(&k_tiles, MTL::DataTypeUInt, NS::UInteger(907));
    constants->setConstantValue(&q_batch_stride, MTL::DataTypeUInt, NS::UInteger(908));
    constants->setConstantValue(&k_batch_stride, MTL::DataTypeUInt, NS::UInteger(909));
    constants->setConstantValue(&v_batch_stride, MTL::DataTypeUInt, NS::UInteger(910));
    constants->setConstantValue(&q_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(911));
    constants->setConstantValue(&kv_scale_batch_stride, MTL::DataTypeUInt, NS::UInteger(912));
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    auto pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = kernelDescriptor();
  auto kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createAttentionPipeline(kernel));
  auto quantizeQ = NS::TransferPtr(createQuantizePipeline(kernel, "quantize_q"));
  auto quantizeK = NS::TransferPtr(createQuantizePipeline(kernel, "quantize_k"));
  auto quantizeV = NS::TransferPtr(createQuantizePipeline(kernel, "quantize_v"));

  PipelineValue<NAInt8AttentionKernel>* output = new PipelineValue<NAInt8AttentionKernel> { kernel, pipeline };
  output->second = quantizeQ;
  output->third = quantizeK;
  output->fourth = quantizeV;
  return std::make_pair(kernelDesc, output);
}
