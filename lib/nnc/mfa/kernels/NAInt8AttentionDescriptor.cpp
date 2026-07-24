#include "NAInt8AttentionDescriptor.hpp"
#include "NAInt8AttentionKernelDescriptor.hpp"
#include "NAInt8AttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

namespace {

static void serializeBinaries(MTL::BinaryArchive *const binaryArchive, const std::string& pathToWrite) noexcept {
  NS::Error *error = nil;
  binaryArchive->serializeToURL(NS::URL::fileURLWithPath(NS::String::string(pathToWrite.c_str(), NS::UTF8StringEncoding)), &error);
}

}

bool NAInt8AttentionDescriptor::operator==(const NAInt8AttentionDescriptor& rhs) const {
  return
    batchDimension == rhs.batchDimension &&
    Hq == rhs.Hq &&
    Hk == rhs.Hk &&
    type == rhs.type &&
    ioPrecision == rhs.ioPrecision &&
    lowPrecisionIntermediates == rhs.lowPrecisionIntermediates &&
    scale == rhs.scale &&
    isCausal == rhs.isCausal &&
    masked == rhs.masked &&
    isVarlen == rhs.isVarlen &&
    attentionSinks == rhs.attentionSinks &&
    maskBatchStride == rhs.maskBatchStride &&
    batchStrides == rhs.batchStrides &&
    simd_all(matrixDimensions == rhs.matrixDimensions);
}

std::size_t std::hash<NAInt8AttentionDescriptor>::operator()(const NAInt8AttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.batchDimension);
  combine_32(seed, hash.Hq);
  combine_32(seed, hash.Hk);
  combine_32(seed, (uint16_t)hash.type.value);
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)hash.ioPrecision.value,
      (uint16_t)(hash.lowPrecisionIntermediates ? 1 : 0) }));
  combine_32(seed, pack_32(simd::ushort2 {
      (uint16_t)(hash.isCausal ? 1 : 0),
      (uint16_t)(hash.masked ? 1 : 0) }));
  combine_32(seed, hash.isVarlen ? 1 : 0);
  combine_32(seed, hash.attentionSinks ? 1 : 0);
  combine_32(seed, hash.maskBatchStride);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&hash.scale));
  return seed;
}

NAInt8AttentionKernelDescriptor NAInt8AttentionDescriptor::kernelDescriptor() const noexcept {
  const uint16_t qScaleTileSize =
      type == AttentionKernelType::forward ? 16 : 32;
  const uint16_t kvScaleTileSize = 64;
  const bool lowPrecisionBackward =
      type != AttentionKernelType::forward &&
      ioPrecision != GEMMOperandPrecision::FP32;
  const bool splitHeadBackward = lowPrecisionBackward && matrixDimensions[2] == 128;
  const bool splitHeadBackwardKeyValue =
      type == AttentionKernelType::backwardKeyValue && splitHeadBackward;
  const uint16_t blockD =
      type == AttentionKernelType::forward ?
      (matrixDimensions[2] >= 192 ? 64 : 32) :
      (type == AttentionKernelType::backwardQuery && splitHeadBackward ?
          32 :
          (splitHeadBackward ? 64 : (uint16_t)matrixDimensions[2]));
  const uint16_t blockC =
      type == AttentionKernelType::backwardKeyValue ?
      (splitHeadBackward ? 32 : 16) : 64;
  const uint16_t queryBlockC =
      type == AttentionKernelType::backwardQuery && splitHeadBackward ? 32 : blockC;
  const simd::ushort3 blockDimensions { 16, queryBlockC, blockD };
  const uint16_t executionSIMDGroups =
      type == AttentionKernelType::forward ?
      (matrixDimensions[2] > 192 ? 16 : 4) :
      (splitHeadBackwardKeyValue ?
          (Hq > Hk ? 8 : 16) :
          4);
  const uint16_t vMeanThreads =
      matrixDimensions[1] <= 20480 ?
      NAInt8AttentionKernel::smallSequenceVMeanThreads :
      NAInt8AttentionKernel::largeSequenceVMeanThreads;
  const bool has_c_remainder =
      type == AttentionKernelType::forward &&
      (isVarlen || (matrixDimensions[1] % blockDimensions[1]) != 0);
  const bool has_causal_empty_rows =
      type == AttentionKernelType::forward &&
      isCausal &&
      !masked &&
      (isVarlen || matrixDimensions[0] > matrixDimensions[1]);
  return NAInt8AttentionKernelDescriptor(
      blockDimensions,
      (unsigned short)matrixDimensions[2],
      Hq,
      Hk,
      qScaleTileSize,
      kvScaleTileSize,
      executionSIMDGroups,
      vMeanThreads,
      has_c_remainder,
      isCausal ? 0 : 2,
      ioPrecision,
      lowPrecisionIntermediates,
      type,
      scale,
      isCausal,
      masked,
      has_causal_empty_rows,
      isVarlen,
      attentionSinks);
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

  auto createKernel =
  [=](const NAInt8AttentionKernelDescriptor& descriptor) -> NAInt8AttentionKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    NAInt8AttentionKernel* kernel = new NAInt8AttentionKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<NAInt8AttentionKernel>(kernel);
    return kernel;
  };

  auto createPipeline =
  [=](NAInt8AttentionKernel* kernel, MTL::FunctionConstantValues* constants, const char* functionNameString) -> MTL::ComputePipelineState* {
    NS::Error* error = nil;
    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    pipelineDescriptor->setComputeFunction(function.get());
    MTL::ComputePipelineState* pipeline = nullptr;
    if (binaryArchivesToRead) {
      pipelineDescriptor->setBinaryArchives(binaryArchivesToRead);
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss, nullptr, &error);
    }
    if (pipeline == nullptr) {
      error = nil;
      pipeline = device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
      if (binaryArchiveToWrite != nullptr) {
        NS::Error* archiveError = nil;
        binaryArchiveToWrite->addComputePipelineFunctions(pipelineDescriptor.get(), &archiveError);
        serializeBinaries(binaryArchiveToWrite, pathToWrite);
      }
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = kernelDescriptor();
  auto kernel = createKernel(kernelDesc);
  const uint32_t q_tiles = (matrixDimensions[0] + kernelDesc.qScaleTileSize - 1) / kernelDesc.qScaleTileSize;
  const uint32_t k_tiles = (matrixDimensions[1] + kernelDesc.kvScaleTileSize - 1) / kernelDesc.kvScaleTileSize;

  auto attentionConstants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t rowDimension = matrixDimensions[0];
  const uint32_t columnDimension = matrixDimensions[1];
  const uint32_t qBatchStride = batchStrides[AttentionOperand::Q].value_or(0);
  const uint32_t kBatchStride = batchStrides[AttentionOperand::K].value_or(0);
  const uint32_t vBatchStride = batchStrides[AttentionOperand::V].value_or(0);
  const uint32_t oBatchStride = batchStrides[AttentionOperand::O].value_or(0);
  const uint32_t dOBatchStride = batchStrides[AttentionOperand::dO].value_or(0);
  const uint32_t dVBatchStride = batchStrides[AttentionOperand::dV].value_or(0);
  const uint32_t dKBatchStride = batchStrides[AttentionOperand::dK].value_or(0);
  const uint32_t dQBatchStride = batchStrides[AttentionOperand::dQ].value_or(0);
  const uint32_t qScaleBatchStride = batchDimension > 1 ? Hq * q_tiles : 0;
  const uint32_t kScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  const uint32_t vScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  const uint32_t dOScaleBatchStride = batchDimension > 1 ? Hq * q_tiles : 0;
  const uint32_t vMeanBatchStride = batchDimension > 1 ? Hk * matrixDimensions[2] : 0;
  const uint32_t maskBatchStride = masked ? this->maskBatchStride : 0;
  const uint32_t blockMaskBatchStride = masked && maskBatchStride > 0 ? q_tiles * k_tiles : 0;
  attentionConstants->setConstantValue(&rowDimension, MTL::DataTypeUInt, NS::UInteger(0));
  attentionConstants->setConstantValue(&columnDimension, MTL::DataTypeUInt, NS::UInteger(1));
  attentionConstants->setConstantValue(&qBatchStride, MTL::DataTypeUInt, NS::UInteger(2));
  attentionConstants->setConstantValue(&kBatchStride, MTL::DataTypeUInt, NS::UInteger(3));
  attentionConstants->setConstantValue(&vBatchStride, MTL::DataTypeUInt, NS::UInteger(4));
  attentionConstants->setConstantValue(&oBatchStride, MTL::DataTypeUInt, NS::UInteger(5));
  attentionConstants->setConstantValue(&dOBatchStride, MTL::DataTypeUInt, NS::UInteger(6));
  attentionConstants->setConstantValue(&dVBatchStride, MTL::DataTypeUInt, NS::UInteger(7));
  attentionConstants->setConstantValue(&dKBatchStride, MTL::DataTypeUInt, NS::UInteger(8));
  attentionConstants->setConstantValue(&dQBatchStride, MTL::DataTypeUInt, NS::UInteger(9));
  attentionConstants->setConstantValue(&qScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(10));
  attentionConstants->setConstantValue(&kScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(11));
  attentionConstants->setConstantValue(&vScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(12));
  attentionConstants->setConstantValue(&dOScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(13));
  attentionConstants->setConstantValue(&vMeanBatchStride, MTL::DataTypeUInt, NS::UInteger(14));
  if (masked) {
    attentionConstants->setConstantValue(&maskBatchStride, MTL::DataTypeUInt, NS::UInteger(15));
    attentionConstants->setConstantValue(&blockMaskBatchStride, MTL::DataTypeUInt, NS::UInteger(16));
  }

  auto quantizeConstants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t qSequence = matrixDimensions[0];
  const uint32_t kvSequence = matrixDimensions[1];
  const uint32_t qHeads = Hq;
  const uint32_t kvHeads = Hk;
  const uint32_t qTileSize = kernel->qScaleTileSize;
  const uint32_t kvTileSize = kernel->kvScaleTileSize;
  const uint32_t qBatchStrideQ = batchStrides[AttentionOperand::Q].value_or(0);
  const uint32_t kBatchStrideQ = batchStrides[AttentionOperand::K].value_or(0);
  const uint32_t vBatchStrideQ = batchStrides[AttentionOperand::V].value_or(0);
  const uint32_t kvScaleBatchStride = batchDimension > 1 ? Hk * k_tiles : 0;
  quantizeConstants->setConstantValue(&qSequence, MTL::DataTypeUInt, NS::UInteger(900));
  quantizeConstants->setConstantValue(&kvSequence, MTL::DataTypeUInt, NS::UInteger(901));
  quantizeConstants->setConstantValue(&qHeads, MTL::DataTypeUInt, NS::UInteger(902));
  quantizeConstants->setConstantValue(&kvHeads, MTL::DataTypeUInt, NS::UInteger(903));
  quantizeConstants->setConstantValue(&qTileSize, MTL::DataTypeUInt, NS::UInteger(904));
  quantizeConstants->setConstantValue(&kvTileSize, MTL::DataTypeUInt, NS::UInteger(905));
  quantizeConstants->setConstantValue(&q_tiles, MTL::DataTypeUInt, NS::UInteger(906));
  quantizeConstants->setConstantValue(&k_tiles, MTL::DataTypeUInt, NS::UInteger(907));
  quantizeConstants->setConstantValue(&qBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(908));
  quantizeConstants->setConstantValue(&kBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(909));
  quantizeConstants->setConstantValue(&vBatchStrideQ, MTL::DataTypeUInt, NS::UInteger(910));
  quantizeConstants->setConstantValue(&qScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(911));
  quantizeConstants->setConstantValue(&kvScaleBatchStride, MTL::DataTypeUInt, NS::UInteger(912));

  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> second;
  NS::SharedPtr<MTL::ComputePipelineState> third;
  NS::SharedPtr<MTL::ComputePipelineState> fourth;
  NS::SharedPtr<MTL::ComputePipelineState> fifth;
  NS::SharedPtr<MTL::ComputePipelineState> sixth;
  switch (type.value) {
  case AttentionKernelType::forward:
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_attention"));
    second = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_q"));
    third = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_k"));
    fourth = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_v"));
    fifth = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "compute_v_mean"));
    if (masked) {
      sixth = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "generate_int8_attention_block_mask"));
    }
    break;
  case AttentionKernelType::backwardQuery:
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_backward_query"));
    second = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "compute_d"));
    third = NS::TransferPtr(createPipeline(kernel, quantizeConstants.get(), "quantize_q"));
    break;
  case AttentionKernelType::backwardKeyValue:
    pipeline = NS::TransferPtr(createPipeline(kernel, attentionConstants.get(), "int8_backward_keyvalue"));
    break;
  }

  PipelineValue<NAInt8AttentionKernel>* output = new PipelineValue<NAInt8AttentionKernel> { kernel, pipeline };
  output->second = second;
  output->third = third;
  output->fourth = fourth;
  output->fifth = fifth;
  output->sixth = sixth;
  return std::make_pair(kernelDesc, output);
}
