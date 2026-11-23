#include "MessageGateDescriptor.hpp"
#include "MessageGateKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool MessageGateDescriptor::operator==(const MessageGateDescriptor& rhs) const {
  return memoryPrecision == rhs.memoryPrecision &&
    downPrecision == rhs.downPrecision &&
    weightPrecision == rhs.weightPrecision &&
    downQuantized == rhs.downQuantized &&
    quantized == rhs.quantized &&
    rank == rhs.rank &&
    mixRows == rhs.mixRows &&
    channels == rhs.channels &&
    epsilon == rhs.epsilon &&
    kind == rhs.kind;
}

std::size_t std::hash<MessageGateDescriptor>::operator()(const MessageGateDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  seed = combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value | ((unsigned int)hash.downPrecision.value << 8) | ((unsigned int)hash.weightPrecision.value << 16), (unsigned int)hash.downQuantized | ((unsigned int)hash.quantized << 1) | ((unsigned int)hash.kind << 2) }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.rank, hash.channels }));
  seed = combine_64(seed, pack_64(simd::uint2 { hash.mixRows, *reinterpret_cast<const uint32_t*>(&hash.epsilon) }));
  return seed;
}

std::pair<MessageGateKernelDescriptor, PipelineValue<MessageGateKernel>*> MessageGateDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MessageGateKernelDescriptor, std::unique_ptr<MessageGateKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](MessageGateKernelDescriptor descriptor) -> MessageGateKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      MessageGateKernel* kernel = new MessageGateKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<MessageGateKernel>(kernel);
      return kernel;
    }
  };

  const MessageGateKernelDescriptor kernelDesc = { memoryPrecision, downPrecision, weightPrecision, downQuantized, quantized, rank, mixRows };

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&channels, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&epsilon, MTL::DataTypeFloat, NS::UInteger(1));

    NS::String* swiftName = NS::String::string(kind == mix ? "message_gate_mix" : (kind == normalize ? "message_gate_normalize" : "message_gate"), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  MessageGateKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<MessageGateKernel>* output = new PipelineValue<MessageGateKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
