#include "RMSNormGatedDescriptor.hpp"
#include "RMSNormGatedKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

static bool _rmsnorm_gated_descriptor_equals(const float epsilon, const GEMMOperandPrecision a_precision, const GEMMOperandPrecision gate_precision, const GEMMOperandPrecision scale_precision, const uint32_t column_count, const float rhs_epsilon, const GEMMOperandPrecision rhs_a_precision, const GEMMOperandPrecision rhs_gate_precision, const GEMMOperandPrecision rhs_scale_precision, const uint32_t rhs_column_count)
{
  return epsilon == rhs_epsilon &&
    a_precision == rhs_a_precision &&
    gate_precision == rhs_gate_precision &&
    scale_precision == rhs_scale_precision &&
    column_count == rhs_column_count;
}

bool RMSNormGatedKernelDescriptor::operator==(const RMSNormGatedKernelDescriptor& rhs) const {
  return _rmsnorm_gated_descriptor_equals(epsilon, aPrecision, gatePrecision, scalePrecision, columnCount, rhs.epsilon, rhs.aPrecision, rhs.gatePrecision, rhs.scalePrecision, rhs.columnCount);
}

bool RMSNormGatedDescriptor::operator==(const RMSNormGatedDescriptor& rhs) const {
  return _rmsnorm_gated_descriptor_equals(epsilon, aPrecision, gatePrecision, scalePrecision, columnCount, rhs.epsilon, rhs.aPrecision, rhs.gatePrecision, rhs.scalePrecision, rhs.columnCount);
}

static std::size_t _rmsnorm_gated_descriptor_hash(const float epsilon, const GEMMOperandPrecision a_precision, const GEMMOperandPrecision gate_precision, const GEMMOperandPrecision scale_precision, const uint32_t column_count) noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)a_precision.value, (unsigned int)gate_precision.value }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)scale_precision.value, column_count }));
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&epsilon));
  return seed;
}

std::size_t std::hash<RMSNormGatedKernelDescriptor>::operator()(const RMSNormGatedKernelDescriptor& hash) const noexcept {
  return _rmsnorm_gated_descriptor_hash(hash.epsilon, hash.aPrecision, hash.gatePrecision, hash.scalePrecision, hash.columnCount);
}

std::size_t std::hash<RMSNormGatedDescriptor>::operator()(const RMSNormGatedDescriptor& hash) const noexcept {
  return _rmsnorm_gated_descriptor_hash(hash.epsilon, hash.aPrecision, hash.gatePrecision, hash.scalePrecision, hash.columnCount);
}

std::pair<RMSNormGatedKernelDescriptor, PipelineValue<RMSNormGatedKernel>*> RMSNormGatedDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<RMSNormGatedKernelDescriptor, std::unique_ptr<RMSNormGatedKernel>>* const libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  RMSNormGatedKernelDescriptor kernelDesc = {
    .epsilon = epsilon,
    .aPrecision = aPrecision,
    .gatePrecision = gatePrecision,
    .scalePrecision = scalePrecision,
    .columnCount = columnCount,
  };

  auto createKernel =
  [=](RMSNormGatedKernelDescriptor descriptor) -> RMSNormGatedKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      RMSNormGatedKernel* kernel = new RMSNormGatedKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<RMSNormGatedKernel>(kernel);
      return kernel;
    }
  };

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    NS::String* swiftName = NS::String::string("rmsnorm_gated", NS::UTF8StringEncoding);
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    NS::Error* error = nil;
    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  RMSNormGatedKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<RMSNormGatedKernel>* output = new PipelineValue<RMSNormGatedKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
