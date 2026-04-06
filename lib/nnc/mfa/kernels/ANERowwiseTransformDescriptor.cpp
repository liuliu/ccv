#include "ANERowwiseTransformDescriptor.hpp"
#include "ANERowwiseTransformKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool ANERowwiseTransformDescriptor::operator==(const ANERowwiseTransformDescriptor& rhs) const
{
  return
      memoryPrecision == rhs.memoryPrecision &&
      M == rhs.M &&
      paddedM == rhs.paddedM &&
      N == rhs.N &&
      K == rhs.K;
}

std::size_t std::hash<ANERowwiseTransformDescriptor>::operator()(const ANERowwiseTransformDescriptor& hash) const noexcept
{
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, (uint32_t)hash.memoryPrecision.value);
  combine_32(seed, hash.M);
  combine_32(seed, hash.paddedM);
  combine_32(seed, hash.N);
  combine_32(seed, hash.K);
  return seed;
}

std::pair<ANERowwiseTransformKernelDescriptor, PipelineValue<ANERowwiseTransformKernel>*> ANERowwiseTransformDescriptor::findKernel(
    MTL::Device* const device,
    const DeviceProperties& dprops,
    NS::Array* const binaryArchivesToRead,
    MTL::BinaryArchive* const binaryArchiveToWrite,
    const std::string& pathToWrite,
    std::unordered_map<ANERowwiseTransformKernelDescriptor, std::unique_ptr<ANERowwiseTransformKernel>> *const libraryCache) const noexcept
{
  (void)dprops;

  auto createKernel =
  [=](const ANERowwiseTransformKernelDescriptor& descriptor) -> ANERowwiseTransformKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end())
      return iterator->second.get();
    ANERowwiseTransformKernel* kernel = new ANERowwiseTransformKernel(descriptor, device);
    (*libraryCache)[descriptor] = std::unique_ptr<ANERowwiseTransformKernel>(kernel);
    return kernel;
  };

  auto createPipeline =
  [=](ANERowwiseTransformKernel* kernel, const char* functionNameString) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&paddedM, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(3));

    auto functionName = NS::String::string(functionNameString, NS::UTF8StringEncoding);
    NS::Error* error = nil;
    auto function = NS::TransferPtr(kernel->library->newFunction(functionName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    error = nil;
    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };

  auto kernelDesc = ANERowwiseTransformKernelDescriptor(memoryPrecision);
  auto kernel = createKernel(kernelDesc);
  auto computeActivationScales = NS::TransferPtr(createPipeline(kernel, "compute_activation_scales"));
  auto quantizeActivation = NS::TransferPtr(createPipeline(kernel, "quantize_transpose_activation"));
  auto dequantizeOutputTransposed = NS::TransferPtr(createPipeline(kernel, "dequantize_output_transposed"));
  auto dequantizeOutputTransposedBias = NS::TransferPtr(createPipeline(kernel, "dequantize_output_transposed_bias"));

  PipelineValue<ANERowwiseTransformKernel>* output =
      new PipelineValue<ANERowwiseTransformKernel> { kernel, computeActivationScales };
  output->second = quantizeActivation;
  output->third = dequantizeOutputTransposed;
  output->fourth = dequantizeOutputTransposedBias;
  return std::make_pair(kernelDesc, output);
}
