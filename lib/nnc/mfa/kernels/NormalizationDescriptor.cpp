#include "NormalizationDescriptor.hpp"
#include "NormalizationKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

static bool _normalization_descriptor_equals(const uint64_t data_type, const uint32_t channel_count, const uint32_t channel_groups, const uint32_t sequence_count, const float epsilon, const float scale, const uint8_t elementwise_affine, const uint8_t scale_translation_batched, const uint8_t normalization_type, const uint8_t reuse_saved_statistics, const bool load_m, const uint32_t src_batch_stride, const uint32_t dst_batch_stride, const uint64_t rhs_data_type, const uint32_t rhs_channel_count, const uint32_t rhs_channel_groups, const uint32_t rhs_sequence_count, const float rhs_epsilon, const float rhs_scale, const uint8_t rhs_elementwise_affine, const uint8_t rhs_scale_translation_batched, const uint8_t rhs_normalization_type, const uint8_t rhs_reuse_saved_statistics, const bool rhs_load_m, const uint32_t rhs_src_batch_stride, const uint32_t rhs_dst_batch_stride)
{
  return data_type == rhs_data_type &&
    channel_count == rhs_channel_count &&
    channel_groups == rhs_channel_groups &&
    load_m == rhs_load_m &&
    (load_m || sequence_count == rhs_sequence_count) &&
    epsilon == rhs_epsilon &&
    scale == rhs_scale &&
    elementwise_affine == rhs_elementwise_affine &&
    scale_translation_batched == rhs_scale_translation_batched &&
    normalization_type == rhs_normalization_type &&
    reuse_saved_statistics == rhs_reuse_saved_statistics &&
    src_batch_stride == rhs_src_batch_stride &&
    dst_batch_stride == rhs_dst_batch_stride;
}

bool NormalizationKernelDescriptor::operator==(const NormalizationKernelDescriptor& rhs) const {
  return _normalization_descriptor_equals(dataType, channelCount, channelGroups, sequenceCount, epsilon, scale, elementwiseAffine, scaleTranslationBatched, normalizationType, reuseSavedStatistics, loadM, srcBatchStride, dstBatchStride, rhs.dataType, rhs.channelCount, rhs.channelGroups, rhs.sequenceCount, rhs.epsilon, rhs.scale, rhs.elementwiseAffine, rhs.scaleTranslationBatched, rhs.normalizationType, rhs.reuseSavedStatistics, rhs.loadM, rhs.srcBatchStride, rhs.dstBatchStride);
}

bool NormalizationDescriptor::operator==(const NormalizationDescriptor& rhs) const {
  return _normalization_descriptor_equals(dataType, channelCount, channelGroups, sequenceCount, epsilon, scale, elementwiseAffine, scaleTranslationBatched, normalizationType, reuseSavedStatistics, loadM, srcBatchStride, dstBatchStride, rhs.dataType, rhs.channelCount, rhs.channelGroups, rhs.sequenceCount, rhs.epsilon, rhs.scale, rhs.elementwiseAffine, rhs.scaleTranslationBatched, rhs.normalizationType, rhs.reuseSavedStatistics, rhs.loadM, rhs.srcBatchStride, rhs.dstBatchStride);
}

static std::size_t _normalization_descriptor_hash(const uint64_t data_type, const uint32_t channel_count, const uint32_t channel_groups, const uint32_t sequence_count, const float epsilon, const float scale, const uint8_t elementwise_affine, const uint8_t scale_translation_batched, const uint8_t normalization_type, const uint8_t reuse_saved_statistics, const bool load_m, const uint32_t src_batch_stride, const uint32_t dst_batch_stride) noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, data_type);
  combine_64(seed, pack_64(simd::uint2 { channel_count, channel_groups }));
  combine_64(seed, pack_64(simd::uint2 { load_m ? 0 : sequence_count, *reinterpret_cast<const uint32_t*>(&epsilon) }));
  combine_32(seed, *reinterpret_cast<const uint32_t*>(&scale));
  combine_32(seed, pack_32(simd::uchar4 { elementwise_affine, scale_translation_batched, normalization_type, reuse_saved_statistics }));
  combine_32(seed, load_m ? 1 : 0);
  combine_64(seed, pack_64(simd::uint2 { src_batch_stride, dst_batch_stride }));
  return seed;
}

std::size_t std::hash<NormalizationKernelDescriptor>::operator()(const NormalizationKernelDescriptor& hash) const noexcept {
  return _normalization_descriptor_hash(hash.dataType, hash.channelCount, hash.channelGroups, hash.sequenceCount, hash.epsilon, hash.scale, hash.elementwiseAffine, hash.scaleTranslationBatched, hash.normalizationType, hash.reuseSavedStatistics, hash.loadM, hash.srcBatchStride, hash.dstBatchStride);
}

std::size_t std::hash<NormalizationDescriptor>::operator()(const NormalizationDescriptor& hash) const noexcept {
  return _normalization_descriptor_hash(hash.dataType, hash.channelCount, hash.channelGroups, hash.sequenceCount, hash.epsilon, hash.scale, hash.elementwiseAffine, hash.scaleTranslationBatched, hash.normalizationType, hash.reuseSavedStatistics, hash.loadM, hash.srcBatchStride, hash.dstBatchStride);
}

std::pair<NormalizationKernelDescriptor, PipelineValue<NormalizationKernel>*> NormalizationDescriptor::findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<NormalizationKernelDescriptor, std::unique_ptr<NormalizationKernel>> *const libraryCache) const noexcept {
  (void)dprops;
  (void)binaryArchivesToRead;
  (void)binaryArchiveToWrite;
  (void)pathToWrite;

  auto createKernel =
  [=](NormalizationKernelDescriptor descriptor) -> NormalizationKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      NormalizationKernel* kernel = new NormalizationKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<NormalizationKernel>(kernel);
      return kernel;
    }
  };

  NormalizationKernelDescriptor kernelDesc = {
    .dataType = dataType,
    .channelCount = channelCount,
    .channelGroups = channelGroups,
    .sequenceCount = sequenceCount,
    .epsilon = epsilon,
    .scale = scale,
    .elementwiseAffine = elementwiseAffine,
    .scaleTranslationBatched = scaleTranslationBatched,
    .normalizationType = normalizationType,
    .reuseSavedStatistics = reuseSavedStatistics,
    .loadM = loadM,
    .srcBatchStride = srcBatchStride,
    .dstBatchStride = dstBatchStride,
  };

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    NS::String* swiftName = NS::String::string("normalization", NS::UTF8StringEncoding);
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    NS::Error* error = nil;

    auto function = NS::TransferPtr(library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  NormalizationKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<NormalizationKernel>* output = new PipelineValue<NormalizationKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
