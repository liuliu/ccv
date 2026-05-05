#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_ane_rowwise_internal.hpp"
using namespace ccv::nnc;

#include <iostream>

// MARK: - C

mfa::context* ccv_nnc_init_mfa_context(MTL::Device* device) {
  return new mfa::context(device);
}

void ccv_nnc_mfa_clear_pipeline_cache(ccv_nnc_mfa_context_t* context) {
  context->kernel_cache.evict();
  ccv_nnc_mfa_ane_rowwise_gemm_cleanup(context);
}

void ccv_nnc_deinit_mfa_context(mfa::context* context) {
  ccv_nnc_mfa_ane_rowwise_gemm_cleanup(context);
  delete context;
}

uint8_t ccv_nnc_mfa_context_supported(mfa::context* context) {
  return context->supported ? 1 : 0;
}

uint8_t ccv_nnc_mfa_supports_int8_ane(ccv_nnc_mfa_context_t* context) {
  auto device = context->device;
  // Public CoreML rowwise GEMM path requires Apple9 or newer.
  return (device->supportsFamily(MTL::GPUFamily(1009)));
}

uint8_t ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_mfa_context_t* context) {
  auto device = context->device;
  // Only apple10 has Neural Accelerators.
  return (device->supportsFamily(MTL::GPUFamily(1010)));
}

uint8_t ccv_nnc_mfa_neural_accelerators_support_bfloat(ccv_nnc_mfa_context_t* context) {
  if (!ccv_nnc_mfa_has_neural_accelerators(context)) {
    return 0;
  }
#ifdef __has_builtin
#if __has_builtin(__builtin_available)
  if (__builtin_available(macos 26.1, iOS 26.1, macCatalyst 26.1, tvOS 26.1, *)) {
    return 1;
  } else {
    return 0;
  }
#else
  return 0;
#endif
#else
  return 0;
#endif
}

uint16_t ccv_nnc_mfa_context_log_level(mfa::context* context) {
  return context->log_level;
}

mtl_device_t* ccv_nnc_mfa_context_device(ccv_nnc_mfa_context_t* context) {
  return context->device.get();
}

void* ccv_nnc_mfa_context_get_ane_rowwise_gemm_cache(ccv_nnc_mfa_context_t* context) {
  return context->ane_rowwise_gemm_cache;
}

void ccv_nnc_mfa_context_set_ane_rowwise_gemm_cache(ccv_nnc_mfa_context_t* context, void* cache) {
  context->ane_rowwise_gemm_cache = cache;
}

PipelineValue<ANERowwiseTransformKernel>* ccv_nnc_mfa_prepare_ane_rowwise_transform(
    ccv_nnc_mfa_context_t* context,
    ANERowwiseTransformDescriptor descriptor) {
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto& shaderCache = context->kernel_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipelineValue =
    shaderCache.findKernel<ANERowwiseTransformKernel, ANERowwiseTransformDescriptor, ANERowwiseTransformKernelDescriptor>(
        descriptor, context->device.get(), dprops);
  pool->drain();
  return pipelineValue;
}

mtl_buffer_t* ccv_nnc_mfa_request_scratch(ccv_nnc_mfa_context_t* context, const uint64_t size) {
  return context->request_scratch(size);
}

void ccv_nnc_mfa_set_binary_archives(ccv_nnc_mfa_context_t* const context, const char** const paths_to_read, const int paths_to_read_size, const char* const path_to_write) {
  std::vector<std::string> paths_to_read_vec;
  for (int i = 0; i < paths_to_read_size; i++) {
    paths_to_read_vec.push_back(std::string(paths_to_read[i]));
  }
  std::string path_to_write_str = path_to_write != nullptr ? std::string(path_to_write) : std::string();
  context->kernel_cache.setBinaryArchives(context->device.get(), paths_to_read_vec, path_to_write_str);
}

void ccv_nnc_mfa_log_message(const char* message) {
  std::cerr << METAL_LOG_HEADER << message << std::endl;
}

MTL::CommandBatch* ccv_nnc_start_command_batch(MTL::CommandQueue* command_queue) {
  return new MTL::CommandBatch(command_queue);
}

MTL::CommandBatch* ccv_nnc_start_command_batch_from_command_buffer(MTL::CommandBuffer* command_buffer, int commit_on_finish) {
  return new MTL::CommandBatch(command_buffer, commit_on_finish != 0);
}

void ccv_nnc_finish_command_batch(MTL::CommandBatch* command_batch) {
  delete command_batch;
}

// MARK: - C++

mfa::context::context(MTL::Device* device)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  this->log_level = 0;
#if CCV_METAL_LOGGING_ENABLE
  const char* log_level_repr = getenv("CCV_METAL_LOG_LEVEL");
  if (log_level_repr) {
    int log_level_raw = atoi(log_level_repr);
    std::cerr << std::endl;
    std::cerr << METAL_LOG_HEADER << "Using log level: " << log_level_raw << std::endl;
    CCV_NNC_MFA_PRECONDITION(log_level_raw >= 0 && log_level_raw <= 4)
    
    this->log_level = uint16_t(log_level_raw);
  }
#endif
  
  this->device = NS::RetainPtr(device);

  this->scratch = NS::TransferPtr(device->newBuffer(65536, 0));
  this->ane_rowwise_gemm_cache = nullptr;

  // Check whether the device architecture is supported.
  this->supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!supported) {
    if (METAL_LOG_LEVEL(this) >= 1) {
      std::cerr << METAL_LOG_HEADER << "Device architecture not supported by MFA." << std::endl;
    }
    pool->drain();
    return;
  }

  pool->drain();
}

MTL::Buffer* mfa::context::request_scratch(uint64_t size) {
  if (size > scratch->length()) {
    uint64_t rounded_size = size;
    if (size < 0x20000000) { // If it is less than 512MiB, we pad it, otherwise we don't pad, just release & allocate. In this way, even we allocate a bit more, we allocate precisely what we need.
      uint64_t padded_size = std::max(int64_t(0), int64_t(size) - 1);
      uint64_t leading_zeroes = __builtin_clzll(padded_size);
      rounded_size = (uint64_t)1 << uint64_t(64 - leading_zeroes);
    }
    this->scratch.reset();
    auto buffer = device->newBuffer(rounded_size, MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked);
    CCV_NNC_MFA_PRECONDITION(buffer != nullptr);
    this->scratch = NS::TransferPtr(buffer);
  }
  return scratch.get();
}

MTL::CommandBatch::CommandBatch(MTL::CommandQueue* commandQueue) {
  commandBuffer = commandQueue->commandBuffer();
  commandEncoder = commandBuffer->computeCommandEncoder();
}

MTL::CommandBatch::CommandBatch(MTL::CommandBuffer* commandBuffer, bool commitOnDestruct) {
  this->commandBuffer = commandBuffer;
  this->commandEncoder = commandBuffer->computeCommandEncoder();
  this->commitOnDestruct = commitOnDestruct;
}

MTL::ComputeCommandEncoder* MTL::CommandBatch::startCommand() {
  CCV_NNC_MFA_PRECONDITION(commandActive == 0)
  commandActive = 1;
  return commandEncoder;
}

void MTL::CommandBatch::finishCommand(MTL::ComputeCommandEncoder* commandEncoder) {
  CCV_NNC_MFA_PRECONDITION(commandActive == 1)
  commandActive = 0;
  batchedCommandCount += 1;
}

MTL::CommandBatch::~CommandBatch() {
  CCV_NNC_MFA_PRECONDITION(commandActive == 0)
  if (commandEncoder) {
    commandEncoder->endEncoding();
  }
  if (commandBuffer && commitOnDestruct) {
    commandBuffer->commit();
  }
}
