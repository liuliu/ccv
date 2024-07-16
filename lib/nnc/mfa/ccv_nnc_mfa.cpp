#include "ccv_nnc_mfa.hpp"
#include "libmfa.inc"
using namespace ccv::nnc;

// MARK: - Testing the New GEMM Kernel

#include "ccv_nnc_mfa_error.hpp"
#include "GEMM/CoreCount.hpp"
#include "GEMM/GEMMDescriptor.hpp"
#include "GEMM/GEMMKernel.hpp"
#include "GEMM/GEMMShaderCache.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

void testFunction() {
  // insert code here...
  std::cout << "Hello, World!\n";
  
  // M1 Max
  //
  // 511^3, BF16, NN | 5149 GFLOPS
  // 511^3, BF16, NT | 4316 GFLOPS or 5559 GFLOPS
  // 511^3, BF16, TN | 4415 GFLOPS
  // 511^3, BF16, TT | 4282 GFLOPS or 5310 GFLOPS
  //
  // 512^3, BF16, NN | 5201 GFLOPS
  // 512^3, BF16, NT | 5265 GFLOPS
  // 512^3, BF16, TN | 4556 GFLOPS or 5880 GFLOPS
  // 512^3, BF16, TT | 5492 GFLOPS
  //
  // 1488^3, BF16, NN | 8371 GFLOPS
  // 1488^3, BF16, NT | 8683 GFLOPS
  // 1488^3, BF16, TN | 8807 GFLOPS
  // 1488^3, BF16, TT | 9041 GFLOPS
  //
  // 1489^3, BF16, NN | 8039 GFLOPS
  // 1489^3, BF16, NT | 8395 GFLOPS
  // 1489^3, BF16, TN | 8378 GFLOPS
  // 1489^3, BF16, TT | 8642 GFLOPS
  
  // Specify the problem configuration.
  int64_t problemSize = 10;
  
  // Instantiate the descriptor.
  GEMMDescriptor gemmDesc;
  gemmDesc.matrixDimensions = simd::uint3 {
    uint32_t(problemSize),
    uint32_t(problemSize),
    uint32_t(problemSize),
  };
  gemmDesc.memoryPrecisions = {
    .A = GEMMOperandPrecision::BF16,
    .B = GEMMOperandPrecision::BF16,
    .C = GEMMOperandPrecision::BF16,
  };
  gemmDesc.transposeState = simd::uchar2 { false, false };
  
  // Instantiate the kernel.
  auto pool = NS::AutoreleasePool::alloc()->init();
  GEMMShaderCache::fetchKernel(gemmDesc);
  auto pipelineValue = GEMMShaderCache::fetchKernel(gemmDesc);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;
  
  // Instantiate the device.
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  
  // Set up the diagonal matrix multiplication.
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> C;
  {
    // A 5x5 matrix defining the upper submatrix of B.
    std::vector<float> B_contents = {
      1.0, 2.0, 3.0, 4.0, 5.0,
      1.0, 2.0, 3.0, 4.0, 5.0,
      2.0, 3.0, 4.0, 5.0, 6.0,
      2.0, 4.0, 6.0, 8.0, 10.0,
      5.0, 4.0, 3.0, 2.0, 1.0,
    };
    for (int64_t rowID = 0; rowID < problemSize; ++rowID) {
      for (int64_t columnID = 0; columnID < problemSize; ++columnID) {
        if (rowID == columnID) {
          A.push_back(2);
        } else {
          A.push_back(0);
        }
        
        if (rowID < 5 && columnID < 5) {
          int64_t address = rowID * 5 + columnID;
          float value = B_contents[address];
          B.push_back(value);
        } else if (rowID == columnID) {
          B.push_back(1);
        } else {
          B.push_back(0);
        }
        
        C.push_back(0);
      }
    }
  }
  
  // Utility functions for type casting.
  auto memcpyDeviceTransfer =
  [=]
  (void *gpu, void *cpu, int64_t elements, bool isCPUToGPU,
   GEMMOperandPrecision type) {
    for (int64_t i = 0; i < elements; ++i) {
      if (type == GEMMOperandPrecision::FP32) {
        // FP32
        auto* gpuPointer = (float*)gpu + i;
        auto* cpuPointer = (float*)cpu + i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[0];
        } else {
          cpuPointer[0] = gpuPointer[0];
        }
      } else if (type.value == GEMMOperandPrecision::FP16) {
        // FP16
        auto* gpuPointer = (_Float16*)gpu + i;
        auto* cpuPointer = (float*)cpu + i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[0];
        } else {
          cpuPointer[0] = gpuPointer[0];
        }
      } else if (type.value == GEMMOperandPrecision::BF16) {
        // BF16
        auto* gpuPointer = (uint16_t*)gpu + i;
        auto* cpuPointer = (uint16_t*)cpu + 2 * i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[1];
        } else {
          cpuPointer[0] = 0;
          cpuPointer[1] = gpuPointer[0];
        }
      }
    }
  };
  
  // Allocate and fill the buffers.
  int64_t squareMatrixBytes = problemSize * problemSize * sizeof(float);
  auto bufferA = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  auto bufferB = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  auto bufferC = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  {
    int64_t elements = problemSize * problemSize;
    memcpyDeviceTransfer
    (bufferA->contents(), A.data(), elements, true,
     gemmDesc.memoryPrecisions.value().A);
    memcpyDeviceTransfer
    (bufferB->contents(), B.data(), elements, true,
     gemmDesc.memoryPrecisions.value().B);
  }
  
  // Instantiate the command queue.
  auto commandQueue = NS::TransferPtr(device->newCommandQueue());
  
  // Multiply A with B.
  int64_t maxGFLOPS = 0;
  int64_t occupancy = pipeline->maxTotalThreadsPerThreadgroup();
  for (int64_t trialID = 0; trialID < 15; ++trialID) {
    int64_t duplicatedCommandCount = 20;
    
    auto commandBuffer = commandQueue->commandBuffer();
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(bufferA.get(), 0, 0);
    encoder->setBuffer(bufferB.get(), 0, 1);
    encoder->setBuffer(bufferC.get(), 0, 2);
    
    for (int64_t commandID = 0; commandID < duplicatedCommandCount; ++commandID) {
      auto ceilDivide =
      [=](int64_t target, uint16_t granularity) -> int64_t {
        return (target + int64_t(granularity) - 1) / int64_t(granularity);
      };
      MTL::Size gridSize
      (ceilDivide(problemSize, kernel->blockDimensions[1]),
       ceilDivide(problemSize, kernel->blockDimensions[0]),
       1);
      MTL::Size groupSize
      (int64_t(kernel->threadgroupSize), 1, 1);
      encoder->dispatchThreadgroups(gridSize, groupSize);
    }
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Determine the time taken.
    double start = commandBuffer->GPUStartTime();
    double end = commandBuffer->GPUEndTime();
    double latency = end - start;
    
    // Determine the amount of work done.
    int64_t operations = 2 * problemSize * problemSize * problemSize;
    operations *= duplicatedCommandCount;
    int64_t gflops = int64_t(double(operations) / double(latency) / 1e9);
    
    // Report the results.
    maxGFLOPS = std::max(maxGFLOPS, gflops);
  }
  
  // Copy the results to C.
  {
    int64_t elements = problemSize * problemSize;
    memcpyDeviceTransfer
    (bufferC->contents(), C.data(), elements, false,
     gemmDesc.memoryPrecisions.value().C);
  }
  
  if (true) {
    // Display the matrices.
    auto displayMatrix =
    [=](float* matrix) {
      int64_t loopCount = std::min(int64_t(problemSize), int64_t(10));
      for (int64_t rowID = 0; rowID < loopCount; ++rowID) {
        for (int64_t columnID = 0; columnID < loopCount; ++columnID) {
          auto address = rowID * problemSize + columnID;
          float entry = matrix[address];
          
          std::cout << std::setprecision(4);
          std::cout << entry << " ";
        }
        std::cout << "\n";
      }
    };
    
    std::cout << "\n";
    std::cout << "A:\n";
    displayMatrix(A.data());
    
    std::cout << "\n";
    std::cout << "B:\n";
    displayMatrix(B.data());
    
    std::cout << "\n";
    std::cout << "C:\n";
    displayMatrix(C.data());
  }
  
  // Choose an error threshold.
  float errorThreshold = 1e-5;
  if (gemmDesc.memoryPrecisions.value().A == GEMMOperandPrecision::BF16) {
    errorThreshold = 2e-1;
  }
  if (gemmDesc.memoryPrecisions.value().B == GEMMOperandPrecision::BF16) {
    errorThreshold = 2e-1;
  }
  
  // Check the results.
  {
    int64_t errorCount = 0;
    for (int64_t rowID = 0; rowID < problemSize; ++rowID) {
      for (int64_t columnID = 0; columnID < problemSize; ++columnID) {
        float entryB = B[rowID * problemSize + columnID];
        float entryC;
        if (gemmDesc.transposeState.value()[1]) {
          entryC = C[columnID * problemSize + rowID];
        } else {
          entryC = C[rowID * problemSize + columnID];
        }
        
        float actual = entryC;
        float expected = entryB * 2;
        float error = actual - expected;
        if (error < 0) {
          error = -error;
        }
        
        if (error < errorThreshold) {
          // Skip ahead to the next iteration. There is no error message to
          // throw.
          continue;
        }
        if (errorCount > 10) {
          // Don't send too many messages to the console.
          continue;
        }
        errorCount += 1;
        
        std::cout << "C[" << rowID << "][" << columnID << "] | ";
        std::cout << "error: " << error << " | ";
        std::cout << "actual: " << actual << " | ";
        std::cout << "expected: " << expected << " | ";
        std::cout << std::endl;
      }
    }
  }
  
  // Report the performance.
  std::cout << std::endl;
  GEMMShaderCache::fetchKernel(gemmDesc);
  std::cout << maxGFLOPS << " GFLOPS ";
  std::cout << std::endl;
  std::cout << occupancy << " threads/core ";
  std::cout << std::endl;
}

// MARK: - C

#include <iostream>

mfa::context* ccv_nnc_init_mfa_context(MTL::Device* device) {
  {
    ccv_nnc_mfa_log_message("test function starting");
    testFunction();
    ccv_nnc_mfa_log_message("test function ending");
  }
  
  return new mfa::context(device);
}

void ccv_nnc_deinit_mfa_context(mfa::context* context) {
  delete context;
}

uint8_t ccv_nnc_mfa_context_supported(mfa::context* context) {
  return context->supported ? 1 : 0;
}

uint16_t ccv_nnc_mfa_context_log_level(mfa::context* context) {
  return context->log_level;
}

mtl_buffer_t* ccv_nnc_mfa_request_scratch(ccv_nnc_mfa_context_t* context, const uint64_t size) {
  return context->request_scratch(size);
}

void ccv_nnc_mfa_log_message(const char* message) {
  std::cerr << METAL_LOG_HEADER << message << std::endl;
}

MTL::CommandBatch* ccv_nnc_start_command_batch(MTL::CommandQueue* command_queue) {
  return new MTL::CommandBatch(command_queue);
}

void ccv_nnc_finish_command_batch(MTL::CommandBatch* command_batch) {
  delete command_batch;
}

// MARK: - C++

template <typename T, typename U>
mfa::cache<T, U>::cache()
{
  map = {};
}

template <typename T, typename U>
mfa::cache<T, U>::~cache()
{
  for (auto it = map.begin(); it != map.end(); ++it) {
    delete it->second;
  }
}

// This is a workaround. If we use a template member function directly, the
// symbols won't link.
template <typename T, typename U>
inline void _mfa_cache_prepare(std::unordered_map<T, U*>* map, mfa::context* context, T hash)
{
  if (map->find(hash) == map->end()) {
    if (METAL_LOG_LEVEL(context) >= 2) {
      std::cout << METAL_LOG_HEADER << "PSO cache miss." << std::endl;
      std::cout << METAL_LOG_HEADER << "  Contents of map (before):" << std::endl;
      for (auto it = map->begin(); it != map->end(); ++it) {
        std::cout << METAL_LOG_HEADER << "    " << it->first << ": " << it->second << std::endl;
      }
    }
    
    auto* pipeline = new U(context, hash);
    (*map)[hash] = pipeline;
    
    if (METAL_LOG_LEVEL(context) >= 2) {
      std::cout << METAL_LOG_HEADER << "  Contents of map (after):" << std::endl;
      for (auto it = map->begin(); it != map->end(); ++it) {
        std::cout << METAL_LOG_HEADER << "    " << it->first << ": " << it->second << std::endl;
      }
    }
  }
}

template <>
void mfa::cache<mfa::attention::hash, mfa::attention::pipeline>::prepare(mfa::context* context, mfa::attention::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::gemm::hash, mfa::gemm::pipeline>::prepare(mfa::context* context, mfa::gemm::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::normalization::hash, mfa::normalization::pipeline>::prepare(mfa::context* context, mfa::normalization::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::depalettize::hash, mfa::depalettize::pipeline>::prepare(mfa::context* context, mfa::depalettize::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::adam::hash, mfa::adam::pipeline>::prepare(mfa::context* context, mfa::adam::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::cmul::hash, mfa::cmul::pipeline>::prepare(mfa::context* context, mfa::cmul::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

template <>
void mfa::cache<mfa::gemv::hash, mfa::gemv::pipeline>::prepare(mfa::context* context, mfa::gemv::hash hash)
{
  _mfa_cache_prepare(&map, context, hash);
}

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
  
  // Example: /usr/local/MetalFlashAttention/lib/libMetalFlashAttention.metallib
  // We need to have two different variants based on the operating system. macOS
  // will not accept a metallib compiled for iOS/tvOS/visionOS and vice versa.
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Started loading 'libMetalFlashAttention.metallib'." << std::endl;
  }

  this->device = NS::RetainPtr(device);

  this->scratch = NS::TransferPtr(device->newBuffer(65536, 0));

  // Check whether the device architecture is supported.
  this->supported = device->supportsFamily(MTL::GPUFamilyApple7);
  if (!supported) {
    if (METAL_LOG_LEVEL(this) >= 1) {
      std::cerr << METAL_LOG_HEADER << "Device architecture not supported by Metal FlashAttention." << std::endl;
    }
    pool->drain();
    return;
  }
  
  const char *external_metallib_path = nullptr;
#if CCV_NNC_MFA_EXTERNAL_METALLIB_ENABLE
  external_metallib_path = getenv("CCV_NNC_MFA_METALLIB_PATH");
#endif
  if (METAL_LOG_LEVEL(this) >= 1) {
    if (external_metallib_path) {
      std::cerr << METAL_LOG_HEADER << "Loading from path '" << external_metallib_path << "'." << std::endl;
    } else {
      std::cerr << METAL_LOG_HEADER << "Loading from embedded string." << std::endl;
    }
  }
  
  // Attempt to load the library, otherwise crash with a detailed log message.
  NS::Error* error = nullptr;
  if (external_metallib_path) {
    auto string = NS::String::string(external_metallib_path, NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    this->library = NS::TransferPtr(device->newLibrary(url, &error));
  } else {
#if TARGET_OS_IPHONE
    dispatch_data_t data = dispatch_data_create(libmfaios16_v1_0_2_a_metallib, sizeof(libmfaios16_v1_0_2_a_metallib), NULL, 0);
#else
    dispatch_data_t data = dispatch_data_create(libmfamacos13_v1_0_2_a_metallib, sizeof(libmfamacos13_v1_0_2_a_metallib), NULL, 0);
#endif
    this->library = NS::TransferPtr(device->newLibrary(data, &error));
    dispatch_release(data);
  }
  if (!this->library) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  // Notify that this finished successfully, and is not just stalling on one of
  // the previous lines of code.
  if (METAL_LOG_LEVEL(this) >= 1) {
    std::cerr << METAL_LOG_HEADER << "Finished loading 'libMetalFlashAttention.metallib'." << std::endl;
  }
  
  pool->drain();
}

MTL::Buffer* mfa::context::request_scratch(uint64_t size) {
  if (size > scratch->length()) {
    uint64_t padded_size = std::max(int64_t(0), int64_t(size) - 1);
    uint64_t leading_zeroes = __builtin_clzll(padded_size);
    uint64_t rounded_size = 1 << uint64_t(64 - leading_zeroes);
    
    auto buffer = device->newBuffer(rounded_size, MTL::ResourceStorageModePrivate);
    CCV_NNC_MFA_PRECONDITION(buffer != nullptr);
    this->scratch = NS::TransferPtr(buffer);
  }
  return scratch.get();
}

MTL::CommandBatch::CommandBatch(MTL::CommandQueue* commandQueue) {
  commandBuffer = commandQueue->commandBuffer();
  commandEncoder = commandBuffer->computeCommandEncoder();
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
  commandEncoder->endEncoding();
  if (commandBuffer) {
    commandBuffer->commit();
  }
}
