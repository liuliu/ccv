#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <QuartzCore/QuartzCore.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/GemvDescriptor.hpp"
#include "nnc/mfa/kernels/GemvKernel.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

using half_float = _Float16;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
  int duplicated_dispatches = 8;
};

struct GemvCase {
  uint32_t M = 4096;
  uint32_t K = 4096;
  uint32_t pack = 16;
};

struct Stats {
  double average_seconds = 0;
  double best3_average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct ValidationStats {
  double max_abs = 0;
  double max_rel = 0;
};

struct SimplePipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  uint16_t threadgroup_size = 256;
};

struct ScalarGemvPipeline {
  GemvKernel* kernel = nullptr;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  uint32_t rows_per_threadgroup = 0;
};

struct MatmulPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  MTL::Size grid_size = MTL::Size(1, 1, 1);
  MTL::Size group_size = MTL::Size(32, 1, 1);
};

struct GemvPipelines {
  ScalarGemvPipeline scalar;
  SimplePipeline pack_padded;
  SimplePipeline pack_striped;
  SimplePipeline extract_first;
  SimplePipeline reduce_diagonal;
  MatmulPipeline padded_matmul;
  MatmulPipeline packed_matmul;
};

constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

uint32_t ceil_div(uint32_t x, uint32_t y)
{
  return (x + y - 1) / y;
}

template <typename T>
std::vector<T> make_data(size_t size, float scale, int phase)
{
  std::vector<T> values(size);
  for (size_t i = 0; i < size; ++i) {
    const int centered = (int)((i * 17 + phase * 23) % 41) - 20;
    values[i] = (T)(centered * scale);
  }
  return values;
}

std::vector<float> cpu_reference(
    const std::vector<half_float>& matrix,
    const std::vector<half_float>& vector,
    const GemvCase& bench)
{
  std::vector<float> output(bench.M, 0);
  for (uint32_t row = 0; row < bench.M; ++row) {
    float sum = 0;
    for (uint32_t k = 0; k < bench.K; ++k)
      sum += (float)matrix[(size_t)row * bench.K + k] * (float)vector[k];
    output[row] = sum;
  }
  return output;
}

std::vector<half_float> pack_weights_striped(
    const std::vector<half_float>& matrix,
    const GemvCase& bench)
{
  const uint32_t kpack = ceil_div(bench.K, bench.pack);
  std::vector<half_float> packed((size_t)bench.M * bench.pack * kpack, (half_float)0);
  for (uint32_t row = 0; row < bench.M; ++row)
    for (uint32_t p = 0; p < bench.pack; ++p)
      for (uint32_t kp = 0; kp < kpack; ++kp) {
        const uint32_t k = kp * bench.pack + p;
        if (k < bench.K)
          packed[((size_t)row * bench.pack + p) * kpack + kp] = matrix[(size_t)row * bench.K + k];
      }
  return packed;
}

ValidationStats validate_output(
    const std::vector<float>& reference,
    const half_float* actual,
    uint32_t length)
{
  ValidationStats stats;
  for (uint32_t i = 0; i < length; ++i) {
    const float lhs = reference[i];
    const float rhs = (float)actual[i];
    const float abs = std::fabs(lhs - rhs);
    const float rel = abs / std::max(1.0f, std::max(std::fabs(lhs), std::fabs(rhs)));
    stats.max_abs = std::max(stats.max_abs, (double)abs);
    stats.max_rel = std::max(stats.max_rel, (double)rel);
  }
  return stats;
}

bool benchmark(const BenchmarkConfig& config, const std::function<double()>& run_once, Stats* stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double elapsed = run_once();
    if (!(elapsed >= 0))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(elapsed);
  }
  if (samples.empty())
    return false;
  stats->average_seconds =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats->best3_average_seconds =
      std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

void print_stats(const char* label, const Stats& stats)
{
  std::cout << label
            << " avg_ms=" << std::fixed << std::setprecision(4) << stats.average_seconds * 1e3
            << " best3_avg_ms=" << stats.best3_average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << '\n';
}

ScalarGemvPipeline create_scalar_pipeline(
    MTL::Device* device,
    ShaderCache& shader_cache,
    const GemvCase& bench)
{
  GemvDescriptor descriptor;
  descriptor.fusedBias = 0;
  descriptor.mrows = 1;
  descriptor.memoryPrecision = GEMMOperandPrecision::FP16;
  descriptor.nrows = bench.M;
  descriptor.ncols = bench.K;
  DeviceProperties dprops{};
  auto pipeline_value =
      shader_cache.findKernel<GemvKernel, GemvDescriptor, GemvKernelDescriptor>(
          descriptor, device, dprops);
  ScalarGemvPipeline pipeline;
  pipeline.kernel = pipeline_value->kernel;
  pipeline.pipeline = pipeline_value->pipeline;
  pipeline.rows_per_threadgroup = GemvDescriptor::rowsPerThreadgroup(device);
  return pipeline;
}

std::string create_source(uint16_t block_m, uint16_t pack, uint16_t block_k, uint16_t simdgroups)
{
  std::ostringstream source;
  source
      << "#include <metal_stdlib>\n"
      << "#include <metal_tensor>\n"
      << "#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>\n"
      << "using namespace metal;\n"
      << "using namespace mpp::tensor_ops;\n"
      << "constant uint M [[function_constant(0)]];\n"
      << "constant uint N [[function_constant(1)]];\n"
      << "constant uint K [[function_constant(2)]];\n"
      << "constant uint PACK [[function_constant(3)]];\n"
      << "constant uint KPACK [[function_constant(4)]];\n"
      << "kernel void pack_padded_vector(\n"
      << "    device const half* src [[buffer(0)]],\n"
      << "    device half* dst [[buffer(1)]],\n"
      << "    uint gid [[thread_position_in_grid]])\n"
      << "{\n"
      << "  const uint total = PACK * K;\n"
      << "  if (gid >= total)\n"
      << "    return;\n"
      << "  const uint row = gid / K;\n"
      << "  const uint col = gid - row * K;\n"
      << "  dst[gid] = row == 0 ? src[col] : half(0);\n"
      << "}\n"
      << "kernel void pack_striped_vector(\n"
      << "    device const half* src [[buffer(0)]],\n"
      << "    device half* dst [[buffer(1)]],\n"
      << "    uint gid [[thread_position_in_grid]])\n"
      << "{\n"
      << "  const uint total = PACK * KPACK;\n"
      << "  if (gid >= total)\n"
      << "    return;\n"
      << "  const uint row = gid / KPACK;\n"
      << "  const uint col = gid - row * KPACK;\n"
      << "  const uint src_index = col * PACK + row;\n"
      << "  dst[gid] = src_index < K ? src[src_index] : half(0);\n"
      << "}\n"
      << "kernel void extract_first_column(\n"
      << "    device const half* src [[buffer(0)]],\n"
      << "    device half* dst [[buffer(1)]],\n"
      << "    uint gid [[thread_position_in_grid]])\n"
      << "{\n"
      << "  if (gid >= M)\n"
      << "    return;\n"
      << "  dst[gid] = src[gid * PACK];\n"
      << "}\n"
      << "kernel void reduce_diagonal(\n"
      << "    device const half* src [[buffer(0)]],\n"
      << "    device half* dst [[buffer(1)]],\n"
      << "    uint gid [[thread_position_in_grid]])\n"
      << "{\n"
      << "  if (gid >= M)\n"
      << "    return;\n"
      << "  float sum = 0;\n"
      << "  #pragma clang loop unroll(full)\n"
      << "  for (uint p = 0; p < PACK; ++p)\n"
      << "    sum += (float)src[((gid * PACK + p) * PACK) + p];\n"
      << "  dst[gid] = half(sum);\n"
      << "}\n"
      << "kernel void gemv_matmul(\n"
      << "    device half* A_buf [[buffer(0)]],\n"
      << "    device half* B_buf [[buffer(1)]],\n"
      << "    device half* C_buf [[buffer(2)]],\n"
      << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
      << "{\n"
      << "  const uint row_block = tgid.y * " << block_m << ";\n"
      << "  const uint col_block = tgid.x * " << pack << ";\n"
      << "  if (row_block >= M || col_block >= N)\n"
      << "    return;\n"
      << "  const uint row_size = min((uint)" << block_m << ", M - row_block);\n"
      << "  const uint col_size = min((uint)" << pack << ", N - col_block);\n"
      << "  auto A = tensor<device half, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M));\n"
      << "  auto B = tensor<device half, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N));\n"
      << "  if (row_size == " << block_m << " && col_size == " << pack << ") {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << block_m << ",\n"
      << "        " << pack << ",\n"
      << "        " << block_k << ",\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << simdgroups << ">> matmul_op;\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(A), decltype(B), half>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = half(0);\n"
      << "    }\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (uint k0 = 0; k0 + " << block_k << " <= K; k0 += " << block_k << ") {\n"
      << "      auto mA = A.slice<" << block_k << ", " << block_m << ">(k0, row_block);\n"
      << "      auto mB = B.slice<" << block_k << ", " << pack << ">(k0, col_block);\n"
      << "      matmul_op.run(mA, mB, cT);\n"
      << "    }\n"
      << "    if ((K % " << block_k << ") != 0) {\n"
      << "      constexpr auto residual_descriptor = matmul2d_descriptor(\n"
      << "          " << block_m << ",\n"
      << "          " << pack << ",\n"
      << "          dynamic_length_v<int>,\n"
      << "          false,\n"
      << "          true,\n"
      << "          true,\n"
      << "          matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "      matmul2d<residual_descriptor, execution_simdgroups<" << simdgroups << ">> residual_op;\n"
      << "      auto mAr = A.slice<dynamic_extent, " << block_m << ">(K / " << block_k << " * " << block_k << ", row_block);\n"
      << "      auto mBr = B.slice<dynamic_extent, " << pack << ">(K / " << block_k << " * " << block_k << ", col_block);\n"
      << "      residual_op.run(mAr, mBr, cT);\n"
      << "    }\n"
      << "    device half* mC = C_buf + row_block * N + col_block;\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        mC[idx[1] * N + idx[0]] = cT[i];\n"
      << "      }\n"
      << "    }\n"
      << "  } else {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << block_m << ",\n"
      << "        " << pack << ",\n"
      << "        dynamic_length_v<int>,\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << simdgroups << ">> matmul_op;\n"
      << "    auto mA = A.slice(0, row_block);\n"
      << "    auto mB = B.slice(0, col_block);\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = half(0);\n"
      << "    }\n"
      << "    matmul_op.run(mA, mB, cT);\n"
      << "    device half* mC = C_buf + row_block * N + col_block;\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        if (idx[0] < col_size && idx[1] < row_size)\n"
      << "          mC[idx[1] * N + idx[0]] = cT[i];\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "}\n";
  return source.str();
}

NS::SharedPtr<MTL::Library> create_library(
    MTL::Device* device,
    uint16_t block_m,
    uint16_t pack,
    uint16_t block_k,
    uint16_t simdgroups)
{
  const std::string source = create_source(block_m, pack, block_k, simdgroups);
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return library;
}

NS::SharedPtr<MTL::FunctionConstantValues> create_constants(
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t pack,
    uint32_t kpack)
{
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&pack, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&kpack, MTL::DataTypeUInt, NS::UInteger(4));
  return constants;
}

NS::SharedPtr<MTL::ComputePipelineState> create_pipeline(
    MTL::Device* device,
    MTL::Library* library,
    const char* name,
    MTL::FunctionConstantValues* constants)
{
  NS::Error* error = nil;
  auto function_name = NS::String::string(name, NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(function_name, constants, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

GemvPipelines create_pipelines(
    MTL::Device* device,
    ShaderCache& shader_cache,
    const GemvCase& bench,
    uint16_t block_m,
    uint16_t block_k,
    uint16_t simdgroups)
{
  const uint32_t kpack = ceil_div(bench.K, bench.pack);
  auto library = create_library(device, block_m, (uint16_t)bench.pack, block_k, simdgroups);

  auto helper_constants = create_constants(bench.M, bench.pack, bench.K, bench.pack, kpack);
  auto baseline_constants = create_constants(bench.M, bench.pack, bench.K, bench.pack, kpack);
  auto packed_constants = create_constants(bench.M * bench.pack, bench.pack, kpack, bench.pack, kpack);

  GemvPipelines pipelines;
  pipelines.scalar = create_scalar_pipeline(device, shader_cache, bench);
  pipelines.pack_padded.pipeline =
      create_pipeline(device, library.get(), "pack_padded_vector", helper_constants.get());
  pipelines.pack_striped.pipeline =
      create_pipeline(device, library.get(), "pack_striped_vector", helper_constants.get());
  pipelines.extract_first.pipeline =
      create_pipeline(device, library.get(), "extract_first_column", helper_constants.get());
  pipelines.reduce_diagonal.pipeline =
      create_pipeline(device, library.get(), "reduce_diagonal", helper_constants.get());
  pipelines.padded_matmul.pipeline =
      create_pipeline(device, library.get(), "gemv_matmul", baseline_constants.get());
  pipelines.packed_matmul.pipeline =
      create_pipeline(device, library.get(), "gemv_matmul", packed_constants.get());

  pipelines.pack_padded.threadgroup_size = 256;
  pipelines.pack_striped.threadgroup_size = 256;
  pipelines.extract_first.threadgroup_size = 256;
  pipelines.reduce_diagonal.threadgroup_size = 256;

  pipelines.padded_matmul.grid_size =
      MTL::Size(1, ceil_div(bench.M, block_m), 1);
  pipelines.padded_matmul.group_size = MTL::Size(simdgroups * 32, 1, 1);
  pipelines.packed_matmul.grid_size =
      MTL::Size(1, ceil_div(bench.M * bench.pack, block_m), 1);
  pipelines.packed_matmul.group_size = MTL::Size(simdgroups * 32, 1, 1);
  return pipelines;
}

double run_padded_once(
    MTL::CommandQueue* command_queue,
    const GemvCase& bench,
    const GemvPipelines& pipelines,
    int duplicated_dispatches,
    bool include_pack,
    MTL::Buffer* raw_vector,
    MTL::Buffer* padded_vector,
    MTL::Buffer* weights,
    MTL::Buffer* matmul_output,
    MTL::Buffer* final_output)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int duplicate = 0; duplicate < duplicated_dispatches; ++duplicate) {
    if (include_pack) {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.pack_padded.pipeline.get());
      encoder->setBuffer(raw_vector, 0, 0);
      encoder->setBuffer(padded_vector, 0, 1);
      const uint32_t total = bench.pack * bench.K;
      encoder->dispatchThreads(MTL::Size(total, 1, 1),
                               MTL::Size(pipelines.pack_padded.threadgroup_size, 1, 1));
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.padded_matmul.pipeline.get());
      encoder->setBuffer(weights, 0, 0);
      encoder->setBuffer(padded_vector, 0, 1);
      encoder->setBuffer(matmul_output, 0, 2);
      encoder->dispatchThreadgroups(
          pipelines.padded_matmul.grid_size,
          pipelines.padded_matmul.group_size);
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.extract_first.pipeline.get());
      encoder->setBuffer(matmul_output, 0, 0);
      encoder->setBuffer(final_output, 0, 1);
      encoder->dispatchThreads(MTL::Size(bench.M, 1, 1),
                               MTL::Size(pipelines.extract_first.threadgroup_size, 1, 1));
      encoder->endEncoding();
    }
  }
  const double start = CACurrentMediaTime();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const double end = CACurrentMediaTime();
  return (end - start) / duplicated_dispatches;
}

double run_packed_once(
    MTL::CommandQueue* command_queue,
    const GemvCase& bench,
    const GemvPipelines& pipelines,
    int duplicated_dispatches,
    bool include_pack,
    MTL::Buffer* raw_vector,
    MTL::Buffer* striped_vector,
    MTL::Buffer* packed_weights,
    MTL::Buffer* matmul_output,
    MTL::Buffer* final_output)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  for (int duplicate = 0; duplicate < duplicated_dispatches; ++duplicate) {
    if (include_pack) {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.pack_striped.pipeline.get());
      encoder->setBuffer(raw_vector, 0, 0);
      encoder->setBuffer(striped_vector, 0, 1);
      const uint32_t kpack = ceil_div(bench.K, bench.pack);
      const uint32_t total = bench.pack * kpack;
      encoder->dispatchThreads(MTL::Size(total, 1, 1),
                               MTL::Size(pipelines.pack_striped.threadgroup_size, 1, 1));
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.packed_matmul.pipeline.get());
      encoder->setBuffer(packed_weights, 0, 0);
      encoder->setBuffer(striped_vector, 0, 1);
      encoder->setBuffer(matmul_output, 0, 2);
      encoder->dispatchThreadgroups(
          pipelines.packed_matmul.grid_size,
          pipelines.packed_matmul.group_size);
      encoder->endEncoding();
    }
    {
      auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
      encoder->setComputePipelineState(pipelines.reduce_diagonal.pipeline.get());
      encoder->setBuffer(matmul_output, 0, 0);
      encoder->setBuffer(final_output, 0, 1);
      encoder->dispatchThreads(MTL::Size(bench.M, 1, 1),
                               MTL::Size(pipelines.reduce_diagonal.threadgroup_size, 1, 1));
      encoder->endEncoding();
    }
  }
  const double start = CACurrentMediaTime();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const double end = CACurrentMediaTime();
  return (end - start) / duplicated_dispatches;
}

double run_scalar_once(
    MTL::CommandQueue* command_queue,
    const GemvCase& bench,
    const GemvPipelines& pipelines,
    int duplicated_dispatches,
    MTL::Buffer* matrix,
    MTL::Buffer* raw_vector,
    MTL::Buffer* final_output)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipelines.scalar.pipeline.get());
  encoder->setBuffer(matrix, 0, 0);
  encoder->setBuffer(raw_vector, 0, 1);
  encoder->setBuffer(final_output, 0, 2);
  encoder->useResource(matrix, MTL::ResourceUsageRead);
  encoder->useResource(raw_vector, MTL::ResourceUsageRead);
  encoder->useResource(final_output, MTL::ResourceUsageWrite);
  const auto grid = MTL::Size(ceil_div(bench.M, pipelines.scalar.rows_per_threadgroup), 1, 1);
  for (int duplicate = 0; duplicate < duplicated_dispatches; ++duplicate)
    encoder->dispatchThreadgroups(
        grid, MTL::Size(pipelines.scalar.rows_per_threadgroup * 32, 1, 1));
  encoder->endEncoding();
  const double start = CACurrentMediaTime();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const double end = CACurrentMediaTime();
  return (end - start) / duplicated_dispatches;
}

} // namespace

int main(int argc, char** argv)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Failed to create command queue.\n";
    pool->drain();
    return 1;
  }

  ShaderCache shader_cache;

  GemvCase bench;
  BenchmarkConfig config;
  if (argc >= 3) {
    bench.M = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    bench.K = (uint32_t)std::strtoul(argv[2], nullptr, 10);
  }
  if (argc >= 4)
    bench.pack = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  if (argc >= 5)
    config.warmup_iterations = std::atoi(argv[4]);
  if (argc >= 6)
    config.timed_iterations = std::atoi(argv[5]);
  if (argc >= 7)
    config.duplicated_dispatches = std::atoi(argv[6]);

  if (bench.pack != 8 && bench.pack != 16) {
    std::cerr << "pack must be 8 or 16.\n";
    pool->drain();
    return 2;
  }

  uint16_t block_m = 16;
  uint16_t block_k = 16;
  uint16_t simdgroups = 1;
  if (argc >= 8)
    block_m = (uint16_t)std::strtoul(argv[7], nullptr, 10);
  if (argc >= 9)
    block_k = (uint16_t)std::strtoul(argv[8], nullptr, 10);
  if (argc >= 10)
    simdgroups = (uint16_t)std::strtoul(argv[9], nullptr, 10);
  const uint32_t kpack = ceil_div(bench.K, bench.pack);

  const auto matrix = make_data<half_float>((size_t)bench.M * bench.K, 0.03125f, 1);
  const auto vector = make_data<half_float>(bench.K, 0.0625f, 2);
  const auto packed_weights = pack_weights_striped(matrix, bench);
  const auto reference = cpu_reference(matrix, vector, bench);

  auto matrix_buffer = NS::TransferPtr(device->newBuffer(
      matrix.data(),
      matrix.size() * sizeof(half_float),
      kSharedResourceOptions));
  auto vector_buffer = NS::TransferPtr(device->newBuffer(
      vector.data(),
      vector.size() * sizeof(half_float),
      kSharedResourceOptions));
  auto packed_weight_buffer = NS::TransferPtr(device->newBuffer(
      packed_weights.data(),
      packed_weights.size() * sizeof(half_float),
      kSharedResourceOptions));
  auto padded_vector_buffer = NS::TransferPtr(device->newBuffer(
      (size_t)bench.pack * bench.K * sizeof(half_float),
      kSharedResourceOptions));
  auto striped_vector_buffer = NS::TransferPtr(device->newBuffer(
      (size_t)bench.pack * kpack * sizeof(half_float),
      kSharedResourceOptions));
  auto padded_matmul_output = NS::TransferPtr(device->newBuffer(
      (size_t)bench.M * bench.pack * sizeof(half_float),
      kSharedResourceOptions));
  auto packed_matmul_output = NS::TransferPtr(device->newBuffer(
      (size_t)bench.M * bench.pack * bench.pack * sizeof(half_float),
      kSharedResourceOptions));
  auto padded_final_output = NS::TransferPtr(device->newBuffer(
      (size_t)bench.M * sizeof(half_float),
      kSharedResourceOptions));
  auto packed_final_output = NS::TransferPtr(device->newBuffer(
      (size_t)bench.M * sizeof(half_float),
      kSharedResourceOptions));
  auto scalar_output = NS::TransferPtr(device->newBuffer(
      (size_t)bench.M * sizeof(half_float),
      kSharedResourceOptions));

  const auto pipelines = create_pipelines(device.get(), shader_cache, bench, block_m, block_k, simdgroups);

  const double padded_check = run_padded_once(
      command_queue.get(), bench, pipelines, 1, true, vector_buffer.get(), padded_vector_buffer.get(),
      matrix_buffer.get(), padded_matmul_output.get(), padded_final_output.get());
  const double packed_check = run_packed_once(
      command_queue.get(), bench, pipelines, 1, true, vector_buffer.get(), striped_vector_buffer.get(),
      packed_weight_buffer.get(), packed_matmul_output.get(), packed_final_output.get());
  const double scalar_check = run_scalar_once(
      command_queue.get(), bench, pipelines, 1, matrix_buffer.get(), vector_buffer.get(), scalar_output.get());
  (void)padded_check;
  (void)packed_check;
  (void)scalar_check;

  const auto padded_validation = validate_output(
      reference,
      static_cast<const half_float*>(padded_final_output->contents()),
      bench.M);
  const auto packed_validation = validate_output(
      reference,
      static_cast<const half_float*>(packed_final_output->contents()),
      bench.M);
  const auto scalar_validation = validate_output(
      reference,
      static_cast<const half_float*>(scalar_output->contents()),
      bench.M);

  Stats scalar_stats;
  Stats padded_compute_stats;
  Stats packed_compute_stats;
  Stats padded_total_stats;
  Stats packed_total_stats;

  if (!benchmark(
          config,
          [&]() {
            return run_scalar_once(
                command_queue.get(), bench, pipelines, config.duplicated_dispatches,
                matrix_buffer.get(), vector_buffer.get(), scalar_output.get());
          },
          &scalar_stats)) {
    std::cerr << "scalar benchmark failed\n";
    pool->drain();
    return 3;
  }
  if (!benchmark(
          config,
          [&]() {
            return run_padded_once(
                command_queue.get(), bench, pipelines, config.duplicated_dispatches, false, vector_buffer.get(),
                padded_vector_buffer.get(), matrix_buffer.get(),
                padded_matmul_output.get(), padded_final_output.get());
          },
          &padded_compute_stats)) {
    std::cerr << "padded compute benchmark failed\n";
    pool->drain();
    return 3;
  }
  if (!benchmark(
          config,
          [&]() {
            return run_packed_once(
                command_queue.get(), bench, pipelines, config.duplicated_dispatches, false, vector_buffer.get(),
                striped_vector_buffer.get(), packed_weight_buffer.get(),
                packed_matmul_output.get(), packed_final_output.get());
          },
          &packed_compute_stats)) {
    std::cerr << "packed compute benchmark failed\n";
    pool->drain();
    return 3;
  }
  if (!benchmark(
          config,
          [&]() {
            return run_padded_once(
                command_queue.get(), bench, pipelines, config.duplicated_dispatches, true, vector_buffer.get(),
                padded_vector_buffer.get(), matrix_buffer.get(),
                padded_matmul_output.get(), padded_final_output.get());
          },
          &padded_total_stats)) {
    std::cerr << "padded total benchmark failed\n";
    pool->drain();
    return 3;
  }
  if (!benchmark(
          config,
          [&]() {
            return run_packed_once(
                command_queue.get(), bench, pipelines, config.duplicated_dispatches, true, vector_buffer.get(),
                striped_vector_buffer.get(), packed_weight_buffer.get(),
                packed_matmul_output.get(), packed_final_output.get());
          },
          &packed_total_stats)) {
    std::cerr << "packed total benchmark failed\n";
    pool->drain();
    return 3;
  }

  std::cout << "shape M=" << bench.M
            << " K=" << bench.K
            << " pack=" << bench.pack
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << " duplicate=" << config.duplicated_dispatches
            << " blockM=" << block_m
            << " blockN=" << bench.pack
            << " blockK=" << block_k
            << " simdgroups=" << simdgroups
            << '\n';
  std::cout << "validation scalar max_abs=" << std::setprecision(6) << scalar_validation.max_abs
            << " max_rel=" << scalar_validation.max_rel << '\n';
  std::cout << "validation padded max_abs=" << std::setprecision(6) << padded_validation.max_abs
            << " max_rel=" << padded_validation.max_rel << '\n';
  std::cout << "validation packed max_abs=" << packed_validation.max_abs
            << " max_rel=" << packed_validation.max_rel << '\n';
  print_stats("scalar", scalar_stats);
  print_stats("padded_compute", padded_compute_stats);
  print_stats("packed_compute", packed_compute_stats);
  print_stats("padded_total", padded_total_stats);
  print_stats("packed_total", packed_total_stats);
  std::cout << "speedup total packed_vs_scalar="
            << (scalar_stats.median_seconds / packed_total_stats.median_seconds)
            << " padded_vs_scalar="
            << (scalar_stats.median_seconds / padded_total_stats.median_seconds)
            << '\n';
  std::cout << "speedup compute packed_vs_padded="
            << (padded_compute_stats.median_seconds / packed_compute_stats.median_seconds)
            << " total packed_vs_padded="
            << (padded_total_stats.median_seconds / packed_total_stats.median_seconds)
            << '\n';

  fflush(stdout);
  fflush(stderr);
  std::_Exit(0);
}
