#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/NAMatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAMatMulKernel.hpp"
#include "nnc/mfa/kernels/NAMatMulKernelDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.hpp"

namespace {

using half_float = _Float16;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
};

struct BenchmarkCase {
  uint32_t M = 4096;
  uint32_t N = 4096;
  uint32_t K = 4096;
};

struct VariantConfig {
  simd::ushort3 block_dimensions = simd::ushort3 { 128, 128, 128 };
  uint16_t execution_simd_groups = 8;
  uint16_t activation_quant_threads = 256;
  uint32_t group_m = UINT32_MAX;
  uint32_t group_n = UINT32_MAX;
  uint16_t split_k = 1;
  bool load_m = false;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double best3_average_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct ValidationStats {
  bool passed = false;
  bool full_reference = false;
  size_t checked_rows = 0;
  size_t checked_cols = 0;
  double max_abs = 0;
  double max_rel = 0;
};

struct QuantizationValidationStats {
  bool passed = false;
  size_t mismatched_values = 0;
  double max_abs_scale = 0;
};

struct RowwiseQuantizedMatrix {
  std::vector<int8_t> values;
  std::vector<float> scales;
};

struct BaselinePipeline {
  NAMatMulDescriptor descriptor;
  std::unique_ptr<NAMatMulKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct DynamicPipeline {
  std::unique_ptr<NAInt8MatMulKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  bool load_m = false;
};

struct RawPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

struct SplitKPipeline {
  NS::SharedPtr<MTL::ComputePipelineState> partial_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> dequant_pipeline;
  uint16_t threadgroup_size = 0;
};

struct QuantizePipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  uint16_t threadgroup_size = 0;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

size_t a_index(const BenchmarkCase& bench, uint32_t row, uint32_t k)
{
  return (size_t)row * bench.K + k;
}

size_t b_index(const BenchmarkCase& bench, uint32_t row, uint32_t k)
{
  return (size_t)row * bench.K + k;
}

size_t c_index(const BenchmarkCase& bench, uint32_t row, uint32_t col)
{
  return (size_t)row * bench.N + col;
}

std::vector<half_float> make_half_matrix(uint32_t rows, uint32_t cols, float scale, int phase)
{
  std::vector<half_float> values((size_t)rows * cols);
  for (uint32_t row = 0; row < rows; ++row) {
    const float row_gain = 0.2f + 0.8f * (float)(((row * 7 + phase * 5) % 23) + 1) / 24.0f;
    for (uint32_t col = 0; col < cols; ++col) {
      const int centered = (int)((row * 131 + col * 17 + phase * 29) % 127) - 63;
      values[(size_t)row * cols + col] = (half_float)(centered * scale * row_gain);
    }
  }
  return values;
}

std::vector<float> half_to_float_vector(const std::vector<half_float>& values)
{
  std::vector<float> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(), [](half_float value) {
    return (float)value;
  });
  return output;
}

RowwiseQuantizedMatrix quantize_rowwise(
    const std::vector<float>& values,
    uint32_t rows,
    uint32_t cols)
{
  RowwiseQuantizedMatrix quantized;
  quantized.values.resize(values.size());
  quantized.scales.resize(rows);
  for (uint32_t row = 0; row < rows; ++row) {
    float max_abs = 0;
    for (uint32_t col = 0; col < cols; ++col)
      max_abs = std::max(max_abs, std::fabs(values[(size_t)row * cols + col]));
    const float scale = max_abs > 0 ? max_abs / 127.0f : (1.0f / 127.0f);
    const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
    quantized.scales[row] = scale;
    for (uint32_t col = 0; col < cols; ++col) {
      const int rounded = (int)std::lrint(values[(size_t)row * cols + col] * inv_scale);
      quantized.values[(size_t)row * cols + col] = (int8_t)std::max(-127, std::min(127, rounded));
    }
  }
  return quantized;
}

uint32_t groupM(uint32_t M) noexcept
{
  return M >= 4096 ? 4096 : 0;
}

uint32_t groupN(uint32_t N) noexcept
{
  return N >= 4096 ? 4096 : 0;
}

uint32_t groupM(const BenchmarkCase& bench, const VariantConfig& variant) noexcept
{
  return variant.group_m == UINT32_MAX ? groupM(bench.M) : variant.group_m;
}

uint32_t groupN(const BenchmarkCase& bench, const VariantConfig& variant) noexcept
{
  return variant.group_n == UINT32_MAX ? groupN(bench.N) : variant.group_n;
}

BaselinePipeline create_baseline_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    bool load_m)
{
  BaselinePipeline bundle;
  bundle.descriptor.batchDimension = 1;
  bundle.descriptor.matrixDimensions = simd::uint3 { bench.M, bench.N, bench.K };
  bundle.descriptor.memoryPrecisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  bundle.descriptor.registerPrecisionC = std::make_optional(GEMMOperandPrecision::FP16);
  bundle.descriptor.batchStrides = std::nullopt;
  bundle.descriptor.transposeState = simd::uchar3 { 0, 1, 0 };
  bundle.descriptor.useBias = false;
  bundle.descriptor.loadM = load_m;
  bundle.descriptor.supportIndirectCommandBuffers = false;

  const GEMMOperandPrecisions register_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const NAMatMulKernelDescriptor kernel_descriptor(
      simd::ushort3 { 128, 64, 64 },
      bundle.descriptor.memoryPrecisions,
      register_precisions,
      1,
      4,
      false,
      bundle.descriptor.transposeState,
      false,
      load_m,
      groupM(bench.M),
      groupN(bench.N));
  bundle.kernel = std::make_unique<NAMatMulKernel>(kernel_descriptor, device);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  const bool batched = false;
  const uint32_t zero = 0;
  if (!load_m)
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&batched, MTL::DataTypeBool, NS::UInteger(11));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(15));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(16));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(17));
  constants->setConstantValue(&zero, MTL::DataTypeUInt, NS::UInteger(18));

  NS::Error* error = nil;
  auto function_name = NS::String::string("matmul", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(bundle.kernel->library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

DynamicPipeline create_dynamic_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
{
  DynamicPipeline bundle;
  const NAInt8MatMulKernelDescriptor kernel_descriptor(
      variant.block_dimensions,
      variant.execution_simd_groups,
      GEMMOperandPrecision::FP16,
      false,
      variant.load_m,
      variant.activation_quant_threads,
      groupM(bench, variant),
      groupN(bench, variant));
  bundle.kernel = std::make_unique<NAInt8MatMulKernel>(kernel_descriptor, device);
  bundle.load_m = variant.load_m;

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  if (!variant.load_m)
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));

  NS::Error* error = nil;
  auto function_name = NS::String::string("int8_matmul", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(bundle.kernel->library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());
  bundle.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return bundle;
}

RawPipeline create_raw_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
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
      << "inline uint compact_morton_even_bits(uint x) {\n"
      << "  x &= 0x55555555u;\n"
      << "  x = (x | (x >> 1)) & 0x33333333u;\n"
      << "  x = (x | (x >> 2)) & 0x0f0f0f0fu;\n"
      << "  x = (x | (x >> 4)) & 0x00ff00ffu;\n"
      << "  x = (x | (x >> 8)) & 0x0000ffffu;\n"
      << "  return x;\n"
      << "}\n"
      << "inline uint2 morton_decode_2d(uint code) {\n"
      << "  return uint2(compact_morton_even_bits(code), compact_morton_even_bits(code >> 1));\n"
      << "}\n"
      << "inline uint lower_bits_mask(uint bit_count) {\n"
      << "  if (bit_count == 0)\n"
      << "    return 0;\n"
      << "  return (1u << bit_count) - 1;\n"
      << "}\n"
      << "inline uint2 morton_decode_rectangular_2d(uint code, uint x_bits, uint y_bits) {\n"
      << "  const uint paired_bits = min(x_bits, y_bits);\n"
      << "  const uint paired_code = code & lower_bits_mask(paired_bits * 2);\n"
      << "  uint2 tile = morton_decode_2d(paired_code);\n"
      << "  uint tail = code >> (paired_bits * 2);\n"
      << "  if (x_bits > paired_bits) {\n"
      << "    const uint x_extra_bits = x_bits - paired_bits;\n"
      << "    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;\n"
      << "    tail >>= x_extra_bits;\n"
      << "  }\n"
      << "  if (y_bits > paired_bits)\n"
      << "    tile.y |= tail << paired_bits;\n"
      << "  return tile;\n"
      << "}\n"
      << "kernel void int8_matmul_raw(\n"
      << "    device int8_t *A_buf [[buffer(0)]],\n"
      << "    device int8_t *B_buf [[buffer(1)]],\n"
      << "    device int *C_buf [[buffer(2)]],\n"
      << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
      << "{\n"
      << "  const uint M_tiles = (M + " << variant.block_dimensions[0] << " - 1) / " << variant.block_dimensions[0] << ";\n"
      << "  const uint N_tiles = (N + " << variant.block_dimensions[1] << " - 1) / " << variant.block_dimensions[1] << ";\n"
      << "  const uint M_tile_bits = M_tiles <= 1 ? 0 : 32 - clz(M_tiles - 1);\n"
      << "  const uint N_tile_bits = N_tiles <= 1 ? 0 : 32 - clz(N_tiles - 1);\n"
      << "  uint2 morton_tile = morton_decode_rectangular_2d(tgid.x, N_tile_bits, M_tile_bits);\n"
      << "  tgid.x = morton_tile.x;\n"
      << "  tgid.y = morton_tile.y;\n"
      << "  if (tgid.x >= N_tiles || tgid.y >= M_tiles)\n"
      << "    return;\n"
      << "  const uint M_block_start = tgid.y * " << variant.block_dimensions[0] << ";\n"
      << "  const uint M_block_size = min((uint)" << variant.block_dimensions[0] << ", M - M_block_start);\n"
      << "  const uint N_block_start = tgid.x * " << variant.block_dimensions[1] << ";\n"
      << "  const uint N_block_size = min((uint)" << variant.block_dimensions[1] << ", N - N_block_start);\n"
      << "  const uint M_group_start = " << groupM(bench, variant) << " ? (M_block_start / " << groupM(bench, variant) << ") * " << groupM(bench, variant) << " : M_block_start;\n"
      << "  const uint M_group_offset = M_block_start - M_group_start;\n"
      << "  const uint M_group_size = M - M_group_start;\n"
      << "  const uint N_group_start = " << groupN(bench, variant) << " ? (N_block_start / " << groupN(bench, variant) << ") * " << groupN(bench, variant) << " : N_block_start;\n"
      << "  const uint N_group_offset = N_block_start - N_group_start;\n"
      << "  const uint N_group_size = N - N_group_start;\n"
      << "  A_buf += M_group_start * K;\n"
      << "  B_buf += N_group_start * K;\n"
      << "  C_buf += M_group_start * N;\n"
      << "  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_group_size));\n"
      << "  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_group_size));\n"
      << "  if (N_block_start + " << variant.block_dimensions[1] << " - 1 < N && M_block_start + " << variant.block_dimensions[0] << " - 1 < M) {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << variant.block_dimensions[0] << ",\n"
      << "        " << variant.block_dimensions[1] << ",\n"
      << "        " << variant.block_dimensions[2] << ",\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> matmul_op;\n"
      << "    auto mA = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(0, M_group_offset);\n"
      << "    auto mB = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(0, N_group_offset);\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = 0;\n"
      << "    }\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (uint k = 0; k + " << variant.block_dimensions[2] << " <= K; k += " << variant.block_dimensions[2] << ") {\n"
      << "      auto mAk = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(k, M_group_offset);\n"
      << "      auto mBk = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(k, N_group_offset);\n"
      << "      matmul_op.run(mAk, mBk, cT);\n"
      << "    }\n"
      << "    if (K % " << variant.block_dimensions[2] << " != 0) {\n"
      << "      constexpr auto residual_descriptor = matmul2d_descriptor(\n"
      << "          " << variant.block_dimensions[0] << ",\n"
      << "          " << variant.block_dimensions[1] << ",\n"
      << "          dynamic_length_v<int>,\n"
      << "          false,\n"
      << "          true,\n"
      << "          true,\n"
      << "          matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "      matmul2d<residual_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> residual_op;\n"
      << "      auto mAr = A.slice<dynamic_extent, " << variant.block_dimensions[0] << ">(K / " << variant.block_dimensions[2] << " * " << variant.block_dimensions[2] << ", M_group_offset);\n"
      << "      auto mBr = B.slice<dynamic_extent, " << variant.block_dimensions[1] << ">(K / " << variant.block_dimensions[2] << " * " << variant.block_dimensions[2] << ", N_group_offset);\n"
      << "      residual_op.run(mAr, mBr, cT);\n"
      << "    }\n"
      << "    auto mC = C_buf + M_group_offset * N + N_block_start;\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        mC[idx[1] * N + idx[0]] = cT[i];\n"
      << "      }\n"
      << "    }\n"
      << "  } else {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << variant.block_dimensions[0] << ",\n"
      << "        " << variant.block_dimensions[1] << ",\n"
      << "        dynamic_length_v<int>,\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> matmul_op;\n"
      << "    auto mA = A.slice(0, M_group_offset);\n"
      << "    auto mB = B.slice(0, N_group_offset);\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = 0;\n"
      << "    }\n"
      << "    matmul_op.run(mA, mB, cT);\n"
      << "    auto mC = C_buf + M_group_offset * N + N_block_start;\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        if (idx[0] < N_block_size && idx[1] < M_block_size)\n"
      << "          mC[idx[1] * N + idx[0]] = cT[i];\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "}\n";

  auto string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));

  auto function_name = NS::String::string("int8_matmul_raw", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());

  RawPipeline pipeline;
  pipeline.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return pipeline;
}

SplitKPipeline create_splitk_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
{
  CCV_NNC_MFA_PRECONDITION(variant.split_k > 1);
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
      << "constant uint SPLIT_K [[function_constant(3)]];\n"
      << "inline uint compact_morton_even_bits(uint x) {\n"
      << "  x &= 0x55555555u;\n"
      << "  x = (x | (x >> 1)) & 0x33333333u;\n"
      << "  x = (x | (x >> 2)) & 0x0f0f0f0fu;\n"
      << "  x = (x | (x >> 4)) & 0x00ff00ffu;\n"
      << "  x = (x | (x >> 8)) & 0x0000ffffu;\n"
      << "  return x;\n"
      << "}\n"
      << "inline uint2 morton_decode_2d(uint code) {\n"
      << "  return uint2(compact_morton_even_bits(code), compact_morton_even_bits(code >> 1));\n"
      << "}\n"
      << "inline uint lower_bits_mask(uint bit_count) {\n"
      << "  if (bit_count == 0)\n"
      << "    return 0;\n"
      << "  return (1u << bit_count) - 1;\n"
      << "}\n"
      << "inline uint2 morton_decode_rectangular_2d(uint code, uint x_bits, uint y_bits) {\n"
      << "  const uint paired_bits = min(x_bits, y_bits);\n"
      << "  const uint paired_code = code & lower_bits_mask(paired_bits * 2);\n"
      << "  uint2 tile = morton_decode_2d(paired_code);\n"
      << "  uint tail = code >> (paired_bits * 2);\n"
      << "  if (x_bits > paired_bits) {\n"
      << "    const uint x_extra_bits = x_bits - paired_bits;\n"
      << "    tile.x |= (tail & lower_bits_mask(x_extra_bits)) << paired_bits;\n"
      << "    tail >>= x_extra_bits;\n"
      << "  }\n"
      << "  if (y_bits > paired_bits)\n"
      << "    tile.y |= tail << paired_bits;\n"
      << "  return tile;\n"
      << "}\n"
      << "constant uint BLOCK_N [[function_constant(4)]];\n"
      << "kernel void int8_matmul_splitk(\n"
      << "    device int8_t *A_buf [[buffer(0)]],\n"
      << "    device int8_t *B_buf [[buffer(1)]],\n"
      << "    device int *C_accum [[buffer(2)]],\n"
      << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
      << "{\n"
      << "  const uint M_tiles = (M + " << variant.block_dimensions[0] << " - 1) / " << variant.block_dimensions[0] << ";\n"
      << "  const uint N_tiles = (N + " << variant.block_dimensions[1] << " - 1) / " << variant.block_dimensions[1] << ";\n"
      << "  const uint M_tile_bits = M_tiles <= 1 ? 0 : 32 - clz(M_tiles - 1);\n"
      << "  const uint N_tile_bits = N_tiles <= 1 ? 0 : 32 - clz(N_tiles - 1);\n"
      << "  const uint tile_codes = 1u << (M_tile_bits + N_tile_bits);\n"
      << "  const uint split_index = tgid.x / tile_codes;\n"
      << "  uint2 morton_tile = morton_decode_rectangular_2d(tgid.x % tile_codes, N_tile_bits, M_tile_bits);\n"
      << "  tgid.x = morton_tile.x;\n"
      << "  tgid.y = morton_tile.y;\n"
      << "  if (split_index >= SPLIT_K || tgid.x >= N_tiles || tgid.y >= M_tiles)\n"
      << "    return;\n"
      << "  const uint M_block_start = tgid.y * " << variant.block_dimensions[0] << ";\n"
      << "  const uint M_block_size = min((uint)" << variant.block_dimensions[0] << ", M - M_block_start);\n"
      << "  const uint N_block_start = tgid.x * " << variant.block_dimensions[1] << ";\n"
      << "  const uint N_block_size = min((uint)" << variant.block_dimensions[1] << ", N - N_block_start);\n"
      << "  const uint M_group_start = " << groupM(bench, variant) << " ? (M_block_start / " << groupM(bench, variant) << ") * " << groupM(bench, variant) << " : M_block_start;\n"
      << "  const uint M_group_offset = M_block_start - M_group_start;\n"
      << "  const uint M_group_size = M - M_group_start;\n"
      << "  const uint N_group_start = " << groupN(bench, variant) << " ? (N_block_start / " << groupN(bench, variant) << ") * " << groupN(bench, variant) << " : N_block_start;\n"
      << "  const uint N_group_offset = N_block_start - N_group_start;\n"
      << "  const uint N_group_size = N - N_group_start;\n"
      << "  const uint K_split = K / SPLIT_K / " << variant.block_dimensions[2] << " * " << variant.block_dimensions[2] << ";\n"
      << "  A_buf += M_group_start * K;\n"
      << "  B_buf += N_group_start * K;\n"
      << "  C_accum += M_group_start * N * SPLIT_K;\n"
      << "  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(K, M_group_size));\n"
      << "  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(K, N_group_size));\n"
      << "  if (N_block_start + " << variant.block_dimensions[1] << " - 1 < N && M_block_start + " << variant.block_dimensions[0] << " - 1 < M) {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << variant.block_dimensions[0] << ",\n"
      << "        " << variant.block_dimensions[1] << ",\n"
      << "        " << variant.block_dimensions[2] << ",\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> matmul_op;\n"
      << "    auto mA = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(0, M_group_offset);\n"
      << "    auto mB = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(0, N_group_offset);\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = 0;\n"
      << "    }\n"
      << "    if (split_index == 0) {\n"
      << "      #pragma clang loop unroll(full)\n"
      << "      for (uint k = 0; k < K_split; k += " << variant.block_dimensions[2] << ") {\n"
      << "        auto mAk = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(k, M_group_offset);\n"
      << "        auto mBk = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(k, N_group_offset);\n"
      << "        matmul_op.run(mAk, mBk, cT);\n"
      << "      }\n"
      << "    } else if (split_index + 1 < SPLIT_K) {\n"
      << "      const uint k_start = split_index * K_split;\n"
      << "      #pragma clang loop unroll(full)\n"
      << "      for (uint i_k = 0; i_k < K_split; i_k += " << variant.block_dimensions[2] << ") {\n"
      << "        const uint k = k_start + i_k;\n"
      << "        auto mAk = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(k, M_group_offset);\n"
      << "        auto mBk = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(k, N_group_offset);\n"
      << "        matmul_op.run(mAk, mBk, cT);\n"
      << "      }\n"
      << "    } else {\n"
      << "      const uint k_tail_start = (SPLIT_K - 1) * K_split;\n"
      << "      #pragma clang loop unroll(full)\n"
      << "      for (uint k = k_tail_start; k + " << variant.block_dimensions[2] << " <= K; k += " << variant.block_dimensions[2] << ") {\n"
      << "        auto mAk = A.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[0] << ">(k, M_group_offset);\n"
      << "        auto mBk = B.slice<" << variant.block_dimensions[2] << ", " << variant.block_dimensions[1] << ">(k, N_group_offset);\n"
      << "        matmul_op.run(mAk, mBk, cT);\n"
      << "      }\n"
      << "      if (K % " << variant.block_dimensions[2] << " != 0) {\n"
      << "        constexpr auto residual_descriptor = matmul2d_descriptor(\n"
      << "            " << variant.block_dimensions[0] << ",\n"
      << "            " << variant.block_dimensions[1] << ",\n"
      << "            dynamic_length_v<int>,\n"
      << "            false,\n"
      << "            true,\n"
      << "            true,\n"
      << "            matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "        matmul2d<residual_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> residual_op;\n"
      << "        auto mAr = A.slice<dynamic_extent, " << variant.block_dimensions[0] << ">(K / " << variant.block_dimensions[2] << " * " << variant.block_dimensions[2] << ", M_group_offset);\n"
      << "        auto mBr = B.slice<dynamic_extent, " << variant.block_dimensions[1] << ">(K / " << variant.block_dimensions[2] << " * " << variant.block_dimensions[2] << ", N_group_offset);\n"
      << "        residual_op.run(mAr, mBr, cT);\n"
      << "      }\n"
      << "    }\n"
      << "    auto mC = C_accum + M_group_offset * N * SPLIT_K + N_block_start * SPLIT_K + split_index * " << variant.block_dimensions[1] << ";\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        mC[idx[1] * N * SPLIT_K + idx[0]] = cT[i];\n"
      << "      }\n"
      << "    }\n"
      << "  } else {\n"
      << "    constexpr auto matmul_descriptor = matmul2d_descriptor(\n"
      << "        " << variant.block_dimensions[0] << ",\n"
      << "        " << variant.block_dimensions[1] << ",\n"
      << "        dynamic_length_v<int>,\n"
      << "        false,\n"
      << "        true,\n"
      << "        true,\n"
      << "        matmul2d_descriptor::mode::multiply_accumulate);\n"
      << "    matmul2d<matmul_descriptor, execution_simdgroups<" << variant.execution_simd_groups << ">> matmul_op;\n"
      << "    auto mA = A.slice(0, M_group_offset);\n"
      << "    auto mB = B.slice(0, N_group_offset);\n"
      << "    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i))\n"
      << "        cT[i] = 0;\n"
      << "    }\n"
      << "    if (split_index == 0) {\n"
      << "      auto mA0 = A.slice(0, M_group_offset);\n"
      << "      auto mB0 = B.slice(0, N_group_offset);\n"
      << "      matmul_op.run(mA0, mB0, cT);\n"
      << "    }\n"
      << "    auto mC = C_accum + M_group_offset * N * SPLIT_K + N_block_start * SPLIT_K + split_index * " << variant.block_dimensions[1] << ";\n"
      << "    #pragma clang loop unroll(full)\n"
      << "    for (unsigned short i = 0; i < cT.get_capacity(); ++i) {\n"
      << "      if (cT.is_valid_element(i)) {\n"
      << "        auto idx = cT.get_multidimensional_index(i);\n"
      << "        if (idx[0] < N_block_size && idx[1] < M_block_size) {\n"
      << "          mC[idx[1] * N * SPLIT_K + idx[0]] = cT[i];\n"
      << "        }\n"
      << "      }\n"
      << "    }\n"
      << "  }\n"
      << "}\n"
      << "kernel void reduce_dequant_accumulator(\n"
      << "    device const int *C_accum [[buffer(0)]],\n"
      << "    device half *C_out [[buffer(1)]],\n"
      << "    device const half *A_scale [[buffer(2)]],\n"
      << "    device const half *B_scale [[buffer(3)]],\n"
      << "    uint gid [[thread_position_in_grid]])\n"
      << "{\n"
      << "  const uint total = M * N;\n"
      << "  if (gid >= total)\n"
      << "    return;\n"
      << "  const uint row = gid / N;\n"
      << "  const uint col = gid - row * N;\n"
      << "  const uint accum_offset = row * N * SPLIT_K + (col / BLOCK_N) * BLOCK_N * SPLIT_K + (col % BLOCK_N);\n"
      << "  int accumulator = C_accum[accum_offset];\n"
      << "  #pragma clang loop unroll(full)\n"
      << "  for (uint k = 1; k < SPLIT_K; ++k)\n"
      << "    accumulator += C_accum[accum_offset + k * BLOCK_N];\n"
      << "  C_out[gid] = half((float)accumulator * (float)A_scale[row] * (float)B_scale[col]);\n"
      << "}\n";

  auto string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t N = bench.N;
  const uint32_t K = bench.K;
  const uint32_t split_k = variant.split_k;
  const uint32_t block_n = variant.block_dimensions[1];
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&N, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&split_k, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&block_n, MTL::DataTypeUInt, NS::UInteger(4));

  auto partial_name = NS::String::string("int8_matmul_splitk", NS::UTF8StringEncoding);
  auto partial_fn = NS::TransferPtr(library->newFunction(partial_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto partial_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  partial_desc->setComputeFunction(partial_fn.get());

  auto dequant_name = NS::String::string("reduce_dequant_accumulator", NS::UTF8StringEncoding);
  auto dequant_fn = NS::TransferPtr(library->newFunction(dequant_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto dequant_desc = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  dequant_desc->setComputeFunction(dequant_fn.get());

  SplitKPipeline pipeline;
  pipeline.partial_pipeline = NS::TransferPtr(device->newComputePipelineState(partial_desc.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  pipeline.dequant_pipeline = NS::TransferPtr(device->newComputePipelineState(dequant_desc.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  pipeline.threadgroup_size = 256;
  return pipeline;
}


QuantizePipeline create_quantize_pipeline(
    MTL::Device* device,
    const BenchmarkCase& bench,
    const VariantConfig& variant)
{
  CCV_NNC_MFA_PRECONDITION(variant.activation_quant_threads % 32 == 0);
  const bool vectorize4 = (bench.K % 4) == 0;
  const uint16_t quant_simdgroups = variant.activation_quant_threads / 32;
  std::ostringstream source;
  source
      << "#include <metal_stdlib>\n"
      << "using namespace metal;\n"
      << "constant uint M [[function_constant(0)]];\n"
      << "constant uint K [[function_constant(1)]];\n"
      << "inline float quantize_reduce_max(float value,\n"
      << "                                 threadgroup float* scratch,\n"
      << "                                 ushort sgid,\n"
      << "                                 ushort lane_id)\n"
      << "{\n"
      << "  value = max(value, simd_shuffle_xor(value, 16));\n"
      << "  value = max(value, simd_shuffle_xor(value, 8));\n"
      << "  value = max(value, simd_shuffle_xor(value, 4));\n"
      << "  value = max(value, simd_shuffle_xor(value, 2));\n"
      << "  value = max(value, simd_shuffle_xor(value, 1));\n"
      << "  if (lane_id == 0)\n"
      << "    scratch[sgid] = value;\n"
      << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  if (sgid == 0) {\n"
      << "    value = lane_id < " << quant_simdgroups << " ? scratch[lane_id] : 0.0f;\n"
      << "    value = max(value, simd_shuffle_xor(value, 16));\n"
      << "    value = max(value, simd_shuffle_xor(value, 8));\n"
      << "    value = max(value, simd_shuffle_xor(value, 4));\n"
      << "    value = max(value, simd_shuffle_xor(value, 2));\n"
      << "    value = max(value, simd_shuffle_xor(value, 1));\n"
      << "    if (lane_id == 0)\n"
      << "      scratch[0] = value;\n"
      << "  }\n"
      << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "  return scratch[0];\n"
      << "}\n";
  if (vectorize4) {
    source
        << "kernel void quantize_activation(\n"
        << "    device const half* src [[buffer(0)]],\n"
        << "    device int8_t* dst [[buffer(1)]],\n"
        << "    device half* scales [[buffer(2)]],\n"
        << "    uint tid [[thread_index_in_threadgroup]],\n"
        << "    ushort sgid [[simdgroup_index_in_threadgroup]],\n"
        << "    ushort lane_id [[thread_index_in_simdgroup]],\n"
        << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
        << "{\n"
        << "  threadgroup float scratch[" << quant_simdgroups << "];\n"
        << "  const uint row = tgid.x;\n"
        << "  if (row >= M)\n"
        << "    return;\n"
        << "  const uint vectors_per_row = K / 4;\n"
        << "  device const half4* src4 = reinterpret_cast<device const half4*>(src);\n"
        << "  device char4* dst4 = reinterpret_cast<device char4*>(dst);\n"
        << "  float local_max = 0.0f;\n"
        << "  const uint base = row * vectors_per_row;\n"
        << "  for (uint i = tid; i < vectors_per_row; i += " << variant.activation_quant_threads << ") {\n"
        << "    const float4 value = float4(src4[base + i]);\n"
        << "    local_max = max(local_max, max(max(fabs(value[0]), fabs(value[1])), max(fabs(value[2]), fabs(value[3]))));\n"
        << "  }\n"
        << "  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);\n"
        << "  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);\n"
        << "  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;\n"
        << "  if (tid == 0)\n"
        << "    scales[row] = half(scale);\n"
        << "  for (uint i = tid; i < vectors_per_row; i += " << variant.activation_quant_threads << ") {\n"
        << "    const int4 rounded = int4(rint(float4(src4[base + i]) * inv_scale));\n"
        << "    dst4[base + i] = char4(clamp(rounded, int4(-127), int4(127)));\n"
        << "  }\n"
        << "}\n";
  } else {
    source
        << "kernel void quantize_activation(\n"
        << "    device const half* src [[buffer(0)]],\n"
        << "    device int8_t* dst [[buffer(1)]],\n"
        << "    device half* scales [[buffer(2)]],\n"
        << "    uint tid [[thread_index_in_threadgroup]],\n"
        << "    ushort sgid [[simdgroup_index_in_threadgroup]],\n"
        << "    ushort lane_id [[thread_index_in_simdgroup]],\n"
        << "    uint3 tgid [[threadgroup_position_in_grid]])\n"
        << "{\n"
        << "  threadgroup float scratch[" << quant_simdgroups << "];\n"
        << "  const uint row = tgid.x;\n"
        << "  if (row >= M)\n"
        << "    return;\n"
        << "  float local_max = 0.0f;\n"
        << "  const uint base = row * K;\n"
        << "  for (uint i = tid; i < K; i += " << variant.activation_quant_threads << ")\n"
        << "    local_max = max(local_max, fabs((float)src[base + i]));\n"
        << "  const float max_abs = quantize_reduce_max(local_max, scratch, sgid, lane_id);\n"
        << "  const float scale = max_abs > 0.0f ? max_abs / 127.0f : (1.0f / 127.0f);\n"
        << "  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;\n"
        << "  if (tid == 0)\n"
        << "    scales[row] = half(scale);\n"
        << "  for (uint i = tid; i < K; i += " << variant.activation_quant_threads << ") {\n"
        << "    const int rounded = (int)rint((float)src[base + i] * inv_scale);\n"
        << "    dst[base + i] = (int8_t)clamp(rounded, -127, 127);\n"
        << "  }\n"
        << "}\n";
  }

  auto string = NS::String::string(source.str().c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t M = bench.M;
  const uint32_t K = bench.K;
  constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&K, MTL::DataTypeUInt, NS::UInteger(1));

  auto function_name = NS::String::string("quantize_activation", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(function_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto descriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  descriptor->setComputeFunction(function.get());

  QuantizePipeline pipeline;
  pipeline.pipeline = NS::TransferPtr(device->newComputePipelineState(descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  pipeline.threadgroup_size = variant.activation_quant_threads;
  return pipeline;
}

void upload_buffer(
    MTL::CommandQueue* command_queue,
    MTL::Buffer* source,
    MTL::Buffer* destination,
    size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

void download_buffer(
    MTL::CommandQueue* command_queue,
    MTL::Buffer* source,
    MTL::Buffer* destination,
    size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

double run_baseline_once(
    MTL::CommandQueue* command_queue,
    const BaselinePipeline& bundle,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_b,
    MTL::Buffer* buffer_c,
    MTL::Buffer* buffer_load_m)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a, 0, 0);
  encoder->setBuffer(buffer_b, 0, 1);
  encoder->setBuffer(buffer_c, 0, 2);
  if (bundle.descriptor.loadM)
    encoder->setBytes(buffer_load_m->contents(), sizeof(uint32_t), 3);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(bundle.descriptor),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get(), bundle.descriptor), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const QuantizePipeline& bundle,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_q,
    MTL::Buffer* buffer_scales)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a, 0, 0);
  encoder->setBuffer(buffer_q, 0, 1);
  encoder->setBuffer(buffer_scales, 0, 2);
  encoder->dispatchThreadgroups(
      MTL::Size(bench.M, 1, 1),
      MTL::Size(bundle.threadgroup_size, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_dynamic_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const DynamicPipeline& bundle,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_c,
    MTL::Buffer* buffer_a_scale,
    MTL::Buffer* buffer_b_scale,
    MTL::Buffer* buffer_load_m)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(bundle.pipeline.get());
  encoder->setBuffer(buffer_a_q, 0, 0);
  encoder->setBuffer(buffer_b_q, 0, 1);
  encoder->setBuffer(buffer_c, 0, 2);
  encoder->setBuffer(buffer_a_scale, 0, 3);
  encoder->setBuffer(buffer_b_scale, 0, 4);
  if (bundle.load_m)
    encoder->setBuffer(buffer_load_m, 0, 5);
  encoder->dispatchThreadgroups(
      bundle.kernel->threadgroupsPerGrid(bench.M, bench.N, 1),
      MTL::Size(bundle.kernel->threadgroupSize(bundle.pipeline.get()), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_raw_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const DynamicPipeline& dynamic,
    const RawPipeline& raw,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_c_i32)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(raw.pipeline.get());
  encoder->setBuffer(buffer_a_q, 0, 0);
  encoder->setBuffer(buffer_b_q, 0, 1);
  encoder->setBuffer(buffer_c_i32, 0, 2);
  encoder->dispatchThreadgroups(
      dynamic.kernel->threadgroupsPerGrid(bench.M, bench.N, 1),
      MTL::Size(dynamic.kernel->threadgroupSize(raw.pipeline.get()), 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_quantize_and_dynamic_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const QuantizePipeline& quantize,
    const DynamicPipeline& dynamic,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_a_scale,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_b_scale,
    MTL::Buffer* buffer_c,
    MTL::Buffer* buffer_load_m)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(quantize.pipeline.get());
    encoder->setBuffer(buffer_a, 0, 0);
    encoder->setBuffer(buffer_a_q, 0, 1);
    encoder->setBuffer(buffer_a_scale, 0, 2);
    encoder->dispatchThreadgroups(
        MTL::Size(bench.M, 1, 1),
        MTL::Size(quantize.threadgroup_size, 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(dynamic.pipeline.get());
    encoder->setBuffer(buffer_a_q, 0, 0);
    encoder->setBuffer(buffer_b_q, 0, 1);
    encoder->setBuffer(buffer_c, 0, 2);
    encoder->setBuffer(buffer_a_scale, 0, 3);
    encoder->setBuffer(buffer_b_scale, 0, 4);
    if (dynamic.load_m)
      encoder->setBuffer(buffer_load_m, 0, 5);
    encoder->dispatchThreadgroups(
        dynamic.kernel->threadgroupsPerGrid(bench.M, bench.N, 1),
        MTL::Size(dynamic.kernel->threadgroupSize(dynamic.pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

double run_splitk_once(
    MTL::CommandQueue* command_queue,
    const BenchmarkCase& bench,
    const DynamicPipeline& dynamic,
    const SplitKPipeline& splitk,
    const VariantConfig& variant,
    MTL::Buffer* buffer_a_q,
    MTL::Buffer* buffer_b_q,
    MTL::Buffer* buffer_c_accum,
    MTL::Buffer* buffer_c,
    MTL::Buffer* buffer_a_scale,
    MTL::Buffer* buffer_b_scale)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(splitk.partial_pipeline.get());
    encoder->setBuffer(buffer_a_q, 0, 0);
    encoder->setBuffer(buffer_b_q, 0, 1);
    encoder->setBuffer(buffer_c_accum, 0, 2);
    const auto base_grid = dynamic.kernel->threadgroupsPerGrid(bench.M, bench.N, 1);
    encoder->dispatchThreadgroups(
        MTL::Size(base_grid.width * variant.split_k, base_grid.height, base_grid.depth),
        MTL::Size(dynamic.kernel->threadgroupSize(splitk.partial_pipeline.get()), 1, 1));
    encoder->endEncoding();
  }
  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(splitk.dequant_pipeline.get());
    encoder->setBuffer(buffer_c_accum, 0, 0);
    encoder->setBuffer(buffer_c, 0, 1);
    encoder->setBuffer(buffer_a_scale, 0, 2);
    encoder->setBuffer(buffer_b_scale, 0, 3);
    const uint32_t total = bench.M * bench.N;
    const uint16_t threadgroup_size = splitk.threadgroup_size;
    encoder->dispatchThreads(
        MTL::Size(total, 1, 1),
        MTL::Size(threadgroup_size, 1, 1));
    encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

template <typename RunOnce>
bool benchmark(const BenchmarkConfig& config, RunOnce&& run_once, Stats* stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double seconds = run_once();
    if (!(seconds > 0) || std::isnan(seconds))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(seconds);
  }
  stats->average_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  stats->median_seconds = samples[samples.size() / 2];
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats->best3_average_seconds =
      std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

std::vector<uint32_t> make_sample_points(uint32_t dimension, const std::vector<uint32_t>& suggestions)
{
  std::vector<uint32_t> sample_points;
  for (const auto suggestion : suggestions)
    if (suggestion < dimension)
      sample_points.push_back(suggestion);
  if (dimension > 0) {
    sample_points.push_back(dimension / 2);
    sample_points.push_back(dimension - 1);
  }
  std::sort(sample_points.begin(), sample_points.end());
  sample_points.erase(std::unique(sample_points.begin(), sample_points.end()), sample_points.end());
  return sample_points;
}

QuantizationValidationStats validate_quantization(
    const RowwiseQuantizedMatrix& reference,
    const int8_t* values,
    const half_float* scales)
{
  QuantizationValidationStats stats;
  for (size_t i = 0; i < reference.values.size(); ++i)
    stats.mismatched_values += (reference.values[i] != values[i]);
  for (size_t i = 0; i < reference.scales.size(); ++i)
    stats.max_abs_scale = std::max(stats.max_abs_scale, std::fabs((double)(half_float)reference.scales[i] - (double)scales[i]));
  stats.passed = (stats.mismatched_values == 0 && stats.max_abs_scale <= 1e-5);
  return stats;
}

float compute_quantized_reference_value(
    const BenchmarkCase& bench,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    uint32_t row,
    uint32_t col)
{
  float accumulator = 0;
  for (uint32_t k = 0; k < bench.K; ++k)
    accumulator +=
        (int32_t)a_quantized.values[a_index(bench, row, k)] *
        (int32_t)b_quantized.values[b_index(bench, col, k)];
  accumulator *= a_quantized.scales[row] * b_quantized.scales[col];
  return accumulator;
}

int32_t compute_quantized_reference_accumulator(
    const BenchmarkCase& bench,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    uint32_t row,
    uint32_t col)
{
  int32_t accumulator = 0;
  for (uint32_t k = 0; k < bench.K; ++k)
    accumulator +=
        (int32_t)a_quantized.values[a_index(bench, row, k)] *
        (int32_t)b_quantized.values[b_index(bench, col, k)];
  return accumulator;
}

float compute_float_reference_value(
    const BenchmarkCase& bench,
    const std::vector<float>& a_values,
    const std::vector<float>& b_values,
    uint32_t row,
    uint32_t col)
{
  float accumulator = 0;
  for (uint32_t k = 0; k < bench.K; ++k)
    accumulator +=
        a_values[a_index(bench, row, k)] *
        b_values[b_index(bench, col, k)];
  return accumulator;
}

ValidationStats validate_output(
    const BenchmarkCase& bench,
    const std::vector<float>& a_values,
    const std::vector<float>& b_values,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    const float* output,
    bool quantized_reference,
    bool half_reference)
{
  ValidationStats stats;
  const uint64_t reference_work = (uint64_t)bench.M * bench.N * bench.K;
  stats.full_reference = reference_work <= (1ull << 26);
  const auto row_points = stats.full_reference ? make_sample_points(bench.M, {}) : make_sample_points(bench.M, { 0, 1, 127, 128, 1023, 1024, 4095, 4096 });
  const auto col_points = stats.full_reference ? make_sample_points(bench.N, {}) : make_sample_points(bench.N, { 0, 1, 63, 64, 1023, 1024, 4095, 4096 });
  for (const auto row : row_points) {
    for (const auto col : col_points) {
      const float reference = quantized_reference ?
          compute_quantized_reference_value(bench, a_quantized, b_quantized, row, col) :
          compute_float_reference_value(bench, a_values, b_values, row, col);
      const float compared_reference = half_reference ? (float)(half_float)reference : reference;
      const float actual = output[c_index(bench, row, col)];
      const double abs_diff = std::fabs(compared_reference - actual);
      const double rel_diff = abs_diff / std::max<double>(std::max(std::fabs(compared_reference), std::fabs(actual)), 1.0);
      stats.max_abs = std::max(stats.max_abs, abs_diff);
      stats.max_rel = std::max(stats.max_rel, rel_diff);
    }
  }
  stats.checked_rows = row_points.size();
  stats.checked_cols = col_points.size();
  if (quantized_reference)
    stats.passed = half_reference ? (stats.max_abs <= 5e-3 || stats.max_rel <= 5e-3)
                                  : (stats.max_abs <= 1e-4 || stats.max_rel <= 1e-4);
  else
    stats.passed = true;
  return stats;
}

void print_stats(const char* label, const BenchmarkCase& bench, const Stats& stats)
{
  const double flops = 2.0 * (double)bench.M * bench.N * bench.K;
  std::cout << label
            << " avg_ms=" << std::fixed << std::setprecision(3) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " best3_avg_ms=" << stats.best3_average_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " avg_gflops=" << flops / stats.average_seconds / 1e9
            << '\n';
}

void print_quantization_validation(const QuantizationValidationStats& stats)
{
  std::cout << "quant-validation"
            << " mismatched_values=" << stats.mismatched_values
            << " max_abs_scale=" << stats.max_abs_scale
            << '\n';
}

void print_validation(const char* label, const ValidationStats& stats)
{
  std::cout << label
            << " mode=" << (stats.full_reference ? "full" : "sampled")
            << " rows=" << stats.checked_rows
            << " cols=" << stats.checked_cols
            << " max_abs=" << stats.max_abs
            << " max_rel=" << stats.max_rel
            << '\n';
}

bool validate_raw_output(
    const BenchmarkCase& bench,
    const RowwiseQuantizedMatrix& a_quantized,
    const RowwiseQuantizedMatrix& b_quantized,
    const int32_t* output,
    double* max_abs_diff)
{
  *max_abs_diff = 0;
  const auto row_points = make_sample_points(bench.M, { 0, 1, 127, 128, 1023, 1024, 4095, 4096 });
  const auto col_points = make_sample_points(bench.N, { 0, 1, 63, 64, 1023, 1024, 4095, 4096 });
  for (const auto row : row_points)
    for (const auto col : col_points) {
      const int32_t reference = compute_quantized_reference_accumulator(bench, a_quantized, b_quantized, row, col);
      const int32_t actual = output[c_index(bench, row, col)];
      *max_abs_diff = std::max(*max_abs_diff, std::fabs((double)reference - (double)actual));
      if (reference != actual)
        return false;
    }
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  BenchmarkCase bench;
  BenchmarkConfig config;
  VariantConfig variant;
  bool baseline_load_m = false;
  if (argc >= 4) {
    bench.M = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    bench.N = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    bench.K = (uint32_t)std::strtoul(argv[3], nullptr, 10);
  }
  if (argc >= 6) {
    config.warmup_iterations = std::atoi(argv[4]);
    config.timed_iterations = std::atoi(argv[5]);
  }
  if (argc >= 9) {
    variant.block_dimensions = simd::ushort3 {
      (uint16_t)std::strtoul(argv[6], nullptr, 10),
      (uint16_t)std::strtoul(argv[7], nullptr, 10),
      (uint16_t)std::strtoul(argv[8], nullptr, 10),
    };
  }
  if (argc >= 10)
    variant.execution_simd_groups = (uint16_t)std::strtoul(argv[9], nullptr, 10);
  if (argc >= 11)
    variant.activation_quant_threads = (uint16_t)std::strtoul(argv[10], nullptr, 10);
  if (argc >= 12)
    variant.group_m = (uint32_t)std::strtoul(argv[11], nullptr, 10);
  if (argc >= 13)
    variant.group_n = (uint32_t)std::strtoul(argv[12], nullptr, 10);
  if (argc >= 14)
    variant.split_k = (uint16_t)std::strtoul(argv[13], nullptr, 10);
  if (argc >= 15)
    variant.load_m = std::strtoul(argv[14], nullptr, 10) != 0;
  if (argc >= 16)
    baseline_load_m = std::strtoul(argv[15], nullptr, 10) != 0;

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    pool->drain();
    return 1;
  }

  const auto a_half_values = make_half_matrix(bench.M, bench.K, 0.03125f, 1);
  const auto b_half_values = make_half_matrix(bench.N, bench.K, 0.015625f, 2);
  const auto a_float_values = half_to_float_vector(a_half_values);
  const auto b_float_values = half_to_float_vector(b_half_values);
  const auto a_quantized_reference = quantize_rowwise(a_float_values, bench.M, bench.K);
  const auto b_quantized_reference = quantize_rowwise(b_float_values, bench.N, bench.K);

  const size_t a_half_bytes = a_half_values.size() * sizeof(half_float);
  const size_t b_half_bytes = b_half_values.size() * sizeof(half_float);
  const size_t a_int8_bytes = a_quantized_reference.values.size() * sizeof(int8_t);
  const size_t b_int8_bytes = b_quantized_reference.values.size() * sizeof(int8_t);
  const size_t a_scale_bytes = a_quantized_reference.scales.size() * sizeof(half_float);
  const size_t b_scale_bytes = b_quantized_reference.scales.size() * sizeof(half_float);
  const size_t c_half_bytes = (size_t)bench.M * bench.N * sizeof(half_float);
  const size_t c_dynamic_half_bytes = (size_t)bench.M * bench.N * sizeof(half_float);
  const size_t c_raw_i32_bytes = (size_t)bench.M * bench.N * sizeof(int32_t);
  const size_t c_splitk_i32_bytes = (size_t)bench.M * bench.N * variant.split_k * sizeof(int32_t);
  const bool benchmark_raw = !variant.load_m && c_raw_i32_bytes <= (size_t(512) << 20);
  const bool benchmark_splitk = !variant.load_m && variant.split_k > 1 && c_splitk_i32_bytes <= (size_t(512) << 20);

  std::vector<half_float> b_half_scales(b_quantized_reference.scales.size());
  std::transform(b_quantized_reference.scales.begin(), b_quantized_reference.scales.end(), b_half_scales.begin(), [](float value) {
    return (half_float)value;
  });
  const uint32_t runtime_m = bench.M;

  auto a_half_stage = NS::TransferPtr(device->newBuffer(a_half_values.data(), a_half_bytes, kSharedResourceOptions));
  auto b_half_stage = NS::TransferPtr(device->newBuffer(b_half_values.data(), b_half_bytes, kSharedResourceOptions));
  auto b_int8_stage = NS::TransferPtr(device->newBuffer(b_quantized_reference.values.data(), b_int8_bytes, kSharedResourceOptions));
  auto b_scale_stage = NS::TransferPtr(device->newBuffer(b_half_scales.data(), b_scale_bytes, kSharedResourceOptions));
  auto c_half_stage = NS::TransferPtr(device->newBuffer(c_half_bytes, kSharedResourceOptions));
  auto a_int8_stage = NS::TransferPtr(device->newBuffer(a_int8_bytes, kSharedResourceOptions));
  auto a_scale_stage = NS::TransferPtr(device->newBuffer(a_scale_bytes, kSharedResourceOptions));
  auto c_dynamic_half_stage = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kSharedResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_raw_i32_stage;
  if (benchmark_raw)
    c_raw_i32_stage = NS::TransferPtr(device->newBuffer(c_raw_i32_bytes, kSharedResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_splitk_i32_stage;
  if (benchmark_splitk)
    c_splitk_i32_stage = NS::TransferPtr(device->newBuffer(c_splitk_i32_bytes, kSharedResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_splitk_half_stage;
  if (benchmark_splitk)
    c_splitk_half_stage = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kSharedResourceOptions));

  auto a_half_buffer = NS::TransferPtr(device->newBuffer(a_half_bytes, kPrivateResourceOptions));
  auto b_half_buffer = NS::TransferPtr(device->newBuffer(b_half_bytes, kPrivateResourceOptions));
  auto a_int8_buffer = NS::TransferPtr(device->newBuffer(a_int8_bytes, kPrivateResourceOptions));
  auto b_int8_buffer = NS::TransferPtr(device->newBuffer(b_int8_bytes, kPrivateResourceOptions));
  auto a_scale_buffer = NS::TransferPtr(device->newBuffer(a_scale_bytes, kPrivateResourceOptions));
  auto b_scale_buffer = NS::TransferPtr(device->newBuffer(b_scale_bytes, kPrivateResourceOptions));
  auto m_buffer = NS::TransferPtr(device->newBuffer(&runtime_m, sizeof(uint32_t), kSharedResourceOptions));
  auto c_half_buffer = NS::TransferPtr(device->newBuffer(c_half_bytes, kPrivateResourceOptions));
  auto c_dynamic_half_buffer = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kPrivateResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_raw_i32_buffer;
  if (benchmark_raw)
    c_raw_i32_buffer = NS::TransferPtr(device->newBuffer(c_raw_i32_bytes, kPrivateResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_splitk_i32_buffer;
  if (benchmark_splitk)
    c_splitk_i32_buffer = NS::TransferPtr(device->newBuffer(c_splitk_i32_bytes, kPrivateResourceOptions));
  NS::SharedPtr<MTL::Buffer> c_splitk_half_buffer;
  if (benchmark_splitk)
    c_splitk_half_buffer = NS::TransferPtr(device->newBuffer(c_dynamic_half_bytes, kPrivateResourceOptions));

  upload_buffer(command_queue.get(), a_half_stage.get(), a_half_buffer.get(), a_half_bytes);
  upload_buffer(command_queue.get(), b_half_stage.get(), b_half_buffer.get(), b_half_bytes);
  upload_buffer(command_queue.get(), b_int8_stage.get(), b_int8_buffer.get(), b_int8_bytes);
  upload_buffer(command_queue.get(), b_scale_stage.get(), b_scale_buffer.get(), b_scale_bytes);

  auto baseline = create_baseline_pipeline(device.get(), bench, baseline_load_m);
  auto quantize = create_quantize_pipeline(device.get(), bench, variant);
  auto dynamic = create_dynamic_pipeline(device.get(), bench, variant);
  RawPipeline raw;
  if (benchmark_raw)
    raw = create_raw_pipeline(device.get(), bench, variant);
  SplitKPipeline splitk;
  if (benchmark_splitk)
    splitk = create_splitk_pipeline(device.get(), bench, variant);

  std::cout << "shape"
            << " M=" << bench.M
            << " N=" << bench.N
            << " K=" << bench.K
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << " blockM=" << variant.block_dimensions[0]
            << " blockN=" << variant.block_dimensions[1]
            << " blockK=" << variant.block_dimensions[2]
            << " simdgroups=" << variant.execution_simd_groups
            << " quantThreads=" << variant.activation_quant_threads
            << " groupM=" << groupM(bench, variant)
            << " groupN=" << groupN(bench, variant)
            << " loadM=" << (variant.load_m ? 1 : 0)
            << " baselineLoadM=" << (baseline_load_m ? 1 : 0)
            << " rawInt32=" << (benchmark_raw ? 1 : 0)
            << " splitK=" << variant.split_k
            << '\n';

  const double quantize_validation_seconds =
      run_quantize_once(command_queue.get(), bench, quantize, a_half_buffer.get(), a_int8_buffer.get(), a_scale_buffer.get());
  if (!(quantize_validation_seconds > 0)) {
    std::cerr << "activation quantization dispatch failed\n";
    pool->drain();
    return 1;
  }
  download_buffer(command_queue.get(), a_int8_buffer.get(), a_int8_stage.get(), a_int8_bytes);
  download_buffer(command_queue.get(), a_scale_buffer.get(), a_scale_stage.get(), a_scale_bytes);
  const auto quantization_validation = validate_quantization(
      a_quantized_reference,
      (const int8_t*)a_int8_stage->contents(),
      (const half_float*)a_scale_stage->contents());
  print_quantization_validation(quantization_validation);
  if (!quantization_validation.passed) {
    std::cerr << "activation quantization validation failed\n";
    pool->drain();
    return 1;
  }

  const double dynamic_validation_seconds =
      run_dynamic_once(
          command_queue.get(),
          bench,
          dynamic,
          a_int8_buffer.get(),
          b_int8_buffer.get(),
          c_dynamic_half_buffer.get(),
          a_scale_buffer.get(),
          b_scale_buffer.get(),
          m_buffer.get());
  if (!(dynamic_validation_seconds > 0)) {
    std::cerr << "dynamic int8 matmul dispatch failed\n";
    pool->drain();
    return 1;
  }
  download_buffer(command_queue.get(), c_dynamic_half_buffer.get(), c_dynamic_half_stage.get(), c_dynamic_half_bytes);
  std::vector<float> c_output((size_t)bench.M * bench.N);
  {
    const auto* c_half_output = (const half_float*)c_dynamic_half_stage->contents();
    std::transform(c_half_output, c_half_output + c_output.size(), c_output.begin(), [](half_float value) {
      return (float)value;
    });
  }
  const auto exact_validation = validate_output(
      bench,
      a_float_values,
      b_float_values,
      a_quantized_reference,
      b_quantized_reference,
      c_output.data(),
      true,
      true);
  print_validation("exact-validation", exact_validation);
  if (!exact_validation.passed) {
    std::cerr << "dynamic int8 matmul exact validation failed\n";
    pool->drain();
    return 1;
  }
  const auto accuracy_validation = validate_output(
      bench,
      a_float_values,
      b_float_values,
      a_quantized_reference,
      b_quantized_reference,
      c_output.data(),
      false,
      false);
  print_validation("float-reference", accuracy_validation);

  ValidationStats splitk_validation;
  if (benchmark_splitk) {
    const double splitk_validation_seconds =
        run_splitk_once(
            command_queue.get(),
            bench,
            dynamic,
            splitk,
            variant,
            a_int8_buffer.get(),
            b_int8_buffer.get(),
            c_splitk_i32_buffer.get(),
            c_splitk_half_buffer.get(),
            a_scale_buffer.get(),
            b_scale_buffer.get());
    if (!(splitk_validation_seconds > 0)) {
      std::cerr << "splitK int8 matmul dispatch failed\n";
      pool->drain();
      return 1;
    }
    download_buffer(command_queue.get(), c_splitk_half_buffer.get(), c_splitk_half_stage.get(), c_dynamic_half_bytes);
    std::vector<float> c_splitk_output((size_t)bench.M * bench.N);
    {
      const auto* c_half_output = (const half_float*)c_splitk_half_stage->contents();
      std::transform(c_half_output, c_half_output + c_splitk_output.size(), c_splitk_output.begin(), [](half_float value) {
        return (float)value;
      });
    }
    splitk_validation = validate_output(
        bench,
        a_float_values,
        b_float_values,
        a_quantized_reference,
        b_quantized_reference,
        c_splitk_output.data(),
        true,
        true);
    print_validation("splitk-validation", splitk_validation);
    if (!splitk_validation.passed) {
      std::cerr << "splitK int8 matmul exact validation failed\n";
      pool->drain();
      return 1;
    }
  }

  if (benchmark_raw) {
    const double raw_validation_seconds =
        run_raw_once(
            command_queue.get(),
            bench,
            dynamic,
            raw,
            a_int8_buffer.get(),
            b_int8_buffer.get(),
            c_raw_i32_buffer.get());
    if (!(raw_validation_seconds > 0)) {
      std::cerr << "raw int32 matmul dispatch failed\n";
      pool->drain();
      return 1;
    }
    download_buffer(command_queue.get(), c_raw_i32_buffer.get(), c_raw_i32_stage.get(), c_raw_i32_bytes);
    double max_abs_diff = 0;
    const bool raw_valid = validate_raw_output(
        bench,
        a_quantized_reference,
        b_quantized_reference,
        (const int32_t*)c_raw_i32_stage->contents(),
        &max_abs_diff);
    std::cout << "raw-validation max_abs_diff=" << max_abs_diff << '\n';
    if (!raw_valid) {
      std::cerr << "raw int32 matmul validation failed\n";
      pool->drain();
      return 1;
    }
  }

  Stats baseline_stats;
  if (!benchmark(config, [&]() {
        return run_baseline_once(command_queue.get(), baseline, a_half_buffer.get(), b_half_buffer.get(), c_half_buffer.get(), m_buffer.get());
      }, &baseline_stats)) {
    std::cerr << "baseline benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats quantize_stats;
  if (!benchmark(config, [&]() {
        return run_quantize_once(command_queue.get(), bench, quantize, a_half_buffer.get(), a_int8_buffer.get(), a_scale_buffer.get());
      }, &quantize_stats)) {
    std::cerr << "quantize benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats dynamic_stats;
  if (!benchmark(config, [&]() {
        return run_dynamic_once(
            command_queue.get(),
            bench,
            dynamic,
            a_int8_buffer.get(),
            b_int8_buffer.get(),
            c_dynamic_half_buffer.get(),
            a_scale_buffer.get(),
            b_scale_buffer.get(),
            m_buffer.get());
      }, &dynamic_stats)) {
    std::cerr << "dynamic int8 benchmark failed\n";
    pool->drain();
    return 1;
  }

  Stats raw_stats;
  if (benchmark_raw) {
    if (!benchmark(config, [&]() {
          return run_raw_once(
              command_queue.get(),
              bench,
              dynamic,
              raw,
              a_int8_buffer.get(),
              b_int8_buffer.get(),
              c_raw_i32_buffer.get());
        }, &raw_stats)) {
      std::cerr << "raw int32 benchmark failed\n";
      pool->drain();
      return 1;
    }
  }

  Stats splitk_stats;
  if (benchmark_splitk) {
    if (!benchmark(config, [&]() {
          return run_splitk_once(
              command_queue.get(),
              bench,
              dynamic,
              splitk,
              variant,
              a_int8_buffer.get(),
              b_int8_buffer.get(),
              c_splitk_i32_buffer.get(),
              c_splitk_half_buffer.get(),
              a_scale_buffer.get(),
              b_scale_buffer.get());
        }, &splitk_stats)) {
      std::cerr << "splitK benchmark failed\n";
      pool->drain();
      return 1;
    }
  }

  Stats combined_stats;
  if (!benchmark(config, [&]() {
        return run_quantize_and_dynamic_once(
            command_queue.get(),
            bench,
            quantize,
            dynamic,
            a_half_buffer.get(),
            a_int8_buffer.get(),
            a_scale_buffer.get(),
            b_int8_buffer.get(),
            b_scale_buffer.get(),
            c_dynamic_half_buffer.get(),
            m_buffer.get());
      }, &combined_stats)) {
    std::cerr << "combined benchmark failed\n";
    pool->drain();
    return 1;
  }

  print_stats("baseline-fp16", bench, baseline_stats);
  print_stats("quantize-activation", bench, quantize_stats);
  if (benchmark_raw)
    print_stats("int8-int8-raw-int32", bench, raw_stats);
  else
    std::cout << "int8-int8-raw-int32 skipped=1 reason=" << (variant.load_m ? "load_m" : "buffer_too_large") << '\n';
  print_stats("int8-int8-inline-dequant", bench, dynamic_stats);
  if (benchmark_splitk)
    print_stats("int8-int8-splitk-inline-dequant", bench, splitk_stats);
  else if (variant.split_k > 1)
    std::cout << "int8-int8-splitk-inline-dequant skipped=1 reason=" << (variant.load_m ? "load_m" : "buffer_too_large") << '\n';
  print_stats("quantize-plus-int8", bench, combined_stats);
  std::cout << "speedup";
  if (benchmark_raw)
    std::cout << " raw_kernel_avg=" << baseline_stats.average_seconds / raw_stats.average_seconds
              << " raw_kernel_median=" << baseline_stats.median_seconds / raw_stats.median_seconds
              << " raw_kernel_best3=" << baseline_stats.best3_average_seconds / raw_stats.best3_average_seconds;
  std::cout << " kernel_avg=" << baseline_stats.average_seconds / dynamic_stats.average_seconds
            << " kernel_median=" << baseline_stats.median_seconds / dynamic_stats.median_seconds
            << " kernel_best3=" << baseline_stats.best3_average_seconds / dynamic_stats.best3_average_seconds;
  if (benchmark_splitk)
    std::cout << " splitk_kernel_avg=" << baseline_stats.average_seconds / splitk_stats.average_seconds
              << " splitk_kernel_median=" << baseline_stats.median_seconds / splitk_stats.median_seconds
              << " splitk_kernel_best3=" << baseline_stats.best3_average_seconds / splitk_stats.best3_average_seconds;
  std::cout << " end_to_end_avg=" << baseline_stats.average_seconds / combined_stats.average_seconds
            << " end_to_end_median=" << baseline_stats.median_seconds / combined_stats.median_seconds
            << " end_to_end_best3=" << baseline_stats.best3_average_seconds / combined_stats.best3_average_seconds
            << '\n';
  std::cout.flush();
  std::_Exit(0);
}
