#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <QuartzCore/QuartzCore.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/GEMMHeaders.hpp"

namespace {

using half_float = _Float16;

struct Conv3DShape {
  uint32_t batch;
  uint32_t input_d;
  uint32_t input_h;
  uint32_t input_w;
  uint32_t input_c;
  uint32_t output_c;
  uint32_t kernel_d;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t pad_d;
  uint32_t pad_h;
  uint32_t pad_w;
};

struct KernelParams {
  uint32_t batch;
  uint32_t input_d;
  uint32_t input_h;
  uint32_t input_w;
  uint32_t input_c;
  uint32_t output_d;
  uint32_t output_h;
  uint32_t output_w;
  uint32_t output_c;
  uint32_t kernel_d;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t pad_d;
  uint32_t pad_h;
  uint32_t pad_w;
  uint32_t gemm_m;
  uint32_t gemm_n;
  uint32_t gemm_k;
};

size_t input_index(
    const Conv3DShape& shape,
    uint32_t n,
    uint32_t d,
    uint32_t h,
    uint32_t w,
    uint32_t c)
{
  return ((((size_t)n * shape.input_d + d) * shape.input_h + h) * shape.input_w + w) *
      shape.input_c + c;
}

size_t weight_index_oidhw(
    const Conv3DShape& shape,
    uint32_t o,
    uint32_t i,
    uint32_t kd,
    uint32_t kh,
    uint32_t kw)
{
  return ((((size_t)o * shape.input_c + i) * shape.kernel_d + kd) * shape.kernel_h + kh) *
      shape.kernel_w + kw;
}

size_t output_index(
    const Conv3DShape& shape,
    uint32_t output_d,
    uint32_t output_h,
    uint32_t output_w,
    uint32_t n,
    uint32_t d,
    uint32_t h,
    uint32_t w,
    uint32_t o)
{
  return ((((size_t)n * output_d + d) * output_h + h) * output_w + w) * shape.output_c + o;
}

template <typename T>
std::vector<T> make_data(size_t size, float scale)
{
  std::vector<T> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] = static_cast<T>(
        static_cast<float>(static_cast<int>(i % 17) - 8) * scale);
  return values;
}

std::vector<half_float> permute_weights_to_ok(
    const Conv3DShape& shape,
    const std::vector<half_float>& weights)
{
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;
  std::vector<half_float> permuted((size_t)shape.output_c * gemm_k);
  for (uint32_t oc = 0; oc < shape.output_c; ++oc)
  {
    uint32_t column = 0;
    for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
      for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
        for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
          for (uint32_t ic = 0; ic < shape.input_c; ++ic, ++column)
            permuted[(size_t)oc * gemm_k + column] =
                weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)];
  }
  return permuted;
}

std::vector<float> cpu_conv3d_same(
    const Conv3DShape& shape,
    const std::vector<half_float>& input,
    const std::vector<half_float>& weights)
{
  std::vector<float> output(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.output_c, 0.0f);
  for (uint32_t n = 0; n < shape.batch; ++n)
    for (uint32_t od = 0; od < shape.input_d; ++od)
      for (uint32_t oh = 0; oh < shape.input_h; ++oh)
        for (uint32_t ow = 0; ow < shape.input_w; ++ow)
          for (uint32_t oc = 0; oc < shape.output_c; ++oc)
          {
            float sum = 0.0f;
            for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
              for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
                for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
                {
                  const int id = static_cast<int>(od) + static_cast<int>(kd) -
                      static_cast<int>(shape.pad_d);
                  const int ih = static_cast<int>(oh) + static_cast<int>(kh) -
                      static_cast<int>(shape.pad_h);
                  const int iw = static_cast<int>(ow) + static_cast<int>(kw) -
                      static_cast<int>(shape.pad_w);
                  if (id < 0 || id >= static_cast<int>(shape.input_d) || ih < 0 ||
                      ih >= static_cast<int>(shape.input_h) || iw < 0 ||
                      iw >= static_cast<int>(shape.input_w))
                    continue;
                  for (uint32_t ic = 0; ic < shape.input_c; ++ic)
                    sum += static_cast<float>(
                               input[input_index(shape, n, id, ih, iw, ic)]) *
                        static_cast<float>(
                            weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)]);
                }
            output[output_index(
                shape, shape.input_d, shape.input_h, shape.input_w, n, od, oh, ow, oc)] = sum;
          }
  return output;
}

std::string create_shader_source()
{
  std::string source = createMetalSimdgroupMatrixStorage(false);
  source += R"(
using namespace metal;

struct KernelParams {
  uint batch;
  uint input_d;
  uint input_h;
  uint input_w;
  uint input_c;
  uint output_d;
  uint output_h;
  uint output_w;
  uint output_c;
  uint kernel_d;
  uint kernel_h;
  uint kernel_w;
  uint pad_d;
  uint pad_h;
  uint pad_w;
  uint gemm_m;
  uint gemm_n;
  uint gemm_k;
};

constant bool B_trans = true;
constant ushort M_group = 32;
constant ushort N_group = 32;
constant ushort REGISTER_M = 32;
constant ushort REGISTER_N = 32;

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

template <typename T>
METAL_FUNC const device T* apply_offset_const(
  const device T *src,
  uint elements_per_row,
  uint2 matrix_origin,
  bool transpose_matrix = false
) {
  if (transpose_matrix) {
    return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
  } else {
    return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
  }
}

METAL_FUNC void multiply_accumulate_implicit(
  const device half *input,
  const device half *weights,
  constant KernelParams& params,
  uint N_offset,
  uint batch,
  uint od,
  uint oh,
  uint ow_base,
  bool interior_tile,
  ushort2 morton_offset,
  ushort2 offset_in_group,
  thread simdgroup_matrix_storage<half> *A_sram,
  thread simdgroup_matrix_storage<half> *B_sram,
  thread simdgroup_matrix_storage<half> *C_sram
) {
#pragma clang loop unroll(full)
  for (uint kd = 0; kd < params.kernel_d; ++kd) {
    const int input_d = int(od) + int(kd) - int(params.pad_d);
    const bool valid_d = input_d >= 0 && input_d < int(params.input_d);
#pragma clang loop unroll(full)
    for (uint kh = 0; kh < params.kernel_h; ++kh) {
      const int input_h = int(oh) + int(kh) - int(params.pad_h);
      uint row_plane_base = 0;
      bool valid_dh = false;
      if (valid_d && input_h >= 0 && input_h < int(params.input_h)) {
        valid_dh = true;
        row_plane_base =
            (((batch * params.input_d + uint(input_d)) * params.input_h +
              uint(input_h)) *
             params.input_w) *
            params.input_c;
      }
#pragma clang loop unroll(full)
      for (uint kw = 0; kw < params.kernel_w; ++kw) {
        const int input_w_base = int(ow_base) + int(kw) - int(params.pad_w);
        const uint k_spatial_base =
            ((kd * params.kernel_h + kh) * params.kernel_w + kw) *
            params.input_c;
#pragma clang loop unroll(enable)
        for (uint c_base = 0; c_base < params.input_c; c_base += 8) {
          const uint lane_channel = c_base + morton_offset.x;

#pragma clang loop unroll(full)
          for (ushort m = 0; m < REGISTER_M; m += 8) {
            const ushort row = m + morton_offset.y;
            half2 values(half(0), half(0));
            if (interior_tile) {
              const uint address =
                  row_plane_base +
                  (uint(input_w_base) + uint(row)) * params.input_c +
                  lane_channel;
              values = *((const device half2*)(input + address));
            } else if (valid_dh) {
              const int input_w = input_w_base + int(row);
              if (input_w >= 0 && input_w < int(params.input_w)) {
                const uint address =
                    row_plane_base + uint(input_w) * params.input_c + lane_channel;
                values = *((const device half2*)(input + address));
              }
            }
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            *A = simdgroup_matrix_storage<half>(values);
          }

          const uint k_base = k_spatial_base + c_base;
          uint2 B_offset(N_offset, k_base);
          B_offset += uint2(offset_in_group.x, morton_offset.y);
          auto B_src = apply_offset_const(
              weights, params.gemm_k, B_offset, B_trans);

#pragma clang loop unroll(full)
          for (ushort n = 0; n < REGISTER_N; n += 8) {
            auto B = get_sram(B_sram, REGISTER_N, ushort2(n, 0));
            B->load(B_src, params.gemm_k, ushort2(n, 0), B_trans);
          }

#pragma clang loop unroll(full)
          for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
            for (ushort n = 0; n < REGISTER_N; n += 8) {
              auto A = get_sram(A_sram, 8, ushort2(0, m));
              auto B = get_sram(B_sram, REGISTER_N, ushort2(n, 0));
              auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
              C->multiply(*A, *B);
            }
          }
        }
      }
    }
  }
}

kernel void implicit_conv3d_bench(
    const device half* input [[buffer(0)]],
    const device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    const constant KernelParams& params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint M_offset = gid.y * M_group;
  const uint N_offset = gid.x * N_group;
  if (M_offset >= params.gemm_m || N_offset >= params.gemm_n)
    return;

  const ushort2 morton_offset = morton_order(lane_id);
  const ushort2 offset_in_group(morton_offset.x, morton_offset.y);

  const uint output_hw = params.output_h * params.output_w;
  const uint output_dhw = params.output_d * output_hw;
  const uint batch = M_offset / output_dhw;
  const uint odhw = M_offset % output_dhw;
  const uint od = odhw / output_hw;
  const uint hw = odhw % output_hw;
  const uint oh = hw / params.output_w;
  const uint ow_base = hw % params.output_w;

  const bool interior_tile =
      (od >= params.pad_d) &&
      (od + (params.kernel_d - params.pad_d - 1) < params.input_d) &&
      (oh >= params.pad_h) &&
      (oh + (params.kernel_h - params.pad_h - 1) < params.input_h) &&
      (ow_base >= params.pad_w) &&
      (ow_base + M_group + (params.kernel_w - params.pad_w - 1) <= params.input_w);

  thread simdgroup_matrix_storage<half> C_sram[(REGISTER_M / 8) * (REGISTER_N / 8)];
#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
      *C = simdgroup_matrix_storage<half>(0);
    }
  }

  thread simdgroup_matrix_storage<half> A_sram[(REGISTER_M / 8)];
  thread simdgroup_matrix_storage<half> B_sram[(REGISTER_N / 8)];
  multiply_accumulate_implicit(
      input,
      weights,
      params,
      N_offset,
      batch,
      od,
      oh,
      ow_base,
      interior_tile,
      morton_offset,
      offset_in_group,
      A_sram,
      B_sram,
      C_sram);

  uint2 C_offset(N_offset + offset_in_group.x, M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<half>::apply_offset(
      output, params.gemm_n, C_offset);
#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
      C->store(C_dst, params.gemm_n, ushort2(n, m));
    }
  }
}
)";
  return source;
}

void run_validation(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    MTL::ComputePipelineState* pipeline)
{
  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 8,
      .input_w = 32,
      .input_c = 8,
      .output_c = 32,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
      .pad_d = 1,
      .pad_h = 1,
      .pad_w = 1,
  };
  const uint32_t gemm_m = shape.batch * shape.input_d * shape.input_h * shape.input_w;
  const uint32_t gemm_n = shape.output_c;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;

  const auto input = make_data<half_float>(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.03125f);
  const auto weights_oidhw = make_data<half_float>(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h *
          shape.kernel_w,
      -0.015625f);
  const auto reference = cpu_conv3d_same(shape, input, weights_oidhw);
  const auto weights_ok = permute_weights_to_ok(shape, weights_oidhw);
  std::vector<half_float> output((size_t)gemm_m * gemm_n, half_float(0));

  KernelParams params{
      .batch = shape.batch,
      .input_d = shape.input_d,
      .input_h = shape.input_h,
      .input_w = shape.input_w,
      .input_c = shape.input_c,
      .output_d = shape.input_d,
      .output_h = shape.input_h,
      .output_w = shape.input_w,
      .output_c = shape.output_c,
      .kernel_d = shape.kernel_d,
      .kernel_h = shape.kernel_h,
      .kernel_w = shape.kernel_w,
      .pad_d = shape.pad_d,
      .pad_h = shape.pad_h,
      .pad_w = shape.pad_w,
      .gemm_m = gemm_m,
      .gemm_n = gemm_n,
      .gemm_k = gemm_k,
  };

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weight_buffer = NS::TransferPtr(device->newBuffer(
      weights_ok.data(),
      weights_ok.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(input_buffer.get(), 0, 0);
  encoder->setBuffer(weight_buffer.get(), 0, 1);
  encoder->setBuffer(output_buffer.get(), 0, 2);
  encoder->setBytes(&params, sizeof(params), 3);
  encoder->dispatchThreadgroups(
      MTL::Size(gemm_n / 32, gemm_m / 32, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const auto* raw = static_cast<const half_float*>(output_buffer->contents());
  float max_abs_error = 0.0f;
  size_t max_error_index = 0;
  for (size_t i = 0; i < output.size(); ++i)
  {
    const float abs_error =
        std::fabs(static_cast<float>(raw[i]) - reference[i]);
    if (abs_error > max_abs_error)
    {
      max_abs_error = abs_error;
      max_error_index = i;
    }
  }
  std::fprintf(
      stderr,
      "implicit conv3d bench validation: M=%u N=%u K=%u max_abs_error=%.6f "
      "max_error_index=%zu\n",
      gemm_m,
      gemm_n,
      gemm_k,
      max_abs_error,
      max_error_index);
  if (!(max_abs_error <= 0.05f))
    std::_Exit(2);
}

} // namespace

int main()
{
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    std::fprintf(stderr, "Metal device unavailable.\n");
    std::_Exit(1);
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue)
  {
    std::fprintf(stderr, "Metal command queue unavailable.\n");
    std::_Exit(1);
  }

  auto source = create_shader_source();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto function = NS::TransferPtr(
      library->newFunction(NS::String::string("implicit_conv3d_bench", NS::UTF8StringEncoding)));
  if (!function)
  {
    std::fprintf(stderr, "Failed to locate Metal function implicit_conv3d_bench.\n");
    std::_Exit(1);
  }
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  run_validation(device.get(), command_queue.get(), pipeline.get());

  const Conv3DShape shape{
      .batch = 1,
      .input_d = 16,
      .input_h = 128,
      .input_w = 128,
      .input_c = 256,
      .output_c = 256,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
      .pad_d = 1,
      .pad_h = 1,
      .pad_w = 1,
  };
  const uint32_t gemm_m = shape.batch * shape.input_d * shape.input_h * shape.input_w;
  const uint32_t gemm_n = shape.output_c;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;

  const auto input = make_data<half_float>(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.03125f);
  const auto weights_oidhw = make_data<half_float>(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h *
          shape.kernel_w,
      -0.015625f);
  const auto weights_ok = permute_weights_to_ok(shape, weights_oidhw);
  std::vector<half_float> output((size_t)gemm_m * gemm_n, half_float(0));

  KernelParams params{
      .batch = shape.batch,
      .input_d = shape.input_d,
      .input_h = shape.input_h,
      .input_w = shape.input_w,
      .input_c = shape.input_c,
      .output_d = shape.input_d,
      .output_h = shape.input_h,
      .output_w = shape.input_w,
      .output_c = shape.output_c,
      .kernel_d = shape.kernel_d,
      .kernel_h = shape.kernel_h,
      .kernel_w = shape.kernel_w,
      .pad_d = shape.pad_d,
      .pad_h = shape.pad_h,
      .pad_w = shape.pad_w,
      .gemm_m = gemm_m,
      .gemm_n = gemm_n,
      .gemm_k = gemm_k,
  };

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weight_buffer = NS::TransferPtr(device->newBuffer(
      weights_ok.data(),
      weights_ok.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(half_float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  const uint32_t grid_x = gemm_n / 32;
  const uint32_t grid_y = gemm_m / 32;
  const int warmup_iterations = 2;
  const int timed_iterations = 10;
  double total_seconds = 0.0;

  for (int iteration = 0; iteration < warmup_iterations + timed_iterations; ++iteration)
  {
    auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(input_buffer.get(), 0, 0);
    encoder->setBuffer(weight_buffer.get(), 0, 1);
    encoder->setBuffer(output_buffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreadgroups(
        MTL::Size(grid_x, grid_y, 1), MTL::Size(32, 1, 1));
    encoder->endEncoding();

    const double start_time = CACurrentMediaTime();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    const double end_time = CACurrentMediaTime();

    if (iteration >= warmup_iterations)
      total_seconds += end_time - start_time;
  }

  const double average_seconds = total_seconds / timed_iterations;
  const double flops =
      2.0 * static_cast<double>(gemm_m) * static_cast<double>(gemm_n) *
      static_cast<double>(gemm_k);
  const double tflops = flops / average_seconds / 1e12;
  const auto* raw_output = static_cast<const half_float*>(output_buffer->contents());
  double checksum = 0.0;
  const size_t checksum_count = std::min<size_t>(1024, output.size());
  for (size_t i = 0; i < checksum_count; ++i)
    checksum += static_cast<double>(raw_output[i]);

  std::fprintf(
      stderr,
      "implicit conv3d bench: device=%s input=1x16x128x128x256 output=1x16x128x128x256 "
      "kernel=3x3x3 padding=same M=%u N=%u K=%u grid=%u x %u warmup=%d timed=%d "
      "avg_ms=%.3f tflops=%.3f checksum1024=%.6f\n",
      device->name()->utf8String(),
      gemm_m,
      gemm_n,
      gemm_k,
      grid_x,
      grid_y,
      warmup_iterations,
      timed_iterations,
      average_seconds * 1e3,
      tflops,
      checksum);
  std::fflush(stderr);
  std::fflush(stdout);
  std::_Exit(0);
}
