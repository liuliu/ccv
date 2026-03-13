#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/GEMMHeaders.hpp"

namespace {

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

std::vector<float> make_data(size_t size, float bias)
{
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] = bias + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f;
  return values;
}

std::vector<float> cpu_conv3d(
    const Conv3DShape& shape,
    uint32_t output_d,
    uint32_t output_h,
    uint32_t output_w,
    const std::vector<float>& input,
    const std::vector<float>& weights)
{
  std::vector<float> output(
      (size_t)shape.batch * output_d * output_h * output_w * shape.output_c, 0.0f);
  for (uint32_t n = 0; n < shape.batch; ++n)
    for (uint32_t od = 0; od < output_d; ++od)
      for (uint32_t oh = 0; oh < output_h; ++oh)
        for (uint32_t ow = 0; ow < output_w; ++ow)
          for (uint32_t oc = 0; oc < shape.output_c; ++oc)
          {
            float sum = 0.0f;
            for (uint32_t kd = 0; kd < shape.kernel_d; ++kd)
              for (uint32_t kh = 0; kh < shape.kernel_h; ++kh)
                for (uint32_t kw = 0; kw < shape.kernel_w; ++kw)
                  for (uint32_t ic = 0; ic < shape.input_c; ++ic)
                    sum += input[input_index(shape, n, od + kd, oh + kh, ow + kw, ic)] *
                        weights[weight_index_oidhw(shape, oc, ic, kd, kh, kw)];
            output[output_index(shape, output_d, output_h, output_w, n, od, oh, ow, oc)] = sum;
          }
  return output;
}

std::vector<float> permute_weights_to_ok(
    const Conv3DShape& shape,
    const std::vector<float>& weights)
{
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;
  std::vector<float> permuted((size_t)shape.output_c * gemm_k);
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
  uint gemm_m;
  uint gemm_n;
  uint gemm_k;
};

constant bool A_trans = false;
constant bool B_trans = true;
constant ushort M_group = 32;
constant ushort N_group = 32;
constant ushort K_group = 8;
constant ushort REGISTER_M = 32;
constant ushort REGISTER_N = 32;
constant ushort A_leading_dimension = 8;
constant ushort B_leading_dimension = 8;

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

METAL_FUNC void multiply_accumulate(
  const threadgroup float *A_src,
  const threadgroup float *B_src,
  thread simdgroup_matrix_storage<float> *A_sram,
  thread simdgroup_matrix_storage<float> *B_sram,
  thread simdgroup_matrix_storage<float> *C_sram,
  ushort k
) {
#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
    A->load(A_src, A_leading_dimension, ushort2(k, m), A_trans);
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < REGISTER_N; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, REGISTER_N, origin);
    B->load(B_src, B_leading_dimension, ushort2(n, k), B_trans);
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

kernel void implicit_conv3d(
    const device float* input [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    const constant KernelParams& params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint M_offset = gid.y * M_group;
  const uint N_offset = gid.x * N_group;
  if (M_offset >= params.gemm_m || N_offset >= params.gemm_n)
    return;

  const ushort2 sid(0, 0);
  const ushort2 morton_offset = morton_order(lane_id);
  const ushort2 offset_in_group(
      sid.x * REGISTER_N + morton_offset.x,
      sid.y * REGISTER_M + morton_offset.y);

  threadgroup float A_block[M_group * K_group];
  threadgroup float B_block[N_group * K_group];
  thread simdgroup_matrix_storage<float> C_sram[(REGISTER_M / 8) * (REGISTER_N / 8)];

#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
      *C = simdgroup_matrix_storage<float>(0);
    }
  }

  for (uint k_base = 0; k_base < params.gemm_k; k_base += K_group) {
    for (uint e = lane_id; e < M_group * K_group; e += 32) {
      const uint tile_row = e / K_group;
      const uint tile_col = e % K_group;
      const uint global_m = M_offset + tile_row;
      const uint global_k = k_base + tile_col;
      float value = 0.0f;
      if (global_m < params.gemm_m && global_k < params.gemm_k) {
        const uint odhw = global_m;
        const uint od = odhw / (params.output_h * params.output_w);
        const uint hw = odhw % (params.output_h * params.output_w);
        const uint oh = hw / params.output_w;
        const uint ow = hw % params.output_w;

        uint linear = global_k;
        const uint ic = linear % params.input_c;
        linear /= params.input_c;
        const uint kw = linear % params.kernel_w;
        linear /= params.kernel_w;
        const uint kh = linear % params.kernel_h;
        const uint kd = linear / params.kernel_h;

        const uint input_index =
            ((((kd + od) * params.input_h + (kh + oh)) * params.input_w + (kw + ow)) * params.input_c) + ic;
        value = input[input_index];
      }
      A_block[tile_row * A_leading_dimension + tile_col] = value;
    }

    for (uint e = lane_id; e < N_group * K_group; e += 32) {
      const uint tile_row = e / K_group;
      const uint tile_col = e % K_group;
      const uint global_n = N_offset + tile_row;
      const uint global_k = k_base + tile_col;
      float value = 0.0f;
      if (global_n < params.gemm_n && global_k < params.gemm_k)
        value = weights[global_n * params.gemm_k + global_k];
      B_block[tile_row * B_leading_dimension + tile_col] = value;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto A_block_src = (threadgroup float*)A_block;
    auto B_block_src = (threadgroup float*)B_block;
    A_block_src = simdgroup_matrix_storage<float>::apply_offset(
        A_block_src, A_leading_dimension, ushort2(morton_offset.x, offset_in_group.y), A_trans);
    B_block_src = simdgroup_matrix_storage<float>::apply_offset(
        B_block_src, B_leading_dimension, ushort2(offset_in_group.x, morton_offset.y), B_trans);

    thread simdgroup_matrix_storage<float> A_sram[(REGISTER_M / 8)];
    thread simdgroup_matrix_storage<float> B_sram[(REGISTER_N / 8)];
    multiply_accumulate(A_block_src, B_block_src, A_sram, B_sram, C_sram, 0);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint2 C_offset(N_offset + offset_in_group.x, M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<float>::apply_offset(
      output, params.gemm_n, C_offset);
#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, REGISTER_N, origin);
      C->store(C_dst, params.gemm_n, origin);
    }
  }
}
)";
  return source;
}

} // namespace

int main()
{
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    fprintf(stderr, "Metal device unavailable.\n");
    std::_Exit(1);
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue)
  {
    fprintf(stderr, "Metal command queue unavailable.\n");
    std::_Exit(1);
  }

  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 6,
      .input_w = 10,
      .input_c = 4,
      .output_c = 64,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
  };
  const uint32_t output_d = shape.input_d - shape.kernel_d + 1;
  const uint32_t output_h = shape.input_h - shape.kernel_h + 1;
  const uint32_t output_w = shape.input_w - shape.kernel_w + 1;
  const uint32_t gemm_m = shape.batch * output_d * output_h * output_w;
  const uint32_t gemm_n = shape.output_c;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;

  const auto input = make_data(
      (size_t)shape.batch * shape.input_d * shape.input_h * shape.input_w * shape.input_c,
      0.25f);
  const auto weights = make_data(
      (size_t)shape.output_c * shape.input_c * shape.kernel_d * shape.kernel_h * shape.kernel_w,
      -0.125f);
  const auto permuted_weights = permute_weights_to_ok(shape, weights);
  const auto reference = cpu_conv3d(shape, output_d, output_h, output_w, input, weights);
  std::vector<float> output((size_t)gemm_m * gemm_n, 0.0f);

  auto source = create_shader_source();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto function = NS::TransferPtr(
      library->newFunction(NS::String::string("implicit_conv3d", NS::UTF8StringEncoding)));
  if (!function)
  {
    fprintf(stderr, "Failed to locate Metal function implicit_conv3d.\n");
    std::_Exit(1);
  }
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  KernelParams params{
      .batch = shape.batch,
      .input_d = shape.input_d,
      .input_h = shape.input_h,
      .input_w = shape.input_w,
      .input_c = shape.input_c,
      .output_d = output_d,
      .output_h = output_h,
      .output_w = output_w,
      .output_c = shape.output_c,
      .kernel_d = shape.kernel_d,
      .kernel_h = shape.kernel_h,
      .kernel_w = shape.kernel_w,
      .gemm_m = gemm_m,
      .gemm_n = gemm_n,
      .gemm_k = gemm_k,
  };

  auto input_buffer = NS::TransferPtr(device->newBuffer(
      input.data(),
      input.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto weight_buffer = NS::TransferPtr(device->newBuffer(
      permuted_weights.data(),
      permuted_weights.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto output_buffer = NS::TransferPtr(device->newBuffer(
      output.data(),
      output.size() * sizeof(float),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));

  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline.get());
  encoder->setBuffer(input_buffer.get(), 0, 0);
  encoder->setBuffer(weight_buffer.get(), 0, 1);
  encoder->setBuffer(output_buffer.get(), 0, 2);
  encoder->setBytes(&params, sizeof(params), 3);
  const auto ceil_divide =
      [](uint32_t target, uint32_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const uint32_t grid_x = ceil_divide(gemm_n, 32);
  const uint32_t grid_y = ceil_divide(gemm_m, 32);
  encoder->dispatchThreadgroups(
      MTL::Size(grid_x, grid_y, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  const auto* raw = static_cast<const float*>(output_buffer->contents());
  float max_abs_error = 0.0f;
  size_t max_error_index = 0;
  for (size_t i = 0; i < output.size(); ++i)
  {
    const float abs_error = std::fabs(raw[i] - reference[i]);
    if (abs_error > max_abs_error)
    {
      max_abs_error = abs_error;
      max_error_index = i;
    }
  }

  fprintf(
      stderr,
      "implicit conv3d scaffold: output=%ux%ux%u M=%u N=%u K=%u grid=%u x %u "
      "max_abs_error=%.6f max_error_index=%zu\n",
      output_d,
      output_h,
      output_w,
      gemm_m,
      gemm_n,
      gemm_k,
      grid_x,
      grid_y,
      max_abs_error,
      max_error_index);
  fflush(stderr);
  fflush(stdout);
  std::_Exit(max_abs_error <= 1e-3f ? 0 : 2);
}
