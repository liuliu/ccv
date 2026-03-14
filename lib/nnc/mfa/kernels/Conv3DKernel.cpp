#include "Conv3DKernel.hpp"
#include "Conv3DDescriptor.hpp"
#include "CodeWriter.hpp"
#include "GEMMHeaders.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

Conv3DKernel::Conv3DKernel(Conv3DKernelDescriptor descriptor, MTL::Device *const device) {
  blockDimensions = descriptor.blockDimensions;
  CCV_NNC_MFA_PRECONDITION(blockDimensions[0] > 0 && blockDimensions[1] > 0);
  CCV_NNC_MFA_PRECONDITION(blockDimensions[0] <= 512 && (blockDimensions[0] % 32) == 0);
  CCV_NNC_MFA_PRECONDITION(blockDimensions[1] == 32);
  kernelDimensions = descriptor.kernelDimensions;
  dataType = descriptor.dataType;
  inputChannels = descriptor.inputChannels;
  outputChannels = descriptor.outputChannels;
  paddingLeft = descriptor.paddingLeft;
  paddingRight = descriptor.paddingRight;
  paddingTop = descriptor.paddingTop;
  paddingBottom = descriptor.paddingBottom;
  useBias = descriptor.useBias;

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

uint16_t Conv3DKernel::permutationThreadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept {
  return std::min<uint32_t>(std::max<uint32_t>(pipelineState->threadExecutionWidth(), 1), pipelineState->maxTotalThreadsPerThreadgroup());
}

uint16_t Conv3DKernel::threadgroupSize(MTL::ComputePipelineState *const pipelineState, const Conv3DDescriptor &descriptor) const noexcept {
  const uint16_t simdgroups_n = blockDimensions[0] > 32 ? blockDimensions[0] / 32 : 1;
  const uint16_t threadgroup_size = 32 * simdgroups_n;
  CCV_NNC_MFA_PRECONDITION(threadgroup_size <= pipelineState->maxTotalThreadsPerThreadgroup());
  return threadgroup_size;
}

MTL::Size Conv3DKernel::threadgroupsPerGrid(const Conv3DDescriptor &descriptor) const noexcept {
  auto ceilDivide =
  [=](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const uint32_t width_tiles = ceilDivide(descriptor.matrixDimensions[2], blockDimensions[1]);
  return MTL::Size(
      ceilDivide(descriptor.outputChannels, blockDimensions[0]),
      width_tiles * descriptor.matrixDimensions[0] * descriptor.matrixDimensions[1],
      descriptor.batchDimension);
}

std::string Conv3DKernel::createSource() const noexcept {
  CodeWriter source;
  source += createMetalSimdgroupMatrixStorage(false) + "\n";
  source += "using namespace metal;\n\n";
  source += R"(
constant uint DEPTH [[function_constant(0)]];
constant uint HEIGHT [[function_constant(1)]];
constant uint WIDTH [[function_constant(2)]];
)";

  source.SetValue("SCALAR_NAME", dataType == 3 ? "float" : "half");
  source.SetValue("SCALAR2_NAME", dataType == 3 ? "float2" : "half2");
  source.SetValue("KERNEL_DEPTH", std::to_string(kernelDimensions[0]));
  source.SetValue("KERNEL_HEIGHT", std::to_string(kernelDimensions[1]));
  source.SetValue("KERNEL_WIDTH", std::to_string(kernelDimensions[2]));
  source.SetValue("INPUT_CHANNELS", std::to_string(inputChannels));
  source.SetValue("OUTPUT_CHANNELS", std::to_string(outputChannels));
  source.SetValue("PADDING_LEFT", std::to_string(paddingLeft));
  source.SetValue("PADDING_RIGHT", std::to_string(paddingRight));
  source.SetValue("PADDING_TOP", std::to_string(paddingTop));
  source.SetValue("PADDING_BOTTOM", std::to_string(paddingBottom));
  const uint16_t register_m = 32;
  const uint16_t register_n = 32;
  const uint16_t simdgroups_n = blockDimensions[0] / register_n;
  source.SetValue("M_GROUP", std::to_string(register_m));
  source.SetValue("N_GROUP", std::to_string(blockDimensions[0]));
  source.SetValue("REGISTER_M", std::to_string(register_m));
  source.SetValue("REGISTER_N", std::to_string(register_n));
  source.SetValue("SIMDGROUPS_N", std::to_string(simdgroups_n));
  source.SetValue("SIMDGROUP_COUNT", std::to_string(simdgroups_n));
  source.SetValue("USE_BIAS", useBias ? "1" : "0");

  source += R"(
constant uint INPUT_CHANNELS = {{INPUT_CHANNELS}};
constant uint OUTPUT_CHANNELS = {{OUTPUT_CHANNELS}};
constant uint KERNEL_DEPTH = {{KERNEL_DEPTH}};
constant uint KERNEL_HEIGHT = {{KERNEL_HEIGHT}};
constant uint KERNEL_WIDTH = {{KERNEL_WIDTH}};
constant uint GEMM_K = KERNEL_DEPTH * KERNEL_HEIGHT * KERNEL_WIDTH * INPUT_CHANNELS;
constant int INPUT_DEPTH = int(DEPTH) + int(KERNEL_DEPTH) - 1;
constant int INPUT_HEIGHT = int(HEIGHT) + int(KERNEL_HEIGHT) - 1 - {{PADDING_TOP}} - {{PADDING_BOTTOM}};
constant int INPUT_WIDTH = int(WIDTH) + int(KERNEL_WIDTH) - 1 - {{PADDING_LEFT}} - {{PADDING_RIGHT}};
constant int PADDING_LEFT = {{PADDING_LEFT}};
constant int PADDING_RIGHT = {{PADDING_RIGHT}};
constant int PADDING_TOP = {{PADDING_TOP}};
constant int PADDING_BOTTOM = {{PADDING_BOTTOM}};

constant bool B_trans = true;
constant ushort M_group = {{M_GROUP}};
constant ushort N_group = {{N_GROUP}};
constant ushort REGISTER_M = {{REGISTER_M}};
constant ushort REGISTER_N = {{REGISTER_N}};
constant ushort SIMDGROUPS_N = {{SIMDGROUPS_N}};
constant ushort SIMDGROUP_COUNT = {{SIMDGROUP_COUNT}};

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

METAL_FUNC void initialize_accumulator(
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *C_sram,
)";
  if (useBias) {
    source += R"(
  const device {{SCALAR_NAME}} *bias,
)";
  }
  source += R"(
  uint N_offset,
  ushort2 morton_offset
) {
#pragma clang loop unroll(full)
  for (ushort n = 0; n < REGISTER_N; n += 8) {
    {{SCALAR2_NAME}} values({{SCALAR_NAME}}(0), {{SCALAR_NAME}}(0));
)";
  if (useBias) {
    source += R"(
    const uint channel = N_offset + n + morton_offset.x;
    if (channel + 1 < OUTPUT_CHANNELS) {
      values = *((const device {{SCALAR2_NAME}}*)(bias + channel));
    } else if (channel < OUTPUT_CHANNELS) {
      values[0] = bias[channel];
    }
)";
  }
  source += R"(
#pragma clang loop unroll(full)
    for (ushort m = 0; m < REGISTER_M; m += 8) {
      auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
      *C = simdgroup_matrix_storage<{{SCALAR_NAME}}>(values);
    }
  }
}

METAL_FUNC void multiply_accumulate_implicit_interior(
  const device {{SCALAR_NAME}} *input,
  const device {{SCALAR_NAME}} *weights,
  uint output_depth,
  uint output_height,
  uint M_offset,
  uint N_offset,
  ushort2 morton_offset,
  ushort2 offset_in_group,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *A_sram,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *B_sram,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *C_sram
) {
#pragma clang loop unroll(full)
  for (uint kd = 0; kd < KERNEL_DEPTH; ++kd) {
    const uint input_depth = output_depth + kd;
#pragma clang loop unroll(full)
    for (uint kh = 0; kh < KERNEL_HEIGHT; ++kh) {
      const uint row_plane_base =
        (((input_depth * uint(INPUT_HEIGHT) +
           (output_height + kh - uint(PADDING_TOP))) *
          uint(INPUT_WIDTH)) * INPUT_CHANNELS);
#pragma clang loop unroll(full)
      for (uint kw = 0; kw < KERNEL_WIDTH; ++kw) {
        const int input_width_base = int(M_offset) + int(kw) - PADDING_LEFT;
        const uint k_spatial_base =
          ((kd * KERNEL_HEIGHT + kh) * KERNEL_WIDTH + kw) * INPUT_CHANNELS;
#pragma clang loop unroll(enable)
        for (uint c_base = 0; c_base < INPUT_CHANNELS; c_base += 8) {
          const uint lane_channel = c_base + morton_offset.x;
          const device {{SCALAR_NAME}} *A_row =
            input + row_plane_base +
            (uint(input_width_base) + uint(morton_offset.y)) * INPUT_CHANNELS +
            lane_channel;
#pragma clang loop unroll(full)
          for (ushort m = 0; m < REGISTER_M; m += 8) {
            const {{SCALAR2_NAME}} values =
              *((const device {{SCALAR2_NAME}}*)A_row);
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            *A = simdgroup_matrix_storage<{{SCALAR_NAME}}>(values);
            A_row += 8 * INPUT_CHANNELS;
          }

          const uint k_base = k_spatial_base + c_base;
          const device {{SCALAR_NAME}} *B_src =
            weights + ulong(N_offset + offset_in_group.x) * GEMM_K +
            k_base + morton_offset.y;
#pragma clang loop unroll(full)
          for (ushort n = 0; n < REGISTER_N; n += 8) {
            auto B = get_sram(B_sram, REGISTER_N, ushort2(n, 0));
            B->load(B_src, GEMM_K, ushort2(n, 0), B_trans);
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

METAL_FUNC void multiply_accumulate_implicit(
  const device {{SCALAR_NAME}} *input,
  const device {{SCALAR_NAME}} *weights,
  uint output_depth,
  uint output_height,
  uint M_offset,
  uint N_offset,
  bool interior_tile,
  bool full_n_tile,
  ushort2 morton_offset,
  ushort2 offset_in_group,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *A_sram,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *B_sram,
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> *C_sram
) {
#pragma clang loop unroll(full)
  for (uint kd = 0; kd < KERNEL_DEPTH; ++kd) {
    const uint input_depth = output_depth + kd;
#pragma clang loop unroll(full)
    for (uint kh = 0; kh < KERNEL_HEIGHT; ++kh) {
      const int input_height = int(output_height) + int(kh) - PADDING_TOP;
      uint row_plane_base = 0;
      bool valid_h = false;
      if (input_height >= 0 && input_height < INPUT_HEIGHT) {
        valid_h = true;
        row_plane_base =
          (((input_depth * uint(INPUT_HEIGHT) + uint(input_height)) *
            uint(INPUT_WIDTH)) * INPUT_CHANNELS);
      }
#pragma clang loop unroll(full)
      for (uint kw = 0; kw < KERNEL_WIDTH; ++kw) {
        const int input_width_base = int(M_offset) + int(kw) - PADDING_LEFT;
        const uint k_spatial_base =
          ((kd * KERNEL_HEIGHT + kh) * KERNEL_WIDTH + kw) * INPUT_CHANNELS;
#pragma clang loop unroll(enable)
        for (uint c_base = 0; c_base < INPUT_CHANNELS; c_base += 8) {
          const uint lane_channel = c_base + morton_offset.x;

#pragma clang loop unroll(full)
          for (ushort m = 0; m < REGISTER_M; m += 8) {
            const ushort row = m + morton_offset.y;
            {{SCALAR2_NAME}} values({{SCALAR_NAME}}(0), {{SCALAR_NAME}}(0));
            if (interior_tile) {
              const uint address =
                row_plane_base +
                (uint(input_width_base) + uint(row)) * INPUT_CHANNELS +
                lane_channel;
              values = *((const device {{SCALAR2_NAME}}*)(input + address));
            } else if (valid_h) {
              const int input_width = input_width_base + int(row);
              if (input_width >= 0 && input_width < INPUT_WIDTH) {
                const uint address =
                  row_plane_base +
                  uint(input_width) * INPUT_CHANNELS +
                  lane_channel;
                values = *((const device {{SCALAR2_NAME}}*)(input + address));
              }
            }
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            *A = simdgroup_matrix_storage<{{SCALAR_NAME}}>(values);
          }

          const uint k_base = k_spatial_base + c_base;
          if (full_n_tile) {
            uint2 B_offset(N_offset, k_base);
            B_offset += uint2(offset_in_group.x, morton_offset.y);
            auto B_src = apply_offset_const(weights, GEMM_K, B_offset, B_trans);
#pragma clang loop unroll(full)
            for (ushort n = 0; n < REGISTER_N; n += 8) {
              auto B = get_sram(B_sram, REGISTER_N, ushort2(n, 0));
              B->load(B_src, GEMM_K, ushort2(n, 0), B_trans);
            }
          } else {
            const uint k = k_base + morton_offset.y;
#pragma clang loop unroll(full)
            for (ushort n = 0; n < REGISTER_N; n += 8) {
              const uint channel = N_offset + n + morton_offset.x;
              {{SCALAR2_NAME}} values({{SCALAR_NAME}}(0), {{SCALAR_NAME}}(0));
              if (channel + 1 < OUTPUT_CHANNELS) {
                values[0] = weights[channel * GEMM_K + k];
                values[1] = weights[(channel + 1) * GEMM_K + k];
              } else if (channel < OUTPUT_CHANNELS) {
                values[0] = weights[channel * GEMM_K + k];
              }
              auto B = get_sram(B_sram, REGISTER_N, ushort2(n, 0));
              *B = simdgroup_matrix_storage<{{SCALAR_NAME}}>(values);
            }
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

kernel void conv3d(device const {{SCALAR_NAME}} *input [[buffer(0)]],
                   device const {{SCALAR_NAME}} *weights [[buffer(1)]],
                   device {{SCALAR_NAME}} *output [[buffer(2)]],)";
  if (useBias) {
    source += R"(
                   device const {{SCALAR_NAME}} *bias [[buffer(3)]],)";
  }
  source += R"(
                   uint3 gid [[threadgroup_position_in_grid]],
                   ushort sidx [[simdgroup_index_in_threadgroup]],
                   ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint N_base = gid.x * N_group;
  const uint width_tiles = (WIDTH + M_group - 1) / M_group;
  const uint width_tile = gid.y % width_tiles;
  const uint height_depth = gid.y / width_tiles;
  const uint output_height = height_depth % HEIGHT;
  const uint output_depth = height_depth / HEIGHT;
  const uint M_base = width_tile * M_group;
  const uint N_offset = N_base + uint(sidx) * REGISTER_N;
  const uint M_offset = M_base;
  const uint batch = gid.z;

  if (N_offset >= OUTPUT_CHANNELS || M_offset >= WIDTH) {
    return;
  }

  const uint input_batch_base = batch * uint(INPUT_DEPTH) * uint(INPUT_HEIGHT) * uint(INPUT_WIDTH) * INPUT_CHANNELS;
  const uint output_batch_base = batch * DEPTH * HEIGHT * WIDTH * OUTPUT_CHANNELS;
  input += input_batch_base;
  output += output_batch_base + (((output_depth * HEIGHT + output_height) * WIDTH) * OUTPUT_CHANNELS);
)";
  if (useBias) {
    source += "\n  (void)bias;\n";
  }
  source += R"(

  const ushort2 morton_offset = morton_order(lane_id);
  const ushort2 offset_in_group(morton_offset.x, morton_offset.y);
  const bool full_m_tile = (M_offset + REGISTER_M <= WIDTH);
  const bool full_n_tile = (N_offset + REGISTER_N <= OUTPUT_CHANNELS);
  const bool interior_tile =
      full_m_tile && full_n_tile &&
      (output_height >= uint(PADDING_TOP)) &&
      (output_height + (KERNEL_HEIGHT - uint(PADDING_TOP) - 1) < uint(INPUT_HEIGHT)) &&
      (M_offset >= uint(PADDING_LEFT)) &&
      (M_offset + REGISTER_M + (KERNEL_WIDTH - uint(PADDING_LEFT) - 1) <= uint(INPUT_WIDTH));

  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> C_sram[(REGISTER_M / 8) * (REGISTER_N / 8)];
)";
  if (useBias) {
    source += R"(
  initialize_accumulator(C_sram, bias, N_offset, morton_offset);
)";
  } else {
    source += R"(
  initialize_accumulator(C_sram, N_offset, morton_offset);
)";
  }
  source += R"(
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> A_sram[(REGISTER_M / 8)];
  thread simdgroup_matrix_storage<{{SCALAR_NAME}}> B_sram[(REGISTER_N / 8)];
  if (interior_tile && full_n_tile) {
    multiply_accumulate_implicit_interior(
        input,
        weights,
        output_depth,
        output_height,
        M_offset,
        N_offset,
        morton_offset,
        offset_in_group,
        A_sram,
        B_sram,
        C_sram);
  } else {
    multiply_accumulate_implicit(
        input,
        weights,
        output_depth,
        output_height,
        M_offset,
        N_offset,
        interior_tile,
        full_n_tile,
        morton_offset,
        offset_in_group,
        A_sram,
        B_sram,
        C_sram);
  }

	  if (full_m_tile && full_n_tile) {
    uint2 C_offset(N_offset + offset_in_group.x, M_offset + offset_in_group.y);
    auto C_dst = simdgroup_matrix_storage<{{SCALAR_NAME}}>::apply_offset(
        output, OUTPUT_CHANNELS, C_offset);
#pragma clang loop unroll(full)
    for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
      for (ushort n = 0; n < REGISTER_N; n += 8) {
        auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
        C->store(C_dst, OUTPUT_CHANNELS, ushort2(n, m));
      }
    }
	  } else {
	    threadgroup {{SCALAR_NAME}} C_block[SIMDGROUP_COUNT * REGISTER_M * REGISTER_N];
	    threadgroup {{SCALAR_NAME}}* C_tile = C_block + uint(sidx) * REGISTER_M * REGISTER_N;
	    auto C_block_dst =
	      simdgroup_matrix_storage<{{SCALAR_NAME}}>::apply_offset(
	        C_tile, REGISTER_N, offset_in_group);
#pragma clang loop unroll(full)
	    for (ushort m = 0; m < REGISTER_M; m += 8) {
#pragma clang loop unroll(full)
	      for (ushort n = 0; n < REGISTER_N; n += 8) {
	        auto C = get_sram(C_sram, REGISTER_N, ushort2(n, m));
	        C->store(C_block_dst, REGISTER_N, ushort2(n, m));
	      }
	    }
	    threadgroup_barrier(mem_flags::mem_threadgroup);

	    for (uint i = lane_id; i < uint(REGISTER_M) * uint(REGISTER_N); i += 32) {
	      const uint row = i / uint(REGISTER_N);
	      const uint col = i % uint(REGISTER_N);
	      if (M_offset + row < WIDTH && N_offset + col < OUTPUT_CHANNELS) {
	        output[(M_offset + row) * OUTPUT_CHANNELS + (N_offset + col)] = C_tile[i];
	      }
	    }
	  }
}

kernel void permute_oidhw_to_ok(device const {{SCALAR_NAME}} *source [[buffer(0)]],
                                device {{SCALAR_NAME}} *destination [[buffer(1)]],
                                constant uint &outputChannels [[buffer(2)]],
                                constant uint &inputChannels [[buffer(3)]],
                                constant uint &kernelDepth [[buffer(4)]],
                                constant uint &kernelHeight [[buffer(5)]],
                                constant uint &kernelWidth [[buffer(6)]],
                                uint gid [[thread_position_in_grid]])
{
  const uint gemmK = kernelDepth * kernelHeight * kernelWidth * inputChannels;
  const uint elementCount = outputChannels * gemmK;
  if (gid >= elementCount) {
    return;
  }

  uint linear = gid;
  const uint ic = linear % inputChannels;
  linear /= inputChannels;
  const uint kw = linear % kernelWidth;
  linear /= kernelWidth;
  const uint kh = linear % kernelHeight;
  linear /= kernelHeight;
  const uint kd = linear % kernelDepth;
  const uint oc = linear / kernelDepth;

  const uint sourceIndex = ((((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth) + kw;
  destination[gid] = source[sourceIndex];
}
)";

  return source.ToString();
}
