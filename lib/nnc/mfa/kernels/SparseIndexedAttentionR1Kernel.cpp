#include "SparseIndexedAttentionR1Kernel.hpp"

#include "../ccv_nnc_mfa.hpp"

SparseIndexedAttentionR1Kernel::SparseIndexedAttentionR1Kernel(
    SparseIndexedAttentionR1KernelDescriptor descriptor,
    MTL::Device* const device)
{
  memoryPrecision = descriptor.memoryPrecision;
  loadK = descriptor.loadK;
  attentionSinks = descriptor.attentionSinks;
  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

static uint32_t alignUp(uint32_t value, uint32_t alignment) noexcept {
  return (value + alignment - 1) / alignment * alignment;
}

uint32_t SparseIndexedAttentionR1Kernel::threadgroupMemoryAllocation(
    const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept
{
  // Every SIMD group publishes the first invalid sparse row it scanned.
  const uint32_t scanBytes = descriptor.simdgroups * sizeof(uint32_t);
  if (descriptor.mode == SparseIndexedAttentionR1Descriptor::Mode::splitReduce) {
    return alignUp(scanBytes, 16);
  }
  // Direct mode also merges every SIMD group's output, sum, and maximum.
  const uint32_t partialBytes =
      descriptor.simdgroups * (descriptor.D + 2) * sizeof(float);
  return alignUp(partialBytes + scanBytes, 16);
}

uint32_t SparseIndexedAttentionR1Kernel::threadgroupSize(
    const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept
{
  return 32 * descriptor.simdgroups;
}

uint32_t SparseIndexedAttentionR1Kernel::reduceThreadgroupSize(
    const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept
{
  // One output component per thread.
  return (descriptor.D + 31) / 32 * 32;
}

// Shared by both modes. The caller defines h, stream, streams, and partial_invalid.
// All SIMD groups scan for the terminator together; each then visits rows stream,
// stream + streams, ... and leaves its softmax state in q, acc, row_m, and row_s.
// exitPastHeads retires SIMD groups past the last head after the scan barrier.
static std::string traversalSource(const bool attentionSinks, const char* sinkOwner, const bool exitPastHeads, const uint32_t rowsPerStep) {
  std::string source = R"(
  // All SIMD groups share the scan for the first invalid virtual sparse row.
  // Blocked indices expand to block_size rows each.
  const uint sparse_capacity = K * block_size;
  uint first_invalid = sparse_capacity;
  for (uint i = sgid * 32 + lane_id; i < sparse_capacity; i += NSG * 32) {
    const int index = sparse_row(Indices, i);
    if (index < 0 || uint(index) >= sparse_rows) {
      first_invalid = i;
      break;
    }
  }
  first_invalid = simd_min(first_invalid);
  if (lane_id == 0) {
    partial_invalid[sgid] = first_invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
)";
  if (exitPastHeads) {
    source += R"(
  if (h >= H) {
    return;
  }
)";
  }
  source += R"(
  const uint sparse_count = simd_min(lane_id < NSG ? partial_invalid[lane_id] : sparse_capacity);
  // One query: blocked attention sees only its incomplete block densely.
  const uint dense_count =
      block_size > 1 ? dense_rows % block_size :
      sliding_window > 0 && sliding_window < dense_rows ?
      sliding_window : dense_rows;
  const uint dense_start = dense_rows - dense_count;
  const uint total_rows = dense_count + sparse_count;

  // Cache contiguous query and accumulator components in registers.
  const uint elements = (D_LEN + 31) / 32;
  float q[16], acc[16];
  for (uint i = 0; i < elements; ++i) {
    const uint d = lane_id * elements + i;
    q[i] = d < D_LEN ? (float)Q[h * D_LEN + d] * scale_log2e : 0;
    acc[i] = 0;
  }
  float row_m = )";
  if (attentionSinks) {
    source += std::string("(") + sinkOwner + " ? (float)Sinks[h * sink_head_stride] * 1.442695041 : -INFINITY)";
  } else {
    source += "-INFINITY";
  }
  source += R"(;
  float row_s = )";
  source += attentionSinks ? std::string("(") + sinkOwner + " ? 1 : 0)" : std::string("0");
  source += R"(;

  // Lane j resolves the row of step j in a 32-step batch, which keeps the
  // index load off the serial softmax chain.
  for (uint batch = stream; batch < total_rows; batch += 32 * streams) {
    const uint lane_row = batch + lane_id * streams;
    uint code = 0;
    if (lane_row < dense_count) {
      code = (dense_start + lane_row) | 0x80000000u;
    } else if (lane_row < total_rows) {
      code = uint(sparse_row(Indices, lane_row - dense_count));
    }
    const uint steps = min(32u, (total_rows - batch + streams - 1) / streams);
    uint j = 0;
)";
  if (rowsPerStep > 1) {
    const std::string rows = std::to_string(rowsPerStep);
    source += R"(
    // Several rows per step keep their key/value loads and reductions in flight.
    for (; j + )" + rows + R"( <= steps; j += )" + rows + R"() {
      float kv[)" + rows + R"(][16], score[)" + rows + R"(];
      #pragma clang loop unroll(full)
      for (uint r = 0; r < )" + rows + R"(; ++r) {
        device const real* kv_row = row_pointer(DenseKV, SparseKV, simd_shuffle(code, ushort(j + r)));
        score[r] = 0;
        for (uint i = 0; i < elements; ++i) {
          const uint d = lane_id * elements + i;
          if (d < D_LEN) {
            kv[r][i] = (float)kv_row[d];
            score[r] += q[i] * kv[r][i];
          }
        }
      }
      float new_m = row_m;
      #pragma clang loop unroll(full)
      for (uint r = 0; r < )" + rows + R"(; ++r) {
        score[r] = simd_sum(score[r]);
        new_m = max(new_m, score[r]);
      }
      const float old_scale = fast::exp2(row_m - new_m);
      row_s *= old_scale;
      for (uint i = 0; i < elements; ++i) {
        acc[i] *= old_scale;
      }
      #pragma clang loop unroll(full)
      for (uint r = 0; r < )" + rows + R"(; ++r) {
        const float row_scale = fast::exp2(score[r] - new_m);
        row_s += row_scale;
        for (uint i = 0; i < elements; ++i) {
          const uint d = lane_id * elements + i;
          if (d < D_LEN) {
            acc[i] += row_scale * kv[r][i];
          }
        }
      }
      row_m = new_m;
    }
)";
  }
  source += R"(
    for (; j < steps; ++j) {
      device const real* kv_row = row_pointer(DenseKV, SparseKV, simd_shuffle(code, ushort(j)));
      float kv[16];
      float score = 0;
      for (uint i = 0; i < elements; ++i) {
        const uint d = lane_id * elements + i;
        if (d < D_LEN) {
          kv[i] = (float)kv_row[d];
          score += q[i] * kv[i];
        }
      }
      score = simd_sum(score);
      const float new_m = max(row_m, score);
      const float old_scale = fast::exp2(row_m - new_m);
      const float row_scale = fast::exp2(score - new_m);
      for (uint i = 0; i < elements; ++i) {
        const uint d = lane_id * elements + i;
        if (d < D_LEN) {
          acc[i] = acc[i] * old_scale + row_scale * kv[i];
        }
      }
      row_s = row_s * old_scale + row_scale;
      row_m = new_m;
    }
  }
)";
  return source;
}

std::string SparseIndexedAttentionR1Kernel::createSource() const noexcept {
  std::string source = createConstants();
  source += R"(
#include <metal_stdlib>
using namespace metal;

// Dense rows carry the high bit; sparse rows index SparseKV directly.
inline device const real* row_pointer(device const real* dense, device const real* sparse, const uint code) {
  return (code & 0x80000000u) ? dense + (code & 0x7fffffffu) * D_LEN : sparse + code * D_LEN;
}
)";
  for (const bool partial : {false, true}) {
    source += partial ?
        "\nkernel void sparse_indexed_attention_r1_split_partials(\n" :
        "\nkernel void sparse_indexed_attention_r1_direct(\n";
    source += R"(    device const real* Q [[buffer(0)]],
    device const real* DenseKV [[buffer(1)]],
    device const real* SparseKV [[buffer(2)]],
    device const int* Indices [[buffer(3)]],
)";
    if (attentionSinks) {
      source += "    device const real* Sinks [[buffer(4)]],\n";
    }
    source += partial ?
        "    device float* Partial [[buffer(5)]],\n" :
        "    device real* O [[buffer(5)]],\n";
    if (loadK) {
      source += "    const device uint* shape [[buffer(6)]],\n";
    }
    if (attentionSinks) {
      source += "    constant uint& sink_head_stride [[buffer(7)]],\n";
    }
    source += R"(    threadgroup uchar* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
)";
    if (loadK) {
      source += R"(  const uniform<uint> dense_rows = make_uniform(shape[0]);
  const uniform<uint> sparse_rows = make_uniform(shape[1]);
  const uniform<uint> K = make_uniform(shape[2]);
)";
    }
    if (partial) {
      // NSG heads per workgroup walk the same rows, so their key/value loads
      // share cache lines. Each SIMD group owns one head of one partition.
      source += R"(  const uint h = tgid.x * NSG + sgid;
  const uint stream = tgid.z;
  const uint streams = NWG;
  threadgroup uint* partial_invalid = (threadgroup uint*)scratch;
)";
      source += traversalSource(attentionSinks, "stream == 0", true, 4);
      source += R"(
  device float* partial = Partial + (h * NWG + stream) * (D_LEN + 2);
  for (uint i = 0; i < elements; ++i) {
    const uint d = lane_id * elements + i;
    if (d < D_LEN) {
      partial[d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial[D_LEN] = row_s;
    partial[D_LEN + 1] = row_m;
  }
}
)";
    } else {
      // One workgroup per head; its SIMD groups split the rows and merge here.
      source += R"(  const uint h = tgid.x;
  const uint stream = sgid;
  const uint streams = NSG;
  threadgroup float* partial_o = (threadgroup float*)scratch;
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;
  threadgroup uint* partial_invalid = (threadgroup uint*)(partial_m + NSG);
)";
      source += traversalSource(attentionSinks, "sgid == 0", false, 2);
      source += R"(
  for (uint i = 0; i < elements; ++i) {
    const uint d = lane_id * elements + i;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const bool valid = lane_id < NSG && partial_s[lane_id] > 0;
  const float local_m = valid ? partial_m[lane_id] : -INFINITY;
  const float global_m = simd_max(local_m);
  const float factor = valid ? fast::exp2(local_m - global_m) : 0;
  const float global_s = simd_sum(valid ? partial_s[lane_id] * factor : 0);
  device real* o_row = O + h * D_LEN;
  // Keep all lanes active for the shuffles, including a partial output tile.
  for (uint base = sgid * 32; base < D_LEN; base += NSG * 32) {
    const uint d = base + lane_id;
    float numerator = 0;
    for (uint i = 0; i < NSG; ++i) {
      const float rescale = simd_shuffle(factor, i);
      if (d < D_LEN) numerator += partial_o[i * D_LEN + d] * rescale;
    }
    if (d < D_LEN) o_row[d] = global_s > 0 ? (real)(numerator / global_s) : (real)0;
  }
}
)";
    }
  }
  source += R"(
kernel void sparse_indexed_attention_r1_split_reduce(
    device const float* Partial [[buffer(0)]],
    device real* O [[buffer(1)]],
    uint h [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort nsg [[simdgroups_per_threadgroup]])
{
  // One lane per partition (NWG <= 32); shuffles broadcast each rescale.
  device const float* base = Partial + h * NWG * (D_LEN + 2);
  const bool valid = lane_id < NWG && base[lane_id * (D_LEN + 2) + D_LEN] > 0;
  const float local_m = valid ? base[lane_id * (D_LEN + 2) + D_LEN + 1] : -INFINITY;
  const float global_m = simd_max(local_m);
  const float factor = valid ? fast::exp2(local_m - global_m) : 0;
  const float global_s = simd_sum(valid ? base[lane_id * (D_LEN + 2) + D_LEN] * factor : 0);
  device real* o_row = O + h * D_LEN;
  for (uint tile = sgid * 32; tile < D_LEN; tile += nsg * 32) {
    const uint d = tile + lane_id;
    float numerator = 0;
    for (uint i = 0; i < NWG; ++i) {
      const float rescale = simd_shuffle(factor, i);
      if (d < D_LEN) numerator += base[i * (D_LEN + 2) + d] * rescale;
    }
    if (d < D_LEN) o_row[d] = global_s > 0 ? (real)(numerator / global_s) : (real)0;
  }
}
)";
  return source;
}

std::string SparseIndexedAttentionR1Kernel::createConstants() const noexcept {
  std::string defines;
  if (memoryPrecision == GEMMOperandPrecision::FP32) {
    defines += "typedef float real;\n";
  } else if (memoryPrecision == GEMMOperandPrecision::BF16) {
    defines += "typedef bfloat real;\n";
  } else {
    defines += "typedef half real;\n";
  }
  defines += "constant uint H [[function_constant(0)]];\n";
  defines += "constant uint D_LEN [[function_constant(1)]];\n";
  defines += "constant uint NSG [[function_constant(2)]];\n";
  defines += "constant uint NWG [[function_constant(3)]];\n";
  defines += "constant float dot_product_scale [[function_constant(4)]];\n";
  defines += "constant uint sliding_window [[function_constant(5)]];\n";
  if (!loadK) {
    defines += "constant uint dense_rows [[function_constant(6)]];\n";
    defines += "constant uint sparse_rows [[function_constant(7)]];\n";
    defines += "constant uint K [[function_constant(8)]];\n";
  }
  defines += "constant float scale_log2e = dot_product_scale * 1.442695041;\n";
  defines += "constant uint block_size [[function_constant(9)]];\n";
  // Blocked sparse indices: virtual row i reads block Indices[i / block_size].
  defines +=
      "inline int sparse_row(device const int* row_indices, uint i) {\n"
      "  if (block_size <= 1) {\n"
      "    return row_indices[i];\n"
      "  }\n"
      "  const int base = row_indices[i / block_size];\n"
      "  return base < 0 ? base : base * int(block_size) + int(i % block_size);\n"
      "}\n";
  return defines;
}
