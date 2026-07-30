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
  const uint32_t queryBytes = alignUp(
      descriptor.D * (uint32_t)descriptor.memoryPrecision.size(),
      sizeof(float));
  const uint32_t partialBytes =
      descriptor.simdgroups * (descriptor.D + 2) * sizeof(float);
  return alignUp(queryBytes + partialBytes + sizeof(uint32_t), 16);
}

uint32_t SparseIndexedAttentionR1Kernel::threadgroupSize(
    const SparseIndexedAttentionR1Descriptor& descriptor) const noexcept
{
  return 32 * descriptor.simdgroups;
}

std::string SparseIndexedAttentionR1Kernel::createSource() const noexcept {
  std::string source = createConstants();
  source += R"(
#include <metal_stdlib>
using namespace metal;

kernel void sparse_indexed_attention_r1_direct(
    device const real* Q [[buffer(0)]],
    device const real* DenseKV [[buffer(1)]],
    device const real* SparseKV [[buffer(2)]],
    device const int* Indices [[buffer(3)]],
)";
  if (attentionSinks) {
    source += R"(
    device const real* Sinks [[buffer(4)]],
)";
  }
  source += R"(
    device real* O [[buffer(5)]],
)";
  if (loadK) {
    source += R"(
    const device uint* shape [[buffer(6)]],
)";
  }
  if (attentionSinks) {
    source += R"(
    constant uint& sink_head_stride [[buffer(7)]],
)";
  }
  source += R"(
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint h [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
)";
  if (loadK) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(shape[0]);
  const uniform<uint> sparse_rows = make_uniform(shape[1]);
  const uniform<uint> K = make_uniform(shape[2]);
)";
  }
  source += R"(
  const uint q_shared_bytes = ((D_LEN * (uint)sizeof(real) + 3) / 4) * 4;
  threadgroup real* q_shared = (threadgroup real*)scratch;
  threadgroup float* partial_o = (threadgroup float*)(scratch + q_shared_bytes);
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;
  threadgroup uint* sparse_count_shared = (threadgroup uint*)(partial_m + NSG);

  if (sgid == 0) {
    device const real* q_row = Q + h * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = q_row[d];
    }
    uint first_invalid = K;
    for (uint i = lane_id; i < K; i += 32) {
      const int index = Indices[i];
      if ((index < 0 || uint(index) >= sparse_rows) && i < first_invalid) {
        first_invalid = i;
      }
    }
    first_invalid = simd_min(first_invalid);
    if (lane_id == 0) {
      sparse_count_shared[0] = first_invalid;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sparse_count = sparse_count_shared[0];
  const uint dense_count =
      sliding_window > 0 && sliding_window < dense_rows ?
      sliding_window : dense_rows;
  const uint dense_start = dense_rows - dense_count;
  const uint total_rows = dense_count + sparse_count;

  float acc[16];
  for (uint i = 0; i < 16; ++i) {
    acc[i] = 0;
  }
  float row_m = )";
  source += attentionSinks ?
      "(sgid == 0 ? (float)Sinks[h * sink_head_stride] * 1.442695041 : -INFINITY)" :
      "-INFINITY";
  source += R"(;
  float row_s = )";
  source += attentionSinks ? "(sgid == 0 ? 1 : 0)" : "0";
  source += R"(;

  for (uint row = sgid; row < total_rows; row += NSG) {
    device const real* kv_row;
    if (row < dense_count) {
      kv_row = DenseKV + (dense_start + row) * D_LEN;
    } else {
      const uint sparse_offset = row - dense_count;
      kv_row = SparseKV + uint(Indices[sparse_offset]) * D_LEN;
    }
    float dot_acc = 0;
    for (uint i = 0; i < 16; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        dot_acc += (float)q_shared[d] * (float)kv_row[d];
      }
    }
    const float score = simd_sum(dot_acc) * scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float row_scale = fast::exp2(score - new_m);
    for (uint i = 0; i < 16; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + row_scale * (float)kv_row[d];
      }
    }
    row_s = row_s * old_scale + row_scale;
    row_m = new_m;
  }

  for (uint i = 0; i < 16; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      if (partial_s[i] > 0) {
        global_m = max(global_m, partial_m[i]);
      }
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      if (partial_s[i] > 0) {
        global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
      }
    }
    device real* o_row = O + h * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        if (partial_s[i] > 0) {
          numerator += partial_o[i * D_LEN + d] *
              fast::exp2(partial_m[i] - global_m);
        }
      }
      o_row[d] = global_s > 0 ? (real)(numerator / global_s) : (real)0;
    }
  }
}

kernel void sparse_indexed_attention_r1_split_partials(
    device const real* Q [[buffer(0)]],
    device const real* DenseKV [[buffer(1)]],
    device const real* SparseKV [[buffer(2)]],
    device const int* Indices [[buffer(3)]],
)";
  if (attentionSinks) {
    source += R"(
    device const real* Sinks [[buffer(4)]],
)";
  }
  source += R"(
    device float* Partial [[buffer(5)]],
)";
  if (loadK) {
    source += R"(
    const device uint* shape [[buffer(6)]],
)";
  }
  if (attentionSinks) {
    source += R"(
    constant uint& sink_head_stride [[buffer(7)]],
)";
  }
  source += R"(
    threadgroup uchar* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint h = tgid.x;
  const uint iwg = tgid.z;
)";
  if (loadK) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(shape[0]);
  const uniform<uint> sparse_rows = make_uniform(shape[1]);
  const uniform<uint> K = make_uniform(shape[2]);
)";
  }
  source += R"(
  const uint q_shared_bytes = ((D_LEN * (uint)sizeof(real) + 3) / 4) * 4;
  threadgroup real* q_shared = (threadgroup real*)scratch;
  threadgroup float* partial_o = (threadgroup float*)(scratch + q_shared_bytes);
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;
  threadgroup uint* sparse_count_shared = (threadgroup uint*)(partial_m + NSG);

  if (sgid == 0) {
    device const real* q_row = Q + h * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = q_row[d];
    }
    uint first_invalid = K;
    for (uint i = lane_id; i < K; i += 32) {
      const int index = Indices[i];
      if ((index < 0 || uint(index) >= sparse_rows) && i < first_invalid) {
        first_invalid = i;
      }
    }
    first_invalid = simd_min(first_invalid);
    if (lane_id == 0) {
      sparse_count_shared[0] = first_invalid;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint sparse_count = sparse_count_shared[0];
  const uint dense_count =
      sliding_window > 0 && sliding_window < dense_rows ?
      sliding_window : dense_rows;
  const uint dense_start = dense_rows - dense_count;
  const uint total_rows = dense_count + sparse_count;

  float acc[16];
  for (uint i = 0; i < 16; ++i) {
    acc[i] = 0;
  }
  float row_m = )";
  source += attentionSinks ?
      "(iwg == 0 && sgid == 0 ? (float)Sinks[h * sink_head_stride] * 1.442695041 : -INFINITY)" :
      "-INFINITY";
  source += R"(;
  float row_s = )";
  source += attentionSinks ? "(iwg == 0 && sgid == 0 ? 1 : 0)" : "0";
  source += R"(;

  for (uint row = iwg * NSG + sgid; row < total_rows; row += NWG * NSG) {
    device const real* kv_row;
    if (row < dense_count) {
      kv_row = DenseKV + (dense_start + row) * D_LEN;
    } else {
      const uint sparse_offset = row - dense_count;
      kv_row = SparseKV + uint(Indices[sparse_offset]) * D_LEN;
    }
    float dot_acc = 0;
    for (uint i = 0; i < 16; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        dot_acc += (float)q_shared[d] * (float)kv_row[d];
      }
    }
    const float score = simd_sum(dot_acc) * scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float row_scale = fast::exp2(score - new_m);
    for (uint i = 0; i < 16; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + row_scale * (float)kv_row[d];
      }
    }
    row_s = row_s * old_scale + row_scale;
    row_m = new_m;
  }

  for (uint i = 0; i < 16; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      if (partial_s[i] > 0) {
        global_m = max(global_m, partial_m[i]);
      }
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      if (partial_s[i] > 0) {
        global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
      }
    }
    device float* row =
        Partial + (h * NWG + iwg) * (D_LEN + 2);
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        if (partial_s[i] > 0) {
          numerator += partial_o[i * D_LEN + d] *
              fast::exp2(partial_m[i] - global_m);
        }
      }
      row[d] = numerator;
    }
    if (lane_id == 0) {
      row[D_LEN] = global_s;
      row[D_LEN + 1] = global_m;
    }
  }
}

kernel void sparse_indexed_attention_r1_split_reduce(
    device const float* Partial [[buffer(0)]],
    device real* O [[buffer(1)]],
    uint h [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  device const float* base = Partial + h * NWG * (D_LEN + 2);
  float local_m = -INFINITY;
  if (lane_id < NWG) {
    device const float* row = base + lane_id * (D_LEN + 2);
    if (row[D_LEN] > 0) {
      local_m = row[D_LEN + 1];
    }
  }
  const float global_m = simd_max(local_m);
  float local_s = 0;
  if (lane_id < NWG) {
    device const float* row = base + lane_id * (D_LEN + 2);
    if (row[D_LEN] > 0) {
      local_s = row[D_LEN] * fast::exp2(row[D_LEN + 1] - global_m);
    }
  }
  const float global_s = simd_sum(local_s);
  device real* o_row = O + h * D_LEN;
  for (uint d = lane_id; d < D_LEN; d += 32) {
    float numerator = 0;
    for (uint i = 0; i < NWG; ++i) {
      device const float* row = base + i * (D_LEN + 2);
      if (row[D_LEN] > 0) {
        numerator += row[d] * fast::exp2(row[D_LEN + 1] - global_m);
      }
    }
    o_row[d] = global_s > 0 ? (real)(numerator / global_s) : (real)0;
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
  return defines;
}
