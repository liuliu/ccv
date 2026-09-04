#include "SparseIndexedAttentionKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

uint32_t SparseIndexedAttentionKernel::threadgroupMemoryAllocation() const noexcept {
  const uint32_t allocation = rowsPerBlock * maxHeadDimension * (uint32_t)memoryPrecision.size() + (rowsPerBlock + 2) * sizeof(uint32_t);
  return (allocation + 15) & ~15;
}

MTL::Size SparseIndexedAttentionKernel::threadgroupSize() const noexcept {
  return MTL::Size(threadsPerThreadgroup, 1, 1);
}

MTL::Size SparseIndexedAttentionKernel::threadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept {
  return MTL::Size((H + simdGroupsPerThreadgroup - 1) / simdGroupsPerThreadgroup, T, 1);
}

SparseIndexedAttentionKernel::SparseIndexedAttentionKernel(SparseIndexedAttentionKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecision = descriptor.memoryPrecision;
  attentionSinks = descriptor.attentionSinks;
  loadRows = descriptor.loadRows;
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string SparseIndexedAttentionKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("REAL", memoryPrecision.name());
  source.SetValue("THREADS", std::to_string(threadsPerThreadgroup));
  source.SetValue("SIMD_GROUPS", std::to_string(simdGroupsPerThreadgroup));
  source.SetValue("MAX_HEAD_DIMENSION", std::to_string(maxHeadDimension));
  source.SetValue("VALUES_PER_LANE", std::to_string(valuesPerLane));
  source.SetValue("ROWS_PER_BLOCK", std::to_string(rowsPerBlock));
  source += R"(
#include <metal_stdlib>

using namespace metal;

typedef {{REAL}} real;

constant uint T [[function_constant(0)]];\)";
  if (!loadRows) {
    source += R"(
constant uint dense_rows [[function_constant(1)]];
constant uint sparse_rows [[function_constant(2)]];\)";
  }
  source += R"(
constant uint H [[function_constant(3)]];
constant uint D [[function_constant(4)]];
constant uint K [[function_constant(5)]];
constant bool is_causal [[function_constant(6)]];
constant uint sink_head_stride [[function_constant(7)]];
constant float scale [[function_constant(8)]];
constant uint sliding_window [[function_constant(9)]];

constant float log2_e = 1.442695041f;
constant float scale_log2e = scale * log2_e;

kernel void sparse_indexed_attention(
  device const real* q [[buffer(0)]],
  device const real* dense_kv [[buffer(1)]],
  device const real* sparse_kv [[buffer(2)]],
  device const int* indices [[buffer(3)]],
)";
  if (attentionSinks) {
    source += R"(
  device const real* sinks [[buffer(4)]],
)";
  }
  source += R"(
  device real* out [[buffer(5)]],\)";
  if (loadRows) {
    source += R"(
  constant uint2& runtime_rows [[buffer(6)]],\)";
  }
  source += R"(
  threadgroup uchar* scratch [[threadgroup(0)]],
  ushort tid [[thread_index_in_threadgroup]],
  ushort lane [[thread_index_in_simdgroup]],
  ushort sg [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {\)";
  if (loadRows) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(runtime_rows.x);
  const uniform<uint> sparse_rows = make_uniform(runtime_rows.y);\)";
  }
  source += R"(
  threadgroup real* kv_shared = (threadgroup real*)scratch;
  const uint h = tgid.x * {{SIMD_GROUPS}}u + uint(sg);
  const uint t = tgid.y;
  const bool active = h < H && t < T && D <= {{MAX_HEAD_DIMENSION}}u;
  const uint q_base = (t * H + h) * D;

  float qv[{{VALUES_PER_LANE}}];
  float acc[{{VALUES_PER_LANE}}];
  for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
    const uint d = uint(lane) + i * 32u;
    qv[i] = (active && d < D) ? float(q[q_base + d]) : 0;
    acc[i] = 0;
  }
  float row_m = -INFINITY;
  float row_s = 0;

  uint dense_start = 0;
  uint dense_end = dense_rows;
  if (is_causal) {
    int visible = int(dense_rows) - int(T) + int(t) + 1;
    visible = max(visible, 0);
    visible = min(visible, int(dense_rows));
    dense_end = uint(visible);
    if (sliding_window > 0 && dense_end > sliding_window) {
      dense_start = dense_end - sliding_window;
    }
  }
  for (uint row0 = dense_start; row0 < dense_end; row0 += {{ROWS_PER_BLOCK}}u) {
    const uint rows = min({{ROWS_PER_BLOCK}}u, dense_end - row0);
    for (uint off = uint(tid); off < rows * D; off += {{THREADS}}u) {
      const uint row = off / D;
      const uint d = off - row * D;
      kv_shared[off] = dense_kv[(row0 + row) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint row = 0; row < rows; ++row) {
        float dot = 0;
        for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
          const uint d = uint(lane) + i * 32u;
          if (d < D) {
            dot += qv[i] * float(kv_shared[row * D + d]);
          }
        }
        const float score = simd_sum(dot) * scale_log2e;
        const float new_m = max(row_m, score);
        const float old_scale = fast::exp2(row_m - new_m);
        const float row_scale = fast::exp2(score - new_m);
        for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
          const uint d = uint(lane) + i * 32u;
          if (d < D) {
            acc[i] = acc[i] * old_scale + float(kv_shared[row * D + d]) * row_scale;
          }
        }
        row_s = row_s * old_scale + row_scale;
        row_m = new_m;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i0 = 0; i0 < K; i0 += {{ROWS_PER_BLOCK}}u) {
    threadgroup int* row_ids = (threadgroup int*)(scratch + {{ROWS_PER_BLOCK}}u * {{MAX_HEAD_DIMENSION}}u * sizeof(real));
    threadgroup uint* row_count = (threadgroup uint*)(row_ids + {{ROWS_PER_BLOCK}}u);
    threadgroup uint* stop_flag = row_count + 1;
    if (tid == 0) {
      uint rows = 0;
      uint stop = 0;
      for (uint i = 0; i < {{ROWS_PER_BLOCK}}u && i0 + i < K; ++i) {
        const int index = indices[t * K + i0 + i];
        if (index < 0 || uint(index) >= sparse_rows) {
          stop = 1;
          break;
        }
        row_ids[rows++] = index;
      }
      row_count[0] = rows;
      stop_flag[0] = stop;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint rows = row_count[0];
    for (uint off = uint(tid); off < rows * D; off += {{THREADS}}u) {
      const uint row = off / D;
      const uint d = off - row * D;
      kv_shared[off] = sparse_kv[uint(row_ids[row]) * D + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
      for (uint row = 0; row < rows; ++row) {
        float dot = 0;
        for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
          const uint d = uint(lane) + i * 32u;
          if (d < D) {
            dot += qv[i] * float(kv_shared[row * D + d]);
          }
        }
        const float score = simd_sum(dot) * scale_log2e;
        const float new_m = max(row_m, score);
        const float old_scale = fast::exp2(row_m - new_m);
        const float row_scale = fast::exp2(score - new_m);
        for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
          const uint d = uint(lane) + i * 32u;
          if (d < D) {
            acc[i] = acc[i] * old_scale + float(kv_shared[row * D + d]) * row_scale;
          }
        }
        row_s = row_s * old_scale + row_scale;
        row_m = new_m;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (stop_flag[0] != 0) {
      break;
    }
  }
)";
  if (attentionSinks) {
    source += R"(
  if (active) {
    const float score = float(sinks[h * sink_head_stride]) * log2_e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float row_scale = fast::exp2(score - new_m);
    for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
      acc[i] *= old_scale;
    }
    row_s = row_s * old_scale + row_scale;
  }
)";
  }
  source += R"(
  if (active) {
    const float inv_s = row_s == 0 ? 0 : fast::divide(1.0f, row_s);
    for (uint i = 0; i < {{VALUES_PER_LANE}}u; ++i) {
      const uint d = uint(lane) + i * 32u;
      if (d < D) {
        out[q_base + d] = (real)(acc[i] * inv_s);
      }
    }
  }
}
)";
  return source.ToString();
}
