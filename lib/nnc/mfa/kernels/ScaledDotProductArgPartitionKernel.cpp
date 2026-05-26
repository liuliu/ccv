#include "ScaledDotProductArgPartitionKernel.hpp"
#include "CodeWriter.hpp"
#include "GEMMHeaders.hpp"
#include "../ccv_nnc_mfa.hpp"

ScaledDotProductArgPartitionKernel::ScaledDotProductArgPartitionKernel(ScaledDotProductArgPartitionKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecision = descriptor.memoryPrecision;
  kth = descriptor.kth;
  scoreBlockM = descriptor.scoreBlockM;
  scoreBlockN = descriptor.scoreBlockN;
  scoreSIMDGroups = descriptor.scoreSIMDGroups;
  CCV_NNC_MFA_PRECONDITION(scoreBlockM == 16);
  CCV_NNC_MFA_PRECONDITION(scoreBlockN == 32);
  CCV_NNC_MFA_PRECONDITION(scoreSIMDGroups == 4);
  scoreThreadgroupSize = MTL::Size(scoreSIMDGroups * 32, 1, 1);
  topKThreadgroupSize = MTL::Size(1, 1, 1);
  topKTileThreadgroupSize = MTL::Size(512, 1, 1);
  topKMergeThreadgroupSize = MTL::Size(512, 1, 1);
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string ScaledDotProductArgPartitionKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("memory_precision", memoryPrecision.name());
  source.SetValue("register_precision", memoryPrecision == GEMMOperandPrecision::BF16 ? "float" : memoryPrecision.name());
  source.SetValue("load_function", memoryPrecision == GEMMOperandPrecision::BF16 ? "load_bfloat" : "load");
  source.SetValue("kth", std::to_string(kth));
  source.SetValue("score_block_m", std::to_string(scoreBlockM));
  source.SetValue("score_block_n", std::to_string(scoreBlockN));
  source.SetValue("score_block_d", "128");
  source.SetValue("score_register_m", "8");
  source.SetValue("score_register_n", "16");
  source.SetValue("score_register_n_8", "2");
  source.SetValue("topk_tile_c", "2048");
  source.SetValue("topk_threads", "512");
  source.SetValue("topk_values_per_thread", "4");
  source.SetValue("topk_sort_values", "2048");
  source += createMetalSimdgroupMatrixStorage(memoryPrecision == GEMMOperandPrecision::BF16) + "\n";
  source += R"(
using namespace metal;

typedef {{memory_precision}} real;
typedef {{register_precision}} register_real;

constant uint T [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];
constant uint compression_ratio [[function_constant(4)]];
constant bool is_causal [[function_constant(5)]];
constant float scale [[function_constant(6)]];

inline uint visible_count_for_token(uint t) {
  if (!is_causal) {
    return C;
  }
  const int q_start = int(C * compression_ratio) - int(T);
  int visible = (q_start + int(t) + 1) / int(compression_ratio);
  visible = max(visible, 0);
  visible = min(visible, int(C));
  return uint(visible);
}

inline bool better_pair(float lhs_score, int lhs_idx, float rhs_score, int rhs_idx) {
  if (lhs_idx < 0) {
    return false;
  }
  if (rhs_idx < 0) {
    return true;
  }
  return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_idx < rhs_idx);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

inline void thread_sort4(thread float (&scores)[{{topk_values_per_thread}}], thread int (&indices)[{{topk_values_per_thread}}]) {
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < {{topk_values_per_thread}}; ++i) {
    #pragma clang loop unroll(full)
    for (ushort j = i & 1; j < {{topk_values_per_thread}} - 1; j += 2) {
      if (better_pair(scores[j + 1], indices[j + 1], scores[j], indices[j])) {
        const float score = scores[j];
        const int index = indices[j];
        scores[j] = scores[j + 1];
        indices[j] = indices[j + 1];
        scores[j + 1] = score;
        indices[j + 1] = index;
      }
    }
  }
}

inline short merge_partition_pairs(
  threadgroup const float* a_scores,
  threadgroup const int* a_indices,
  threadgroup const float* b_scores,
  threadgroup const int* b_indices,
  short a_size,
  short b_size,
  short sort_mid
) {
  short a_start = max(short(0), short(sort_mid - b_size));
  short a_end = min(sort_mid, a_size);
  while (a_start < a_end) {
    const short mid = a_start + (a_end - a_start) / 2;
    const short b_pos = sort_mid - 1 - mid;
    if (better_pair(b_scores[b_pos], b_indices[b_pos], a_scores[mid], a_indices[mid])) {
      a_end = mid;
    } else {
      a_start = mid + 1;
    }
  }
  return a_end;
}

inline void merge_step_pairs(
  threadgroup const float* a_scores,
  threadgroup const int* a_indices,
  threadgroup const float* b_scores,
  threadgroup const int* b_indices,
  short a_size,
  short b_size,
  thread float (&scores)[{{topk_values_per_thread}}],
  thread int (&indices)[{{topk_values_per_thread}}]
) {
  short a_pos = 0;
  short b_pos = 0;
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < {{topk_values_per_thread}}; ++i) {
    const bool take_b = b_pos < b_size && (a_pos >= a_size || better_pair(b_scores[b_pos], b_indices[b_pos], a_scores[a_pos], a_indices[a_pos]));
    if (take_b) {
      scores[i] = b_scores[b_pos];
      indices[i] = b_indices[b_pos];
      ++b_pos;
    } else if (a_pos < a_size) {
      scores[i] = a_scores[a_pos];
      indices[i] = a_indices[a_pos];
      ++a_pos;
    } else {
      scores[i] = -3.402823466e+38f;
      indices[i] = -1;
    }
  }
}

inline void block_merge_sort_pairs(threadgroup float* group_scores, threadgroup int* group_indices, ushort tid) {
  const ushort base = tid * {{topk_values_per_thread}};
  thread float local_scores[{{topk_values_per_thread}}];
  thread int local_indices[{{topk_values_per_thread}}];
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < {{topk_values_per_thread}}; ++i) {
    local_scores[i] = group_scores[base + i];
    local_indices[i] = group_indices[base + i];
  }
  thread_sort4(local_scores, local_indices);
  for (ushort merge_threads = 2; merge_threads <= {{topk_threads}}; merge_threads <<= 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < {{topk_values_per_thread}}; ++i) {
      group_scores[base + i] = local_scores[i];
      group_indices[base + i] = local_indices[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort merge_group = tid / merge_threads;
    const ushort merge_lane = tid - merge_group * merge_threads;
    const short sort_size = {{topk_values_per_thread}} * merge_threads;
    const short sort_start = sort_size * merge_group;
    threadgroup const float* a_scores = group_scores + sort_start;
    threadgroup const int* a_indices = group_indices + sort_start;
    threadgroup const float* b_scores = group_scores + sort_start + sort_size / 2;
    threadgroup const int* b_indices = group_indices + sort_start + sort_size / 2;
    short a_size = sort_size / 2;
    short b_size = sort_size / 2;
    const short sort_mid = {{topk_values_per_thread}} * merge_lane;
    const short partition = merge_partition_pairs(a_scores, a_indices, b_scores, b_indices, a_size, b_size, sort_mid);
    a_scores += partition;
    a_indices += partition;
    b_scores += sort_mid - partition;
    b_indices += sort_mid - partition;
    a_size -= partition;
    b_size -= sort_mid - partition;
    merge_step_pairs(a_scores, a_indices, b_scores, b_indices, a_size, b_size, local_scores, local_indices);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < {{topk_values_per_thread}}; ++i) {
    group_scores[base + i] = local_scores[i];
    group_indices[base + i] = local_indices[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float edge_score_cell(
  device const real* q,
  device const real* k,
  device const real* head_w,
  uint t,
  uint c
) {
  float accum = 0;
  for (uint h = 0; h < H; ++h) {
    float dot = 0;
    #pragma clang loop unroll(full)
    for (uint d = 0; d < {{score_block_d}}; ++d) {
      dot += float(q[(t * H + h) * D + d]) * float(k[c * D + d]);
    }
    if (dot > 0) {
      accum += dot * float(head_w[t * H + h]) * scale;
    }
  }
  return accum;
}

kernel void index_score(
  device real* q [[buffer(0)]],
  device real* k [[buffer(1)]],
  device const real* head_w [[buffer(2)]],
  device float* scores [[buffer(3)]],
  ushort lane_id [[thread_index_in_simdgroup]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {
  const uint c_start = tgid.x * {{score_block_n}};
  const uint t_start = tgid.y * {{score_block_m}};
  const uint sg_m = uint(sgid) / 2;
  const uint sg_n = uint(sgid) - sg_m * 2;
  const uint m_offset = t_start + sg_m * {{score_register_m}};
  const uint n_offset = c_start + sg_n * {{score_register_n}};
  const ushort2 morton_offset = morton_order(lane_id);
  const bool full_tile = m_offset + {{score_register_m}} <= T && n_offset + {{score_register_n}} <= C;
  if (full_tile) {
    thread simdgroup_matrix_storage<float> accum[{{score_register_n_8}}];
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < {{score_register_n}}; n += 8) {
      auto accum_tile = get_sram(accum, {{score_register_n}}, ushort2(n, 0));
      *accum_tile = simdgroup_matrix_storage<float>(float2(0));
    }
    thread simdgroup_matrix_storage<register_real> A_sram[1];
    thread simdgroup_matrix_storage<register_real> B_sram[{{score_register_n_8}}];
    thread simdgroup_matrix_storage<float> dot_sram[{{score_register_n_8}}];
    for (uint h = 0; h < H; ++h) {
      #pragma clang loop unroll(full)
      for (ushort n = 0; n < {{score_register_n}}; n += 8) {
        auto dot_tile = get_sram(dot_sram, {{score_register_n}}, ushort2(n, 0));
        *dot_tile = simdgroup_matrix_storage<float>(float2(0));
      }
      #pragma clang loop unroll(full)
      for (uint d = 0; d < {{score_block_d}}; d += 8) {
        auto A_src = simdgroup_matrix_storage<real>::apply_offset(q + h * D, H * D, uint2(d + uint(morton_offset.x), m_offset + uint(morton_offset.y)), false);
        A_sram[0].{{load_function}}(A_src, H * D, ushort2(0, 0), false);
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < {{score_register_n}}; n += 8) {
          auto B = get_sram(B_sram, {{score_register_n}}, ushort2(n, 0));
          auto B_src = simdgroup_matrix_storage<real>::apply_offset(k, D, uint2(n_offset + uint(n) + uint(morton_offset.x), d + uint(morton_offset.y)), true);
          B->{{load_function}}(B_src, D, ushort2(0, 0), true);
          auto dot_tile = get_sram(dot_sram, {{score_register_n}}, ushort2(n, 0));
          dot_tile->multiply(A_sram[0], *B);
        }
      }
      const float weight = float(head_w[(m_offset + uint(morton_offset.y)) * H + h]) * scale;
      #pragma clang loop unroll(full)
      for (ushort n = 0; n < {{score_register_n}}; n += 8) {
        auto dot_tile = get_sram(dot_sram, {{score_register_n}}, ushort2(n, 0));
        auto accum_tile = get_sram(accum, {{score_register_n}}, ushort2(n, 0));
        const float2 dot_values = *(dot_tile->thread_elements());
        float2 accum_values = *(accum_tile->thread_elements());
        accum_values += max(dot_values, float2(0)) * weight;
        *(accum_tile->thread_elements()) = accum_values;
      }
    }
    const uint t = m_offset + uint(morton_offset.y);
    const uint visible = visible_count_for_token(t);
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < {{score_register_n}}; n += 8) {
      auto accum_tile = get_sram(accum, {{score_register_n}}, ushort2(n, 0));
      const float2 score_values = *(accum_tile->thread_elements());
      const uint c0 = n_offset + uint(n) + uint(morton_offset.x);
      const uint c1 = c0 + 1;
      scores[t * C + c0] = c0 < visible ? score_values[0] : -3.402823466e+38f;
      scores[t * C + c1] = c1 < visible ? score_values[1] : -3.402823466e+38f;
    }
  } else {
    const uint t = m_offset + uint(morton_offset.y);
    if (t < T) {
      const uint visible = visible_count_for_token(t);
      #pragma clang loop unroll(full)
      for (ushort n = 0; n < {{score_register_n}}; n += 8) {
        const uint c0 = n_offset + uint(n) + uint(morton_offset.x);
        const uint c1 = c0 + 1;
        if (c0 < C) {
          scores[t * C + c0] = c0 < visible ? edge_score_cell(q, k, head_w, t, c0) : -3.402823466e+38f;
        }
        if (c1 < C) {
          scores[t * C + c1] = c1 < visible ? edge_score_cell(q, k, head_w, t, c1) : -3.402823466e+38f;
        }
      }
    }
  }
}

kernel void topk_serial(
  device const float* scores [[buffer(0)]],
  device int* selected [[buffer(1)]],
  uint t [[thread_position_in_grid]]
) {
  if (t >= T) {
    return;
  }
  float top_scores[{{kth}}];
  int top_indices[{{kth}}];
  for (uint i = 0; i < {{kth}}; ++i) {
    top_scores[i] = -3.402823466e+38f;
    top_indices[i] = -1;
    selected[t * {{kth}} + i] = -1;
  }
  const uint visible = visible_count_for_token(t);
  uint top_count = 0;
  for (uint c = 0; c < visible; ++c) {
    const float score = scores[t * C + c];
    if (top_count == {{kth}} && !better_pair(score, int(c), top_scores[{{kth}} - 1], top_indices[{{kth}} - 1])) {
      continue;
    }
    uint pos = top_count < {{kth}} ? top_count++ : {{kth}} - 1;
    while (pos > 0 && better_pair(score, int(c), top_scores[pos - 1], top_indices[pos - 1])) {
      top_scores[pos] = top_scores[pos - 1];
      top_indices[pos] = top_indices[pos - 1];
      --pos;
    }
    top_scores[pos] = score;
    top_indices[pos] = int(c);
  }
  const uint write_count = min(top_count, uint({{kth}}));
  for (uint i = 0; i < write_count; ++i) {
    selected[t * {{kth}} + i] = top_indices[i];
  }
}

kernel void topk_tile(
  device const float* scores [[buffer(0)]],
  device float* candidate_scores [[buffer(1)]],
  device int* candidate_indices [[buffer(2)]],
  uint2 tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  const uint tile = tgid.x;
  const uint t = tgid.y;
  if (t >= T) {
    return;
  }
  threadgroup float tile_scores[{{topk_sort_values}}];
  threadgroup int tile_indices[{{topk_sort_values}}];
  const uint c_start = tile * {{topk_tile_c}};
  const uint visible = visible_count_for_token(t);
  for (uint i = tid; i < {{topk_sort_values}}; i += {{topk_threads}}) {
    const uint c = c_start + i;
    if (c < C && c < visible) {
      tile_scores[i] = scores[t * C + c];
      tile_indices[i] = int(c);
    } else {
      tile_scores[i] = -3.402823466e+38f;
      tile_indices[i] = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort_pairs(tile_scores, tile_indices, tid);
  const uint num_tiles = (C + {{topk_tile_c}} - 1) / {{topk_tile_c}};
  const uint out_base = (t * num_tiles + tile) * {{kth}};
  for (uint i = tid; i < {{kth}}; i += {{topk_threads}}) {
    candidate_scores[out_base + i] = tile_scores[i];
    candidate_indices[out_base + i] = tile_indices[i];
  }
}

kernel void topk_merge(
  device const float* candidate_scores [[buffer(0)]],
  device const int* candidate_indices [[buffer(1)]],
  device int* selected [[buffer(2)]],
  uint t [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  if (t >= T) {
    return;
  }
  threadgroup float merge_scores[{{topk_sort_values}}];
  threadgroup int merge_indices[{{topk_sort_values}}];
  const uint num_tiles = (C + {{topk_tile_c}} - 1) / {{topk_tile_c}};
  const uint merge_count = num_tiles * {{kth}};
  const uint in_base = t * merge_count;
  for (uint i = tid; i < {{topk_sort_values}}; i += {{topk_threads}}) {
    if (i < merge_count) {
      merge_scores[i] = candidate_scores[in_base + i];
      merge_indices[i] = candidate_indices[in_base + i];
    } else {
      merge_scores[i] = -3.402823466e+38f;
      merge_indices[i] = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  block_merge_sort_pairs(merge_scores, merge_indices, tid);
  for (uint i = tid; i < {{kth}}; i += {{topk_threads}}) {
    selected[t * {{kth}} + i] = merge_indices[i];
  }
}
)";
  return source.ToString();
}
