#include "ScaledDotProductArgPartitionKernel.hpp"
#include "CodeWriter.hpp"
#include "GEMMHeaders.hpp"
#include "../ccv_nnc_mfa.hpp"

ScaledDotProductArgPartitionKernel::ScaledDotProductArgPartitionKernel(ScaledDotProductArgPartitionKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecision = descriptor.memoryPrecision;
  kth = descriptor.kth;
  scoreMode = descriptor.scoreMode;
  scoreBlockM = descriptor.scoreBlockM;
  scoreBlockN = descriptor.scoreBlockN;
  scoreSIMDGroups = descriptor.scoreSIMDGroups;
  loadC = descriptor.loadC;
  loadM = descriptor.loadM;
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
  source.SetValue("CANDIDATE_INDICES_ARGUMENT", (scoreMode == 1 || scoreMode == 3) ? "  device const int* block_ids [[buffer(5)]],\n" : "");
  const bool dense = scoreMode == 0 || scoreMode == 4;
  const std::string topKVisible = dense ? "c < visible" : (scoreMode == 3 ? "c < visible && (uint(block_ids[t * (((C + candidate_block_size - 1) / candidate_block_size + 31) / 32) + (c / candidate_block_size) / 32]) & (1u << ((c / candidate_block_size) % 32))) != 0" : "scores[t * C + c] > -3.402823466e+38f");
  source.SetValue("TOPK_VISIBLE", topKVisible);
  source.SetValue("TOPK_SERIAL_FILTER", dense ? "" : "    if (!(" + topKVisible + ")) { continue; }\n");
  source.SetValue("CANDIDATE_RUNTIME_FIELDS", scoreMode != 0 ? "  uint key_count;\n" : "");
  source.SetValue("TOPK_LIMIT", dense || scoreMode == 3 ? "visible" : "C");
  source.SetValue("TOPK_INDEX", scoreMode == 1 ? "block_ids[t * candidate_count + c / candidate_block_size] * int(candidate_block_size) + int(c % candidate_block_size)" : "int(c)");
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
  source.SetValue("T_FUNCTION_CONSTANT", loadM ? "" : "constant uint T [[function_constant(0)]];\n");
  source.SetValue("LOAD_M_VALUE", loadM ? "  const uniform<uint> T = make_uniform(runtime_params.T);\n" : "");
  source.SetValue("TOPK_MERGE_M_ARGUMENT", loadM ? "  constant SDPAPRuntimeParams& runtime_params [[buffer(5)]],\n" : "");
  source.SetValue("C_FUNCTION_CONSTANT", loadC ? "" : "constant uint C [[function_constant(1)]];\n");
  source.SetValue("QUERY_OFFSET_FUNCTION_CONSTANT", loadC ? "" : "constant int query_offset [[function_constant(7)]];\n");
  source.SetValue("LOAD_C_PARAMETER", loadC ? ", uniform<uint> C, uniform<int> query_offset" : "");
  source.SetValue("VISIBLE_COUNT_FOR_TOKEN", loadC ? "visible_count_for_token(t, C, query_offset)" : "visible_count_for_token(t)");
  source.SetValue("INDEX_SCORE_C_ARGUMENT", (loadC || loadM) ? "  constant SDPAPRuntimeParams& runtime_params [[buffer(4)]],\n" : "");
  source.SetValue("TOPK_SERIAL_C_ARGUMENT", (loadC || loadM) ? "  constant SDPAPRuntimeParams& runtime_params [[buffer(2)]],\n" : "");
  source.SetValue("TOPK_C_ARGUMENT", (loadC || loadM) ? "  constant SDPAPRuntimeParams& runtime_params [[buffer(3)]],\n" : "");
  source.SetValue("LOAD_C_VALUE", loadC ? "  const uniform<uint> C = make_uniform(runtime_params.C);\n  const uniform<int> query_offset = make_uniform(runtime_params.query_offset);\n" : "");
  source += createMetalSimdgroupMatrixStorage(memoryPrecision == GEMMOperandPrecision::BF16) + "\n";
  source += R"(
using namespace metal;

typedef {{memory_precision}} real;
typedef {{register_precision}} register_real;

{{T_FUNCTION_CONSTANT}}{{C_FUNCTION_CONSTANT}}constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];
constant uint compression_ratio [[function_constant(4)]];
constant bool is_causal [[function_constant(5)]];
constant float scale [[function_constant(6)]];
{{QUERY_OFFSET_FUNCTION_CONSTANT}}

struct SDPAPRuntimeParams {
  uint C;
  int query_offset;
  uint T;
{{CANDIDATE_RUNTIME_FIELDS}}};
)";
  if (scoreMode != 0) {
    source += R"(
constant uint candidate_block_size [[function_constant(8)]];
constant uint candidate_count [[function_constant(9)]];
inline uint candidate_visible(uint t, constant SDPAPRuntimeParams& p) {
  return is_causal ? uint(clamp((p.query_offset + int(t) + 1) / int(compression_ratio), 0, int(p.key_count))) : p.key_count;
}
)";
  }
  source += R"(
inline uint visible_count_for_token(uint t{{LOAD_C_PARAMETER}}) {
  if (!is_causal) {
    return C;
  }
  int visible = (query_offset + int(t) + 1) / int(compression_ratio);
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

inline void merge_pair_at(
  threadgroup const float* a_scores,
  threadgroup const int* a_indices,
  threadgroup const float* b_scores,
  threadgroup const int* b_indices,
  short a_size,
  short b_size,
  short rank,
  thread float& score,
  thread int& index
) {
  const short a_pos = merge_partition_pairs(a_scores, a_indices, b_scores, b_indices, a_size, b_size, rank);
  const short b_pos = rank - a_pos;
  const bool take_b = b_pos < b_size && (a_pos >= a_size || better_pair(b_scores[b_pos], b_indices[b_pos], a_scores[a_pos], a_indices[a_pos]));
  if (take_b) {
    score = b_scores[b_pos];
    index = b_indices[b_pos];
  } else if (a_pos < a_size) {
    score = a_scores[a_pos];
    index = a_indices[a_pos];
  } else {
    score = -3.402823466e+38f;
    index = -1;
  }
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
    const short sort_mid = {{topk_values_per_thread}} * merge_lane;
    short a_size = min(short(sort_size / 2), short({{kth}}));
    short b_size = a_size;
    if (sort_mid < {{kth}}) {
      const short partition = merge_partition_pairs(a_scores, a_indices, b_scores, b_indices, a_size, b_size, sort_mid);
      a_scores += partition;
      a_indices += partition;
      b_scores += sort_mid - partition;
      b_indices += sort_mid - partition;
      a_size -= partition;
      b_size -= sort_mid - partition;
      merge_step_pairs(a_scores, a_indices, b_scores, b_indices, a_size, b_size, local_scores, local_indices);
    }
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

)";
  if (scoreMode == 1) {
    source += R"(
kernel void index_score(
  device const real* q [[buffer(0)]],
  device const real* k [[buffer(1)]],
  device const real* head_w [[buffer(2)]],
  device float* scores [[buffer(3)]],
  constant SDPAPRuntimeParams& runtime_params [[buffer(4)]],
  device const int* block_ids [[buffer(5)]],
  ushort lane [[thread_index_in_simdgroup]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {
  const uint t = tgid.y;
  const uint c = tgid.x * 4 + sgid;
  if (t >= runtime_params.T || c >= runtime_params.C) { return; }
  const int block = block_ids[t * candidate_count + c / candidate_block_size];
  const uint row = uint(max(block, 0)) * candidate_block_size + c % candidate_block_size;
  const uint visible = candidate_visible(t, runtime_params);
  float score = -3.402823466e+38f;
  if (block >= 0 && row < visible) {
    score = 0;
    for (uint h = 0; h < H; ++h) {
      float dot = 0;
      for (uint d = lane; d < D; d += 32) {
        dot += float(q[(t * H + h) * D + d]) * float(k[row * D + d]);
      }
      dot = simd_sum(dot);
      score += max(dot, 0.0f) * float(head_w[t * H + h]) * scale;
    }
  }
  if (lane == 0) { scores[t * runtime_params.C + c] = score; }
}
)";
  } else if (scoreMode == 2) {
    source += R"(
kernel void index_score(
  device const float* scores [[buffer(0)]],
  device float* block_scores [[buffer(3)]],
  constant SDPAPRuntimeParams& runtime_params [[buffer(4)]],
  uint2 gid [[thread_position_in_grid]]
) {
  const uint block = gid.x;
  const uint t = gid.y;
  if (block >= runtime_params.C || t >= runtime_params.T) { return; }
  const uint keys = runtime_params.key_count;
  const uint visible = candidate_visible(t, runtime_params);
  const uint start = block * candidate_block_size;
  float score = -3.402823466e+38f;
  for (uint c = start; c < min(start + candidate_block_size, visible); ++c) {
    score = max(score, scores[t * keys + c]);
  }
  if (visible > 0 && block == (visible - 1) / candidate_block_size) {
    score = INFINITY;
  }
  block_scores[t * runtime_params.C + block] = score;
}
)";
  } else {
    source += R"(
kernel void index_score(
  device real* q [[buffer(0)]],
  device real* k [[buffer(1)]],
  device const real* head_w [[buffer(2)]],
  device float* scores [[buffer(3)]],
{{INDEX_SCORE_C_ARGUMENT}}  ushort lane_id [[thread_index_in_simdgroup]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {
{{LOAD_C_VALUE}}{{LOAD_M_VALUE}}  const uint c_start = tgid.x * {{score_block_n}};
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
    const uint visible = {{VISIBLE_COUNT_FOR_TOKEN}};
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
      const uint visible = {{VISIBLE_COUNT_FOR_TOKEN}};
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

)";
  }
  source += R"(
kernel void topk_serial(
{{CANDIDATE_INDICES_ARGUMENT}}  device const float* scores [[buffer(0)]],
  device int* selected [[buffer(1)]],
{{TOPK_SERIAL_C_ARGUMENT}}  uint t [[thread_position_in_grid]]
) {
{{LOAD_C_VALUE}}{{LOAD_M_VALUE}}  if (t >= T) {
    return;
  }
  float top_scores[{{kth}}];
  int top_indices[{{kth}}];
  for (uint i = 0; i < {{kth}}; ++i) {
    top_scores[i] = -3.402823466e+38f;
    top_indices[i] = -1;
    selected[t * {{kth}} + i] = -1;
  }
  const uint visible = {{VISIBLE_COUNT_FOR_TOKEN}};
  uint top_count = 0;
  for (uint c = 0; c < {{TOPK_LIMIT}}; ++c) {
{{TOPK_SERIAL_FILTER}}    const float score = scores[t * C + c];
    if (top_count == {{kth}} && !better_pair(score, {{TOPK_INDEX}}, top_scores[{{kth}} - 1], top_indices[{{kth}} - 1])) {
      continue;
    }
    uint pos = top_count < {{kth}} ? top_count++ : {{kth}} - 1;
    while (pos > 0 && better_pair(score, {{TOPK_INDEX}}, top_scores[pos - 1], top_indices[pos - 1])) {
      top_scores[pos] = top_scores[pos - 1];
      top_indices[pos] = top_indices[pos - 1];
      --pos;
    }
    top_scores[pos] = score;
    top_indices[pos] = {{TOPK_INDEX}};
  }
  const uint write_count = min(top_count, uint({{kth}}));
  for (uint i = 0; i < write_count; ++i) {
    selected[t * {{kth}} + i] = top_indices[i];
  }
}

kernel void topk_tile(
{{CANDIDATE_INDICES_ARGUMENT}}  device const float* scores [[buffer(0)]],
  device float* candidate_scores [[buffer(1)]],
  device int* candidate_indices [[buffer(2)]],
{{TOPK_C_ARGUMENT}}  uint2 tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
{{LOAD_C_VALUE}}{{LOAD_M_VALUE}}  const uint tile = tgid.x;
  const uint t = tgid.y;
  if (t >= T) {
    return;
  }
  threadgroup float tile_scores[{{topk_sort_values}}];
  threadgroup int tile_indices[{{topk_sort_values}}];
  const uint c_start = tile * {{topk_tile_c}};
  const uint visible = {{VISIBLE_COUNT_FOR_TOKEN}};
  for (uint i = tid; i < {{topk_sort_values}}; i += {{topk_threads}}) {
    const uint c = c_start + i;
    if (c < C && {{TOPK_VISIBLE}}) {
      tile_scores[i] = scores[t * C + c];
      tile_indices[i] = {{TOPK_INDEX}};
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

)";
  if (scoreMode != 0 && kth > 512) {
    source += R"(
kernel void topk_merge(
  device const float* candidate_scores [[buffer(0)]],
  device const int* candidate_indices [[buffer(1)]],
  device float* reduced_scores [[buffer(2)]],
  device int* reduced_indices [[buffer(3)]],
  constant uint& num_lists [[buffer(4)]],
{{TOPK_MERGE_M_ARGUMENT}}  uint2 tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
{{LOAD_M_VALUE}}  const uint first = tgid.x * 2;
  const uint lists = min(num_lists - first, 2u);
  const uint in_base = (tgid.y * num_lists + first) * {{kth}};
  const uint out_base = (tgid.y * ((num_lists + 1) / 2) + tgid.x) * {{kth}};
  threadgroup float s[2 * {{kth}}];
  threadgroup int ids[2 * {{kth}}];
  for (uint i = tid; i < lists * {{kth}}; i += 512) {
    s[i] = candidate_scores[in_base + i];
    ids[i] = candidate_indices[in_base + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < {{kth}}; i += 512) {
    float score; int index;
    merge_pair_at(s, ids, s + {{kth}}, ids + {{kth}}, {{kth}}, lists == 2 ? {{kth}} : 0, short(i), score, index);
    reduced_scores[out_base + i] = score;
    reduced_indices[out_base + i] = index;
  }
}
)";
  } else {
    source += R"(
kernel void topk_merge(
  device const float* candidate_scores [[buffer(0)]],
  device const int* candidate_indices [[buffer(1)]],
  device float* reduced_scores [[buffer(2)]],
  device int* reduced_indices [[buffer(3)]],
  constant uint& num_lists [[buffer(4)]],
{{TOPK_MERGE_M_ARGUMENT}}  uint2 tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
{{LOAD_M_VALUE}}  const uint group = tgid.x;
  const uint t = tgid.y;
  if (t >= T) {
    return;
  }
  const uint first_list = group * 4;
  const uint list_count = min(num_lists - first_list, 4u);
  const uint in_base = (t * num_lists + first_list) * {{kth}};
  const uint output_lists = (num_lists + 3) / 4;
  const uint out_base = (t * output_lists + group) * {{kth}};
  if (list_count == 1) {
    for (uint i = tid; i < {{kth}}; i += {{topk_threads}}) {
      reduced_scores[out_base + i] = candidate_scores[in_base + i];
      reduced_indices[out_base + i] = candidate_indices[in_base + i];
    }
    return;
  }
  threadgroup float merge_scores[{{topk_sort_values}}];
  threadgroup int merge_indices[{{topk_sort_values}}];
  const uint merge_count = list_count * {{kth}};
  for (uint i = tid; i < merge_count; i += {{topk_threads}}) {
    merge_scores[i] = candidate_scores[in_base + i];
    merge_indices[i] = candidate_indices[in_base + i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  thread float pair_scores[2];
  thread int pair_indices[2];
  if (tid < {{kth}}) {
    merge_pair_at(merge_scores, merge_indices, merge_scores + {{kth}}, merge_indices + {{kth}}, {{kth}}, {{kth}}, short(tid), pair_scores[0], pair_indices[0]);
    if (list_count <= 2) {
      reduced_scores[out_base + tid] = pair_scores[0];
      reduced_indices[out_base + tid] = pair_indices[0];
    } else {
      merge_pair_at(merge_scores + 2 * {{kth}}, merge_indices + 2 * {{kth}}, merge_scores + 3 * {{kth}}, merge_indices + 3 * {{kth}}, {{kth}}, list_count == 4 ? {{kth}} : 0, short(tid), pair_scores[1], pair_indices[1]);
    }
  }
  if (list_count <= 2) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < {{kth}}) {
    merge_scores[tid] = pair_scores[0];
    merge_indices[tid] = pair_indices[0];
    merge_scores[{{kth}} + tid] = pair_scores[1];
    merge_indices[{{kth}} + tid] = pair_indices[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < {{kth}}) {
    float score;
    int index;
    merge_pair_at(merge_scores, merge_indices, merge_scores + {{kth}}, merge_indices + {{kth}}, {{kth}}, {{kth}}, short(tid), score, index);
    reduced_scores[out_base + tid] = score;
    reduced_indices[out_base + tid] = index;
  }
}
)";
  }
  if (scoreMode != 0) {
    source += R"(
kernel void index_ids(
  device const int* input [[buffer(0)]],
  device int* output [[buffer(1)]],
  constant SDPAPRuntimeParams& runtime_params [[buffer(2)]],
  constant uint2& options [[buffer(3)]],
  uint t [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
  const uint length = options.x;
  const uint mode = options.y; // 0: normalize, 1: sort, 2: enumerate rows, 3: bitset, 4: enumerate blocks, 5: enumerate restricted rows.
  threadgroup int ids[2048];
  if (mode == 5 && candidate_block_size > 0) { // C <= kth <= 1024: enumerate the pool's eligible rows.
    const uint visible = candidate_visible(t, runtime_params);
    const uint blocks = (visible + candidate_block_size - 1) / candidate_block_size;
    // Produced full pools begin with every visible block in order. Check that
    // prefix on the GPU so arbitrary external pools still take the filtering path.
    bool complete = candidate_count >= blocks;
    if (complete) {
      for (uint i = tid; i < blocks; i += 256) { complete &= input[t * candidate_count + i] == int(i); }
    }
    const bool simd_complete = simd_all(complete);
    if (tid % 32 == 0) { ids[tid / 32] = simd_complete; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    complete = true;
    for (uint i = 0; i < 8; ++i) { complete &= ids[i] != 0; }
    if (complete) {
      for (uint i = tid; i < length; i += 256) { output[t * length + i] = i < visible ? int(i) : -1; }
      return;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint* bits = reinterpret_cast<threadgroup atomic_uint*>(ids);
    if (tid < 32) { atomic_store_explicit(bits + tid, 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < candidate_count; i += 256) {
      const int id = input[t * candidate_count + i];
      if (id >= 0 && uint(id) < blocks) {
        atomic_fetch_or_explicit(bits + uint(id) / 32, 1u << (uint(id) % 32), memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = 0;
    for (uint word = 0; word < (blocks + 31) / 32; ++word) {
      count += popcount(atomic_load_explicit(bits + word, memory_order_relaxed)) * candidate_block_size;
    }
    if (visible % candidate_block_size != 0 &&
        (atomic_load_explicit(bits + (blocks - 1) / 32, memory_order_relaxed) & (1u << ((blocks - 1) % 32))) != 0) {
      count -= candidate_block_size - visible % candidate_block_size;
    }
    for (uint row = tid; row < visible; row += 256) {
      const uint block = row / candidate_block_size;
      const uint word = block / 32;
      const uint mask = 1u << (block % 32);
      const uint value = atomic_load_explicit(bits + word, memory_order_relaxed);
      if ((value & mask) != 0) {
        uint preceding = popcount(value & (mask - 1));
        for (uint i = 0; i < word; ++i) {
          preceding += popcount(atomic_load_explicit(bits + i, memory_order_relaxed));
        }
        output[t * length + preceding * candidate_block_size + row % candidate_block_size] = int(row);
      }
    }
    for (uint i = count + tid; i < length; i += 256) { output[t * length + i] = -1; }
    return;
  }
  if (mode == 3 && candidate_block_size > 0) { // Dense-reader membership bitset; duplicates are idempotent.
    const uint blocks = (runtime_params.key_count + candidate_block_size - 1) / candidate_block_size;
    const uint words = (blocks + 31) / 32;
    device atomic_uint* bits = reinterpret_cast<device atomic_uint*>(output) + t * words;
    for (uint i = tid; i < words; i += 256) { atomic_store_explicit(bits + i, 0, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < length; i += 256) {
      const int id = input[t * length + i];
      if (id >= 0 && uint(id) < blocks) { atomic_fetch_or_explicit(bits + uint(id) / 32, 1u << (uint(id) % 32), memory_order_relaxed); }
    }
    return;
  }
  if (mode == 2 || mode == 4) {
    const uint visible = candidate_visible(t, runtime_params);
    const uint count = mode == 4 && candidate_block_size > 0 ? (visible + candidate_block_size - 1) / candidate_block_size : visible;
    for (uint i = tid; i < length; i += 256) { output[t * length + i] = i < count ? int(i) : -1; }
    return;
  }
  uint width = 1;
  while (width < length) { width *= 2; }
  const uint blocks = candidate_block_size > 0 ? (runtime_params.key_count + candidate_block_size - 1) / candidate_block_size : 0;
  for (uint i = tid; i < width; i += 256) {
    const int id = i < length ? input[t * length + i] : -1;
    ids[i] = id < 0 || (mode == 0 && uint(id) >= blocks) ? 2147483647 : id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= width; k *= 2) {
    for (uint j = k / 2; j > 0; j /= 2) {
      for (uint i = tid; i < width; i += 256) {
        const uint other = i ^ j;
        if (other > i) {
          const int a = ids[i], b = ids[other];
          const bool ascending = (i & k) == 0;
          ids[i] = ascending ? min(a, b) : max(a, b);
          ids[other] = ascending ? max(a, b) : min(a, b);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = tid; i < length; i += 256) {
    const int id = ids[i];
    output[t * length + i] = id == 2147483647 || (mode == 0 && i > 0 && id == ids[i - 1]) ? -1 : id;
  }
}
)";
  }
  return source.ToString();
}
