#include "NAScaledDotProductArgPartitionKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

NAScaledDotProductArgPartitionKernel::NAScaledDotProductArgPartitionKernel(NAScaledDotProductArgPartitionKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecision = descriptor.memoryPrecision;
  kth = descriptor.kth;
  scoreBlockM = descriptor.scoreBlockM;
  scoreBlockN = descriptor.scoreBlockN;
  scoreSIMDGroups = descriptor.scoreSIMDGroups;
  loadC = descriptor.loadC;
  isCausal = descriptor.isCausal;
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

std::string NAScaledDotProductArgPartitionKernel::createSource() const noexcept {
  CodeWriter source;
  source.SetValue("memory_precision", memoryPrecision.name());
  source.SetValue("kth", std::to_string(kth));
  source.SetValue("score_block_m", std::to_string(scoreBlockM));
  source.SetValue("score_block_n", std::to_string(scoreBlockN));
  source.SetValue("score_block_d", "128");
  source.SetValue("score_simdgroups", std::to_string(scoreSIMDGroups));
  source.SetValue("topk_tile_c", "2048");
  source.SetValue("topk_threads", "512");
  source.SetValue("topk_values_per_thread", "4");
  source.SetValue("topk_sort_values", "2048");
  source.SetValue("C_FUNCTION_CONSTANT", loadC ? "" : "constant uint C [[function_constant(1)]];\n");
  source.SetValue("LOAD_C_PARAMETER", loadC ? ", uniform<uint> C" : "");
  source.SetValue("VISIBLE_COUNT_FOR_TOKEN", loadC ? "visible_count_for_token(t, C)" : "visible_count_for_token(t)");
  source.SetValue("INDEX_SCORE_C_ARGUMENT", loadC ? "  constant uint& C_buf [[buffer(4)]],\n" : "");
  source.SetValue("TOPK_SERIAL_C_ARGUMENT", loadC ? "  constant uint& C_buf [[buffer(2)]],\n" : "");
  source.SetValue("TOPK_C_ARGUMENT", loadC ? "  constant uint& C_buf [[buffer(3)]],\n" : "");
  source.SetValue("LOAD_C_VALUE", loadC ? "  const uniform<uint> C = make_uniform(C_buf);\n" : "");
  std::string scoreTileCulling;
  if (isCausal) {
    const std::string visibleCount = loadC ? "visible_count_for_token(last_t - 1, C)" : "visible_count_for_token(last_t - 1)";
    scoreTileCulling =
      "  if (t_start >= T) {\n"
      "    return;\n"
      "  }\n"
      "  const uint last_t = min(t_start + uint(" + std::to_string(scoreBlockM) + "), T);\n"
      "  const uint max_visible = " + visibleCount + ";\n"
      "  if (c_start >= max_visible) {\n"
      "    for (uint i = tid; i < " + std::to_string(scoreBlockM * scoreBlockN) + "; i += " + std::to_string(scoreSIMDGroups * 32) + ") {\n"
      "      const uint row = i / " + std::to_string(scoreBlockN) + ";\n"
      "      const uint col = i - row * " + std::to_string(scoreBlockN) + ";\n"
      "      const uint t = t_start + row;\n"
      "      const uint c = c_start + col;\n"
      "      if (t < T && c < C) {\n"
      "        scores[t * C + c] = -3.402823466e+38f;\n"
      "      }\n"
      "    }\n"
      "    return;\n"
      "  }\n";
  }
  source.SetValue("SCORE_TILE_CULLING", scoreTileCulling);
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace metal;
using namespace mpp::tensor_ops;

typedef {{memory_precision}} real;

constant uint T [[function_constant(0)]];
{{C_FUNCTION_CONSTANT}}constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];
constant uint compression_ratio [[function_constant(4)]];
constant bool is_causal [[function_constant(5)]];
constant float scale [[function_constant(6)]];
constant int query_offset [[function_constant(7)]];

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

kernel void index_score(
  device real* q [[buffer(0)]],
  device real* k [[buffer(1)]],
  device const real* head_w [[buffer(2)]],
  device float* scores [[buffer(3)]],
{{INDEX_SCORE_C_ARGUMENT}}  ushort tid [[thread_index_in_threadgroup]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {
{{LOAD_C_VALUE}}  const uint c_start = tgid.x * {{score_block_n}};
  const uint t_start = tgid.y * {{score_block_m}};
{{SCORE_TILE_CULLING}}  auto Q = tensor<device real, dextents<int32_t, 2>, tensor_inline>(q, dextents<int32_t, 2>(H * D, T));
  auto K = tensor<device real, dextents<int32_t, 2>, tensor_inline>(k, dextents<int32_t, 2>(D, C));
  constexpr auto score_desc = matmul2d_descriptor({{score_block_m}}, {{score_block_n}}, {{score_block_d}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<score_desc, execution_simdgroups<1>> score_matmul_op;
  auto mQ0 = Q.slice<{{score_block_d}}, {{score_block_m}}>(sgid * D, t_start);
  auto mK0 = K.slice<{{score_block_d}}, {{score_block_n}}>(0, c_start);
  auto cAccum = score_matmul_op.get_destination_cooperative_tensor<decltype(mQ0), decltype(mK0), float>();
  auto cDot = score_matmul_op.get_destination_cooperative_tensor<decltype(mQ0), decltype(mK0), float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cAccum.get_capacity(); ++i) {
    if (cAccum.is_valid_element(i)) {
      cAccum[i] = 0;
    }
  }
  for (uint h = sgid; h < H; h += {{score_simdgroups}}) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cDot.get_capacity(); ++i) {
      if (cDot.is_valid_element(i)) {
        cDot[i] = 0;
      }
    }
    auto mQ = Q.slice<{{score_block_d}}, {{score_block_m}}>(h * D, t_start);
    auto mK = K.slice<{{score_block_d}}, {{score_block_n}}>(0, c_start);
    score_matmul_op.run(mQ, mK, cDot);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cDot.get_capacity(); ++i) {
      if (cDot.is_valid_element(i)) {
        auto idx = cDot.get_multidimensional_index(i);
        const uint t = t_start + idx[1];
        if (t < T) {
          const float dot = cDot[i];
          if (dot > 0) {
            cAccum[i] += dot * float(head_w[t * H + h]) * scale;
          }
        }
      }
    }
  }
  threadgroup float partial_scores[{{score_simdgroups}} * {{score_block_m}} * {{score_block_n}}];
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cAccum.get_capacity(); ++i) {
    if (cAccum.is_valid_element(i)) {
      auto idx = cAccum.get_multidimensional_index(i);
      partial_scores[(uint(sgid) * {{score_block_m}} + idx[1]) * {{score_block_n}} + idx[0]] = cAccum[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < {{score_block_m}} * {{score_block_n}}; i += {{score_simdgroups}} * 32) {
    const uint row = i / {{score_block_n}};
    const uint col = i - row * {{score_block_n}};
    const uint t = t_start + row;
    const uint c = c_start + col;
    if (t < T && c < C) {
      const uint visible = {{VISIBLE_COUNT_FOR_TOKEN}};
      float score = -3.402823466e+38f;
      if (c < visible) {
        score = 0;
        #pragma clang loop unroll(full)
        for (ushort sg = 0; sg < {{score_simdgroups}}; ++sg) {
          score += partial_scores[(uint(sg) * {{score_block_m}} + row) * {{score_block_n}} + col];
        }
      }
      scores[t * C + c] = score;
    }
  }
}

kernel void topk_serial(
  device const float* scores [[buffer(0)]],
  device int* selected [[buffer(1)]],
{{TOPK_SERIAL_C_ARGUMENT}}  uint t [[thread_position_in_grid]]
) {
{{LOAD_C_VALUE}}  if (t >= T) {
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
{{TOPK_C_ARGUMENT}}  uint2 tgid [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
{{LOAD_C_VALUE}}  const uint tile = tgid.x;
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
{{TOPK_C_ARGUMENT}}  uint t [[threadgroup_position_in_grid]],
  uint tid [[thread_index_in_threadgroup]]
) {
{{LOAD_C_VALUE}}  if (t >= T) {
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
