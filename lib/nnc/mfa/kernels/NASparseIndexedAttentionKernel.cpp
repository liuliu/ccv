#include "NASparseIndexedAttentionKernel.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

uint16_t NASparseIndexedAttentionKernel::sparseHeadGroup() const noexcept {
  switch (variant) {
    case NASparseIndexedAttentionVariant::Threadgroup24:
      return 24;
    case NASparseIndexedAttentionVariant::Threadgroup64:
      return 16;
    case NASparseIndexedAttentionVariant::Threadgroup64D128:
      return 16;
    case NASparseIndexedAttentionVariant::Threadgroup16:
    default:
      return 16;
  }
}

uint16_t NASparseIndexedAttentionKernel::sparseExecutionSIMDGroups() const noexcept {
  if (variant == NASparseIndexedAttentionVariant::Threadgroup64D128) {
    return 4;
  }
  return variant == NASparseIndexedAttentionVariant::Threadgroup64 ? 4 : 1;
}

uint32_t NASparseIndexedAttentionKernel::threadgroupMemoryAllocation() const noexcept {
  if (denseOnly) {
    return headGroup * denseBlockColumns * denseExecutionSIMDGroups * memoryPrecision.size();
  }
  if (variant == NASparseIndexedAttentionVariant::Threadgroup64D128) {
    return (threadgroupHeadDimensionD128 * threadgroupRowBlockD128 + sparseHeadGroup() * sparseExecutionSIMDGroups() * threadgroupRowBlockD128) * memoryPrecision.size() + threadgroupRowBlockD128 * sizeof(uint32_t);
  }
  return (headDimension * threadgroupRowBlock + sparseHeadGroup() * sparseExecutionSIMDGroups() * threadgroupRowBlock) * memoryPrecision.size() + threadgroupRowBlock * sizeof(uint32_t);
}

MTL::Size NASparseIndexedAttentionKernel::threadgroupSize() const noexcept {
  if (denseOnly) {
    return MTL::Size(simdGroupSize * denseExecutionSIMDGroups, 1, 1);
  }
  if (variant == NASparseIndexedAttentionVariant::Threadgroup64D128) {
    return MTL::Size(simdGroupSize * sparseExecutionSIMDGroups(), 1, 1);
  }
  return MTL::Size(simdGroupSize * sparseExecutionSIMDGroups(), 1, 1);
}

MTL::Size NASparseIndexedAttentionKernel::threadgroupsPerGrid(uint32_t T, uint32_t H) const noexcept {
  if (denseOnly) {
    return MTL::Size((H + headGroup - 1) / headGroup, (T + denseExecutionSIMDGroups - 1) / denseExecutionSIMDGroups, 1);
  }
  const uint32_t heads_per_threadgroup = sparseHeadGroup() * sparseExecutionSIMDGroups();
  return MTL::Size((H + heads_per_threadgroup - 1) / heads_per_threadgroup, T, 1);
}

NASparseIndexedAttentionKernel::NASparseIndexedAttentionKernel(NASparseIndexedAttentionKernelDescriptor descriptor, MTL::Device *const device) {
  memoryPrecision = descriptor.memoryPrecision;
  attentionSinks = descriptor.attentionSinks;
  denseOnly = descriptor.denseOnly;
  loadRows = descriptor.loadRows;
  variant = descriptor.variant;
  source = createSource();
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

void NASparseIndexedAttentionKernel::createThreadgroupAttendBlock(CodeWriter& source) const noexcept {
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      cS[i] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint load_idx = uint(tid); load_idx < {{HEAD_DIMENSION}}u * {{THREADGROUP_ROW_BLOCK}}u; load_idx += {{SPARSE_THREADS}}u) {
    const uint d = load_idx % {{HEAD_DIMENSION}}u;
    const uint row = load_idx / {{HEAD_DIMENSION}}u;
    KV_buf[load_idx] = (row < block_rows) ? kv_source[row_ids[row] * {{HEAD_DIMENSION}}u + d] : (real)0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for (ushort k = 0; k < K_edge; k += {{DIM_BLOCK}}u) {
    auto mQ_k = Q.slice<{{DIM_BLOCK}}, {{SPARSE_HEAD_GROUP}}>(k, token * H + head_base);
    auto mK_k = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>(k, 0);
    qk_op.run(mQ_k, mK_k, cS);
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      cS[i] = (idx[0] >= int(block_rows) || head_base + uint(idx[1]) >= H) ? -numeric_limits<float>::infinity() : cS[i];
    }
  }
  auto cM_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      correction[i] = 1;
      const float new_m = cM_new[i] * dot_scale;
      if (new_m > cM[i]) {
        correction[i] = fast::exp2(cM[i] - new_m);
        cM[i] = new_m;
      }
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      auto it = cS.get_iterator(i);
      auto dst_it = cM.map_iterator(it);
      cS[i] = (idx[0] >= int(block_rows)) ? 0 : fast::exp2(cS[i] * dot_scale - *dst_it);
    }
  }
  auto cL_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cL.get_capacity(); ++i) {
    if (cL.is_valid_element(i)) {
      cL[i] = cL[i] * correction[i] + cL_new[i];
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      auto it = cO0.get_iterator(i);
      auto dst_it = correction.map_iterator(it);
      cO0[i] *= *dst_it;
      cO1[i] *= *dst_it;
      cO2[i] *= *dst_it;
      cO3[i] *= *dst_it;
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      P_buf[idx[0] + idx[1] * {{THREADGROUP_ROW_BLOCK}}] = (real)cS[i];
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  {
    auto mV0 = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>(0, 0);
    pv_op.run(P, mV0, cO0);
  }
  {
    auto mV1 = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>({{DIM_BLOCK}}, 0);
    pv_op.run(P, mV1, cO1);
  }
  {
    auto mV2 = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>({{DIM_BLOCK_2}}, 0);
    pv_op.run(P, mV2, cO2);
  }
  {
    auto mV3 = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>({{DIM_BLOCK_3}}, 0);
    pv_op.run(P, mV3, cO3);
  }
)";
}

std::string NASparseIndexedAttentionKernel::createSource() const noexcept {
  if (denseOnly) {
    return createDenseOnlySource();
  }
  if (variant == NASparseIndexedAttentionVariant::Threadgroup64D128) {
    return createThreadgroupD128Source();
  }
  return createThreadgroupSource();
}

std::string NASparseIndexedAttentionKernel::createDenseOnlySource() const noexcept {
  CodeWriter source;
  source.SetValue("REAL", memoryPrecision.name());
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("HEAD_GROUP", std::to_string(headGroup));
  source.SetValue("DENSE_BLOCK_COLUMNS", std::to_string(denseBlockColumns));
  source.SetValue("DENSE_EXECUTION_SIMD_GROUPS", std::to_string(denseExecutionSIMDGroups));
  source.SetValue("DIM_BLOCK", std::to_string(dimBlock));
  source.SetValue("DIM_BLOCK_2", std::to_string(dimBlock * 2));
  source.SetValue("DIM_BLOCK_3", std::to_string(dimBlock * 3));
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

typedef {{REAL}} real;

constant uint T [[function_constant(0)]];\)";
  if (!loadRows) {
    source += R"(
constant uint dense_rows [[function_constant(1)]];\)";
  }
  source += R"(
constant uint H [[function_constant(3)]];
constant bool is_causal [[function_constant(5)]];
constant uint sink_head_stride [[function_constant(6)]];
constant float scale [[function_constant(7)]];
constant uint sliding_window [[function_constant(8)]];

constant uint K_edge = {{HEAD_DIMENSION}}u + 1u - {{DIM_BLOCK}}u;
constant float log2_e = 1.442695041f;

kernel void sparse_indexed_attention(
  device real* q [[buffer(0)]],
  device real* dense_k [[buffer(1)]],
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
  threadgroup uchar* threadgroup_block [[threadgroup(0)]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {\)";
  if (loadRows) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(runtime_rows.x);\)";
  }
  source += R"(
  const uint head_base = tgid.x * {{HEAD_GROUP}}u;
  const uint token = tgid.y * {{DENSE_EXECUTION_SIMD_GROUPS}}u + uint(sgid);
  if (head_base >= H || token >= T) {
    return;
  }
  auto Q = tensor<device real, dextents<int32_t, 2>, tensor_inline>(q, dextents<int32_t, 2>({{HEAD_DIMENSION}}, int(T * H)));
  auto K = tensor<device real, dextents<int32_t, 2>, tensor_inline>(dense_k, dextents<int32_t, 2>({{HEAD_DIMENSION}}, int(dense_rows)));
  auto V = tensor<device real, dextents<int32_t, 2>, tensor_inline>(dense_k, dextents<int32_t, 2>({{HEAD_DIMENSION}}, int(dense_rows)));
  threadgroup real* P_buf = (threadgroup real*)threadgroup_block + {{DENSE_BLOCK_COLUMNS}}u * {{HEAD_GROUP}}u * uint(sgid);
  auto P = tensor<threadgroup real, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{DENSE_BLOCK_COLUMNS}}, {{HEAD_GROUP}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{HEAD_GROUP}}, {{DENSE_BLOCK_COLUMNS}}, {{DIM_BLOCK}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> qk_op;
  constexpr auto pv_desc = matmul2d_descriptor({{HEAD_GROUP}}, {{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> pv_op;
  auto mQ = Q.slice<{{DIM_BLOCK}}, {{HEAD_GROUP}}>(0, token * H + head_base);
  auto mK = K.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>(0, 0);
  auto cS = qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto mV = V.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>(0, 0);
  auto cO0 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO1 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO2 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO3 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
)";
  if (attentionSinks) {
    source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      auto idx = cM.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[0]);
      cM[i] = (head >= H) ? -numeric_limits<float>::infinity() : (float)sinks[head * sink_head_stride] * log2_e;
      cL[i] = (head >= H) ? numeric_limits<float>::denorm_min() : 1;
    }
  }
)";
  } else {
    source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      cM[i] = -numeric_limits<float>::infinity();
      cL[i] = numeric_limits<float>::denorm_min();
    }
  }
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      cO0[i] = 0;
      cO1[i] = 0;
      cO2[i] = 0;
      cO3[i] = 0;
    }
  }
  uint dense_start = 0;
  uint dense_end = dense_rows;
  if (is_causal) {
    const int causal_end = int(dense_rows) - int(T) + int(token) + 1;
    dense_end = uint(clamp(causal_end, 0, int(dense_rows)));
    if (sliding_window > 0 && dense_end > sliding_window) {
      dense_start = dense_end - sliding_window;
    }
  }
  const uint dense_count = dense_end - dense_start;
  const uint dense_full_count = (dense_count / {{DENSE_BLOCK_COLUMNS}}u) * {{DENSE_BLOCK_COLUMNS}}u;
  const uint dense_remainder = dense_count - dense_full_count;
  const float dot_scale = scale * log2_e;
  for (uint off = 0; off < dense_full_count; off += {{DENSE_BLOCK_COLUMNS}}u) {
    const uint c = dense_start + off;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        cS[i] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < K_edge; k += {{DIM_BLOCK}}u) {
      auto mQ_k = Q.slice<{{DIM_BLOCK}}, {{HEAD_GROUP}}>(k, token * H + head_base);
      auto mK_k = K.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>(k, c);
      qk_op.run(mQ_k, mK_k, cS);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto idx = cS.get_multidimensional_index(i);
        const uint head = head_base + uint(idx[1]);
        cS[i] = (head >= H) ? -numeric_limits<float>::infinity() : cS[i];
      }
    }
    auto cM_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cM.get_capacity(); ++i) {
      if (cM.is_valid_element(i)) {
        correction[i] = 1;
        const float new_m = cM_new[i] * dot_scale;
        if (new_m > cM[i]) {
          correction[i] = fast::exp2(cM[i] - new_m);
          cM[i] = new_m;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto it = cS.get_iterator(i);
        auto dst_it = cM.map_iterator(it);
        cS[i] = fast::exp2(cS[i] * dot_scale - *dst_it);
      }
    }
    auto cL_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cL.get_capacity(); ++i) {
      if (cL.is_valid_element(i)) {
        cL[i] = cL[i] * correction[i] + cL_new[i];
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cO0.get_capacity(); ++i) {
      if (cO0.is_valid_element(i)) {
        auto it = cO0.get_iterator(i);
        auto dst_it = correction.map_iterator(it);
        cO0[i] *= *dst_it;
        cO1[i] *= *dst_it;
        cO2[i] *= *dst_it;
        cO3[i] *= *dst_it;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto idx = cS.get_multidimensional_index(i);
        P_buf[idx[0] + idx[1] * {{DENSE_BLOCK_COLUMNS}}] = (real)cS[i];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    {
      auto mV0 = V.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>(0, c);
      pv_op.run(P, mV0, cO0);
    }
    {
      auto mV1 = V.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>({{DIM_BLOCK}}, c);
      pv_op.run(P, mV1, cO1);
    }
    {
      auto mV2 = V.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>({{DIM_BLOCK_2}}, c);
      pv_op.run(P, mV2, cO2);
    }
    {
      auto mV3 = V.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>({{DIM_BLOCK_3}}, c);
      pv_op.run(P, mV3, cO3);
    }
  }
  if (dense_remainder > 0) {
    const uint c = dense_start + dense_full_count;
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto idx = cS.get_multidimensional_index(i);
        cS[i] = idx[0] >= int(dense_remainder) ? -numeric_limits<float>::infinity() : 0;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < K_edge; k += {{DIM_BLOCK}}u) {
      auto mQ_k = Q.slice<{{DIM_BLOCK}}, {{HEAD_GROUP}}>(k, token * H + head_base);
      auto mK_k = K.slice<{{DIM_BLOCK}}, {{DENSE_BLOCK_COLUMNS}}>(k, c);
      qk_op.run(mQ_k, mK_k, cS);
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto idx = cS.get_multidimensional_index(i);
        const uint head = head_base + uint(idx[1]);
        if (idx[0] >= int(dense_remainder) || head >= H) {
          cS[i] = -numeric_limits<float>::infinity();
        }
      }
    }
    auto cM_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cM.get_capacity(); ++i) {
      if (cM.is_valid_element(i)) {
        correction[i] = 1;
        const float new_m = cM_new[i] * dot_scale;
        if (new_m > cM[i]) {
          correction[i] = fast::exp2(cM[i] - new_m);
          cM[i] = new_m;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto it = cS.get_iterator(i);
        auto dst_it = cM.map_iterator(it);
        auto idx = cS.get_multidimensional_index(i);
        const uint head = head_base + uint(idx[1]);
        if (idx[0] >= int(dense_remainder) || head >= H) {
          cS[i] = 0;
        } else {
          cS[i] = fast::exp2(cS[i] * dot_scale - *dst_it);
        }
      }
    }
    auto cL_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cL.get_capacity(); ++i) {
      if (cL.is_valid_element(i)) {
        cL[i] = cL[i] * correction[i] + cL_new[i];
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cO0.get_capacity(); ++i) {
      if (cO0.is_valid_element(i)) {
        auto it = cO0.get_iterator(i);
        auto dst_it = correction.map_iterator(it);
        cO0[i] *= *dst_it;
        cO1[i] *= *dst_it;
        cO2[i] *= *dst_it;
        cO3[i] *= *dst_it;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < cS.get_capacity(); ++i) {
      if (cS.is_valid_element(i)) {
        auto idx = cS.get_multidimensional_index(i);
        if (idx[0] >= int(dense_remainder)) {
          P_buf[idx[0] - dense_remainder + idx[1] * {{DENSE_BLOCK_COLUMNS}}] = 0;
        } else {
          P_buf[{{DENSE_BLOCK_COLUMNS}}u - dense_remainder + uint(idx[0]) + uint(idx[1]) * {{DENSE_BLOCK_COLUMNS}}u] = (real)cS[i];
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mP = P.slice<dynamic_extent, {{HEAD_GROUP}}>({{DENSE_BLOCK_COLUMNS}}u - dense_remainder, 0);
    constexpr auto pv_remainder_desc = matmul2d_descriptor({{HEAD_GROUP}}, {{DIM_BLOCK}}, dynamic_length_v<int>, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<pv_remainder_desc, execution_simdgroups<1>> pv_remainder_op;
    {
      auto mV0 = V.slice<{{DIM_BLOCK}}, dynamic_extent>(0, c);
      pv_remainder_op.run(mP, mV0, cO0);
    }
    {
      auto mV1 = V.slice<{{DIM_BLOCK}}, dynamic_extent>({{DIM_BLOCK}}, c);
      pv_remainder_op.run(mP, mV1, cO1);
    }
    {
      auto mV2 = V.slice<{{DIM_BLOCK}}, dynamic_extent>({{DIM_BLOCK_2}}, c);
      pv_remainder_op.run(mP, mV2, cO2);
    }
    {
      auto mV3 = V.slice<{{DIM_BLOCK}}, dynamic_extent>({{DIM_BLOCK_3}}, c);
      pv_remainder_op.run(mP, mV3, cO3);
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      auto idx = cO0.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[1]);
      if (head < H) {
        auto it = cO0.get_iterator(i);
        auto dst_it = cL.map_iterator(it);
        const float inv_l = fast::divide(1, *dst_it);
        device real* out_head = out + (token * H + head) * {{HEAD_DIMENSION}}u;
        out_head[idx[0]] = (real)(cO0[i] * inv_l);
        out_head[idx[0] + {{DIM_BLOCK}}u] = (real)(cO1[i] * inv_l);
        out_head[idx[0] + {{DIM_BLOCK_2}}u] = (real)(cO2[i] * inv_l);
        out_head[idx[0] + {{DIM_BLOCK_3}}u] = (real)(cO3[i] * inv_l);
      }
    }
  }
}
)";
  return source.ToString();
}

std::string NASparseIndexedAttentionKernel::createThreadgroupSource() const noexcept {
  CodeWriter source;
  source.SetValue("REAL", memoryPrecision.name());
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("SPARSE_EXECUTION_SIMD_GROUPS", std::to_string(sparseExecutionSIMDGroups()));
  source.SetValue("SPARSE_HEAD_GROUP", std::to_string(sparseHeadGroup()));
  source.SetValue("SPARSE_THREADS", std::to_string(simdGroupSize * sparseExecutionSIMDGroups()));
  source.SetValue("THREADGROUP_ROW_BLOCK", std::to_string(threadgroupRowBlock));
  source.SetValue("DIM_BLOCK", std::to_string(dimBlock));
  source.SetValue("DIM_BLOCK_2", std::to_string(dimBlock * 2));
  source.SetValue("DIM_BLOCK_3", std::to_string(dimBlock * 3));
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

typedef {{REAL}} real;

constant uint T [[function_constant(0)]];\)";
  if (!loadRows) {
    source += R"(
constant uint dense_rows [[function_constant(1)]];
constant uint sparse_rows [[function_constant(2)]];\)";
  }
  source += R"(
constant uint H [[function_constant(3)]];
constant uint K [[function_constant(4)]];
constant bool is_causal [[function_constant(5)]];
constant uint sink_head_stride [[function_constant(6)]];
constant float scale [[function_constant(7)]];
constant uint sliding_window [[function_constant(8)]];

constant uint K_edge = {{HEAD_DIMENSION}}u + 1u - {{DIM_BLOCK}}u;
constant float log2_e = 1.442695041f;

kernel void sparse_indexed_attention(
  device real* q [[buffer(0)]],
  device real* dense_k [[buffer(1)]],
  device real* sparse_k [[buffer(2)]],
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
  threadgroup uchar* threadgroup_block [[threadgroup(0)]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  ushort tid [[thread_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {\)";
  if (loadRows) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(runtime_rows.x);
  const uniform<uint> sparse_rows = make_uniform(runtime_rows.y);\)";
  }
  source += R"(
  const uint head_base = (tgid.x * {{SPARSE_EXECUTION_SIMD_GROUPS}}u + uint(sgid)) * {{SPARSE_HEAD_GROUP}}u;
  const uint token = tgid.y;
  if (token >= T || head_base >= H) {
    return;
  }
  threadgroup real* KV_buf = (threadgroup real*)threadgroup_block;
  threadgroup uint* row_ids = (threadgroup uint*)(threadgroup_block + {{HEAD_DIMENSION}}u * {{THREADGROUP_ROW_BLOCK}}u * sizeof(real));
  threadgroup real* P_buf = (threadgroup real*)(threadgroup_block + {{HEAD_DIMENSION}}u * {{THREADGROUP_ROW_BLOCK}}u * sizeof(real) + {{THREADGROUP_ROW_BLOCK}}u * sizeof(uint)) + {{THREADGROUP_ROW_BLOCK}}u * {{SPARSE_HEAD_GROUP}}u * uint(sgid);
  auto Q = tensor<device real, dextents<int32_t, 2>, tensor_inline>(q, dextents<int32_t, 2>({{HEAD_DIMENSION}}, int(T * H)));
  auto KV = tensor<threadgroup real, dextents<int32_t, 2>, tensor_inline>(KV_buf, extents<int32_t, {{HEAD_DIMENSION}}, {{THREADGROUP_ROW_BLOCK}}>());
  auto P = tensor<threadgroup real, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{THREADGROUP_ROW_BLOCK}}, {{SPARSE_HEAD_GROUP}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{SPARSE_HEAD_GROUP}}, {{THREADGROUP_ROW_BLOCK}}, {{DIM_BLOCK}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> qk_op;
  constexpr auto pv_desc = matmul2d_descriptor({{SPARSE_HEAD_GROUP}}, {{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> pv_op;
  auto mQ = Q.slice<{{DIM_BLOCK}}, {{SPARSE_HEAD_GROUP}}>(0, token * H + head_base);
  auto mK = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>(0, 0);
  auto cS = qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto mV = KV.slice<{{DIM_BLOCK}}, {{THREADGROUP_ROW_BLOCK}}>(0, 0);
  auto cO0 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO1 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO2 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO3 = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      cM[i] = -numeric_limits<float>::infinity();
      cL[i] = 0;
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      cO0[i] = 0;
      cO1[i] = 0;
      cO2[i] = 0;
      cO3[i] = 0;
    }
  }
  const float dot_scale = scale * log2_e;
  uint dense_start = 0;
  uint dense_end = dense_rows;
  if (is_causal) {
    const int causal_end = int(dense_rows) - int(T) + int(token) + 1;
    dense_end = uint(clamp(causal_end, 0, int(dense_rows)));
    if (sliding_window > 0 && dense_end > sliding_window) {
      dense_start = dense_end - sliding_window;
    }
  }
  for (uint dense_base = dense_start; dense_base < dense_end; dense_base += {{THREADGROUP_ROW_BLOCK}}u) {
    uint block_rows = min({{THREADGROUP_ROW_BLOCK}}u, dense_end - dense_base);
    for (uint j = uint(tid); j < block_rows; j += {{SPARSE_THREADS}}u) {
      row_ids[j] = dense_base + j;
    }
    device real* kv_source = dense_k;
)";
  createThreadgroupAttendBlock(source);
  source += R"(
  }
  device const int* row_indices = indices + token * K;
  bool stop_sparse = false;
  for (uint sparse_base = 0; sparse_base < K && !stop_sparse; sparse_base += {{THREADGROUP_ROW_BLOCK}}u) {
    uint block_rows = 0;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < {{THREADGROUP_ROW_BLOCK}}u; j++) {
      if (sparse_base + j >= K) {
        stop_sparse = true;
        break;
      }
      const int idx = row_indices[sparse_base + j];
      if (idx < 0) {
        stop_sparse = true;
        break;
      }
      if (uint(idx) >= sparse_rows) {
        stop_sparse = true;
        break;
      }
      block_rows++;
    }
    if (block_rows == 0) {
      continue;
    }
    for (uint j = uint(tid); j < block_rows; j += {{SPARSE_THREADS}}u) {
      row_ids[j] = uint(row_indices[sparse_base + j]);
    }
    device real* kv_source = sparse_k;
)";
  createThreadgroupAttendBlock(source);
  source += R"(
  }
)";
  if (attentionSinks) {
    source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      auto idx = cM.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[0]);
      const float score = (head >= H) ? -numeric_limits<float>::infinity() : (float)sinks[head * sink_head_stride] * log2_e;
      const float old_m = cM[i];
      const float new_m = max(old_m, score);
      correction[i] = fast::exp2(old_m - new_m);
      cL[i] = cL[i] * correction[i] + fast::exp2(score - new_m);
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      auto it = cO0.get_iterator(i);
      auto dst_it = correction.map_iterator(it);
      cO0[i] *= *dst_it;
      cO1[i] *= *dst_it;
      cO2[i] *= *dst_it;
      cO3[i] *= *dst_it;
    }
  }
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO0.get_capacity(); ++i) {
    if (cO0.is_valid_element(i)) {
      auto idx = cO0.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[1]);
      if (head >= H) {
        continue;
      }
      auto it = cO0.get_iterator(i);
      auto dst_it = cL.map_iterator(it);
      const float inv_l = (*dst_it == 0) ? 0 : fast::divide(1, *dst_it);
      device real* out_head = out + (token * H + head) * {{HEAD_DIMENSION}}u;
      out_head[idx[0]] = (real)(cO0[i] * inv_l);
      out_head[idx[0] + {{DIM_BLOCK}}u] = (real)(cO1[i] * inv_l);
      out_head[idx[0] + {{DIM_BLOCK_2}}u] = (real)(cO2[i] * inv_l);
      out_head[idx[0] + {{DIM_BLOCK_3}}u] = (real)(cO3[i] * inv_l);
    }
  }
}
)";
  return source.ToString();
}

void NASparseIndexedAttentionKernel::createThreadgroupD128AttendBlock(CodeWriter& source) const noexcept {
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      cS[i] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint load_idx = uint(tid); load_idx < {{HEAD_DIMENSION_D128}}u * {{THREADGROUP_ROW_BLOCK_D128}}u; load_idx += {{SPARSE_THREADS_D128}}u) {
    const uint d = load_idx % {{HEAD_DIMENSION_D128}}u;
    const uint row = load_idx / {{HEAD_DIMENSION_D128}}u;
    KV_buf[load_idx] = (row < block_rows) ? kv_source[row_ids[row] * {{HEAD_DIMENSION_D128}}u + d] : (real)0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  qk_op.run(mQ, mK, cS);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      cS[i] = (idx[0] >= int(block_rows) || head_base + uint(idx[1]) >= H) ? -numeric_limits<float>::infinity() : cS[i];
    }
  }
  auto cM_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  reduce_rows(cS, cM_new, reduction_operation::max, -numeric_limits<float>::infinity());
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      correction[i] = 1;
      const float new_m = cM_new[i] * dot_scale;
      if (new_m > cM[i]) {
        correction[i] = fast::exp2(cM[i] - new_m);
        cM[i] = new_m;
      }
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      auto it = cS.get_iterator(i);
      auto dst_it = cM.map_iterator(it);
      cS[i] = (idx[0] >= int(block_rows)) ? 0 : fast::exp2(cS[i] * dot_scale - *dst_it);
    }
  }
  auto cL_new = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cL.get_capacity(); ++i) {
    if (cL.is_valid_element(i)) {
      cL[i] = cL[i] * correction[i] + cL_new[i];
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) {
    if (cO.is_valid_element(i)) {
      auto it = cO.get_iterator(i);
      auto dst_it = correction.map_iterator(it);
      cO[i] *= *dst_it;
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cS.get_capacity(); ++i) {
    if (cS.is_valid_element(i)) {
      auto idx = cS.get_multidimensional_index(i);
      P_buf[idx[0] + idx[1] * {{THREADGROUP_ROW_BLOCK_D128}}] = (real)cS[i];
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  pv_op.run(P, mV, cO);
)";
}

std::string NASparseIndexedAttentionKernel::createThreadgroupD128Source() const noexcept {
  CodeWriter source;
  source.SetValue("REAL", memoryPrecision.name());
  source.SetValue("HEAD_DIMENSION_D128", std::to_string(threadgroupHeadDimensionD128));
  source.SetValue("THREADGROUP_ROW_BLOCK_D128", std::to_string(threadgroupRowBlockD128));
  source.SetValue("SPARSE_EXECUTION_SIMD_GROUPS_D128", std::to_string(sparseExecutionSIMDGroups()));
  source.SetValue("SPARSE_HEAD_GROUP_D128", std::to_string(sparseHeadGroup()));
  source.SetValue("SPARSE_THREADS_D128", std::to_string(simdGroupSize * sparseExecutionSIMDGroups()));
  source += R"(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

typedef {{REAL}} real;

constant uint T [[function_constant(0)]];\)";
  if (!loadRows) {
    source += R"(
constant uint dense_rows [[function_constant(1)]];
constant uint sparse_rows [[function_constant(2)]];\)";
  }
  source += R"(
constant uint H [[function_constant(3)]];
constant uint K [[function_constant(4)]];
constant bool is_causal [[function_constant(5)]];
constant uint sink_head_stride [[function_constant(6)]];
constant float scale [[function_constant(7)]];
constant uint sliding_window [[function_constant(8)]];

constant float log2_e = 1.442695041f;

kernel void sparse_indexed_attention(
  device real* q [[buffer(0)]],
  device real* dense_k [[buffer(1)]],
  device real* sparse_k [[buffer(2)]],
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
  threadgroup uchar* threadgroup_block [[threadgroup(0)]],
  ushort sgid [[simdgroup_index_in_threadgroup]],
  ushort tid [[thread_index_in_threadgroup]],
  uint2 tgid [[threadgroup_position_in_grid]]
) {\)";
  if (loadRows) {
    source += R"(
  const uniform<uint> dense_rows = make_uniform(runtime_rows.x);
  const uniform<uint> sparse_rows = make_uniform(runtime_rows.y);\)";
  }
  source += R"(
  const uint head_base = (tgid.x * {{SPARSE_EXECUTION_SIMD_GROUPS_D128}}u + uint(sgid)) * {{SPARSE_HEAD_GROUP_D128}}u;
  const uint token = tgid.y;
  if (token >= T || head_base >= H) {
    return;
  }
  threadgroup real* KV_buf = (threadgroup real*)threadgroup_block;
  threadgroup uint* row_ids = (threadgroup uint*)(threadgroup_block + {{HEAD_DIMENSION_D128}}u * {{THREADGROUP_ROW_BLOCK_D128}}u * sizeof(real));
  threadgroup real* P_buf = (threadgroup real*)(threadgroup_block + {{HEAD_DIMENSION_D128}}u * {{THREADGROUP_ROW_BLOCK_D128}}u * sizeof(real) + {{THREADGROUP_ROW_BLOCK_D128}}u * sizeof(uint)) + {{THREADGROUP_ROW_BLOCK_D128}}u * {{SPARSE_HEAD_GROUP_D128}}u * uint(sgid);
  auto Q = tensor<device real, dextents<int32_t, 2>, tensor_inline>(q, dextents<int32_t, 2>({{HEAD_DIMENSION_D128}}, int(T * H)));
  auto KV = tensor<threadgroup real, dextents<int32_t, 2>, tensor_inline>(KV_buf, extents<int32_t, {{HEAD_DIMENSION_D128}}, {{THREADGROUP_ROW_BLOCK_D128}}>());
  auto P = tensor<threadgroup real, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, {{THREADGROUP_ROW_BLOCK_D128}}, {{SPARSE_HEAD_GROUP_D128}}>());
  constexpr auto qk_desc = matmul2d_descriptor({{SPARSE_HEAD_GROUP_D128}}, {{THREADGROUP_ROW_BLOCK_D128}}, {{HEAD_DIMENSION_D128}}, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> qk_op;
  constexpr auto pv_desc = matmul2d_descriptor({{SPARSE_HEAD_GROUP_D128}}, {{HEAD_DIMENSION_D128}}, {{THREADGROUP_ROW_BLOCK_D128}}, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> pv_op;
  auto mQ = Q.slice<{{HEAD_DIMENSION_D128}}, {{SPARSE_HEAD_GROUP_D128}}>(0, token * H + head_base);
  auto mK = KV.slice<{{HEAD_DIMENSION_D128}}, {{THREADGROUP_ROW_BLOCK_D128}}>(0, 0);
  auto cS = qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto mV = KV.slice<{{HEAD_DIMENSION_D128}}, {{THREADGROUP_ROW_BLOCK_D128}}>(0, 0);
  auto cO = pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      cM[i] = -numeric_limits<float>::infinity();
      cL[i] = 0;
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) {
    if (cO.is_valid_element(i)) {
      cO[i] = 0;
    }
  }
  const float dot_scale = scale * log2_e;
  uint dense_start = 0;
  uint dense_end = dense_rows;
  if (is_causal) {
    const int causal_end = int(dense_rows) - int(T) + int(token) + 1;
    dense_end = uint(clamp(causal_end, 0, int(dense_rows)));
    if (sliding_window > 0 && dense_end > sliding_window) {
      dense_start = dense_end - sliding_window;
    }
  }
  for (uint dense_base = dense_start; dense_base < dense_end; dense_base += {{THREADGROUP_ROW_BLOCK_D128}}u) {
    uint block_rows = min({{THREADGROUP_ROW_BLOCK_D128}}u, dense_end - dense_base);
    for (uint j = uint(tid); j < block_rows; j += {{SPARSE_THREADS_D128}}u) {
      row_ids[j] = dense_base + j;
    }
    device real* kv_source = dense_k;
)";
  createThreadgroupD128AttendBlock(source);
  source += R"(
  }
  device const int* row_indices = indices + token * K;
  bool stop_sparse = false;
  for (uint sparse_base = 0; sparse_base < K && !stop_sparse; sparse_base += {{THREADGROUP_ROW_BLOCK_D128}}u) {
    uint block_rows = 0;
    #pragma clang loop unroll(full)
    for (uint j = 0; j < {{THREADGROUP_ROW_BLOCK_D128}}u; j++) {
      if (sparse_base + j >= K) {
        stop_sparse = true;
        break;
      }
      const int idx = row_indices[sparse_base + j];
      if (idx < 0) {
        stop_sparse = true;
        break;
      }
      if (uint(idx) >= sparse_rows) {
        stop_sparse = true;
        break;
      }
      block_rows++;
    }
    if (block_rows == 0) {
      continue;
    }
    for (uint j = uint(tid); j < block_rows; j += {{SPARSE_THREADS_D128}}u) {
      row_ids[j] = uint(row_indices[sparse_base + j]);
    }
    device real* kv_source = sparse_k;
)";
  createThreadgroupD128AttendBlock(source);
  source += R"(
  }
)";
  if (attentionSinks) {
    source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cM.get_capacity(); ++i) {
    if (cM.is_valid_element(i)) {
      auto idx = cM.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[0]);
      const float score = (head >= H) ? -numeric_limits<float>::infinity() : (float)sinks[head * sink_head_stride] * log2_e;
      const float old_m = cM[i];
      const float new_m = max(old_m, score);
      correction[i] = fast::exp2(old_m - new_m);
      cL[i] = cL[i] * correction[i] + fast::exp2(score - new_m);
    }
  }
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) {
    if (cO.is_valid_element(i)) {
      auto it = cO.get_iterator(i);
      auto dst_it = correction.map_iterator(it);
      cO[i] *= *dst_it;
    }
  }
)";
  }
  source += R"(
  #pragma clang loop unroll(full)
  for (ushort i = 0; i < cO.get_capacity(); ++i) {
    if (cO.is_valid_element(i)) {
      auto idx = cO.get_multidimensional_index(i);
      const uint head = head_base + uint(idx[1]);
      if (head >= H) {
        continue;
      }
      auto it = cO.get_iterator(i);
      auto dst_it = cL.map_iterator(it);
      const float inv_l = (*dst_it == 0) ? 0 : fast::divide(1, *dst_it);
      device real* out_head = out + (token * H + head) * {{HEAD_DIMENSION_D128}}u;
      out_head[idx[0]] = (real)(cO[i] * inv_l);
    }
  }
}
)";
  return source.ToString();
}
