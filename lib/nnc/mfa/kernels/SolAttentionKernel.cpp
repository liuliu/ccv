#include "SolAttentionKernel.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include <algorithm>

namespace {

// Sol owns its forward shader generator. Geometry and precision are fixed here
// rather than inherited from the native attention selector.
struct SolOperand {
  enum Value { Q, K, V, O, L, S, P };
  Value value;
  constexpr SolOperand(Value value) : value(value) {}
  std::string name() const {
    static const char* names[] = {"Q", "K", "V", "O", "L", "S", "P"};
    return names[value];
  }
  int bufferIndex() const { return int(value); }
};

struct SolAccumulate {
  SolOperand A, B, C;
  std::string firstIteration, everyIterationScale, lastIterationScale;
  SolAccumulate(SolOperand a, SolOperand b, SolOperand c, const std::string& every, const std::string& last)
      : A(a), B(b), C(c), everyIterationScale(every), lastIterationScale(last) {}
};

enum class SolAddressSpace { device, threadgroup };

struct SolAttentionSource {
  simd::ushort3 blockDimensions;
  static constexpr uint16_t headDimension = 128;
  bool preferAsyncCache, preferAsyncLoad;
  bool disableAsyncCopy = false;
  uint16_t threadgroupSize, threadgroupMemoryAllocation;

  SolAttentionSource(uint16_t rows, uint16_t columns, bool apple9)
      : blockDimensions{rows, columns, 32}, preferAsyncCache(apple9), preferAsyncLoad(!apple9),
        threadgroupSize(32 * (rows / 8)),
        threadgroupMemoryAllocation(std::max(rows * 32 * 2, columns * 32 * 2)) {}

  std::string memoryName(SolOperand) const { return "half"; }
  std::string registerName(SolOperand op) const { return op.value == SolOperand::O ? "float" : "half"; }
  std::string loadFunction(SolOperand) const { return "load"; }
  std::string storeFunction(SolOperand) const { return "store"; }
  bool transposed(SolOperand) const { return false; }
  std::string leadingDimension(SolOperand op) const { return op.name() + "_leading_dimension"; }
  uint16_t leadingBlockDimension(SolOperand) const { return blockDimensions[2]; }
  std::string parallelizationDimensionValue() const { return "R"; }
  std::string parallelizationGroupOffsetValue() const { return "parallelization_group_offset"; }
  std::string unsafeParallelizationThreadOffsetValue() const {
    return "parallelization_group_offset + sidx * 8 + morton_offset.y";
  }
  std::string clampedParallelizationThreadOffsetValue() const {
    return "min(" + unsafeParallelizationThreadOffsetValue() + ", R - 1)";
  }
  std::string traversalDimensionValue() const { return "C"; }
  std::string traversalOffsetValue() const { return "c"; }
  uint16_t paddedHeadDimensionValue() const { return headDimension; }
  uint16_t paddedHeadEdgeValue() const { return blockDimensions[2]; }
  std::string operandLocationValue(SolOperand op) const { return op.name(); }
  std::string operandLocationWithHeadOffsetValue(SolOperand op) const {
    if (op.value == SolOperand::L)
      return "L + (gid.z * Hq + gid.y) * R";
    const auto head = op.value == SolOperand::K || op.value == SolOperand::V ? "gid.y / H_Hk_ratio" : "gid.y";
    return op.name() + " + gid.z * " + op.name() + "_batch_stride + " + head + " * 128";
  }
  std::string paddedTraversalEdgeValue() const noexcept;
  std::string createConstants() const noexcept;
  std::string createBufferBindings() const noexcept;
  std::string accumulate(const SolAccumulate&) const noexcept;
  std::string storeOutput() const noexcept;
  std::string createSetup() const noexcept;
  std::string createCleanup() const noexcept;
  std::string maskAttentionMatrixEdge() const noexcept;
  std::string onlineReduceMaximum() const noexcept;
  std::string onlineCorrectO() const noexcept;
  std::string onlineReduceSum() const noexcept;
  std::string softmax() const noexcept;
};

std::string SolAttentionSource::paddedTraversalEdgeValue() const noexcept {
  auto blockDim = blockDimensions[1];
  auto remainder = traversalDimensionValue() + " % " + std::to_string(blockDim);

  std::string output = "(" + remainder + " == 0) ? " + std::to_string(blockDim) + " : " + remainder;
  output = "((" + output + ") + 7) / 8 * 8";
  return output;
}

std::string SolAttentionSource::createConstants() const noexcept {
  const SolOperand operands[] = {SolOperand::Q, SolOperand::K, SolOperand::V, SolOperand::O};
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  constant uint " + operand.name() + "_batch_stride [[function_constant(";
    output += std::to_string(operand.bufferIndex() + 5) + ")]];\n";
    {
      output += "  constant uint " + operand.name() + "_leading_dimension [[function_constant(";
      output += std::to_string(operand.bufferIndex() + 15) + ")]];\n";
    }
  }
  return R"(

    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    // Hq = number of query heads.
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];

    constant uint Hq [[function_constant(2)]];
    constant uint H_Hk_ratio [[function_constant(3)]];

	constant float dot_product_scale_derivative [[function_constant(4)]];
	constant float dot_product_scale = dot_product_scale_derivative * 1.442695041;

)" + output;
}

std::string SolAttentionSource::createBufferBindings() const noexcept {
  const SolOperand operands[] = {SolOperand::Q, SolOperand::K, SolOperand::V, SolOperand::O, SolOperand::L};
  std::string output = "";
  for (const auto& operand : operands) {
    output += "  device ";
    output += memoryName(operand);
    output += "* " + operand.name() + " [[buffer(";
    output += std::to_string(operand.bufferIndex()) + ")]],\n";
  }
  return output;
}

std::string SolAttentionSource::accumulate(const SolAccumulate& accumulateDesc) const noexcept {

  struct LoopIterationDescriptor {
    SolAddressSpace addressSpaceLHS;
    SolAddressSpace addressSpaceRHS;
    std::string registerOffset;
    unsigned short registerSize;
    LoopIterationDescriptor(SolAddressSpace aAddressSpaceLHS, SolAddressSpace aAddressSpaceRHS)
        : addressSpaceLHS(aAddressSpaceLHS), addressSpaceRHS(aAddressSpaceRHS), registerOffset(""), registerSize(0) {}
  };

  // MARK: - Initialize
  auto A = accumulateDesc.A;
  auto B = accumulateDesc.B;
  auto C = accumulateDesc.C;

  auto initializeAccumulator = [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("REGISTER_NAME_C", registerName(C));
    source.SetValue("C", C.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      auto {{C}} = {{C}}_sram + ({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8;
      *{{C}} = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(0);
    }

)";
    return source.ToString();
  };

  auto scaleAccumulator = [=](std::string scale, LoopIterationDescriptor descriptor) -> std::string {
    if (scale.empty()) {
      return "";
    }
    CodeWriter source;
    source.SetValue("SCALE", scale);
    source.SetValue("C", C.name());
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      auto {{C}} = {{C}}_sram + ({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8;
      *({{C}}->thread_elements()) *= {{SCALE}};
    }

)";
    return source.ToString();
  };

  // MARK: - Load RHS

  auto leadingDimensionRHS = [=](LoopIterationDescriptor descriptor) -> std::string {
    switch (descriptor.addressSpaceRHS) {
    case SolAddressSpace::device:
      return leadingDimension(B);
    case SolAddressSpace::threadgroup:
      return std::to_string(leadingBlockDimension(B));
    }
  };

  auto declareRHSLocation = [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("B", B.name());
    source.SetValue("B_LOCATION", operandLocationValue(B));
    source.SetValue("MEMORY_NAME_B", memoryName(B));
    source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
    source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
    source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    switch (descriptor.addressSpaceRHS) {
    case SolAddressSpace::device:
      source += R"(

      uint2 {{B}}_src_offset(
        morton_offset.x + d_outer,
        morton_offset.y + {{TRAVERSAL_OFFSET}});
      auto {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
        {{B}}_src_offset, {{TRANSPOSED_B}});

)";
      break;
    case SolAddressSpace::threadgroup:
      source += R"(

      ushort2 {{B}}_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto {{B}}_src = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);
      {{B}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
      ::apply_offset(
        {{B}}_src, {{LEADING_BLOCK_DIMENSION_B}},
        {{B}}_block_offset, {{TRANSPOSED_B}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      break;
    }
    return source.ToString();
  };

  auto loadRHS = [=](LoopIterationDescriptor descriptor) -> std::string {
    switch (descriptor.addressSpaceRHS) {
    case SolAddressSpace::device:
      return declareRHSLocation(descriptor);
    case SolAddressSpace::threadgroup:
      CodeWriter source;
      source.SetValue("B", B.name());
      source.SetValue("B_LOCATION", operandLocationValue(B));
      source.SetValue("MEMORY_NAME_B", memoryName(B));
      source.SetValue("LEADING_DIMENSION_B", leadingDimension(B));
      source.SetValue("LEADING_BLOCK_DIMENSION_B", std::to_string(leadingBlockDimension(B)));
      source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
      source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
      source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
      source.SetValue("PADDED_TRAVERSAL_EDGE", paddedTraversalEdgeValue());
      source.SetValue("DECLARE_RHS_LOCATION", declareRHSLocation(descriptor));
      if (disableAsyncCopy) {
        source.SetValue("ASYNC_LANE_ID", ", lane_id");
      } else {
        source.SetValue("ASYNC_LANE_ID", "");
      }
      source += R"(

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{B}}_offset(d_outer, {{TRAVERSAL_OFFSET}});
        auto src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>
        ::apply_offset(
          {{B_LOCATION}}, {{LEADING_DIMENSION_B}},
          {{B}}_offset, {{TRANSPOSED_B}});
        auto dst = (threadgroup {{MEMORY_NAME_B}}*)(threadgroup_block);

        ushort D_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort C_src_dimension = min(
          uint({{BLOCK_DIMENSIONS_TRAVERSAL}}),
          uint({{TRAVERSAL_DIMENSION}} - {{TRAVERSAL_OFFSET}}));
        ushort C_dst_dimension = max(
          ushort({{PADDED_TRAVERSAL_EDGE}}),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_B}}, 32>(
          dst, tile_dst,
          src, {{LEADING_DIMENSION_B}}, tile_src{{ASYNC_LANE_ID}}, {{TRANSPOSED_B}});
        simdgroup_event::wait(1, &event);
      }

      {{DECLARE_RHS_LOCATION}}

 )";
      return source.ToString();
    }
    return "";
  };

  // MARK: - Inner Loop

  auto innerLoopHead = [=](LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("DESCRIPTOR_REGISTER_OFFSET", descriptor.registerOffset);
    source.SetValue("REGISTER_NAME_B", registerName(B));
    source.SetValue("A", A.name());
    source.SetValue("B", B.name());
    source.SetValue("C", C.name());
    source.SetValue("LOAD_FUNCTION_B", loadFunction(B));
    source.SetValue("TRANSPOSED_B", transposed(B) ? "true" : "false");
    source.SetValue("LEADING_DIMENSION_RHS", leadingDimensionRHS(descriptor));
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{DESCRIPTOR_REGISTER_SIZE}}; d += 8) {
      // Load the RHS from memory.
      ushort2 {{B}}_origin(d, c);
      simdgroup_matrix_storage<{{REGISTER_NAME_B}}> {{B}};
      {{B}}.{{LOAD_FUNCTION_B}}(
        {{B}}_src, {{LEADING_DIMENSION_RHS}},
        {{B}}_origin, {{TRANSPOSED_B}});

      // Issue one SIMD matmul instruction.
      {{C}}_sram[({{DESCRIPTOR_REGISTER_OFFSET}} + d) / 8].multiply(
        {{A}}_sram[c / 8], {{B}}, /*accumulate=*/true);
    }

)";
    return source.ToString();
  };

  auto innerLoopTraversal = [=](std::string traversalStart, std::string traversalEnd,
                                LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("TRAVERSAL_START", traversalStart);
    source.SetValue("TRAVERSAL_END", traversalEnd);
    source.SetValue("INNER_LOOP_HEAD", innerLoopHead(descriptor));
    source += R"(

    #pragma clang loop unroll(full)
    for (ushort c = {{TRAVERSAL_START}}; c < {{TRAVERSAL_END}}; c += 8) {
      {{INNER_LOOP_HEAD}}
    }

)";
    return source.ToString();
  };

  // MARK: - Outer Loop

  auto loopIteration = [=](LoopIterationDescriptor descriptor) -> std::string {
    auto multiplyAB = [=]() -> std::string {
      CodeWriter source;
      if (descriptor.addressSpaceLHS == SolAddressSpace::device ||
          descriptor.addressSpaceRHS == SolAddressSpace::device) {
        auto blockDim = blockDimensions[1];
        source.SetValue("INNER_LOOP_TRAVERSAL", innerLoopTraversal("0", std::to_string(blockDim), descriptor));
        source.SetValue("BLOCK_DIM", std::to_string(blockDim));
        source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
        source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
        source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.lastIterationScale, descriptor));

        source += R"(

        {{INNER_LOOP_TRAVERSAL}}
        if (
          ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) &&
          ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} == {{TRAVERSAL_DIMENSION}})
        ) {
           {{SCALE_ACCUMULATOR}}
        }

)";

      } else {
        source.SetValue("INNER_LOOP_TRAVERSAL_0", innerLoopTraversal("0", paddedTraversalEdgeValue(), descriptor));
        source.SetValue("INNER_LOOP_TRAVERSAL_1",
                        innerLoopTraversal(paddedTraversalEdgeValue(), std::to_string(blockDimensions[1]), descriptor));
        source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
        source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
        source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
        source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.lastIterationScale, descriptor));

        source += R"(

        {{INNER_LOOP_TRAVERSAL_0}}
        if ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIMENSIONS_TRAVERSAL}}
            < {{TRAVERSAL_DIMENSION}}) {
          {{INNER_LOOP_TRAVERSAL_1}}
        } else {
          {{SCALE_ACCUMULATOR}}
        }

)";
      }
      return source.ToString();
    };

    CodeWriter source;
    source.SetValue("ALLOCATE_ACCUMULATOR", "");
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("ACCUMULATOR_IS_UNINITIALIZED",
                    !accumulateDesc.firstIteration.empty() ? accumulateDesc.firstIteration : "c == 0");
    source.SetValue("INITIALIZE_ACCUMULATOR", initializeAccumulator(descriptor));
    source.SetValue("LOAD_ACCUMULATOR", "");
    source.SetValue("STORE_ACCUMULATOR", "");
    source.SetValue("LOAD_RHS", loadRHS(descriptor));
    source.SetValue("MULTIPLY_AB", multiplyAB());
    source.SetValue("SCALE_ACCUMULATOR", scaleAccumulator(accumulateDesc.everyIterationScale, descriptor));
    source += R"(

    {{ALLOCATE_ACCUMULATOR}}
    if ({{ACCUMULATOR_IS_UNINITIALIZED}}) {
      {{INITIALIZE_ACCUMULATOR}}
    } else {
      {{LOAD_ACCUMULATOR}}
      {{SCALE_ACCUMULATOR}}
    }
    {{LOAD_RHS}}
    {{MULTIPLY_AB}}
    {{STORE_ACCUMULATOR}}

)";
    return source.ToString();
  };

  auto gatedLoopIteration = [=](LoopIterationDescriptor descriptor) -> std::string {
    auto descriptorThreadgroup = descriptor;
    descriptorThreadgroup.addressSpaceLHS = SolAddressSpace::threadgroup;
    descriptorThreadgroup.addressSpaceRHS = SolAddressSpace::threadgroup;
    if (preferAsyncCache && preferAsyncLoad) {
      return loopIteration(descriptorThreadgroup);
    }

    auto descriptorDevice = descriptor;
    if (preferAsyncCache) {
      descriptorDevice.addressSpaceLHS = SolAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceLHS = SolAddressSpace::device;
    }
    if (preferAsyncLoad) {
      descriptorDevice.addressSpaceRHS = SolAddressSpace::threadgroup;
    } else {
      descriptorDevice.addressSpaceRHS = SolAddressSpace::device;
    }

    auto blockDim = blockDimensions[1];
    CodeWriter source;
    source.SetValue("BLOCK_DIM", std::to_string(blockDim));
    source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
    source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("DESCRIPTOR_REGISTER_SIZE", std::to_string(descriptor.registerSize));
    source.SetValue("LOOP_ITERATION_DEVICE", loopIteration(descriptorDevice));
    source.SetValue("LOOP_ITERATION_THREADGROUP", loopIteration(descriptorThreadgroup));

    source += R"(

    if ((
          ({{TRAVERSAL_DIMENSION}} % {{BLOCK_DIM}} == 0) ||
          ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} <= {{TRAVERSAL_DIMENSION}})
        ) && (
          ({{HEAD_DIMENSION}} % 8 == 0) ||
          (d_outer + {{DESCRIPTOR_REGISTER_SIZE}} <= {{HEAD_DIMENSION}})
        )) {
      {{LOOP_ITERATION_DEVICE}}
    } else {
      {{LOOP_ITERATION_THREADGROUP}}
    }

)";
    return source.ToString();
  };

  // MARK: - Top Level Specification

  auto loopEnd = [=]() -> unsigned short { return paddedHeadDimensionValue(); };

  auto loopEndFloor = [=]() -> unsigned short { return loopEnd() - loopEnd() % blockDimensions[2]; };

  auto unrollStatement = [=]() -> std::string {
    return "#pragma clang loop unroll(full)";
  };

  auto registerOffset = [=]() -> std::string {
    return "d_outer";
  };

  auto firstIterations = [=]() -> std::string {
    LoopIterationDescriptor descriptor(SolAddressSpace::device, SolAddressSpace::device);
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = blockDimensions[2];
    CodeWriter source;
    source.SetValue("UNROLL_STATEMENT", unrollStatement());
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));

    source += R"(

    {{UNROLL_STATEMENT}}
    for (
      ushort d_outer = 0;
      d_outer < {{LOOP_END_FLOOR}};
      d_outer += {{BLOCK_DIMENSIONS_HEAD}}
    ) {
      {{GATED_LOOP_ITERATION}}
    }

)";
    return source.ToString();
  };

  auto lastIteration = [=]() -> std::string {
    LoopIterationDescriptor descriptor(SolAddressSpace::device, SolAddressSpace::device);
    descriptor.registerOffset = registerOffset();
    descriptor.registerSize = paddedHeadEdgeValue();

    CodeWriter source;
    source.SetValue("LOOP_END_FLOOR", std::to_string(loopEndFloor()));
    source.SetValue("LOOP_END_FLOOR_LESS_LOOP_END", (loopEndFloor() < loopEnd()) ? "true" : "false");
    source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration(descriptor));

    source += R"(

    if ({{LOOP_END_FLOOR_LESS_LOOP_END}}) {
      ushort d_outer = {{LOOP_END_FLOOR}};
      {{GATED_LOOP_ITERATION}}
    }

)";
    return source.ToString();
  };

  // Collect all of the statements into one string.
  return "\n" + firstIterations() + "\n" + lastIteration() + "\n";
}

std::string SolAttentionSource::storeOutput() const noexcept {
  const SolOperand operand = SolOperand::O;
  // MARK: - Operand

  auto asyncAccessOperand = [=]() -> std::string {
    CodeWriter source;
    source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
    source.SetValue("OPERAND", operand.name());
    source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
    source.SetValue("LEADING_BLOCK_DIMENSION_OPERAND", std::to_string(leadingBlockDimension(operand)));
    source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimension(operand));
    source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
    source.SetValue("PARALLELIZATION_GROUP_OFFSET", parallelizationGroupOffsetValue());
    source.SetValue("BLOCK_DIMENSIONS_PARALLELIZATION", std::to_string(blockDimensions[0]));
    if (disableAsyncCopy) {
      source.SetValue("ASYNC_LANE_ID", ", lane_id");
    } else {
      source.SetValue("ASYNC_LANE_ID", "");
    }
    source += R"(

      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 {{OPERAND}}_offset(d_outer, {{PARALLELIZATION_GROUP_OFFSET}});
        auto src = (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
        ::apply_offset(
          {{OPERAND_LOCATION}}, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_offset, {{TRANSPOSED_OPERAND}});

        ushort D_dimension = min(
          ushort({{BLOCK_DIMENSIONS_HEAD}}),
          ushort({{HEAD_DIMENSION}} - d_outer));
        ushort R_dimension = min(
          uint({{BLOCK_DIMENSIONS_PARALLELIZATION}}),
          uint({{PARALLELIZATION_DIMENSION}} - {{PARALLELIZATION_GROUP_OFFSET}}));
        ushort2 tile(D_dimension, R_dimension);

        simdgroup_event event;
        event.async_copy<{{LEADING_BLOCK_DIMENSION_OPERAND}}, 32>(
          dst, {{LEADING_DIMENSION_OPERAND}}, tile,
          src, tile{{ASYNC_LANE_ID}},
          {{TRANSPOSED_OPERAND}});
        simdgroup_event::wait(1, &event);
      }

)";
    return source.ToString();
  };

  struct LoopIterationDescriptor {
    SolAddressSpace addressSpace;
  };

  auto leadingDimensionOperand = [=](LoopIterationDescriptor descriptor) -> std::string {
    if (descriptor.addressSpace == SolAddressSpace::device) {
      return leadingDimension(operand);
    } else {
      return std::to_string(leadingBlockDimension(operand));
    }
  };

  auto declareOperandLocation = [=](LoopIterationDescriptor descriptor) -> std::string {
    if (descriptor.addressSpace == SolAddressSpace::device) {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("OPERAND_LOCATION", operandLocationValue(operand));
      source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimension(operand));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
      source += R"(

      uint2 {{OPERAND}}_src_offset(
        morton_offset.x + d_outer,
        {{CLAMPED_PARALLELIZATION_THREAD_OFFSET}});
      auto {{OPERAND}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
      ::apply_offset(
        {{OPERAND_LOCATION}}, {{LEADING_DIMENSION_OPERAND}},
        {{OPERAND}}_src_offset, {{TRANSPOSED_OPERAND}});

)";
      return source.ToString();
    } else {
      CodeWriter source;
      source.SetValue("MEMORY_NAME_OPERAND", memoryName(operand));
      source.SetValue("OPERAND", operand.name());
      source.SetValue("LEADING_BLOCK_DIMENSION_OPERAND", std::to_string(leadingBlockDimension(operand)));
      source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");
      source += R"(

      ushort2 {{OPERAND}}_block_offset(
        morton_offset.x,
        morton_offset.y + sidx * 8);
      auto {{OPERAND}}_src =
      (threadgroup {{MEMORY_NAME_OPERAND}}*)(threadgroup_block);

      {{OPERAND}}_src = simdgroup_matrix_storage<{{MEMORY_NAME_OPERAND}}>
      ::apply_offset(
        {{OPERAND}}_src, {{LEADING_BLOCK_DIMENSION_OPERAND}},
        {{OPERAND}}_block_offset, {{TRANSPOSED_OPERAND}});
      threadgroup_barrier(mem_flags::mem_threadgroup);

)";
      return source.ToString();
    }
  };

  // MARK: - Inner Loop

  auto innerLoopHead = [=](unsigned short headStart, unsigned short headEnd,
                           LoopIterationDescriptor descriptor) -> std::string {
    CodeWriter source;
    source.SetValue("HEAD_START", std::to_string(headStart));
    source.SetValue("HEAD_END", std::to_string(headEnd));
    source.SetValue("OPERAND", operand.name());
    source.SetValue("LEADING_DIMENSION_OPERAND", leadingDimensionOperand(descriptor));
    source.SetValue("TRANSPOSED_OPERAND", transposed(operand) ? "true" : "false");

    source.SetValue("STORE_FUNCTION_OPERAND", storeFunction(operand));
    source += R"(

      #pragma clang loop unroll(full)
      for (ushort d = {{HEAD_START}}; d < {{HEAD_END}}; d += 8) {
        ushort2 {{OPERAND}}_origin(d, 0);
        {{OPERAND}}_sram[(d_outer + d) / 8].{{STORE_FUNCTION_OPERAND}}(
          {{OPERAND}}_src, {{LEADING_DIMENSION_OPERAND}},
          {{OPERAND}}_origin, {{TRANSPOSED_OPERAND}});
      }

)";

    return source.ToString();
  };

  // MARK: - Outer Loop

  auto loopIteration = [=](LoopIterationDescriptor descriptor) -> std::string {
    auto loadOperand = [=]() -> std::string {
      return "";
    };

    auto storeOperand = [=]() -> std::string {
      return asyncAccessOperand();
    };

    if (descriptor.addressSpace == SolAddressSpace::device) {
      CodeWriter source;
      source.SetValue("DECLARE_OPERAND_LOCATION", declareOperandLocation(descriptor));
      source.SetValue("TYPE_IS_LOAD", "false");
      source.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
      source.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
      source.SetValue("INNER_LOOP_HEAD", innerLoopHead(0, blockDimensions[2], descriptor));
      source += R"(

      {{DECLARE_OPERAND_LOCATION}}
      if (
        {{TYPE_IS_LOAD}} ||
        ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}})
      ) {
      {{INNER_LOOP_HEAD}}
      }

)";
      return source.ToString();
    } else {
      CodeWriter source;
      source.SetValue("LOAD_OPERAND", loadOperand());
      source.SetValue("DECLARE_OPERAND_LOCATION", declareOperandLocation(descriptor));
      source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
      source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
      source.SetValue("INNER_LOOP_HEAD_0", innerLoopHead(0, blockDimensions[2], descriptor));
      source.SetValue("INNER_LOOP_HEAD_1", innerLoopHead(0, headDimension % blockDimensions[2], descriptor));
      source.SetValue("STORE_OPERAND", storeOperand());
      source += R"(

      {{LOAD_OPERAND}}
      {{DECLARE_OPERAND_LOCATION}}
      if (d_outer + {{BLOCK_DIMENSIONS_HEAD}} <= {{HEAD_DIMENSION}}) {
        {{INNER_LOOP_HEAD_0}}
      } else {
        {{INNER_LOOP_HEAD_1}}
      }
      {{STORE_OPERAND}}

)";
      return source.ToString();
    }
  };

  auto gatedLoopIteration = [=]() -> std::string {
    LoopIterationDescriptor descriptorDevice;
    LoopIterationDescriptor descriptorThreadgroup;
    descriptorDevice.addressSpace = SolAddressSpace::device;
    descriptorThreadgroup.addressSpace = SolAddressSpace::threadgroup;
    CodeWriter source;
    source.SetValue("NOT_PREFER_ASYNC_CACHE", !preferAsyncCache ? "true" : "false");
    source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
    source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
    source.SetValue("LOOP_ITERATION_DEVICE", loopIteration(descriptorDevice));
    source.SetValue("LOOP_ITERATION_THREADGROUP", loopIteration(descriptorThreadgroup));

    source += R"(

    if ({{NOT_PREFER_ASYNC_CACHE}} && (
      ({{HEAD_DIMENSION}} % {{BLOCK_DIMENSIONS_HEAD}} == 0) ||
      (d_outer + {{BLOCK_DIMENSIONS_HEAD}} <= {{HEAD_DIMENSION}})
    )) {
      {{LOOP_ITERATION_DEVICE}}
    } else {
      {{LOOP_ITERATION_THREADGROUP}}
    }
)";
    return source.ToString();
  };

  CodeWriter source;
  source.SetValue("ALLOCATE_OPERAND", "");
  source.SetValue("HEAD_DIMENSION", std::to_string(headDimension));
  source.SetValue("BLOCK_DIMENSIONS_HEAD", std::to_string(blockDimensions[2]));
  source.SetValue("GATED_LOOP_ITERATION", gatedLoopIteration());
  source += R"(

  {{ALLOCATE_OPERAND}}

  #pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < {{HEAD_DIMENSION}};
    d_outer += {{BLOCK_DIMENSIONS_HEAD}}
  ) {
    {{GATED_LOOP_ITERATION}}
  }

)";
  return source.ToString();
}

std::string SolAttentionSource::createSetup() const noexcept {
  // Allocate registers for the specified operand.
  auto allocate = [=](SolOperand operand) -> std::string {
    CodeWriter source;
    source.SetValue("REGISTER_NAME_OPERAND", registerName(operand));
    source.SetValue("OPERAND", operand.name());
    source.SetValue("PADDED_HEAD_DIMENSION_8", std::to_string(paddedHeadDimensionValue() / 8));
    source += R"(

    simdgroup_matrix_storage<{{REGISTER_NAME_OPERAND}}> {{OPERAND}}_sram[{{PADDED_HEAD_DIMENSION_8}}];

)";
    return source.ToString();
  };

  // Initialize the output string.
  CodeWriter output;

  output += allocate(SolOperand::O);
  output += R"(

    float m = -numeric_limits<float>::max();
    float l = numeric_limits<float>::denorm_min();

)";
  return output.ToString();
}

std::string SolAttentionSource::createCleanup() const noexcept {
  // Initialize the output string.
  CodeWriter output;

  output += storeOutput();

  // L is always either FP16 or FP32, so we don't need custom type
  // conversion code here.
  output.SetValue("L_LOCATION", operandLocationValue(SolOperand::L));
  output.SetValue("UNSAFE_PARALLELIZATION_THREAD_OFFSET", unsafeParallelizationThreadOffsetValue());
  output.SetValue("PARALLELIZATION_DIMENSION", parallelizationDimensionValue());
  output.SetValue("CLAMPED_PARALLELIZATION_THREAD_OFFSET", clampedParallelizationThreadOffsetValue());
  output += R"(

    if ({{UNSAFE_PARALLELIZATION_THREAD_OFFSET}} < {{PARALLELIZATION_DIMENSION}}) {
      // Premultiplied by log_base_2(e).
      float L_sram = m + fast::log2(l);
      ({{L_LOCATION}})[{{CLAMPED_PARALLELIZATION_THREAD_OFFSET}}] = L_sram;
    }

)";
  return output.ToString();
}

std::string SolAttentionSource::maskAttentionMatrixEdge() const noexcept {
  auto blockDim = blockDimensions[1];
  std::string remainder = "(" + traversalDimensionValue() + " % " + std::to_string(blockDim) + ")";
  std::string remainderFloor = "(" + remainder + " - (" + remainder + " % 8))";
  float logBase2E = 1.442695041;

  CodeWriter source;
  source.SetValue("REMAINDER", remainder);
  source.SetValue("REMAINDER_FLOOR", remainderFloor);
  source.SetValue("TRAVERSAL_OFFSET", traversalOffsetValue());
  source.SetValue("BLOCK_DIM", std::to_string(blockDim));
  source.SetValue("TRAVERSAL_DIMENSION", traversalDimensionValue());
  source.SetValue("LOG_BASE_2E", std::to_string(logBase2E));
  source.SetValue("REGISTER_NAME_S", registerName(SolOperand::S));
  source += R"(

  if (({{REMAINDER}} != 0) &&
      ({{TRAVERSAL_OFFSET}} + {{BLOCK_DIM}} > {{TRAVERSAL_DIMENSION}})) {
    // Prevent the value from becoming -INF during the FMA before the
    // exponentiation. If the multiplication during FMA returns -INF,
    // subtracting a positive 'm' value will turn it into zero. We don't want
    // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
    const {{REGISTER_NAME_S}} mask_value =
    (0.875 / {{LOG_BASE_2E}}) * -numeric_limits<{{REGISTER_NAME_S}}>::max();

    #pragma clang loop unroll(full)
    for (ushort index = 0; index < 2; ++index) {
      if (morton_offset.x + index >= {{REMAINDER}} - {{REMAINDER_FLOOR}}) {
        auto S_elements = S_sram[{{REMAINDER_FLOOR}} / 8].thread_elements();
        (*S_elements)[index] = mask_value;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort c = {{REMAINDER_FLOOR}} + 8; c < {{BLOCK_DIM}}; c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      *S_elements = mask_value;
    }
  }

)";
  return source.ToString();
}

std::string SolAttentionSource::onlineReduceMaximum() const noexcept {
  CodeWriter source;
  source.SetValue("REGISTER_NAME_S", registerName(SolOperand::S));
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source.SetValue("M_NEW_SCALE", "m_new *= dot_product_scale;");
  source += R"(

  // update 'm'
  vec<{{REGISTER_NAME_S}}, 2> m_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
    auto S_elements = S_sram[c / 8].thread_elements();
    if (c == 0) {
      m_new_accumulator = *S_elements;
    } else {
      m_new_accumulator = max(m_new_accumulator, *S_elements);
    }
  }
  float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
  m_new = max(m_new, simd_shuffle_xor(m_new, 1));
  m_new = max(m_new, simd_shuffle_xor(m_new, 8));
  {{M_NEW_SCALE}}

)";
  return source.ToString();
}

std::string SolAttentionSource::onlineCorrectO() const noexcept {
  return R"(

  // update 'O'
  float correction = 1;
  if (m_new > m) {
    correction = fast::exp2(m - m_new);
    m = m_new;
  }

)";
}

std::string SolAttentionSource::onlineReduceSum() const noexcept {
  CodeWriter source;
  source.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  source += R"(

  // update 'l'
  float2 l_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
    auto P_elements = P_sram[c / 8].thread_elements();
    if (c == 0) {
      l_new_accumulator = float2(*P_elements);
    } else {
      l_new_accumulator += float2(*P_elements);
    }
  }
  float l_new = l_new_accumulator[0] + l_new_accumulator[1];
  l_new += simd_shuffle_xor(l_new, 1);
  l_new += simd_shuffle_xor(l_new, 8);
  l = l * correction + l_new;

)";
  return source.ToString();
}

std::string SolAttentionSource::softmax() const noexcept {
  CodeWriter allocation, elements, loop, source;
  allocation.SetValue("REGISTER_NAME_P", "half");
  allocation.SetValue("BLOCK_DIM", std::to_string(blockDimensions[1]));
  allocation += R"(

      simdgroup_matrix_storage<{{REGISTER_NAME_P}}> P_sram[{{BLOCK_DIM}} / 8];

)";
  elements.SetValue("REGISTER_NAME_P", "half");
  elements.SetValue("SCALE", "dot_product_scale");
  elements += R"(

      auto S = *(S_sram[c / 8].thread_elements());
      auto P = vec<{{REGISTER_NAME_P}}, 2>(
        fast::exp2(float2(S) * {{SCALE}} - float2(L_elements)));
      *(P_sram[c / 8].thread_elements()) = P;

)";
  loop.SetValue("BLOCK_DIMENSIONS_TRAVERSAL", std::to_string(blockDimensions[1]));
  loop.SetValue("OVERWRITE_ATTENTION_MATRIX_ELEMENTS", elements.ToString());
  loop += R"(

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < {{BLOCK_DIMENSIONS_TRAVERSAL}}; c += 8) {
        auto L_elements = m;
        {{OVERWRITE_ATTENTION_MATRIX_ELEMENTS}}
      }

)";
  source.SetValue("ALLOCATE_OUTPUT", allocation.ToString());
  source.SetValue("INNER_LOOP", loop.ToString());
  source += R"(

    {{ALLOCATE_OUTPUT}}
    {
      {{INNER_LOOP}}
    }

)";
  return source.ToString();
}

// Sol's diagonal-threshold / pooled-block approximation. The byte route map
// stores one decision per query group / key block, never a T x T score matrix.
static const char* solPreparationSource = R"METAL(
struct SolAttentionParams {
  uint N, T, H, B, begin, end;
  float scale, tau;
  uint QB;
  uint local_block_radius;
};

kernel void sol_pool(device const half* Q [[buffer(0)]],
  device const half* K [[buffer(1)]], device const half* V [[buffer(2)]],
  device float* QC [[buffer(4)]], device half* KC [[buffer(5)]],
  device half* VC [[buffer(6)]], constant SolAttentionParams& p [[buffer(10)]],
  uint d [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint J = (C + SOL_BLOCK_SIZE - 1) / SOL_BLOCK_SIZE;
  const uint j = group.x, nh = group.y, n = nh / Hq, h = nh % Hq;
  const uint JP = (J + 63) / 64 * 64;
  if (j >= J) {
    KC[(ulong(nh) * JP + j) * 128 + d] = 0;
    VC[(ulong(nh) * JP + j) * 128 + d] = 0;
    return;
  }
  const uint count = min(uint(SOL_BLOCK_SIZE), C - j * SOL_BLOCK_SIZE);
  const uint QJ = (C + SOL_QB - 1) / SOL_QB;
  float q[4] = {0, 0, 0, 0}; float k = 0, v = 0;
  for (uint t = j * SOL_BLOCK_SIZE; t < j * SOL_BLOCK_SIZE + count; ++t) {
    const ulong index = ((ulong(n) * C + t) * Hq + h) * 128 + d;
    q[(t - j * SOL_BLOCK_SIZE) / SOL_QB] += float(Q[index]); k += float(K[index]); v += float(V[index]);
  }
  for (uint part = 0; part < SOL_BLOCK_SIZE / SOL_QB; ++part) {
    const uint qb = j * SOL_BLOCK_SIZE / SOL_QB + part;
    if (qb < QJ) QC[(ulong(nh) * QJ + qb) * 128 + d] = q[part] / min(SOL_QB, C - qb * SOL_QB);
  }
  KC[(ulong(nh) * JP + j) * 128 + d] = half(k / count);
  VC[(ulong(nh) * JP + j) * 128 + d] = half(v / count);
}

kernel void sol_stats(device const half* KC [[buffer(5)]],
  device float2* stats [[buffer(7)]], constant SolAttentionParams& p [[buffer(10)]],
  uint d [[thread_index_in_threadgroup]], uint nh [[threadgroup_position_in_grid]]) {
  const uint J = (C + SOL_BLOCK_SIZE - 1) / SOL_BLOCK_SIZE;
  float m = 0, v = 0;
  for (uint j = 0; j < J; ++j) {
    const float k = float(KC[(ulong(nh) * ((J + 63) / 64 * 64) + j) * 128 + d]);
    m += k; v += k * k;
  }
  m /= J;
  stats[nh * 128 + d] = float2(m, max(0.0f, v / J - m * m));
}

kernel void sol_route(device const float* QC [[buffer(4)]],
  device const half* KC [[buffer(5)]], device const float2* stats [[buffer(7)]],
  device uchar* routes [[buffer(8)]], constant SolAttentionParams& p [[buffer(10)]],
  device uint* counts [[buffer(29)]],
  device uint* route_bits [[buffer(30), function_constant(SOL_ROUTE_BITS)]],
  uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
  uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  const uint J = (C + SOL_BLOCK_SIZE - 1) / SOL_BLOCK_SIZE, qb = group.x, nh = group.y;
  const uint QJ = (C + SOL_QB - 1) / SOL_QB;
  QC += (ulong(nh) * QJ + qb) * 128;
  KC += ulong(nh) * ((J + 63) / 64 * 64) * 128;
  routes += (ulong(nh) * QJ + qb) * J;
  const float g = p.scale * 1.4426950408889634f;
  float mu = 0, variance = 0;
  // Each SIMD group derives the same threshold independently.
  for (uint d = lane; d < 128; d += 32) {
    const float q = QC[d];
    const float2 s = stats[nh * 128 + d];
    mu += q * s.x; variance += q * q * s.y;
  }
  const float threshold = g * simd_sum(mu) + p.tau * sqrt(g * g * simd_sum(variance) + 1e-6f);
  const bool exact_query = qb * SOL_QB < p.begin || min((qb + 1) * SOL_QB, C) > p.end;
  for (uint j = sg; j < J; j += 4) {
    float dot = 0;
    for (uint d = lane; d < 128; d += 32) dot += QC[d] * float(KC[j * 128 + d]);
    const float score = simd_sum(dot) * g;
    if (lane == 0) routes[j] = exact_query || j * SOL_BLOCK_SIZE < p.begin || min((j + 1) * SOL_BLOCK_SIZE, C) > p.end || abs(int(qb * SOL_QB / SOL_BLOCK_SIZE) - int(j)) <= p.local_block_radius || score > threshold;
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (SOL_ROUTE_BITS) {
    const uint words = (J + 31) / 32;
    for (uint word = tid; word < words; word += 128) {
      uint selected = 0;
      for (uint bit = 0; bit < 32 && word * 32 + bit < J; ++bit)
        selected |= uint(routes[word * 32 + bit] != 0) << bit;
      // The device barrier above completes this group's QC reads. Reuse only
      // this query's 128-float row, so other groups' QC reads cannot race.
      route_bits[(ulong(nh) * QJ + qb) * 128 + word] = selected;
    }
  }
  if (tid == 0) {
    uint count = 0;
    for (uint j = 0; j < J; ++j) count += routes[j] != 0;
    counts[ulong(nh) * QJ + qb] = count;
  }
}

)METAL";

std::string createSolSource(SolAttentionSource& dense, SolAttentionSource& sparse, uint32_t blockSize, bool useRouteBits) {
  const auto blockDimensions = dense.blockDimensions;
  const auto threadgroupSize = dense.threadgroupSize;
  CodeWriter code;
  code.SetValue("B", std::to_string(blockSize));
  code.SetValue("BR", std::to_string(blockDimensions[0]));
  code += createMetalSimdgroupEvent(dense.disableAsyncCopy) + "\n";
  code += createMetalSimdgroupMatrixStorage(false) + "\nusing namespace metal;\n";
  code += dense.createConstants();
  code += R"(
constant uint SOL_QB [[function_constant(28)]];
constant uint SOL_BLOCK_SIZE [[function_constant(29)]];
constant bool SOL_ROUTE_BITS [[function_constant(31)]];
constant uint SOL_J = (C + {{B}} - 1) / {{B}};
constant uint SOL_JP = (SOL_J + 63) / 64 * 64;
)";
  code += solPreparationSource;
  const std::string preamble = code.ToString();
  code = CodeWriter();
  code.SetValue("B", std::to_string(blockSize));
  code.SetValue("BR", std::to_string(blockDimensions[0]));
  code += dense.createBufferBindings();
  code += R"(
  device half* KC [[buffer(5)]],
  device half* VC [[buffer(6)]],
  device const uchar* routes [[buffer(8)]],
  device const uint* counts [[buffer(9)]],
  device const uint* route_bits [[buffer(10), function_constant(SOL_ROUTE_BITS)]],
  threadgroup uchar* threadgroup_block [[threadgroup(0)]],
  uint3 gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]) {
  ushort2 morton_offset = morton_order(lane_id);
  const uint tiles = (R + {{BR}} - 1) / {{BR}};
  gid = {gid.x % tiles, gid.x / tiles % Hq, gid.x / (tiles * Hq)};
  uint parallelization_group_offset = gid.x;
  parallelization_group_offset *= {{BR}};
  if (parallelization_group_offset >= R) return;
  const uint nh = gid.z * Hq + gid.y;
  const uint route_row = nh * ((R + SOL_QB - 1) / SOL_QB) + parallelization_group_offset / SOL_QB;
  routes += ulong(route_row) * SOL_J;
  const uniform<bool> all_exact = make_uniform(counts[route_row] == SOL_J);
)";
  const std::string functionHead = code.ToString();
  code = CodeWriter();
  code.SetValue("B", std::to_string(blockSize));
  code.SetValue("BR", std::to_string(blockDimensions[0]));
  code += preamble + "kernel void sol_attention_dense(\n" + functionHead;
  code += "if (!(all_exact && dot_product_scale > 0)) return;\n";
  // Default K/V use the public NHWC layout, just like native attention.
  auto adjustOffsets = [](SolAttentionSource& kernel) {
    std::string text;
    for (SolOperand op : {SolOperand::Q, SolOperand::O, SolOperand::L})
      text += op.name() + " = " + kernel.operandLocationWithHeadOffsetValue(op) + ";\n";
    for (SolOperand op : {SolOperand::K, SolOperand::V})
      text += op.name() + " = " + kernel.operandLocationWithHeadOffsetValue(op) + ";\n";
    return text;
  };
  // A query tile stays in threadgroup memory throughout traversal. K/V always
  // retain their input layout on the direct path; no full-size copies are needed.
  auto queryTile = [&](SolAttentionSource& kernel) -> std::string {
    CodeWriter q;
    q.SetValue("BR", std::to_string(blockDimensions[0]));
    q.SetValue("THREADS", std::to_string(threadgroupSize));
    q.SetValue("Q_OFFSET", std::to_string(kernel.threadgroupMemoryAllocation));
    q += R"(
  threadgroup half* Q_tile = (threadgroup half*)(threadgroup_block + {{Q_OFFSET}});
  for (uint i = (sidx * 32 + lane_id) * 4; i < {{BR}} * 128; i += {{THREADS}} * 4) {
    const uint row = parallelization_group_offset + i / 128;
    ((threadgroup half4*)Q_tile)[i / 4] = row < R ?
      *((device half4*)(Q + ulong(row) * Q_leading_dimension + i % 128)) : half4(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    return q.ToString();
  };
  // Separate the full and partial tile loads outside the channel loops. A
  // per-element tail branch here substantially slows unaligned sequence lengths.
  auto qk = [&](SolAttentionSource& kernel) -> std::string {
    CodeWriter q;
    q.SetValue("BC", std::to_string(kernel.blockDimensions[1]));
    q.SetValue("BD", std::to_string(kernel.blockDimensions[2]));
    // Unroll the dense C128 dot product; the mixed path keeps a smaller body.
    q.SetValue("UNROLL_D", kernel.blockDimensions[1] == 128 ? "full" : "disable");
    q += R"(
  simdgroup_matrix_storage<half> S_sram[{{BC}} / 8];
  #pragma clang loop unroll(full)
  for (ushort x = 0; x < {{BC}} / 8; ++x) *S_sram[x].thread_elements() = half2(0);
  const uniform<bool> full_tile = make_uniform((C % {{BC}} == 0) || c + {{BC}} <= C);
)";
    for (bool full : {true, false}) {
      q += full ? "if (full_tile) {\n" : "} else {\n";
      q += R"(
  #pragma clang loop unroll({{UNROLL_D}})
  for (ushort d_outer = 0; d_outer < 128; d_outer += {{BD}}) {
    simdgroup_matrix_storage<half> Q_sram[{{BD}} / 8];
    auto q_src = Q_tile + (sidx * 8 + morton_offset.y) * 128 + morton_offset.x + d_outer;
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{BD}}; d += 8) Q_sram[d / 8].load(q_src, 128, ushort2(d, 0), false);
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < {{BD}}; d += 8) {
      #pragma clang loop unroll(full)
      for (ushort x = 0; x < {{BC}}; x += 8) {
        simdgroup_matrix_storage<half> key;
)";
      q += full ? R"(
        auto k_src = K + (c + morton_offset.x) * K_leading_dimension + d_outer + morton_offset.y;
        key.load(k_src, K_leading_dimension, ushort2(x, d), true);
)"
                : R"(
        const uint row = c + x + morton_offset.x;
        const uint channel = d_outer + d + morton_offset.y;
        *key.thread_elements() = half2(row < C ? K[row * K_leading_dimension + channel] : half(0),
          row + 1 < C ? K[(row + 1) * K_leading_dimension + channel] : half(0));
)";
      q += R"(
        S_sram[x / 8].multiply(Q_sram[d / 8], key);
      }
    }
  }
)";
    }
    q += "}\n";
    return q.ToString();
  };
  std::string denseLoop;
  {
    SolAccumulate pv(SolOperand::P, SolOperand::V, SolOperand::O, "correction", "");
    const auto step = qk(dense) + dense.maskAttentionMatrixEdge() + dense.onlineReduceMaximum() +
                      dense.onlineCorrectO() + dense.softmax() + dense.onlineReduceSum() + dense.accumulate(pv);
    CodeWriter loop;
    loop.SetValue("BC", std::to_string(dense.blockDimensions[1]));
    loop += R"(
  const uint full_C = C / {{BC}} * {{BC}};
  {
    const uint C = full_C;
    for (uint c = 0; c < C; c += {{BC}}) {
)";
    loop += step;
    // Keep the two SIMD groups near the same K/V tile during long traversals.
    loop += "if ((c / {{BC}}) % 2 == 0) threadgroup_barrier(mem_flags::mem_none);\n";
    loop += R"(
    }
  }
  if (full_C < C) {
    const uint c = full_C;
)";
    loop += step;
    loop += R"(
  }
  #pragma clang loop unroll(full)
  for (ushort d = 0; d < 16; ++d)
    *O_sram[d].thread_elements() *= fast::divide(1.0f, l);
)";
    denseLoop = loop.ToString();
  }
  code += adjustOffsets(dense) + queryTile(dense);
  code += dense.createSetup() + denseLoop + dense.createCleanup();
  code += "}\n";
  code += "kernel void sol_attention(\n" + functionHead;
  code += "if (all_exact && dot_product_scale > 0) return;\n";
  code += adjustOffsets(sparse) + queryTile(sparse);
  code += sparse.createSetup();
  code += "bool has_accumulated = false;\n";

  auto tile = [&](bool summary) {
    CodeWriter body;
    body.SetValue("B", std::to_string(blockSize));
    body += qk(sparse);
    // Scale in FP32 before reducing the maximum. This also handles signed/zero
    // scales and fully masked summary tiles without 0 * infinity or NaNs.
    body += R"(
  float2 logits[{{B}} / 8];
  float m_new = m;
  #pragma clang loop unroll(full)
  for (ushort x = 0; x < {{B}}; x += 8) {
    const uint column = c + x + morton_offset.x;
    logits[x / 8] = float2(*S_sram[x / 8].thread_elements()) * dot_product_scale;
    for (ushort i = 0; i < 2; ++i) {
)";
    body += summary ? R"(
      const bool valid = column + i < C && !routes[column + i];
)"
                    : R"(
      const bool valid = column + i < C;
)";
    body += R"(
      if (!valid) logits[x / 8][i] = -INFINITY;
      m_new = max(m_new, logits[x / 8][i]);
    }
  }
  m_new = max(m_new, simd_shuffle_xor(m_new, 1));
  m_new = max(m_new, simd_shuffle_xor(m_new, 8));
)";
    body += sparse.onlineCorrectO();
    body += R"(
  simdgroup_matrix_storage<half> P_sram[{{B}} / 8];
  #pragma clang loop unroll(full)
  for (ushort x = 0; x < {{B}}; x += 8) {
    float2 probability = fast::exp2(logits[x / 8] - m);
)";
    if (summary)
      body += R"(
    const uint column = c + x + morton_offset.x;
    for (ushort i = 0; i < 2; ++i)
      probability[i] *= column + i < C ? min(uint({{B}}), R - (column + i) * {{B}}) : 0;
)";
    body += R"(
    *P_sram[x / 8].thread_elements() = half2(probability);
  }
)";
    body += sparse.onlineReduceSum();
    SolAccumulate pv(SolOperand::P, SolOperand::V, SolOperand::O, "correction", "");
    pv.firstIteration = "!has_accumulated";
    body += sparse.accumulate(pv);
    body += "has_accumulated = true;\n";
    return body.ToString();
  };
  code += R"(
  if (!all_exact) {
    // Pooled keys / values have a compact per-head layout. Shadowing the
    // sequence and leading dimensions reuses the Sol SIMD loaders.
    const uint C = SOL_J;
    const uint K_leading_dimension = 128, V_leading_dimension = 128;
    device half* K = KC + ulong(nh) * SOL_JP * 128;
    device half* V = VC + ulong(nh) * SOL_JP * 128;
    for (uint c = 0; c < C; c += {{B}}) {
)";
  code += tile(true);
  code += "}\n}\n";
  if (useRouteBits)
    code += R"(
  const uint words = (SOL_J + 31) / 32;
  route_bits += ulong(route_row) * 128;
  uint visited_blocks = 0;
  for (uint word = 0; word < words; ++word) {
    uint selected = route_bits[word];
    while (selected != 0) {
      const uint c = (word * 32 + ctz(selected)) * {{B}};
      selected &= selected - 1;
)";
  else
    code += R"(
  for (uint c = 0; c < C; c += {{B}}) {
    if (!routes[c / {{B}}]) continue;
)";
  code += tile(false);
  if (useRouteBits)
    code += R"(
      // Keep SIMD groups near the same K/V block on long mixed traversals.
      // Every thread follows this query group's route, including padded rows.
      if ((++visited_blocks % 2) == 0)
        threadgroup_barrier(mem_flags::mem_none);
    }
)";
  code += R"(
  }
  #pragma clang loop unroll(full)
  for (ushort d = 0; d < 16; ++d)
    *O_sram[d].thread_elements() *= fast::divide(1.0f, l);
)";
  code += sparse.createCleanup();
  code += "}\n";
  return code.ToString();
}

} // namespace

SolAttentionKernel::SolAttentionKernel(MTL::Device* device, uint32_t blockSize, bool useRouteBits) {
  const uint16_t rows = 16;
  const bool apple9 = device->supportsFamily(MTL::GPUFamily(1009));
  SolAttentionSource dense(rows, 128, apple9);
  SolAttentionSource sparse(rows, blockSize, apple9);
  blockDimensions = dense.blockDimensions;
  threadgroupSize = dense.threadgroupSize;
  // Keep Q beyond the Sol loaders' temporary storage. Dense and mixed
  // kernels reserve their own sizes so mixed occupancy is not limited by C128.
  const uint16_t queryBytes = blockDimensions[0] * 128 * sizeof(uint16_t);
  threadgroupMemoryAllocation = sparse.threadgroupMemoryAllocation + queryBytes;
  denseThreadgroupMemoryAllocation = dense.threadgroupMemoryAllocation + queryBytes;
  source = createSolSource(dense, sparse, blockSize, useRouteBits);
  NS::Error* error = nullptr;
  library =
      NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr, &error));
  // Newer shader compilers reject the private async-copy intrinsic. Native
  // attention may use a precompiled library; Sol retains a portable load path.
  if (!library) {
    dense.preferAsyncCache = dense.preferAsyncLoad = false;
    sparse.preferAsyncCache = sparse.preferAsyncLoad = false;
    dense.disableAsyncCopy = sparse.disableAsyncCopy = true;
    source = createSolSource(dense, sparse, blockSize, useRouteBits);
    error = nullptr;
    library = NS::TransferPtr(
        device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr, &error));
  }
  CCV_NNC_MFA_CHECK_ERROR(error);
}

bool SolAttentionDescriptor::operator==(const SolAttentionDescriptor& r) const {
  return N == r.N && T == r.T && H == r.H && blockSize == r.blockSize && queryBlockSize == r.queryBlockSize &&
         scale == r.scale && useRouteBits == r.useRouteBits;
}
size_t std::hash<SolAttentionDescriptor>::operator()(const SolAttentionDescriptor& d) const noexcept {
  size_t h = std::hash<float>{}(d.scale);
  for (uint32_t v : {d.N, d.T, d.H, d.blockSize, d.queryBlockSize})
    h = h * 31 + v;
  h = h * 31 + d.useRouteBits;
  return h;
}
std::pair<SolAttentionKernelKey, PipelineValue<SolAttentionKernel>*> SolAttentionDescriptor::findKernel(
    MTL::Device* device, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&,
    std::unordered_map<SolAttentionKernelKey, std::unique_ptr<SolAttentionKernel>>* cache) const noexcept {
  const auto key = SolAttentionKernelKey(blockSize | uint32_t(useRouteBits) << 8);
  auto& kernel = (*cache)[key];
  if (!kernel)
    kernel = std::make_unique<SolAttentionKernel>(device, blockSize, useRouteBits);
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t ratio = 1, stride = T * H * 128, leading = H * 128;
  constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&T, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&H, MTL::DataTypeUInt, 2);
  constants->setConstantValue(&ratio, MTL::DataTypeUInt, 3);
  constants->setConstantValue(&scale, MTL::DataTypeFloat, 4);
  for (uint32_t i = 0; i < 4; ++i) {
    constants->setConstantValue(&stride, MTL::DataTypeUInt, 5 + i);
    constants->setConstantValue(&leading, MTL::DataTypeUInt, 15 + i);
  }
  constants->setConstantValue(&queryBlockSize, MTL::DataTypeUInt, 28);
  constants->setConstantValue(&blockSize, MTL::DataTypeUInt, 29);
  constants->setConstantValue(&useRouteBits, MTL::DataTypeBool, 31);
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(kernel->library->newFunction(
      NS::String::string("sol_attention", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto mixedDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  mixedDescriptor->setComputeFunction(function.get());
  mixedDescriptor->setMaxTotalThreadsPerThreadgroup(256);
  auto pipeline =
      NS::TransferPtr(device->newComputePipelineState(mixedDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto denseFunction = NS::TransferPtr(kernel->library->newFunction(
      NS::String::string("sol_attention_dense", NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipelineDescriptor = NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  pipelineDescriptor->setComputeFunction(denseFunction.get());
  pipelineDescriptor->setMaxTotalThreadsPerThreadgroup(256);
  auto densePipeline = NS::TransferPtr(
      device->newComputePipelineState(pipelineDescriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto value = new PipelineValue<SolAttentionKernel>{kernel.get(), pipeline};
  value->second = densePipeline;
  return {key, value};
}

bool SolAttentionPreparationDescriptor::operator==(const SolAttentionPreparationDescriptor& rhs) const {
  return useRouteBits == rhs.useRouteBits && entry == rhs.entry && blockSize == rhs.blockSize &&
    N == rhs.N && T == rhs.T && H == rhs.H && queryBlockSize == rhs.queryBlockSize;
}
std::pair<SolAttentionKernelKey, PipelineValue<SolAttentionKernel>*> SolAttentionPreparationDescriptor::findKernel(MTL::Device* device, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&, std::unordered_map<SolAttentionKernelKey, std::unique_ptr<SolAttentionKernel>>* cache) const noexcept {
  const auto key = SolAttentionKernelKey(blockSize | uint32_t(useRouteBits) << 8);
  auto& kernel = (*cache)[key];
  if (!kernel) kernel = std::make_unique<SolAttentionKernel>(device, blockSize, useRouteBits);
  const char* names[] = { "sol_pool", "sol_stats", "sol_route" };
  // Shapes specialize pipelines, not the source / kernel-object cache.
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&T, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&H, MTL::DataTypeUInt, 2);
  constants->setConstantValue(&queryBlockSize, MTL::DataTypeUInt, 28);
  constants->setConstantValue(&blockSize, MTL::DataTypeUInt, 29);
  constants->setConstantValue(&useRouteBits, MTL::DataTypeBool, 31);
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string(names[entry], NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return { key, new PipelineValue<SolAttentionKernel>{ kernel.get(), pipeline } };
}

size_t std::hash<SolAttentionPreparationDescriptor>::operator()(const SolAttentionPreparationDescriptor& d) const noexcept {
  size_t hash = d.entry + (d.blockSize << 8);
  for (uint32_t dimension : { d.N, d.T, d.H, d.queryBlockSize, uint32_t(d.useRouteBits) })
    hash = hash * 31 + dimension;
  return hash;
}
