#include "GEMMKernel.hpp"
#include "GEMMHeaders.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

GEMMKernel::GEMMKernel(GEMMKernelDescriptor descriptor, MTL::Device *const device) {
  auto blockDimensions = descriptor.blockDimensions;
  auto memoryPrecisions = descriptor.memoryPrecisions;
  auto preferAsyncStore = descriptor.preferAsyncStore;
  auto registerPrecisions = descriptor.registerPrecisions;
  auto splits = descriptor.splits;
  auto transposeState = descriptor.transposeState;
  auto useBias = descriptor.useBias;
  this->blockDimensions = blockDimensions;
  this->threadgroupSize = 32 * splits[0] * splits[1];
  
  // Validate the correctness of register precisions.
  auto checkOperandPair =
  [=](GEMMOperandPrecision memory, GEMMOperandPrecision register_) -> bool {
    // Truth table:
    //
    // memory | register | valid |
    // ------ | -------- | ----- |
    // FP32   | FP32     | yes   |
    // FP32   | FP16     | no    |
    // FP32   | BF16     | no    |
    // FP16   | FP32     | yes   |
    // FP16   | FP16     | yes   |
    // FP16   | BF16     | no    |
    // BF16   | FP32     | yes   |
    // BF16   | FP16     | no    |
    // BF16   | BF16     | yes   |
    //
    // Optimized form of the logic:
    //
    // If the register precision matches the memory precision,
    //   return true
    // If the register precision equals FP32,
    //   return true
    // Otherwise,
    //   return false
    //
    // The logic statements will change if you introduce custom quantized
    // formats. The truth table will grow exponentially. You'll need to add
    // more restrictions on accepted pairs to overcome the combinatorial
    // explosion.
    if (register_ == memory) {
      return true;
    } else if (register_.value == GEMMOperandPrecision::FP32) {
      return true;
    } else {
      return false;
    }
  };
  
  CCV_NNC_MFA_PRECONDITION
  (checkOperandPair(memoryPrecisions.A, registerPrecisions.A));
  CCV_NNC_MFA_PRECONDITION
  (checkOperandPair(memoryPrecisions.B, registerPrecisions.B));
  CCV_NNC_MFA_PRECONDITION
  (checkOperandPair(memoryPrecisions.C, registerPrecisions.C));
  if (registerPrecisions.C == GEMMOperandPrecision::BF16) {
    // BF16 has too few mantissa bits to be an accurate accumulator. In
    // addition, switching from FP32 accumulator to BF16 accumulator slows
    // down execution speed on both M1/M2 and M3+.
    CCV_NNC_MFA_PRECONDITION(false);
  }
  bool anyBF16 = (memoryPrecisions.A == GEMMOperandPrecision::BF16) || (memoryPrecisions.B == GEMMOperandPrecision::BF16) || (memoryPrecisions.C == GEMMOperandPrecision::BF16) || (memoryPrecisions.bias == GEMMOperandPrecision::BF16);
  
  // Inject the contents of the headers.
  source += createMetalSimdgroupEvent() + "\n";
  source += createMetalSimdgroupMatrixStorage(anyBF16) + "\n";
  source += "using namespace metal;\n";
  source += "\n";
  
  // Declare the size of M and N within a register allocation.
  {
    uint16_t registerM = blockDimensions[0] / splits[0];
    uint16_t registerN = blockDimensions[1] / splits[1];
    source += "#define REGISTER_M " + std::to_string(registerM) + "\n";
    source += "#define REGISTER_N " + std::to_string(registerN) + "\n";
  }
  
  // Retrieve the "padded" block dimensions, otherwise compute analytically
  // from the true block dimensions.
  simd::ushort2 paddedBlockDimensionsA; // (M, K)
  simd::ushort2 paddedBlockDimensionsB; // (K, N)
  simd::ushort2 paddedBlockDimensionsC; // (M, N)
  if (descriptor.paddedBlockDimensions.has_value()) {
    auto paddedBlockDimensions = descriptor.paddedBlockDimensions.value();
    
    paddedBlockDimensionsA = {
      paddedBlockDimensions[0], paddedBlockDimensions[1]
    };
    paddedBlockDimensionsB = {
      paddedBlockDimensions[2], paddedBlockDimensions[3]
    };
    paddedBlockDimensionsC = {
      paddedBlockDimensions[4], paddedBlockDimensions[5]
    };
  } else {
    auto blockDimensionM = blockDimensions[0];
    auto blockDimensionN = blockDimensions[1];
    auto blockDimensionK = blockDimensions[2];
    paddedBlockDimensionsA = { blockDimensionM, blockDimensionK };
    paddedBlockDimensionsB = { blockDimensionK, blockDimensionN };
    paddedBlockDimensionsC = { blockDimensionM, blockDimensionN };
  }
  
  // Determine the block dimensions from the transpose state.
  std::string leadingDimensionA;
  std::string leadingDimensionB;
  uint16_t leadingBlockDimensionA;
  uint16_t leadingBlockDimensionB;
  if (transposeState[0]) {
    leadingDimensionA = "M";
    leadingBlockDimensionA = paddedBlockDimensionsA[0];
  } else {
    leadingDimensionA = "K";
    leadingBlockDimensionA = paddedBlockDimensionsA[1];
  }
  if (transposeState[1]) {
    leadingDimensionB = "K";
    leadingBlockDimensionB = paddedBlockDimensionsB[0];
  } else {
    leadingDimensionB = "N";
    leadingBlockDimensionB = paddedBlockDimensionsB[1];
  }
  source += "#define LEADING_DIMENSION_A " + leadingDimensionA + "\n";
  source += "#define LEADING_DIMENSION_B " + leadingDimensionB + "\n";
  source += "#define LEADING_BLOCK_DIMENSION_A ";
  source += std::to_string(leadingBlockDimensionA) + "\n";
  source += "#define LEADING_BLOCK_DIMENSION_B ";
  source += std::to_string(leadingBlockDimensionB) + "\n";
  
  // Add the function constants.
  source += R"(
// Dimensions of each matrix.
// - Limitations to matrix size:
//   - 2^32 in each dimension (M/N/K).
//   - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//     good chance this will significantly degrade performance, and require
//     changing the data type of several variables that process addresses. The
//     client is responsible for ensuring correctness and performance with
//     matrices spanning several billion elements in one direction.
//   - The matrix dimensions must be known at compile time, via function
//     constants. Dynamic matrix shapes are beyond the scope of this reference
//     implementation. Dynamic shapes cause a non-negligible regression to
//     shader execution speed. However, they could minimize a compilation
//     latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

)";
  
  // Whether each matrix is transposed.
  source += "constant bool A_trans = ";
  source += std::to_string(bool(transposeState[0])) + ";\n";
  source += "constant bool B_trans = ";
  source += std::to_string(bool(transposeState[1])) + ";\n";
  source += "\n";

  if (useBias) {
    source += "constant bool bias_trans = ";
    source += std::to_string(bool(transposeState[2])) + ";\n";
    source += "\n";
  }
  
  // Define the memory layout of the matrix block.
  source += "constant ushort M_group = ";
  source += std::to_string(blockDimensions[0]) + ";\n";
  source += "constant ushort N_group = ";
  source += std::to_string(blockDimensions[1]) + ";\n";
  source += "constant ushort K_group = ";
  source += std::to_string(blockDimensions[2]) + ";\n";
  source += "\n";
  
  // Thresholds that mark the matrix edge.
  source += "constant uint M_edge = M - (M % M_group);\n";
  source += "constant uint N_edge = N - (N % N_group);\n";
  source += "\n";
  
  // Find the number of elements in the final block. If the matrix
  // dimensions are perfectly divisibly by block dimensions, we don't want
  // this value to be zero. The final block is a full block.
  source += "constant ushort M_remainder = (M % REGISTER_M == 0)\n";
  source += "  ? REGISTER_M : M % REGISTER_M;\n";
  source += "constant ushort N_remainder = (N % REGISTER_N == 0)\n";
  source += "  ? REGISTER_N : N % REGISTER_N;\n";
  source += "constant ushort K_remainder = (K % K_group == 0)\n";
  source += "  ? K_group : K % K_group;\n";
  source += "constant ushort K_remainder_padded = ";
  source += "(K_remainder + 7) / 8 * 8;\n";
  
  // Shift the final block, so it doesn't access out-of-bounds memory.
  source += "constant ushort M_shift = (M < M_group) ";
  source += "? 0 : REGISTER_M - M_remainder;\n";
  source += "constant ushort N_shift = (N < N_group) ";
  source += "? 0 : REGISTER_N - N_remainder;\n";
  
  {
    // Allocate threadgroup memory, using the 'memory precision'. This memory
    // is allocated at runtime, either by the user (explicit API call) or by
    // the driver (behind the scenes).
    std::string memoryNameA = memoryPrecisions.A.name();
    std::string memoryNameB = memoryPrecisions.B.name();
    std::string memoryNameC = memoryPrecisions.C.name();
    std::string memoryNameBias = memoryPrecisions.bias.name();
    source += "#define MEMORY_NAME_A " + memoryNameA + "\n";
    source += "#define MEMORY_NAME_B " + memoryNameB + "\n";
    source += "#define MEMORY_NAME_C " + memoryNameC + "\n";
    source += "#define MEMORY_NAME_BIAS " + memoryNameBias + "\n";
    
    // Allocate thread memory, using the 'register precision'. This memory
    // is allocated by embedding the precision into the assembly code.
    std::string registerNameA = registerPrecisions.A.name();
    std::string registerNameB = registerPrecisions.B.name();
    std::string registerNameC = registerPrecisions.C.name();
    std::string registerNameBias = registerPrecisions.bias.name();
    source += "#define REGISTER_NAME_A " + registerNameA + "\n";
    source += "#define REGISTER_NAME_B " + registerNameB + "\n";
    source += "#define REGISTER_NAME_C " + registerNameC + "\n";
    source += "#define REGISTER_NAME_BIAS " + registerNameBias + "\n";
  }
  
  // Add the utility functions.
  source += R"(
// Indexes into an array of registers.
//
// Calls to this function are expected to be evaluated at compile time. The
// array indices transform into register offsets, which are embedded into the
// assembly code.
template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}
)";
  
  struct MultiplyDescriptor {
    std::optional<std::string> addressSpace;
    std::optional<std::string> leadingDimensionA;
    std::optional<std::string> leadingDimensionB;
    std::optional<std::string> loadFunctionA;
    std::optional<std::string> loadFunctionB;
  };
  
  auto createMultiply =
  [=](MultiplyDescriptor descriptor) -> std::string {
    CCV_NNC_MFA_PRECONDITION(descriptor.addressSpace.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.leadingDimensionA.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.leadingDimensionB.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.loadFunctionA.has_value());
    CCV_NNC_MFA_PRECONDITION(descriptor.loadFunctionB.has_value());
    auto addressSpace = descriptor.addressSpace.value();
    auto leadingDimensionA = descriptor.leadingDimensionA.value();
    auto leadingDimensionB = descriptor.leadingDimensionB.value();
    auto loadFunctionA = descriptor.loadFunctionA.value();
    auto loadFunctionB = descriptor.loadFunctionB.value();
    
    std::string output;
    output += R"(
// One multiply-accumulate loop iteration, or 8 dot products.
METAL_FUNC void multiply_accumulate(
)";
    output += "  const " + addressSpace + " MEMORY_NAME_A *A_src,\n";
    output += "  const " + addressSpace + " MEMORY_NAME_B *B_src,";
    output += R"(
  thread simdgroup_matrix_storage<REGISTER_NAME_A> *A_sram,
  thread simdgroup_matrix_storage<REGISTER_NAME_B> *B_sram,
  thread simdgroup_matrix_storage<REGISTER_NAME_C> *C_sram,
  ushort k
) {
#pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
)";
    output += "    A->" + loadFunctionA + "(A_src, ";
    output += leadingDimensionA + ", ushort2(k, m), A_trans);";
    output += R"(
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < REGISTER_N; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, REGISTER_N, origin);
)";
    output += "    B->" + loadFunctionB + "(B_src, ";
    output += leadingDimensionB + ", ushort2(n, k), B_trans);";
    output += R"(
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
)";
    return output;
  };
  
  // Add the utility functions for the multiply-accumulate inner loop.
  {
    MultiplyDescriptor multiplyDesc;
    if (memoryPrecisions.A == GEMMOperandPrecision::BF16 &&
        registerPrecisions.A == GEMMOperandPrecision::FP32) {
      multiplyDesc.loadFunctionA = "load_bfloat";
    } else {
      multiplyDesc.loadFunctionA = "load";
    }
    if (memoryPrecisions.B == GEMMOperandPrecision::BF16 &&
        registerPrecisions.B == GEMMOperandPrecision::FP32) {
      multiplyDesc.loadFunctionB = "load_bfloat";
    } else {
      multiplyDesc.loadFunctionB = "load";
    }
    
    multiplyDesc.addressSpace = "device";
    multiplyDesc.leadingDimensionA = leadingDimensionA;
    multiplyDesc.leadingDimensionB = leadingDimensionB;
    source += createMultiply(multiplyDesc);
    
    multiplyDesc.addressSpace = "threadgroup";
    multiplyDesc.leadingDimensionA = std::to_string(leadingBlockDimensionA);
    multiplyDesc.leadingDimensionB = std::to_string(leadingBlockDimensionB);
    source += createMultiply(multiplyDesc);
  }
  
  // Add the setup portion where the addresses are prepared.
  {
    uint16_t blockBytesA =
    paddedBlockDimensionsA[0] * paddedBlockDimensionsA[1];
    uint16_t blockBytesB =
    paddedBlockDimensionsB[0] * paddedBlockDimensionsB[1];
    uint16_t blockBytesC =
    paddedBlockDimensionsC[0] * paddedBlockDimensionsC[1];
    
    blockBytesA *= uint16_t(memoryPrecisions.A.size());
    blockBytesB *= uint16_t(memoryPrecisions.B.size());
    blockBytesC *= uint16_t(memoryPrecisions.C.size());
    threadgroupMemoryAllocation = std::max
    (uint16_t(blockBytesA + blockBytesB), blockBytesC);
    
    source += "\n";
    source += "#define BLOCK_BYTES_A " + std::to_string(blockBytesA) + "\n";
    source += "#define SPLITS_N " + std::to_string(splits[1]) + "\n";
    
    source += R"(

// Metal function arguments.
//
// A: the left-hand side matrix
// - dimensions: M x K
//               K x M (transposed)
// - memory precision: memA
// - register precision: regA
//
// B: the right-hand side matrix
// - dimensions: K x N
//               N x K (transposed)
// - memory precision: memB
// - register precision: regB
//
// C: the output matrix, alternatively the dot product accumulator
// - dimensions: M x N
// - memory precision: memC
// - register precision: regC
//
// threadgroup_block: the chunk of threadgroup memory allocated at runtime
// - ideally 10 KB or less
// - precision: void/8-bit integer to make the pointer arithmetic more legible
kernel void gemm(device MEMORY_NAME_A *A [[buffer(0)]],
                 device MEMORY_NAME_B *B [[buffer(1)]],
                 device MEMORY_NAME_C *C [[buffer(2)]],
)";
    if (useBias) {
      source += "\n";
      source += "device MEMORY_NAME_BIAS *bias [[buffer(3)]],\n";
    }
                 
    source += R"(
                 threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                 
                 uint3 gid [[threadgroup_position_in_grid]],
                 ushort sidx [[simdgroup_index_in_threadgroup]],
                 ushort lane_id [[thread_index_in_simdgroup]])
{
  auto A_block = (threadgroup MEMORY_NAME_A*)(threadgroup_block);
  auto B_block = (threadgroup MEMORY_NAME_B*)(threadgroup_block + BLOCK_BYTES_A);
  ushort2 sid(sidx % SPLITS_N, sidx / SPLITS_N);
  ushort2 morton_offset = morton_order(lane_id);
  
  // Return early if the SIMD is out of bounds.
  //
  // There could be some threadgroups where the matrix edge cuts straight
  // through the middle of the block. SIMDs on the right or bottom of the
  // dividing line must be stopped from causing out-of-bounds accesses. This is
  // the reason for the early exit.
  uint M_offset = gid.y * M_group;
  uint N_offset = gid.x * N_group;
  {
    if (M_offset + sid.y * REGISTER_M >= M ||
        N_offset + sid.x * REGISTER_N >= N) {
      return;
    }
  }
  ushort2 offset_in_group(sid.x * REGISTER_N + morton_offset.x,
                          sid.y * REGISTER_M + morton_offset.y);
  
  // Shift the matrix block within bounds, if possible.
  if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
    M_offset -= M_shift;
  }
  if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
    N_offset -= N_shift;
  }
)";
  }
  
  // Add the setup of the accumulator.
  source += R"(
  simdgroup_matrix_storage<REGISTER_NAME_C> C_sram[
    (REGISTER_M / 8) * (REGISTER_N / 8)];
)";

  if (useBias) {
    if (descriptor.preferAsyncLoad) {
      source += "\n";
      source += "#define USE_BIAS_ASYNC_COND false\n";
    } else {
      source += "\n";
      source += "#define USE_BIAS_ASYNC_COND (M >= M_group) && (N >= N_group)\n";
    }
    if (memoryPrecisions.bias == GEMMOperandPrecision::BF16 &&
        registerPrecisions.bias == GEMMOperandPrecision::FP32) {
      source += "\n";
      source += "#define BIAS_LOAD load_bfloat\n";
    } else { 
      source += "\n";
      source += "#define BIAS_LOAD load\n";
    }
    std::string declareBiasLocationDevice;
    std::string declareBiasLocationThreadgroup;
    if (transposeState[2]) {
      declareBiasLocationDevice = R"(
    uint2 bias_offset(uint(M_offset + offset_in_group.y), 0);
    auto bias_src =
      simdgroup_matrix_storage<MEMORY_NAME_BIAS>::apply_offset(
        bias, 0, bias_offset);
)";
      declareBiasLocationThreadgroup = R"(
    ushort2 bias_block_offset(ushort(offset_in_group.y), 0);
    auto bias_src = (threadgroup MEMORY_NAME_BIAS*)(threadgroup_block);
    bias_src = simdgroup_matrix_storage<MEMORY_NAME_BIAS>::apply_offset(
      bias_src, 0, bias_block_offset);
)";
    } else {
      declareBiasLocationDevice = R"(
    uint2 bias_offset(uint(N_offset + offset_in_group.x), 0);
    auto bias_src =
      simdgroup_matrix_storage<MEMORY_NAME_BIAS>::apply_offset(
        bias, 0, bias_offset);
)";
      declareBiasLocationThreadgroup = R"(
    ushort2 bias_block_offset(ushort(offset_in_group.x), 0);
    auto bias_src = (threadgroup MEMORY_NAME_BIAS*)(threadgroup_block);
    bias_src = simdgroup_matrix_storage<MEMORY_NAME_BIAS>::apply_offset(
      bias_src, 0, bias_block_offset);
)";
    }
    std::string loadBiasLoop;
    if (transposeState[2]) {
      loadBiasLoop = R"(
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < REGISTER_M; m += 8) {
      simdgroup_matrix_storage<REGISTER_NAME_BIAS> bias;
      bias.BIAS_LOAD(
        bias_src, 0, ushort2(m, 0));
      bias.thread_elements()[0][1] = bias.thread_elements()[0][0];

      #pragma clang loop unroll(full)
      for (ushort n = 0; n < REGISTER_N; n += 8) {
        vec<REGISTER_NAME_BIAS, 2> biasForm = *(bias.thread_elements());
        auto accumulatorForm = vec<REGISTER_NAME_C, 2>(biasForm);

        ushort2 origin(n, m);
        auto C = get_sram(C_sram, REGISTER_N, origin);
        *C = simdgroup_matrix_storage<REGISTER_NAME_C>(accumulatorForm);
      }
    }
)";
    } else {
      loadBiasLoop = R"(
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      simdgroup_matrix_storage<REGISTER_NAME_BIAS> bias;
      bias.BIAS_LOAD(
        bias_src, 0, ushort2(n, 0));

      #pragma clang loop unroll(full)
      for (ushort m = 0; m < REGISTER_M; m += 8) {
        vec<REGISTER_NAME_BIAS, 2> biasForm = *(bias.thread_elements());
        auto accumulatorForm = vec<REGISTER_NAME_C, 2>(biasForm);

        ushort2 origin(n, m);
        auto C = get_sram(C_sram, REGISTER_N, origin);
        *C = simdgroup_matrix_storage<REGISTER_NAME_C>(accumulatorForm);
      }
    }
)";
    }
    source += R"(
  if (USE_BIAS_ASYNC_COND) {
)";
    source += declareBiasLocationDevice;
    source += loadBiasLoop;
    source += R"(
  } else {
    if (sidx == 0) {
      uint2 bias_offset(bias_trans ? M_offset : N_offset, 0);
      auto bias_dst = (threadgroup MEMORY_NAME_BIAS*)(threadgroup_block);
      auto bias_src =
      simdgroup_matrix_storage<MEMORY_NAME_BIAS>::apply_offset(
        bias, 0, bias_offset);
      
      ushort bias_tile_dimension = bias_trans
      ? min(uint(M_group), M - M_offset)
      : min(uint(N_group), N - N_offset);
    
      // Issue an async copy.
      simdgroup_event event;
      event.async_copy(
        bias_dst, 1, ushort2(bias_tile_dimension, 1),
        bias_src, 1, ushort2(bias_tile_dimension, 1));
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    source += declareBiasLocationThreadgroup;
    source += loadBiasLoop;
    source += R"(
        
    // Add a barrier, because you accessed the entries from threadgroup
    // memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
)";
  } else {
    source += R"(
  // Initialize the accumulator.
  #pragma clang loop unroll(full)
  for (ushort m = 0; m < REGISTER_M; m += 8) {
  #pragma clang loop unroll(full)
    for (ushort n = 0; n < REGISTER_N; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, REGISTER_N, origin);
      *C = simdgroup_matrix_storage<REGISTER_NAME_C>(0);
    }
  }
)";
  }
  
  // Add the matrix multiplication iterations.
  //
  // Async copies are required for correct behavior in edge cases. We attempt
  // to execute most iterations without async copy, and only the necessary
  // ones with async copy.
  {
    std::string asyncIterationsStart;
    if (descriptor.preferAsyncLoad) {
      asyncIterationsStart = "0";
    } else {
      asyncIterationsStart = "(K - (K % K_group))";
    }
    std::string paddedCeilingK = "(K + K_remainder_padded - K_remainder)";
    source += "#define ASYNC_ITERATIONS_START " + asyncIterationsStart + "\n";
    source += "#define PADDED_CEILING_K " + paddedCeilingK + "\n";
    
    source += R"(

// Perform the iterations where async copy is avoided.
for (uint k = 0; k < ASYNC_ITERATIONS_START; k += 8) {
  uint2 A_offset(k, M_offset);
  uint2 B_offset(N_offset, k);
  A_offset += uint2(morton_offset.x, offset_in_group.y);
  B_offset += uint2(offset_in_group.x, morton_offset.y);
  
  auto A_src = simdgroup_matrix_storage<MEMORY_NAME_A>::apply_offset(
    A, LEADING_DIMENSION_A, A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<MEMORY_NAME_B>::apply_offset(
    B, LEADING_DIMENSION_B, B_offset, B_trans);

  simdgroup_matrix_storage<REGISTER_NAME_A> A_sram[(REGISTER_M / 8) * (8 / 8)];
  simdgroup_matrix_storage<REGISTER_NAME_B> B_sram[(8 / 8) * (REGISTER_N / 8)];
  multiply_accumulate(A_src, B_src,
                      A_sram, B_sram, C_sram, 0);
}

// Perform the iterations where async copy is used.
for (uint k = ASYNC_ITERATIONS_START; k < K; k += K_group) {
  // Launch an async copy from device to threadgroup memory.
  if (sidx == 0) {
    uint2 A_offset(k, M_offset);
    uint2 B_offset(N_offset, k);
    auto A_src = simdgroup_matrix_storage<MEMORY_NAME_A>::apply_offset(
      A, LEADING_DIMENSION_A, A_offset, A_trans);
    auto B_src = simdgroup_matrix_storage<MEMORY_NAME_B>::apply_offset(
      B, LEADING_DIMENSION_B, B_offset, B_trans);

    ushort M_tile_dimension = min(uint(M_group), M - M_offset);
    ushort N_tile_dimension = min(uint(N_group), N - N_offset);
    ushort K_tile_dimension = min(uint(K_group), K - k);
    ushort K_tile_padded = min(uint(K_group), PADDED_CEILING_K - k);

    ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
    ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
    ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
    ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

    simdgroup_event events[2];
    events[0].async_copy(A_block, LEADING_BLOCK_DIMENSION_A, A_tile_dst,
                         A_src, LEADING_DIMENSION_A, A_tile_src, A_trans);
    events[1].async_copy(B_block, LEADING_BLOCK_DIMENSION_B, B_tile_dst,
                         B_src, LEADING_DIMENSION_B, B_tile_src, B_trans);
    simdgroup_event::wait(2, events);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
  ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
  auto A_block_src = simdgroup_matrix_storage<MEMORY_NAME_A>::apply_offset(
    A_block, LEADING_BLOCK_DIMENSION_A, A_block_offset, A_trans);
  auto B_block_src = simdgroup_matrix_storage<MEMORY_NAME_B>::apply_offset(
    B_block, LEADING_BLOCK_DIMENSION_B, B_block_offset, B_trans);

  simdgroup_matrix_storage<REGISTER_NAME_A> A_sram[
    (REGISTER_M / 8) * (K_group / 8)];
  simdgroup_matrix_storage<REGISTER_NAME_B> B_sram[
    (K_group / 8) * (REGISTER_N / 8)];
#pragma clang loop unroll(full)
  for (ushort k = 0; k < K_remainder_padded; k += 8) {
    multiply_accumulate(A_block_src, B_block_src,
                        A_sram, B_sram, C_sram, k);
  }

  // Will there be any iterations after this one?
  if (k + K_group < K) {
    // If so, we haven't reached the edge of either input matrix yet.
#pragma clang loop unroll(full)
    for (ushort k = K_remainder_padded; k < K_group; k += 8) {
      multiply_accumulate(A_block_src, B_block_src,
                          A_sram, B_sram, C_sram, k);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
  
)";
  }
  
  // Add the cleanup portion where the accumulator is stored.
  {
    std::string storeFunctionC;
    if (memoryPrecisions.C == GEMMOperandPrecision::BF16 &&
        registerPrecisions.C == GEMMOperandPrecision::FP32) {
      storeFunctionC = "store_bfloat";
    } else {
      storeFunctionC = "store";
    }
    
    std::string condition;
    if (preferAsyncStore) {
      condition = "false";
    } else {
      condition = "(M >= M_group) && (N >= N_group)";
    }
    
    source += "if (" + condition + ") {";
    source += R"(
    // Fast path for matrices that qualify.
    uint2 C_offset(N_offset + offset_in_group.x,
                   M_offset + offset_in_group.y);
    auto C_dst = simdgroup_matrix_storage<MEMORY_NAME_C>::apply_offset(
      C, N, C_offset);
    
    // Write the accumulator to device memory.
  #pragma clang loop unroll(full)
    for (ushort m = 0; m < REGISTER_M; m += 8) {
  #pragma clang loop unroll(full)
      for (ushort n = 0; n < REGISTER_N; n += 8) {
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, REGISTER_N, origin);
)";
    source += "    C->" + storeFunctionC + "(C_dst, N, origin);";
    source += R"(
      }
    }
  } else {
    // Slow path for when memory must be handled more carefully.
    auto C_block = (threadgroup MEMORY_NAME_C*)(threadgroup_block);
    auto C_block_dst = simdgroup_matrix_storage<MEMORY_NAME_C>::apply_offset(
      C_block, N_group, offset_in_group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write the accumulator to threadgroup memory.
  #pragma clang loop unroll(full)
    for (ushort m = 0; m < REGISTER_M; m += 8) {
  #pragma clang loop unroll(full)
      for (ushort n = 0; n < REGISTER_N; n += 8) {
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, REGISTER_N, origin);
)";
    source += "    C->" + storeFunctionC + "(C_block_dst, N_group, origin);";
    source += R"(
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Launch the async copy from threadgroup to device memory.
    if (sidx == 0) {
      uint2 C_offset(gid.x * N_group, gid.y * M_group);
      ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                     min(uint(M_group), M - C_offset.y));
      auto C_dst = simdgroup_matrix_storage<MEMORY_NAME_C>::apply_offset(
        C, N, C_offset);
      
      // If we shift successfully, the garbage zone moves from the bottom right
      // to the top left.
      if ((M_shift != 0) || (N_shift != 0)) {
        ushort2 C_block_shift(0, 0);
        if ((M_shift != 0) && (C_offset.y >= M_edge)) {
          C_block_shift.y = M_shift;
        }
        if ((N_shift != 0) && (C_offset.x >= N_edge)) {
          C_block_shift.x = N_shift;
        }
        C_block = simdgroup_matrix_storage<MEMORY_NAME_C>::apply_offset(
          C_block, N_group, C_block_shift);
      }
      
      simdgroup_event event;
      event.async_copy(C_dst, N, C_tile, C_block, N_group, C_tile);
    }
  }
)";
  }
  
  // Add the final closing brace of the Metal function.
  source += "}\n";

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}
