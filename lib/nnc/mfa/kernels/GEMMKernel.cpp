#include "GEMMKernel.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

std::string GEMMKernel::memoryName(char operand) const noexcept {
  switch (operand) {
  case 'A':
    return memoryPrecisions.A.name();
  case 'B':
    return memoryPrecisions.B.name();
  case 'C':
    return memoryPrecisions.C.name();
  case 'S':
    return memoryPrecisions.bias.name();
  default:
    return "";
  }
}

std::string GEMMKernel::registerName(char operand) const noexcept {
  switch (operand) {
  case 'A':
    return registerPrecisions.A.name();
  case 'B':
    return registerPrecisions.B.name();
  case 'C':
    return registerPrecisions.C.name();
  case 'S':
    return registerPrecisions.bias.name();
  default:
    return "";
  }
}

unsigned short GEMMKernel::createThreadgroupMemoryAllocation() const noexcept {
  unsigned short blockBytesA = blockBytes('A');
  unsigned short blockBytesB = blockBytes('B');
  unsigned short blockBytesC = blockBytes('C');
  return std::max((unsigned short)(blockBytesA + blockBytesB), blockBytesC);
}

bool GEMMKernel::transposed(char operand) const noexcept {
  switch (operand) {
  case 'A':
    return transposeState[0];
  case 'B':
    return transposeState[1];
  case 'C':
    return false;
  default:
    return false;
  }
}

std::string GEMMKernel::leadingDimension(char operand) const noexcept {
  return std::string(1, operand) + "_leading_dimension";
}

unsigned short GEMMKernel::leadingBlockDimension(char operand) const noexcept {
  switch (operand) {
  case 'A':
    return leadingBlockDimensions[0];
  case 'B':
    return leadingBlockDimensions[1];
  case 'C':
    return leadingBlockDimensions[2];
  default:
    return 0;
  }
}

unsigned short GEMMKernel::trailingBlockDimension(char operand) const noexcept {
  auto chooseTrailingBlockDimension =
  [=](bool transposeState, unsigned short untransposedRows, unsigned short untransposedColumns) -> unsigned short {
    if (transposeState) {
      return untransposedColumns;
    } else {
      return untransposedRows;
    }
  };

  switch (operand) {
  case 'A':
    return chooseTrailingBlockDimension(
      transposed('A'), blockDimensions[0], blockDimensions[2]);
  case 'B':
    return chooseTrailingBlockDimension(
      transposed('B'), blockDimensions[2], blockDimensions[1]);
  case 'C':
    return chooseTrailingBlockDimension(
      transposed('C'), blockDimensions[0], blockDimensions[1]);
  default:
    return 0;
  }
}

unsigned short GEMMKernel::blockBytes(char operand) const noexcept {
  unsigned short output = 1;
  output *= leadingBlockDimension(operand);
  output *= trailingBlockDimension(operand);

  GEMMOperandPrecision memoryPrecision;
  switch (operand) {
  case 'A':
    memoryPrecision = memoryPrecisions.A;
  case 'B':
    memoryPrecision = memoryPrecisions.B;
  case 'C':
    memoryPrecision = memoryPrecisions.C;
  }
  output *= memoryPrecision.size();
  return output;
}

GEMMKernel::GEMMKernel(GEMMKernelDescriptor descriptor, MTL::Device *const device) {
  blockDimensions = descriptor.blockDimensions;
  memoryPrecisions = descriptor.memoryPrecisions;
  registerPrecisions = descriptor.registerPrecisions;
  splits = descriptor.splits;
  transposeState = descriptor.transposeState;
  preferAsyncLoad = descriptor.preferAsyncLoad;
  preferAsyncStore = descriptor.preferAsyncStore;
  useBias = descriptor.useBias;
  loadM = descriptor.loadM;
  threadgroupSize = 32 * splits[0] * splits[1];
  disableAsyncCopy = false;
  
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
  
  // Declare the size of M and N within a register allocation.
  registerM = blockDimensions[0] / splits[0];
  registerN = blockDimensions[1] / splits[1];

  // Retrieve the "padded" block dimensions, otherwise compute analytically
  // from the true block dimensions.
  auto chooseLeadingBlockDimension =
  [=](unsigned short specifiedLeading, bool transposeState, unsigned short untransposedRows, unsigned short untransposedColumns) -> unsigned short {
    unsigned short expectedLeading;
    if (transposeState) {
      expectedLeading = untransposedRows;
    } else {
      expectedLeading = untransposedColumns;
    }

    unsigned short actualLeading;
    if (specifiedLeading != 0) {
      if (specifiedLeading < expectedLeading) {
        CCV_NNC_MFA_PRECONDITION(false && "Leading block dimension was too small.");
      }
      actualLeading = specifiedLeading;
    } else {
      actualLeading = expectedLeading;
    }

    return actualLeading;
  };

  leadingBlockDimensions[0] = chooseLeadingBlockDimension(
    descriptor.leadingBlockDimensions.value_or(simd::ushort3())[0], transposeState[0],
    blockDimensions[0], blockDimensions[2]);
  leadingBlockDimensions[1] = chooseLeadingBlockDimension(
    descriptor.leadingBlockDimensions.value_or(simd::ushort3())[1], transposeState[1],
    blockDimensions[2], blockDimensions[1]);
  leadingBlockDimensions[2] = chooseLeadingBlockDimension(
    descriptor.leadingBlockDimensions.value_or(simd::ushort3())[2], false,
    blockDimensions[0], blockDimensions[1]);

  threadgroupMemoryAllocation = createThreadgroupMemoryAllocation();

  source = createSource();

  // Compile the shader source.
  {
    auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    NS::Error* error = nil;
    library = NS::TransferPtr(device->newLibrary(string, nil, &error));
    if (error) {
      error = nil;
      library = NS::TransferPtr(findPrecompiledLibrary(descriptor, device, &error));
      if (!library) {
        preferAsyncLoad = false;
        preferAsyncStore = false;
        disableAsyncCopy = true;
        source = createSource();
        string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
        error = nil;
        library = NS::TransferPtr(device->newLibrary(string, nil, &error));
      }
    }
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
}

#pragma mark - Source

std::string GEMMKernel::createSource() const noexcept {
  CodeWriter source;

  bool injectBF16Methods = (memoryPrecisions.A == GEMMOperandPrecision::BF16) || (memoryPrecisions.B == GEMMOperandPrecision::BF16) || (memoryPrecisions.C == GEMMOperandPrecision::BF16) || (memoryPrecisions.bias == GEMMOperandPrecision::BF16);

  // Inject the contents of the headers.
  source += createMetalSimdgroupEvent(disableAsyncCopy) + "\n";
  source += createMetalSimdgroupMatrixStorage(injectBF16Methods) + "\n";
  source += "using namespace metal;\n\n";

  source.SetValue("TRANSPOSE_STATE_A", std::to_string(bool(transposeState[0])));
  source.SetValue("TRANSPOSE_STATE_B", std::to_string(bool(transposeState[1])));
  source.SetValue("TRANSPOSE_STATE_BIAS", std::to_string(bool(transposeState[2])));
  source.SetValue("BLOCK_DIMENSIONS_M", std::to_string(blockDimensions[0]));
  source.SetValue("BLOCK_DIMENSIONS_N", std::to_string(blockDimensions[1]));
  source.SetValue("BLOCK_DIMENSIONS_K", std::to_string(blockDimensions[2]));
  source.SetValue("REGISTER_M", std::to_string(registerM));
  source.SetValue("REGISTER_N", std::to_string(registerN));

  source += createConstants();

  source.SetValue("MEMORY_NAME_A", memoryName('A'));
  source.SetValue("MEMORY_NAME_B", memoryName('B'));
  source.SetValue("MEMORY_NAME_C", memoryName('C'));
  source.SetValue("MEMORY_NAME_BIAS", memoryName('S'));
  source.SetValue("REGISTER_NAME_A", registerName('A'));
  source.SetValue("REGISTER_NAME_B", registerName('B'));
  source.SetValue("REGISTER_NAME_C", registerName('C'));
  source.SetValue("REGISTER_NAME_BIAS", registerName('S'));
  source.SetValue("SPLITS_N", std::to_string(splits[1]));
  if (disableAsyncCopy) {
    source.SetValue("ASYNC_LANE_ID", ", lane_id");
  } else {
    source.SetValue("ASYNC_LANE_ID", "");
  }

  createUtilities(&source);

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

kernel void gemm(device {{MEMORY_NAME_A}} *A [[buffer(0)]],
                 device {{MEMORY_NAME_B}} *B [[buffer(1)]],
                 device {{MEMORY_NAME_C}} *C [[buffer(2)]],
)";
  if (useBias) {
    source += R"(
                 device {{MEMORY_NAME_BIAS}} *bias [[buffer(3)]],
)";
    if (loadM) {
      source += R"(
                   device uint *loadM [[buffer(4)]],
)";
    }
  } else {
    if (loadM) {
      source += R"(
                   device uint *loadM [[buffer(3)]],
)";
    }
  }
  source += R"(
                 threadgroup uchar *threadgroup_block [[threadgroup(0)]],

                 uint3 gid [[threadgroup_position_in_grid]],
                 ushort sidx [[simdgroup_index_in_threadgroup]],
                 ushort lane_id [[thread_index_in_simdgroup]])
{
  if (batched) {
    A = A + A_batch_stride * gid.z;
    B = B + B_batch_stride * gid.z;
    C = C + C_batch_stride * gid.z;
)";
  if (useBias) {
    source += R"(
    bias = bias + bias_batch_stride * gid.z;
)";
  }
  source += R"(
  }
)";
  if (loadM) {
    source += R"(
  const uint M = loadM[0];
  // Thresholds that mark the matrix edge.
  const uint M_edge = M - (M % M_group);
  // Find the number of elements in the final block. If the matrix
  // dimensions are perfectly divisibly by block dimensions, we don't want
  // this value to be zero. The final block is a full block.
  const ushort M_remainder = (M % {{REGISTER_M}} == 0)
    ? {{REGISTER_M}} : M % {{REGISTER_M}};
  // Shift the final block, so it doesn't access out-of-bounds memory.
  const ushort M_shift = (M < M_group) ? 0 : {{REGISTER_M}} - M_remainder;
)";
  }
  source += R"(
  ushort2 sid(sidx % {{SPLITS_N}}, sidx / {{SPLITS_N}});
  ushort2 morton_offset = morton_order(lane_id);
  
  // Return early if the SIMD is out of bounds.
  //
  // There could be some threadgroups where the matrix edge cuts straight
  // through the middle of the block. SIMDs on the right or bottom of the
  // dividing line must be stopped from causing out-of-bounds accesses. This is
  // the reason for the early exit.
  uint M_offset = gid.y * M_group;
  uint N_offset = gid.x * N_group;
  if (M_offset + sid.y * {{REGISTER_M}} >= M ||
      N_offset + sid.x * {{REGISTER_N}} >= N) {
    return;
  }
  ushort2 offset_in_group(sid.x * {{REGISTER_N}} + morton_offset.x,
                          sid.y * {{REGISTER_M}} + morton_offset.y);
  
  // Shift the matrix block within bounds, if possible.
  if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
    M_offset -= M_shift;
  }
  if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
    N_offset -= N_shift;
  }

)";

  createInitializeC(&source);

  createMultiplyIterations(&source);

  createStoreC(&source);

  source += "}\n\n";

  return source.ToString();
}

std::string GEMMKernel::createConstants() const noexcept {
  std::string constants = R"(
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
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Specify the leading dimensions at PSO creation time.
constant uint A_leading_dimension [[function_constant(5)]];
constant uint B_leading_dimension [[function_constant(6)]];
constant uint C_leading_dimension [[function_constant(7)]];

// Whether to load the previous value of C, and add it to the accumulator.
constant bool load_previous_C [[function_constant(10)]];

// Specify the batch / batch strides at PSO creation time.
constant bool batched [[function_constant(11)]];

constant uint A_batch_stride [[function_constant(15)]];
constant uint B_batch_stride [[function_constant(16)]];
constant uint C_batch_stride [[function_constant(17)]];
constant uint bias_batch_stride [[function_constant(18)]];

// Whether each matrix is transposed.
constant bool A_trans = {{TRANSPOSE_STATE_A}};
constant bool B_trans = {{TRANSPOSE_STATE_B}};
)";
  if (useBias) {
    constants += R"(
constant bool bias_trans = {{TRANSPOSE_STATE_BIAS}};
)";
  }
  constants += R"(

// Define the memory layout of the matrix block.
constant ushort M_group = {{BLOCK_DIMENSIONS_M}};
constant ushort N_group = {{BLOCK_DIMENSIONS_N}};
constant ushort K_group = {{BLOCK_DIMENSIONS_K}};

// Thresholds that mark the matrix edge.
constant uint N_edge = N - (N % N_group);

// Find the number of elements in the final block. If the matrix
// dimensions are perfectly divisibly by block dimensions, we don't want
// this value to be zero. The final block is a full block.
constant ushort N_remainder = (N % {{REGISTER_N}} == 0)
  ? {{REGISTER_N}} : N % {{REGISTER_N}};
constant ushort K_remainder = (K % K_group == 0)
  ? K_group : K % K_group;
constant ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

// Shift the final block, so it doesn't access out-of-bounds memory.
constant ushort N_shift = (N < N_group) ? 0 : {{REGISTER_N}} - N_remainder;

)";
  if (!loadM) {
    constants += R"(
constant uint M [[function_constant(0)]];
// Thresholds that mark the matrix edge.
constant uint M_edge = M - (M % M_group);
// Find the number of elements in the final block. If the matrix
// dimensions are perfectly divisibly by block dimensions, we don't want
// this value to be zero. The final block is a full block.
constant ushort M_remainder = (M % {{REGISTER_M}} == 0)
  ? {{REGISTER_M}} : M % {{REGISTER_M}};
// Shift the final block, so it doesn't access out-of-bounds memory.
constant ushort M_shift = (M < M_group) ? 0 : {{REGISTER_M}} - M_remainder;
)";
  }
  return constants;
}

void GEMMKernel::createUtilities(CodeWriter *const source) const noexcept {
  // Add the utility functions.
  *source += R"(

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

  std::string createMultiply = R"(

// One multiply-accumulate loop iteration, or 8 dot products.
METAL_FUNC void multiply_accumulate(
const {{ADDRESS_SPACE}} {{MEMORY_NAME_A}} *A_src,
const {{ADDRESS_SPACE}} {{MEMORY_NAME_B}} *B_src,
thread simdgroup_matrix_storage<{{REGISTER_NAME_A}}> *A_sram,
thread simdgroup_matrix_storage<{{REGISTER_NAME_B}}> *B_sram,
thread simdgroup_matrix_storage<{{REGISTER_NAME_C}}> *C_sram,
ushort k
) {
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
  ushort2 origin(0, m);
  auto A = get_sram(A_sram, 8, origin);
  A->{{LOAD_FUNCTION_A}}(A_src, {{LEADING_DIMENSION_A}}, ushort2(k, m), A_trans);
}
#pragma clang loop unroll(full)
for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
  ushort2 origin(n, 0);
  auto B = get_sram(B_sram, {{REGISTER_N}}, origin);
  B->{{LOAD_FUNCTION_B}}(B_src, {{LEADING_DIMENSION_B}}, ushort2(n, k), B_trans);
}
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
  for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
    auto A = get_sram(A_sram, 8, ushort2(0, m));
    auto B = get_sram(B_sram, {{REGISTER_N}}, ushort2(n, 0));
    auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
    C->multiply(*A, *B);
  }
}
}

)";

  // Add the utility functions for the multiply-accumulate inner loop.
  if (memoryPrecisions.A == GEMMOperandPrecision::BF16 && registerPrecisions.A == GEMMOperandPrecision::FP32) {
    source->SetValue("LOAD_FUNCTION_A", "load_bfloat");
  } else {
    source->SetValue("LOAD_FUNCTION_A", "load");
  }
  if (memoryPrecisions.B == GEMMOperandPrecision::BF16 && registerPrecisions.B == GEMMOperandPrecision::FP32) {
    source->SetValue("LOAD_FUNCTION_B", "load_bfloat");
  } else {
    source->SetValue("LOAD_FUNCTION_B", "load");
  }

  source->SetValue("ADDRESS_SPACE", "device");
  source->SetValue("LEADING_DIMENSION_A", leadingDimension('A'));
  source->SetValue("LEADING_DIMENSION_B", leadingDimension('B'));

  *source += createMultiply;

  source->SetValue("ADDRESS_SPACE", "threadgroup");
  source->SetValue("LEADING_DIMENSION_A", std::to_string(leadingBlockDimensions[0]));
  source->SetValue("LEADING_DIMENSION_B", std::to_string(leadingBlockDimensions[1]));
  *source += createMultiply;
}

#pragma mark - Caching

void GEMMKernel::createInitializeC(CodeWriter *source) const noexcept {
  source->SetValue("REGISTER_M_8_REGISTER_N_8", std::to_string((registerM / 8) * (registerN / 8)));
  *source += R"(
    
    simdgroup_matrix_storage<{{REGISTER_NAME_C}}> C_sram[
      {{REGISTER_M_8_REGISTER_N_8}}];
    
    if (load_previous_C) {
  )";
  createLoadC(source);
  *source += R"(
    } else {
)";
  if (useBias) {
    if (true) { // TODO: figure why on M3 / M4 this is faster. preferAsyncLoad) {
      source->SetValue("DIRECT_BIAS_ACCESS_CONDITION", "false");
    } else {
      source->SetValue("DIRECT_BIAS_ACCESS_CONDITION", "(M >= M_group) && (N >= N_group)");
    }
    if (memoryPrecisions.bias == GEMMOperandPrecision::BF16 && registerPrecisions.bias == GEMMOperandPrecision::FP32) {
      source->SetValue("LOAD_FUNCTION_BIAS", "load_bfloat");
    } else {
      source->SetValue("LOAD_FUNCTION_BIAS", "load");
    }
    std::string declareBiasLocationDevice;
    std::string declareBiasLocationThreadgroup;
    if (transposeState[2]) {
      declareBiasLocationDevice = R"(
    uint2 bias_offset(uint(M_offset + offset_in_group.y), 0);
    auto bias_src =
      simdgroup_matrix_storage<{{MEMORY_NAME_BIAS}}>::apply_offset(
        bias, 0, bias_offset);
)";
      declareBiasLocationThreadgroup = R"(
    ushort2 bias_block_offset(ushort(offset_in_group.y), 0);
    auto bias_src = (threadgroup {{MEMORY_NAME_BIAS}}*)(threadgroup_block);
    bias_src = simdgroup_matrix_storage<{{MEMORY_NAME_BIAS}}>::apply_offset(
      bias_src, 0, bias_block_offset);
)";
    } else {
      declareBiasLocationDevice = R"(
    uint2 bias_offset(uint(N_offset + offset_in_group.x), 0);
    auto bias_src =
      simdgroup_matrix_storage<{{MEMORY_NAME_BIAS}}>::apply_offset(
        bias, 0, bias_offset);
)";
      declareBiasLocationThreadgroup = R"(
    ushort2 bias_block_offset(ushort(offset_in_group.x), 0);
    auto bias_src = (threadgroup {{MEMORY_NAME_BIAS}}*)(threadgroup_block);
    bias_src = simdgroup_matrix_storage<{{MEMORY_NAME_BIAS}}>::apply_offset(
      bias_src, 0, bias_block_offset);
)";
    }
    std::string loadBiasLoop;
    if (transposeState[2]) {
      loadBiasLoop = R"(
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
      simdgroup_matrix_storage<{{REGISTER_NAME_BIAS}}> bias;
      bias.{{LOAD_FUNCTION_BIAS}}(
        bias_src, 0, ushort2(m, 0));
      bias.thread_elements()[0][1] = bias.thread_elements()[0][0];

      #pragma clang loop unroll(full)
      for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        vec<{{REGISTER_NAME_BIAS}}, 2> biasForm = *(bias.thread_elements());
        auto accumulatorForm = vec<{{REGISTER_NAME_C}}, 2>(biasForm);

        ushort2 origin(n, m);
        auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
        *C = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(accumulatorForm);
      }
    }
)";
    } else {
      loadBiasLoop = R"(
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
      simdgroup_matrix_storage<{{REGISTER_NAME_BIAS}}> bias;
      bias.{{LOAD_FUNCTION_BIAS}}(
        bias_src, 0, ushort2(n, 0));

      #pragma clang loop unroll(full)
      for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
        vec<{{REGISTER_NAME_BIAS}}, 2> biasForm = *(bias.thread_elements());
        auto accumulatorForm = vec<{{REGISTER_NAME_C}}, 2>(biasForm);
        ushort2 origin(n, m);
        auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
        *C = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(accumulatorForm);
      }
    }
)";
    }
    *source += R"(
  if ({{DIRECT_BIAS_ACCESS_CONDITION}}) {
)";
    *source += declareBiasLocationDevice;
    *source += loadBiasLoop;
    *source += R"(
  } else {
    if (sidx == 0) {
      uint2 bias_offset(bias_trans ? M_offset : N_offset, 0);
      auto bias_dst = (threadgroup {{MEMORY_NAME_BIAS}}*)(threadgroup_block);
      auto bias_src =
      simdgroup_matrix_storage<{{MEMORY_NAME_BIAS}}>::apply_offset(
        bias, 0, bias_offset);

      ushort bias_tile_dimension = bias_trans
      ? min(uint(M_group), M - M_offset)
      : min(uint(N_group), N - N_offset);

      // Issue an async copy.
      simdgroup_event event;
      event.async_copy<32>(
        bias_dst, bias_tile_dimension,
        bias_src, bias_tile_dimension{{ASYNC_LANE_ID}});
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
)";
    *source += declareBiasLocationThreadgroup;
    *source += loadBiasLoop;
    *source += R"(
    // Add a barrier, because you accessed the entries from threadgroup
    // memory.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  }
)";
  } else {
    *source += R"(
      #pragma clang loop unroll(full)
      for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
          ushort2 origin(n, m);
          auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
          *C = simdgroup_matrix_storage<{{REGISTER_NAME_C}}>(0);
        }
      }
    }
)";
  }
}

void GEMMKernel::createLoadC(CodeWriter *source) const noexcept {
  if (memoryPrecisions.C == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32) {
    source->SetValue("LOAD_FUNCTION_C", "load_bfloat");
  } else {
    source->SetValue("LOAD_FUNCTION_C", "load");
  }
  source->SetValue("LEADING_DIMENSION_C", leadingDimension('C'));
  source->SetValue("LEADING_BLOCK_DIMENSIONS_C", std::to_string(leadingBlockDimensions[2]));

  if (preferAsyncStore) {
    source->SetValue("DIRECT_ACCESS_CONDITION", "false");
  } else {
    // In the vanilla GEMM kernel, the extra storing code can be optimized
    // away at compile time. The compiler may allocate less registers, and
    // occupancy may be greater.
    std::string output = "(M >= M_group) && (N >= N_group)";

    // When accumulate is supported, there are overlapping writes. We must
    // sanitize the matrix edge with async copy. The optimization from
    // the unified GEMM kernel cannot be applied.
    //
    // Ideally, a client implementation would add a GEMMKernelDescriptor
    // property for whether in-place accumulation was enabled. When false,
    // the statements below are not part of the direct-access condition.
    // The code for loading C from memory would be elided at
    // code-generation time.
    //
    // MFA has settled on a function constant to toggle accumulation.
    output += " && (load_previous_C ? (M_offset == gid.y * M_group) : true)";
    output += " && (load_previous_C ? (N_offset == gid.x * N_group) : true)";
    source->SetValue("DIRECT_ACCESS_CONDITION", output);
  }

  *source += R"(

if ({{DIRECT_ACCESS_CONDITION}}) {
  // Fast path for matrices that qualify.
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
    C, {{LEADING_DIMENSION_C}}, C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
      C->{{LOAD_FUNCTION_C}}(C_dst, {{LEADING_DIMENSION_C}}, origin);
    }
  }
} else {
  // Slow path for when memory must be handled more carefully.
  auto C_block = (threadgroup {{MEMORY_NAME_C}}*)(threadgroup_block);
  auto C_block_dst =
  simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
    C_block, {{LEADING_BLOCK_DIMENSIONS_C}}, offset_in_group);
  
  // Launch the async copy from threadgroup to device memory.
  if (sidx == 0) {
    uint2 C_offset(N_offset, M_offset);
    ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                   min(uint(M_group), M - C_offset.y));
    auto C_dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
      C, {{LEADING_DIMENSION_C}}, C_offset);
    
    simdgroup_event event;
    event.async_copy<{{LEADING_BLOCK_DIMENSIONS_C}}, 32>(
      C_block, C_tile, C_dst, {{LEADING_DIMENSION_C}}, C_tile{{ASYNC_LANE_ID}});
    simdgroup_event::wait(1, &event);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Read the accumulator from threadgroup memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
      C->{{LOAD_FUNCTION_C}}(
        C_block_dst, {{LEADING_BLOCK_DIMENSIONS_C}}, origin);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

)";
}

void GEMMKernel::createStoreC(CodeWriter *source) const noexcept {
  if (memoryPrecisions.C == GEMMOperandPrecision::BF16 && registerPrecisions.C == GEMMOperandPrecision::FP32) {
    source->SetValue("STORE_FUNCTION_C", "store_bfloat");
  } else {
    source->SetValue("STORE_FUNCTION_C", "store");
  }
    
  *source += R"(

if ({{DIRECT_ACCESS_CONDITION}}) {
  // Fast path for matrices that qualify.
  uint2 C_offset(N_offset + offset_in_group.x,
                 M_offset + offset_in_group.y);
  auto C_dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
    C, {{LEADING_DIMENSION_C}}, C_offset);
  
  // Write the accumulator to device memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
      C->{{STORE_FUNCTION_C}}(C_dst, {{LEADING_DIMENSION_C}}, origin);
    }
  }
} else {
  // Slow path for when memory must be handled more carefully.
  auto C_block = (threadgroup {{MEMORY_NAME_C}}*)(threadgroup_block);
  auto C_block_dst =
  simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
    C_block, {{LEADING_BLOCK_DIMENSIONS_C}}, offset_in_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Write the accumulator to threadgroup memory.
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
      ushort2 origin(n, m);
      auto C = get_sram(C_sram, {{REGISTER_N}}, origin);
      C->{{STORE_FUNCTION_C}}(
        C_block_dst, {{LEADING_BLOCK_DIMENSIONS_C}}, origin);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Launch the async copy from threadgroup to device memory.
  if (sidx == 0) {
    uint2 C_offset(gid.x * N_group, gid.y * M_group);
    ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                   min(uint(M_group), M - C_offset.y));
    auto C_dst = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
      C, {{LEADING_DIMENSION_C}}, C_offset);
    
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
      C_block = simdgroup_matrix_storage<{{MEMORY_NAME_C}}>::apply_offset(
        C_block, {{LEADING_BLOCK_DIMENSIONS_C}}, C_block_shift);
    }
    
    simdgroup_event event;
    event.async_copy<{{LEADING_BLOCK_DIMENSIONS_C}}, 32>(
      C_dst, {{LEADING_DIMENSION_C}}, C_tile, C_block, C_tile{{ASYNC_LANE_ID}});
  }
}
)";
}

#pragma mark - Multiply

void GEMMKernel::createMultiplyIterations(CodeWriter *source) const noexcept {
  if (preferAsyncLoad) {
    source->SetValue("ASYNC_ITERATIONS_START", "0");
  } else {
    source->SetValue("ASYNC_ITERATIONS_START", "(K - (K % K_group))");
  }
  source->SetValue("PADDED_CEILING_K", "(K + K_remainder_padded - K_remainder)");
  source->SetValue("LEADING_DIMENSION_A", leadingDimension('A'));
  source->SetValue("LEADING_DIMENSION_B", leadingDimension('B'));
  source->SetValue("LEADING_BLOCK_DIMENSIONS_A", std::to_string(leadingBlockDimensions[0]));
  source->SetValue("LEADING_BLOCK_DIMENSIONS_B", std::to_string(leadingBlockDimensions[1]));
  source->SetValue("BLOCK_BYTES_A", std::to_string(blockBytes('A')));
  source->SetValue("REGISTER_M_8", std::to_string(registerM / 8));
  source->SetValue("REGISTER_N_8", std::to_string(registerN / 8));
    
  *source += R"(

uint k_start;
if (M_group <= M - M_offset && N_group <= N - N_offset) {
  k_start = {{ASYNC_ITERATIONS_START}};
} else {
  k_start = 0;
}
// Perform the iterations where async copy is avoided.
for (uint k = 0; k < k_start; k += 8) {
  uint2 A_offset(k, M_offset);
  uint2 B_offset(N_offset, k);
  A_offset += uint2(morton_offset.x, offset_in_group.y);
  B_offset += uint2(offset_in_group.x, morton_offset.y);

  auto A_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>::apply_offset(
    A, {{LEADING_DIMENSION_A}}, A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>::apply_offset(
    B, {{LEADING_DIMENSION_B}}, B_offset, B_trans);

  simdgroup_matrix_storage<{{REGISTER_NAME_A}}> A_sram[
    {{REGISTER_M_8}} * (8 / 8)];
  simdgroup_matrix_storage<{{REGISTER_NAME_B}}> B_sram[
    (8 / 8) * {{REGISTER_N_8}}];
  multiply_accumulate(A_src, B_src,
                      A_sram, B_sram, C_sram, 0);
}

// Perform the iterations where async copy is used.
for (uint k = k_start; k < K; k += K_group) {
  auto A_block = (threadgroup {{MEMORY_NAME_A}}*)(
    threadgroup_block);
  auto B_block = (threadgroup {{MEMORY_NAME_B}}*)(
    threadgroup_block + {{BLOCK_BYTES_A}});
  
  // Launch an async copy from device to threadgroup memory.
  if (sidx == 0) {
    uint2 A_offset(k, M_offset);
    uint2 B_offset(N_offset, k);
    auto A_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>::apply_offset(
      A, {{LEADING_DIMENSION_A}}, A_offset, A_trans);
    auto B_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>::apply_offset(
      B, {{LEADING_DIMENSION_B}}, B_offset, B_trans);

    ushort M_tile_dimension = min(uint(M_group), M - M_offset);
    ushort N_tile_dimension = min(uint(N_group), N - N_offset);
    ushort K_tile_dimension = min(uint(K_group), K - k);
    ushort K_tile_padded = min(uint(K_group), {{PADDED_CEILING_K}} - k);

    ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
    ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
    ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
    ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

    simdgroup_event events[2];
    events[0].async_copy<{{LEADING_BLOCK_DIMENSIONS_A}}, 32>(
      A_block, A_tile_dst, A_src, {{LEADING_DIMENSION_A}}, A_tile_src{{ASYNC_LANE_ID}}, A_trans);
    events[1].async_copy<{{LEADING_BLOCK_DIMENSIONS_B}}, 32>(
      B_block, B_tile_dst, B_src, {{LEADING_DIMENSION_B}}, B_tile_src{{ASYNC_LANE_ID}}, B_trans);
    simdgroup_event::wait(2, events);
  }
  
  ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
  ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
  auto A_block_src = A_block;
  auto B_block_src = B_block;
  A_block_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>::apply_offset(
    A_block_src, {{LEADING_BLOCK_DIMENSIONS_A}}, A_block_offset, A_trans);
  B_block_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>::apply_offset(
    B_block_src, {{LEADING_BLOCK_DIMENSIONS_B}}, B_block_offset, B_trans);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  simdgroup_matrix_storage<{{REGISTER_NAME_A}}> A_sram[
    {{REGISTER_M_8}} * (K_group / 8)];
  simdgroup_matrix_storage<{{REGISTER_NAME_B}}> B_sram[
    (K_group / 8) * {{REGISTER_N_8}}];
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
