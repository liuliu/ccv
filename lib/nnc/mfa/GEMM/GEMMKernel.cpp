#include "GEMMKernel.hpp"
#include "GEMMHeaders.hpp"
#include "../ccv_nnc_mfa.hpp"

GEMMKernel::GEMMKernel(GEMMKernelDescriptor descriptor) {
  CCV_NNC_MFA_PRECONDITION(descriptor.blockDimensions.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.device.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.memoryPrecisions.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.preferAsyncStore.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.registerPrecisions.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.splits.has_value());
  CCV_NNC_MFA_PRECONDITION(descriptor.transposeState.has_value());
  auto blockDimensions = descriptor.blockDimensions.value();
  auto device = descriptor.device.value();
  auto memoryPrecisions = descriptor.memoryPrecisions.value();
  auto preferAsyncStore = descriptor.preferAsyncStore.value();
  auto registerPrecisions = descriptor.registerPrecisions.value();
  auto splits = descriptor.splits.value();
  auto transposeState = descriptor.transposeState.value();
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
  
  // Inject the contents of the headers.
  source += createMetalSimdgroupEvent() + "\n";
  source += createMetalSimdgroupMatrixStorage() + "\n";
  source += "using namespace metal;\n";
  source += "\n";
  
  // Declare the size of M and N within a register allocation.
  uint16_t registerM = blockDimensions[0] / splits[0];
  uint16_t registerN = blockDimensions[1] / splits[1];
  
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
    paddedBlockDimensionsA = { blockDimensions[0], blockDimensions[1] };
    paddedBlockDimensionsB = { blockDimensions[2], blockDimensions[3] };
    paddedBlockDimensionsC = { blockDimensions[4], blockDimensions[5] };
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
  source += "constant ushort M_remainder = (M % ";
  source += std::to_string(registerM) + " == 0)\n";
  source += "  ? " + std::to_string(registerM);
  source += " : M % " + std::to_string(registerM) + ";\n";
  
  source += "constant ushort N_remainder = (N % ";
  source += std::to_string(registerN) + " == 0)\n";
  source += "  ? " + std::to_string(registerN);
  source += " : N % " + std::to_string(registerN) + ";\n";
  
  source += "constant ushort K_remainder = (K % K_group == 0) \n";
  source += "  ? K_group : K % K_group;\n";
  source += "constant ushort K_remainder_padded = ";
  source += "(K_remainder + 7) / 8 * 8;\n";
  
  // Shift the final block, so it doesn't access out-of-bounds memory.
  source += "constant ushort M_shift = (M < M_group) ";
  source += "? 0 : " + std::to_string(registerM) + " - M_remainder;\n";
  source += "constant ushort N_shift = (N < N_group) ";
  source += "? 0 : " + std::to_string(registerM) + " - N_remainder;\n";
  
  // Compile the shader source.
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}
