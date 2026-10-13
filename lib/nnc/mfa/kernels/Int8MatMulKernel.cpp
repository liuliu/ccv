#include "Int8MatMulKernel.hpp"
#include "GEMMHeaders.hpp"
#include "CodeWriter.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <algorithm>

Int8MatMulKernel::Int8MatMulKernel(Int8MatMulKernelDescriptor descriptor, MTL::Device* const device)
{
	operation = descriptor.operation;
	ioPrecision = descriptor.ioPrecision;
	blockDimensions = descriptor.blockDimensions;
	splits = descriptor.splits;
	kInterleave = descriptor.kInterleave;
	useBias = descriptor.useBias;
	activationHadamard256 = descriptor.activationHadamard256;
	registerM = 0;
	registerN = 0;
	switch (operation) {
		case Int8MatMulDense:
			CCV_NNC_MFA_PRECONDITION(ioPrecision == GEMMOperandPrecision::FP32 || ioPrecision == GEMMOperandPrecision::FP16);
			CCV_NNC_MFA_PRECONDITION(kInterleave == 1 || kInterleave == 2 || kInterleave == 4 || kInterleave == 8);
			CCV_NNC_MFA_PRECONDITION(splits[0] > 0 && splits[1] > 0);
			CCV_NNC_MFA_PRECONDITION(blockDimensions[0] % (8 * splits[0]) == 0 && blockDimensions[1] % (8 * splits[1]) == 0);
			CCV_NNC_MFA_PRECONDITION(blockDimensions[2] > 0 && blockDimensions[2] % (8 * kInterleave) == 0);
			registerM = blockDimensions[0] / splits[0];
			registerN = blockDimensions[1] / splits[1];
			threadgroupSize = MTL::Size(32 * splits[0] * splits[1], 1, 1);
			break;
		case Int8MatMulSegmented:
			CCV_NNC_MFA_PRECONDITION(blockDimensions[0] > 0 && blockDimensions[0] % 8 == 0);
			threadgroupSize = MTL::Size(32, 1, 1);
			break;
		case Int8MatMulQuantizeActivation:
			CCV_NNC_MFA_PRECONDITION(ioPrecision == GEMMOperandPrecision::FP32 || ioPrecision == GEMMOperandPrecision::FP16);
			// The rotation assigns one 256-wide group to each of 8 simdgroups at a time.
			threadgroupSize = MTL::Size(256, 1, 1);
			break;
		case Int8MatMulRotateActivation:
			CCV_NNC_MFA_PRECONDITION(ioPrecision == GEMMOperandPrecision::FP16);
			// Each of 8 simdgroups rotates 4 of the threadgroup's 32 groups.
			threadgroupSize = MTL::Size(256, 1, 1);
			break;
		case Int8MatMulCastWeights:
		case Int8MatMulDequantizeOutput:
		case Int8MatMulDequantizeSegmentedOutput:
			threadgroupSize = MTL::Size(256, 1, 1);
			break;
	}
	source = createSource();
	auto sourceString = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(sourceString, nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}

const char* Int8MatMulKernel::functionName() const noexcept
{
	switch (operation) {
		case Int8MatMulQuantizeActivation:
			return "int8_matmul_quantize_activation";
		case Int8MatMulRotateActivation:
			return "int8_matmul_rotate_activation";
		case Int8MatMulCastWeights:
			return "int8_matmul_cast_weights";
		case Int8MatMulDequantizeOutput:
			return "int8_matmul_dequantize_output";
		case Int8MatMulDequantizeSegmentedOutput:
			return "int8_matmul_dequantize_segmented_output";
		case Int8MatMulSegmented:
			return "int8_matmul_segmented";
		case Int8MatMulDense:
			return "int8_matmul_dense";
	}
	return nullptr;
}

MTL::Size Int8MatMulKernel::threadgroupsPerGrid(const Int8MatMulDescriptor& descriptor) const noexcept
{
	auto ceilDivide = [](uint64_t target, uint64_t granularity) -> uint64_t {
		return (target + granularity - 1) / granularity;
	};
	switch (operation) {
		case Int8MatMulQuantizeActivation:
			return MTL::Size(descriptor.M, 1, 1);
		case Int8MatMulRotateActivation:
			return MTL::Size(ceilDivide((uint64_t)descriptor.M * descriptor.K, 256 * 32), 1, 1);
		case Int8MatMulCastWeights:
			return MTL::Size(ceilDivide((uint64_t)descriptor.expertCount * descriptor.N * descriptor.K, 256), 1, 1);
		case Int8MatMulDequantizeOutput:
			return MTL::Size(ceilDivide((uint64_t)descriptor.M * descriptor.N, 256), 1, 1);
		case Int8MatMulDequantizeSegmentedOutput:
			return MTL::Size(ceilDivide(descriptor.N, 256), descriptor.binCount, 1);
		case Int8MatMulSegmented:
			return MTL::Size(ceilDivide(descriptor.N, 8), descriptor.binCount, 1);
		case Int8MatMulDense:
			return MTL::Size(ceilDivide(descriptor.N, blockDimensions[1]), ceilDivide(descriptor.M, blockDimensions[0]), 1);
	}
	return MTL::Size(0, 0, 0);
}

#pragma mark - Source

std::string Int8MatMulKernel::createSource() const noexcept
{
	CodeWriter source;
	source += createMetalSimdgroupMatrixStorage(false);
	source += R"(
#include <metal_stdlib>
using namespace metal;

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];
constant uint expert_count [[function_constant(3)]];
constant uint bincount [[function_constant(4)]];
constant uint weight_count [[function_constant(5)]];
)";
	switch (operation) {
		case Int8MatMulQuantizeActivation:
			createQuantizeActivation(&source);
			break;
		case Int8MatMulRotateActivation:
			createRotateActivation(&source);
			break;
		case Int8MatMulCastWeights:
			createCastWeights(&source);
			break;
		case Int8MatMulDequantizeOutput:
			createDequantizeOutput(&source);
			break;
		case Int8MatMulDequantizeSegmentedOutput:
			createDequantizeSegmentedOutput(&source);
			break;
		case Int8MatMulSegmented:
			createSegmented(&source);
			break;
		case Int8MatMulDense:
			createDense(&source);
			break;
	}
	return source.ToString();
}

// Matches NAInt8MatMul's activation rotation, so rows rotate the same way with
// or without neural accelerators.
void Int8MatMulKernel::createRotate256(CodeWriter* source) const noexcept
{
	source->SetValue("MEMORY_NAME_IO", ioPrecision.name());
	*source += R"(
// One simdgroup rotates 256 consecutive features; lane i holds features
// 4i to 4i + 3 of each half, in rotated[0] and rotated[1].
// H4 has +1 everywhere except its anti-diagonal (-1). Its fourth Kronecker
// power, normalized by 1/16, is the Hadamard matrix the weights were
// rotated by.
inline void int8_matmul_rotate_256(device const {{MEMORY_NAME_IO}}* source,
                                   ushort lane_id,
                                   thread float4* rotated)
{
  float4 x[2];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < 2; ++j) {
    device const {{MEMORY_NAME_IO}}* features = source + j * 128 + lane_id * 4;
    float4 v = float4(features[0], features[1], features[2], features[3]);
    v = (v.x + v.y + v.z + v.w) - 2.0f * v.wzyx;
    #pragma clang loop unroll(full)
    for (ushort stride = 1; stride <= 4; stride *= 4) {
      const float4 a = simd_shuffle_xor(v, stride);
      const float4 b = simd_shuffle_xor(v, ushort(stride * 2));
      const float4 c = simd_shuffle_xor(v, ushort(stride * 3));
      v = ((v + a) + b) - c;
    }
    x[j] = v;
  }
  const float4 a = simd_shuffle_xor(x[0], 16);
  const float4 b = simd_shuffle_xor(x[1], 16);
  rotated[0] = (((x[0] + a) + x[1]) - b) * (1.0f / 16.0f);
  rotated[1] = (((x[1] + b) + x[0]) - a) * (1.0f / 16.0f);
}
)";
}

// Activations are quantized per row to int8 codes q in [-127, 127] and stored
// in FP16 as q / 128; the row's scale absorbs the factor of 128. Weight codes
// are used as is (the dense kernel), or as q / 128 (cast weights), in which
// case the epilogue restores another factor of 128.
void Int8MatMulKernel::createQuantizeActivation(CodeWriter* source) const noexcept
{
	source->SetValue("MEMORY_NAME_IO", ioPrecision.name());
	if (activationHadamard256)
		createRotate256(source);
	if (ioPrecision == GEMMOperandPrecision::FP16) {
		// Store the scale at IO precision, as NAInt8MatMul does, so FP16 rows
		// quantize the same way with or without neural accelerators.
		source->SetValue("CREATE_SCALE", R"(
  const float scale = 128.0f * float(half(max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f));
)");
	} else {
		source->SetValue("CREATE_SCALE", R"(
  const float scale = max_abs > 0.0f ? (128.0f * max_abs / 127.0f) : (128.0f / 127.0f);
)");
	}
	*source += R"(
inline float int8_matmul_reduce_max(float value,
                                  threadgroup float* scratch,
                                  ushort sgid,
                                  ushort lane_id)
{
  value = max(value, simd_shuffle_xor(value, 16));
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  value = max(value, simd_shuffle_xor(value, 1));
  if (lane_id == 0)
    scratch[sgid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    value = lane_id < 8 ? scratch[lane_id] : 0.0f;
    value = max(value, simd_shuffle_xor(value, 16));
    value = max(value, simd_shuffle_xor(value, 8));
    value = max(value, simd_shuffle_xor(value, 4));
    value = max(value, simd_shuffle_xor(value, 2));
    value = max(value, simd_shuffle_xor(value, 1));
    if (lane_id == 0)
      scratch[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}

kernel void int8_matmul_quantize_activation(
    device const {{MEMORY_NAME_IO}}* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device float* scales [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x;
  if (row >= M)
    return;
  threadgroup float scratch[8];
  const ulong base = (ulong)row * K;
  float local_max = 0.0f;
)";
	if (activationHadamard256) {
		*source += R"(
  // Each simdgroup rotates every eighth 256-wide group, and keeps the rotated
  // row in registers through the scale reduction.
  const uint groups_per_simd = (K / 256 + 7) / 8;
  // Metal cannot size a thread array with a function constant. Specialization
  // eliminates unused slots; the descriptor bounds K to 65536 at 256 threads.
  float4 rotated[64];
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    const uint group = g * 8 + sgid;
    if (group < K / 256) {
      int8_matmul_rotate_256(source + base + group * 256, lane_id, rotated + g * 2);
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 2; ++j) {
        const float4 v = abs(rotated[g * 2 + j]);
        local_max = max(local_max, max(max(v.x, v.y), max(v.z, v.w)));
      }
    }
  }
  const float max_abs = int8_matmul_reduce_max(local_max, scratch, sgid, lane_id);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
{{CREATE_SCALE}}
  if (tid == 0)
    scales[row] = scale;
  device half4* destination4 = reinterpret_cast<device half4*>(destination + base);
  #pragma clang loop unroll(full)
  for (uint g = 0; g < groups_per_simd; ++g) {
    const uint group = g * 8 + sgid;
    if (group < K / 256) {
      #pragma clang loop unroll(full)
      for (uint j = 0; j < 2; ++j) {
        const int4 rounded = int4(rint(rotated[g * 2 + j] * inv_scale));
        destination4[group * 64 + j * 32 + lane_id] = half4(clamp(rounded, int4(-127), int4(127))) * half(0.0078125f);
      }
    }
  }
}
)";
		return;
	}
	*source += R"(
  for (uint col = tid; col < K; col += 256)
    local_max = max(local_max, fabs(float(source[base + col])));
  const float max_abs = int8_matmul_reduce_max(local_max, scratch, sgid, lane_id);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
{{CREATE_SCALE}}
  if (tid == 0)
    scales[row] = scale;
  for (uint col = tid; col < K; col += 256) {
    const int rounded = (int)rint(float(source[base + col]) * inv_scale);
    destination[base + col] = (half)clamp(rounded, -127, 127) * half(0.0078125f);
  }
}
)";
}

// FP16 activations need no quantization; H256 weights only need them rotated.
// A rotated value is at most 1/16 of its group's absolute sum, so it fits in
// FP16 unless the group's activations average over 4094 in magnitude.
void Int8MatMulKernel::createRotateActivation(CodeWriter* source) const noexcept
{
	createRotate256(source);
	*source += R"(
kernel void int8_matmul_rotate_activation(
    device const half* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  // Every row holds whole groups, so group g starts at element g * 256. Four
  // groups per simdgroup keep more loads in flight than one: up to a third faster
  // on M3 Ultra, and the same on M4 Pro, where the pass is bandwidth bound.
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 4; ++i) {
    const uint group = (tgid.x * 4 + i) * 8 + sgid;
    if (group < M * (K / 256)) {
      float4 rotated[2];
      int8_matmul_rotate_256(source + (ulong)group * 256, lane_id, rotated);
      device half4* destination4 = reinterpret_cast<device half4*>(destination + (ulong)group * 256);
      destination4[lane_id] = half4(rotated[0]);
      destination4[32 + lane_id] = half4(rotated[1]);
    }
  }
}
)";
}

void Int8MatMulKernel::createCastWeights(CodeWriter* source) const noexcept
{
	*source += R"(
kernel void int8_matmul_cast_weights(
    device const char* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    uint3 position [[thread_position_in_grid]])
{
  const uint index = position.x;
  if (index < weight_count)
    destination[index] = (half)source[index] * half(0.0078125f);
}
)";
}

void Int8MatMulKernel::createDequantizeOutput(CodeWriter* source) const noexcept
{
	*source += R"(
kernel void int8_matmul_dequantize_output(
    device float* output [[buffer(0)]],
    device const float* activation_scales [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    uint3 position [[thread_position_in_grid]])
{
  const uint index = position.x;
  const ulong count = (ulong)M * N;
  if ((ulong)index < count) {
    const uint row = index / N;
    const uint col = index - row * N;
    output[index] *= activation_scales[row] * (128.0f * weight_scales[col]);
  }
}
)";
}

void Int8MatMulKernel::createDequantizeSegmentedOutput(CodeWriter* source) const noexcept
{
	*source += R"(
kernel void int8_matmul_dequantize_segmented_output(
    device float* output [[buffer(0)]],
    device const float* activation_scales [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    device const int* indices [[buffer(3)]],
    device const int* counts [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  const uint bin = tgid.y;
  if (bin >= bincount)
    return;
  const int expert = indices[bin];
  const int count_i = counts[bin];
  if (expert < 0 || expert >= (int)expert_count || count_i <= 0)
    return;
  uint row_offset = 0;
  for (uint i = 0; i < bin; ++i)
    row_offset += (uint)max(counts[i], 0);
  const uint col = tgid.x * 256 + tid;
  if (col >= N)
    return;
  const float weight_scale = 128.0f * weight_scales[(ulong)expert * N + col];
  for (uint row = 0; row < (uint)count_i; ++row) {
    const ulong output_index = (ulong)(row_offset + row) * N + col;
    output[output_index] *= activation_scales[row_offset + row] * weight_scale;
  }
}
)";
}

void Int8MatMulKernel::createSegmented(CodeWriter* source) const noexcept
{
	source->SetValue("SEGMENTED_BLOCK_M", std::to_string(blockDimensions[0]));
	*source += R"(
kernel void int8_matmul_segmented(
    device const half* A [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const int* counts [[buffer(2)]],
    device const char* B [[buffer(3)]],
    device float* C [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint bin = gid.y;
  if (bin >= bincount)
    return;
  const int expert = indices[bin];
  const int count_i = counts[bin];
  if (expert < 0 || expert >= (int)expert_count || count_i <= 0)
    return;
  uint row_offset = 0;
  for (uint i = 0; i < bin; ++i)
    row_offset += (uint)max(counts[i], 0);
  const uint n0 = gid.x * 8;
  if (n0 >= N)
    return;
  const ushort2 morton = morton_order(lane_id);
  const device char* B_base = B + (ulong)expert * N * K;
  const uint full_count = (uint)count_i & ~({{SEGMENTED_BLOCK_M}} - 1u);
  for (uint m0 = 0; m0 < full_count; m0 += {{SEGMENTED_BLOCK_M}}) {
    simdgroup_matrix_storage<float> accum[{{SEGMENTED_BLOCK_M}} / 8];
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m)
      accum[m] = simdgroup_matrix_storage<float>(0);
    for (uint k0 = 0; k0 < K; k0 += 8) {
      const device char* B_lane = B_base + (ulong)(n0 + morton.x) * K + k0 + morton.y;
      simdgroup_matrix_storage<half> a_reg[{{SEGMENTED_BLOCK_M}} / 8];
      simdgroup_matrix_storage<half> b_reg;
#pragma clang loop unroll(full)
      for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m) {
        const uint row = m0 + m * 8 + morton.y;
        const device half* A_lane = A + (ulong)(row_offset + row) * K + k0 + morton.x;
        a_reg[m].load(A_lane, K, ushort2(0, 0), false);
      }
      b_reg.load(B_lane, K, ushort2(0, 0), true);
      *b_reg.thread_elements() *= half(0.0078125f);
#pragma clang loop unroll(full)
      for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m)
        accum[m].multiply(a_reg[m], b_reg);
    }
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m) {
      const uint row = m0 + m * 8 + morton.y;
      device float* C_lane = C + (ulong)(row_offset + row) * N + n0 + morton.x;
      accum[m].store(C_lane, N, ushort2(0, 0), false);
    }
  }
  if (full_count < (uint)count_i) {
    simdgroup_matrix_storage<float> accum[{{SEGMENTED_BLOCK_M}} / 8];
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m)
      accum[m] = simdgroup_matrix_storage<float>(0);
    for (uint k0 = 0; k0 < K; k0 += 8) {
      const device char* B_lane = B_base + (ulong)(n0 + morton.x) * K + k0 + morton.y;
      simdgroup_matrix_storage<half> a_reg[{{SEGMENTED_BLOCK_M}} / 8];
      simdgroup_matrix_storage<half> b_reg;
#pragma clang loop unroll(full)
      for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m) {
        const uint row = full_count + m * 8 + morton.y;
        if (row < (uint)count_i) {
          const ulong address = (ulong)(row_offset + row) * K + k0 + morton.x;
          *a_reg[m].thread_elements() = half2(A[address], A[address + 1]);
        } else {
          a_reg[m] = simdgroup_matrix_storage<half>(0);
        }
      }
      b_reg.load(B_lane, K, ushort2(0, 0), true);
      *b_reg.thread_elements() *= half(0.0078125f);
#pragma clang loop unroll(full)
      for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m)
        accum[m].multiply(a_reg[m], b_reg);
    }
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{SEGMENTED_BLOCK_M}} / 8; ++m) {
      const uint row = full_count + m * 8 + morton.y;
      if (row < (uint)count_i) {
        const float2 values = *accum[m].thread_elements();
        const ulong address = (ulong)(row_offset + row) * N + n0 + morton.x;
        C[address] = values.x;
        C[address + 1] = values.y;
      }
    }
  }
}
)";
}

#pragma mark - Dense

// Only FP16 activations are multiplied as they are. Any other precision (FP32,
// or BF16 if supported) is quantized to FP16 codes with row-wise scales, so the
// multiplication stays in FP16.
static bool denseUsesActivationScales(const GEMMOperandPrecision ioPrecision) noexcept
{
	return ioPrecision != GEMMOperandPrecision::FP16;
}

void Int8MatMulKernel::createDense(CodeWriter* source) const noexcept
{
	source->SetValue("MEMORY_NAME_IO", ioPrecision.name());
	source->SetValue("BLOCK_DIMENSIONS_M", std::to_string(blockDimensions[0]));
	source->SetValue("BLOCK_DIMENSIONS_N", std::to_string(blockDimensions[1]));
	source->SetValue("BLOCK_DIMENSIONS_K", std::to_string(blockDimensions[2]));
	source->SetValue("REGISTER_M", std::to_string(registerM));
	source->SetValue("REGISTER_N", std::to_string(registerN));
	source->SetValue("REGISTER_M_8", std::to_string(registerM / 8));
	source->SetValue("REGISTER_N_8", std::to_string(registerN / 8));
	source->SetValue("SPLITS_N", std::to_string(splits[1]));
	source->SetValue("K_INTERLEAVE", std::to_string(kInterleave));
	source->SetValue("BIAS_ARGUMENT", useBias ? "                              device const " + ioPrecision.name() + " *bias [[buffer(5)]]," : "");
	source->SetValue("A_SCALES_ARGUMENT", denseUsesActivationScales(ioPrecision) ? "                              device const float *A_scales [[buffer(3)]]," : "");
	*source += R"(
// Row-wise int8 matrix multiplication, C = (A * B^T) * A_scales * B_scales + bias.
//
// A: M x K FP16 activations. FP16 IO reads them as they are (or rotated by
// int8_matmul_rotate_activation), without A_scales; other IO precisions read
// their FP16 codes and A_scales from int8_matmul_quantize_activation.
// B: N x K int8 weight codes; the right-hand side is B^T.
// C: M x N output. A_scales are per row of C; B_scales and bias per column.
//
// Each simdgroup accumulates a {{REGISTER_M}} x {{REGISTER_N}} tile in FP32 straight
// from device memory, like GEMMKernel on Apple9 and later. Loads at the matrix
// edges are clamped into bounds and only valid elements are stored, so M and N
// are arbitrary; K must be a multiple of 8.
kernel void int8_matmul_dense(device const half *A [[buffer(0)]],
                              device const char *B [[buffer(1)]],
                              device {{MEMORY_NAME_IO}} *C [[buffer(2)]],
{{A_SCALES_ARGUMENT}}
                              device const {{MEMORY_NAME_IO}} *B_scales [[buffer(4)]],
{{BIAS_ARGUMENT}}
                              uint3 gid [[threadgroup_position_in_grid]],
                              ushort sidx [[simdgroup_index_in_threadgroup]],
                              ushort lane_id [[thread_index_in_simdgroup]])
{
  const ushort2 sid(sidx % {{SPLITS_N}}, sidx / {{SPLITS_N}});
  const ushort2 morton_offset = morton_order(lane_id);
  const uint M_offset = gid.y * {{BLOCK_DIMENSIONS_M}} + sid.y * {{REGISTER_M}};
  const uint N_offset = gid.x * {{BLOCK_DIMENSIONS_N}} + sid.x * {{REGISTER_N}};
  // There are no barriers, so a simdgroup past the edge can exit.
  if (M_offset >= M || N_offset >= N)
    return;

  // The row of A, and the two rows of B, each thread reads per 8 x 8 block.
  uint A_offsets[{{REGISTER_M_8}}];
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
    A_offsets[m] = min(M_offset + m * 8 + morton_offset.y, M - 1) * K;
  uint B_offsets[{{REGISTER_N_8}}][2];
#pragma clang loop unroll(full)
  for (ushort n = 0; n < {{REGISTER_N_8}}; ++n) {
    const uint column = N_offset + n * 8 + morton_offset.x;
    B_offsets[n][0] = min(column, N - 1) * K;
    B_offsets[n][1] = min(column + 1, N - 1) * K;
  }

  simdgroup_matrix_storage<float> C_sram[{{REGISTER_M_8}}][{{REGISTER_N_8}}];
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
      C_sram[m][n] = simdgroup_matrix_storage<float>(0);
)";
	createDenseMultiplyIterations(source);
	createDenseStoreC(source);
	*source += "}\n";
}

void Int8MatMulKernel::createDenseMultiplyIterations(CodeWriter* source) const noexcept
{
	// Each load covers kInterleave slices of 8 columns: two adjacent elements of
	// a row of A (the thread's 8 x 8 layout) per slice, and one element in each
	// of two rows of B. Assign column k0 + kInterleave * c + j to slice j, so a
	// thread's elements for every slice are adjacent in memory.
	const unsigned short activationWidth = std::min(4, 2 * kInterleave);
	const unsigned short weightWidth = std::min(4, (int)kInterleave);
	auto vectorName = [](const std::string& scalar, unsigned short width) {
		return width == 1 ? scalar : "vec<" + scalar + ", " + std::to_string(width) + ">";
	};
	auto element = [](const std::string& array, const std::string& index, unsigned short width) {
		if (width == 1)
			return array + "[" + index + "]";
		return array + "[(" + index + ") / " + std::to_string(width) + "][(" + index + ") % " + std::to_string(width) + "]";
	};
	source->SetValue("ACTIVATION_VECTOR", vectorName("half", activationWidth));
	source->SetValue("ACTIVATION_WIDTH", std::to_string(activationWidth));
	source->SetValue("ACTIVATION_LOADS", std::to_string(2 * kInterleave / activationWidth));
	source->SetValue("ACTIVATION_ELEMENT_0", element("A_loads[m]", "j", activationWidth));
	source->SetValue("ACTIVATION_ELEMENT_1", element("A_loads[m]", std::to_string(kInterleave) + " + j", activationWidth));
	source->SetValue("WEIGHT_VECTOR", vectorName("char", weightWidth));
	source->SetValue("WEIGHT_WIDTH", std::to_string(weightWidth));
	source->SetValue("WEIGHT_LOADS", std::to_string(kInterleave / weightWidth));
	source->SetValue("WEIGHT_ELEMENT_0", element("B_loads[n][0]", "j", weightWidth));
	source->SetValue("WEIGHT_ELEMENT_1", element("B_loads[n][1]", "j", weightWidth));
	*source += R"(
  const uint K_floor = K - K % {{BLOCK_DIMENSIONS_K}};
  for (uint k = 0; k < K_floor; k += {{BLOCK_DIMENSIONS_K}}) {
#pragma clang loop unroll(full)
    for (ushort k_slices = 0; k_slices < {{BLOCK_DIMENSIONS_K}}; k_slices += 8 * {{K_INTERLEAVE}}) {
      {{ACTIVATION_VECTOR}} A_loads[{{REGISTER_M_8}}][{{ACTIVATION_LOADS}}];
#pragma clang loop unroll(full)
      for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
#pragma clang loop unroll(full)
        for (ushort l = 0; l < {{ACTIVATION_LOADS}}; ++l)
          A_loads[m][l] = *(const device {{ACTIVATION_VECTOR}} *)(A + A_offsets[m] + k + k_slices + {{K_INTERLEAVE}} * morton_offset.x + l * {{ACTIVATION_WIDTH}});
      {{WEIGHT_VECTOR}} B_loads[{{REGISTER_N_8}}][2][{{WEIGHT_LOADS}}];
#pragma clang loop unroll(full)
      for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
#pragma clang loop unroll(full)
        for (ushort r = 0; r < 2; ++r)
#pragma clang loop unroll(full)
          for (ushort l = 0; l < {{WEIGHT_LOADS}}; ++l)
            B_loads[n][r][l] = *(const device {{WEIGHT_VECTOR}} *)(B + B_offsets[n][r] + k + k_slices + {{K_INTERLEAVE}} * morton_offset.y + l * {{WEIGHT_WIDTH}});
#pragma clang loop unroll(full)
      for (ushort j = 0; j < {{K_INTERLEAVE}}; ++j) {
        simdgroup_matrix_storage<half> A_sram[{{REGISTER_M_8}}];
        simdgroup_matrix_storage<half> B_sram[{{REGISTER_N_8}}];
#pragma clang loop unroll(full)
        for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
          *A_sram[m].thread_elements() = half2({{ACTIVATION_ELEMENT_0}}, {{ACTIVATION_ELEMENT_1}});
#pragma clang loop unroll(full)
        for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
          *B_sram[n].thread_elements() = half2(half({{WEIGHT_ELEMENT_0}}), half({{WEIGHT_ELEMENT_1}}));
#pragma clang loop unroll(full)
        for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
#pragma clang loop unroll(full)
          for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
            C_sram[m][n].multiply(A_sram[m], B_sram[n]);
      }
    }
  }

  // The remaining columns, one 8 x 8 block at a time.
  for (uint k = K_floor; k < K; k += 8) {
    simdgroup_matrix_storage<half> A_sram[{{REGISTER_M_8}}];
    simdgroup_matrix_storage<half> B_sram[{{REGISTER_N_8}}];
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
      *A_sram[m].thread_elements() = *(const device half2 *)(A + A_offsets[m] + k + morton_offset.x);
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
      *B_sram[n].thread_elements() = half2(half(B[B_offsets[n][0] + k + morton_offset.y]), half(B[B_offsets[n][1] + k + morton_offset.y]));
#pragma clang loop unroll(full)
    for (ushort m = 0; m < {{REGISTER_M_8}}; ++m)
#pragma clang loop unroll(full)
      for (ushort n = 0; n < {{REGISTER_N_8}}; ++n)
        C_sram[m][n].multiply(A_sram[m], B_sram[n]);
  }
)";
}

void Int8MatMulKernel::createDenseStoreC(CodeWriter* source) const noexcept
{
	source->SetValue("BIAS_0", useBias ? " + float(bias[column])" : "");
	source->SetValue("BIAS_1", useBias ? " + float(bias[column + 1])" : "");
	const bool activationScales = denseUsesActivationScales(ioPrecision);
	source->SetValue("LOAD_A_SCALE", activationScales ? "    const float A_scale = A_scales[row];\n" : "");
	source->SetValue("A_SCALE", activationScales ? " * A_scale" : "");
	*source += R"(
#pragma clang loop unroll(full)
  for (ushort m = 0; m < {{REGISTER_M_8}}; ++m) {
    const uint row = M_offset + m * 8 + morton_offset.y;
    if (row >= M)
      continue;
{{LOAD_A_SCALE}}#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N_8}}; ++n) {
      const uint column = N_offset + n * 8 + morton_offset.x;
      const float2 C_values = *C_sram[m][n].thread_elements(){{A_SCALE}};
      if (column < N)
        C[row * N + column] = {{MEMORY_NAME_IO}}(C_values[0] * float(B_scales[column]){{BIAS_0}});
      if (column + 1 < N)
        C[row * N + column + 1] = {{MEMORY_NAME_IO}}(C_values[1] * float(B_scales[column + 1]){{BIAS_1}});
    }
  }
)";
}
