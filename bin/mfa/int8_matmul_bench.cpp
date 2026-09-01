#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <QuartzCore/QuartzCore.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/GEMMHeaders.hpp"

namespace {

using half_float = _Float16;

struct Shape {
	const char* name;
	uint32_t rows_per_expert;
	uint32_t N;
	uint32_t K;
	uint32_t expert_count;
};

struct Pipelines {
	NS::SharedPtr<MTL::ComputePipelineState> quantize;
	NS::SharedPtr<MTL::ComputePipelineState> raw_cast;
	NS::SharedPtr<MTL::ComputePipelineState> matmul;
	NS::SharedPtr<MTL::ComputePipelineState> dequantize;
};

std::string direct_source()
{
	return createMetalSimdgroupMatrixStorage(false) + R"(
#include <metal_stdlib>
using namespace metal;

inline float reduce_max(float value, threadgroup float* scratch, ushort sgid, ushort lane_id)
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

kernel void quantize_activation(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x;
  if (row >= row_count)
    return;
  threadgroup float scratch[8];
  const ulong base = (ulong)row * K;
  float local_max = 0.0f;
  for (uint col = tid; col < K; col += 256)
    local_max = max(local_max, fabs(source[base + col]));
  const float max_abs = reduce_max(local_max, scratch, sgid, lane_id);
  const float scale = max_abs > 0.0f ? (128.0f * max_abs / 127.0f) : (128.0f / 127.0f);
  const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
  if (tid == 0)
    scales[row] = scale;
  for (uint col = tid; col < K; col += 256) {
    const int q = clamp((int)rint(source[base + col] * inv_scale), -127, 127);
    destination[base + col] = (half)q * half(0.0078125f);
  }
}

kernel void raw_cast_activation(
    device const float* source [[buffer(0)]],
    device half* destination [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
  const uint row = tgid.x;
  if (row >= row_count)
    return;
  const ulong base = (ulong)row * K;
  if (tid == 0)
    scales[row] = 1.0f;
  for (uint col = tid; col < K; col += 256)
    destination[base + col] = (half)source[base + col];
}

kernel void int8_matmul_direct(
    device const half* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint n0 = gid.x * 8;
  const uint m0 = gid.y * 16;
  if (n0 >= N || m0 >= M)
    return;
  const ushort2 morton = morton_order(lane_id);
  simdgroup_matrix_storage<float> accum[2];
  accum[0] = simdgroup_matrix_storage<float>(0);
  accum[1] = simdgroup_matrix_storage<float>(0);
  for (uint k0 = 0; k0 < K; k0 += 8) {
    const device half* A_lane = A + ulong(m0 + morton.y) * K + k0 + morton.x;
    const device char* B_lane = B + ulong(n0 + morton.x) * K + k0 + morton.y;
    simdgroup_matrix_storage<half> a_reg[2];
    simdgroup_matrix_storage<half> b_reg;
#pragma clang loop unroll(full)
    for (ushort m = 0; m < 2; ++m) {
      if (m0 + m * 8 < M)
        a_reg[m].load(A_lane, K, ushort2(0, m * 8), false);
      else
        a_reg[m] = simdgroup_matrix_storage<half>(0);
    }
    b_reg.load(B_lane, K, ushort2(0, 0), true);
    *b_reg.thread_elements() *= half(0.0078125f);
    accum[0].multiply(a_reg[0], b_reg);
    accum[1].multiply(a_reg[1], b_reg);
  }
  device float* C_lane = C + ulong(m0 + morton.y) * N + n0 + morton.x;
#pragma clang loop unroll(full)
  for (ushort m = 0; m < 2; ++m)
    if (m0 + m * 8 < M)
      accum[m].store(C_lane, N, ushort2(0, m * 8), false);
}

kernel void dequantize_output(
    device float* output [[buffer(0)]],
    device const float* activation_scales [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& rows_per_expert [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint3 position [[thread_position_in_grid]])
{
  const uint index = position.x;
  const ulong count = (ulong)row_count * N;
  if ((ulong)index < count) {
    const uint row = index / N;
    const uint col = index - row * N;
    const uint expert = row / rows_per_expert;
    output[index] *= activation_scales[row] * (128.0f * weight_scales[(ulong)expert * N + col]);
  }
}
)";
}

NS::SharedPtr<MTL::ComputePipelineState> make_pipeline(
	MTL::Library* const library, MTL::Device* const device, const char* const name)
{
	auto function_name = NS::String::string(name, NS::UTF8StringEncoding);
	auto function = NS::TransferPtr(library->newFunction(function_name));
	NS::Error* error = nullptr;
	auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
	if (!pipeline)
	{
		fprintf(stderr, "%s pipeline creation failed: %s\n", name,
			error ? error->localizedDescription()->utf8String() : "unknown error");
		exit(2);
	}
	return pipeline;
}

Pipelines make_pipelines(MTL::Device* const device)
{
	const std::string source = direct_source();
	auto source_string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
	NS::Error* error = nullptr;
	auto library = NS::TransferPtr(device->newLibrary(source_string, nullptr, &error));
	if (!library)
	{
		fprintf(stderr, "Metal library compilation failed: %s\n",
			error ? error->localizedDescription()->utf8String() : "unknown error");
		exit(2);
	}
	return {
		make_pipeline(library.get(), device, "quantize_activation"),
		make_pipeline(library.get(), device, "raw_cast_activation"),
		make_pipeline(library.get(), device, "int8_matmul_direct"),
		make_pipeline(library.get(), device, "dequantize_output"),
	};
}

void encode(
	MTL::ComputeCommandEncoder* const encoder,
	const Pipelines& pipelines,
	MTL::Buffer* const a_float,
	MTL::Buffer* const a_half,
	MTL::Buffer* const activation_scales,
	MTL::Buffer* const weights,
	MTL::Buffer* const weight_scales,
	MTL::Buffer* const c,
	const Shape shape,
	const bool quantized_activation)
{
	const uint32_t row_count = shape.expert_count * shape.rows_per_expert;
	encoder->setComputePipelineState(
		quantized_activation ? pipelines.quantize.get() : pipelines.raw_cast.get());
	encoder->setBuffer(a_float, 0, 0);
	encoder->setBuffer(a_half, 0, 1);
	encoder->setBuffer(activation_scales, 0, 2);
	encoder->setBytes(&row_count, sizeof(row_count), 3);
	encoder->setBytes(&shape.K, sizeof(shape.K), 4);
	encoder->dispatchThreadgroups(MTL::Size(row_count, 1, 1), MTL::Size(256, 1, 1));

	encoder->setComputePipelineState(pipelines.matmul.get());
	for (uint32_t expert = 0; expert < shape.expert_count; ++expert)
	{
		const size_t a_offset = (size_t)expert * shape.rows_per_expert * shape.K * sizeof(half_float);
		const size_t w_offset = (size_t)expert * shape.N * shape.K;
		const size_t c_offset = (size_t)expert * shape.rows_per_expert * shape.N * sizeof(float);
		encoder->setBuffer(a_half, a_offset, 0);
		encoder->setBuffer(weights, w_offset, 1);
		encoder->setBuffer(c, c_offset, 2);
		encoder->setBytes(&shape.rows_per_expert, sizeof(shape.rows_per_expert), 3);
		encoder->setBytes(&shape.N, sizeof(shape.N), 4);
		encoder->setBytes(&shape.K, sizeof(shape.K), 5);
		encoder->dispatchThreadgroups(
			MTL::Size((shape.N + 7) / 8, (shape.rows_per_expert + 15) / 16, 1),
			MTL::Size(32, 1, 1));
	}

	encoder->setComputePipelineState(pipelines.dequantize.get());
	encoder->setBuffer(c, 0, 0);
	encoder->setBuffer(activation_scales, 0, 1);
	encoder->setBuffer(weight_scales, 0, 2);
	encoder->setBytes(&row_count, sizeof(row_count), 3);
	encoder->setBytes(&shape.rows_per_expert, sizeof(shape.rows_per_expert), 4);
	encoder->setBytes(&shape.N, sizeof(shape.N), 5);
	const size_t output_count = (size_t)row_count * shape.N;
	encoder->dispatchThreads(MTL::Size(output_count, 1, 1), MTL::Size(256, 1, 1));
}

double benchmark(
	MTL::Device* const device,
	MTL::CommandQueue* const queue,
	const Pipelines& pipelines,
	const Shape shape,
	const int warmup,
	const int iterations,
	const bool quantized_activation)
{
	const uint32_t row_count = shape.expert_count * shape.rows_per_expert;
	const size_t a_float_size = (size_t)row_count * shape.K * sizeof(float);
	const size_t a_half_size = (size_t)row_count * shape.K * sizeof(half_float);
	const size_t a_scale_size = (size_t)row_count * sizeof(float);
	const size_t w_size = (size_t)shape.expert_count * shape.N * shape.K;
	const size_t w_scale_size = (size_t)shape.expert_count * shape.N * sizeof(float);
	const size_t c_size = (size_t)row_count * shape.N * sizeof(float);
	const MTL::ResourceOptions options = MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
	auto a_float = NS::TransferPtr(device->newBuffer(a_float_size, options));
	auto a_half = NS::TransferPtr(device->newBuffer(a_half_size, options));
	auto activation_scales = NS::TransferPtr(device->newBuffer(a_scale_size, options));
	auto weights = NS::TransferPtr(device->newBuffer(w_size, options));
	auto weight_scales = NS::TransferPtr(device->newBuffer(w_scale_size, options));
	auto c = NS::TransferPtr(device->newBuffer(c_size, options));
	if (!a_float || !a_half || !activation_scales || !weights || !weight_scales || !c)
	{
		fprintf(stderr, "%s allocation failed (A32 %.3f GiB, A16 %.3f GiB, W %.3f GiB, C %.3f GiB)\n",
			shape.name, (double)a_float_size / (1ull << 30), (double)a_half_size / (1ull << 30),
			(double)w_size / (1ull << 30), (double)c_size / (1ull << 30));
		exit(2);
	}
	std::vector<double> samples;
	for (int iteration = 0; iteration < warmup + iterations; ++iteration)
	{
		auto command_buffer = NS::RetainPtr(queue->commandBuffer());
		auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
		encode(encoder.get(), pipelines, a_float.get(), a_half.get(), activation_scales.get(),
			weights.get(), weight_scales.get(), c.get(), shape, quantized_activation);
		encoder->endEncoding();
		const double start = CACurrentMediaTime();
		command_buffer->commit();
		command_buffer->waitUntilCompleted();
		if (command_buffer->status() == MTL::CommandBufferStatusError)
		{
			NS::Error* const error = command_buffer->error();
			fprintf(stderr, "%s command failed: %s\n", shape.name,
				error ? error->localizedDescription()->utf8String() : "unknown error");
			exit(2);
		}
		if (iteration >= warmup)
			samples.push_back((CACurrentMediaTime() - start) * 1000);
	}
	if (samples.empty())
		return 0;
	std::sort(samples.begin(), samples.end());
	const double median = samples[samples.size() / 2];
	const double operations = 2.0 * row_count * shape.N * shape.K;
	printf("direct_%s/%s: experts=%u M/expert=%u N=%u K=%u weight=%.3f GiB median=%.4f ms %.3f TFLOP/s\n",
		quantized_activation ? "qfp16" : "raw_fp16",
		shape.name, shape.expert_count, shape.rows_per_expert, shape.N, shape.K,
		(double)w_size / (1ull << 30), median, operations / (median * 1e9));
	fflush(stdout);
	return median;
}

void validate(MTL::Device* const device, MTL::CommandQueue* const queue, const Pipelines& pipelines)
{
	const Shape shape = { "validation", 16, 8, 32, 1 };
	const size_t a_count = (size_t)shape.rows_per_expert * shape.K;
	const size_t w_count = (size_t)shape.N * shape.K;
	const size_t c_count = (size_t)shape.rows_per_expert * shape.N;
	std::vector<float> a(a_count);
	std::vector<int8_t> weights(w_count);
	std::vector<float> weight_scales(shape.N);
	std::vector<float> c(c_count, 0);
	for (size_t i = 0; i < a.size(); ++i)
		a[i] = (float)((int)(i % 31) - 15) / 32;
	for (uint32_t row = 0; row < shape.rows_per_expert; ++row)
		a[(size_t)row * shape.K] = (row & 1) ? -1e8f : 1e8f;
	for (size_t i = 0; i < weights.size(); ++i)
		weights[i] = (int8_t)((int)(i % 19) - 9);
	for (uint32_t i = 0; i < shape.N; ++i)
		weight_scales[i] = (float)(i + 1) / 256;
	auto a_float = NS::TransferPtr(device->newBuffer(a.data(), a.size() * sizeof(float), MTL::ResourceStorageModeShared));
	auto a_half = NS::TransferPtr(device->newBuffer(a.size() * sizeof(half_float), MTL::ResourceStorageModeShared));
	auto activation_scales = NS::TransferPtr(device->newBuffer(shape.rows_per_expert * sizeof(float), MTL::ResourceStorageModeShared));
	auto w_buffer = NS::TransferPtr(device->newBuffer(weights.data(), weights.size(), MTL::ResourceStorageModeShared));
	auto w_scale_buffer = NS::TransferPtr(device->newBuffer(weight_scales.data(), weight_scales.size() * sizeof(float), MTL::ResourceStorageModeShared));
	auto c_buffer = NS::TransferPtr(device->newBuffer(c.data(), c.size() * sizeof(float), MTL::ResourceStorageModeShared));
	auto command_buffer = NS::RetainPtr(queue->commandBuffer());
	auto encoder = NS::RetainPtr(command_buffer->computeCommandEncoder());
	encode(encoder.get(), pipelines, a_float.get(), a_half.get(), activation_scales.get(),
		w_buffer.get(), w_scale_buffer.get(), c_buffer.get(), shape, true);
	encoder->endEncoding();
	command_buffer->commit();
	command_buffer->waitUntilCompleted();
	if (command_buffer->status() == MTL::CommandBufferStatusError)
	{
		NS::Error* const error = command_buffer->error();
		fprintf(stderr, "validation command failed: %s\n",
			error ? error->localizedDescription()->utf8String() : "unknown error");
		exit(4);
	}
	memcpy(c.data(), c_buffer->contents(), c.size() * sizeof(float));
	double max_rel = 0;
	for (uint32_t m = 0; m < shape.rows_per_expert; ++m)
	{
		float max_abs = 0;
		for (uint32_t k = 0; k < shape.K; ++k)
			max_abs = std::max(max_abs, fabsf(a[(size_t)m * shape.K + k]));
		const float activation_scale = max_abs > 0 ? max_abs / 127 : 1.0f / 127;
		const float inv_scale = max_abs > 0 ? 127 / max_abs : 127;
		for (uint32_t n = 0; n < shape.N; ++n)
		{
			float dot = 0;
			for (uint32_t k = 0; k < shape.K; ++k)
			{
				const int q = std::max(-127, std::min(127, (int)rintf(a[(size_t)m * shape.K + k] * inv_scale)));
				dot += q * weights[(size_t)n * shape.K + k];
			}
			const float expected = dot * activation_scale * weight_scales[n];
			const float actual = c[(size_t)m * shape.N + n];
			if (!std::isfinite(actual))
				exit(3);
			const double diff = fabs((double)actual - expected);
			const double denom = std::max(1.0, std::max(fabs((double)actual), fabs((double)expected)));
			max_rel = std::max(max_rel, diff / denom);
		}
	}
	printf("validation: max_rel=%g\n", max_rel);
	if (max_rel > 1e-5)
		exit(3);
}

} // namespace

int main(int argc, char** argv)
{
	auto* pool = NS::AutoreleasePool::alloc()->init();
	auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!device)
		return 1;
	auto queue = NS::TransferPtr(device->newCommandQueue());
	const Pipelines pipelines = make_pipelines(device.get());
	printf("device: %s\n", device->name()->utf8String());
	fflush(stdout);
	validate(device.get(), queue.get(), pipelines);
	const int warmup = argc > 3 ? atoi(argv[3]) : 1;
	const int iterations = argc > 4 ? atoi(argv[4]) : 5;
	const uint32_t token_count = argc > 2 ? (uint32_t)atoi(argv[2]) : 2048;
	const uint32_t rows_per_expert = (token_count * 6 + 255) / 256;
	const Shape shared_gate = { "shared_gate_or_up", token_count, 2048, 4096, 1 };
	const Shape shared_down = { "shared_down", token_count, 4096, 2048, 1 };
	const Shape routed_gate = { "routed_gate_or_up", rows_per_expert, 2048, 4096, 256 };
	const Shape routed_down = { "routed_down", rows_per_expert, 4096, 2048, 256 };
	const std::string selected = argc > 1 ? argv[1] : "routed_gate";
	const bool quantized_activation = selected.rfind("raw_", 0) != 0;
	const std::string shape_name = quantized_activation ? selected : selected.substr(4);
	if (shape_name == "shared_gate")
		benchmark(device.get(), queue.get(), pipelines, shared_gate, warmup, iterations, quantized_activation);
	else if (shape_name == "shared_down")
		benchmark(device.get(), queue.get(), pipelines, shared_down, warmup, iterations, quantized_activation);
	else if (shape_name == "routed_gate")
		benchmark(device.get(), queue.get(), pipelines, routed_gate, warmup, iterations, quantized_activation);
	else if (shape_name == "routed_down")
		benchmark(device.get(), queue.get(), pipelines, routed_down, warmup, iterations, quantized_activation);
	else
		return 2;
	pool->release();
	return 0;
}
