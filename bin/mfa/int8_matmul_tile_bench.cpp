// Tile sweep for the dense Int8MatMul without neural accelerators (Apple9:
// M3 / M4), against the GEMM routes it replaces.
//
// Candidates are Int8MatMulKernelDescriptor configurations: the threadgroup
// tile (block M / N), simdgroup splits, the K extent of one main-loop iteration
// and the K interleave of the loads. "auto" is the configuration the kernel
// descriptor selects for this device (--cores overrides the core count it sees).
//
// Baselines use the shipping MFA kernels:
// - fp16_gemm: FP16 GEMM on already dequantized weights (FP32 registers).
// - dequant: the row-wise int8 -> FP16 weight dequantization that precedes it.
// - cast_gemm: Int8MatMul's GEMM route (FP16 weight copy, FP32 GEMM output,
//   scale epilogue).
//
// Usage:
//   int8_matmul_tile_bench [--float] [--baselines] [--sweep tiles|occupancy]
//     [--configs BMxBN:SMxSN:BK:I,...] [--shapes MxNxK,...] [--cores N]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/GEMMHeaders.hpp"
#include "nnc/mfa/kernels/GEMMDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernel.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseDescriptor.hpp"
#include "nnc/mfa/kernels/Dequantize8iRowwiseKernel.hpp"
#include "nnc/mfa/kernels/Int8MatMulDescriptor.hpp"
#include "nnc/mfa/kernels/Int8MatMulKernelDescriptor.hpp"
#include "nnc/mfa/kernels/Int8MatMulKernel.hpp"

namespace {

using half_t = _Float16;

struct Shape {
	uint32_t M, N, K;
};

struct Config {
	simd::ushort3 blockDimensions;
	simd::ushort2 splits;
	uint16_t kInterleave;
	bool automatic;
};

std::string configName(const Int8MatMulKernelDescriptor& kernel, bool automatic)
{
	char buffer[96];
	snprintf(buffer, sizeof(buffer), "%s%ux%u_s%ux%u_k%u_i%u", automatic ? "auto=" : "",
		kernel.blockDimensions[0], kernel.blockDimensions[1], kernel.splits[0], kernel.splits[1],
		kernel.blockDimensions[2], kernel.kInterleave);
	return buffer;
}

struct Bench {
	NS::SharedPtr<MTL::Device> device;
	NS::SharedPtr<MTL::CommandQueue> queue;
	NS::SharedPtr<MTL::Library> aux;
	std::unordered_map<GEMMKernelDescriptor, std::unique_ptr<GEMMKernel>> gemmLibraries;
	std::unordered_map<Dequantize8iRowwiseKernelDescriptor, std::unique_ptr<Dequantize8iRowwiseKernel>> dequantLibraries;
	std::unordered_map<Int8MatMulKernelDescriptor, std::unique_ptr<Int8MatMulKernel>> int8Libraries;
	DeviceProperties dprops = { 0 };
	double targetMs = 30;
	int samples = 5;
};

uint32_t gpuCoreCount(MTL::Device* const device)
{
	int cores = 0;
	const io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, IORegistryEntryIDMatching(device->registryID()));
	if (service)
	{
		CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
		if (value && CFGetTypeID(value) == CFNumberGetTypeID())
			CFNumberGetValue((CFNumberRef)value, kCFNumberIntType, &cores);
		if (value)
			CFRelease(value);
		IOObjectRelease(service);
	}
	return cores > 0 ? (uint32_t)cores : 10;
}

// The parts of Int8MatMul's GEMM route outside MFA GEMM.
const char* kCastSource = R"(
#include <metal_stdlib>
using namespace metal;
constant uint count [[function_constant(0)]];
constant uint row_length [[function_constant(1)]];

kernel void cast_weights(device const char* src [[buffer(0)]], device half* dst [[buffer(1)]], uint i [[thread_position_in_grid]])
{
  if (i < count)
    dst[i] = (half)src[i] * half(0.0078125f);
}

kernel void dequantize_output_half(device half* out [[buffer(0)]], device const float* acc [[buffer(1)]],
    device const float* a_scales [[buffer(2)]], device const half* w_scales [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  if (i < count) {
    const uint row = i / row_length;
    const uint col = i - row * row_length;
    out[i] = half(acc[i] * a_scales[row] * (128.0f * float(w_scales[col])));
  }
}

kernel void dequantize_output_float(device float* out [[buffer(0)]],
    device const float* a_scales [[buffer(2)]], device const float* w_scales [[buffer(3)]], uint i [[thread_position_in_grid]])
{
  if (i < count) {
    const uint row = i / row_length;
    const uint col = i - row * row_length;
    out[i] *= a_scales[row] * (128.0f * w_scales[col]);
  }
}
)";

NS::SharedPtr<MTL::ComputePipelineState> auxPipeline(Bench& bench, const char* name, uint32_t count, uint32_t rowLength)
{
	auto values = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	values->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
	values->setConstantValue(&rowLength, MTL::DataTypeUInt, NS::UInteger(1));
	NS::Error* error = nullptr;
	auto function = NS::TransferPtr(bench.aux->newFunction(NS::String::string(name, NS::UTF8StringEncoding), values.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	auto pipeline = NS::TransferPtr(bench.device->newComputePipelineState(function.get(), &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	return pipeline;
}

// Median GPU milliseconds per repetition of `encode`.
template <typename Encode>
double timeGPU(Bench& bench, Encode encode)
{
	auto run = [&](int repetitions) {
		auto commandBuffer = NS::RetainPtr(bench.queue->commandBuffer());
		auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
		for (int i = 0; i < repetitions; ++i)
			encode(encoder.get());
		encoder->endEncoding();
		commandBuffer->commit();
		commandBuffer->waitUntilCompleted();
		if (commandBuffer->status() == MTL::CommandBufferStatusError)
		{
			fprintf(stderr, "command buffer failed\n");
			exit(3);
		}
		return (commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000.0 / repetitions;
	};
	const double probe = run(1);
	run(1);
	const int repetitions = std::max(1, std::min(200, (int)std::ceil(bench.targetMs / std::max(probe, 1e-3))));
	std::vector<double> times;
	for (int i = 0; i < bench.samples; ++i)
		times.push_back(run(repetitions));
	std::sort(times.begin(), times.end());
	return times[times.size() / 2];
}

// Activation codes are q / 128 in FP16 and their scales 128 * s, as
// int8_matmul_quantize_activation writes them.
struct Buffers {
	NS::SharedPtr<MTL::Buffer> a, b, aScales, bScales, c, bHalf, accumulator;
	std::vector<int8_t> qa, qb;
	std::vector<float> sa, sb;
};

Buffers makeBuffers(Bench& bench, const Shape& shape, bool ioHalf)
{
	Buffers buffers;
	std::mt19937 rng(shape.M * 31 + shape.N * 17 + shape.K);
	std::uniform_int_distribution<int> q(-127, 127);
	std::uniform_real_distribution<float> u(0.5f, 1.5f);
	buffers.qa.resize((size_t)shape.M * shape.K);
	buffers.qb.resize((size_t)shape.N * shape.K);
	for (auto& v : buffers.qa)
		v = (int8_t)q(rng);
	for (auto& v : buffers.qb)
		v = (int8_t)q(rng);
	buffers.sa.resize(shape.M);
	buffers.sb.resize(shape.N);
	for (auto& v : buffers.sa)
		v = u(rng) / 127.0f;
	for (auto& v : buffers.sb)
		v = (float)(half_t)(u(rng) / (127.0f * std::sqrt((float)shape.K)));
	const auto shared = MTL::ResourceStorageModeShared;
	std::vector<half_t> a(buffers.qa.size());
	for (size_t i = 0; i < a.size(); ++i)
		a[i] = (half_t)((float)buffers.qa[i] / 128.0f);
	buffers.a = NS::TransferPtr(bench.device->newBuffer(a.data(), a.size() * sizeof(half_t), shared));
	buffers.b = NS::TransferPtr(bench.device->newBuffer(buffers.qb.data(), buffers.qb.size(), shared));
	std::vector<float> aScales(shape.M);
	for (uint32_t i = 0; i < shape.M; ++i)
		aScales[i] = buffers.sa[i] * 128.0f;
	buffers.aScales = NS::TransferPtr(bench.device->newBuffer(aScales.data(), aScales.size() * sizeof(float), shared));
	if (ioHalf)
	{
		std::vector<half_t> bScales(shape.N);
		for (uint32_t i = 0; i < shape.N; ++i)
			bScales[i] = (half_t)buffers.sb[i];
		buffers.bScales = NS::TransferPtr(bench.device->newBuffer(bScales.data(), bScales.size() * sizeof(half_t), shared));
	} else
		buffers.bScales = NS::TransferPtr(bench.device->newBuffer(buffers.sb.data(), buffers.sb.size() * sizeof(float), shared));
	buffers.c = NS::TransferPtr(bench.device->newBuffer((size_t)shape.M * shape.N * (ioHalf ? sizeof(half_t) : sizeof(float)), shared));
	return buffers;
}

// The max relative error over sampled outputs (negative if one is not finite).
double validate(const Buffers& buffers, const Shape& shape, bool ioHalf)
{
	std::vector<std::pair<uint32_t, uint32_t>> points;
	std::mt19937 rng(7);
	for (int i = 0; i < 192; ++i)
		points.push_back({ rng() % shape.M, rng() % shape.N });
	for (uint32_t r : { 0u, shape.M - 1, shape.M / 2 })
		for (uint32_t c : { 0u, shape.N - 1, shape.N / 2 })
			points.push_back({ r, c });
	double maxRel = 0;
	for (auto [r, c] : points)
	{
		int64_t dot = 0;
		for (uint32_t k = 0; k < shape.K; ++k)
			dot += (int64_t)buffers.qa[(size_t)r * shape.K + k] * buffers.qb[(size_t)c * shape.K + k];
		const double want = (double)dot * buffers.sa[r] * buffers.sb[c];
		const size_t index = (size_t)r * shape.N + c;
		const double got = ioHalf ? (double)((const half_t*)buffers.c->contents())[index] : (double)((const float*)buffers.c->contents())[index];
		if (!std::isfinite(got))
			return -1;
		maxRel = std::max(maxRel, std::fabs(got - want) / std::max(1e-2, std::fabs(want)));
	}
	return maxRel;
}

double runInt8MatMul(Bench& bench, const Shape& shape, Buffers& buffers, const Int8MatMulKernelDescriptor& kernelDescriptor, bool ioHalf, double* relOut)
{
	const Int8MatMulDescriptor descriptor = {
		.M = shape.M,
		.N = shape.N,
		.K = shape.K,
		.expertCount = 1,
		.binCount = 0,
		.operation = Int8MatMulDense,
		.ioPrecision = ioHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32,
	};
	auto* pipeline = descriptor.findKernel(bench.device.get(), kernelDescriptor, nullptr, nullptr, "", &bench.int8Libraries).second;
	const MTL::Size grid = pipeline->kernel->threadgroupsPerGrid(descriptor);
	const MTL::Size threads = pipeline->kernel->threadgroupSize;
	memset(buffers.c->contents(), 0xff, buffers.c->length());
	const double ms = timeGPU(bench, [&](MTL::ComputeCommandEncoder* encoder) {
		encoder->setComputePipelineState(pipeline->pipeline.get());
		encoder->setBuffer(buffers.a.get(), 0, 0);
		encoder->setBuffer(buffers.b.get(), 0, 1);
		encoder->setBuffer(buffers.c.get(), 0, 2);
		encoder->setBuffer(buffers.aScales.get(), 0, 3);
		encoder->setBuffer(buffers.bScales.get(), 0, 4);
		encoder->dispatchThreadgroups(grid, threads);
	});
	*relOut = validate(buffers, shape, ioHalf);
	delete pipeline;
	return ms;
}

PipelineValue<GEMMKernel>* gemmPipeline(Bench& bench, const Shape& shape, GEMMOperandPrecision cPrecision)
{
	GEMMDescriptor desc;
	desc.batchDimension = 1;
	desc.matrixDimensions = simd::uint3 { shape.M, shape.N, shape.K };
	desc.memoryPrecisions = { .A = GEMMOperandPrecision::FP16, .B = GEMMOperandPrecision::FP16, .C = cPrecision, .bias = GEMMOperandPrecision::FP16 };
	desc.registerPrecisionC = GEMMOperandPrecision::FP32;
	desc.transposeState = simd::uchar3 { 0, 1, 0 };
	desc.loadPreviousC = false;
	desc.useBias = false;
	desc.loadM = false;
	desc.supportIndirectCommandBuffers = false;
	return desc.findKernel(bench.device.get(), bench.dprops, nullptr, nullptr, "", &bench.gemmLibraries).second;
}

void encodeGEMM(MTL::ComputeCommandEncoder* encoder, PipelineValue<GEMMKernel>* gemm, const Shape& shape, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* c)
{
	encoder->setComputePipelineState(gemm->pipeline.get());
	encoder->setThreadgroupMemoryLength(gemm->kernel->threadgroupMemoryAllocation, 0);
	encoder->setBuffer(a, 0, 0);
	encoder->setBuffer(b, 0, 1);
	encoder->setBuffer(c, 0, 2);
	encoder->dispatchThreadgroups(
		MTL::Size((shape.N + gemm->kernel->blockDimensions[1] - 1) / gemm->kernel->blockDimensions[1],
			(shape.M + gemm->kernel->blockDimensions[0] - 1) / gemm->kernel->blockDimensions[0], 1),
		MTL::Size(gemm->kernel->threadgroupSize, 1, 1));
}

void runBaselines(Bench& bench, const Shape& shape, Buffers& buffers, bool ioHalf, const char* prefix)
{
	const double flops = 2.0 * shape.M * shape.N * shape.K;
	if (!buffers.bHalf)
		buffers.bHalf = NS::TransferPtr(bench.device->newBuffer((size_t)shape.N * shape.K * sizeof(half_t), MTL::ResourceStorageModePrivate));
	// FP16 GEMM on dequantized weights, and the dequantization.
	auto* gemm = gemmPipeline(bench, shape, ioHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32);
	const double gemmMs = timeGPU(bench, [&](MTL::ComputeCommandEncoder* encoder) {
		encodeGEMM(encoder, gemm, shape, buffers.a.get(), buffers.bHalf.get(), buffers.c.get());
	});
	const size_t payload = ((size_t)shape.N * shape.K + 127) & ~(size_t)127;
	auto packed = NS::TransferPtr(bench.device->newBuffer(payload + (size_t)shape.N * sizeof(half_t), MTL::ResourceStorageModePrivate));
	Dequantize8iRowwiseDescriptor dequantDesc;
	dequantDesc.memoryPrecision = GEMMOperandPrecision::FP16;
	dequantDesc.rowLength = shape.K;
	dequantDesc.length = shape.N * shape.K;
	auto* dequant = dequantDesc.findKernel(bench.device.get(), bench.dprops, nullptr, nullptr, "", &bench.dequantLibraries).second;
	const double dequantMs = timeGPU(bench, [&](MTL::ComputeCommandEncoder* encoder) {
		encoder->setComputePipelineState(dequant->pipeline.get());
		encoder->setBuffer(packed.get(), 0, 0);
		encoder->setBuffer(buffers.bHalf.get(), 0, 1);
		encoder->setBuffer(packed.get(), payload, 2);
		encoder->dispatchThreadgroups(dequant->kernel->gridSize(dequantDesc.length), dequant->kernel->threadgroupSize);
	});
	printf("%s,baseline_fp16_gemm,%.4f,%.3f,0\n", prefix, gemmMs, flops / gemmMs / 1e9);
	printf("%s,baseline_dequant,%.4f,0,0\n", prefix, dequantMs);
	printf("%s,baseline_dequant_gemm,%.4f,%.3f,0\n", prefix, gemmMs + dequantMs, flops / (gemmMs + dequantMs) / 1e9);
	// Int8MatMul's GEMM route.
	auto* castGEMM = gemmPipeline(bench, shape, GEMMOperandPrecision::FP32);
	if (ioHalf && !buffers.accumulator)
		buffers.accumulator = NS::TransferPtr(bench.device->newBuffer((size_t)shape.M * shape.N * sizeof(float), MTL::ResourceStorageModePrivate));
	const uint32_t weightCount = shape.N * shape.K;
	const uint32_t outputCount = shape.M * shape.N;
	auto cast = auxPipeline(bench, "cast_weights", weightCount, shape.K);
	auto epilogue = auxPipeline(bench, ioHalf ? "dequantize_output_half" : "dequantize_output_float", outputCount, shape.N);
	MTL::Buffer* const accumulator = ioHalf ? buffers.accumulator.get() : buffers.c.get();
	const double castMs = timeGPU(bench, [&](MTL::ComputeCommandEncoder* encoder) {
		encoder->setComputePipelineState(cast.get());
		encoder->setBuffer(buffers.b.get(), 0, 0);
		encoder->setBuffer(buffers.bHalf.get(), 0, 1);
		encoder->dispatchThreadgroups(MTL::Size((weightCount + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
		encodeGEMM(encoder, castGEMM, shape, buffers.a.get(), buffers.bHalf.get(), accumulator);
		encoder->setComputePipelineState(epilogue.get());
		encoder->setBuffer(buffers.c.get(), 0, 0);
		if (ioHalf)
			encoder->setBuffer(buffers.accumulator.get(), 0, 1);
		encoder->setBuffer(buffers.aScales.get(), 0, 2);
		encoder->setBuffer(buffers.bScales.get(), 0, 3);
		encoder->dispatchThreadgroups(MTL::Size((outputCount + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
	});
	printf("%s,baseline_cast_gemm,%.4f,%.3f,0\n", prefix, castMs, flops / castMs / 1e9);
	delete gemm;
	delete castGEMM;
	delete dequant;
	fflush(stdout);
}

std::vector<Shape> parseShapes(const std::string& text)
{
	std::vector<Shape> shapes;
	size_t start = 0;
	while (start < text.size())
	{
		size_t end = text.find(',', start);
		if (end == std::string::npos)
			end = text.size();
		Shape shape;
		if (sscanf(text.substr(start, end - start).c_str(), "%ux%ux%u", &shape.M, &shape.N, &shape.K) == 3)
			shapes.push_back(shape);
		start = end + 1;
	}
	return shapes;
}

std::vector<Config> parseConfigs(const std::string& text)
{
	std::vector<Config> configs;
	size_t start = 0;
	while (start < text.size())
	{
		size_t end = text.find(',', start);
		if (end == std::string::npos)
			end = text.size();
		unsigned bm, bn, sm, sn, bk, interleave;
		if (sscanf(text.substr(start, end - start).c_str(), "%ux%u:%ux%u:%u:%u", &bm, &bn, &sm, &sn, &bk, &interleave) == 6)
			configs.push_back({ simd::ushort3 { (uint16_t)bm, (uint16_t)bn, (uint16_t)bk }, simd::ushort2 { (uint16_t)sm, (uint16_t)sn }, (uint16_t)interleave, false });
		start = end + 1;
	}
	return configs;
}

// Register tiles (per simdgroup) x splits x inner loops. "tiles" covers GEMM's
// Apple9 tile (32 x 32, one simdgroup) and Apple7 / 8 tiles (48 x 48 over
// 2 x 2 simdgroups) along with wider and taller tiles; "occupancy" the small
// tiles that win when each core gets few of them.
std::vector<Config> sweepConfigs(const std::string& name)
{
	std::vector<std::pair<uint16_t, uint16_t>> registers;
	std::vector<std::pair<uint16_t, uint16_t>> splits;
	std::vector<std::pair<uint16_t, uint16_t>> loops;
	if (name == "tiles")
	{
		registers = { { 16, 16 }, { 16, 32 }, { 32, 16 }, { 32, 32 }, { 24, 24 }, { 48, 48 }, { 32, 64 }, { 64, 32 }, { 64, 64 }, { 40, 40 }, { 48, 32 }, { 32, 48 } };
		splits = { { 1, 1 }, { 1, 2 }, { 2, 1 }, { 2, 2 } };
		loops = { { 32, 4 }, { 64, 4 }, { 64, 1 } };
	} else if (name == "occupancy") {
		registers = { { 8, 8 }, { 8, 16 }, { 16, 8 }, { 8, 32 }, { 16, 16 }, { 16, 32 }, { 32, 16 }, { 24, 24 }, { 32, 24 }, { 24, 32 }, { 32, 32 } };
		splits = { { 1, 1 }, { 2, 2 } };
		loops = { { 64, 1 }, { 64, 2 }, { 64, 4 } };
	} else if (name == "loops") {
		registers = { { 32, 32 } };
		splits = { { 1, 1 } };
		for (uint16_t interleave : { 1, 2, 4, 8 })
			for (uint16_t kGroup : { 8, 16, 32, 64, 128 })
				if (kGroup % (8 * interleave) == 0)
					loops.push_back({ kGroup, interleave });
	}
	std::vector<Config> configs;
	for (auto [rm, rn] : registers)
		for (auto [sm, sn] : splits)
			for (auto [kGroup, interleave] : loops)
				configs.push_back({ simd::ushort3 { (uint16_t)(rm * sm), (uint16_t)(rn * sn), kGroup }, simd::ushort2 { sm, sn }, interleave, false });
	return configs;
}

} // namespace

int main(int argc, char** argv)
{
	auto* pool = NS::AutoreleasePool::alloc()->init();
	Bench bench;
	bench.device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	bench.queue = NS::TransferPtr(bench.device->newCommandQueue());
	NS::Error* error = nullptr;
	bench.aux = NS::TransferPtr(bench.device->newLibrary(NS::String::string(kCastSource, NS::UTF8StringEncoding), nullptr, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
	bench.dprops.coreCount = gpuCoreCount(bench.device.get());
	std::string shapesText = "9276x6144x6144";
	std::string configsText;
	std::string sweep;
	bool ioHalf = true;
	bool baselines = false;
	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--shapes" && i + 1 < argc)
			shapesText = argv[++i];
		else if (arg == "--configs" && i + 1 < argc)
			configsText = argv[++i];
		else if (arg == "--sweep" && i + 1 < argc)
			sweep = argv[++i];
		else if (arg == "--float")
			ioHalf = false;
		else if (arg == "--baselines")
			baselines = true;
		else if (arg == "--cores" && i + 1 < argc)
			bench.dprops.coreCount = (uint32_t)atoi(argv[++i]);
		else if (arg == "--target-ms" && i + 1 < argc)
			bench.targetMs = atof(argv[++i]);
		else if (arg == "--samples" && i + 1 < argc)
			bench.samples = atoi(argv[++i]);
	}
	std::vector<Config> configs = parseConfigs(configsText);
	const std::vector<Config> swept = sweepConfigs(sweep);
	configs.insert(configs.end(), swept.begin(), swept.end());
	configs.insert(configs.begin(), Config { simd::ushort3 { 0, 0, 0 }, simd::ushort2 { 0, 0 }, 0, true });
	printf("device,%s,cores,%u\n", bench.device->name()->utf8String(), bench.dprops.coreCount);
	printf("shape,io,kernel,ms,tflops,max_rel\n");
	fflush(stdout);
	const GEMMOperandPrecision ioPrecision = ioHalf ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32;
	for (const Shape& shape : parseShapes(shapesText))
	{
		auto shapePool = NS::AutoreleasePool::alloc()->init();
		Buffers buffers = makeBuffers(bench, shape, ioHalf);
		const double flops = 2.0 * shape.M * shape.N * shape.K;
		char prefix[64];
		snprintf(prefix, sizeof(prefix), "%ux%ux%u,%s", shape.M, shape.N, shape.K, ioHalf ? "f16" : "f32");
		if (baselines)
			runBaselines(bench, shape, buffers, ioHalf, prefix);
		for (const Config& config : configs)
		{
			auto configPool = NS::AutoreleasePool::alloc()->init();
			Int8MatMulKernelDescriptor kernel(Int8MatMulDense, ioPrecision, config.blockDimensions, config.splits, config.kInterleave, false);
			if (config.automatic)
			{
				const Int8MatMulDescriptor descriptor = { .M = shape.M, .N = shape.N, .K = shape.K, .expertCount = 1, .binCount = 0, .operation = Int8MatMulDense, .ioPrecision = ioPrecision };
				kernel = Int8MatMulKernelDescriptor(descriptor, bench.dprops);
			}
			double rel = 0;
			const double ms = runInt8MatMul(bench, shape, buffers, kernel, ioHalf, &rel);
			printf("%s,%s,%.4f,%.3f,%.3g\n", prefix, configName(kernel, config.automatic).c_str(), ms, flops / ms / 1e9, rel);
			fflush(stdout);
			configPool->release();
		}
		shapePool->release();
	}
	pool->release();
	return 0;
}
