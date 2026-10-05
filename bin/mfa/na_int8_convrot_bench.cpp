// Fused regular H256 activation quantization: correctness and paired GPU timings.
// See na_int8_convrot_bench.md for the transform contract and shape sources.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"

namespace {
using Precision = GEMMOperandPrecision;
using Pipeline = PipelineValue<NAInt8MatMulKernel>;
using Cache = std::unordered_map<NAInt8MatMulKernelDescriptor, std::unique_ptr<NAInt8MatMulKernel>>;

void require(bool condition, const char* message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

size_t element_size(Precision p) { return p == Precision::FP32 ? 4 : 2; }

void store_value(void* data, size_t i, float value, Precision p)
{
  if (p == Precision::FP32)
    static_cast<float*>(data)[i] = value;
  else if (p == Precision::FP16)
    static_cast<_Float16*>(data)[i] = value;
  else {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    static_cast<uint16_t*>(data)[i] = bits >> 16;
  }
}

float read_value(const void* data, size_t i, Precision p)
{
  if (p == Precision::FP32)
    return static_cast<const float*>(data)[i];
  if (p == Precision::FP16)
    return static_cast<const _Float16*>(data)[i];
  const uint32_t bits = uint32_t(static_cast<const uint16_t*>(data)[i]) << 16;
  float value;
  std::memcpy(&value, &bits, 4);
  return value;
}

NS::SharedPtr<MTL::Buffer> buffer(MTL::Device* device, size_t bytes)
{
  auto b = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
  require(b.get() != nullptr, "Buffer allocation failed");
  std::memset(b->contents(), 0, bytes);
  return b;
}

std::unique_ptr<Pipeline> pipeline(MTL::Device* device, Cache& cache, const NAInt8MatMulDescriptor& d, bool register_operands = false)
{
  auto kernel = NAInt8MatMulKernelDescriptor(d, DeviceProperties());
  if (register_operands) {
    kernel.useRegisterOperands = true;
    kernel.blockDimensions = simd::ushort3 {64, 128, 32};
    kernel.executionSIMDGroups = 8;
    kernel.groupM = kernel.groupN = 0;
  }
  return std::unique_ptr<Pipeline>(d.findKernel(device, kernel, nullptr, nullptr, "", &cache).second);
}

void validate_cache(MTL::Device* device)
{
  // Equal dynamic descriptors must select the same source configuration on
  // every device profile, including the M-dependent address admission limits.
  size_t checked = 0;
  for (uint32_t cores : {0u, 10u, 35u, 36u, 40u, 48u, 64u, 80u, 128u})
    for (auto shape : {simd::uint2{1024, 4096}, {1536, 6144}, {2048, 8192},
        {2176, 8448}, {2432, 9216}, {2560, 8192}, {2560, 9216}, {4096, 12288},
        {6144, 24576}, {16384, 4096}, {262272, 16384}, {2147483500u, 4096},
        {1536, 0}, {2560, 65536}}) {
      std::unordered_map<NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor> ranges;
      std::vector<uint32_t> rows;
      for (uint32_t M = 1; M <= 4352; ++M) rows.push_back(M);
      for (uint64_t boundary : {uint64_t(UINT32_MAX) / std::max(1u, shape[0]) + 1,
          uint64_t(UINT32_MAX) / std::max(1u, shape[1]) + 1,
          uint64_t(INT32_MAX) - 62, uint64_t(UINT32_MAX)})
        for (int offset : {-1, 0, 1})
          if (boundary + offset > 0 && boundary + offset <= UINT32_MAX)
            rows.push_back(boundary + offset);
      for (uint32_t M : rows) {
        NAInt8MatMulDescriptor d;
        d.matrixDimensions = simd::uint3{M, shape[0], shape[1]};
        d.loadM = true;
        const auto config = NAInt8MatMulKernelDescriptor(d, DeviceProperties{cores});
        auto previous = ranges.emplace(d, config);
        require(previous.second || previous.first->second == config,
          "Equal dynamic descriptors select different source configurations");
        d.loadM = false;
        auto fixed = NAInt8MatMulKernelDescriptor(d, DeviceProperties{cores});
        fixed.loadM = true;
        require(fixed == config, "Dynamic and fixed M select different configurations");
        ++checked;
      }
    }

  // Compile representative entries in both call orders. Check actual cache
  // hits against uncached selection, not just descriptor equality in isolation.
  for (uint32_t cores : {10u, 36u, 40u, 80u})
    for (bool reverse : {false, true}) {
      ShaderCache cache;
      auto find = [&](const NAInt8MatMulDescriptor& d) {
        return cache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(
          d, device, DeviceProperties{cores});
      };
      for (auto shape : {simd::uint2{2560, 9216}, {2176, 8448}, {2560, 8192}, {1536, 6144}}) {
        std::vector<uint32_t> rows = {511, 512, 513, 575, 576, 577, 2457, 2458, 4095, 4096, 4097};
        if (reverse) std::reverse(rows.begin(), rows.end());
        for (uint32_t M : rows) {
          NAInt8MatMulDescriptor d;
          d.matrixDimensions = simd::uint3{M, shape[0], shape[1]}; d.loadM = true;
          const auto expected = NAInt8MatMulKernelDescriptor(d, DeviceProperties{cores});
          auto actual = find(d)->kernel;
          require(actual->useRegisterOperands == expected.useRegisterOperands &&
              simd_all(actual->blockDimensions == expected.blockDimensions) &&
              actual->groupM == expected.groupM,
            "Cached dynamic-M configuration depends on call order");
        }
      }
      NAInt8MatMulDescriptor d;
      d.matrixDimensions = simd::uint3{513, 2560, 9216}; d.loadM = true;
      auto first = find(d);
      d.matrixDimensions[0] = 576;
      require(find(d) == first, "Dynamic-M pipeline stores exact M within a tile-count range");
      d.matrixDimensions[0] = 2458;
      first = find(d);
      d.matrixDimensions[0] = 4095;
      require(find(d) == first, "Large-output dynamic-M range does not share its pipeline");
      d.matrixDimensions[0] = 4096;
      first = find(d);
      d.matrixDimensions[0] = 4097;
      require(find(d) == first, "Large-M traversal range does not share its pipeline");
      d.matrixDimensions[2] += 32;
      auto k_tail = find(d);
      require(k_tail != first && k_tail->kernel == first->kernel,
        "K specialization does not share the source kernel");
      d.matrixDimensions = simd::uint3{512, 2560, 9216}; d.loadM = false;
      first = find(d);
      require(first->kernel->useRegisterOperands == (cores >= 40), "Fixed-M register selection changed");
      d.matrixDimensions[0] = 513;
      require(find(d) != first, "Static-M pipeline omits M");
    }
  // Optional presence changes source / constants even at sentinel values.
  NAInt8MatMulDescriptor d;
  d.matrixDimensions = simd::uint3{511, 2560, 9216}; d.loadM = true;
  auto equal = d;
  equal.batchStrides = simd::uint4(UINT32_MAX);
  equal.leadingDimensions = simd::uint2(UINT32_MAX);
  require(!(d == equal), "Explicit strides alias absent optional strides");
  equal = d; equal.matrixDimensions[0] = 512;
  require(!(d == equal) && std::hash<NAInt8MatMulDescriptor>{}(d) != std::hash<NAInt8MatMulDescriptor>{}(equal),
    "Dynamic-M selection boundary omitted from equality/hash");
  std::cout << "validation descriptor_cache configurations=" << checked << " PASS\n";
}

// Independent dense FP64 reference: directly form each entry of H4^tensor4.
// This intentionally doesn't mirror the SIMD butterfly implementation.
std::vector<float> rotate(const std::vector<float>& values, uint32_t K)
{
  require(K > 0 && K % 256 == 0 && values.size() % K == 0, "Invalid CPU rotation shape");
  static const int h4[4][4] = {{1,1,1,-1}, {1,1,-1,1}, {1,-1,1,1}, {-1,1,1,1}};
  std::vector<float> result(values.size());
  for (size_t base = 0; base < values.size(); base += 256)
    for (uint32_t j = 0; j < 256; ++j) {
      double sum = 0;
      for (uint32_t k = 0; k < 256; ++k) {
        int sign = 1;
        for (uint32_t shift = 0; shift < 8; shift += 2)
          sign *= h4[(k >> shift) & 3][(j >> shift) & 3];
        sum += values[base + k] * sign;
      }
      result[base + j] = sum / 16;
    }
  return result;
}

struct Buffers {
  NS::SharedPtr<MTL::Buffer> a, aq, as, bq, bs, c, bias;
};

// mode: 0 = quantize, 1 = GEMM, 2 = quantize + GEMM. Repetitions include
// dependency barriers through separate encoders, matching production staging.
double run(MTL::CommandQueue* queue, const NAInt8MatMulDescriptor& d,
    Pipeline* quant, Pipeline* gemm, const Buffers& b, uint32_t actual_m,
    int mode, int repeats = 1)
{
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  const uint32_t K = d.matrixDimensions[2], N = d.matrixDimensions[1];
  const auto strides = d.batchStrides.value_or(simd::uint4(0));
  uint32_t dims[] = { actual_m, strides[0], strides[2],
    d.packedABatchStride.value_or(actual_m * K), d.aScaleBatchStride.value_or(actual_m) };
  auto command = NS::RetainPtr(queue->commandBuffer());
  for (int r = 0; r < repeats; ++r) {
    if (mode != 1) {
      auto enc = NS::RetainPtr(command->computeCommandEncoder());
      enc->setComputePipelineState(quant->second.get());
      enc->setBuffer(b.a.get(), 0, 0);
      enc->setBuffer(b.aq.get(), 0, 1);
      enc->setBuffer(b.as.get(), 0, 2);
      if (d.loadM)
        enc->setBytes(dims, sizeof(dims), 3);
      enc->dispatchThreadgroups(MTL::Size(d.matrixDimensions[0], 1, d.batchDimension),
        MTL::Size(quant->kernel->activationQuantizeThreads, 1, 1));
      enc->endEncoding();
    }
    if (mode != 0) {
      auto enc = NS::RetainPtr(command->computeCommandEncoder());
      enc->setComputePipelineState(gemm->pipeline.get());
      enc->setBuffer(b.aq.get(), 0, 0);
      enc->setBuffer(b.bq.get(), 0, 1);
      enc->setBuffer(b.c.get(), 0, 2);
      enc->setBuffer(b.as.get(), 0, 3);
      enc->setBuffer(b.bs.get(), 0, 4);
      if (d.useBias)
        enc->setBuffer(b.bias.get(), 0, 5);
      if (d.loadM)
        enc->setBytes(dims, sizeof(dims), d.useBias ? 6 : 5);
      enc->dispatchThreadgroups(gemm->kernel->threadgroupsPerGrid(actual_m, N, d.batchDimension),
        gemm->kernel->threadsPerThreadgroup(gemm->pipeline.get()));
      enc->endEncoding();
    }
  }
  command->commit();
  command->waitUntilCompleted();
  if (command->error())
    std::cerr << command->error()->localizedDescription()->utf8String() << '\n';
  require(command->status() == MTL::CommandBufferStatusCompleted, "GPU execution failed");
  return (command->GPUEndTime() - command->GPUStartTime()) * 1000 / repeats;
}

void validate(MTL::Device* device, MTL::CommandQueue* queue, Cache& cache,
    Precision precision, uint32_t K, bool dynamic, bool strided, bool register_operands = false)
{
  const uint32_t M = register_operands ? 67 : K > 2304 ? 7 : 19;
  const uint32_t N = register_operands ? 137 : K > 2304 ? 19 : 137, batches = strided ? 2 : 1;
  const uint32_t actual_m = dynamic ? M - 2 : M;
  const uint32_t lda = K + (strided ? 4 : 0), ldc = N + (strided ? 5 : 0);
  const uint32_t stride_a = M * lda + 8, stride_aq = M * K + 256, stride_as = M + 3;
  const uint32_t stride_c = M * ldc + 9;
  const size_t bytes = element_size(precision);
  NAInt8MatMulDescriptor d;
  d.matrixDimensions = simd::uint3 {M, N, K};
  d.ioPrecision = precision;
  d.batchDimension = batches;
  d.loadM = dynamic;
  d.useBias = strided || (register_operands && dynamic);
  d.activationHadamard256 = true;
  if (strided) {
    d.leadingDimensions = simd::uint2 {lda, ldc};
    d.batchStrides = simd::uint4 {stride_a, N * K, stride_c, N};
    d.packedABatchStride = stride_aq;
    d.aScaleBatchStride = stride_as;
  }
  auto p = pipeline(device, cache, d, register_operands);
  // A growing K must reuse the source object and specialize register storage via
  // function constants. Alternating the flag must produce a separate cache entry.
  auto plain_d = d;
  plain_d.activationHadamard256 = false;
  require(!(d == plain_d), "Descriptor omits rotation");
  auto plain = pipeline(device, cache, plain_d, register_operands);
  require(p->kernel != plain->kernel, "Kernel cache aliases rotation modes");
  const size_t cache_size = cache.size();
  auto larger_d = d;
  larger_d.matrixDimensions[2] = K == 65536 ? K - 256 : K + 256;
  if (strided)
    larger_d.leadingDimensions = simd::uint2 {larger_d.matrixDimensions[2] + 4, ldc};
  auto larger = pipeline(device, cache, larger_d, register_operands);
  require(larger->kernel == p->kernel && cache.size() == cache_size, "Kernel cache stores shape");
  Buffers b {buffer(device, batches * stride_a * bytes), buffer(device, batches * stride_aq),
    buffer(device, batches * stride_as * bytes), buffer(device, batches * N * K),
    buffer(device, batches * N * bytes), buffer(device, batches * stride_c * bytes),
    buffer(device, batches * N * bytes)};
  std::memset(b.aq->contents(), 0x55, b.aq->length());
  std::mt19937 rng(42 + K);
  std::normal_distribution<float> normal(0, 0.25f);
  std::vector<float> a(batches * M * K), w(batches * N * K);
  for (uint32_t batch = 0; batch < batches; ++batch) {
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        float value = row == 0 ? 0 : row == 1 ? 0.5f : normal(rng);
        if (row > 1 && k % 256 == 17)
          value += 12;
        const size_t i = batch * stride_a + row * lda + k;
        store_value(b.a->contents(), i, value, precision);
        a[(batch * M + row) * K + k] = read_value(b.a->contents(), i, precision);
      }
  }
  for (float& value : w)
    value = normal(rng);
  const auto ar = rotate(a, K), wr = rotate(w, K);
  double errors[2] = {0, 0}, signal = 0;
  for (int rotated = 0; rotated < 2; ++rotated) {
    const auto& weights = rotated ? wr : w;
    for (uint32_t row = 0; row < batches * N; ++row) {
      float maximum = 0;
      for (uint32_t k = 0; k < K; ++k)
        maximum = std::max(maximum, std::abs(weights[row * K + k]));
      store_value(b.bs->contents(), row, maximum / 127, precision);
      for (uint32_t k = 0; k < K; ++k)
        static_cast<int8_t*>(b.bq->contents())[row * K + k] = std::lrint(weights[row * K + k] * (127 / maximum));
      store_value(b.bias->contents(), row, (int(row % 7) - 3) * 0.02f, precision);
    }
    auto exec_d = d;
    exec_d.activationHadamard256 = rotated;
    run(queue, exec_d, rotated ? p.get() : plain.get(), rotated ? p.get() : plain.get(), b, actual_m, 2);
    const auto& reference = rotated ? ar : a;
    double worst_quant = 0, worst_matmul = 0;
    for (uint32_t batch = 0; batch < batches; ++batch) {
      const size_t qbase = batches > 1 ? batch * stride_aq : 0;
      const size_t sbase = batches > 1 ? batch * stride_as : 0;
      for (uint32_t row = 0; row < actual_m; ++row) {
        float maximum = 0;
        for (uint32_t k = 0; k < K; ++k)
          maximum = std::max(maximum, std::abs(reference[(batch * M + row) * K + k]));
        const float inv = maximum > 0 ? 127 / maximum : 127;
        uint32_t scale_bits = 0;
        store_value(&scale_bits, 0, maximum > 0 ? maximum / 127 : 1.0f / 127, precision);
        const float expected_scale = read_value(&scale_bits, 0, precision);
        const float scale = read_value(b.as->contents(), sbase + row, precision);
        require(std::abs(scale - expected_scale) <= std::max(1e-8f, expected_scale * 1e-5f), "Activation scale mismatch");
        for (uint32_t k = 0; k < K; ++k) {
          const float q = static_cast<int8_t*>(b.aq->contents())[qbase + row * K + k];
          worst_quant = std::max(worst_quant, double(std::abs(q - reference[(batch * M + row) * K + k] * inv)));
        }
        for (uint32_t col = 0; col < N; ++col) {
          int32_t acc = 0;
          double ref = 0;
          for (uint32_t k = 0; k < K; ++k) {
            acc += int32_t(static_cast<int8_t*>(b.aq->contents())[qbase + row * K + k]) *
              static_cast<int8_t*>(b.bq->contents())[(batch * N + col) * K + k];
            ref += double(a[(batch * M + row) * K + k]) * w[(batch * N + col) * K + k];
          }
          const float bias = d.useBias ? read_value(b.bias->contents(), batch * N + col, precision) : 0;
          ref += bias;
          const float expected = float(acc) * scale * read_value(b.bs->contents(), batch * N + col, precision) + bias;
          const float actual = read_value(b.c->contents(), (batches > 1 ? batch * stride_c : 0) + row * ldc + col, precision);
          const double relative = std::abs(actual - expected) / std::max(1.0f, std::abs(expected));
          worst_matmul = std::max(worst_matmul, relative);
          errors[rotated] += (actual - ref) * (actual - ref);
          if (!rotated)
            signal += ref * ref;
        }
      }
      if (dynamic) {
        for (uint32_t row = actual_m; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k)
            require(static_cast<uint8_t*>(b.aq->contents())[qbase + row * K + k] == 0x55, "Dynamic M overwrote inactive row");
        for (uint32_t row = actual_m; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col)
            require(read_value(b.c->contents(), (batches > 1 ? batch * stride_c : 0) + row * ldc + col, precision) == 0,
              "Dynamic M overwrote inactive output row");
      }
    }
    require(worst_quant <= 0.501, "Rotated activation differs from dense H256 reference");
    const double tolerance = precision == Precision::FP32 ? 2e-5 : precision == Precision::FP16 ? 0.002 : 0.016;
    require(worst_matmul < tolerance, "Matmul disagrees with staged INT8 reference");
    if (register_operands && dynamic) {
      // Reuse the same pipeline across aligned / partial M and a smaller
      // grid. Active values must match; inactive rows must remain untouched.
      std::vector<uint8_t> expected(b.c->length());
      std::memcpy(expected.data(), b.c->contents(), expected.size());
      for (const uint32_t rows : {64u, 33u, actual_m}) {
        std::memset(b.c->contents(), 0, b.c->length());
        std::memset(b.aq->contents(), 0x55, b.aq->length());
        run(queue, exec_d, rotated ? p.get() : plain.get(), rotated ? p.get() : plain.get(), b, rows, 2);
        require(std::memcmp(expected.data(), b.c->contents(), size_t(rows) * N * bytes) == 0,
          "Reusing dynamic-M register pipeline changed active output");
        for (size_t i = size_t(rows) * N * bytes; i < b.c->length(); ++i)
          require(static_cast<uint8_t*>(b.c->contents())[i] == 0,
            "Reusing dynamic-M register pipeline overwrote inactive output");
        for (size_t i = size_t(rows) * K; i < b.aq->length(); ++i)
          require(static_cast<uint8_t*>(b.aq->contents())[i] == 0x55,
            "Reusing dynamic-M register pipeline overwrote inactive activation");
      }
    }
  }
  require(errors[1] < errors[0], "Rotation didn't improve synthetic outlier error");
  if (strided) {
    // The fused path also supports source rows / batches not aligned to a
    // vector load. Repack the same values and require identical staged output.
    const std::vector<int8_t> expected(static_cast<int8_t*>(b.aq->contents()),
      static_cast<int8_t*>(b.aq->contents()) + b.aq->length());
    auto odd = d;
    odd.leadingDimensions = simd::uint2 {K + 3, ldc};
    odd.batchStrides = simd::uint4 {stride_a - 1, N * K, stride_c, N};
    for (uint32_t batch = 0; batch < batches; ++batch)
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          store_value(b.a->contents(), batch * (stride_a - 1) + row * (K + 3) + k,
            a[(batch * M + row) * K + k], precision);
    auto odd_pipeline = pipeline(device, cache, odd);
    run(queue, odd, odd_pipeline.get(), plain.get(), b, actual_m, 0);
    require(std::memcmp(b.aq->contents(), expected.data(), expected.size()) == 0,
      "Unaligned source strides changed activation quantization");
  }
  std::cout << "validation precision=" << precision.name() << " K=" << K << " dynamic=" << dynamic
    << " register_operands=" << register_operands << " strided_batched_bias=" << strided << " plain_nrmse=" << std::sqrt(errors[0] / signal)
    << " h256_nrmse=" << std::sqrt(errors[1] / signal) << " PASS\n";
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double percentile(std::vector<double> values, double fraction)
{
  std::sort(values.begin(), values.end());
  return values[size_t(fraction * (values.size() - 1))];
}

void benchmark(MTL::Device* device, MTL::CommandQueue* queue, Cache& cache,
    uint32_t M, uint32_t N, uint32_t K, int iterations, int repeats)
{
  NAInt8MatMulDescriptor d;
  d.matrixDimensions = simd::uint3 {M, N, K};
  auto plain = pipeline(device, cache, d);
  d.activationHadamard256 = true;
  auto rotated = pipeline(device, cache, d);
  Buffers b {buffer(device, size_t(M) * K * 2), buffer(device, size_t(M) * K),
    buffer(device, M * 2), buffer(device, size_t(N) * K), buffer(device, N * 2),
    buffer(device, size_t(M) * N * 2), buffer(device, N * 2)};
  std::mt19937 rng(17);
  for (size_t i = 0; i < size_t(M) * K; ++i)
    static_cast<_Float16*>(b.a->contents())[i] = (int(rng() % 2001) - 1000) / 1000.0f;
  for (size_t i = 0; i < size_t(N) * K; ++i)
    static_cast<int8_t*>(b.bq->contents())[i] = int(rng() % 255) - 127;
  for (uint32_t n = 0; n < N; ++n)
    static_cast<_Float16*>(b.bs->contents())[n] = 1.0f / 127;
  // Fixed packed weight bytes isolate activation staging cost.
  // Numerical equivalence is checked separately with matching offline weights.
  std::vector<double> samples[2][3], paired[3];
  for (int i = -5; i < iterations; ++i)
    for (int mode : {2, 0, 1}) {
      if (mode == 1) {
        const double ms = run(queue, d, plain.get(), plain.get(), b, M, 1, repeats);
        if (i >= 0)
          samples[0][1].push_back(ms);
        continue;
      }
      double ms[2];
      for (int order = 0; order < 2; ++order) {
        const int arm = (i & 1) ^ order;
        ms[arm] = run(queue, d, arm ? rotated.get() : plain.get(), plain.get(), b, M, mode, repeats);
        if (i >= 0)
          samples[arm][mode].push_back(ms[arm]);
      }
      if (i >= 0)
        paired[mode].push_back((ms[1] / ms[0] - 1) * 100);
    }
  std::cout << std::fixed << std::setprecision(6) << "shape M=" << M << " N=" << N << " K=" << K
    << " iterations=" << iterations << " repeats=" << repeats
    << " plain_quant_ms=" << median(samples[0][0]) << " h256_quant_ms=" << median(samples[1][0])
    << " gemm_ms=" << median(samples[0][1])
    << " plain_total_ms=" << median(samples[0][2]) << " h256_total_ms=" << median(samples[1][2])
    << " overhead_pct=" << (median(samples[1][2]) / median(samples[0][2]) - 1) * 100
    << " paired_overhead_pct=" << median(paired[2])
    << " paired_p10_pct=" << percentile(paired[2], 0.1)
    << " paired_p90_pct=" << percentile(paired[2], 0.9)
    << " staging_delta_pct=" << (median(samples[1][0]) - median(samples[0][0])) / median(samples[0][2]) * 100 << '\n';
}
}

int main(int argc, char** argv)
{
  std::cout << std::unitbuf;
  bool validate_only = false, skip_validation = false, validate_register = false;
  uint32_t M = 0, N = 0, K = 0;
  int iterations = 25, repeats = 3;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--validate-register") { validate_register = true; validate_only = true; }
    else if (arg == "--validate-only") validate_only = true;
    else if (arg == "--skip-validation") skip_validation = true;
    else if (arg == "--shape" && i + 3 < argc) {
      M = std::stoul(argv[++i]); N = std::stoul(argv[++i]); K = std::stoul(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
    else if (arg == "--repeats" && i + 1 < argc) repeats = std::stoi(argv[++i]);
    else {
      std::cerr << "Usage: " << argv[0] << " [--validate-only|--validate-register|--skip-validation] [--shape M N K] [--iterations 25] [--repeats 3]\n";
      return 1;
    }
  }
  require(iterations > 0 && repeats > 0 && (!M || (N > 0 && K > 0 && K % 256 == 0)), "Invalid benchmark dimensions or iterations");
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  require(device.get() != nullptr, "No Metal device available");
  auto queue = NS::TransferPtr(device->newCommandQueue());
  std::cout << "device=" << device->name()->utf8String() << '\n';
  Cache cache;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!skip_validation) {
    validate_cache(device.get());
    for (Precision p : {Precision::FP16, Precision::FP32, Precision::BF16})
      for (uint32_t k : {256u, 768u, 2304u}) {
        validate(device.get(), queue.get(), cache, p, k, false, false, validate_register);
        validate(device.get(), queue.get(), cache, p, k, true, !validate_register, validate_register);
      }
    if (!validate_register)
      for (uint32_t k : {5376u, 16384u, 65536u})
        validate(device.get(), queue.get(), cache, Precision::FP16, k, true, true);
  }
  if (!validate_only) {
    if (M)
      benchmark(device.get(), queue.get(), cache, M, N, K, iterations, repeats);
    else
      for (uint32_t m : {4096u, 16384u})
        for (auto shape : {simd::uint2{6144,6144}, {1536,6144}, {16384,6144}, {6144,16384},
                            {7168,5376}, {5376,7168}, {14336,5376}, {5376,14336}})
          benchmark(device.get(), queue.get(), cache, m, shape[0], shape[1], iterations, repeats);
  }
  ccv_nnc_deinit_mfa_context(context);
  return 0;
}
