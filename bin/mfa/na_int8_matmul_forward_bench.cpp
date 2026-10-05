// GPU timings through the shipping INT8 GEMM dispatcher, including activation quantization.
// Usage: na_int8_matmul_forward_bench M N K warmup samples dynamic_m
// FP16 A[M,K] and output; rowwise INT8 W[N,K] is prepared outside GPU timing.
#include <CommonCrypto/CommonDigest.h>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

int main(int argc, char** argv)
{
  if (argc != 7) {
    std::fprintf(stderr, "Usage: %s M N K warmup samples dynamic_m\n", argv[0]);
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(argv[i], &end, 10);
    if (errno || !*argv[i] || *end || value < 0 || value > INT_MAX) return 2;
  }
  const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
  const int warmup = atoi(argv[4]), timed = atoi(argv[5]), dynamicM = atoi(argv[6]);
  if (std::min({m,n,k,timed}) <= 0 || dynamicM > 1) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  const long double limit = device->maxBufferLength() / sizeof(float);
  if (static_cast<long double>(m)*k > limit ||
      static_cast<long double>(n)*k > limit ||
      static_cast<long double>(m)*n > limit ||
      m > INT32_MAX - 128 || n > INT32_MAX - 128)
    return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  printf("gpu_core_count=%u\n", context->device_properties.coreCount);
  if (!ccv_nnc_mfa_has_neural_accelerators(context)) return 2;
  if (dynamicM) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const size_t ac = size_t(m)*k;
  const size_t bc = size_t(n)*k;
  const size_t oc = size_t(m)*n;
  const size_t scaleOffset = (bc + 127) / 128 * 128;
  std::array<NS::SharedPtr<MTL::Buffer>, 4> buffers;
  const size_t sizes[4] = {ac*2, bc*2, oc*2, scaleOffset + size_t(n)*2};
  for (int i=0;i<4;++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(sizes[i], MTL::ResourceStorageModeShared));
    if (!buffers[i]) return 2;
  }
  for (int phase=1;phase<=2;++phase) {
    auto p = static_cast<_Float16*>(buffers[phase-1]->contents());
    const size_t count = phase == 1 ? ac : bc;
    for (size_t i=0;i<count;++i)
      p[i] = float(int((i*17+phase*29)%127)-63)/64;
  }
  {
    const auto w = static_cast<const _Float16*>(buffers[1]->contents());
    auto q = static_cast<int8_t*>(buffers[3]->contents());
    auto scales = reinterpret_cast<_Float16*>(q + scaleOffset);
    for (int row=0;row<n;++row) {
      float maximum=0;
      for (int d=0;d<k;++d) maximum=std::max(maximum,std::abs(float(w[size_t(row)*k+d])));
      const float scale=maximum>0 ? maximum/127 : 1.0f/127;
      scales[row]=scale;
      for (int d=0;d<k;++d) q[size_t(row)*k+d]=int8_t(std::round(float(w[size_t(row)*k+d])/scale));
    }
  }
  ccv_nnc_mfa_scaled_gemm_params_t ip = {};
  ip.data_type=MTL::DataTypeHalf; ip.M=m; ip.N=n; ip.K=k;
  ip.use_neural_accelerators=1; ip.batch_dimension=1; ip.loadM=dynamicM;
  ccv_nnc_mfa_scaled_gemv_params_t ivp = {};
  ivp.data_type=MTL::DataTypeHalf; ivp.mrows=m; ivp.nrows=n; ivp.ncols=k;
  // Mirror the dense frontend's vector exception (gemm_mps.m).
  const bool gemv = m<=3;
  printf("device=%s route=%s quantized=1 dynamic_m=%d\n", device->name()->utf8String(),
    gemv ? "GEMV" : "MFA-GEMM", dynamicM);
  std::vector<double> samples;
  double warmSeconds=0;
  const double minWarm=std::getenv("CCV_NA_WARMUP_SECONDS") ? std::atof(std::getenv("CCV_NA_WARMUP_SECONDS")) : 0.1;
  if (!std::isfinite(minWarm) || minWarm < 0) return 2;
  for (int i=0;samples.size()<size_t(timed);++i) {
    auto iterationPool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto cb=NS::RetainPtr(queue->commandBuffer());
    auto batch=ccv_nnc_start_command_batch_from_command_buffer(cb.get(),0);
    MTL::Buffer* tensors[] = {buffers[0].get(),buffers[3].get(),buffers[2].get(),nullptr};
    size_t offsets[4] = {};
    if (gemv) {
      std::swap(tensors[0],tensors[1]);
      ccv_nnc_mfa_encode_scaled_gemv(context,ivp,batch,tensors,offsets);
    } else ccv_nnc_mfa_encode_scaled_gemm(context,ip,batch,tensors,offsets);
    ccv_nnc_finish_command_batch(batch);
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status()!=MTL::CommandBufferStatusCompleted) return 3;
    const double elapsed=cb->GPUEndTime()-cb->GPUStartTime();
    warmSeconds+=elapsed;
    if (i>=warmup && warmSeconds>=minWarm) samples.push_back(elapsed*1000);
  }
  const auto a=static_cast<const _Float16*>(buffers[0]->contents());
  const auto w=static_cast<const _Float16*>(buffers[1]->contents());
  const auto out=static_cast<const _Float16*>(buffers[2]->contents());
  if (const char* path=std::getenv("CCV_NA_FORWARD_DUMP")) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(out),oc*2);
    if (!file) return 4;
  }

  for (size_t i=0;i<oc;++i) if (!std::isfinite(float(out[i]))) return 4;
  {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    const auto bytes = reinterpret_cast<const unsigned char*>(out);
    for (size_t offset = 0; offset < oc*sizeof(_Float16); offset += 1u << 20)
      CC_SHA256_Update(&hash, bytes + offset, CC_LONG(std::min<size_t>(1u << 20, oc*sizeof(_Float16) - offset)));
    CC_SHA256_Final(digest, &hash);
    printf("output_sha256=");
    for (auto byte : digest) printf("%02x",byte);
    printf("\n");
  }

  double error=0,norm=0,maxError=0;
  for(int sample=0;sample<64;++sample) {
    const int r=size_t(sample)*7919%m, c=size_t(sample)*104729%n;
    double expected=0;
    for(int d=0;d<k;++d) expected+=float(a[size_t(r)*k+d])*float(w[size_t(c)*k+d]);
    const double diff=float(out[size_t(r)*n+c])-expected;
    error+=diff*diff; norm+=expected*expected; maxError=std::max(maxError,std::abs(diff));
  }
  const double l2=std::sqrt(error/std::max(norm,1e-30));
  printf("cpu_sample_max_abs=%.9g normalized_l2=%.9g all_finite=1\n",maxError,l2);
  const bool accurate = l2 <= 0.05;
  printf("gpu_samples_ms=");
  for (size_t i=0;i<samples.size();++i) printf("%s%.9g",i ? "," : "",samples[i]);
  printf("\n");
  std::sort(samples.begin(),samples.end());
  printf("frontend gpu_median_ms=%.9g min_ms=%.9g max_ms=%.9g samples=%zu\n",
    (samples[(samples.size()-1)/2]+samples[samples.size()/2])*0.5,samples.front(),samples.back(),samples.size());
  ccv_nnc_deinit_mfa_context(context);
  return accurate ? 0 : 4;
}
