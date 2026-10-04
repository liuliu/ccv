// Application-shaped GPU timings through the shipping MFA dispatchers.
// Usage: na_application_bench attention|backward|matmul M N K B Hq Hkv warmup samples causal int8 upcast flags
// FP16 tensors; matmul uses A[M,K], W[N,K]. Rowwise INT8 weights are prepared offline.
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
  if (argc != 14) {
    std::fprintf(stderr, "Usage: %s attention|backward|matmul M N K B Hq Hkv warmup samples causal int8 upcast flags\n", argv[0]);
    return 2;
  }
  for (int i = 2; i < argc; ++i) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(argv[i], &end, 10);
    if (errno || !*argv[i] || *end || value < 0 || value > INT_MAX) return 2;
  }
  for (int i = 10; i <= 12; ++i) if (std::atoi(argv[i]) > 1) return 2;
  const bool backward = !strcmp(argv[1], "backward");
  const bool attention = backward || !strcmp(argv[1], "attention");
  if (!attention && strcmp(argv[1], "matmul")) return 2;
  const int m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]);
  const int b = atoi(argv[5]), hq = atoi(argv[6]), hk = atoi(argv[7]);
  const int warmup = atoi(argv[8]), timed = atoi(argv[9]);
  const bool causal = atoi(argv[10]), quantized = atoi(argv[11]), upcast = atoi(argv[12]);
  // Bit 0: dynamic M; bit 1: dynamic C, as used by Local Code's growing KV cache.
  const int flags = atoi(argv[13]);
  if (std::min({m,n,k,b,hq,hk,timed}) <= 0 || warmup < 0 || hq % hk ||
      (!attention && b != 1) || (flags & ~3)) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  const long double limit = device->maxBufferLength() / sizeof(float);
  if ((attention && (static_cast<long double>(b)*m*hq*k > limit ||
                     static_cast<long double>(b)*n*hk*k > limit)) ||
      (!attention && (static_cast<long double>(m)*k > limit ||
                      static_cast<long double>(n)*k > limit ||
                      static_cast<long double>(m)*n > limit)) ||
      uint64_t(m)*b > INT32_MAX || m > INT32_MAX - 128 || n > INT32_MAX - 128)
    return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  printf("gpu_core_count=%u\n", context->device_properties.coreCount);
  if (!ccv_nnc_mfa_has_neural_accelerators(context)) return 2;
  if (flags & 1) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
  if (flags & 2) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ATTENTION_SPECIALIZING_C);
  auto queue = NS::TransferPtr(device->newCommandQueue());
  if (backward && causal) return 2;
  const size_t ac = attention ? size_t(b)*m*hq*k : size_t(m)*k;
  const size_t bc = attention ? size_t(b)*n*hk*k : size_t(n)*k;
  const size_t oc = attention ? ac : size_t(m)*n;
  const size_t scaleOffset = (bc + 127) / 128 * 128;
  std::array<NS::SharedPtr<MTL::Buffer>, 5> buffers;
  const size_t sizes[5] = {ac*2, bc*2, attention ? bc*2 : oc*2, attention ? oc*2 : 2,
    scaleOffset + size_t(n)*4};
  for (int i=0;i<5;++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(sizes[i], MTL::ResourceStorageModeShared));
    if (!buffers[i]) return 2;
  }
  for (int phase=1;phase<=(attention ? 3 : 2);++phase) {
    auto p = static_cast<_Float16*>(buffers[phase-1]->contents());
    const size_t count = phase == 1 ? ac : bc;
    for (size_t i=0;i<count;++i)
      p[i] = attention ? float(int((i*17+phase*13)%29)-14)*(9-phase)/256 : float(int((i*17+phase*29)%127)-63)/64;
  }
  if (!attention && quantized) {
    const auto w = static_cast<const _Float16*>(buffers[1]->contents());
    auto q = static_cast<int8_t*>(buffers[4]->contents());
    auto scales = reinterpret_cast<_Float16*>(q + scaleOffset);
    for (int row=0;row<n;++row) {
      float maximum=0;
      for (int d=0;d<k;++d) maximum=std::max(maximum,std::abs(float(w[size_t(row)*k+d])));
      const float scale=maximum/127;
      scales[row]=scale;
      for (int d=0;d<k;++d) q[size_t(row)*k+d]=int8_t(std::round(float(w[size_t(row)*k+d])/scale));
    }
  }
  const bool output_cast = !attention && std::getenv("CCV_NA_OUTPUT_CAST") && std::atoi(std::getenv("CCV_NA_OUTPUT_CAST"));
  const bool fused_cast = std::getenv("CCV_GEMM_FUSED_CAST") && std::atoi(std::getenv("CCV_GEMM_FUSED_CAST"));
  if (fused_cast && (!output_cast || attention)) return 2;
  if (output_cast && oc > UINT32_MAX / sizeof(float)) return 2;
  auto float_output = output_cast ? NS::TransferPtr(device->newBuffer(oc * sizeof(float), MTL::ResourceStorageModeShared)) : NS::SharedPtr<MTL::Buffer>();
  if (output_cast && !float_output) return 2;
  ccv_nnc_mfa_cast_params_t cp = {};
  cp.original_data_type = MTL::DataTypeHalf; cp.data_type = MTL::DataTypeFloat; cp.length = (uint32_t)oc;
  printf("output_cast=%d\n", output_cast);
  ccv_nnc_mfa_attention_params_t ap = {};
  ap.data_type=MTL::DataTypeHalf; ap.R=m; ap.C=n; ap.D=k; ap.Hq=hq; ap.Hk=hk;
  ap.output_rows=m*b; ap.K_trans=1; ap.alpha=1/std::sqrt(float(k)); ap.is_causal=causal;
  ap.batched=b>1; ap.batch_dims_q[0]=b>1 ? b : 0; ap.upcast=upcast;
  // Mirror the dense SDPA frontend's short D256 inference fallback. All probe
  // buffers are distinct and contiguous, so its alias/layout guards hold.
  const bool prefer_fp16 = attention && !backward && quantized && !upcast &&
      k == 256 && m > 64 &&
      ((causal && m <= n && n <= INT32_MAX - 96) ||
       (!causal && !ccv_nnc_mfa_attention_split_output_shape(m, n, k)));
  ap.use_neural_accelerators=1; ap.use_quantized_attention=quantized && !prefer_fp16;
  ap.is_inference=!backward;
  printf("attention_fp16_fallback=%d\n", int(prefer_fp16));
  std::array<NS::SharedPtr<MTL::Buffer>, 5> gradients;
  if (backward) {
    const size_t bytes[5]={ac*2,ac*2,bc*2,bc*2,size_t(b)*hq*m*4};
    for (int j=0;j<5;++j) {
      gradients[j]=NS::TransferPtr(device->newBuffer(bytes[j],MTL::ResourceStorageModeShared));
      if (!gradients[j]) return 2;
    }
    auto cot=static_cast<_Float16*>(gradients[0]->contents());
    for (size_t i=0;i<ac;++i) cot[i]=float(int((i*17+4*13)%29)-14)*5/256;
    auto cb=queue->commandBuffer();
    auto batch=ccv_nnc_start_command_batch_from_command_buffer(cb,0);
    MTL::Buffer* tensors[10]={buffers[0].get(),buffers[1].get(),buffers[2].get(),buffers[3].get(),nullptr,gradients[4].get()};
    size_t offsets[10]={};
    ccv_nnc_mfa_encode_attention(context,ap,batch,tensors,offsets);
    ccv_nnc_finish_command_batch(batch); cb->commit(); cb->waitUntilCompleted();
    if (cb->status()!=MTL::CommandBufferStatusCompleted) return 3;
    ap.type=1;
    printf("backward_route=%s saved_forward_excluded=1\n",k>128 ? "MFA-fallback" : (quantized ? "NA-int8" : "NA"));
  }
  ccv_nnc_mfa_gemm_params_t gp = {};
  gp.data_type=MTL::DataTypeHalf; gp.M=m; gp.N=n; gp.K=k; gp.B_trans=1;
  gp.register_float=upcast; gp.use_neural_accelerators=1; gp.batch_dimension=1; gp.loadM=flags&1;
  ccv_nnc_mfa_scaled_gemm_params_t ip = {};
  ip.data_type=MTL::DataTypeHalf; ip.M=m; ip.N=n; ip.K=k;
  ip.use_neural_accelerators=1; ip.batch_dimension=1; ip.loadM=flags&1;
  ccv_nnc_mfa_gemv_params_t vp = {};
  vp.data_type=MTL::DataTypeHalf; vp.mrows=m; vp.nrows=n; vp.ncols=k;
  ccv_nnc_mfa_scaled_gemv_params_t ivp = {};
  ivp.data_type=MTL::DataTypeHalf; ivp.mrows=m; ivp.nrows=n; ivp.ncols=k;
  // Mirror the dense frontend's vector exception (gemm_mps.m).
  const bool gemv = !attention && m<=3 && !(m==1 && k<5120 && !upcast && !quantized);
  printf("device=%s route=%s upcast=%d quantized=%d dynamic_m=%d dynamic_c=%d\n", device->name()->utf8String(),
    attention ? (m==1 && !quantized ? "AttentionR1" : "MFA-attention") : (gemv ? "GEMV" : "MFA-GEMM"), upcast, quantized, flags&1, (flags>>1)&1);
  std::vector<double> samples;
  double warmSeconds=0;
  const double minWarm=std::getenv("CCV_NA_WARMUP_SECONDS") ? std::atof(std::getenv("CCV_NA_WARMUP_SECONDS")) : 0.1;
  for (int i=0;samples.size()<size_t(timed);++i) {
    auto iterationPool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto cb=NS::RetainPtr(queue->commandBuffer());
    auto batch=ccv_nnc_start_command_batch_from_command_buffer(cb.get(),0);
    MTL::Buffer* tensors[10] = {buffers[0].get(),buffers[1].get(),buffers[2].get(),attention ? buffers[3].get() : nullptr};
    size_t offsets[10] = {};
    if (backward) {
      tensors[4]=gradients[4].get(); tensors[5]=gradients[0].get();
      for (int j=0;j<3;++j) tensors[6+j]=gradients[1+j].get();
    }
    bool cast_encoded = false;
    if (attention) ccv_nnc_mfa_encode_attention(context,ap,batch,tensors,offsets);
    else {
      if (quantized) tensors[1]=buffers[4].get();
      if (gemv) {
        std::swap(tensors[0],tensors[1]);
        if (quantized) ccv_nnc_mfa_encode_scaled_gemv(context,ivp,batch,tensors,offsets);
        else ccv_nnc_mfa_encode_gemv(context,vp,batch,tensors,offsets);
      } else if (quantized) {
        if (fused_cast) {
          tensors[2]=float_output.get();
          cast_encoded=ccv_nnc_mfa_encode_scaled_gemm_cast(context,ip,batch,tensors,offsets);
          tensors[2]=buffers[2].get();
        }
        if (!cast_encoded) ccv_nnc_mfa_encode_scaled_gemm(context,ip,batch,tensors,offsets);
      }
      else ccv_nnc_mfa_encode_gemm(context,gp,batch,tensors,offsets);
    }
    if (i==0) printf("cast_fused=%d\n",int(cast_encoded));
    if (output_cast && !cast_encoded) {
      MTL::Buffer* cast_tensors[3] = {buffers[2].get(), float_output.get(), nullptr};
      size_t cast_offsets[3] = {};
      ccv_nnc_mfa_encode_cast(context,cp,batch,cast_tensors,cast_offsets);
    }
    ccv_nnc_finish_command_batch(batch);
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status()!=MTL::CommandBufferStatusCompleted) return 3;
    const double elapsed=cb->GPUEndTime()-cb->GPUStartTime();
    warmSeconds+=elapsed;
    if (i>=warmup && warmSeconds>=minWarm) samples.push_back(elapsed*1000);
  }
  if (fused_cast) {
    // Only for the existing sampled CPU checker, after timing. Equivalence is
    // checked separately by hashing the complete Float32 output of each mode.
    auto half_values=static_cast<_Float16*>(buffers[2]->contents());
    const auto float_values=static_cast<const float*>(float_output->contents());
    for (size_t i=0;i<oc;++i) half_values[i]=_Float16(float_values[i]);
  }
  const auto a=static_cast<const _Float16*>(buffers[0]->contents());
  const auto w=static_cast<const _Float16*>(buffers[1]->contents());
  const auto v=static_cast<const _Float16*>(buffers[2]->contents());
  const auto out=static_cast<const _Float16*>(buffers[attention ? 3 : 2]->contents());
  if (const char* path=std::getenv("CCV_NA_FORWARD_DUMP")) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(out),oc*2);
    if (!file) return 4;
  }

  for (size_t i=0;i<oc;++i) if (!std::isfinite(float(out[i]))) return 4;
  if (!output_cast) {
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

  if (output_cast) {
    const float* converted = static_cast<const float*>(float_output->contents());
    for (size_t i=0;i<oc;++i) if (!std::isfinite(converted[i]) || converted[i] != float(out[i])) return 4;
    printf("cast_exact_values=%zu\n",oc);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    const auto bytes = reinterpret_cast<const unsigned char*>(converted);
    for (size_t offset = 0; offset < oc*sizeof(float); offset += 1u << 20)
      CC_SHA256_Update(&hash, bytes + offset, CC_LONG(std::min<size_t>(1u << 20, oc*sizeof(float) - offset)));
    CC_SHA256_Final(digest, &hash);
    printf("output_sha256=");
    for (auto byte : digest) printf("%02x",byte);
    printf("\n");

  }

  double error=0,norm=0,maxError=0;
  if (attention) {
    for (int z : {0,b-1}) for (int h : {0,hq-1}) for (int r : {0,m/2,m-1}) {
      const int kh=h/(hq/hk), cols=causal ? std::min(n,r+std::max(0,n-m)+1) : n;
      std::vector<double> scores(cols);
      for (int c=0;c<cols;++c) {
        double dot=0;
        for(int d=0;d<k;++d) dot+=float(a[((size_t(z)*m+r)*hq+h)*k+d])*float(w[((size_t(z)*n+c)*hk+kh)*k+d]);
        scores[c]=dot/std::sqrt(double(k));
      }
      const double maximum=*std::max_element(scores.begin(),scores.end());
      double sum=0;
      for(auto& s:scores) { s=std::exp(s-maximum); sum+=s; }
      for (int d : {0,k/2,k-1}) {
        double expected=0;
        for(int c=0;c<cols;++c) expected+=scores[c]*float(v[((size_t(z)*n+c)*hk+kh)*k+d]);
        expected/=sum;
        const double diff=float(out[((size_t(z)*m+r)*hq+h)*k+d])-expected;
        error+=diff*diff; norm+=expected*expected; maxError=std::max(maxError,std::abs(diff));
      }
    }
  } else {
    for(int sample=0;sample<64;++sample) {
      const int r=size_t(sample)*7919%m, c=size_t(sample)*104729%n;
      double expected=0;
      for(int d=0;d<k;++d) expected+=float(a[size_t(r)*k+d])*float(w[size_t(c)*k+d]);
      const double diff=float(out[size_t(r)*n+c])-expected;
      error+=diff*diff; norm+=expected*expected; maxError=std::max(maxError,std::abs(diff));
    }
  }
  const double l2=std::sqrt(error/std::max(norm,1e-30));
  printf("cpu_sample_max_abs=%.9g normalized_l2=%.9g all_finite=1\n",maxError,l2);
  const bool accurate = l2 <= (quantized && !prefer_fp16 ? 0.05 : 0.01);
  if (backward) for (int j=0;j<3;++j) {
    const auto values=static_cast<const _Float16*>(gradients[1+j]->contents());
    const size_t count=j==0 ? ac : bc;
    for (size_t i=0;i<count;++i) if (!std::isfinite(float(values[i]))) return 4;
    if (const char* path=std::getenv("CCV_NA_DUMP")) {
      std::ofstream file(std::string(path)+"."+std::to_string(j),std::ios::binary);
      file.write(reinterpret_cast<const char*>(values),count*2);
      if (!file) return 4;
    }
  }
  printf("gpu_samples_ms=");
  for (size_t i=0;i<samples.size();++i) printf("%s%.9g",i ? "," : "",samples[i]);
  printf("\n");
  std::sort(samples.begin(),samples.end());
  printf("frontend gpu_median_ms=%.9g min_ms=%.9g max_ms=%.9g samples=%zu\n",
    (samples[(samples.size()-1)/2]+samples[samples.size()/2])*0.5,samples.front(),samples.back(),samples.size());
  ccv_nnc_deinit_mfa_context(context);
  return accurate ? 0 : 4;
}
