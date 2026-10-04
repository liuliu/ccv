// GPU timestamps for production MLX primitives. Link with a Metal-JIT MLX build.
// Usage: mlx_na_bench matmul|attention|backward M N K [B Hq Hkv warmup samples causal]
// Attention uses B/R/H/D storage, matching CCV. Backward excludes saved forward.
#include <cerrno>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <functional>
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"

namespace mx = mlx::core;
mx::array input(mx::Shape shape, int phase, bool attention = false) {
  size_t count = 1;
  for (int d : shape) count *= d;
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) values[i] = attention ? float(int((i * 17 + phase * 13) % 29) - 14) * (9 - phase) / 256 : float(int((i * 17 + phase * 29) % 127) - 63) / 64;
  return mx::array(values.data(), shape, mx::float16);
}
int main(int argc, char** argv) {
  if (argc != 11) {
    std::cerr << "Usage: " << argv[0] << " matmul|attention|backward M N K B Hq Hkv warmup samples causal\n";
    return 1;
  }
  for (int i = 2; i < argc; ++i) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(argv[i], &end, 10);
    if (errno || !*argv[i] || *end || value < 0 || value > INT_MAX) return 1;
  }
  if (std::atoi(argv[10]) > 1) return 1;
  const std::string mode = argv[1];
  const bool attention = mode != "matmul", backward = mode == "backward";
  const bool output_cast = !attention && std::getenv("CCV_NA_OUTPUT_CAST") && std::atoi(std::getenv("CCV_NA_OUTPUT_CAST"));
  if (mode != "matmul" && mode != "attention" && !backward) return 1;
  const int m = atoi(argv[2]), n = atoi(argv[3]), k = atoi(argv[4]);
  const int batch = argc > 5 ? atoi(argv[5]) : 1;
  const int hq = argc > 6 ? atoi(argv[6]) : 8, hk = argc > 7 ? atoi(argv[7]) : hq;
  const int warmup = argc > 8 ? atoi(argv[8]) : 200, timed = argc > 9 ? atoi(argv[9]) : 50;
  const bool causal = argc > 10 && atoi(argv[10]);
  if (std::min({m,n,k,batch,hq,hk,timed}) <= 0 || warmup < 0 || hq % hk) return 1;
  if (backward && causal) return 1;
  const auto stream = mx::default_stream(mx::Device::gpu);
  auto a = input(attention ? mx::Shape{batch,m,hq,k} : mx::Shape{batch,m,k}, 1, attention);
  auto b = input(attention ? mx::Shape{batch,n,hk,k} : mx::Shape{batch,n,k}, 2, attention);
  auto v = input(attention ? mx::Shape{batch,n,hk,k} : mx::Shape{1}, 3, attention);
  if (attention) {
    a = mx::transpose(a, {0,2,1,3});
    b = mx::transpose(b, {0,2,1,3});
    v = mx::transpose(v, {0,2,1,3});
  } else b = mx::swapaxes(b, -1, -2);
  mx::eval(a,b,v);
  auto sdpa = [&](const std::vector<mx::array>& x) {
    return std::vector<mx::array>{mx::fast::scaled_dot_product_attention(
        x[0],x[1],x[2],1.0f/std::sqrt(float(k)),causal ? "causal" : "")};
  };
  std::vector<mx::array> outputs;
  if (backward) {
    auto cot = mx::transpose(input({batch,m,hq,k},4,true), {0,2,1,3});
    outputs = mx::vjp(sdpa, {a,b,v}, {cot}).second;
  } else outputs.push_back(attention ? sdpa({a,b,v})[0] : mx::matmul(a,b));
  if (output_cast) outputs[0] = mx::astype(outputs[0],mx::float32);
  // Reject an unexpected backward lowering before evaluating its inputs: a
  // fallback graph could otherwise run backward work outside the timed region.
  if (backward && (outputs.size() != 3 || !outputs[0].has_primitive() ||
      std::string(outputs[0].primitive().name()) != "ScaledDotProductAttentionVJP")) {
    std::cerr << "Expected the three-output MLX SDPA VJP primitive\n";
    return 5;
  }
  // Materialize O and LSE (and any views) before timing the VJP primitive.
  auto inputs = outputs[0].inputs();
  if (backward) mx::eval(inputs);
  mx::synchronize(stream);
  // Wider heads may lower to matmul/softmax/view primitives. Visit the entire
  // unevaluated graph so a final view or matmul cannot masquerade as SDPA time.
  std::vector<std::vector<mx::array>> stages;
  std::unordered_set<uintptr_t> visited;
  std::function<void(const mx::array&)> visit = [&](const mx::array& out) {
    if (!out.has_primitive() || out.is_available() || !visited.insert(out.primitive_id()).second) return;
    for (const auto& in : out.inputs()) visit(in);
    stages.push_back(out.outputs());
  };
  for (const auto& out : outputs) visit(out);
  if (stages.empty()) return 5;
  std::cout << "nax=" << mx::metal::is_nax_available() << " arch=" << mx::metal::device(mx::Device::gpu).get_architecture() << " primitive=";
  std::cout << outputs[0].primitive().name();
  std::cout << " outputs=" << outputs.size() << " stages=" << stages.size() << std::endl;
  for (const auto& stage : stages) std::cout << "stage=" << stage[0].primitive().name() << std::endl;
  if (!mx::metal::is_nax_available()) return 3;
  std::vector<double> samples;
  const double minimum_warmup = std::getenv("CCV_NA_WARMUP_SECONDS") ? std::atof(std::getenv("CCV_NA_WARMUP_SECONDS")) : 0;
  double warmup_seconds = 0;
  for (int i=0; samples.size() < size_t(timed); ++i) {
    auto& encoder = mx::metal::get_command_encoder(stream);
    auto cb = NS::RetainPtr(encoder.get_command_buffer());
    for (auto& stage : stages) stage[0].primitive().eval_gpu(stage[0].inputs(), stage);
    // These direct primitives encode into one buffer at the recorded revision.
    // Reject future internal commits instead of reporting only the last buffer.
    if (cb.get() != encoder.get_command_buffer()) {
      std::cerr << "Primitive changed the timed command buffer\n";
      return 5;
    }
    encoder.end_encoding();
    encoder.commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) return 2;
    warmup_seconds += cb->GPUEndTime()-cb->GPUStartTime();
    if (i >= warmup && warmup_seconds >= minimum_warmup) samples.push_back((cb->GPUEndTime()-cb->GPUStartTime())*1000);
    if (i == 0 && attention && !backward) {
      double max_error = 0;
      const auto* q = a.data<mx::float16_t>();
      const auto* key = b.data<mx::float16_t>();
      const auto* val = v.data<mx::float16_t>();
      const auto* result = outputs[0].data<mx::float16_t>();
      const auto qs = a.strides(), ks = b.strides(), vs = v.strides(), os = outputs[0].strides();
      for (int z=0;z<batch;++z) for (int h=0;h<hq;++h)
        for (int r=0;r<m;++r) for (int d=0;d<k;++d)
          if (!std::isfinite(float(result[z*os[0]+h*os[1]+r*os[2]+d*os[3]]))) return 6;
      std::cout << "all_finite=1" << std::endl;
      for (int z : {0,batch-1}) for (int h : {0,hq-1}) for (int r : {0,m-1}) {
        const int kh = h / (hq/hk);
        const int columns = causal ? std::min(n,r+std::max(0,n-m)+1) : n;
        std::vector<float> scores(columns);
        for (int c=0;c<columns;++c) {
          float dot=0;
          for (int d=0;d<k;++d)
            dot += float(q[z*qs[0]+h*qs[1]+r*qs[2]+d*qs[3]]) * float(key[z*ks[0]+kh*ks[1]+c*ks[2]+d*ks[3]]);
          scores[c]=dot/std::sqrt(float(k));
        }
        const float maximum=*std::max_element(scores.begin(),scores.end());
        double sum=0;
        for (auto& score:scores) { score=std::exp(score-maximum);sum+=score; }
        for (int d : {0,k-1}) {
          double expected=0;
          for (int c=0;c<columns;++c) expected+=scores[c]*float(val[z*vs[0]+kh*vs[1]+c*vs[2]+d*vs[3]]);
          expected/=sum;
          const float actual=float(result[z*os[0]+h*os[1]+r*os[2]+d*os[3]]);
          if (!std::isfinite(actual)) return 6;
          max_error=std::max(max_error,std::abs(actual-expected));
        }
      }
      std::cout << "cpu_sample_max_abs=" << max_error << std::endl;
      if (max_error>0.005) return 6;
    }
    if (i == 0 && !attention) {
      const auto& result=outputs[0];
      const auto* av=a.data<mx::float16_t>(); const auto* bv=b.data<mx::float16_t>();
      const auto as=a.strides(),bs=b.strides(),cs=result.strides();
      auto value=[&](size_t j) { return output_cast ? result.data<float>()[j] : float(result.data<mx::float16_t>()[j]); };
      const size_t count=size_t(batch)*m*n;
      for(size_t j=0;j<count;++j) if(!std::isfinite(value(j))) return 6;
      if(output_cast) {
        const auto& half_result=result.inputs()[0];
        const auto* half_values=half_result.data<mx::float16_t>();
        for(size_t j=0;j<count;++j) if(value(j)!=float(half_values[j])) return 6;
        std::cout<<"cast_exact_values="<<count<<std::endl;
      }
      double error2=0,norm2=0,max_error=0;
      for(int z : {0,batch-1}) for(int row : {0,m/2,m-1}) for(int col : {0,n/2,n-1}) {
        double expected=0;
        for(int d=0;d<k;++d) expected+=double(float(av[z*as[0]+row*as[1]+d*as[2]]))*float(bv[z*bs[0]+d*bs[1]+col*bs[2]]);
        const double delta=value(z*cs[0]+row*cs[1]+col*cs[2])-expected;
        error2+=delta*delta;norm2+=expected*expected;max_error=std::max(max_error,std::abs(delta));
      }
      double relative=std::sqrt(error2/std::max(norm2,1e-20));
      std::cout<<"normalized_l2="<<relative<<" cpu_sample_max_abs="<<max_error<<std::endl;
      if(relative>.01) return 6;
    }
    if (i == 0) {
      for (auto& out : outputs) {
        const float first = output_cast ? out.data<float>()[0] : float(out.data<mx::float16_t>()[0]);
        if (!std::isfinite(first)) return 4;
        std::cout << "first=" << first << '\n';
      }
    }
  }
  if (backward) {
    if (const char* dump = std::getenv("CCV_NA_DUMP")) {
      for (int i = 0; i < 3; ++i) {
        const auto& out = outputs[i];
        const auto* ptr = out.data<mx::float16_t>();
        const auto strides = out.strides();
        std::ofstream file(std::string(dump) + "." + std::to_string(i), std::ios::binary);
        // Store B/R/H/D, including any non-contiguous production output strides.
        for (int b=0; b<out.shape(0); ++b)
          for (int r=0; r<out.shape(2); ++r)
            for (int h=0; h<out.shape(1); ++h)
              for (int d=0; d<out.shape(3); ++d) {
                auto value = ptr[b*strides[0]+h*strides[1]+r*strides[2]+d*strides[3]];
                if (!std::isfinite(float(value))) return 4;
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
              }
        if (!file) return 4;
      }
    }
  }
  std::cout << "gpu_samples_ms=";
  for (size_t i=0;i<samples.size();++i) std::cout << (i ? "," : "") << samples[i];
  std::cout << std::endl;
  std::sort(samples.begin(), samples.end());
  std::cout << mode << " gpu_median_ms=" << samples[samples.size()/2] << " min_ms=" << samples[0] << std::endl;
}
