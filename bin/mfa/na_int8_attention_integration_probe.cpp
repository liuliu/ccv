#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/mps/ccv_nnc_mps.h"
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/ccv_nnc_mfa_attention.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"

namespace {

void fill_data(float* data, int count, int phase)
{
  for (int i = 0; i < count; ++i)
  {
    const int centered = (i * 17 + phase * 13) % 29 - 14;
    data[i] = centered * 0.03125f;
  }
}

double average_ms(const std::vector<double>& values)
{
  double sum = 0;
  for (const double value : values)
    sum += value;
  return sum / values.size();
}

double median_ms(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

}

int main(int argc, char** argv)
{
  const int B = argc > 1 ? atoi(argv[1]) : 1;
  const int R = argc > 2 ? atoi(argv[2]) : 8192;
  const int C = argc > 3 ? atoi(argv[3]) : 8192;
  const int H = argc > 4 ? atoi(argv[4]) : 32;
  const int D = argc > 5 ? atoi(argv[5]) : 128;
  const int warmup = argc > 6 ? atoi(argv[6]) : 3;
  const int timed = argc > 7 ? atoi(argv[7]) : 10;

  if (ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) <= 0)
  {
    std::cerr << "no GPU stream context available\n";
    return 1;
  }

  const float scale = 1.0f / std::sqrt((float)D);

  ccv_nnc_mfa_attention_params_t params = {};
  params.Q_trans = 0;
  params.K_trans = 1;
  params.V_trans = 0;
  params.O_trans = 0;
  params.batched = 1;
  params.masked = 0;
  params.upcast = 0;
  params.type = 0;
  params.use_neural_accelerators = 1;
  params.use_quantized_attention = 1;
  params.R = (uint32_t)R;
  params.C = (uint32_t)C;
  params.Hq = (uint32_t)H;
  params.Hk = (uint32_t)H;
  params.D = (uint32_t)D;
  params.alpha = scale;
  params.data_type = MTL::DataTypeHalf;
  params.batch_dims_q[0] = B;
  params.batch_dims_mask[0] = B;

  const uint16_t block_d = D >= 192 ? 64 : 32;
  const uint16_t execution_simdgroups = D > 192 ? 16 : 4;

  std::cout << "integration descriptor"
            << " blockR=16"
            << " blockC=64"
            << " blockD=" << block_d
            << " simdgroups=" << execution_simdgroups
            << " threadBarrierOverC=true"
            << " mortonOrder=true"
            << " qQuantThreads=" << NAInt8AttentionKernel::qQuantizeThreads
            << " kvQuantThreads=" << NAInt8AttentionKernel::kvQuantizeThreads
            << "\n";

  ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
  ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
  ccv_nnc_tensor_t* const v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
  ccv_nnc_tensor_t* const q16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
  ccv_nnc_tensor_t* const k16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
  ccv_nnc_tensor_t* const v16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
  const int q_count = B * R * H * D;
  const int kv_count = B * C * H * D;
  fill_data(q->data.f32, q_count, 1);
  fill_data(k->data.f32, kv_count, 2);
  fill_data(v->data.f32, kv_count, 3);
  ccv_float_to_half_precision(q->data.f32, (uint16_t*)q16->data.f16, q_count);
  ccv_float_to_half_precision(k->data.f32, (uint16_t*)k16->data.f16, kv_count);
  ccv_float_to_half_precision(v->data.f32, (uint16_t*)v16->data.f16, kv_count);

  ccv_nnc_tensor_t* const gpu_q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
  ccv_nnc_tensor_t* const gpu_k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
  ccv_nnc_tensor_t* const gpu_v = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
  ccv_nnc_tensor_t* const gpu_o = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);

  ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
  ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0,
      TENSOR_LIST(q16, k16, v16), TENSOR_LIST(gpu_q, gpu_k, gpu_v), stream);
  ccv_nnc_stream_context_wait(stream);

  ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
  gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;

  for (int i = 0; i < warmup; ++i)
  {
    ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_q, gpu_k, gpu_v), TENSOR_LIST(gpu_o), stream);
    ccv_nnc_stream_context_wait(stream);
  }

  std::vector<double> samples;
  samples.reserve(timed);
  for (int i = 0; i < timed; ++i)
  {
    const auto start = std::chrono::steady_clock::now();
    ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0,
        TENSOR_LIST(gpu_q, gpu_k, gpu_v), TENSOR_LIST(gpu_o), stream);
    ccv_nnc_stream_context_wait(stream);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::cout << "integration timing"
            << " avg_ms=" << average_ms(samples)
            << " median_ms=" << median_ms(samples)
            << " warmup=" << warmup
            << " timed=" << timed
            << "\n";

  ccv_nnc_stream_context_free(stream);
  ccv_nnc_tensor_free(gpu_o);
  ccv_nnc_tensor_free(gpu_v);
  ccv_nnc_tensor_free(gpu_k);
  ccv_nnc_tensor_free(gpu_q);
  ccv_nnc_tensor_free(v16);
  ccv_nnc_tensor_free(k16);
  ccv_nnc_tensor_free(q16);
  ccv_nnc_tensor_free(v);
  ccv_nnc_tensor_free(k);
  ccv_nnc_tensor_free(q);
  return 0;
}
