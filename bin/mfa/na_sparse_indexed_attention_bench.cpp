#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
}

namespace {

struct Config {
  int T = 512;
  int dense_rows = 128;
  int sparse_rows = 8192;
  int H = 64;
  int D = 512;
  int K = 512;
  int is_causal = 1;
  int attention_sinks = 1;
  int warmup = 2;
  int timed = 5;
};

struct Stats {
  double average_ms = 0;
  double median_ms = 0;
  double min_ms = 0;
  double max_ms = 0;
};

Stats compute_stats(std::vector<double> values)
{
  Stats stats;
  double sum = 0;
  for (const double value : values)
    sum += value;
  stats.average_ms = sum / values.size();
  std::sort(values.begin(), values.end());
  stats.min_ms = values.front();
  stats.max_ms = values.back();
  stats.median_ms = values[values.size() / 2];
  return stats;
}

void print_stats(const char* const name, const Stats& stats)
{
  std::cout << name
            << " avg_ms=" << stats.average_ms
            << " median_ms=" << stats.median_ms
            << " min_ms=" << stats.min_ms
            << " max_ms=" << stats.max_ms
            << "\n";
}

void restore_flag(const uint64_t saved_flags, const uint64_t flag)
{
  if (saved_flags & flag)
    ccv_nnc_enable_flag(flag);
  else
    ccv_nnc_disable_flag(flag);
}

bool parse_positive(const char* const text, int* const value)
{
  const int parsed = std::atoi(text);
  if (parsed <= 0)
    return false;
  *value = parsed;
  return true;
}

void print_usage(const char* const argv0)
{
  std::cerr << "usage: " << argv0 << " [T dense_rows sparse_rows H D K causal attention_sinks warmup timed]\n";
}

void fill_sparse_indexed_attention_inputs(
  std::vector<float>& q,
  std::vector<float>& dense,
  std::vector<float>& sparse,
  std::vector<int>& indices,
  std::vector<float>& sinks,
  const Config& config)
{
  for (size_t i = 0; i < q.size(); ++i)
    q[i] = (float)(((i * 17 + 11) % 97) - 48) / 256.0f;
  for (size_t i = 0; i < dense.size(); ++i)
    dense[i] = (float)(((i * 19 + 7) % 89) - 44) / 256.0f;
  for (size_t i = 0; i < sparse.size(); ++i)
    sparse[i] = (float)(((i * 23 + 5) % 101) - 50) / 256.0f;
  if (config.K == 0)
  {
    for (int t = 0; t < config.T; ++t)
      indices[t] = -1;
  } else {
    for (int t = 0; t < config.T; ++t)
      for (int k = 0; k < config.K; ++k)
        indices[t * config.K + k] = (t * 17 + k * 13) % config.sparse_rows;
  }
  for (int h = 0; h < config.H; ++h)
    sinks[h] = (float)(((h * 7 + 3) % 17) - 8) / 32.0f;
}

void fill_sdpa_baseline_inputs(
  std::vector<float>& source,
  std::vector<int>& shared_indices,
  std::vector<int>& token_indices,
  std::vector<float>& shared_mask,
  std::vector<float>& token_mask,
  const std::vector<float>& dense,
  const std::vector<float>& sparse,
  const std::vector<int>& indices,
  const Config& config)
{
  const int baseline_rows = config.dense_rows + config.K;
  for (int r = 0; r < config.dense_rows; ++r)
    std::memcpy(source.data() + (size_t)r * config.D, dense.data() + (size_t)r * config.D, sizeof(float) * config.D);
  for (int r = 0; r < config.sparse_rows; ++r)
    std::memcpy(source.data() + (size_t)(config.dense_rows + r) * config.D, sparse.data() + (size_t)r * config.D, sizeof(float) * config.D);
  const float masked = -65504.0f;
  std::vector<uint8_t> shared_sparse_visible(config.K);
  bool shared_terminated = false;
  for (int c = 0; c < baseline_rows; ++c)
  {
    if (c < config.dense_rows) {
      shared_indices[c] = c;
    } else {
      const int k = c - config.dense_rows;
      const int idx = config.K > 0 ? indices[k] : -1;
      const bool valid = !shared_terminated && idx >= 0 && idx < config.sparse_rows;
      if (!valid)
        shared_terminated = true;
      shared_sparse_visible[k] = valid;
      shared_indices[c] = valid ? config.dense_rows + idx : 0;
    }
  }
  for (int t = 0; t < config.T; ++t)
  {
    const int dense_end = config.is_causal ? std::max(0, std::min(config.dense_rows, config.dense_rows - config.T + t + 1)) : config.dense_rows;
    bool token_terminated = false;
    for (int c = 0; c < baseline_rows; ++c)
    {
      bool token_visible = true;
      bool shared_visible = true;
      if (c < config.dense_rows) {
        token_indices[(size_t)t * baseline_rows + c] = c;
        token_visible = c < dense_end;
        shared_visible = token_visible;
      } else {
        const int k = c - config.dense_rows;
        const int token_idx = config.K > 0 ? indices[(size_t)t * config.K + k] : -1;
        const bool token_valid = !token_terminated && token_idx >= 0 && token_idx < config.sparse_rows;
        if (!token_valid)
          token_terminated = true;
        token_indices[(size_t)t * baseline_rows + c] = token_valid ? config.dense_rows + token_idx : 0;
        token_visible = token_valid;
        shared_visible = shared_sparse_visible[k] != 0;
      }
      shared_mask[(size_t)t * baseline_rows + c] = shared_visible ? 0 : masked;
      token_mask[(size_t)t * baseline_rows + c] = token_visible ? 0 : masked;
    }
  }
}

bool benchmark_sparse_indexed_attention(
  const ccv_nnc_cmd_t cmd,
  ccv_nnc_tensor_t* const q,
  ccv_nnc_tensor_t* const dense,
  ccv_nnc_tensor_t* const sparse,
  ccv_nnc_tensor_t* const indices,
  ccv_nnc_tensor_t* const sinks,
  ccv_nnc_tensor_t* const out,
  ccv_nnc_stream_context_t* const stream,
  const Config& config,
  Stats* const stats)
{
  for (int i = 0; i < config.warmup; ++i)
  {
    const int status = config.attention_sinks ?
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices, sinks), TENSOR_LIST(out), stream) :
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices), TENSOR_LIST(out), stream);
    ccv_nnc_stream_context_wait(stream);
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
  }
  std::vector<double> samples;
  samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
  {
    const auto start = std::chrono::steady_clock::now();
    const int status = config.attention_sinks ?
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices, sinks), TENSOR_LIST(out), stream) :
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices), TENSOR_LIST(out), stream);
    ccv_nnc_stream_context_wait(stream);
    const auto end = std::chrono::steady_clock::now();
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  *stats = compute_stats(samples);
  return true;
}

bool benchmark_sdpa_with_index_select(
  const ccv_nnc_cmd_t index_select_cmd,
  const ccv_nnc_cmd_t cmd,
  ccv_nnc_tensor_t* const source,
  ccv_nnc_tensor_t* const gather_indices,
  ccv_nnc_tensor_t* const gathered_kv,
  ccv_nnc_tensor_t* const q,
  ccv_nnc_tensor_t* const k,
  ccv_nnc_tensor_t* const v,
  ccv_nnc_tensor_t* const mask,
  ccv_nnc_tensor_t* const sinks,
  ccv_nnc_tensor_t* const out,
  ccv_nnc_stream_context_t* const stream,
  const Config& config,
  Stats* const stats)
{
  for (int i = 0; i < config.warmup; ++i)
  {
    int status = ccv_nnc_cmd_exec(index_select_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(source, gather_indices), TENSOR_LIST(gathered_kv), stream);
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
    status = config.attention_sinks ?
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, mask, 0, 0, 0, 0, sinks), TENSOR_LIST(out), stream) :
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, mask), TENSOR_LIST(out), stream);
    ccv_nnc_stream_context_wait(stream);
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
  }
  std::vector<double> samples;
  samples.reserve(config.timed);
  for (int i = 0; i < config.timed; ++i)
  {
    const auto start = std::chrono::steady_clock::now();
    int status = ccv_nnc_cmd_exec(index_select_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(source, gather_indices), TENSOR_LIST(gathered_kv), stream);
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
    status = config.attention_sinks ?
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, mask, 0, 0, 0, 0, sinks), TENSOR_LIST(out), stream) :
      ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, v, mask), TENSOR_LIST(out), stream);
    ccv_nnc_stream_context_wait(stream);
    const auto end = std::chrono::steady_clock::now();
    if (status != CCV_NNC_EXEC_SUCCESS)
      return false;
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  *stats = compute_stats(samples);
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  ccv_nnc_init();
  Config config;
  if (argc > 1 && argc != 11)
  {
    print_usage(argv[0]);
    return 1;
  }
  if (argc == 11)
  {
    if (!parse_positive(argv[1], &config.T) ||
        !parse_positive(argv[2], &config.dense_rows) ||
        !parse_positive(argv[3], &config.sparse_rows) ||
        !parse_positive(argv[4], &config.H) ||
        !parse_positive(argv[5], &config.D) ||
        !parse_positive(argv[9], &config.warmup) ||
        !parse_positive(argv[10], &config.timed))
    {
      print_usage(argv[0]);
      return 1;
    }
    config.K = std::atoi(argv[6]);
    if (config.K < 0)
    {
      print_usage(argv[0]);
      return 1;
    }
    config.is_causal = std::atoi(argv[7]) != 0;
    config.attention_sinks = std::atoi(argv[8]) != 0;
  }
  if (config.H != 64 || (config.D != 512 && config.D != 128))
  {
    std::cerr << "NASparseIndexedAttention benchmark requires H=64 and D=512 or D=128\n";
    return 1;
  }
  if (config.D == 128 && config.K == 0)
  {
    std::cerr << "NASparseIndexedAttention D=128 benchmark requires K > 0\n";
    return 1;
  }
  if (ccv_nnc_device_count(CCV_STREAM_CONTEXT_GPU) <= 0)
  {
    std::cerr << "no GPU stream context available\n";
    return 1;
  }
  if (!ccv_nnc_cmd_ok(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) ||
      !ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) ||
      !ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
  {
    std::cerr << "required MPS commands are not available\n";
    return 1;
  }

  const int baseline_rows = config.dense_rows + config.K;
  const int index_columns = config.K;
  const float scale = 1.0f / std::sqrt((float)config.D);
  const size_t q_count = (size_t)config.T * config.H * config.D;
  const size_t dense_count = (size_t)config.dense_rows * config.D;
  const size_t sparse_count = (size_t)config.sparse_rows * config.D;
  const size_t index_count = (size_t)config.T * std::max(index_columns, 1);
  const int baseline_Hk = 1;
  const size_t baseline_source_count = (size_t)(config.dense_rows + config.sparse_rows) * config.D;
  const size_t shared_index_count = baseline_rows;
  const size_t token_index_count = (size_t)config.T * baseline_rows;
  const size_t baseline_mask_count = (size_t)config.T * baseline_rows;

  std::vector<float> q_f32(q_count);
  std::vector<float> dense_f32(dense_count);
  std::vector<float> sparse_f32(sparse_count);
  std::vector<int> indices_i32(index_count);
  std::vector<float> sinks_f32(config.H);
  std::vector<float> baseline_source_f32(baseline_source_count);
  std::vector<int> shared_gather_indices_i32(shared_index_count);
  std::vector<int> token_gather_indices_i32(token_index_count);
  std::vector<float> shared_mask_f32(baseline_mask_count);
  std::vector<float> token_mask_f32(baseline_mask_count);
  fill_sparse_indexed_attention_inputs(q_f32, dense_f32, sparse_f32, indices_i32, sinks_f32, config);
  fill_sdpa_baseline_inputs(baseline_source_f32, shared_gather_indices_i32, token_gather_indices_i32, shared_mask_f32, token_mask_f32, dense_f32, sparse_f32, indices_i32, config);

  ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const hdense = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.dense_rows, 1, config.D), 0);
  ccv_nnc_tensor_t* const hsparse = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, config.sparse_rows, 1, config.D), 0);
  const ccv_nnc_tensor_param_t hindices_params = (index_columns == 0) ? CPU_TENSOR_NHWC(32S, config.T) : CPU_TENSOR_NHWC(32S, config.T, index_columns);
  ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, hindices_params, 0);
  ccv_nnc_tensor_t* const hsinks = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, config.H, 1), 0);
  ccv_nnc_tensor_t* const hbaseline_source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, config.dense_rows + config.sparse_rows, config.D), 0);
  ccv_nnc_tensor_t* const hshared_gather_indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, baseline_rows), 0);
  ccv_nnc_tensor_t* const htoken_gather_indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)token_index_count), 0);
  ccv_nnc_tensor_t* const hshared_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, config.T, baseline_rows), 0);
  ccv_nnc_tensor_t* const htoken_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, config.T, 1, 1, baseline_rows), 0);
  ccv_float_to_half_precision(q_f32.data(), (uint16_t*)hq->data.f16, (int)q_count);
  ccv_float_to_half_precision(dense_f32.data(), (uint16_t*)hdense->data.f16, (int)dense_count);
  ccv_float_to_half_precision(sparse_f32.data(), (uint16_t*)hsparse->data.f16, (int)sparse_count);
  std::memcpy(hindices->data.i32, indices_i32.data(), sizeof(int) * index_count);
  ccv_float_to_half_precision(sinks_f32.data(), (uint16_t*)hsinks->data.f16, config.H);
  ccv_float_to_half_precision(baseline_source_f32.data(), (uint16_t*)hbaseline_source->data.f16, (int)baseline_source_count);
  std::memcpy(hshared_gather_indices->data.i32, shared_gather_indices_i32.data(), sizeof(int) * shared_index_count);
  std::memcpy(htoken_gather_indices->data.i32, token_gather_indices_i32.data(), sizeof(int) * token_index_count);
  ccv_float_to_half_precision(shared_mask_f32.data(), (uint16_t*)hshared_mask->data.f16, (int)baseline_mask_count);
  ccv_float_to_half_precision(token_mask_f32.data(), (uint16_t*)htoken_mask->data.f16, (int)baseline_mask_count);

  ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const dense = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.dense_rows, 1, config.D), 0);
  ccv_nnc_tensor_t* const sparse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.sparse_rows, 1, config.D), 0);
  const ccv_nnc_tensor_param_t indices_params = (index_columns == 0) ? GPU_TENSOR_NHWC(000, 32S, config.T) : GPU_TENSOR_NHWC(000, 32S, config.T, index_columns);
  ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, indices_params, 0);
  ccv_nnc_tensor_t* const sinks = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, config.H, 1), 0);
  ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const baseline_source = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.dense_rows + config.sparse_rows, config.D), 0);
  ccv_nnc_tensor_t* const shared_gather_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, baseline_rows), 0);
  ccv_nnc_tensor_t* const token_gather_indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)token_index_count), 0);
  ccv_nnc_tensor_t* const shared_gathered_kv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, baseline_rows, config.D), 0);
  ccv_nnc_tensor_t* const token_gathered_kv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, (int)token_index_count, config.D), 0);
  ccv_nnc_tensor_t* const shared_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, config.T, baseline_rows), 0);
  ccv_nnc_tensor_t* const token_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.T, 1, 1, baseline_rows), 0);
  ccv_nnc_tensor_t* const sdpa_o_shared = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), 0);
  ccv_nnc_tensor_t* const sdpa_o_token = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, config.T, 1, config.H, config.D), 0);
  ccv_nnc_tensor_view_t* const shared_q = ccv_nnc_tensor_view_new(q, GPU_TENSOR_NHWC(000, 16F, 1, config.T, config.H, config.D), DIM_ALLOC(), DIM_ALLOC(config.T * config.H * config.D, config.H * config.D, config.D, 1));
  ccv_nnc_tensor_view_t* const token_q = ccv_nnc_tensor_view_new(q, GPU_TENSOR_NHWC(000, 16F, config.T, 1, config.H, config.D), DIM_ALLOC(), DIM_ALLOC(config.H * config.D, config.H * config.D, config.D, 1));
  ccv_nnc_tensor_view_t* const shared_kv = ccv_nnc_tensor_view_new(shared_gathered_kv, GPU_TENSOR_NHWC(000, 16F, 1, baseline_rows, baseline_Hk, config.D), DIM_ALLOC(), DIM_ALLOC(baseline_rows * baseline_Hk * config.D, baseline_Hk * config.D, config.D, 1));
  ccv_nnc_tensor_view_t* const token_kv = ccv_nnc_tensor_view_new(token_gathered_kv, GPU_TENSOR_NHWC(000, 16F, config.T, baseline_rows, baseline_Hk, config.D), DIM_ALLOC(), DIM_ALLOC(baseline_rows * baseline_Hk * config.D, baseline_Hk * config.D, config.D, 1));
  ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
  ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hdense, hsparse, hindices, hsinks, hbaseline_source, hshared_gather_indices, htoken_gather_indices, hshared_mask, htoken_mask), TENSOR_LIST(q, dense, sparse, indices, sinks, baseline_source, shared_gather_indices, token_gather_indices, shared_mask, token_mask), stream);
  ccv_nnc_stream_context_wait(stream);

  ccv_nnc_cmd_t sia_cmd = CMD_SPARSE_INDEXED_ATTENTION_FORWARD(scale, config.is_causal, config.attention_sinks);
  ccv_nnc_cmd_t sdpa_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
  sdpa_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
  sdpa_cmd.info.scaled_dot_product_attention.attention_sinks = config.attention_sinks;
  const ccv_nnc_cmd_t index_select_cmd = CMD_INDEX_SELECT_FORWARD();

  std::cout << "shape"
            << " T=" << config.T
            << " dense_rows=" << config.dense_rows
            << " sparse_rows=" << config.sparse_rows
            << " H=" << config.H
            << " Hk=" << baseline_Hk
            << " D=" << config.D
            << " K=" << config.K
            << " index_columns=" << index_columns
            << " causal=" << config.is_causal
            << " attention_sinks=" << config.attention_sinks
            << " baseline_C=" << baseline_rows
            << " warmup=" << config.warmup
            << " timed=" << config.timed
            << " dtype=fp16\n";

  const uint64_t saved_flags = ccv_nnc_flags();
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ATTENTION);
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);

  struct Variant {
    int algorithm;
    const char* name;
  };
  const Variant variants[] = {
    {0, "sparse_indexed_attention_mfa_tg_h16"},
    {1, "sparse_indexed_attention_mfa_tg_h24"},
    {3, "sparse_indexed_attention_mfa_tg_h64"},
    {5, "sparse_indexed_attention_mfa_generic"},
    {6, "sparse_indexed_attention_mfa_r1"},
  };
  const Variant variants_d128[] = {
    {4, "sparse_indexed_attention_mfa_tg_h64_d128"},
    {6, "sparse_indexed_attention_mfa_r1"},
  };
  const Variant* const active_variants = (config.D == 128) ? variants_d128 : variants;
  const int variant_count = (config.D == 128) ? (int)(sizeof(variants_d128) / sizeof(variants_d128[0])) : (int)(sizeof(variants) / sizeof(variants[0]));
  std::vector<Stats> sia_stats(variant_count);
  Stats sdpa_shared_stats;
  Stats sdpa_token_stats;
  for (int i = 0; i < variant_count; ++i)
  {
    ccv_nnc_cmd_t variant_cmd = sia_cmd;
    variant_cmd.algorithm = active_variants[i].algorithm;
    if (!benchmark_sparse_indexed_attention(variant_cmd, q, dense, sparse, indices, sinks, out, stream, config, &sia_stats[i]))
    {
      std::cerr << active_variants[i].name << " benchmark failed\n";
      return 1;
    }
  }
  if (!benchmark_sdpa_with_index_select(index_select_cmd, sdpa_cmd, baseline_source, shared_gather_indices, shared_gathered_kv, (ccv_nnc_tensor_t*)shared_q, (ccv_nnc_tensor_t*)shared_kv, (ccv_nnc_tensor_t*)shared_kv, shared_mask, sinks, sdpa_o_shared, stream, config, &sdpa_shared_stats))
  {
    std::cerr << "shared-index NAAttention baseline failed\n";
    return 1;
  }
  if (!benchmark_sdpa_with_index_select(index_select_cmd, sdpa_cmd, baseline_source, token_gather_indices, token_gathered_kv, (ccv_nnc_tensor_t*)token_q, (ccv_nnc_tensor_t*)token_kv, (ccv_nnc_tensor_t*)token_kv, token_mask, sinks, sdpa_o_token, stream, config, &sdpa_token_stats))
  {
    std::cerr << "per-token-index NAAttention baseline failed\n";
    return 1;
  }
  for (int i = 0; i < variant_count; ++i)
    print_stats(active_variants[i].name, sia_stats[i]);
  print_stats("sdpa_na_attention_shared_index_select_baseline", sdpa_shared_stats);
  print_stats("sdpa_na_attention_token_index_select_baseline", sdpa_token_stats);
  for (int i = 0; i < variant_count; ++i)
    std::cout << active_variants[i].name
              << " speedup_over_shared_index_select_na_attention_median=" << sdpa_shared_stats.median_ms / sia_stats[i].median_ms
              << " sia_over_shared_index_select_na_attention_median=" << sia_stats[i].median_ms / sdpa_shared_stats.median_ms
              << " speedup_over_token_index_select_na_attention_median=" << sdpa_token_stats.median_ms / sia_stats[i].median_ms
              << " sia_over_token_index_select_na_attention_median=" << sia_stats[i].median_ms / sdpa_token_stats.median_ms
              << "\n";

  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA);
  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA_ATTENTION);
  restore_flag(saved_flags, CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);

  ccv_nnc_stream_context_free(stream);
  ccv_nnc_tensor_view_free(token_kv);
  ccv_nnc_tensor_view_free(shared_kv);
  ccv_nnc_tensor_view_free(token_q);
  ccv_nnc_tensor_view_free(shared_q);
  ccv_nnc_tensor_free(sdpa_o_token);
  ccv_nnc_tensor_free(sdpa_o_shared);
  ccv_nnc_tensor_free(token_mask);
  ccv_nnc_tensor_free(shared_mask);
  ccv_nnc_tensor_free(token_gathered_kv);
  ccv_nnc_tensor_free(shared_gathered_kv);
  ccv_nnc_tensor_free(token_gather_indices);
  ccv_nnc_tensor_free(shared_gather_indices);
  ccv_nnc_tensor_free(baseline_source);
  ccv_nnc_tensor_free(out);
  ccv_nnc_tensor_free(sinks);
  ccv_nnc_tensor_free(indices);
  ccv_nnc_tensor_free(sparse);
  ccv_nnc_tensor_free(dense);
  ccv_nnc_tensor_free(q);
  ccv_nnc_tensor_free(htoken_mask);
  ccv_nnc_tensor_free(hshared_mask);
  ccv_nnc_tensor_free(htoken_gather_indices);
  ccv_nnc_tensor_free(hshared_gather_indices);
  ccv_nnc_tensor_free(hbaseline_source);
  ccv_nnc_tensor_free(hsinks);
  ccv_nnc_tensor_free(hindices);
  ccv_nnc_tensor_free(hsparse);
  ccv_nnc_tensor_free(hdense);
  ccv_nnc_tensor_free(hq);
  return 0;
}
