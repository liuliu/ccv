#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
extern "C" {
#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
}
#include "nnc/mfa/ccv_nnc_mfa.hpp"

int main(int argc, char** argv)
{
  if (argc != 3 && (argc < 5 || argc > 9)) {
    std::cerr << "usage: " << argv[0] << " T C [warmup timed [H [use_na [16F|32F|16BF [sort_indices]]]]] (0 0: verify only)\n";
    return 1;
  }
  const int T = std::atoi(argv[1]), C = std::atoi(argv[2]);
  const int warmup = argc >= 5 ? std::atoi(argv[3]) : 10;
  const int timed = argc >= 5 ? std::atoi(argv[4]) : 30;
  const int H = argc >= 6 ? std::atoi(argv[5]) : 32;
  const bool useNA = argc >= 7 ? std::atoi(argv[6]) != 0 : true;
  const char* dtype = argc >= 8 ? argv[7] : "16F";
  const int sortIndices = argc >= 9 ? std::atoi(argv[8]) : 1;
  const int datatype = !std::strcmp(dtype, "32F") ? CCV_32F : (!std::strcmp(dtype, "16BF") ? CCV_16BF : CCV_16F);
  if (std::strcmp(dtype, "32F") && std::strcmp(dtype, "16F") && std::strcmp(dtype, "16BF")) return 1;
  const int elementSize = datatype == CCV_32F ? 4 : 2;
  const int modes = 5;
  const char* names[] = { sortIndices ? "dense_sorted" : "dense", "source_pool", "candidate_reader", "scattered_reader", "legacy" };
  if (T <= 0 || C <= 0 || T > C || H <= 0 || warmup < 0 || timed < 0 || sortIndices < 0 || sortIndices > 1) return 1;
  ccv_nnc_init();
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto queue = NS::TransferPtr(device->newCommandQueue());
  ccv_nnc_mfa_context_t* contexts[modes];
  for (int mode = 0; mode < modes; ++mode) contexts[mode] = ccv_nnc_init_mfa_context(device.get());
  if (useNA && !ccv_nnc_mfa_has_neural_accelerators(contexts[0])) {
    std::cerr << "This probe requires neural accelerators.\n";
    return 1;
  }
  if (datatype == CCV_16BF && !ccv_nnc_mfa_neural_accelerators_support_bfloat(contexts[0])) {
    std::cerr << "BF16 MFA is unavailable on this device / OS.\n";
    return 1;
  }
  const auto options = MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;
  auto q = NS::TransferPtr(device->newBuffer((size_t)T * H * 128 * elementSize, options));
  auto k = NS::TransferPtr(device->newBuffer((size_t)C * 128 * elementSize, options));
  auto w = NS::TransferPtr(device->newBuffer((size_t)T * H * elementSize, options));
  auto selected = NS::TransferPtr(device->newBuffer((size_t)T * 512 * 4, options));
  auto candidates = NS::TransferPtr(device->newBuffer((size_t)T * 2048 * 4, options));
  auto scattered = NS::TransferPtr(device->newBuffer((size_t)T * 2048 * 4, options));
  for (int t = 0; t < T; ++t) {
    const int visible = (C * 2 - T + t + 1) / 2;
    const int blocks = (visible + 7) / 8;
    for (int i = 0; i < 2048; ++i)
      ((int32_t*)scattered->contents())[t * 2048 + i] = i < blocks ? (int32_t)((int64_t)i * (blocks - 1) / std::max(1, std::min(blocks, 2048) - 1)) : -1;
  }
  auto store = [&](const float* source, MTL::Buffer* destination, size_t offset, size_t count) {
    auto ptr = (uint8_t*)destination->contents() + offset * elementSize;
    if (datatype == CCV_32F) std::memcpy(ptr, source, count * sizeof(float));
    else if (datatype == CCV_16BF) ccv_float_to_bfloat(source, (uint16_t*)ptr, count);
    else ccv_float_to_half_precision(source, (uint16_t*)ptr, count);
  };
  std::vector<float> row(128);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      for (int d = 0; d < 128; ++d) row[d] = ((t * 17 + h * 13 + d * 5) % 31 - 15) / 64.0f;
      store(row.data(), q.get(), ((size_t)t * H + h) * 128, 128);
    }
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < 128; ++d) row[d] = ((c * 19 + d * 7) % 37 - 18) / 64.0f;
    store(row.data(), k.get(), (size_t)c * 128, 128);
  }
  std::vector<float> weights((size_t)T * H, 1.0f);
  store(weights.data(), w.get(), 0, weights.size());
  ccv_nnc_mfa_scaled_dot_product_arg_partition_params_t params = {};
  params.data_type = datatype == CCV_32F ? MTL::DataTypeFloat : (datatype == CCV_16BF ? MTL::DataTypeBFloat : MTL::DataTypeHalf);
  params.T = T; params.C = C; params.H = H; params.D = 128; params.kth = 512;
  params.compression_ratio = 2; params.query_offset = C * 2 - T;
  params.scale = 1.0f / 64; params.is_causal = 1; params.use_neural_accelerators = useNA;
  auto run = [&](int mode) {
    auto scope = NS::AutoreleasePool::alloc()->init();
    auto cb = queue->commandBuffer();
    params.candidate_block_size = mode >= 1 && mode <= 3 ? 8 : 0;
    params.candidate_count = mode >= 1 && mode <= 3 ? 2048 : 0;
    params.has_candidates = mode == 2 || mode == 3;
    params.output_candidates = mode == 1;
    params.sort_indices = mode != 4 && sortIndices;
    MTL::Buffer* tensors[] = { q.get(), k.get(), w.get(), selected.get(), mode == 3 ? scattered.get() : candidates.get(), candidates.get(), nullptr };
    size_t offsets[6] = {};
    {
      MTL::CommandBatch batch(cb, false);
      if ((mode == 0 || mode == 4) && C <= 512) {
        ccv_nnc_mfa_scaled_dot_product_arg_partition_enumerate_params_t enumerate = {};
        enumerate.T = T; enumerate.C = C; enumerate.kth = 512;
        enumerate.compression_ratio = 2; enumerate.query_offset = C * 2 - T; enumerate.is_causal = 1;
        MTL::Buffer* outputs[] = { selected.get(), nullptr };
        ccv_nnc_mfa_encode_scaled_dot_product_arg_partition_enumerate(contexts[mode], enumerate, &batch, outputs, offsets);
      } else {
        ccv_nnc_mfa_encode_scaled_dot_product_arg_partition(contexts[mode], params, &batch, tensors, offsets);
      }
    }
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) { std::cerr << "GPU command failed\n"; std::exit(2); }
    const double ms = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000;
    scope->drain();
    return ms;
  };
  run(1); // Seed the pool before timing readers.
  if (C <= 2048 * 8) {
    run(4);
    std::vector<int32_t> denseIDs((int32_t*)selected->contents(), (int32_t*)selected->contents() + (size_t)T * 512);
    for (int t = 0; t < T; ++t)
      std::sort(denseIDs.begin() + t * 512, denseIDs.begin() + (t + 1) * 512);
    for (int mode = 1; mode <= 3; ++mode) {
      run(mode);
      std::vector<int32_t> actualIDs((int32_t*)selected->contents(), (int32_t*)selected->contents() + (size_t)T * 512);
      for (int t = 0; t < T; ++t)
        std::sort(actualIDs.begin() + t * 512, actualIDs.begin() + (t + 1) * 512);
      const bool equal = actualIDs == denseIDs;
      std::cout << "full_pool_verify " << names[mode] << " dense_set_equal=" << equal << std::endl;
      if (!equal) return 2;
    }
  }
  const auto preheatEnd = std::chrono::steady_clock::now() + std::chrono::seconds(timed > 0 ? 1 : 0);
  while (std::chrono::steady_clock::now() < preheatEnd)
    for (int mode = 0; mode < modes; ++mode) run(mode);
  std::vector<double> times[modes];
  for (int i = 0; i < warmup + timed; ++i)
    for (int j = 0; j < modes; ++j) {
      const int mode = (i + j) % modes;
      const double ms = run(mode);
      if (i >= warmup) times[mode].push_back(ms);
    }
  std::cout << "device=" << device->name()->utf8String() << " T=" << T << " C=" << C
            << " H=" << H << " D=128 kth=512 block_size=8 pool=2048 dtype=" << dtype << " use_na=" << useNA << " sort_indices=" << sortIndices << " warmup=" << warmup << " timed=" << timed << '\n';
  for (int mode = 0; mode < modes; ++mode) {
    if (!timed) continue;
    std::sort(times[mode].begin(), times[mode].end());
    std::cout << names[mode] << " median_gpu_ms=" << times[mode][timed / 2]
              << " min_gpu_ms=" << times[mode].front() << " max_gpu_ms=" << times[mode].back()
              << " allocated_scratch_bytes=" << contexts[mode]->scratch->length() << '\n';
  }
  // Exercise the public dispatcher separately. These timings include host
  // encoding, submission and synchronization, unlike the GPU timestamps above.
  // With CCV_METAL_LOGGING_ENABLE=1 in the library, CCV_METAL_LOG_LEVEL=3
  // and warmup=timed=0 record each actual backend route.
  std::vector<ccv_nnc_tensor_t*> owned;
  auto upload = [&](void* data, ccv_nnc_tensor_param_t info) {
    auto host = ccv_nnc_tensor_new(data, info, 0);
    info.type = CCV_TENSOR_GPU_MEMORY;
    auto gpu = ccv_nnc_tensor_new(nullptr, info, 0);
    if (ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(gpu), nullptr) != CCV_NNC_EXEC_SUCCESS) std::exit(2);
    ccv_nnc_tensor_free(host);
    owned.push_back(gpu);
    return gpu;
  };
  ccv_nnc_tensor_param_t qi = CPU_TENSOR_NHWC(32F, T, H, 128);
  ccv_nnc_tensor_param_t ki = CPU_TENSOR_NHWC(32F, C, 128);
  ccv_nnc_tensor_param_t wi = CPU_TENSOR_NHWC(32F, T, H);
  qi.datatype = ki.datatype = wi.datatype = datatype;
  auto cq = upload(q->contents(), qi);
  auto ck = upload(k->contents(), ki);
  auto cw = upload(w->contents(), wi);
  auto cp = upload(candidates->contents(), CPU_TENSOR_NHWC(32S, T, 2048));
  auto cs = upload(scattered->contents(), CPU_TENSOR_NHWC(32S, T, 2048));
  auto co = ccv_nnc_tensor_new(nullptr, GPU_TENSOR_NHWC(000, 32S, T, 512), 0);
  auto ho = ccv_nnc_tensor_new(nullptr, CPU_TENSOR_NHWC(32S, T, 512), 0);
  auto hp = ccv_nnc_tensor_new(nullptr, CPU_TENSOR_NHWC(32S, T, 2048), 0);
  owned.insert(owned.end(), {co, ho, hp});
  const uint64_t oldFlags = ccv_nnc_flags();
  ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
  if (useNA) ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  else ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  auto stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
  auto publicRun = [&](int mode) {
    auto cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(512, params.scale, 1, 2, C * 2 - T);
    cmd.info.scaled_dot_product_arg_partition.candidate_block_size = mode >= 1 && mode <= 3 ? 8 : 0;
    cmd.info.scaled_dot_product_arg_partition.candidate_kth = mode >= 1 && mode <= 3 ? 2048 : 0;
    cmd.info.scaled_dot_product_arg_partition.sort_indices = mode != 4 && sortIndices;
    ccv_nnc_tensor_t* inputs[] = {cq, ck, cw, mode == 3 ? cs : cp};
    ccv_nnc_tensor_t* outputs[] = {co, cp};
    const auto start = std::chrono::steady_clock::now();
    const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, inputs, mode == 2 || mode == 3 ? 4 : 3, outputs, mode == 1 ? 2 : 1, stream);
    ccv_nnc_stream_context_wait(stream);
    if (status != CCV_NNC_EXEC_SUCCESS) std::exit(2);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  };
  for (int mode = 0; mode < modes; ++mode) {
    run(mode);
    std::cout << "public_verify_begin " << names[mode] << std::endl;
    publicRun(mode);
    ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(co, cp), TENSOR_LIST(ho, hp), nullptr);
    const bool equal = !std::memcmp(ho->data.i32, selected->contents(), (size_t)T * 512 * sizeof(int32_t)) &&
      (mode != 1 || !std::memcmp(hp->data.i32, candidates->contents(), (size_t)T * 2048 * sizeof(int32_t)));
    std::cout << "public_verify_end " << names[mode] << " outputs_equal=" << equal << std::endl;
    if (!equal) return 2;
  }
  std::vector<double> wallTimes[modes];
  for (int i = 0; i < warmup + timed; ++i)
    for (int j = 0; j < modes; ++j) {
      const int mode = (i + j) % modes;
      const double ms = publicRun(mode);
      if (i >= warmup) wallTimes[mode].push_back(ms);
    }
  for (int mode = 0; mode < modes; ++mode) {
    if (!timed) continue;
    std::sort(wallTimes[mode].begin(), wallTimes[mode].end());
    std::cout << "public_" << names[mode] << " median_wall_ms=" << wallTimes[mode][timed / 2] << '\n';
  }
  if (oldFlags & CCV_NNC_DISABLE_MFA) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
  if (oldFlags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
  ccv_nnc_stream_context_free(stream);
  for (auto tensor : owned) ccv_nnc_tensor_free(tensor);
  for (int mode = 0; mode < modes; ++mode) ccv_nnc_deinit_mfa_context(contexts[mode]);
  pool->drain();
  return 0;
}
