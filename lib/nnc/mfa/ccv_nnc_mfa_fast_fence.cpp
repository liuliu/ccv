#include "ccv_nnc_mfa.hpp"

#include <algorithm>
#include <mutex>

using namespace ccv::nnc;

// A wait spins at 220 to 300 ns per iteration on M5 Max, so it gives up after 30 to 40 seconds
// (slower GPUs wait longer). A live CPU producer finishes far sooner; a full-prefill layer load takes a few
// seconds. One whose producer is gone (process exit, crash, a suspended loader) would otherwise
// spin forever, and a running kernel keeps its process from finishing exit and freeing its memory.
static const uint32_t ccv_nnc_mfa_fast_fence_wait_budget = 1u << 27;

// Every wait shares one abort word, so the first timeout or a cancellation releases all queued
// waits at once rather than each spinning out its own budget.
enum {
	CCV_NNC_MFA_FAST_FENCE_WAITING = 0,
	CCV_NNC_MFA_FAST_FENCE_TIMED_OUT = 1,
	CCV_NNC_MFA_FAST_FENCE_CANCELLED = 2,
};

static uint32_t* ccv_nnc_mfa_fast_fence_abort_state; // Published once under the pipeline lock, so cancellation needs no lock.

static const char* _ccv_nnc_mfa_fast_fence_source(void)
{
	return R"(
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
#include <metal_stdlib>
#include <metal_atomic>
namespace metal {
constexpr constant metal::thread_scope thread_scope_system = static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
using namespace metal;

kernel void ccv_nnc_mfa_fast_fence_input_coherent(
  volatile coherent(system) device uint* input [[buffer(0)]],
  constant uint& word_offset [[buffer(1)]],
  constant uint& word_count [[buffer(2)]],
  uint index [[thread_position_in_grid]]
) {
  if (index < word_count)
    input[word_offset + index] = input[word_offset + index];
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
}

kernel void ccv_nnc_mfa_fast_fence_update(
  device atomic_uint* timestamp [[buffer(0)]],
  constant uint& expected [[buffer(1)]],
  constant uint& value [[buffer(2)]]
) {
  uint expected_value = expected;
  while (expected_value == expected &&
    !atomic_compare_exchange_weak_explicit(timestamp, &expected_value, value, memory_order_relaxed, memory_order_relaxed)) {}
  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
}

kernel void ccv_nnc_mfa_fast_fence_wait(
  volatile coherent(system) device uint* timestamp [[buffer(0)]],
  constant uint& value [[buffer(1)]],
  volatile coherent(system) device uint* abort_state [[buffer(2)]],
  constant uint& budget [[buffer(3)]]
) {
  for (uint spin = 0; ; spin++) {
    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
    // A second coherent read nearly doubles each spin, and the abort word only changes at teardown
    // or after a timeout, so check it every 8192 spins (first spin included).
    if (timestamp[0] >= value || ((spin & 8191) == 0 && abort_state[0] != 0))
      break;
    if (spin == budget) {
      abort_state[0] = 1; // CCV_NNC_MFA_FAST_FENCE_TIMED_OUT
      metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_system);
      break;
    }
  }
}

)";
}

static int _ccv_nnc_mfa_ensure_fast_fence_pipeline(mfa::context* const context, NS::SharedPtr<MTL::ComputePipelineState>* const coherent_pipeline_ref, NS::SharedPtr<MTL::ComputePipelineState>* const update_pipeline_ref, NS::SharedPtr<MTL::ComputePipelineState>* const wait_pipeline_ref, NS::SharedPtr<MTL::Buffer>* const abort_buffer_ref)
{
	if (!context || !ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		return 0;
	static std::mutex mutex;
	static int attempted = 0;
	static NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	static NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	static NS::SharedPtr<MTL::ComputePipelineState> wait_pipeline;
	static NS::SharedPtr<MTL::Buffer> abort_buffer;
	std::lock_guard<std::mutex> lock(mutex);
	if (coherent_pipeline.get() && update_pipeline.get() && wait_pipeline.get() && abort_buffer.get())
	{
		*coherent_pipeline_ref = coherent_pipeline;
		*update_pipeline_ref = update_pipeline;
		*wait_pipeline_ref = wait_pipeline;
		if (abort_buffer_ref)
			*abort_buffer_ref = abort_buffer;
		return 1;
	}
	if (attempted)
		return 0;
	attempted = 1;
	auto string = NS::String::string(_ccv_nnc_mfa_fast_fence_source(), NS::UTF8StringEncoding);
	NS::Error* error = nil;
	auto library = NS::TransferPtr(context->device->newLibrary(string, nil, &error));
	if (!library.get() || error)
		return 0;
	auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
	auto coherent_name = NS::String::string("ccv_nnc_mfa_fast_fence_input_coherent", NS::UTF8StringEncoding);
	auto update_name = NS::String::string("ccv_nnc_mfa_fast_fence_update", NS::UTF8StringEncoding);
	auto wait_name = NS::String::string("ccv_nnc_mfa_fast_fence_wait", NS::UTF8StringEncoding);
	auto coherent_function = NS::TransferPtr(library->newFunction(coherent_name, constants.get(), &error));
	if (!coherent_function.get() || error)
		return 0;
	auto update_function = NS::TransferPtr(library->newFunction(update_name, constants.get(), &error));
	if (!update_function.get() || error)
		return 0;
	auto wait_function = NS::TransferPtr(library->newFunction(wait_name, constants.get(), &error));
	if (!wait_function.get() || error)
		return 0;
	coherent_pipeline = NS::TransferPtr(context->device->newComputePipelineState(coherent_function.get(), &error));
	if (!coherent_pipeline.get() || error)
		return 0;
	update_pipeline = NS::TransferPtr(context->device->newComputePipelineState(update_function.get(), &error));
	if (!update_pipeline.get() || error)
		return 0;
	wait_pipeline = NS::TransferPtr(context->device->newComputePipelineState(wait_function.get(), &error));
	if (!wait_pipeline.get() || error)
		return 0;
	// The CPU and GPU order accesses to this word themselves; Metal hazard tracking would only
	// serialize unrelated waits.
	abort_buffer = NS::TransferPtr(context->device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeDefaultCache | MTL::ResourceHazardTrackingModeUntracked));
	if (!abort_buffer.get())
		return 0;
	uint32_t* const abort_state = (uint32_t*)abort_buffer->contents();
	__atomic_store_n(abort_state, CCV_NNC_MFA_FAST_FENCE_WAITING, __ATOMIC_RELAXED);
	__atomic_store_n(&ccv_nnc_mfa_fast_fence_abort_state, abort_state, __ATOMIC_RELEASE);
	*coherent_pipeline_ref = coherent_pipeline;
	*update_pipeline_ref = update_pipeline;
	*wait_pipeline_ref = wait_pipeline;
	if (abort_buffer_ref)
		*abort_buffer_ref = abort_buffer;
	return 1;
}

int ccv_nnc_mfa_prepare_fast_fence(ccv_nnc_mfa_context_t* const context)
{
	NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> wait_pipeline;
	return _ccv_nnc_mfa_ensure_fast_fence_pipeline(context, &coherent_pipeline, &update_pipeline, &wait_pipeline, 0);
}

int ccv_nnc_mfa_encode_fast_fence(ccv_nnc_mfa_context_t* const context, const ccv_nnc_mfa_fast_fence_params_t params, mtl_command_batch_t* const command_batch, mtl_buffer_t** const tensors, size_t* const tensor_offsets)
{
	NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> wait_pipeline;
	if (!_ccv_nnc_mfa_ensure_fast_fence_pipeline(context, &coherent_pipeline, &update_pipeline, &wait_pipeline, 0))
		return 0;
	if (!command_batch || !tensors || !tensor_offsets || !tensors[0] || !tensors[1] || params.word_count == 0)
		return 0;
	auto encoder = command_batch->startCommand();
	const NS::UInteger threads = std::min<NS::UInteger>(1024, coherent_pipeline->maxTotalThreadsPerThreadgroup());
	const NS::UInteger groups = (params.word_count + threads - 1) / threads;
	encoder->setComputePipelineState(coherent_pipeline.get());
	encoder->setBuffer(tensors[0], tensor_offsets[0], NS::UInteger(0));
	encoder->setBytes(&params.word_offset, sizeof(params.word_offset), NS::UInteger(1));
	encoder->setBytes(&params.word_count, sizeof(params.word_count), NS::UInteger(2));
	encoder->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));
	encoder->memoryBarrier(MTL::BarrierScopeBuffers);
	encoder->setComputePipelineState(update_pipeline.get());
	encoder->setBuffer(tensors[1], tensor_offsets[1], NS::UInteger(0));
	encoder->setBytes(&params.pending, sizeof(params.pending), NS::UInteger(1));
	encoder->setBytes(&params.complete, sizeof(params.complete), NS::UInteger(2));
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
	command_batch->finishCommand(encoder);
	return 1;
}

int ccv_nnc_mfa_encode_fast_fence_wait(ccv_nnc_mfa_context_t* const context, const uint32_t value, mtl_command_batch_t* const command_batch, mtl_buffer_t* const timestamp, const size_t timestamp_offset)
{
	NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> wait_pipeline;
	NS::SharedPtr<MTL::Buffer> abort_buffer;
	if (!_ccv_nnc_mfa_ensure_fast_fence_pipeline(context, &coherent_pipeline, &update_pipeline, &wait_pipeline, &abort_buffer) ||
		!command_batch || !timestamp || value == 0)
		return 0;
	// Work queued behind a timed-out wait may already have run without its data. Report that, then
	// let later waits wait again: the process is alive, so their producers are too.
	uint32_t timed_out = CCV_NNC_MFA_FAST_FENCE_TIMED_OUT;
	if (__atomic_compare_exchange_n((uint32_t*)abort_buffer->contents(), &timed_out, CCV_NNC_MFA_FAST_FENCE_WAITING, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
		ccv_nnc_mfa_log_message("A GPU wait for CPU-produced data timed out; GPU work that depended on it ran without it.");
	auto encoder = command_batch->startCommand();
	encoder->setComputePipelineState(wait_pipeline.get());
	encoder->setBuffer(timestamp, timestamp_offset, NS::UInteger(0));
	encoder->setBytes(&value, sizeof(value), NS::UInteger(1));
	encoder->setBuffer(abort_buffer.get(), 0, NS::UInteger(2));
	encoder->setBytes(&ccv_nnc_mfa_fast_fence_wait_budget, sizeof(ccv_nnc_mfa_fast_fence_wait_budget), NS::UInteger(3));
	encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
	encoder->memoryBarrier(MTL::BarrierScopeBuffers);
	command_batch->finishCommand(encoder);
	return 1;
}

void ccv_nnc_mfa_cancel_fast_fence_waits(void)
{
	// Never cleared: once cancelled, waits encoded later pass immediately too.
	uint32_t* const abort_state = __atomic_load_n(&ccv_nnc_mfa_fast_fence_abort_state, __ATOMIC_ACQUIRE);
	if (abort_state)
		__atomic_store_n(abort_state, CCV_NNC_MFA_FAST_FENCE_CANCELLED, __ATOMIC_RELEASE);
}
