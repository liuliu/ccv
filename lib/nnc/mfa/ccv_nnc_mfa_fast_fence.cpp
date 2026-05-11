#include "ccv_nnc_mfa.hpp"

#include <algorithm>
#include <mutex>

using namespace ccv::nnc;

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
)";
}

static int _ccv_nnc_mfa_ensure_fast_fence_pipeline(mfa::context* const context, NS::SharedPtr<MTL::ComputePipelineState>* const coherent_pipeline_ref, NS::SharedPtr<MTL::ComputePipelineState>* const update_pipeline_ref)
{
	if (!context || !ccv_nnc_mfa_context_supported(context) || (ccv_nnc_flags() & CCV_NNC_DISABLE_MFA))
		return 0;
	static std::mutex mutex;
	static int attempted = 0;
	static NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	static NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	std::lock_guard<std::mutex> lock(mutex);
	if (coherent_pipeline.get() && update_pipeline.get())
	{
		*coherent_pipeline_ref = coherent_pipeline;
		*update_pipeline_ref = update_pipeline;
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
	auto coherent_function = NS::TransferPtr(library->newFunction(coherent_name, constants.get(), &error));
	if (!coherent_function.get() || error)
		return 0;
	auto update_function = NS::TransferPtr(library->newFunction(update_name, constants.get(), &error));
	if (!update_function.get() || error)
		return 0;
	coherent_pipeline = NS::TransferPtr(context->device->newComputePipelineState(coherent_function.get(), &error));
	if (!coherent_pipeline.get() || error)
		return 0;
	update_pipeline = NS::TransferPtr(context->device->newComputePipelineState(update_function.get(), &error));
	if (!update_pipeline.get() || error)
		return 0;
	*coherent_pipeline_ref = coherent_pipeline;
	*update_pipeline_ref = update_pipeline;
	return 1;
}

int ccv_nnc_mfa_prepare_fast_fence(ccv_nnc_mfa_context_t* const context)
{
	NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	return _ccv_nnc_mfa_ensure_fast_fence_pipeline(context, &coherent_pipeline, &update_pipeline);
}

int ccv_nnc_mfa_encode_fast_fence(ccv_nnc_mfa_context_t* const context, const ccv_nnc_mfa_fast_fence_params_t params, mtl_command_batch_t* const command_batch, mtl_buffer_t** const tensors, size_t* const tensor_offsets)
{
	NS::SharedPtr<MTL::ComputePipelineState> coherent_pipeline;
	NS::SharedPtr<MTL::ComputePipelineState> update_pipeline;
	if (!_ccv_nnc_mfa_ensure_fast_fence_pipeline(context, &coherent_pipeline, &update_pipeline))
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
