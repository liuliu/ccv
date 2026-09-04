#include "MoEWeightsStreamingKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

MoEWeightsStreamingKernel::MoEWeightsStreamingKernel(
	MoEWeightsStreamingKernelDescriptor, MTL::Device* const device)
{
	const std::string shader = R"(
#pragma METAL internals : enable
#ifndef __METAL_MEMORY_SCOPE_SYSTEM__
#define __METAL_MEMORY_SCOPE_SYSTEM__ 3
#endif
#include <metal_stdlib>
namespace metal {
constexpr constant metal::thread_scope thread_scope_system =
	static_cast<thread_scope>(__METAL_MEMORY_SCOPE_SYSTEM__);
}
using namespace metal;

struct Plan {
	uint generation;
	uint desired_count;
	uint load_count;
	uint invalid;
};

struct Params {
	uint generation;
	uint index_count;
	uint expert_count;
	uint resident_slots;
	uint routing_width;
	uint route_weight_count;
	uint route_weight_bytes;
};

kernel void moe_weights_streaming(
	device const int* input_indices [[buffer(0)]],
	device const int* input_counts [[buffer(1)]],
	device const uchar* input_route_weights [[buffer(2)]],
	device int* output_indices [[buffer(3)]],
	device int* output_counts [[buffer(4)]],
	device uchar* output_route_weights [[buffer(5)]],
	device int* logical_to_slot [[buffer(6)]],
	device int* slot_to_logical [[buffer(7)]],
	device uint* last_used [[buffer(8)]],
	device Plan* plan [[buffer(9)]],
	device atomic_uint* ready_generation [[buffer(10)]],
	constant Params& p [[buffer(11)]])
{
	device int* desired_experts = (device int*)(plan + 1);
	device int* desired_slots = desired_experts + p.expert_count;
	device int* load_experts = desired_slots + p.expert_count;
	device int* load_slots = load_experts + p.expert_count;
	plan->generation = p.generation;
	plan->desired_count = 0;
	plan->load_count = 0;
	plan->invalid = 0;
	const bool prefill = p.route_weight_count != p.routing_width;
	if (p.generation == 1)
	{
		for (uint i = 0; i < p.expert_count; i++)
			logical_to_slot[i] = -1;
		for (uint i = 0; i < p.resident_slots; i++)
		{
			slot_to_logical[i] = -1;
			last_used[i] = 0;
		}
	}
	if (p.index_count == 0 || p.index_count > p.expert_count ||
		p.resident_slots == 0 || p.resident_slots > p.expert_count)
	{
		plan->invalid = 1;
		return;
	}
	uint total = 0;
	for (uint i = 0; i < p.index_count; i++)
	{
		const int count = input_counts[i];
		output_counts[i] = prefill ? 0 : count;
		output_indices[i] = 0;
		if (count < 0 || uint(count) > p.route_weight_count - total)
		{
			plan->invalid = 1;
			return;
		}
		total += uint(count);
		if (count == 0)
			continue;
		const int expert = input_indices[i];
		if (expert < 0 || uint(expert) >= p.expert_count ||
			plan->desired_count >= (prefill ? p.expert_count : p.resident_slots))
		{
			plan->invalid = 1;
			return;
		}
		for (uint j = 0; j < plan->desired_count; j++)
			if (desired_experts[j] == expert)
			{
				plan->invalid = 1;
				return;
			}
		const uint desired = plan->desired_count++;
		desired_experts[desired] = expert;
		desired_slots[desired] = -1;
	}
	if (total != p.route_weight_count)
	{
		plan->invalid = 1;
		return;
	}
	for (uint i = 0; i < plan->desired_count; i++)
	{
		const int expert = desired_experts[i];
		int slot = logical_to_slot[expert];
		if (slot >= 0 && (uint(slot) >= p.resident_slots || slot_to_logical[slot] != expert))
		{
			plan->invalid = 1;
			return;
		}
		if (prefill)
		{
			desired_slots[i] = slot;
			if (slot < 0)
			{
				const uint load = plan->load_count++;
				load_experts[load] = expert;
				load_slots[load] = int(i);
			}
			continue;
		}
		if (slot < 0)
		{
			for (uint j = 0; j < p.resident_slots; j++)
				if (slot_to_logical[j] < 0)
				{
					slot = int(j);
					break;
				}
			if (slot < 0)
			{
				uint oldest = 0xffffffffu;
				for (uint j = 0; j < p.resident_slots; j++)
				{
					const int resident = slot_to_logical[j];
					bool protected_slot = false;
					for (uint k = 0; k < plan->desired_count; k++)
						if (desired_experts[k] == resident)
						{
							protected_slot = true;
							break;
						}
					if (!protected_slot && last_used[j] < oldest)
					{
						oldest = last_used[j];
						slot = int(j);
					}
				}
			}
			if (slot < 0)
			{
				plan->invalid = 1;
				return;
			}
			const int evicted = slot_to_logical[slot];
			if (evicted >= 0)
				logical_to_slot[evicted] = -1;
			logical_to_slot[expert] = slot;
			slot_to_logical[slot] = expert;
			const uint load = plan->load_count++;
			load_experts[load] = expert;
			load_slots[load] = slot;
		}
		last_used[slot] = p.generation * p.expert_count + i + 1;
		desired_slots[i] = slot;
	}
	uint desired = 0;
	for (uint i = 0; i < p.index_count; i++)
		if (input_counts[i] > 0)
		{
			if (prefill)
			{
				output_indices[desired] = int(desired);
				output_counts[desired] = input_counts[i];
			} else
				output_indices[i] = desired_slots[desired];
			desired++;
		}
	for (uint i = 0; i < p.route_weight_bytes; i++)
		output_route_weights[i] = input_route_weights[i];
	const bool cpu_work = prefill ? plan->desired_count != 0 : plan->load_count != 0;
	if (!cpu_work)
	{
		uint ready = atomic_load_explicit(ready_generation, memory_order_relaxed);
		while (ready < p.generation && !atomic_compare_exchange_weak_explicit(
			ready_generation, &ready, p.generation, memory_order_relaxed,
			memory_order_relaxed)) {}
		atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
			thread_scope_system);
	}
}
)";
	NS::Error* error = nil;
	library = NS::TransferPtr(device->newLibrary(
		NS::String::string(shader.c_str(), NS::UTF8StringEncoding), nil, &error));
	CCV_NNC_MFA_CHECK_ERROR(error);
}
