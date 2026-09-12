#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/MoERoutingKernel.hpp"

// Compare launch policies using the production shader. Compile and warm both
// pipelines before timing, alternate their order, and use GPU command-buffer
// timestamps over repeated serial dispatches to amortize submission overhead.
int main(int argc, char** argv)
{
	const int iterations = argc > 1 ? atoi(argv[1]) : 80;
	const int dispatches = argc > 2 ? atoi(argv[2]) : 256;
	const int requested_experts = argc > 3 ? atoi(argv[3]) : 0;
	const int requested_hidden = argc > 4 ? atoi(argv[4]) : 5120;
	const int warmup = 10;
	const uint32_t hidden = requested_hidden;
	const uint32_t kth = 6;
	if (iterations <= 0 || dispatches <= 0 || requested_hidden <= 0 || requested_experts < 0 || requested_experts > 512 ||
		(requested_experts > 0 && requested_experts < (int)kth))
	{
		fprintf(stderr, "usage: %s [iterations>0] [dispatches>0] [experts=0(sweep)|6..512] [hidden>0]\n", argv[0]);
		return 1;
	}
	auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
	auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!device)
	{
		fprintf(stderr, "Metal device unavailable.\n");
		return 1;
	}
	auto queue = NS::TransferPtr(device->newCommandQueue());
	std::unordered_map<MoERoutingKernelDescriptor, std::unique_ptr<MoERoutingKernel>> kernels;
	std::vector<uint32_t> expert_counts = {8, 32, 128, 256, 257, 383, 384, 385, 511, 512};
	if (requested_experts)
		expert_counts = {(uint32_t)requested_experts};
	printf("device=%s warmup=%d iterations=%d dispatches=%d H=%u K=%u logits=fp32 activation=fp16\n",
		device->name()->utf8String(), warmup, iterations, dispatches, hidden, kth);
	printf("E,mode,activation,bucket_threads,aligned_threads,bucket_median_us,aligned_median_us,paired_speedup,outputs_equal\n");
	fflush(stdout);
	for (const uint32_t expert_count : expert_counts)
		for (int mode = 0; mode < 4; ++mode)
		{
			auto case_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
			const bool preselected = (mode & 2) != 0;
			const bool single_input_token = (mode & 1) != 0;
			const float weight_scale = 1.5f;
			const uint32_t threads[] = {expert_count <= 256 ? 256u : 512u, ((expert_count + 31) / 32) * 32};
			NS::SharedPtr<MTL::ComputePipelineState> pipelines[2];
			for (int policy = 0; policy < 2; ++policy)
			{
				const MoERoutingKernelDescriptor descriptor {16, 3, threads[policy] / 32};
				auto iterator = kernels.find(descriptor);
				if (iterator == kernels.end())
					iterator = kernels.try_emplace(descriptor, std::make_unique<MoERoutingKernel>(descriptor, device.get())).first;
				auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
				constants->setConstantValue(&expert_count, MTL::DataTypeUInt, NS::UInteger(0));
				constants->setConstantValue(&kth, MTL::DataTypeUInt, 1);
				constants->setConstantValue(&hidden, MTL::DataTypeUInt, 2);
				constants->setConstantValue(&weight_scale, MTL::DataTypeFloat, 3);
				constants->setConstantValue(&preselected, MTL::DataTypeBool, 4);
				constants->setConstantValue(&single_input_token, MTL::DataTypeBool, 5);
				NS::Error* error = nullptr;
				auto function = NS::TransferPtr(iterator->second->library->newFunction(NS::String::string("moe_routing_t1", NS::UTF8StringEncoding), constants.get(), &error));
				CCV_NNC_MFA_CHECK_ERROR(error);
				pipelines[policy] = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
				CCV_NNC_MFA_CHECK_ERROR(error);
				CCV_NNC_MFA_PRECONDITION(pipelines[policy]->threadExecutionWidth() == 32);
				CCV_NNC_MFA_PRECONDITION(threads[policy] <= pipelines[policy]->maxTotalThreadsPerThreadgroup());
			}
			const size_t gathered_count = (single_input_token ? 1 : kth) * hidden;
			const size_t sizes[] = {expert_count * sizeof(float), expert_count * sizeof(float), hidden * sizeof(_Float16),
				gathered_count * sizeof(_Float16), kth * sizeof(float), kth * sizeof(int), kth * sizeof(int), kth * sizeof(int)};
			NS::SharedPtr<MTL::Buffer> buffers[8];
			for (int i = 0; i < 8; ++i)
			{
				buffers[i] = NS::TransferPtr(device->newBuffer(sizes[i], MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
				CCV_NNC_MFA_PRECONDITION(buffers[i]);
				memset(buffers[i]->contents(), 0, sizes[i]);
			}
			std::mt19937 random(42 + expert_count);
			std::uniform_real_distribution<float> distribution(-4, 4);
			for (uint32_t i = 0; i < expert_count; ++i)
			{
				((float*)buffers[0]->contents())[i] = distribution(random);
				((float*)buffers[1]->contents())[i] = distribution(random) * 0.01f;
			}
			if (preselected)
				for (uint32_t i = 0; i < kth; ++i)
					((int*)buffers[1]->contents())[i] = (i * 43 + 7) % expert_count;
			for (uint32_t i = 0; i < hidden; ++i)
				((_Float16*)buffers[2]->contents())[i] = (_Float16)distribution(random);
			const auto run = [&](const int policy, const int count) {
				auto run_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
				auto command_buffer = NS::RetainPtr(queue->commandBuffer());
				auto encoder = command_buffer->computeCommandEncoder();
				encoder->setComputePipelineState(pipelines[policy].get());
				for (uint32_t i = 0; i < 8; ++i)
					encoder->setBuffer(buffers[i].get(), 0, i);
				encoder->setThreadgroupMemoryLength(expert_count * sizeof(float), 0);
				for (int i = 0; i < count; ++i)
					encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(threads[policy], 1, 1));
				encoder->endEncoding();
				command_buffer->commit();
				command_buffer->waitUntilCompleted();
				CCV_NNC_MFA_CHECK_ERROR(command_buffer->error());
				const double seconds = command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
				CCV_NNC_MFA_PRECONDITION(seconds > 0);
				return seconds * 1e6 / count;
			};
			run(0, 1);
			std::vector<unsigned char> expected[5];
			for (int i = 3; i < 8; ++i)
			{
				const unsigned char* const bytes = (const unsigned char*)buffers[i]->contents();
				expected[i - 3].assign(bytes, bytes + sizes[i]);
				memset(buffers[i]->contents(), 0xff, sizes[i]);
			}
			run(1, 1);
			for (int i = 3; i < 8; ++i)
				if (memcmp(expected[i - 3].data(), buffers[i]->contents(), sizes[i]) != 0)
				{
					fprintf(stderr, "Launch policies disagree: E=%u mode=%d buffer=%d\n", expert_count, mode, i);
					return 1;
				}
			std::vector<double> samples[2];
			std::vector<double> speedups;
			for (int iteration = 0; iteration < warmup + iterations; ++iteration)
			{
				double elapsed[2];
				for (int order = 0; order < 2; ++order)
				{
					const int policy = (iteration + order) & 1;
					elapsed[policy] = run(policy, dispatches);
					if (iteration >= warmup)
						samples[policy].push_back(elapsed[policy]);
				}
				if (iteration >= warmup)
					speedups.push_back(elapsed[0] / elapsed[1]);
			}
			const auto median = [](std::vector<double>& values) {
				std::sort(values.begin(), values.end());
				return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
			};
			printf("%u,%s,%s,%u,%u,%.4f,%.4f,%.4f,yes\n", expert_count,
				preselected ? "preselected" : "top6", single_input_token ? "single" : "expanded",
				threads[0], threads[1], median(samples[0]), median(samples[1]), median(speedups));
			fflush(stdout);
		}
	return 0;
}
