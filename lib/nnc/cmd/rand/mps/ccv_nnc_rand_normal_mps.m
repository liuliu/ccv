#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_random_normal(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
	const float std = cmd.info.blas.a[0];
	const float mean = cmd.info.blas.a[1];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_get_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[i];
			MPSGraph *graph = [MPSGraph new];
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
			for (j = 0; j < nd; j++)
			{
				[shape addObject:@(a->info.dim[j])];
				[axes addObject:@(j)];
			}
			MPSGraphTensor* mps_state = [graph randomPhiloxStateTensorWithSeed:(NSUInteger)seed name:nil];
			MPSGraphRandomOpDescriptor* descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionNormal dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			descriptor.mean = mean;
			descriptor.standardDeviation = std;
			// Using a while loop until found a tensor without nan. Somehow MPSGraph returns random tensor with some nans. This can be reproduced when you set the seed to 2069102477
			// with philox state as: {counter_high = -5820494845676086657 : i64, counter_low = 149709389795223126 : i64, key = 5952859558996586250 : i64}
			NSArray<MPSGraphTensor*>* mps_r = [graph whileWithInitialInputs:@[mps_state] before:^MPSGraphTensor*(NSArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				NSArray<MPSGraphTensor*>* mps_results = [graph randomTensorWithShape:shape descriptor:descriptor stateTensor:inputTensors[0] name:nil];
				[resultTensors addObject:mps_results[0]];
				[resultTensors addObject:mps_results[1]];
				MPSGraphTensor* mps_nan = [graph isNaNWithTensor:mps_results[0] name:nil];
				return [graph reductionOrWithTensor:mps_nan axes:axes name:nil];
			} after:^NSArray<MPSGraphTensor*>*(NSArray<MPSGraphTensor*>* bodyBlockArguments) {
				return @[bodyBlockArguments[1]];
			} name:nil];
			[shape release];
			[axes release];
			ccv_nnc_mps_graph_result(graph, command_buffer, @{}, mps_r[0], a, a->info.dim, a->stride);
			[graph release];
		}
		ccv_nnc_stream_context_commit_command_buffer(stream_context, command_buffer);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_NORMAL_FORWARD, CCV_NNC_BACKEND_MPS)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_normal;
}
