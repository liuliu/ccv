#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

// State independent and portable srand48 / mrand48 implementation.
// These are used by MPSGraph to generate counterLow, counterHigh, key from seed.

#define A 0x5DEECE66DULL
#define C 0xBULL
#define M (1ULL << 48)

static uint64_t stateless_srand48(long s)
{
	return (((uint64_t)s) << 16) | 0x330EULL;
}

// Generate a pseudo-random number
static uint32_t stateless_mrand48(uint64_t* seed)
{
	seed[0] = (A * seed[0] + C) & (M - 1);
	// Return 32 significant bits as a signed long
	return (uint32_t)(seed[0] >> (48 - 32));
}

static int _ccv_nnc_random_normal(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	const float std = cmd.info.blas.a[0];
	const float mean = cmd.info.blas.a[1];
	@autoreleasepool {
		MPSCommandBuffer* command_buffer = ccv_nnc_stream_context_start_mps_command_buffer(stream_context);
		for (i = 0; i < output_size; i++)
		{
			uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
			ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)outputs[i];
			NSMutableArray<NSNumber*>* shape = [NSMutableArray new];
			const int nd = ccv_nnc_tensor_nd(a->info.dim);
			NSMutableArray<NSNumber*>* axes = [NSMutableArray new];
			for (j = 0; j < nd; j++)
			{
				[shape addObject:@(a->info.dim[j])];
				[axes addObject:@(j)];
			}
			ccv_nnc_mps_graph_key_t key = ccv_nnc_mps_graph_key_new(cmd, 0, hint, flags, 0, 0, outputs + i, 1);
			int indices[1];
			MPSGraphExecutable* executable = ccv_nnc_mps_graph_executable_cache(key, indices, ^void (MPSGraph* graph, NSMutableArray<MPSGraphTensor*>* inputTensors, NSMutableArray<MPSGraphShapedType*>* inputShapedTypes, NSMutableArray<MPSGraphTensor*>* resultTensors) {
				MPSGraphTensor* mps_state = [graph placeholderWithShape:@[@7] dataType:MPSDataTypeInt32 name:nil];
				[inputTensors addObject:mps_state];
				MPSGraphShapedType* mps_state_shape = [[MPSGraphShapedType alloc] initWithShape:@[@7] dataType:MPSDataTypeInt32];
				[inputShapedTypes addObject:mps_state_shape];
				[mps_state_shape release];
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
				[resultTensors addObject:mps_r[0]];
			});
			uint32_t states[7];
			states[0] = 1;
			// Note that MPSGraph uses srand48 to initialize from seed. We have to simulate that to have compatibility with old implementation.
			// This new implementation allows us to cache the MPS executable to avoid the compilation penalty / framework-related memory leaks
			// every time use the random number generator.
			uint64_t rand48_seed = stateless_srand48((long)seed);
			states[2] = stateless_mrand48(&rand48_seed); // counterLow
			states[1] = stateless_mrand48(&rand48_seed);
			states[4] = stateless_mrand48(&rand48_seed); // counterHigh
			states[3] = stateless_mrand48(&rand48_seed);
			states[6] = stateless_mrand48(&rand48_seed); // key
			states[5] = stateless_mrand48(&rand48_seed);
			NSData* state = [[NSData alloc] initWithBytesNoCopy:states length:7 freeWhenDone:NO];
			MPSGraphTensorData* data_state = [[MPSGraphTensorData alloc] initWithDevice:ccv_nnc_default_mps_device() data:state shape:@[@7] dataType:MPSDataTypeInt32];
			ccv_nnc_mps_graph_executable_result(executable, command_buffer, @[data_state], &a, (int*[]){ a->info.dim }, (int*[]){ a->stride }, 1, 0);
			[shape release];
			[axes release];
			[data_state release];
			[state release];
		}
		ccv_nnc_stream_context_finish_mps_command_buffer(stream_context, command_buffer);
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
