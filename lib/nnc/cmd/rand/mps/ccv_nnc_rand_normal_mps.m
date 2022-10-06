#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>

static int _ccv_nnc_random_normal(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	int i, j;
	const uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
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
			for (j = 0; j < nd; j++)
				[shape addObject:@(a->info.dim[j])];
			MPSGraphRandomOpDescriptor* descriptor = [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionNormal dataType:ccv_nnc_mps_datatype(a->info.datatype)];
			descriptor.mean = mean;
			descriptor.standardDeviation = std;
			MPSGraphTensor* mps_a = [graph randomTensorWithShape:shape descriptor:descriptor seed:(NSUInteger)seed name:nil];
			[shape release];
			ccv_nnc_mps_graph_result(graph, command_buffer, @{}, mps_a, a, a->info.dim, a->stride);
			[graph release];
		}
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
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
