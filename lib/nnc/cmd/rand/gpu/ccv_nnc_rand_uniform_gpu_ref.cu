extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <curand_kernel.h>

__global__ void _ccv_nnc_random_uniform_kernel(const int count, const int seed, const float l, const float u, float* const a)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(seed, id, 0, &state);
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float r = curand_uniform(&state); // This is from 0 to 1 open close.
		a[i] = r * u + (1 - r) * l;
	}
}

static int _ccv_nnc_random_uniform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == 1);
	ccv_nnc_tensor_t* const a = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	const int count = ccv_nnc_tensor_count(a->info);
	const int seed = (int)(intptr_t)a->data.f32;
	const float l = cmd.info.blas.a[0];
	const float u = cmd.info.blas.a[1];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	const int device = ccv_nnc_stream_context_get_device(stream_context);
	cudaSetDevice(device);
	_ccv_nnc_random_uniform_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, seed, l, u, a->data.f32);
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_uniform;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_UNIFORM_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_uniform;
}
