extern "C" {
#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
}
#include <nnc/gpu/ccv_nnc_compat.h>
#include <curand_kernel.h>

template<typename NUM>
__global__ void _ccv_nnc_random_uniform_kernel_x4(const int count, const uint32_t seed, const float l, const float u, NUM* const a)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t state;
	curand_init(seed, id, 0, &state);
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float4 r = curand_uniform4(&state); // This is from 0 to 1 open close.
		a[i * 4] = r.x * u + (1 - r.x) * l;
		a[i * 4 + 1] = r.y * u + (1 - r.y) * l;
		a[i * 4 + 2] = r.z * u + (1 - r.z) * l;
		a[i * 4 + 3] = r.w * u + (1 - r.w) * l;
	}
}

template<typename NUM>
__global__ void _ccv_nnc_random_uniform_kernel(const int count, const uint32_t seed, const float l, const float u, NUM* const a)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	curandStatePhilox4_32_10_t state;
	curand_init(seed, id, 0, &state);
	CUDA_1D_KERNEL_LOOP(i, count) {
		const float r = curand_uniform(&state); // This is from 0 to 1 open close.
		a[i] = r * u + (1 - r) * l;
	}
}

static int _ccv_nnc_random_uniform(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(output_size == 1);
	ccv_nnc_tensor_t* const a = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	const int count = ccv_nnc_tensor_count(a->info);
	const uint32_t seed = ccv_nnc_stream_context_genrand_uint32(stream_context);
	const float l = cmd.info.blas.a[0];
	const float u = cmd.info.blas.a[1];
	cudaStream_t stream = ccv_nnc_stream_context_get_stream(stream_context);
	if (count % 4 == 0)
	{
		const int count_4 = count / 4;
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_random_uniform_kernel_x4<<<CUDA_GET_BLOCKS(count_4), CUDA_NUM_THREADS, 0, stream>>>(count_4, seed, l, u, a->data.f32);
		else if (a->info.datatype == CCV_16F)
			_ccv_nnc_random_uniform_kernel_x4<<<CUDA_GET_BLOCKS(count_4), CUDA_NUM_THREADS, 0, stream>>>(count_4, seed, l, u, (__half*)a->data.f16);
	} else {
		if (a->info.datatype == CCV_32F)
			_ccv_nnc_random_uniform_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, seed, l, u, a->data.f32);
		else if (a->info.datatype == CCV_16F)
			_ccv_nnc_random_uniform_kernel<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(count, seed, l, u, (__half*)a->data.f16);
	}
	return CCV_NNC_EXEC_SUCCESS;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_UNIFORM_FORWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_uniform;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_RANDOM_UNIFORM_BACKWARD, CCV_NNC_BACKEND_GPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_16F;
	registry->tensor_memory = CCV_TENSOR_GPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_random_uniform;
}
