extern "C" {
#include "ccv_nnc_cuda.h"
}
#include <cuda.h>

void* gcmalloc(size_t size)
{
	void* ptr = 0;
	cudaMalloc(&ptr, size);
	return ptr;
}

void gcmemcpy(void* dst, const void* src, size_t size, int kind)
{
	if (kind == CCV_NNC_MEMCPY_CPU_TO_GPU)
		cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	else if (kind == CCV_NNC_MEMCPY_GPU_TO_CPU)
		cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void gcfree(void* ptr)
{
	cudaFree(ptr);
}
