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

void gcfree(void* ptr)
{
	cudaFree(ptr);
}
