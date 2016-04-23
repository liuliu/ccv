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

typedef struct {
} ccv_nnc_cuda_stream_unit_t;

ccv_nnc_stream_unit_t* ccv_nnc_cuda_stream_unit_new(void)
{
	ccv_nnc_cuda_stream_unit_t* stream_unit = (ccv_nnc_cuda_stream_unit_t*)ccmalloc(sizeof(ccv_nnc_cuda_stream_unit_t));
	return stream_unit;
}

void ccv_nnc_cuda_stream_unit_free(ccv_nnc_stream_unit_t* stream_unit)
{
	ccfree(stream_unit);
}
