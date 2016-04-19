/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_cuda_h
#define GUARD_ccv_nnc_cuda_h

#include <stddef.h>

enum {
	CCV_NNC_MEMCPY_CPU_TO_GPU = 0x1,
	CCV_NNC_MEMCPY_GPU_TO_CPU = 0x2,
};

void* gcmalloc(size_t size);
void gcmemcpy(void* dst, const void* src, size_t size, int kind);
void gcfree(void* ptr);

#endif
