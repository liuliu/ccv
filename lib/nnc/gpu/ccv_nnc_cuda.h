/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_cuda_h
#define GUARD_ccv_nnc_cuda_h

#include "../ccv_nnc.h"

#ifndef HAVE_CUDA
#error "This file requires to be compiled with CUDA support."
#endif

// Simple counterparts of ccmalloc / ccfree.
void* gcmalloc(size_t size);
void gcfree(void* ptr);

// Real functions.
CCV_WARN_UNUSED(ccv_nnc_stream_unit_t*) ccv_nnc_cuda_stream_unit_new(void);
void ccv_nnc_cuda_stream_unit_free(ccv_nnc_stream_unit_t* stream_unit);

#endif
