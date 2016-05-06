/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_compat_h
#define GUARD_ccv_nnc_compat_h

#include "../ccv_nnc.h"

#ifndef HAVE_CUDA
#error "This file requires to be compiled with CUDA support."
#endif

// Simple counterparts of ccmalloc / ccfree.
void* cumalloc(int device, size_t size);
void cufree(int device, void* ptr);

// Stream context
CCV_WARN_UNUSED(ccv_nnc_stream_context_t*) ccv_nnc_init_stream_context(ccv_nnc_stream_context_t* stream_context);
void ccv_nnc_synchronize_stream_context(const ccv_nnc_stream_context_t* stream_context);
void ccv_nnc_deinit_stream_context(ccv_nnc_stream_context_t* stream_context);

#endif
