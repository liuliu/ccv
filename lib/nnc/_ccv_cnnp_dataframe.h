/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_cnnp_dataframe_internal_h
#define GUARD_ccv_cnnp_dataframe_internal_h

#include "ccv_nnc.h"

typedef struct {
	int size;
} ccv_cnnp_dataframe_tuple_t;

CCV_WARN_UNUSED(void*) ccv_cnnp_dataframe_column_context(const ccv_cnnp_dataframe_t* const dataframe, const int column_idx);

static inline char* ccv_cnnp_column_copy_name(const char* const name)
{
	if (name)
	{
		const size_t len = strnlen(name, 1023);
		const size_t n = len + 1;
		char* copy = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		memcpy(copy, name, n);
		copy[len] = 0;
		return copy;
	}
	return 0;
}

#endif
