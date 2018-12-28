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

#endif
