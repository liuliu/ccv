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
#include "3rdparty/khash/khash.h"
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#else
#include "3rdparty/sfmt/SFMT.h"
#endif

typedef struct {
	void (*deinit)(ccv_cnnp_dataframe_t* const self); /**< It can be nil. */
} ccv_cnnp_dataframe_vtab_t;

KHASH_MAP_INIT_INT64(ctx, ccv_array_t*)

struct ccv_cnnp_dataframe_s {
	ccv_cnnp_dataframe_vtab_t isa;
	int row_size;
	int column_size;
	int* shuffled_idx;
#ifdef HAVE_GSL
	gsl_rng* rng;
#else
	sfmt_t sfmt;
#endif
	khash_t(ctx)* data_ctx; // The stream context based cache for data entity of columns. This helps us to avoid allocations when iterate through data.
	ccv_array_t* derived_column_data;
	ccv_cnnp_column_data_t column_data[1];
};

typedef struct {
	int column_idx_size;
	int* column_idxs;
	void* context;
	ccv_cnnp_column_data_deinit_f deinit;
	ccv_cnnp_column_data_map_f map;
} ccv_cnnp_derived_column_data_t;

#endif
