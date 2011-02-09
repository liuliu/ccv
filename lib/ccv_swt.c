#include "ccv.h"

void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert(a->type & CCV_C1);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_swt", 7, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
}

ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params)
{
	return 0;
}
