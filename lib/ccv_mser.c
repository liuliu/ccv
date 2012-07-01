#include "ccv.h"
#include "ccv_internal.h"

ccv_array_t* ccv_mser(ccv_dense_matrix_t* a, ccv_dense_matrix_t* h, ccv_dense_matrix_t** b, int type, ccv_mser_param_t params)
{
	ccv_declare_derived_signature_case(bsig, ccv_sign_with_format(64, "ccv_mser(matrix)"), ccv_sign_if(h == 0 && a->sig != 0, a->sig, 0), ccv_sign_if(h != 0 && a->sig != 0 && h->sig != 0, a->sig, h->sig, 0));
	ccv_declare_derived_signature_case(rsig, ccv_sign_with_format(64, "ccv_mser(array)"), ccv_sign_if(h == 0 && a->sig != 0, a->sig, 0), ccv_sign_if(h != 0 && a->sig != 0 && h->sig != 0, a->sig, h->sig, 0));
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_GET_CHANNEL(a->type) | CCV_ALL_DATA_TYPE, type, bsig);
	ccv_array_t* seq = ccv_array_new(sizeof(ccv_rect_t), 64, rsig);
	ccv_object_return_if_cached(seq, db, seq);
	ccv_revive_object_if_cached(db, seq);
	return seq;
}
