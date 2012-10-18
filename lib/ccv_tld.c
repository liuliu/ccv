#include "ccv.h"
#include "ccv_internal.h"

#define TLD_GRID_SPARSITY (10)

static ccv_rect_t _ccv_tld_short_term_track(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_rect_t box, ccv_tld_param_t params)
{
	ccv_rect_t newbox = ccv_rect(0, 0, 0, 0);
	ccv_array_t* point_a = ccv_array_new(sizeof(ccv_decimal_point_t), (TLD_GRID_SPARSITY - 1) * (TLD_GRID_SPARSITY - 1), 0);
	float gapx = (float)box.width / TLD_GRID_SPARSITY;
	float gapy = (float)box.height / TLD_GRID_SPARSITY;
	for (float x = gapx * 0.5; x < box.width; x += gapx)
		for (float y = gapy * 0.5; y < box.height; y += gapy)
		{
			ccv_decimal_point_t point = ccv_decimal_point(box.x + x, box.y + y);
			ccv_array_push(point_a, &point);
		}
	ccv_array_t* point_b = 0;
	ccv_optical_flow_lucas_kanade(a, b, point_a, &point_b, params.win_size, params.level, params.min_eigen);
	ccv_array_t* point_c = 0;
	ccv_optical_flow_lucas_kanade(b, a, point_b, &point_c, params.win_size, params.level, params.min_eigen);
	// compute forward-backward error
	int i;
	for (i = 0; i < point_a->rnum; i++)
	{
		ccv_decimal_point_with_status_t* p1 = (ccv_decimal_point_with_status_t*)ccv_array_get(point_b, i);
		ccv_decimal_point_with_status_t* p2 = (ccv_decimal_point_with_status_t*)ccv_array_get(point_c, i);
		if (p1->status && p2->status)
		{
		}
	}
	ccv_array_free(point_c);
	ccv_array_free(point_b);
	ccv_array_free(point_a);
	return newbox;
}

ccv_tld_t* __attribute__((warn_unused_result)) ccv_tld_new(ccv_dense_matrix_t* a, ccv_rect_t box, ccv_tld_param_t params)
{
	ccv_tld_t* tld = (ccv_tld_t*)ccmalloc(sizeof(ccv_tld_t));
	tld->params = params;
	tld->box.rect = box;
	return tld;
}

// since there is no refcount syntax for ccv yet, we won't implicitly retain any matrix in ccv_tld_t
// instead, you should pass the previous frame and the current frame into the track function
ccv_comp_t ccv_tld_track_object(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	ccv_comp_t result;
	ccv_rect_t newbox = _ccv_tld_short_term_track(a, b, tld->box.rect, tld->params);
	return result;
}

void ccv_tld_free(ccv_tld_t* tld)
{
	ccfree(tld);
}
