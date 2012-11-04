#include "ccv.h"
#include "ccv_internal.h"
#include "3rdparty/sfmt/SFMT.h"

const ccv_tld_param_t ccv_tld_default_params = {
	.win_size = {
		15,
		15,
	},
	.level = 5,
	.min_forward_backward_error = 100,
	.min_eigen = 0.05,
	.min_win = 20,
	.interval = 3,
	.shift = 0.1,
	.top_n = 50,
	.include_overlap = 0.7,
	.exclude_overlap = 0.2,
	.structs = 30,
	.features = 13,
	.ncc_same = 0.95,
	.ncc_thres = 0.65,
	.ncc_collect = 0.5,
	.bad_patches = 100,
};

#define TLD_GRID_SPARSITY (10)
#define TLD_PATCH_SIZE (10)

static CCV_IMPLEMENT_MEDIAN(_ccv_tld_median, float)

static float _ccv_tld_norm_cross_correlate(ccv_dense_matrix_t* r0, ccv_dense_matrix_t* r1)
{
	assert(CCV_GET_CHANNEL(r0->type) == CCV_C1 && CCV_GET_DATA_TYPE(r0->type) == CCV_8U);
	assert(CCV_GET_CHANNEL(r1->type) == CCV_C1 && CCV_GET_DATA_TYPE(r1->type) == CCV_8U);
	assert(r0->rows == r1->rows && r0->cols == r1->cols);
	int x, y;
	int sum0 = 0, sum1 = 0;
	unsigned char* r0_ptr = r0->data.u8;
	unsigned char* r1_ptr = r1->data.u8;
	for (y = 0; y < r0->rows; y++)
	{
		for (x = 0; x < r0->cols; x++)
		{
			sum0 += r0_ptr[x];
			sum1 += r1_ptr[x];
		}
		r0_ptr += r0->step;
		r1_ptr += r1->step;
	}
	r0_ptr = r0->data.u8;
	r1_ptr = r1->data.u8;
	float mr0 = (float)sum0 / (r0->rows * r0->cols);
	float mr1 = (float)sum1 / (r1->rows * r1->cols);
	float r0r1 = 0, r0r0 = 0, r1r1 = 0;
	for (y = 0; y < r0->rows; y++)
	{
		for (x = 0; x < r0->cols; x++)
		{
			float r0f = r0_ptr[x] - mr0;
			float r1f = r1_ptr[x] - mr1;
			r0r1 += r0f * r1f;
			r0r0 += r0f * r0f;
			r1r1 += r1f * r1f;
		}
		r0_ptr += r0->step;
		r1_ptr += r1->step;
	}
	if (r0r0 * r1r1 < 1e-6)
		return 0;
	return r0r1 / sqrtf(r0r0 * r1r1);
}

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
	if (point_a->rnum <= 0)
	{
		ccv_array_free(point_a);
		return newbox;
	}
	ccv_array_t* point_b = 0;
	ccv_optical_flow_lucas_kanade(a, b, point_a, &point_b, params.win_size, params.level, params.min_eigen);
	if (point_b->rnum <= 0)
	{
		ccv_array_free(point_b);
		ccv_array_free(point_a);
		return newbox;
	}
	ccv_array_t* point_c = 0;
	ccv_optical_flow_lucas_kanade(b, a, point_b, &point_c, params.win_size, params.level, params.min_eigen);
	// compute forward-backward error
	ccv_dense_matrix_t* r0 = (ccv_dense_matrix_t*)alloca(ccv_compute_dense_matrix_size(TLD_PATCH_SIZE, TLD_PATCH_SIZE, CCV_8U | CCV_C1));
	ccv_dense_matrix_t* r1 = (ccv_dense_matrix_t*)alloca(ccv_compute_dense_matrix_size(TLD_PATCH_SIZE, TLD_PATCH_SIZE, CCV_8U | CCV_C1));
	r0 = ccv_dense_matrix_new(TLD_PATCH_SIZE, TLD_PATCH_SIZE, CCV_8U | CCV_C1, r0, 0);
	r1 = ccv_dense_matrix_new(TLD_PATCH_SIZE, TLD_PATCH_SIZE, CCV_8U | CCV_C1, r1, 0);
	int i, j, k, size;
	int* wrt = (int*)alloca(sizeof(int) * point_a->rnum);
	{ // will reclaim the stack
	float* fberr = (float*)alloca(sizeof(float) * point_a->rnum);
	float* sim = (float*)alloca(sizeof(float) * point_a->rnum);
	for (i = 0, k = 0; i < point_a->rnum; i++)
	{
		ccv_decimal_point_t* p0 = (ccv_decimal_point_t*)ccv_array_get(point_a, i);
		ccv_decimal_point_with_status_t* p1 = (ccv_decimal_point_with_status_t*)ccv_array_get(point_b, i);
		ccv_decimal_point_with_status_t* p2 = (ccv_decimal_point_with_status_t*)ccv_array_get(point_c, i);
		if (p1->status && p2->status &&
			p1->point.x >= 0 && p1->point.x < a->cols && p1->point.y >= 0 && p1->point.y < a->rows &&
			p2->point.x >= 0 && p2->point.x < a->cols && p2->point.y >= 0 && p2->point.y < a->rows)
		{
			fberr[k] = (p2->point.x - p0->x) * (p2->point.x - p0->x) + (p2->point.y - p0->y) * (p2->point.y - p0->y);
			ccv_decimal_slice(a, &r0, 0, p0->y - (TLD_PATCH_SIZE - 1) * 0.5, p0->x - (TLD_PATCH_SIZE - 1) * 0.5, TLD_PATCH_SIZE, TLD_PATCH_SIZE);
			ccv_decimal_slice(a, &r1, 0, p1->point.y - (TLD_PATCH_SIZE - 1) * 0.5, p1->point.x - (TLD_PATCH_SIZE - 1) * 0.5, TLD_PATCH_SIZE, TLD_PATCH_SIZE);
			sim[k] = _ccv_tld_norm_cross_correlate(r0, r1);
			wrt[k] = i;
			++k;
		}
	}
	ccv_array_free(point_c);
	if (k == 0)
	{
		// early termination because we don't have qualified tracking points
		ccv_array_free(point_b);
		ccv_array_free(point_a);
		return newbox;
	}
	size = k;
	float simmd = _ccv_tld_median(sim, 0, size - 1);
	for (i = 0, k = 0; i < size; i++)
		if (sim[i] > simmd)
		{
			fberr[k] = fberr[i];
			wrt[k] = wrt[i];
			++k;
		}
	size = k;
	float fberrmd = _ccv_tld_median(fberr, 0, size - 1);
	if (fberrmd >= params.min_forward_backward_error)
	{
		// early termination because we don't have qualified tracking points
		ccv_array_free(point_b);
		ccv_array_free(point_a);
		return newbox;
	}
	size = k;
	for (i = 0, k = 0; i < size; i++)
		if (fberr[i] <= fberrmd)
			wrt[k++] = wrt[i];
	size = k;
	if (k == 0)
	{
		// early termination because we don't have qualified tracking points
		ccv_array_free(point_b);
		ccv_array_free(point_a);
		return newbox;
	}
	} // reclaim stack
	float dx, dy;
	{ // will reclaim the stack
	float* offx = (float*)alloca(sizeof(float) * size);
	float* offy = (float*)alloca(sizeof(float) * size);
	for (i = 0; i < size; i++)
	{
		ccv_decimal_point_t* p0 = (ccv_decimal_point_t*)ccv_array_get(point_a, wrt[i]);
		ccv_decimal_point_t* p1 = (ccv_decimal_point_t*)ccv_array_get(point_b, wrt[i]);
		offx[i] = p1->x - p0->x;
		offy[i] = p1->y - p0->y;
	}
	dx = _ccv_tld_median(offx, 0, size - 1);
	dy = _ccv_tld_median(offy, 0, size - 1);
	} // reclaim stack
	if (size > 1)
	{
		float* s = (float*)alloca(sizeof(float) * size * (size - 1) / 2);
		k = 0;
		for (i = 0; i < size - 1; i++)
		{
			ccv_decimal_point_t* p0i = (ccv_decimal_point_t*)ccv_array_get(point_a, wrt[i]);
			ccv_decimal_point_t* p1i = (ccv_decimal_point_t*)ccv_array_get(point_b, wrt[i]);
			for (j = i + 1; j < size; j++)
			{
				ccv_decimal_point_t* p0j = (ccv_decimal_point_t*)ccv_array_get(point_a, wrt[j]);
				ccv_decimal_point_t* p1j = (ccv_decimal_point_t*)ccv_array_get(point_b, wrt[j]);
				s[k] = sqrtf(((p1i->x - p1j->x) * (p1i->x - p1j->x) + (p1i->y - p1j->y) * (p1i->y - p1j->y)) /
							 ((p0i->x - p0j->x) * (p0i->x - p0j->x) + (p0i->y - p0j->y) * (p0i->y - p0j->y)));
				++k;
			}
		}
		assert(size * (size - 1) / 2 == k);
		float ds = _ccv_tld_median(s, 0, size * (size - 1) / 2 - 1);
		newbox.x = (int)(box.x + dx - box.width * (ds - 1.0) * 0.5 + 0.5);
		newbox.y = (int)(box.y + dy - box.height * (ds - 1.0) * 0.5 + 0.5);
		newbox.width = (int)(box.width * ds + 0.5);
		newbox.height = (int)(box.height * ds + 0.5);
	} else {
		newbox.width = box.width;
		newbox.height = box.height;
		newbox.x = (int)(box.x + dx + 0.5);
		newbox.y = (int)(box.y + dy + 0.5);
	}
	ccv_array_free(point_b);
	ccv_array_free(point_a);
	return newbox;
}

static inline float _ccv_tld_rect_intersect(const ccv_rect_t r1, const ccv_rect_t r2)
{
	int intersect = ccv_max(0, ccv_min(r1.x + r1.width, r2.x + r2.width) - ccv_max(r1.x, r2.x)) * ccv_max(0, ccv_min(r1.y + r1.height, r2.y + r2.height) - ccv_max(r1.y, r2.y));
	return (float)intersect / (r1.width * r1.height + r2.width * r2.height - intersect);
}

#define for_each_size(new_width, new_height, box_width, box_height, interval, min_win, image_width, image_height) \
	{ \
		double INTERNAL_CATCH_UNIQUE_NAME(scale) = pow(2.0, 1.0 / (interval + 1.0)); \
		int INTERNAL_CATCH_UNIQUE_NAME(scale_upto) = (int)(log((double)ccv_min((double)image_width / box_width, (double)image_height / box_height)) / log(INTERNAL_CATCH_UNIQUE_NAME(scale))); \
		int INTERNAL_CATCH_UNIQUE_NAME(s); \
		double INTERNAL_CATCH_UNIQUE_NAME(ss) = 1.0; \
		for (INTERNAL_CATCH_UNIQUE_NAME(s) = 0; INTERNAL_CATCH_UNIQUE_NAME(s) < INTERNAL_CATCH_UNIQUE_NAME(scale_upto); INTERNAL_CATCH_UNIQUE_NAME(s)++) \
		{ \
			int new_width = (int)(box_width * INTERNAL_CATCH_UNIQUE_NAME(ss) + 0.5); \
			int new_height = (int)(box_height * INTERNAL_CATCH_UNIQUE_NAME(ss) + 0.5); \
			int INTERNAL_CATCH_UNIQUE_NAME(min_side) = ccv_min(new_width, new_height); \
			INTERNAL_CATCH_UNIQUE_NAME(ss) *= INTERNAL_CATCH_UNIQUE_NAME(scale); \
			if (INTERNAL_CATCH_UNIQUE_NAME(min_side) < min_win || new_width > image_width || new_height > image_height) \
				continue;
#define end_for_each_size } }

#define for_each_box(new_comp, box_width, box_height, interval, shift, min_win, image_width, image_height) \
	{ \
		int INTERNAL_CATCH_UNIQUE_NAME(is) = 0; \
		for_each_size(INTERNAL_CATCH_UNIQUE_NAME(width), INTERNAL_CATCH_UNIQUE_NAME(height), box_width, box_height, interval, min_win, image_width, image_height) \
			++INTERNAL_CATCH_UNIQUE_NAME(is); \
			float INTERNAL_CATCH_UNIQUE_NAME(x), INTERNAL_CATCH_UNIQUE_NAME(y); \
			int INTERNAL_CATCH_UNIQUE_NAME(piy) = -1; \
			for (INTERNAL_CATCH_UNIQUE_NAME(y) = 0; \
				 INTERNAL_CATCH_UNIQUE_NAME(y) < image_height - INTERNAL_CATCH_UNIQUE_NAME(height); \
				 INTERNAL_CATCH_UNIQUE_NAME(y) += shift * INTERNAL_CATCH_UNIQUE_NAME(min_side)) /* min_side is exposed by for_each_size, and because the ubiquity of this macro, its name gets leaked */ \
			{ \
				int INTERNAL_CATCH_UNIQUE_NAME(iy) = (int)(INTERNAL_CATCH_UNIQUE_NAME(y) + 0.5); \
				if (INTERNAL_CATCH_UNIQUE_NAME(iy) == INTERNAL_CATCH_UNIQUE_NAME(piy)) \
					continue; \
				INTERNAL_CATCH_UNIQUE_NAME(piy) = INTERNAL_CATCH_UNIQUE_NAME(iy); \
				int INTERNAL_CATCH_UNIQUE_NAME(pix) = -1; \
				for (INTERNAL_CATCH_UNIQUE_NAME(x) = 0; \
					 INTERNAL_CATCH_UNIQUE_NAME(x) < image_width - INTERNAL_CATCH_UNIQUE_NAME(width); \
					 INTERNAL_CATCH_UNIQUE_NAME(x) += shift * INTERNAL_CATCH_UNIQUE_NAME(min_side)) \
				{ \
					int INTERNAL_CATCH_UNIQUE_NAME(ix) = (int)(INTERNAL_CATCH_UNIQUE_NAME(x) + 0.5); \
					if (INTERNAL_CATCH_UNIQUE_NAME(ix) == INTERNAL_CATCH_UNIQUE_NAME(pix)) \
						continue; \
					INTERNAL_CATCH_UNIQUE_NAME(pix) = INTERNAL_CATCH_UNIQUE_NAME(ix); \
					ccv_comp_t new_comp; \
					new_comp.rect = ccv_rect(INTERNAL_CATCH_UNIQUE_NAME(ix), INTERNAL_CATCH_UNIQUE_NAME(iy), INTERNAL_CATCH_UNIQUE_NAME(width), INTERNAL_CATCH_UNIQUE_NAME(height)); \
					new_comp.id = INTERNAL_CATCH_UNIQUE_NAME(is) - 1;
#define end_for_each_box } } end_for_each_size }

static void _ccv_tld_box_percolate_down(ccv_array_t* good, int i)
{
	for (;;)
	{
		int left = 2 * (i + 1) - 1;
		int right = 2 * (i + 1);
		int smallest = i;
		ccv_comp_t* smallest_comp = (ccv_comp_t*)ccv_array_get(good, smallest);
		if (left < good->rnum)
		{
			ccv_comp_t* left_comp = (ccv_comp_t*)ccv_array_get(good, left);
			if (left_comp->confidence < smallest_comp->confidence)
			{
				smallest = left;
				smallest_comp = left_comp;
			}
		}
		if (right < good->rnum)
		{
			ccv_comp_t* right_comp = (ccv_comp_t*)ccv_array_get(good, right);
			if (right_comp->confidence < smallest_comp->confidence)
			{
				smallest = right;
				smallest_comp = right_comp;
			}
		}
		if (smallest == i)
			break;
		ccv_comp_t c = *(ccv_comp_t*)ccv_array_get(good, smallest);
		*(ccv_comp_t*)ccv_array_get(good, smallest) = *(ccv_comp_t*)ccv_array_get(good, i);
		*(ccv_comp_t*)ccv_array_get(good, i) = c;
		i = smallest;
	}
}

static void _ccv_tld_box_percolate_up(ccv_array_t* good, int smallest)
{
	for (;;)
	{
		int one = smallest;
		int parent = (smallest + 1) / 2 - 1;
		if (parent < 0)
			break;
		ccv_comp_t* parent_comp = (ccv_comp_t*)ccv_array_get(good, parent);
		ccv_comp_t* smallest_comp = (ccv_comp_t*)ccv_array_get(good, smallest);
		if (smallest_comp->confidence < parent_comp->confidence)
		{
			smallest = parent;
			smallest_comp = parent_comp;
		}
		// if current one is no smaller than the parent one, stop
		if (smallest == one)
			break;
		ccv_comp_t c = *(ccv_comp_t*)ccv_array_get(good, smallest);
		*(ccv_comp_t*)ccv_array_get(good, smallest) = *(ccv_comp_t*)ccv_array_get(good, one);
		*(ccv_comp_t*)ccv_array_get(good, one) = c;
		int other = 2 * (parent + 1) - !(one & 1);
		if (other < good->rnum)
		{
			ccv_comp_t* other_comp = (ccv_comp_t*)ccv_array_get(good, other);
			// if current one is no smaller than the other one, stop, and this requires a percolating down
			if (other_comp->confidence < smallest_comp->confidence)
				break;
		}
	}
	// have to percolating down to avoid it is bigger than the other side of the sub-tree
	_ccv_tld_box_percolate_down(good, smallest);
}

static ccv_comp_t _ccv_tld_generate_box_for(ccv_size_t image_size, ccv_size_t input_size, ccv_rect_t box, int gcap, ccv_array_t** good, ccv_array_t** bad, ccv_tld_param_t params)
{
	assert(gcap > 0);
	ccv_array_t* agood = *good = ccv_array_new(sizeof(ccv_comp_t), gcap, 0);
	ccv_array_t* abad = *bad = ccv_array_new(sizeof(ccv_comp_t), 64, 0);
	double max_overlap = -DBL_MAX;
	ccv_comp_t best_box = {
		.id = 0,
		.rect = ccv_rect(0, 0, 0, 0),
	};
	int i = 0;
	for_each_box(comp, input_size.width, input_size.height, params.interval, params.shift, params.min_win, image_size.width, image_size.height)
		double overlap = _ccv_tld_rect_intersect(comp.rect, box);
		comp.neighbors = i++;
		comp.confidence = overlap;
		if (overlap > params.include_overlap)
		{
			if (overlap > max_overlap)
			{
				max_overlap = overlap;
				best_box = comp;
			}
			if (agood->rnum < gcap)
			{
				ccv_array_push(agood, &comp);
				_ccv_tld_box_percolate_up(agood, agood->rnum - 1);
			} else {
				ccv_comp_t* p = (ccv_comp_t*)ccv_array_get(agood, 0);
				if (overlap > p->confidence)
				{
					*(ccv_comp_t*)ccv_array_get(agood, 0) = comp;
					_ccv_tld_box_percolate_down(agood, 0);
				}
			}
		} else if (overlap < params.exclude_overlap)
			ccv_array_push(abad, &comp);
	end_for_each_box;
	best_box.neighbors = i;
	return best_box;
}

static void _ccv_tld_ferns_feature_for(ccv_ferns_t* ferns, ccv_dense_matrix_t* a, ccv_comp_t box, uint32_t* fern)
{
	ccv_dense_matrix_t roi = ccv_dense_matrix(box.rect.height, box.rect.width, CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type), ccv_get_dense_matrix_cell(a, box.rect.y, box.rect.x, 0), 0);
	roi.step = a->step;
	ccv_ferns_feature(ferns, &roi, box.id, fern);
}

static void _ccv_tld_fetch_patch(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_rect_t box)
{
	ccv_dense_matrix_t* c = 0;
	if (box.width == tld->patch.width && box.height == tld->patch.height)
		ccv_slice(a, (ccv_matrix_t**)b, type, box.y, box.x, box.height, box.width);
	else {
		assert((box.width >= tld->patch.width && box.height >= tld->patch.height) ||
			   (box.width <= tld->patch.width && box.height <= tld->patch.height));
		ccv_slice(a, (ccv_matrix_t**)&c, type, box.y, box.x, box.height, box.width);
		ccv_resample(c, b, type, tld->patch.height, tld->patch.width, CCV_INTER_AREA | CCV_INTER_CUBIC);
		ccv_matrix_free(c);
	}
}

static float _ccv_tld_sv_classify(ccv_tld_t* tld, ccv_dense_matrix_t* a, int pnum, int nnum, int* anyp, int* anyn)
{
	assert(a->rows == tld->patch.height && a->cols == tld->patch.width);
	int i;
	pnum = (pnum <= 0) ? tld->sv[1]->rnum : ccv_min(pnum, tld->sv[1]->rnum);
	if (pnum == 0)
		return 0;
	nnum = (nnum <= 0) ? tld->sv[0]->rnum : ccv_min(nnum, tld->sv[0]->rnum);
	if (nnum == 0)
		return 1;
	float maxp = -1;
	for (i = 0; i < pnum; i++)
	{
		ccv_dense_matrix_t* b = *(ccv_dense_matrix_t**)ccv_array_get(tld->sv[1], i);
		float ncc = _ccv_tld_norm_cross_correlate(a, b);
		if (ncc > maxp)
			maxp = ncc;
	}
	maxp = (maxp + 1) * 0.5; // make it in 0~1 range
	if (anyp)
		*anyp = (maxp > tld->params.ncc_same);
	float maxn = -1;
	for (i = 0; i < nnum; i++)
	{
		ccv_dense_matrix_t* b = *(ccv_dense_matrix_t**)ccv_array_get(tld->sv[0], i);
		float ncc = _ccv_tld_norm_cross_correlate(a, b);
		if (ncc > maxn)
			maxn = ncc;
	}
	maxn = (maxn + 1) * 0.5; // make it in 0~1 range
	if (anyn)
		*anyn = (maxn > tld->params.ncc_same);
	return (1 - maxn) / (2 - maxn - maxp);
}

// return 0 means that we will retain the given example (thus, you don't want to free it)
static int _ccv_tld_sv_correct(ccv_tld_t* tld, ccv_dense_matrix_t* a, int y)
{
	int anyp, anyn;
	if (y == 1 && tld->sv[1]->rnum == 0)
	{
		ccv_array_push(tld->sv[1], &a);
		return 0;
	}
	float conf = _ccv_tld_sv_classify(tld, a, 0, 0, &anyp, &anyn);
	if (y == 1 && conf < tld->params.ncc_thres)
	{
		ccv_array_push(tld->sv[1], &a);
		return 0;
	} else if (y == 0 && conf > tld->params.ncc_collect) {
		ccv_array_push(tld->sv[0], &a);
		return 0;
	}
	return -1;
}

#define PATCH_SIZE (20)

static void _ccv_tld_check_params(ccv_tld_param_t params)
{
	assert(params.top_n > 0);
	assert(params.structs > 0);
	assert(params.features > 0 && params.features <= 32);
	assert(params.win_size.width > 0 && params.win_size.height > 0);
	assert((params.win_size.width & 1) == 1 && (params.win_size.height & 1) == 1);
	assert(params.level >= 0);
	assert(params.min_eigen > 0);
	assert(params.min_forward_backward_error > 0);
	assert(params.bad_patches > 0);
	assert(params.interval >= 0);
	assert(params.shift > 0 && params.shift < 1);
}

static float _ccv_tld_ferns_compute_threshold(ccv_ferns_t* ferns, float ferns_thres, ccv_dense_matrix_t* ga, ccv_array_t* bad)
{
	int i;
	uint32_t* fern = (uint32_t*)alloca(sizeof(uint32_t) * ferns->structs);
	for (i = 0; i < bad->rnum; i++)
	{
		ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(bad, i);
		_ccv_tld_ferns_feature_for(ferns, ga, *box, fern);
		float c = ccv_ferns_predict(ferns, fern);
		if (c > ferns_thres)
			ferns_thres = c;
	}
	return ferns_thres;
}

ccv_tld_t* ccv_tld_new(ccv_dense_matrix_t* a, ccv_rect_t box, ccv_tld_param_t params)
{
	_ccv_tld_check_params(params);
	ccv_size_t patch = ccv_size((int)(sqrtf(params.min_win * params.min_win * (float)box.width / box.height) + 0.5),
								(int)(sqrtf(params.min_win * params.min_win * (float)box.height / box.width) + 0.5));
	ccv_array_t* good = 0;
	ccv_array_t* bad = 0;
	ccv_comp_t best_box = _ccv_tld_generate_box_for(ccv_size(a->cols, a->rows), patch, box, 20, &good, &bad, params);
	ccv_tld_t* tld = (ccv_tld_t*)ccmalloc(sizeof(ccv_tld_t) + sizeof(uint32_t) * params.structs * best_box.neighbors);
	tld->patch = patch;
	tld->params = params;
	tld->ncc_thres = params.ncc_thres;
	tld->box.rect = box;
	{
	double scale = pow(2.0, 1.0 / (params.interval + 1.0));
	int scale_upto = (int)(log((double)ccv_min((double)a->cols / patch.width, (double)a->rows / patch.height)) / log(scale));
	ccv_size_t* scales = (ccv_size_t*)alloca(sizeof(ccv_size_t) * scale_upto);
	int is = 0;
	for_each_size(width, height, patch.width, patch.height, params.interval, params.min_win, a->cols, a->rows)
		scales[is] = ccv_size(width, height);
		++is;
	end_for_each_size;
	tld->ferns = ccv_ferns_new(params.structs, params.features, is, scales);
	}
	tld->sv[0] = ccv_array_new(sizeof(ccv_dense_matrix_t*), 64, 0);
	tld->sv[1] = ccv_array_new(sizeof(ccv_dense_matrix_t*), 64, 0);
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, (uint32_t)a);
	sfmt_genrand_shuffle(&sfmt, ccv_array_get(bad, 0), bad->rnum, bad->rsize);
	int badex = (bad->rnum + 1) / 2;
	int* idx = (int*)ccmalloc(sizeof(int) * (badex + good->rnum));
	int i, j, k;
	for (i = 0; i < badex + good->rnum; i++)
		idx[i] = i;
	sfmt_genrand_shuffle(&sfmt, idx, badex + good->rnum, sizeof(int));
	// train the fern classifier
	uint32_t* fern = (uint32_t*)ccmalloc(sizeof(uint32_t) * tld->ferns->structs * (badex + good->rnum));
	ccv_dense_matrix_t* ga = 0;
	ccv_blur(a, &ga, 0, 1.5);
	uint32_t* pfern = fern;
	for (j = 0; j < badex + good->rnum; j++)
	{
		k = idx[j];
		if (k < badex)
		{
			ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(bad, k);
			_ccv_tld_ferns_feature_for(tld->ferns, ga, *box, pfern);
		} else {
			k -= badex;
			ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(good, k);
			_ccv_tld_ferns_feature_for(tld->ferns, ga, *box, pfern);
		}
		pfern += tld->ferns->structs;
	}
	for (i = 0; i < 10; i++)
	{
		pfern = fern;
		for (j = 0; j < badex + good->rnum; j++)
		{
			k = idx[j];
			if (k < badex)
			{
				// fix the thresholding for negative
				if (ccv_ferns_predict(tld->ferns, pfern) >= tld->ferns->threshold)
					ccv_ferns_correct(tld->ferns, pfern, 0, 2);
			} else {
				k -= badex;
				// fix the thresholding for positive
				if (ccv_ferns_predict(tld->ferns, pfern) <= tld->ferns->threshold)
					ccv_ferns_correct(tld->ferns, pfern, 1, 2);
			}
			pfern += tld->ferns->structs;
		}
	}
	tld->ferns_thres = _ccv_tld_ferns_compute_threshold(tld->ferns, tld->ferns->threshold, ga, bad);
	ccv_matrix_free(ga);
	ccfree(fern);
	ccv_array_free(good);
	ccfree(idx);
	// train the nearest-neighbor classifier
	ccv_dense_matrix_t* b = 0;
	_ccv_tld_fetch_patch(tld, a, &b, 0, best_box.rect);
	ccv_array_push(tld->sv[1], &b);
	for (i = 0; i < ccv_min(tld->params.bad_patches, bad->rnum); i++)
	{
		ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(bad, i);
		ccv_dense_matrix_t* b = 0;
		_ccv_tld_fetch_patch(tld, a, &b, 0, box->rect);
		if (_ccv_tld_sv_correct(tld, b, 0) != 0)
			ccv_matrix_free(b);
	}
	ccv_array_free(bad);
	// init tld params
	tld->found = 1; // assume last time has found (we just started)
	// top is ccv_tld_feature_t, and its continuous memory region for a feature
	tld->top = ccv_array_new(sizeof(ccv_comp_t), params.top_n, 0);
	tld->top->rnum = 0;
	return tld;
}

static void _ccv_tld_quick_learn(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_comp_t dd)
{
	ccv_dense_matrix_t* b = 0;
	float scale = sqrtf((float)(dd.rect.width * dd.rect.height) / (tld->patch.width * tld->patch.height));
	// regularize the rect to conform patch's aspect ratio
	dd.rect = ccv_rect((int)(dd.rect.x + (dd.rect.width - tld->patch.width * scale) + 0.5),
					   (int)(dd.rect.y + (dd.rect.height - tld->patch.height * scale) + 0.5),
					   (int)(tld->patch.width * scale + 0.5),
					   (int)(tld->patch.height * scale + 0.5));
	_ccv_tld_fetch_patch(tld, a, &b, 0, dd.rect);
	int anyp, anyn;
	float c = _ccv_tld_sv_classify(tld, b, 0, 0, &anyp, &anyn);
	if (c > tld->ncc_thres && !anyn)
	{
		ccv_array_t* good = 0;
		ccv_array_t* bad = 0;
		ccv_comp_t best_box = _ccv_tld_generate_box_for(ccv_size(a->cols, a->rows), tld->patch, dd.rect, 10, &good, &bad, tld->params);
		sfmt_t sfmt;
		sfmt_init_gen_rand(&sfmt, (uint32_t)a);
		sfmt_genrand_shuffle(&sfmt, ccv_array_get(bad, 0), bad->rnum, bad->rsize);
		int* idx = (int*)ccmalloc(sizeof(int) * (bad->rnum + good->rnum));
		int i, j, k;
		for (i = 0; i < bad->rnum + good->rnum; i++)
			idx[i] = i;
		sfmt_genrand_shuffle(&sfmt, idx, bad->rnum + good->rnum, sizeof(int));
		uint32_t* fern = (uint32_t*)ccmalloc(sizeof(uint32_t) * tld->ferns->structs * (bad->rnum + good->rnum));
		uint32_t* pfern =fern;
		for (j = 0; j < bad->rnum + good->rnum; j++)
		{
			k = idx[j];
			if (k < bad->rnum)
			{
				// fix the thresholding for negative
				ccv_comp_t *box = (ccv_comp_t*)ccv_array_get(bad, k);
				assert(box->neighbors >= 0 && box->neighbors < best_box.neighbors);
				memcpy(pfern, tld->fern_buffer + box->neighbors * tld->ferns->structs, sizeof(uint32_t) * tld->ferns->structs);
			} else {
				ccv_comp_t *box = (ccv_comp_t*)ccv_array_get(good, k - bad->rnum);
				assert(box->neighbors >= 0 && box->neighbors < best_box.neighbors);
				memcpy(pfern, tld->fern_buffer + box->neighbors * tld->ferns->structs, sizeof(uint32_t) * tld->ferns->structs);
			}
			pfern += tld->ferns->structs;
		}
		// train the fern classifier
		for (i = 0; i < 10; i++)
		{
			pfern = fern;
			for (j = 0; j < bad->rnum + good->rnum; j++)
			{
				k = idx[j];
				if (k < bad->rnum)
				{
					// fix the thresholding for negative
					if (ccv_ferns_predict(tld->ferns, pfern) >= tld->ferns_thres)
						ccv_ferns_correct(tld->ferns, pfern, 0, 1);
				} else {
					// fix the thresholding for positive
					if (ccv_ferns_predict(tld->ferns, pfern) <= tld->ferns_thres)
						ccv_ferns_correct(tld->ferns, pfern, 1, 1);
				}
				pfern += tld->ferns->structs;
			}
		}
		ccfree(fern);
		ccv_array_free(good);
		ccfree(idx);
		// train the nearest-neighbor classifier
		ccv_dense_matrix_t* b = 0;
		_ccv_tld_fetch_patch(tld, a, &b, 0, best_box.rect);
		if (_ccv_tld_sv_correct(tld, b, 1) != 0)
			ccv_matrix_free(b);
		for (i = 0; i < ccv_min(tld->params.bad_patches, bad->rnum); i++)
		{
			ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(bad, i);
			ccv_dense_matrix_t* b = 0;
			_ccv_tld_fetch_patch(tld, a, &b, 0, box->rect);
			if (_ccv_tld_sv_correct(tld, b, 0) != 0)
				ccv_matrix_free(b);
		}
		ccv_array_free(bad);
	}
	ccv_matrix_free(b);
}

static ccv_array_t* _ccv_tld_long_term_detect(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_tld_info_t* info)
{
	int i;
	tld->top->rnum = 0;
	uint32_t* fern = tld->fern_buffer;
	ccv_dense_matrix_t* ga = 0;
	ccv_blur(a, &ga, 0, 1.5);
	for_each_box(box, tld->patch.width, tld->patch.height, tld->params.interval, tld->params.shift, tld->params.min_win, a->cols, a->rows)
		_ccv_tld_ferns_feature_for(tld->ferns, ga, box, fern);
		box.confidence = ccv_ferns_predict(tld->ferns, fern);
		if (box.confidence > tld->ferns_thres)
		{
			ccv_comp_t* top_box = (ccv_comp_t*)ccv_array_get(tld->top, 0);
			if (tld->top->rnum < tld->params.top_n)
			{
				ccv_array_push(tld->top, &box);
				_ccv_tld_box_percolate_up(tld->top, tld->top->rnum - 1);
			} else if (top_box->confidence < box.confidence) {
				*(ccv_comp_t*)ccv_array_get(tld->top, 0) = box;
				_ccv_tld_box_percolate_down(tld->top, 0);
			}
		}
		fern += tld->ferns->structs;
	end_for_each_box;
	ccv_matrix_free(ga);
	if (info)
		info->fern_detects = tld->top->rnum;
	ccv_array_t* seq = ccv_array_new(sizeof(ccv_comp_t), tld->top->rnum, 0);
	for (i = 0; i < tld->top->rnum; i++)
	{
		ccv_comp_t* box = (ccv_comp_t*)ccv_array_get(tld->top, i);
		int anyp = 0, anyn = 0;
		ccv_dense_matrix_t* b = 0;
		_ccv_tld_fetch_patch(tld, a, &b, 0, box->rect);
		float c = _ccv_tld_sv_classify(tld, b, 0, 0, &anyp, &anyn);
		ccv_matrix_free(b);
		if (c > tld->ncc_thres)
		{
			box->confidence = c;
			ccv_array_push(seq, box);
		}
	}
	return seq;
}

static int _ccv_is_equal(const void* _r1, const void* _r2, void* data)
{
	const ccv_comp_t* r1 = (const ccv_comp_t*)_r1;
	const ccv_comp_t* r2 = (const ccv_comp_t*)_r2;
	return _ccv_tld_rect_intersect(r1->rect, r2->rect) > 0.5;
}

// since there is no refcount syntax for ccv yet, we won't implicitly retain any matrix in ccv_tld_t
// instead, you should pass the previous frame and the current frame into the track function
ccv_comp_t ccv_tld_track_object(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_tld_info_t* info)
{
	ccv_comp_t result;
	int tracked = 0;
	int verified = 0;
	if (info)
		info->perform_track = tld->found;
	if (tld->found)
	{
		result.rect = _ccv_tld_short_term_track(a, b, tld->box.rect, tld->params);
		if (!ccv_rect_is_zero(result.rect))
		{
			tracked = 1;
			int anyp = 0, anyn = 0;
			ccv_dense_matrix_t* c = 0;
			_ccv_tld_fetch_patch(tld, b, &c, 0, result.rect);
			result.confidence = _ccv_tld_sv_classify(tld, c, 0, 0, &anyp, &anyn);
			ccv_matrix_free(c);
			if (result.confidence > tld->ncc_thres)
				verified = 1;
		}
	}
	if (info)
		info->track_success = tracked;
	ccv_array_t* dd = _ccv_tld_long_term_detect(tld, b, info);
	if (info)
		info->ncc_detects = dd->rnum;
	int i;
	// cluster detected result
	if (dd->rnum > 1)
	{
		ccv_array_t* idx_dd = 0;
		// group retrieved rectangles in order to filter out noise
		int ncomp = ccv_array_group(dd, &idx_dd, _ccv_is_equal, 0);
		ccv_comp_t* comps = (ccv_comp_t*)ccmalloc(ncomp * sizeof(ccv_comp_t));
		memset(comps, 0, ncomp * sizeof(ccv_comp_t));
		for (i = 0; i < dd->rnum; i++)
		{
			ccv_comp_t r1 = *(ccv_comp_t*)ccv_array_get(dd, i);
			int idx = *(int*)ccv_array_get(idx_dd, i);
			++comps[idx].neighbors;
			comps[idx].rect.x += r1.rect.x;
			comps[idx].rect.y += r1.rect.y;
			comps[idx].rect.width += r1.rect.width;
			comps[idx].rect.height += r1.rect.height;
			comps[idx].confidence += r1.confidence;
		}
		ccv_array_clear(dd);
		for(i = 0; i < ncomp; i++)
		{
			int n = comps[i].neighbors;
			ccv_comp_t comp;
			comp.rect.x = (comps[i].rect.x * 2 + n) / (2 * n);
			comp.rect.y = (comps[i].rect.y * 2 + n) / (2 * n);
			comp.rect.width = (comps[i].rect.width * 2 + n) / (2 * n);
			comp.rect.height = (comps[i].rect.height * 2 + n) / (2 * n);
			comp.neighbors = comps[i].neighbors;
			comp.confidence = comps[i].confidence / n;
			ccv_array_push(dd, &comp);
		}
		ccv_array_free(idx_dd);
		ccfree(comps);
	}
	if (info)
	{
		info->clustered_detects = dd->rnum;
		info->confident_matches = info->close_matches = 0;
	}
	if (tracked)
	{
		if (dd->rnum > 0)
		{
			ccv_comp_t* ddcomp = 0;
			int confident_matches = 0;
			for (i = 0; i < dd->rnum; i++)
			{
				ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(dd, i);
				if (_ccv_tld_rect_intersect(result.rect, comp->rect) < 0.5 && comp->confidence > result.confidence)
				{
					++confident_matches;
					ddcomp = comp;
				}
			}
			if (info)
				info->confident_matches = confident_matches;
			if (confident_matches == 1)
			{
				// only one match, reinitialize tracking
				result = *ddcomp;
				// but the result is not a valid tracking
				verified = 0;
			} else {
				// too much confident matches, we will focus on close matches instead
				int close_matches = 0;
				ccv_rect_t ddc = ccv_rect(0, 0, 0, 0);
				for (i = 0; i < dd->rnum; i++)
				{
					ccv_comp_t* comp = (ccv_comp_t*)ccv_array_get(dd, i);
					if (_ccv_tld_rect_intersect(result.rect, comp->rect) > 0.7)
					{
						ddc.y += comp->rect.y;
						ddc.x += comp->rect.x;
						ddc.height += comp->rect.height;
						ddc.width += comp->rect.width;
						++close_matches;
					}
				}
				if (info)
					info->close_matches = close_matches;
				if (close_matches > 0)
				{
					// reweight the tracking result
					result.rect.x = (20 * result.rect.x + ddc.x * 2 + close_matches + 10) / (20 + 2 * close_matches);
					result.rect.y = (20 * result.rect.y + ddc.y * 2 + close_matches + 10) / (20 + 2 * close_matches);
					result.rect.width = (20 * result.rect.width + ddc.width * 2 + close_matches + 10) / (20 + 2 * close_matches);
					result.rect.height = (20 * result.rect.height + ddc.height * 2 + close_matches + 10) / (20 + 2 * close_matches);
				}
			}
		}
	} else if (dd->rnum == 1) {
		// only reinitialize tracker when detection result is exactly one
		result = *(ccv_comp_t*)ccv_array_get(dd, 0);
		tld->found = 1;
	} else {
		// failed to found anything
		tld->found = 0;
	}
	ccv_array_free(dd);
	if (info)
		info->perform_learn = verified;
	if (verified)
		_ccv_tld_quick_learn(tld, b, result);
	tld->box = result;
	return result;
}

void ccv_tld_free(ccv_tld_t* tld)
{
	int i;
	for (i = 0; i < tld->sv[0]->rnum; i++)
		ccv_matrix_free(*(ccv_dense_matrix_t**)ccv_array_get(tld->sv[0], i));
	ccv_array_free(tld->sv[0]);
	for (i = 0; i < tld->sv[1]->rnum; i++)
		ccv_matrix_free(*(ccv_dense_matrix_t**)ccv_array_get(tld->sv[1], i));
	ccv_array_free(tld->sv[1]);
	ccv_array_free(tld->top);
	ccv_ferns_free(tld->ferns);
	ccfree(tld);
}
