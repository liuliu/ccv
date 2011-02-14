#include "ccv.h"

static inline int __ccv_median(int* buf, int low, int high)
{
	int middle, ll, hh, w;
	int median = (low + high) / 2;
	for (;;)
	{
		if (high <= low)
			return buf[median];
		if (high == low + 1)
		{
			if (buf[low] > buf[high])
				CCV_SWAP(buf[low], buf[high], w);
			return buf[median];
		}
		middle = (low + high) / 2;
		if (buf[middle] > buf[high])
			CCV_SWAP(buf[middle], buf[high], w);
		if (buf[low] > buf[high])
			CCV_SWAP(buf[low], buf[high], w);
		if (buf[middle] > buf[low])
			CCV_SWAP(buf[middle], buf[low], w);
		CCV_SWAP(buf[middle], buf[low + 1], w);
		ll = low + 1;
		hh = high;
		for (;;)
		{
			do ll++; while (buf[low] > buf[ll]);
			do hh--; while (buf[hh] > buf[low]);
			if (hh < ll)
				break;
			CCV_SWAP(buf[ll], buf[hh], w);
		}
		CCV_SWAP(buf[low], buf[hh], w);
		if (hh <= median)
			low = ll;
		else if (hh >= median)
			high = hh - 1;
	}
}

/* ccv_swt is only the method to generate stroke width map */
void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_swt_param_t params)
{
	assert(a->type & CCV_C1);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_swt(%d,%d,%lf,%lf)", params.direct, params.size, params.low_thresh, params.high_thresh);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	ccv_dense_matrix_t* c = 0;
	ccv_canny(a, &c, 0, params.size, params.low_thresh, params.high_thresh);
	ccv_dense_matrix_t* dx = 0;
	ccv_sobel(a, &dx, 0, params.size, 0);
	ccv_dense_matrix_t* dy = 0;
	ccv_sobel(a, &dy, 0, 0, params.size);
	int i, j, k;
	int* buf = (int*)alloca(sizeof(int) * ccv_max(a->cols, a->rows));
	unsigned char* b_ptr = db->data.ptr;
	unsigned char* c_ptr = c->data.ptr;
	unsigned char* dx_ptr = dx->data.ptr;
	unsigned char* dy_ptr = dy->data.ptr;
	ccv_zero(db);
	int adx, ady, sx, sy, err, e2, x0, x1, y0, y1;
	int dx9[] = {-2, -1, 0, 1, 2, 0, 0, 0, 0};
	int dy9[] = {0, 0, 0, 0, 0, -2, -1, 1, 2};
	int dx25[] = {-2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2, -2, -1, 0, 1, 2};
	int dy25[] = {0, 0, 0, 0, 0, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
#define ray_reset() \
	err = adx - ady; e2 = 0; \
	x0 = j; y0 = i;
#define ray_increment() \
	e2 = 2 * err; \
	if (e2 > -ady) \
	{ \
		err -= ady; \
		x0 += sx; \
	} \
	if (e2 < adx) \
	{ \
		err += adx; \
		y0 += sy; \
	}
#define for_block(__for_get_d, __for_set_b, __for_get_b) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			if (c_ptr[j]) \
			{ \
				adx = abs(__for_get_d(dx_ptr, j, 0)); \
				ady = abs(__for_get_d(dy_ptr, j, 0)); \
				sx = __for_get_d(dx_ptr, j, 0) > 0 ? params.direct : -params.direct; \
				sy = __for_get_d(dy_ptr, j, 0) > 0 ? params.direct : -params.direct; \
				/* Bresenham's line algorithm */ \
				ray_reset(); \
				for (;;) \
				{ \
					ray_increment(); \
					if (x0 >= a->cols - 2 || x0 <= 1 || y0 >= a->rows - 2 || y0 <= 1) \
						break; \
					if (abs(i - y0) < 5 && abs(j - x0) < 5) \
					{ \
						if (c_ptr[x0 + (y0 - i) * c->step]) \
							break; \
					} else { /* ideally, I can encounter another edge directly, but in practice, we should search in a small region around it */ \
						int flag = 0; \
						for (k = 0; k < 9; k++) \
							if (c_ptr[x0 + dx9[k] + (y0 - i + dy9[k]) * c->step]) \
							{ \
								flag = 1; \
								break; \
							} \
						if (flag) \
							break; \
					} \
				} \
				if (x0 < a->cols - 2 && x0 > 1 && y0 < a->rows - 2 && y0 > 1) \
				{ \
					/* the opposite angle should be in d_p -/+ PI / 6 (otherwise discard),
					 * a faster computation should be:
					 * Tan(d_q - d_p) = (Tan(d_q) - Tan(d_p)) / (1 + Tan(d_q) * Tan(d_p))
					 * and -1 / sqrt(3) < Tan(d_q - d_p) < 1 / sqrt(3)
					 * also, we needs to check the whole 5x5 neighborhood in a hope that we don't miss one or two of them */ \
					int flag = 0; \
					for (k = 0; k < 25; k++) \
					{ \
						int tn = __for_get_d(dy_ptr, j, 0) * __for_get_d(dx_ptr + (y0 - i + dy25[k]) * dx->step, x0 + dx25[k], 0) - \
								 __for_get_d(dx_ptr, j, 0) * __for_get_d(dy_ptr + (y0 - i + dy25[k]) * dy->step, x0 + dx25[k], 0); \
						int td = __for_get_d(dx_ptr, j, 0) * __for_get_d(dx_ptr + (y0 - i + dy25[k]) * dx->step, x0 + dx25[k], 0) + \
								 __for_get_d(dy_ptr, j, 0) * __for_get_d(dy_ptr + (y0 - i + dy25[k]) * dy->step, x0 + dx25[k], 0); \
						if (tn * 7 < -td * 4 && tn * 7 > td * 4) \
						{ \
							flag = 1; \
							break; \
						} \
					} \
					if (flag) \
					{ \
						x1 = x0; y1 = y0; \
						int n = 0; \
						ray_reset(); \
						int w = (int)(sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) + 0.5); \
						/* extend the line to be width of 1 */ \
						for (;;) \
						{ \
							if (__for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) == 0 || __for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > w) \
							{ \
								__for_set_b(b_ptr + (y0 - i) * db->step, x0, w, 0); \
								buf[n++] = w; \
							} else if (__for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) != 0) \
								buf[n++] = __for_get_b(b_ptr + (y0 - i) * db->step, x0, 0); \
							if (x0 == x1 && y0 == y1) \
								break; \
							ray_increment(); \
						} \
						int nw = __ccv_median(buf, 0, n - 1); \
						if (nw != w) \
						{ \
							ray_reset(); \
							for (;;) \
							{ \
								if (__for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > nw) \
									__for_set_b(b_ptr + (y0 - i) * db->step, x0, nw, 0); \
								if (x0 == x1 && y0 == y1) \
									break; \
								ray_increment(); \
							} \
						} \
					} \
				} \
			} \
		b_ptr += db->step; \
		c_ptr += c->step; \
		dx_ptr += dx->step; \
		dy_ptr += dy->step; \
	}
	ccv_matrix_getter(dx->type, ccv_matrix_setter_getter, db->type, for_block);
#undef for_block
#undef ray_reset
#undef ray_increment
}

typedef struct {
	ccv_rect_t rect;
	int thickness;
	int brightness;
} ccv_letter_t;

static ccv_array_t* __ccv_connected_letters(ccv_dense_matrix_t* a, ccv_dense_matrix_t* swt)
{
	ccv_array_t* contours = ccv_connected_component(swt, 0, 1 << 8, 3.0, 1);
	ccv_array_t* letters = ccv_array_new(5, sizeof(ccv_letter_t));
	int i, j, x, y, n;
	int* labels = (int*)ccmalloc(sizeof(int) * swt->rows * swt->cols);
	int* buffer = (int*)ccmalloc(sizeof(int) * swt->rows * swt->cols);
	memset(labels, 0, sizeof(int) * swt->rows * swt->cols);
	for (i = 0; i < contours->rnum; i++)
	{
		ccv_contour_t* contour = *(ccv_contour_t**)ccv_array_get(contours, i);
		for (j = 0; j < contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			labels[point->x + point->y * swt->cols] = i + 1;
		}
	}
	for (i = 0; i < contours->rnum; i++)
	{
		ccv_contour_t* contour = *(ccv_contour_t**)ccv_array_get(contours, i);
		if (contour->rect.height > 300 || contour->rect.height < 10)
		{
			ccv_contour_free(contour);
			continue;
		}
		double ratio = (double)contour->rect.width / (double)contour->rect.height;
		if (ratio < 0.1 || ratio > 10)
		{
			ccv_contour_free(contour);
			continue;
		}
		double mean = 0;
		n = 0;
		for (j = 0; j < contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			mean += swt->data.i[point->x + point->y * swt->cols];
			buffer[n++] = swt->data.i[point->x + point->y * swt->cols];
		}
		mean = mean / contour->size;
		double variance = 0;
		for (j = 0; j < contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			variance += (mean - swt->data.i[point->x + point->y * swt->cols]) * (mean - swt->data.i[point->x + point->y * swt->cols]);
		}
		variance = variance / contour->size;
		if (sqrt(variance) > mean * 0.5)
		{
			ccv_contour_free(contour);
			continue;
		}
		ccv_letter_t letter;
		letter.thickness = __ccv_median(buffer, 0, n - 1);
		if (ccv_min(contour->rect.width, contour->rect.height) > 10 * letter.thickness)
		{
			ccv_contour_free(contour);
			continue;
		}
		letter.rect = contour->rect;
		int another[] = {0, 0};
		int more = 0;
		for (x = contour->rect.x; x < contour->rect.x + contour->rect.width; x++)
			for (y = contour->rect.y; y < contour->rect.y + contour->rect.height; y++)
				if (labels[x + swt->cols * y] && labels[x + swt->cols * y] != i + 1)
				{
					if (another[0])
					{
						if (labels[x + swt->cols * y] != another[0])
						{
							if (another[1])
							{
								if (labels[x + swt->cols * y] != another[1])
								{
									more = 1;
									break;
								}
							} else {
								another[1] = labels[x + swt->cols * y];
							}
						}
					} else {
						another[0] = labels[x + swt->cols * y];
					}
				}
		if (more)
		{
			ccv_contour_free(contour);
			continue;
		}
		letter.brightness = 0;
		for (j = 0; j < contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			letter.brightness += a->data.ptr[point->x + point->y * a->step];
		}
		letter.brightness /= contour->size;
		ccv_array_push(letters, &letter);
		ccv_contour_free(contour);
	}
	ccv_array_free(contours);
	ccfree(labels);
	ccfree(buffer);
	return letters;
}

typedef struct {
	int parent;
	int left;
	int right;
	int dx;
	int dy;
} ccv_letter_pair_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
} ccv_letter_chain_t;

static int __ccv_in_letter_chain(const void* a, const void* b, void* data)
{
	ccv_letter_pair_t* pair1 = (ccv_letter_pair_t*)a;
	ccv_letter_pair_t* pair2 = (ccv_letter_pair_t*)b;
	if (pair1->left == pair2->left || pair1->right == pair2->right)
	{
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the same end, opposite direction
		if (tn * 15 < -td * 4 && tn * 15 > td * 4)
			return 1;
	} else if (pair1->left == pair2->right || pair1->right == pair2->left) {
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the other end, same direction
		if (tn * 15 < td * 4 && tn * 15 > -td * 4)
			return 1;
	}
	return 0;
}

static ccv_array_t* __ccv_merge_textline(ccv_array_t* letters)
{
	int i, j;
	ccv_array_t* pairs = ccv_array_new(letters->rnum * letters->rnum, sizeof(ccv_letter_pair_t));
	for (i = 0; i < letters->rnum - 1; i++)
	{
		ccv_letter_t* li = (ccv_letter_t*)ccv_array_get(letters, i);
		for (j = i + 1; j < letters->rnum; j++)
		{
			ccv_letter_t* lj = (ccv_letter_t*)ccv_array_get(letters, j);
			double ratio = (double)li->thickness / lj->thickness;
			if (ratio > 2.0 || ratio < 0.5)
				continue;
			ratio = (double)li->rect.height / lj->rect.height;
			if (ratio > 2.0 || ratio < 0.5)
				continue;
			if (abs(li->brightness - lj->brightness) > 10)
				continue;
			int dx = li->rect.x - lj->rect.x + (li->rect.width - lj->rect.width) / 2;
			int dy = li->rect.y - lj->rect.y + (li->rect.height - lj->rect.height) / 2;
			int max_width = 3 * ccv_max(li->rect.width, lj->rect.width);
			if (dx * dx + dy * dy > max_width * max_width)
				continue;
			ccv_letter_pair_t pair = { .parent = -1, .left = i, .right = j, .dx = dx, .dy = dy };
			ccv_array_push(pairs, &pair);
		}
	}
	ccv_array_t* idx = 0;
	int nchains = ccv_array_group(pairs, &idx, __ccv_in_letter_chain, 0);
	ccv_letter_chain_t* chain = (ccv_letter_chain_t*)ccmalloc((nchains + 1) * sizeof(ccv_letter_chain_t));
	for (i = 0; i < nchains; i++)
		chain[i].neighbors = 0;
	for (i = 0; i < pairs->rnum; i++)
	{
		j = *(int*)ccv_array_get(idx, i);
		if (chain[j].neighbors == 0)
		{
			ccv_letter_t* li = (ccv_letter_t*)ccv_array_get(letters, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->left);
			ccv_letter_t* lj = (ccv_letter_t*)ccv_array_get(letters, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->right);
			chain[j].rect.x = ccv_min(li->rect.x, lj->rect.x);
			chain[j].rect.y = ccv_min(li->rect.y, lj->rect.y);
			chain[j].rect.width = ccv_max(li->rect.x + li->rect.width, lj->rect.x + lj->rect.width) - chain[j].rect.x;
			chain[j].rect.height = ccv_max(li->rect.y + li->rect.height, lj->rect.y + lj->rect.height) - chain[j].rect.y;
			chain[j].neighbors = 1;
		} else {
			ccv_letter_t* li = (ccv_letter_t*)ccv_array_get(letters, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->left);
			if (li->rect.x < chain[j].rect.x)
			{
				chain[j].rect.width += chain[j].rect.x - li->rect.x;
				chain[j].rect.x = li->rect.x;
			}
			if (li->rect.x + li->rect.width > chain[j].rect.x + chain[j].rect.width)
				chain[j].rect.width = li->rect.x + li->rect.width - chain[j].rect.x;
			if (li->rect.y < chain[j].rect.y)
			{
				chain[j].rect.height += chain[j].rect.y - li->rect.y;
				chain[j].rect.y = li->rect.y;
			}
			if (li->rect.y + li->rect.height > chain[j].rect.y + chain[j].rect.height)
				chain[j].rect.height = li->rect.y + li->rect.height - chain[j].rect.y;
			ccv_letter_t* lj = (ccv_letter_t*)ccv_array_get(letters, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->right);
			if (lj->rect.x < chain[j].rect.x)
			{
				chain[j].rect.width += chain[j].rect.x - lj->rect.x;
				chain[j].rect.x = lj->rect.x;
			}
			if (lj->rect.x + lj->rect.width > chain[j].rect.x + chain[j].rect.width)
				chain[j].rect.width = lj->rect.x + lj->rect.width - chain[j].rect.x;
			if (lj->rect.y < chain[j].rect.y)
			{
				chain[j].rect.height += chain[j].rect.y - lj->rect.y;
				chain[j].rect.y = lj->rect.y;
			}
			if (lj->rect.y + lj->rect.height > chain[j].rect.y + chain[j].rect.height)
				chain[j].rect.height = lj->rect.y + lj->rect.height - chain[j].rect.y;
			chain[j].neighbors++;
		}
	}
	ccv_array_free(pairs);
	ccv_array_t* regions = ccv_array_new(5, sizeof(ccv_rect_t));
	for (i = 0; i < nchains; i++)
		if (chain[i].neighbors > 1)
			ccv_array_push(regions, &chain[i].rect);
	ccfree(chain);
	return regions;
}

ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params)
{
	ccv_dense_matrix_t* swt = 0;
	params.direct = -1;
	ccv_swt(a, &swt, 0, params);
	/* perform connected component analysis */
	ccv_array_t* letters = __ccv_connected_letters(a, swt);
	ccv_array_t* textline = __ccv_merge_textline(letters);
	ccv_array_free(letters);
	ccv_matrix_free(swt);
	swt = 0;
	params.direct = 1;
	ccv_swt(a, &swt, 0, params);
	letters = __ccv_connected_letters(a, swt);
	ccv_matrix_free(swt);
	ccv_array_t* textline2 = __ccv_merge_textline(letters);
	ccv_array_free(letters);
	int i;
	for (i = 0; i < textline2->rnum; i++)
		ccv_array_push(textline, ccv_array_get(textline2, i));
	ccv_array_free(textline2);
	return textline;
}
