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
	int i, j, k, w;
	int* buf = (int*)alloca(sizeof(int) * ccv_max(a->cols, a->rows));
	unsigned char* b_ptr = db->data.ptr;
	unsigned char* c_ptr = c->data.ptr;
	unsigned char* dx_ptr = dx->data.ptr;
	unsigned char* dy_ptr = dy->data.ptr;
	ccv_zero(db);
	int dx5[] = {-1, 0, 1, 0, 0};
	int dy5[] = {0, 0, 0, -1, 1};
	int dx9[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
	int dy9[] = {0, 0, 0, -1, -1, -1, 1, 1, 1};
	int adx, ady, sx, sy, err, e2, x0, x1, y0, y1, sk;
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
	int rdx, rdy, flag;
#define ray_emit(xx, xy, yx, yy, __for_get_d, __for_set_b, __for_get_b) \
	rdx = __for_get_d(dx_ptr, j, 0) * (xx) + __for_get_d(dy_ptr, j, 0) * (xy); \
	rdy = __for_get_d(dx_ptr, j, 0) * (yx) + __for_get_d(dy_ptr, j, 0) * (yy); \
	adx = abs(rdx); \
	ady = abs(rdy); \
	sx = rdx > 0 ? params.direct : -params.direct; \
	sy = rdy > 0 ? params.direct : -params.direct; \
	/* Bresenham's line algorithm */ \
	ray_reset(); \
	flag = 0; \
	sk = 1; \
	for (w = 0; w < 70; w++) \
	{ \
		ray_increment(); \
		if (x0 >= a->cols - 1 || x0 < 1 || y0 >= a->rows - 1 || y0 < 1) \
			break; \
		if (abs(i - y0) < 2 && abs(j - x0) < 2) \
		{ \
			if (c_ptr[x0 + (y0 - i) * c->step]) \
			{ \
				flag = 1; \
				break; \
			} \
		} else { /* ideally, I can encounter another edge directly, but in practice, we should search in a small region around it */ \
			flag = 0; \
			for (k = 0; k < 5; k++) \
				if (c_ptr[x0 + dx5[k] + (y0 - i + dy5[k]) * c->step]) \
				{ \
					sk = k; \
					flag = 1; \
					break; \
				} \
			if (flag) \
				break; \
		} \
	} \
	if (flag && x0 + dx5[sk] < a->cols - 1 && x0 + dx5[sk] > 0 && y0 + dy5[sk] < a->rows - 1 && y0 + dy5[sk] > 0) \
	{ \
		/* the opposite angle should be in d_p -/+ PI / 6 (otherwise discard),
		 * a faster computation should be:
		 * Tan(d_q - d_p) = (Tan(d_q) - Tan(d_p)) / (1 + Tan(d_q) * Tan(d_p))
		 * and -1 / sqrt(3) < Tan(d_q - d_p) < 1 / sqrt(3)
		 * also, we needs to check the whole 5x5 neighborhood in a hope that we don't miss one or two of them */ \
		flag = 0; \
		for (k = 0; k < 9; k++) \
		{ \
			int tn = __for_get_d(dy_ptr, j, 0) * __for_get_d(dx_ptr + (y0 + dy5[sk] - i + dy9[k]) * dx->step, x0 + dx5[sk] + dx9[k], 0) - \
					 __for_get_d(dx_ptr, j, 0) * __for_get_d(dy_ptr + (y0 + dy5[sk] - i + dy9[k]) * dy->step, x0 + dx5[sk] + dx9[k], 0); \
			int td = __for_get_d(dx_ptr, j, 0) * __for_get_d(dx_ptr + (y0 + dy5[sk] - i + dy9[k]) * dx->step, x0 + dx5[sk] + dx9[k], 0) + \
					 __for_get_d(dy_ptr, j, 0) * __for_get_d(dy_ptr + (y0 + dy5[sk] - i + dy9[k]) * dy->step, x0 + dx5[sk] + dx9[k], 0); \
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
			w = (int)(sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)) + 0.5); \
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
	}
#define for_block(__for_get_d, __for_set_b, __for_get_b) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			if (c_ptr[j]) \
			{ \
				ray_emit(1, 0, 0, 1, __for_get_d, __for_set_b, __for_get_b); \
				ray_emit(1, -1, 1, 1, __for_get_d, __for_set_b, __for_get_b); \
				ray_emit(1, 1, -1, 1, __for_get_d, __for_set_b, __for_get_b); \
			} \
		b_ptr += db->step; \
		c_ptr += c->step; \
		dx_ptr += dx->step; \
		dy_ptr += dy->step; \
	}
	ccv_matrix_getter(dx->type, ccv_matrix_setter_getter, db->type, for_block);
#undef for_block
#undef ray_emit
#undef ray_reset
#undef ray_increment
}

ccv_array_t* __ccv_swt_connected_component(ccv_dense_matrix_t* a, double ratio)
{
	int i, j, k;
	int* a_ptr = a->data.i;
	int dx8[] = {-1, 1, -1, 0, 1, -1, 0, 1};
	int dy8[] = {0, 0, -1, -1, -1, 1, 1, 1};
	int* marker = (int*)ccmalloc(sizeof(int) * a->rows * a->cols);
	memset(marker, 0, sizeof(int) * a->rows * a->cols);
	int* m_ptr = marker;
	ccv_point_t* buffer = (ccv_point_t*)ccmalloc(sizeof(ccv_point_t) * a->rows * a->cols);
	ccv_array_t* contours = ccv_array_new(5, sizeof(ccv_contour_t*));
	for (i = 0; i < a->rows; i++)
	{
		for (j = 0; j < a->cols; j++)
			if (a_ptr[j] != 0 && !m_ptr[j])
			{
				m_ptr[j] = 1;
				ccv_contour_t* contour = ccv_contour_new(1);
				ccv_point_t* closed = buffer;
				closed->x = j;
				closed->y = i;
				ccv_point_t* open = buffer + 1;
				for (; closed < open; closed++)
				{
					ccv_contour_push(contour, *closed);
					int color = a_ptr[closed->x + (closed->y - i) * a->cols];
					for (k = 0; k < 8; k++)
					{
						int nx = closed->x + dx8[k];
						int ny = closed->y + dy8[k];
						if (nx >= 0 && nx < a->cols && ny >= 0 && ny < a->rows &&
							a_ptr[nx + (ny - i) * a->cols] != 0 &&
							!m_ptr[nx + (ny - i) * a->cols] &&
							(a_ptr[nx + (ny - i) * a->cols] <= ratio * color && a_ptr[nx + (ny - i) * a->cols] * ratio >= color))
						{
							m_ptr[nx + (ny - i) * a->cols] = 1;
							open->x = nx;
							open->y = ny;
							open++;
						}
					}
				}
				ccv_array_push(contours, &contour);
			}
		a_ptr += a->cols;
		m_ptr += a->cols;
	}
	ccfree(marker);
	ccfree(buffer);
	return contours;
}

typedef struct {
	ccv_rect_t rect;
	int thickness;
	int brightness;
	double variance;
	double mean;
	ccv_contour_t* contour;
} ccv_letter_t;

static ccv_array_t* __ccv_connected_letters(ccv_dense_matrix_t* a, ccv_dense_matrix_t* swt)
{
	ccv_array_t* contours = __ccv_swt_connected_component(swt, 3.0);
	ccv_array_t* letters = ccv_array_new(5, sizeof(ccv_letter_t));
	int i, j, x, y;
	int* buffer = (int*)ccmalloc(sizeof(int) * swt->rows * swt->cols);
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
		double xc = (double)contour->m10 / contour->size;
		double yc = (double)contour->m01 / contour->size;
		double af = (double)contour->m20 / contour->size - xc * xc;
		double bf = 0.5 * ((double)contour->m11 / contour->size - xc * yc);
		double cf = (double)contour->m02 / contour->size - yc * yc;
		double delta = sqrt(bf * bf - (af - cf) * (af - cf));
		ratio = sqrt((af + cf + delta) / (af + cf - delta));
		if (ratio < 0.1 || ratio > 10)
		{
			ccv_contour_free(contour);
			continue;
		}
		double mean = 0;
		for (j = 0; j < contour->size; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			mean += swt->data.i[point->x + point->y * swt->cols];
			buffer[j] = swt->data.i[point->x + point->y * swt->cols];
		}
		mean = mean / contour->size;
		double variance = 0;
		for (j = 0; j < contour->size; j++)
			variance += (mean - buffer[j]) * (mean - buffer[j]);
		variance = variance / contour->size;
		ccv_letter_t letter;
		letter.variance = variance;
		letter.mean = mean;
		letter.thickness = __ccv_median(buffer, 0, contour->size - 1);
		letter.rect = contour->rect;
		letter.brightness = 0;
		letter.contour = contour;
		ccv_array_push(letters, &letter);
	}
	ccv_array_free(contours);
	memset(buffer, 0, sizeof(int) * swt->rows * swt->cols);
	ccv_array_t* new_letters = ccv_array_new(5, sizeof(ccv_letter_t));
	for (i = 0; i < letters->rnum; i++)
	{
		ccv_letter_t* letter = (ccv_letter_t*)ccv_array_get(letters, i);
		for (j = 0; j < letter->contour->size; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(letter->contour->set, j);
			buffer[point->x + point->y * swt->cols] = i + 1;
		}
	}
	for (i = 0; i < letters->rnum; i++)
	{
		ccv_letter_t* letter = (ccv_letter_t*)ccv_array_get(letters, i);
		if (sqrt(letter->variance) > letter->mean)
		{
			ccv_contour_free(letter->contour);
			continue;
		}
		int another[] = {0, 0};
		int more = 0;
		for (x = letter->rect.x; x < letter->rect.x + letter->rect.width; x++)
			for (y = letter->rect.y; y < letter->rect.y + letter->rect.height; y++)
				if (buffer[x + swt->cols * y] && buffer[x + swt->cols * y] != i + 1)
				{
					if (another[0])
					{
						if (buffer[x + swt->cols * y] != another[0])
						{
							if (another[1])
							{
								if (buffer[x + swt->cols * y] != another[1])
								{
									more = 1;
									break;
								}
							} else {
								another[1] = buffer[x + swt->cols * y];
							}
						}
					} else {
						another[0] = buffer[x + swt->cols * y];
					}
				}
		if (more)
		{
			ccv_contour_free(letter->contour);
			continue;
		}
		for (j = 0; j < letter->contour->set->rnum; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(letter->contour->set, j);
			letter->brightness += a->data.ptr[point->x + point->y * a->step];
		}
		letter->brightness /= letter->contour->size;
		ccv_contour_free(letter->contour);
		letter->contour = 0;
		ccv_array_push(new_letters, letter);
	}
	ccv_array_free(letters);
	ccfree(buffer);
	return new_letters;
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
			if (abs(dx) > 3 * ccv_max(li->rect.width, lj->rect.width))
				continue;
			int oy = ccv_min(li->rect.y + li->rect.height, lj->rect.y + lj->rect.height) - ccv_max(li->rect.y, lj->rect.y);
			if (oy * 2 < ccv_min(li->rect.height, lj->rect.height))
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
