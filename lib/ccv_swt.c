#include "ccv.h"

static inline int _ccv_median(int* buf, int low, int high)
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
	snprintf(identifier, 64, "ccv_swt(%d,%d,%lf,%lf)", params.direction, params.size, params.low_thresh, params.high_thresh);
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
	int adx, ady, sx, sy, err, e2, x0, x1, y0, y1, kx, ky;
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
#define ray_emit(xx, xy, yx, yy, _for_get_d, _for_set_b, _for_get_b) \
	rdx = _for_get_d(dx_ptr, j, 0) * (xx) + _for_get_d(dy_ptr, j, 0) * (xy); \
	rdy = _for_get_d(dx_ptr, j, 0) * (yx) + _for_get_d(dy_ptr, j, 0) * (yy); \
	adx = abs(rdx); \
	ady = abs(rdy); \
	sx = rdx > 0 ? params.direction : -params.direction; \
	sy = rdy > 0 ? params.direction : -params.direction; \
	/* Bresenham's line algorithm */ \
	ray_reset(); \
	flag = 0; \
	kx = x0; \
	ky = y0; \
	for (w = 0; w < 70; w++) \
	{ \
		ray_increment(); \
		if (x0 >= a->cols - 1 || x0 < 1 || y0 >= a->rows - 1 || y0 < 1) \
			break; \
		if (abs(i - y0) < 2 && abs(j - x0) < 2) \
		{ \
			if (c_ptr[x0 + (y0 - i) * c->step]) \
			{ \
				kx = x0; \
				ky = y0; \
				flag = 1; \
				break; \
			} \
		} else { /* ideally, I can encounter another edge directly, but in practice, we should search in a small region around it */ \
			flag = 0; \
			for (k = 0; k < 5; k++) \
			{ \
				kx = x0 + dx5[k]; \
				ky = y0 + dy5[k]; \
				if (c_ptr[kx + (ky - i) * c->step]) \
				{ \
					flag = 1; \
					break; \
				} \
			} \
			if (flag) \
				break; \
		} \
	} \
	if (flag && kx < a->cols - 1 && kx > 0 && ky < a->rows - 1 && ky > 0) \
	{ \
		/* the opposite angle should be in d_p -/+ PI / 6 (otherwise discard),
		 * a faster computation should be:
		 * Tan(d_q - d_p) = (Tan(d_q) - Tan(d_p)) / (1 + Tan(d_q) * Tan(d_p))
		 * and -1 / sqrt(3) < Tan(d_q - d_p) < 1 / sqrt(3)
		 * also, we needs to check the whole 3x3 neighborhood in a hope that we don't miss one or two of them */ \
		flag = 0; \
		for (k = 0; k < 9; k++) \
		{ \
			int tn = _for_get_d(dy_ptr, j, 0) * _for_get_d(dx_ptr + (ky - i + dy9[k]) * dx->step, kx + dx9[k], 0) - \
					 _for_get_d(dx_ptr, j, 0) * _for_get_d(dy_ptr + (ky - i + dy9[k]) * dy->step, kx + dx9[k], 0); \
			int td = _for_get_d(dx_ptr, j, 0) * _for_get_d(dx_ptr + (ky - i + dy9[k]) * dx->step, kx + dx9[k], 0) + \
					 _for_get_d(dy_ptr, j, 0) * _for_get_d(dy_ptr + (ky - i + dy9[k]) * dy->step, kx + dx9[k], 0); \
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
				if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) == 0 || _for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > w) \
				{ \
					_for_set_b(b_ptr + (y0 - i) * db->step, x0, w, 0); \
					buf[n++] = w; \
				} else if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) != 0) \
					buf[n++] = _for_get_b(b_ptr + (y0 - i) * db->step, x0, 0); \
				if (x0 == x1 && y0 == y1) \
					break; \
				ray_increment(); \
			} \
			int nw = _ccv_median(buf, 0, n - 1); \
			if (nw != w) \
			{ \
				ray_reset(); \
				for (;;) \
				{ \
					if (_for_get_b(b_ptr + (y0 - i) * db->step, x0, 0) > nw) \
						_for_set_b(b_ptr + (y0 - i) * db->step, x0, nw, 0); \
					if (x0 == x1 && y0 == y1) \
						break; \
					ray_increment(); \
				} \
			} \
		} \
	}
#define for_block(_for_get_d, _for_set_b, _for_get_b) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			if (c_ptr[j]) \
			{ \
				ray_emit(1, 0, 0, 1, _for_get_d, _for_set_b, _for_get_b); \
				ray_emit(1, -1, 1, 1, _for_get_d, _for_set_b, _for_get_b); \
				ray_emit(1, 1, -1, 1, _for_get_d, _for_set_b, _for_get_b); \
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
	ccv_matrix_free(c);
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
}

ccv_array_t* _ccv_swt_connected_component(ccv_dense_matrix_t* a, int ratio)
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
					int w = a_ptr[closed->x + (closed->y - i) * a->cols];
					for (k = 0; k < 8; k++)
					{
						int nx = closed->x + dx8[k];
						int ny = closed->y + dy8[k];
						if (nx >= 0 && nx < a->cols && ny >= 0 && ny < a->rows &&
							a_ptr[nx + (ny - i) * a->cols] != 0 &&
							!m_ptr[nx + (ny - i) * a->cols] &&
							(a_ptr[nx + (ny - i) * a->cols] <= ratio * w && a_ptr[nx + (ny - i) * a->cols] * ratio >= w))
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
	ccv_point_t center;
	int thickness;
	int intensity;
	double variance;
	double mean;
	ccv_contour_t* contour;
} ccv_letter_t;

static ccv_array_t* _ccv_swt_connected_letters(ccv_dense_matrix_t* a, ccv_dense_matrix_t* swt, ccv_swt_param_t params)
{
	ccv_array_t* contours = _ccv_swt_connected_component(swt, 3);
	ccv_array_t* letters = ccv_array_new(5, sizeof(ccv_letter_t));
	int i, j, x, y;
	int* buffer = (int*)ccmalloc(sizeof(int) * swt->rows * swt->cols);
	double aspect_ratio_inv = 1.0 / params.aspect_ratio;
	for (i = 0; i < contours->rnum; i++)
	{
		ccv_contour_t* contour = *(ccv_contour_t**)ccv_array_get(contours, i);
		if (contour->rect.height > params.max_height || contour->rect.height < params.min_height)
		{
			ccv_contour_free(contour);
			continue;
		}
		double ratio = (double)contour->rect.width / (double)contour->rect.height;
		if (ratio < aspect_ratio_inv || ratio > params.aspect_ratio)
		{
			ccv_contour_free(contour);
			continue;
		}
		double xc = (double)contour->m10 / contour->size;
		double yc = (double)contour->m01 / contour->size;
		double af = (double)contour->m20 / contour->size - xc * xc;
		double bf = 2 * ((double)contour->m11 / contour->size - xc * yc);
		double cf = (double)contour->m02 / contour->size - yc * yc;
		double delta = sqrt(bf * bf + (af - cf) * (af - cf));
		ratio = sqrt((af + cf + delta) / (af + cf - delta));
		if (ratio < aspect_ratio_inv || ratio > params.aspect_ratio)
		{
			ccv_contour_free(contour);
			continue;
		}
		double mean = 0;
		for (j = 0; j < contour->size; j++)
		{
			ccv_point_t* point = (ccv_point_t*)ccv_array_get(contour->set, j);
			mean += buffer[j] = swt->data.i[point->x + point->y * swt->cols];
		}
		mean = mean / contour->size;
		double variance = 0;
		for (j = 0; j < contour->size; j++)
			variance += (mean - buffer[j]) * (mean - buffer[j]);
		variance = variance / contour->size;
		ccv_letter_t letter;
		letter.variance = variance;
		letter.mean = mean;
		letter.thickness = _ccv_median(buffer, 0, contour->size - 1);
		letter.rect = contour->rect;
		letter.center.x = letter.rect.x + letter.rect.width / 2;
		letter.center.y = letter.rect.y + letter.rect.height / 2;
		letter.intensity = 0;
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
		if (sqrt(letter->variance) > letter->mean * params.variance_ratio)
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
			letter->intensity += a->data.ptr[point->x + point->y * a->step];
		}
		letter->intensity /= letter->contour->size;
		ccv_contour_free(letter->contour);
		letter->contour = 0;
		ccv_array_push(new_letters, letter);
	}
	ccv_array_free(letters);
	ccfree(buffer);
	return new_letters;
}

typedef struct {
	ccv_letter_t* left;
	ccv_letter_t* right;
	int dx;
	int dy;
} ccv_letter_pair_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	ccv_letter_t** letters;
} ccv_textline_t;

static int _ccv_in_textline(const void* a, const void* b, void* data)
{
	ccv_letter_pair_t* pair1 = (ccv_letter_pair_t*)a;
	ccv_letter_pair_t* pair2 = (ccv_letter_pair_t*)b;
	if (pair1->left == pair2->left || pair1->right == pair2->right)
	{
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the same end, opposite direction
		if (tn * 7 < -td * 4 && tn * 7 > td * 4)
			return 1;
	} else if (pair1->left == pair2->right || pair1->right == pair2->left) {
		int tn = pair1->dy * pair2->dx - pair1->dx * pair2->dy;
		int td = pair1->dx * pair2->dx + pair1->dy * pair2->dy;
		// share the other end, same direction
		if (tn * 7 < td * 4 && tn * 7 > -td * 4)
			return 1;
	}
	return 0;
}

static void _ccv_swt_add_letter(ccv_textline_t* textline, ccv_letter_t* letter)
{
	if (textline->neighbors == 0)
	{
		textline->rect = letter->rect;
		textline->neighbors = 1;
		textline->letters = (ccv_letter_t**)ccmalloc(sizeof(ccv_letter_t*) * textline->neighbors);
		textline->letters[0] = letter;
	} else {
		int i, flag = 0;
		for (i = 0; i < textline->neighbors; i++)
			if (textline->letters[i] == letter)
			{
				flag = 1;
				break;
			}
		if (flag)
			return;
		if (letter->rect.x < textline->rect.x)
		{
			textline->rect.width += textline->rect.x - letter->rect.x;
			textline->rect.x = letter->rect.x;
		}
		if (letter->rect.x + letter->rect.width > textline->rect.x + textline->rect.width)
			textline->rect.width = letter->rect.x + letter->rect.width - textline->rect.x;
		if (letter->rect.y < textline->rect.y)
		{
			textline->rect.height += textline->rect.y - letter->rect.y;
			textline->rect.y = letter->rect.y;
		}
		if (letter->rect.y + letter->rect.height > textline->rect.y + textline->rect.height)
			textline->rect.height = letter->rect.y + letter->rect.height - textline->rect.y;
		textline->neighbors++;
		textline->letters = (ccv_letter_t**)ccrealloc(textline->letters, sizeof(ccv_letter_t*) * textline->neighbors);
		textline->letters[textline->neighbors - 1] = letter;
	}
}

static ccv_array_t* _ccv_swt_merge_textline(ccv_array_t* letters, ccv_swt_param_t params)
{
	int i, j;
	ccv_array_t* pairs = ccv_array_new(letters->rnum * letters->rnum, sizeof(ccv_letter_pair_t));
	double thickness_ratio_inv = 1.0 / params.thickness_ratio;
	double height_ratio_inv = 1.0 / params.height_ratio;
	for (i = 0; i < letters->rnum - 1; i++)
	{
		ccv_letter_t* li = (ccv_letter_t*)ccv_array_get(letters, i);
		for (j = i + 1; j < letters->rnum; j++)
		{
			ccv_letter_t* lj = (ccv_letter_t*)ccv_array_get(letters, j);
			double ratio = (double)li->thickness / lj->thickness;
			if (ratio > params.thickness_ratio || ratio < thickness_ratio_inv)
				continue;
			ratio = (double)li->rect.height / lj->rect.height;
			if (ratio > params.height_ratio || ratio < height_ratio_inv)
				continue;
			if (abs(li->intensity - lj->intensity) > params.intensity_thresh)
				continue;
			int dx = li->rect.x - lj->rect.x + (li->rect.width - lj->rect.width) / 2;
			int dy = li->rect.y - lj->rect.y + (li->rect.height - lj->rect.height) / 2;
			if (abs(dx) > params.distance_ratio * ccv_max(li->rect.width, lj->rect.width))
				continue;
			int oy = ccv_min(li->rect.y + li->rect.height, lj->rect.y + lj->rect.height) - ccv_max(li->rect.y, lj->rect.y);
			if (oy * params.intersect_ratio < ccv_min(li->rect.height, lj->rect.height))
				continue;
			ccv_letter_pair_t pair = { .left = li, .right = lj, .dx = dx, .dy = dy };
			ccv_array_push(pairs, &pair);
		}
	}
	ccv_array_t* idx = 0;
	int nchains = ccv_array_group(pairs, &idx, _ccv_in_textline, 0);
	ccv_textline_t* chain = (ccv_textline_t*)ccmalloc(nchains * sizeof(ccv_textline_t));
	for (i = 0; i < nchains; i++)
		chain[i].neighbors = 0;
	for (i = 0; i < pairs->rnum; i++)
	{
		j = *(int*)ccv_array_get(idx, i);
		_ccv_swt_add_letter(chain + j,((ccv_letter_pair_t*)ccv_array_get(pairs, i))->left);
		_ccv_swt_add_letter(chain + j, ((ccv_letter_pair_t*)ccv_array_get(pairs, i))->right);
	}
	ccv_array_free(pairs);
	ccv_array_t* regions = ccv_array_new(5, sizeof(ccv_textline_t));
	for (i = 0; i < nchains; i++)
		if (chain[i].neighbors >= params.letter_thresh && chain[i].rect.width > chain[i].rect.height * params.elongate_ratio)
			ccv_array_push(regions, chain + i);
		else if (chain[i].neighbors > 0)
			ccfree(chain[i].letters);
	ccfree(chain);
	return regions;
}

#define less_than(a, b, aux) ((a)->center.x < (b)->center.x)
CCV_IMPLEMENT_QSORT(_ccv_sort_letters, ccv_letter_t*, less_than)
#undef less_than

static ccv_array_t* _ccv_swt_break_words(ccv_array_t* textline, ccv_swt_param_t params)
{
	int i, j, n = 0;
	for (i = 0; i < textline->rnum; i++)
	{
		ccv_textline_t* t = (ccv_textline_t*)ccv_array_get(textline, i);
		if (t->neighbors > n + 1)
			n = t->neighbors - 1;
	}
	int* buffer = (int*)alloca(n * sizeof(int));
	ccv_array_t* words = ccv_array_new(textline->rnum, sizeof(ccv_rect_t));
	for (i = 0; i < textline->rnum; i++)
	{
		ccv_textline_t* t = (ccv_textline_t*)ccv_array_get(textline, i);
		_ccv_sort_letters(t->letters, t->neighbors, 0);
		int range = 0;
		double mean = 0;
		for (j = 0; j < t->neighbors - 1; j++)
		{
			buffer[j] = t->letters[j + 1]->center.x - t->letters[j]->center.x;
			if (buffer[j] >= range)
				range = buffer[j] + 1;
			mean += buffer[j];
		}
		ccv_dense_matrix_t otsu = ccv_dense_matrix(1, t->neighbors - 1, CCV_32S | CCV_C1, buffer, 0);
		double var;
		int threshold = ccv_otsu(&otsu, &var, range);
		mean = mean / (t->neighbors - 1);
		if (var > mean * params.breakdown_ratio)
		{
			ccv_textline_t nt = { .neighbors = 0 };
			_ccv_swt_add_letter(&nt, t->letters[0]);
			for (j = 0; j < t->neighbors - 1; j++)
			{
				if (buffer[j] > threshold)
				{
					if (nt.neighbors >= params.letter_thresh && nt.rect.width > nt.rect.height * params.elongate_ratio)
						ccv_array_push(words, &nt.rect);
					nt.neighbors = 0;
				}
				_ccv_swt_add_letter(&nt, t->letters[j + 1]);
			}
			if (nt.neighbors >= params.letter_thresh && nt.rect.width > nt.rect.height * params.elongate_ratio)
				ccv_array_push(words, &nt.rect);
		} else {
			ccv_array_push(words, &(t->rect));
		}
	}
	return words;
}

static int _ccv_is_same_textline(const void* a, const void* b, void* data)
{
	ccv_textline_t* t1 = (ccv_textline_t*)a;
	ccv_textline_t* t2 = (ccv_textline_t*)b;
	int width = ccv_min(t1->rect.x + t1->rect.width, t2->rect.x + t2->rect.width) - ccv_max(t1->rect.x, t2->rect.x);
	int height = ccv_min(t1->rect.y + t1->rect.height, t2->rect.y + t2->rect.height) - ccv_max(t1->rect.y, t2->rect.y);
	/* overlapped 80% */
	return (width > 0 && height > 0 && width * height * 10 > ccv_min(t1->rect.width * t1->rect.height, t2->rect.width * t2->rect.height) * 8);
}

ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params)
{
	ccv_dense_matrix_t* swt = 0;
	params.direction = -1;
	ccv_swt(a, &swt, 0, params);
	/* perform connected component analysis */
	ccv_array_t* lettersB = _ccv_swt_connected_letters(a, swt, params);
	ccv_matrix_free(swt);
	ccv_array_t* textline = _ccv_swt_merge_textline(lettersB, params);
	swt = 0;
	params.direction = 1;
	ccv_swt(a, &swt, 0, params);
	ccv_array_t* lettersF = _ccv_swt_connected_letters(a, swt, params);
	ccv_matrix_free(swt);
	ccv_array_t* textline2 = _ccv_swt_merge_textline(lettersF, params);
	int i;
	for (i = 0; i < textline2->rnum; i++)
		ccv_array_push(textline, ccv_array_get(textline2, i));
	ccv_array_free(textline2);
	ccv_array_t* idx = 0;
	int ntl = ccv_array_group(textline, &idx, _ccv_is_same_textline, 0);
	ccv_array_t* words;
	if (params.breakdown)
	{
		textline2 = ccv_array_new(ntl, sizeof(ccv_textline_t));
		ccv_array_zero(textline2);
		textline2->rnum = ntl;
		for (i = 0; i < textline->rnum; i++)
		{
			ccv_textline_t* r = (ccv_textline_t*)ccv_array_get(textline, i);
			int k = *(int*)ccv_array_get(idx, i);
			ccv_textline_t* r2 = (ccv_textline_t*)ccv_array_get(textline2, k);
			if (r2->rect.width < r->rect.width)
			{
				if (r2->letters != 0)
					ccfree(r2->letters);
				*r2 = *r;
			}
		}
		ccv_array_free(idx);
		ccv_array_free(textline);
		words = _ccv_swt_break_words(textline2, params);
		for (i = 0; i < textline2->rnum; i++)
			ccfree(((ccv_textline_t*)ccv_array_get(textline2, i))->letters);
		ccv_array_free(textline2);
		ccv_array_free(lettersB);
		ccv_array_free(lettersF);
	} else {
		ccv_array_free(lettersB);
		ccv_array_free(lettersF);
		words = ccv_array_new(ntl, sizeof(ccv_rect_t));
		ccv_array_zero(words);
		words->rnum = ntl;
		for (i = 0; i < textline->rnum; i++)
		{
			ccv_textline_t* r = (ccv_textline_t*)ccv_array_get(textline, i);
			ccfree(r->letters);
			int k = *(int*)ccv_array_get(idx, i);
			ccv_rect_t* r2 = (ccv_rect_t*)ccv_array_get(words, k);
			if (r2->width * r2->height < r->rect.width * r->rect.height)
				*r2 = r->rect;
		}
		ccv_array_free(idx);
		ccv_array_free(textline);
	}
	return words;
}
