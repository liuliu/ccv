#include "ccv.h"
#include "ccv_internal.h"

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size)
{
	assert(a->rows >= size && a->cols >= size && (4 + sbin * 3) <= CCV_MAX_CHANNEL);
	int rows = a->rows / size;
	int cols = a->cols / size;
	b_type = (CCV_GET_DATA_TYPE(b_type) == CCV_64F) ? CCV_64F | (4 + sbin * 3) : CCV_32F | (4 + sbin * 3);
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_hog(%d,%d)", sbin, size), a->sig, CCV_EOF_SIGN);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_64F | CCV_32F | (4 + sbin * 3), b_type, sig);
	ccv_object_return_if_cached(, db);
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 1, 1);
	float* agp = ag->data.f32;
	float* mgp = mg->data.f32;
	int i, j, k, ch = CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* cn = ccv_dense_matrix_new(rows, cols, CCV_GET_DATA_TYPE(db->type) | (sbin * 2), 0, 0);
	ccv_dense_matrix_t* ca = ccv_dense_matrix_new(rows, cols, CCV_GET_DATA_TYPE(db->type) | CCV_C1, 0, 0);
	ccv_zero(cn);
	// normalize sbin direction-sensitive and sbin * 2 insensitive over 4 normalization factor
	// accumulating them over sbin * 2 + sbin + 4 channels
	// TNA - truncation - normalization - accumulation
#define TNA(_for_type, idx, a, b, c, d) \
	{ \
		_for_type norm = 1.0 / sqrt(cap[a] + cap[b] + cap[c] + cap[d] + 1e-4); \
		for (k = 0; k < sbin * 2; k++) \
		{ \
			_for_type v = 0.5 * ccv_min(cnp[k] * norm, 0.2); \
			dbp[4 + sbin + k] += v; \
			dbp[idx] += v; \
		} \
		dbp[idx] *= 0.2357; \
		for (k = 0; k < sbin; k++) \
		{ \
			_for_type v = 0.5 * ccv_min((cnp[k] + cnp[k + sbin]) * norm, 0.2); \
			dbp[4 + k] += v; \
		} \
	}
#define for_block(_, _for_type) \
	_for_type* cnp = (_for_type*)ccv_get_dense_matrix_cell(cn, 0, 0, 0); \
	for (i = 0; i < rows * size; i++) \
	{ \
		for (j = 0; j < cols * size; j++) \
		{ \
			_for_type agv = agp[j * ch]; \
			_for_type mgv = mgp[j * ch]; \
			for (k = 1; k < ch; k++) \
				if (mgp[j * ch + k] > mgv) \
				{ \
					mgv = mgp[j * ch + k]; \
					agv = agp[j * ch + k]; \
				} \
			_for_type agr0 = (ccv_clamp(agv, 0, 359.99) / 360.0) * (sbin * 2); \
			int ag0 = (int)agr0; \
			int ag1 = (ag0 + 1 < sbin * 2) ? ag0 + 1 : 0; \
			agr0 = agr0 - ag0; \
			_for_type agr1 = 1.0 - agr0; \
			mgv = mgv / 255.0; \
			_for_type yp = ((_for_type)i + 0.5) / (_for_type)size - 0.5; \
			_for_type xp = ((_for_type)j + 0.5) / (_for_type)size - 0.5; \
			int iyp = (int)floor(yp); \
			assert(iyp < rows); \
			int ixp = (int)floor(xp); \
			assert(ixp < cols); \
			_for_type vy0 = yp - iyp; \
			_for_type vx0 = xp - ixp; \
			_for_type vy1 = 1.0 - vy0; \
			_for_type vx1 = 1.0 - vx0; \
			if (ixp >= 0 && iyp >= 0) \
			{ \
				cnp[iyp * cn->cols * sbin * 2 + ixp * sbin * 2 + ag0] += agr1 * vx1 * vy1 * mgv; \
				cnp[iyp * cn->cols * sbin * 2 + ixp * sbin * 2 + ag1] += agr0 * vx1 * vy1 * mgv; \
			} \
			if (ixp + 1 < cn->cols && iyp >= 0) \
			{ \
				cnp[iyp * cn->cols * sbin * 2 + (ixp + 1) * sbin * 2 + ag0] += agr1 * vx0 * vy1 * mgv; \
				cnp[iyp * cn->cols * sbin * 2 + (ixp + 1) * sbin * 2 + ag1] += agr0 * vx0 * vy1 * mgv; \
			} \
			if (ixp >= 0 && iyp + 1 < cn->rows) \
			{ \
				cnp[(iyp + 1) * cn->cols * sbin * 2 + ixp * sbin * 2 + ag0] += agr1 * vx1 * vy0 * mgv; \
				cnp[(iyp + 1) * cn->cols * sbin * 2 + ixp * sbin * 2 + ag1] += agr0 * vx1 * vy0 * mgv; \
			} \
			if (ixp + 1 < cn->cols && iyp + 1 < cn->rows) \
			{ \
				cnp[(iyp + 1) * cn->cols * sbin * 2 + (ixp + 1) * sbin * 2 + ag0] += agr1 * vx0 * vy0 * mgv; \
				cnp[(iyp + 1) * cn->cols * sbin * 2 + (ixp + 1) * sbin * 2 + ag1] += agr0 * vx0 * vy0 * mgv; \
			} \
		} \
		agp += a->cols * ch; \
		mgp += a->cols * ch; \
	} \
	ccv_matrix_free(ag); \
	ccv_matrix_free(mg); \
	cnp = (_for_type*)ccv_get_dense_matrix_cell(cn, 0, 0, 0); \
	_for_type* cap = (_for_type*)ccv_get_dense_matrix_cell(ca, 0, 0, 0); \
	for (i = 0; i < rows; i++) \
	{ \
		for (j = 0; j < cols; j++) \
		{ \
			*cap = 0; \
			for (k = 0; k < sbin; k++) \
				*cap += (cnp[k] + cnp[k + sbin]) * (cnp[k] + cnp[k + sbin]); \
			cnp += 2 * sbin; \
			cap++; \
		} \
	} \
	cnp = (_for_type*)ccv_get_dense_matrix_cell(cn, 0, 0, 0); \
	cap = (_for_type*)ccv_get_dense_matrix_cell(ca, 0, 0, 0); \
	ccv_zero(db); \
	_for_type* dbp = (_for_type*)ccv_get_dense_matrix_cell(db, 0, 0, 0); \
	TNA(_for_type, 0, 1, cols + 1, cols, 0); \
	TNA(_for_type, 1, 1, 1, 0, 0); \
	TNA(_for_type, 2, 0, cols, cols, 0); \
	TNA(_for_type, 3, 0, 0, 0, 0); \
	cnp += 2 * sbin; \
	dbp += 3 * sbin + 4; \
	cap++; \
	for (j = 1; j < cols - 1; j++) \
	{ \
		TNA(_for_type, 0, 1, cols + 1, cols, 0); \
		TNA(_for_type, 1, 1, 1, 0, 0); \
		TNA(_for_type, 2, -1, cols - 1, cols, 0); \
		TNA(_for_type, 3, -1, -1, 0, 0); \
		cnp += 2 * sbin; \
		dbp += 3 * sbin + 4; \
		cap++; \
	} \
	TNA(_for_type, 0, 0, cols, cols, 0); \
	TNA(_for_type, 1, 0, 0, 0, 0); \
	TNA(_for_type, 2, -1, cols - 1, cols, 0); \
	TNA(_for_type, 3, -1, -1, 0, 0); \
	cnp += 2 * sbin; \
	dbp += 3 * sbin + 4; \
	cap++; \
	for (i = 1; i < rows - 1; i++) \
	{ \
		TNA(_for_type, 0, 1, cols + 1, cols, 0); \
		TNA(_for_type, 1, 1, -cols + 1, -cols, 0); \
		TNA(_for_type, 2, 0, cols, cols, 0); \
		TNA(_for_type, 3, 0, -cols, -cols, 0); \
		cnp += 2 * sbin; \
		dbp += 3 * sbin + 4; \
		cap++; \
		for (j = 1; j < cols - 1; j++) \
		{ \
			TNA(_for_type, 0, 1, cols + 1, cols, 0); \
			TNA(_for_type, 1, 1, -cols + 1, -cols, 0); \
			TNA(_for_type, 2, -1, cols - 1, cols, 0); \
			TNA(_for_type, 3, -1, -cols - 1, -cols, 0); \
			cnp += 2 * sbin; \
			dbp += 3 * sbin + 4; \
			cap++; \
		} \
		TNA(_for_type, 0, 0, cols, cols, 0); \
		TNA(_for_type, 1, 0, -cols, -cols, 0); \
		TNA(_for_type, 2, -1, cols - 1, cols, 0); \
		TNA(_for_type, 3, -1, -cols - 1, -cols, 0); \
		cnp += 2 * sbin; \
		dbp += 3 * sbin + 4; \
		cap++; \
	} \
	TNA(_for_type, 0, 1, 1, 0, 0); \
	TNA(_for_type, 1, 1, -cols + 1, -cols, 0); \
	TNA(_for_type, 2, 0, 0, 0, 0); \
	TNA(_for_type, 3, 0, -cols, -cols, 0); \
	cnp += 2 * sbin; \
	dbp += 3 * sbin + 4; \
	cap++; \
	for (j = 1; j < cols - 1; j++) \
	{ \
		TNA(_for_type, 0, 1, 1, 0, 0); \
		TNA(_for_type, 1, 1, -cols + 1, -cols, 0); \
		TNA(_for_type, 2, -1, -1, 0, 0); \
		TNA(_for_type, 3, -1, -cols - 1, -cols, 0); \
		cnp += 2 * sbin; \
		dbp += 3 * sbin + 4; \
		cap++; \
	} \
	TNA(_for_type, 0, 0, 0, 0, 0); \
	TNA(_for_type, 1, 0, -cols, -cols, 0); \
	TNA(_for_type, 2, -1, -1, 0, 0); \
	TNA(_for_type, 3, -1, -cols - 1, -cols, 0);
	ccv_matrix_typeof(db->type, for_block);
#undef for_block
#undef TNA
	ccv_matrix_free(cn);
	ccv_matrix_free(ca);
}

/* it is a supposely cleaner and faster implementation than original OpenCV (ccv_canny_deprecated,
 * removed, since the newer implementation achieve bit accuracy with OpenCV's), after a lot
 * profiling, the current implementation still uses integer to speed up */
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh)
{
	assert(CCV_GET_CHANNEL(a->type) == CCV_C1);
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_canny(%d,%la,%la)", size, low_thresh, high_thresh), a->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_8U | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_object_return_if_cached(, db);
	if ((a->type & CCV_8U) || (a->type & CCV_32S))
	{
		ccv_dense_matrix_t* dx = 0;
		ccv_dense_matrix_t* dy = 0;
		ccv_sobel(a, &dx, 0, size, 0);
		ccv_sobel(a, &dy, 0, 0, size);
		/* special case, all integer */
		int low = (int)(low_thresh + 0.5);
		int high = (int)(high_thresh + 0.5);
		int* dxi = dx->data.i32;
		int* dyi = dy->data.i32;
		int i, j;
		int* mbuf = (int*)alloca(3 * (a->cols + 2) * sizeof(int));
		memset(mbuf, 0, 3 * (a->cols + 2) * sizeof(int));
		int* rows[3];
		rows[0] = mbuf + 1;
		rows[1] = mbuf + (a->cols + 2) + 1;
		rows[2] = mbuf + 2 * (a->cols + 2) + 1;
		for (j = 0; j < a->cols; j++)
			rows[1][j] = abs(dxi[j]) + abs(dyi[j]);
		dxi += a->cols;
		dyi += a->cols;
		int* map = (int*)ccmalloc(sizeof(int) * (a->rows + 2) * (a->cols + 2));
		int map_cols = a->cols + 2;
		memset(map, 0, sizeof(int) * map_cols);
		int* map_ptr = map + map_cols + 1;
		int** stack = (int**)ccmalloc(sizeof(int*) * a->rows * a->cols);
		int** stack_top = stack;
		int** stack_bottom = stack;
		for (i = 1; i <= a->rows; i++)
		{
			/* the if clause should be unswitched automatically, no need to manually do so */
			if (i == a->rows)
				memset(rows[2], 0, sizeof(int) * a->cols);
			else
				for (j = 0; j < a->cols; j++)
					rows[2][j] = abs(dxi[j]) + abs(dyi[j]);
			int* _dx = dxi - a->cols;
			int* _dy = dyi - a->cols;
			map_ptr[-1] = 0;
			int suppress = 0;
			for (j = 0; j < a->cols; j++)
			{
				int f = rows[1][j];
				if (f > low)
				{
					int x = abs(_dx[j]);
					int y = abs(_dy[j]);
					int s = _dx[j] ^ _dy[j];
					/* x * tan(22.5) */
					int tg22x = x * (int)(0.4142135623730950488016887242097 * (1 << 15) + 0.5);
					/* x * tan(67.5) == 2 * x + x * tan(22.5) */
					int tg67x = tg22x + ((x + x) << 15);
					y <<= 15;
					/* it is a little different from the Canny original paper because we adopted the coordinate system of
					 * top-left corner as origin. Thus, the derivative of y convolved with matrix:
					 * |-1 -2 -1|
					 * | 0  0  0|
					 * | 1  2  1|
					 * actually is the reverse of real y. Thus, the computed angle will be mirrored around x-axis.
					 * In this case, when angle is -45 (135), we compare with north-east and south-west, and for 45,
					 * we compare with north-west and south-east (in traditional coordinate system sense, the same if we
					 * adopt top-left corner as origin for "north", "south", "east", "west" accordingly) */
#define high_block \
					{ \
						if (f > high && !suppress && map_ptr[j - map_cols] != 2) \
						{ \
							map_ptr[j] = 2; \
							suppress = 1; \
							*(stack_top++) = map_ptr + j; \
						} else { \
							map_ptr[j] = 1; \
						} \
						continue; \
					}
					/* sometimes, we end up with same f in integer domain, for that case, we will take the first occurrence
					 * suppressing the second with flag */
					if (y < tg22x)
					{
						if (f > rows[1][j - 1] && f >= rows[1][j + 1])
							high_block;
					} else if (y > tg67x) {
						if (f > rows[0][j] && f >= rows[2][j])
							high_block;
					} else {
						s = s < 0 ? -1 : 1;
						if (f > rows[0][j - s] && f > rows[2][j + s])
							high_block;
					}
#undef high_block
				}
				map_ptr[j] = 0;
				suppress = 0;
			}
			map_ptr[a->cols] = 0;
			map_ptr += map_cols;
			dxi += a->cols;
			dyi += a->cols;
			int* row = rows[0];
			rows[0] = rows[1];
			rows[1] = rows[2];
			rows[2] = row;
		}
		memset(map_ptr - 1, 0, sizeof(int) * map_cols);
		int dr[] = {-1, 1, -map_cols - 1, -map_cols, -map_cols + 1, map_cols - 1, map_cols, map_cols + 1};
		while (stack_top > stack_bottom)
		{
			map_ptr = *(--stack_top);
			for (i = 0; i < 8; i++)
				if (map_ptr[dr[i]] == 1)
				{
					map_ptr[dr[i]] = 2;
					*(stack_top++) = map_ptr + dr[i];
				}
		}
		map_ptr = map + map_cols + 1;
		unsigned char* b_ptr = db->data.u8;
#define for_block(_, _for_set) \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
				_for_set(b_ptr, j, (map_ptr[j] == 2), 0); \
			map_ptr += map_cols; \
			b_ptr += db->step; \
		}
		ccv_matrix_setter(db->type, for_block);
#undef for_block
		ccfree(stack);
		ccfree(map);
		ccv_matrix_free(dx);
		ccv_matrix_free(dy);
	} else {
		/* general case, use all ccv facilities to deal with it */
		ccv_dense_matrix_t* mg = 0;
		ccv_dense_matrix_t* ag = 0;
		ccv_gradient(a, &ag, 0, &mg, 0, size, size);
		ccv_matrix_free(ag);
		ccv_matrix_free(mg);
		/* FIXME: Canny implementation for general case */
	}
}

void ccv_close_outline(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type)
{
	assert((CCV_GET_CHANNEL(a->type) == CCV_C1) && ((a->type & CCV_8U) || (a->type & CCV_32S) || (a->type & CCV_64S)));
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_literal("ccv_close_outline"), a->sig, CCV_EOF_SIGN);
	type = ((type == 0) || (type & CCV_32F) || (type & CCV_64F)) ? CCV_GET_DATA_TYPE(a->type) | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_object_return_if_cached(, db);
	int i, j;
	unsigned char* a_ptr = a->data.u8;
	unsigned char* b_ptr = db->data.u8;
	ccv_zero(db);
#define for_block(_for_get, _for_set_b, _for_get_b) \
	for (i = 0; i < a->rows - 1; i++) \
	{ \
		for (j = 0; j < a->cols - 1; j++) \
		{ \
			if (!_for_get_b(b_ptr, j, 0)) \
				_for_set_b(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
			if (_for_get(a_ptr, j, 0) && _for_get(a_ptr + a->step, j + 1, 0)) \
			{ \
				_for_set_b(b_ptr + a->step, j, 1, 0); \
				_for_set_b(b_ptr, j + 1, 1, 0); \
			} \
			if (_for_get(a_ptr + a->step, j, 0) && _for_get(a_ptr, j + 1, 0)) \
			{ \
				_for_set_b(b_ptr, j, 1, 0); \
				_for_set_b(b_ptr + a->step, j + 1, 1, 0); \
			} \
		} \
		if (!_for_get_b(b_ptr, a->cols - 1, 0)) \
			_for_set_b(b_ptr, a->cols - 1, _for_get(a_ptr, a->cols - 1, 0), 0); \
		a_ptr += a->step; \
		b_ptr += db->step; \
	} \
	for (j = 0; j < a->cols; j++) \
	{ \
		if (!_for_get_b(b_ptr, j, 0)) \
			_for_set_b(b_ptr, j, _for_get(a_ptr, j, 0), 0); \
	}
	ccv_matrix_getter_integer_only(a->type, ccv_matrix_setter_getter_integer_only, db->type, for_block);
#undef for_block
}

int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range)
{
	assert((a->type & CCV_32S) || (a->type & CCV_8U));
	int* histogram = (int*)alloca(range * sizeof(int));
	memset(histogram, 0, sizeof(int) * range);
	int i, j;
	unsigned char* a_ptr = a->data.u8;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			histogram[ccv_clamp((int)_for_get(a_ptr, j, 0), 0, range - 1)]++; \
		a_ptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	double sum = 0, sumB = 0;
	for (i = 0; i < range; i++)
		sum += i * histogram[i];
	int wB = 0, wF = 0, total = a->rows * a->cols;
	double maxVar = 0;
	int threshold = 0;
	for (i = 0; i < range; i++)
	{
		wB += histogram[i];
		if (wB == 0)
			continue;
		wF = total - wB;
		if (wF == 0)
			break;
		sumB += i * histogram[i];
		double mB = sumB / wB;
		double mF = (sum - sumB) / wF;
		double var = wB * wF * (mB - mF) * (mB - mF);
		if (var > maxVar)
		{
			maxVar = var;
			threshold = i;
		}
	}
	if (outvar != 0)
		*outvar = maxVar / total / total;
	return threshold;
}

#define LK_MAX_ITER (30)
#define LK_EPSILON (0.01)

/* this code is a rewrite from OpenCV's legendary Lucas-Kanade optical flow implementation */
void ccv_optical_flow_lucas_kanade(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_array_t* point_a, ccv_array_t** point_b, ccv_size_t win_size, int level, double min_eigen)
{
	assert(a && b && a->rows == b->rows && a->cols == b->cols);
	assert(CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(b->type) && CCV_GET_DATA_TYPE(a->type) == CCV_GET_DATA_TYPE(b->type));
	assert(CCV_GET_CHANNEL(a->type) == 1);
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_8U);
	assert(point_a->rnum > 0);
	level = ccv_clamp(level + 1, 1, (int)(log((double)ccv_min(a->rows, a->cols) / ccv_max(win_size.width * 2, win_size.height * 2)) / log(2.0) + 0.5));
	ccv_declare_derived_signature(sig, a->sig != 0 && b->sig != 0 && point_a->sig != 0, ccv_sign_with_format(128, "ccv_optical_flow_lucas_kanade(%d,%d,%d,%la)", win_size.width, win_size.height, level, min_eigen), a->sig, b->sig, point_a->sig, CCV_EOF_SIGN);
	ccv_array_t* seq = *point_b = ccv_array_new(sizeof(ccv_decimal_point_with_status_t), point_a->rnum, sig);
	ccv_object_return_if_cached(, seq);
	seq->rnum = point_a->rnum;
	ccv_dense_matrix_t** pyr_a = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * level);
	ccv_dense_matrix_t** pyr_a_dx = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * level);
	ccv_dense_matrix_t** pyr_a_dy = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * level);
	ccv_dense_matrix_t** pyr_b = (ccv_dense_matrix_t**)alloca(sizeof(ccv_dense_matrix_t*) * level);
	int i, j, t, x, y;
	/* generating image pyramid */
	pyr_a[0] = a;
	pyr_a_dx[0] = pyr_a_dy[0] = 0;
	ccv_sobel(pyr_a[0], &pyr_a_dx[0], 0, 3, 0);
	ccv_sobel(pyr_a[0], &pyr_a_dy[0], 0, 0, 3);
	pyr_b[0] = b;
	for (i = 1; i < level; i++)
	{
		pyr_a[i] = pyr_a_dx[i] = pyr_a_dy[i] = pyr_b[i] = 0;
		ccv_sample_down(pyr_a[i - 1], &pyr_a[i], 0, 0, 0);
		ccv_sobel(pyr_a[i], &pyr_a_dx[i], 0, 3, 0);
		ccv_sobel(pyr_a[i], &pyr_a_dy[i], 0, 0, 3);
		ccv_sample_down(pyr_b[i - 1], &pyr_b[i], 0, 0, 0);
	}
	int* wi = (int*)alloca(sizeof(int) * win_size.width * win_size.height);
	int* widx = (int*)alloca(sizeof(int) * win_size.width * win_size.height);
	int* widy = (int*)alloca(sizeof(int) * win_size.width * win_size.height);
	ccv_decimal_point_t half_win = ccv_decimal_point((win_size.width - 1) * 0.5f, (win_size.height - 1) * 0.5f);
	const int W_BITS14 = 14, W_BITS7 = 7, W_BITS9 = 9;
	const float FLT_SCALE = 1.0f / (1 << 25);
	// clean up status to 1
	for (i = 0; i < point_a->rnum; i++)
	{
		ccv_decimal_point_with_status_t* point_with_status = (ccv_decimal_point_with_status_t*)ccv_array_get(seq, i);
		point_with_status->status = 1;
	}
	int prev_rows, prev_cols;
	for (t = level - 1; t >= 0; t--)
	{
		ccv_dense_matrix_t* a = pyr_a[t];
		ccv_dense_matrix_t* adx = pyr_a_dx[t];
		ccv_dense_matrix_t* ady = pyr_a_dy[t];
		assert(CCV_GET_DATA_TYPE(adx->type) == CCV_32S);
		assert(CCV_GET_DATA_TYPE(ady->type) == CCV_32S);
		ccv_dense_matrix_t* b = pyr_b[t];
		for (i = 0; i < point_a->rnum; i++)
		{
			ccv_decimal_point_t prev_point = *(ccv_decimal_point_t*)ccv_array_get(point_a, i);
			ccv_decimal_point_with_status_t* point_with_status = (ccv_decimal_point_with_status_t*)ccv_array_get(seq, i);
			prev_point.x = prev_point.x / (float)(1 << t);
			prev_point.y = prev_point.y / (float)(1 << t);
			ccv_decimal_point_t next_point;
			if (t == level - 1)
				next_point = prev_point;
			else {
				next_point.x = point_with_status->point.x * 2 + (a->cols - prev_cols * 2) * 0.5;
				next_point.y = point_with_status->point.y * 2 + (a->rows - prev_rows * 2) * 0.5;
			}
			point_with_status->point = next_point;
			prev_point.x -= half_win.x;
			prev_point.y -= half_win.y;
			ccv_point_t iprev_point = ccv_point((int)prev_point.x, (int)prev_point.y);
			if (iprev_point.x < 0 || iprev_point.x >= a->cols - win_size.width - 1 ||
				iprev_point.y < 0 || iprev_point.y >= a->rows - win_size.height - 1)
			{
				if (t == 0)
					point_with_status->status = 0;
				continue;
			}
			float xd = prev_point.x - iprev_point.x;
			float yd = prev_point.y - iprev_point.y;
			int iw00 = (int)((1 - xd) * (1 - yd) * (1 << W_BITS14) + 0.5);
			int iw01 = (int)(xd * (1 - yd) * (1 << W_BITS14) + 0.5);
			int iw10 = (int)((1 - xd) * yd * (1 << W_BITS14) + 0.5);
			int iw11 = (1 << W_BITS14) - iw00 - iw01 - iw10;
			float a11 = 0, a12 = 0, a22 = 0;
			unsigned char* a_ptr = (unsigned char*)ccv_get_dense_matrix_cell_by(CCV_C1 | CCV_8U, a, iprev_point.y, iprev_point.x, 0);
			int* adx_ptr = (int*)ccv_get_dense_matrix_cell_by(CCV_C1 | CCV_32S, adx, iprev_point.y, iprev_point.x, 0);
			int* ady_ptr = (int*)ccv_get_dense_matrix_cell_by(CCV_C1 | CCV_32S, ady, iprev_point.y, iprev_point.x, 0);
			int* wi_ptr = wi;
			int* widx_ptr = widx;
			int* widy_ptr = widy;
			for (y = 0; y < win_size.height; y++)
			{
				for (x = 0; x < win_size.width; x++)
				{
					wi_ptr[x] = ccv_descale(a_ptr[x] * iw00 + a_ptr[x + 1] * iw01 + a_ptr[x + a->step] * iw10 + a_ptr[x + a->step + 1] * iw11, W_BITS7);
					// because we use 3x3 sobel, which scaled derivative up by 4
					widx_ptr[x] = ccv_descale(adx_ptr[x] * iw00 + adx_ptr[x + 1] * iw01 + adx_ptr[x + adx->cols] * iw10 + adx_ptr[x + adx->cols + 1] * iw11, W_BITS9);
					widy_ptr[x] = ccv_descale(ady_ptr[x] * iw00 + ady_ptr[x + 1] * iw01 + ady_ptr[x + ady->cols] + iw10 + ady_ptr[x + ady->cols + 1] * iw11, W_BITS9);
					a11 += (float)(widx_ptr[x] * widx_ptr[x]);
					a12 += (float)(widx_ptr[x] * widy_ptr[x]);
					a22 += (float)(widy_ptr[x] * widy_ptr[x]);
				}
				a_ptr += a->step;
				adx_ptr += adx->cols;
				ady_ptr += ady->cols;
				wi_ptr += win_size.width;
				widx_ptr += win_size.width;
				widy_ptr += win_size.width;
			}
			a11 *= FLT_SCALE;
			a12 *= FLT_SCALE;
			a22 *= FLT_SCALE;
			float D = a11 * a22 - a12 * a12;
			float eigen = (a22 + a11 - sqrtf((a11 - a22) * (a11 - a22) + 4.0f * a12 * a12)) / (2 * win_size.width * win_size.height);
			if (eigen < min_eigen || D < FLT_EPSILON)
			{
				if (t == 0)
					point_with_status->status = 0;
				continue;
			}
			D = 1.0f / D;
			next_point.x -= half_win.x;
			next_point.y -= half_win.y;
			ccv_decimal_point_t prev_delta;
			for (j = 0; j < LK_MAX_ITER; j++)
			{
				ccv_point_t inext_point = ccv_point((int)next_point.x, (int)next_point.y);
				if (inext_point.x < 0 || inext_point.x >= a->cols - win_size.width - 1 ||
					inext_point.y < 0 || inext_point.y >= a->rows - win_size.height - 1)
					break;
				float xd = next_point.x - inext_point.x;
				float yd = next_point.y - inext_point.y;
				int iw00 = (int)((1 - xd) * (1 - yd) * (1 << W_BITS14) + 0.5);
				int iw01 = (int)(xd * (1 - yd) * (1 << W_BITS14) + 0.5);
				int iw10 = (int)((1 - xd) * yd * (1 << W_BITS14) + 0.5);
				int iw11 = (1 << W_BITS14) - iw00 - iw01 - iw10;
				float b1 = 0, b2 = 0;
				unsigned char* b_ptr = (unsigned char*)ccv_get_dense_matrix_cell_by(CCV_C1 | CCV_8U, b, inext_point.y, inext_point.x, 0);
				int* wi_ptr = wi;
				int* widx_ptr = widx;
				int* widy_ptr = widy;
				for (y = 0; y < win_size.height; y++)
				{
					for (x = 0; x < win_size.width; x++)
					{
						int diff = ccv_descale(b_ptr[x] * iw00 + b_ptr[x + 1] * iw01 + b_ptr[x + b->step] * iw10 + b_ptr[x + b->step + 1] * iw11, W_BITS7) - wi_ptr[x];
						b1 += (float)(diff * widx_ptr[x]);
						b2 += (float)(diff * widy_ptr[x]);
					}
					b_ptr += b->step;
					wi_ptr += win_size.width;
					widx_ptr += win_size.width;
					widy_ptr += win_size.width;
				}
				b1 *= FLT_SCALE;
				b2 *= FLT_SCALE;
				ccv_decimal_point_t delta = ccv_decimal_point((a12 * b2 - a22 * b1) * D, (a12 * b1 - a11 * b2) * D);
				next_point.x += delta.x;
				next_point.y += delta.y;
				if (delta.x * delta.x + delta.y * delta.y < LK_EPSILON)
					break;
				if (j > 0 && fabs(prev_delta.x - delta.x) < 0.01 && fabs(prev_delta.y - delta.y) < 0.01)
				{
					next_point.x -= delta.x * 0.5;
					next_point.y -= delta.y * 0.5;
					break;
				}
				prev_delta = delta;
			}
			ccv_point_t inext_point = ccv_point((int)next_point.x, (int)next_point.y);
			if (inext_point.x < 0 || inext_point.x >= a->cols - win_size.width - 1 ||
				inext_point.y < 0 || inext_point.y >= a->rows - win_size.height - 1)
				point_with_status->status = 0;
			else {
				point_with_status->point.x = next_point.x + half_win.x;
				point_with_status->point.y = next_point.y + half_win.y;
			}
		}
		prev_rows = a->rows;
		prev_cols = a->cols;
		ccv_matrix_free(adx);
		ccv_matrix_free(ady);
		if (t > 0)
		{
			ccv_matrix_free(a);
			ccv_matrix_free(b);
		}
	}
}
