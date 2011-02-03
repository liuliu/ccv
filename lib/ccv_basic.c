#include "ccv.h"

void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int dx, int dy)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sobel(%d,%d)", dx, dy);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	int i, j;
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	if (dx == 1 || dy == 1)
	{
		if (dx > dy)
		{
#define for_block(__for_get, __for_set) \
			for (i = 0; i < a->rows; i++) \
			{ \
				__for_set(b_ptr, 0, __for_get(a_ptr, 1, 0) - __for_get(a_ptr, 0, 0), 0); \
				for (j = 1; j < a->cols - 1; j++) \
					__for_set(b_ptr, j, 2 * (__for_get(a_ptr, j + 1, 0) - __for_get(a_ptr, j - 1, 0)), 0); \
				__for_set(b_ptr, a->cols - 1, __for_get(a_ptr, a->cols - 1, 0) - __for_get(a_ptr, a->cols - 2, 0), 0); \
				b_ptr += db->step; \
				a_ptr += a->step; \
			}
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
		} else {
#define for_block(__for_get, __for_set) \
			for (j = 0; j < a->cols; j++) \
				__for_set(b_ptr, j, __for_get(a_ptr + a->step, j, 0) - __for_get(a_ptr, j, 0), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
			for (i = 1; i < a->rows - 1; i++) \
			{ \
				for (j = 0; j < a->cols; j++) \
					__for_set(b_ptr, j, 2 * (__for_get(a_ptr + a->step, j, 0) - __for_get(a_ptr - a->step, j, 0)), 0); \
				a_ptr += a->step; \
				b_ptr += db->step; \
			} \
			for (j = 0; j < a->cols; j++) \
				__for_set(b_ptr, j, __for_get(a_ptr, j, 0) - __for_get(a_ptr - a->step, j, 0), 0);
			ccv_matrix_getter(a->type, ccv_matrix_setter, db->type, for_block);
#undef for_block
		}
	} else {
		/* FIXME: the calculation indeed requires a second matrix, but, it shouldn't be limited in this case (3x3) only */
		ccv_dense_matrix_t* c = ccv_dense_matrix_new(a->rows, a->cols, CCV_32S | CCV_C1, 0, 0);
		int* c_ptr = c->data.i;
		if (dx > dy)
		{
			for (i = 0; i < a->rows; i++)
			{
				c_ptr[0] = a_ptr[1] - a_ptr[0];
				for (j = 1; j < a->cols - 1; j++)
					c_ptr[j] = a_ptr[j + 1] - a_ptr[j - 1];
				c_ptr[c->cols - 1] = a_ptr[a->cols - 1] - a_ptr[a->cols - 2];
				a_ptr += a->step;
				c_ptr += c->cols;
			}
			c_ptr = c->data.i;
#define for_block(dummy, __for_set) \
			for (j = 0; j < c->cols; j++) \
				__for_set(b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j], 0); \
			b_ptr += db->step; \
			c_ptr += c->cols; \
			for (i = 1; i < c->rows - 1; i++) \
			{ \
				for (j = 0; j < c->cols; j++) \
					__for_set(b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j] + c_ptr[j - c->cols], 0); \
				b_ptr += db->step; \
				c_ptr += c->cols; \
			} \
			for (j = 0; j < c->cols; j++) \
				__for_set(b_ptr, j, 2 * c_ptr[j] + c_ptr[j - c->cols], 0);
			ccv_matrix_setter(db->type, for_block);
#undef for_block
		} else {
			for (j = 0; j < a->cols; j++)
				c_ptr[j] = a_ptr[j + a->step] - a_ptr[j];
			a_ptr += a->step;
			c_ptr += c->cols;
			for (i = 1; i < a->rows - 1; i++)
			{
				for (j = 0; j < a->cols; j++)
					c_ptr[j] = a_ptr[j + a->step] - a_ptr[j - a->step];
				a_ptr += a->step;
				c_ptr += c->cols;
			}
			for (j = 0; j < a->cols; j++)
				c_ptr[j] = a_ptr[j] - a_ptr[j - a->step];
			c_ptr = c->data.i;
#define for_block(dummy, __for_set) \
			for (i = 0; i < c->rows; i++) \
			{ \
				__for_set(b_ptr, 0, c_ptr[1] + 2 * c_ptr[0], 0); \
				for (j = 1; j < c->cols - 1; j++) \
					__for_set(b_ptr, j, c_ptr[j + 1] + 2 * c_ptr[j] + c_ptr[j - 1], 0); \
				__for_set(b_ptr, c->cols - 1, c_ptr[c->cols - 1] + 2 * c_ptr[c->cols - 2], 0); \
				b_ptr += db->step; \
				c_ptr += c->cols; \
			}
			ccv_matrix_setter(db->type, for_block);
#undef for_block
		}
		ccv_matrix_free(c);
	}
}

/* the fast arctan function adopted from OpenCV */
static void __ccv_atan2(float* x, float* y, float* angle, float* mag, int len)
{
	int i = 0;
	float scale = (float)(180.0 / CCV_PI);
#ifndef _WIN32
	union { int i; float fl; } iabsmask; iabsmask.i = 0x7fffffff;
	__m128 eps = _mm_set1_ps((float)1e-6), absmask = _mm_set1_ps(iabsmask.fl);
	__m128 _90 = _mm_set1_ps((float)(3.141592654 * 0.5)), _180 = _mm_set1_ps((float)3.141592654), _360 = _mm_set1_ps((float)(3.141592654 * 2));
	__m128 zero = _mm_setzero_ps(), _0_28 = _mm_set1_ps(0.28f), scale4 = _mm_set1_ps(scale);
	
	for(; i <= len - 4; i += 4)
	{
		__m128 x4 = _mm_loadu_ps(x + i), y4 = _mm_loadu_ps(y + i);
		__m128 xq4 = _mm_mul_ps(x4, x4), yq4 = _mm_mul_ps(y4, y4);
		__m128 xly = _mm_cmplt_ps(xq4, yq4);
		__m128 z4 = _mm_div_ps(_mm_mul_ps(x4, y4), _mm_add_ps(_mm_add_ps(_mm_max_ps(xq4, yq4), _mm_mul_ps(_mm_min_ps(xq4, yq4), _0_28)), eps));

		// a4 <- x < y ? 90 : 0;
		__m128 a4 = _mm_and_ps(xly, _90);
		// a4 <- (y < 0 ? 360 - a4 : a4) == ((x < y ? y < 0 ? 270 : 90) : (y < 0 ? 360 : 0))
		__m128 mask = _mm_cmplt_ps(y4, zero);
		a4 = _mm_or_ps(_mm_and_ps(_mm_sub_ps(_360, a4), mask), _mm_andnot_ps(mask, a4));
		// a4 <- (x < 0 && !(x < y) ? 180 : a4)
		mask = _mm_andnot_ps(xly, _mm_cmplt_ps(x4, zero));
		a4 = _mm_or_ps(_mm_and_ps(_180, mask), _mm_andnot_ps(mask, a4));
		
		// a4 <- (x < y ? a4 - z4 : a4 + z4)
		a4 = _mm_mul_ps(_mm_add_ps(_mm_xor_ps(z4, _mm_andnot_ps(absmask, xly)), a4), scale4);
		__m128 m4 = _mm_sqrt_ps(_mm_add_ps(xq4, yq4));
		_mm_storeu_ps(angle + i, a4);
		_mm_storeu_ps(mag + i, m4);
	}
#endif
	for(; i < len; i++)
	{
		float xf = x[i], yf = y[i];
		float a, x2 = xf * xf, y2 = yf * yf;
		if(y2 <= x2)
			a = xf * yf / (x2 + 0.28f * y2 + (float)1e-6) + (float)(xf < 0 ? CCV_PI : yf >= 0 ? 0 : CCV_PI * 2);
		else
			a = (float)(yf >= 0 ? CCV_PI * 0.5 : CCV_PI * 1.5) - xf * yf / (y2 + 0.28f * x2 + (float)1e-6);
		angle[i] = a * scale;
		mag[i] = sqrt(x2 + y2);
	}
}

void ccv_gradient(ccv_dense_matrix_t* a, ccv_dense_matrix_t** theta, int ttype, ccv_dense_matrix_t** m, int mtype, int dx, int dy)
{
	uint64_t tsig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_gradient_theta", 18, a->sig, 0);
	uint64_t msig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_gradient_m", 14, a->sig, 0);
	ccv_dense_matrix_t* dtheta = *theta = ccv_dense_matrix_renew(*theta, a->rows, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, tsig);
	ccv_dense_matrix_t* dm = *m = ccv_dense_matrix_renew(*m, a->rows, a->cols, CCV_32F | CCV_C1, CCV_32F | CCV_C1, msig);
	if ((dtheta->type & CCV_GARBAGE) && (dm->type & CCV_GARBAGE))
	{
		dtheta->type &= ~CCV_GARBAGE;
		dm->type &= ~CCV_GARBAGE;
		return;
	}
	dtheta->type &= ~CCV_GARBAGE;
	dm->type &= ~CCV_GARBAGE;
	ccv_dense_matrix_t* tx = 0;
	ccv_dense_matrix_t* ty = 0;
	ccv_sobel(a, &tx, CCV_32F | CCV_C1, dx, 0);
	ccv_sobel(a, &ty, CCV_32F | CCV_C1, 0, dy);
	__ccv_atan2(tx->data.fl, ty->data.fl, dtheta->data.fl, dm->data.fl, a->rows * a->cols);
	ccv_matrix_free(tx);
	ccv_matrix_free(ty);
}

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size)
{
	int border_size = size / 2;
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_hog", 7, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows - border_size * 2, (a->cols - border_size * 2) * 8, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	ccv_dense_matrix_t* ag = 0;
	ccv_dense_matrix_t* mg = 0;
	ccv_gradient(a, &ag, 0, &mg, 0, 3, 3);
	int i, j, x, y;
	ag->type = (ag->type & ~CCV_32F) | CCV_32S;
	for (i = 0; i < a->rows * a->cols; i++)
		ag->data.i[i] = ((int)(ag->data.fl[i] / 45 + 0.5)) & 0x7;
	int* agi = ag->data.i;
	float* mgfl = mg->data.fl;
	int* bi = db->data.i;
	for (y = 0; y <= a->rows - size; y++)
	{
		float hog[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		for (i = 0; i < size; i++)
			for (j = 0; j < size; j++)
				hog[agi[i * a->cols + j]] += mgfl[i * a->cols + j];
		for (i = 0; i < 8; i++)
			bi[i] = (int)hog[i];
		bi += 8;
		mgfl++;
		agi++;
		for (x = 1; x <= a->cols - size; x++)
		{
			for (i = 0; i < size; i++)
			{
				hog[agi[i * a->cols - 1]] -= mgfl[i * a->cols - 1];
				hog[agi[i * a->cols - 1 + size]] += mgfl[i * a->cols - 1 + size];
			}
			for (i = 0; i < 8; i++)
				bi[i] = (int)hog[i];
			bi += 8;
			mgfl++;
			agi++;
		}
		agi += border_size * 2;
		mgfl += border_size * 2;
	}
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
}

/* canny detector is adopted from OpenCV */
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double low_thresh, double high_thresh)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_canny(%lf,%lf)", low_thresh, high_thresh);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_32S | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_C1 | CCV_ALL_DATA_TYPE, type, sig);
	ccv_cache_return(db, );
	ccv_dense_matrix_t* dx = 0;
	ccv_dense_matrix_t* dy = 0;
	ccv_sobel(a, &dx, 0, 1, 0);
	ccv_sobel(a, &dy, 0, 0, 1);
	int low = (int)(low_thresh + 0.5);
	int high = (int)(high_thresh + 0.5);
	unsigned char* buffer = (unsigned char*)malloc((a->rows + 2 + 3 * sizeof(int)) * (a->cols + 2));
	int mapstep = a->cols + 2;
	int* mbuf[3];
	mbuf[0] = (int*)buffer;
	mbuf[1] = mbuf[0] + a->cols + 2;
	mbuf[2] = mbuf[1] + a->cols + 2;

	unsigned char* map = (unsigned char*)(mbuf[2] + a->cols + 2);
	memset(mbuf[0], 0, (a->cols + 2) * sizeof(int));
	memset(map, 1, mapstep);
	memset(map + mapstep * (a->rows + 1), 1, mapstep);
	int maxsize = a->rows * a->cols;
	unsigned char** stack = (unsigned char**)malloc(maxsize * sizeof(unsigned char*));
	unsigned char** stack_bottom = stack;
	unsigned char** stack_top = stack;
#define __canny_push(d) *(d) = (unsigned char)2, *stack_top++ = (d)
#define __canny_pop(d) (d) = *--stack_top
	int i, j;
	for (i = 0; i <= a->rows; i++)
	{
		int* mg = mbuf[(i > 0) + 1] + 1;
		const int* _dx = dx->data.i + dx->cols * i;
		const int* _dy = dy->data.i + dy->cols * i;
		int x, y;
		if (i < a->rows)
		{
			mg[-1] = mg[a->cols] = 0;
			for (j = 0; j < a->cols; j++)
				mg[j] = abs(_dx[j]) + abs(_dy[j]);
		} else
			memset(mg - 1, 0, (a->cols + 2) * sizeof(int));
		if (i == 0)
			continue;
		int magstep1 = (int)(mbuf[2] - mbuf[1]);
		int magstep2 = (int)(mbuf[0] - mbuf[1]);
		unsigned char* _map = map + mapstep * i + 1;
		_map[-1] = _map[a->cols] = 1;
		mg = mbuf[1] + 1;
		_dx = dx->data.i + dx->cols * (i - 1);
		_dy = dy->data.i + dy->cols * (i - 1);
		int prev_flag = 0;
		for (j = 0; j < a->cols; j++)
		{
#define CANNY_SHIFT 15
#define TG22 (int)(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) + 0.5)
#define __ccv_high_watermark { if (m > high && !prev_flag && _map[j - mapstep] != 2) \
	{ \
		__canny_push(_map + j); \
		prev_flag = 1; \
	} else \
		_map[j] = 0; \
	continue; }
			x = abs(_dx[j]);
			y = abs(_dy[j]);
			int s = _dx[j] ^ _dy[j];
			int m = mg[j];
			if (m > low)
			{
				int tg22x = x * TG22;
				int tg67x = tg22x + ((x + x) << CANNY_SHIFT);
				y <<= CANNY_SHIFT;
				if (y < tg22x)
				{
					if (m > mg[j - 1] && m >= mg[j + 1])
						__ccv_high_watermark;
				} else if (y > tg67x) {
					if (m > mg[j + magstep2] && m >= mg[j + magstep1])
						__ccv_high_watermark;
				} else {
					s = s < 0 ? - 1 : 1;
					if (m > mg[j + magstep2 - s] && m > mg[j + magstep1 + s])
						__ccv_high_watermark;
				}
			}
			prev_flag = 0;
			_map[j] = 1;
#undef __ccv_high_watermark
#undef TG22
#undef CANNY_SHIFT
		}
		mg = mbuf[0];
		mbuf[0] = mbuf[1];
		mbuf[1] = mbuf[2];
		mbuf[2] = mg;
	}
	int dr[] = {-1, 1, -mapstep - 1, -mapstep, -mapstep + 1, mapstep - 1, mapstep, mapstep + 1};
	while (stack_top > stack_bottom)
	{
		unsigned char* m;
		if ((stack_top - stack_bottom) + 8 > maxsize)
		{
			int sz = (int)(stack_top - stack_bottom);
			maxsize = ccv_max(maxsize * 3 / 2, maxsize + 8);
			stack = (unsigned char**)realloc(stack, maxsize * sizeof(unsigned char*));
			stack_bottom = stack;
			stack_top = stack_bottom + sz;
		}
		__canny_pop(m);
		for (i = 0; i < 8; i++)
			if (!m[dr[i]])
				__canny_push(m + dr[i]);
	}
#undef __canny_pop
#undef __canny_push
	for (i = 0; i < a->rows; i++)
	{
		const unsigned char* _map = map + mapstep * (i + 1) + 1;
		for (j = 0; j < a->cols; j++)
			db->data.i[j + i * db->cols] = (unsigned char)-(_map[j] >> 1);
	}
	free(buffer);
	free(stack);
}

/* area interpolation resample is adopted from OpenCV */

typedef struct {
	int si, di;
	unsigned int alpha;
} ccv_int_alpha;

static void __ccv_resample_area_8u(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	ccv_int_alpha* xofs = (ccv_int_alpha*)alloca(sizeof(ccv_int_alpha) * a->cols * 2);
	int ch = ccv_clamp(CCV_GET_CHANNEL_NUM(a->type), 1, 4);
	double scale_x = (double)a->cols / b->cols;
	double scale_y = (double)a->rows / b->rows;
	// double scale = 1.f / (scale_x * scale_y);
	unsigned int inv_scale_256 = (int)(scale_x * scale_y * 0x10000);
	int dx, dy, sx, sy, i, k;
	for (dx = 0, k = 0; dx < b->cols; dx++)
	{
		double fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
		int sx1 = (int)(fsx1 + 1.0 - 1e-6), sx2 = (int)(fsx2);
		sx1 = ccv_min(sx1, a->cols - 1);
		sx2 = ccv_min(sx2, a->cols - 1);

		if (sx1 > fsx1)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = (sx1 - 1) * ch;
			xofs[k++].alpha = (unsigned int)((sx1 - fsx1) * 0x100);
		}

		for (sx = sx1; sx < sx2; sx++)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx * ch;
			xofs[k++].alpha = 256;
		}

		if (fsx2 - sx2 > 1e-3)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx2 * ch;
			xofs[k++].alpha = (unsigned int)((fsx2 - sx2) * 256);
		}
	}
	int xofs_count = k;
	unsigned int* buf = (unsigned int*)alloca(b->cols * ch * sizeof(unsigned int));
	unsigned int* sum = (unsigned int*)alloca(b->cols * ch * sizeof(unsigned int));
	for (dx = 0; dx < b->cols * ch; dx++)
		buf[dx] = sum[dx] = 0;
	dy = 0;
	for (sy = 0; sy < a->rows; sy++)
	{
		unsigned char* a_ptr = a->data.ptr + a->step * sy;
		for (k = 0; k < xofs_count; k++)
		{
			int dxn = xofs[k].di;
			unsigned int alpha = xofs[k].alpha;
			for (i = 0; i < ch; i++)
				buf[dxn + i] += a_ptr[xofs[k].si + i] * alpha;
		}
		if ((dy + 1) * scale_y <= sy + 1 || sy == a->rows - 1)
		{
			unsigned int beta = (int)(ccv_max(sy + 1 - (dy + 1) * scale_y, 0.f) * 256);
			unsigned int beta1 = 256 - beta;
			unsigned char* b_ptr = b->data.ptr + b->step * dy;
			if (beta <= 0)
			{
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					b_ptr[dx] = ccv_clamp((sum[dx] + buf[dx] * 256) / inv_scale_256, 0, 255);
					sum[dx] = buf[dx] = 0;
				}
			} else {
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					b_ptr[dx] = ccv_clamp((sum[dx] + buf[dx] * beta1) / inv_scale_256, 0, 255);
					sum[dx] = buf[dx] * beta;
					buf[dx] = 0;
				}
			}
			dy++;
		}
		else
		{
			for(dx = 0; dx < b->cols * ch; dx++)
			{
				sum[dx] += buf[dx] * 256;
				buf[dx] = 0;
			}
		}
	}
}

typedef struct {
	int si, di;
	float alpha;
} ccv_decimal_alpha;

static void __ccv_resample_area(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
{
	ccv_decimal_alpha* xofs = (ccv_decimal_alpha*)alloca(sizeof(ccv_decimal_alpha) * a->cols * 2);
	int ch = ccv_clamp(CCV_GET_CHANNEL_NUM(a->type), 1, 4);
	double scale_x = (double)a->cols / b->cols;
	double scale_y = (double)a->rows / b->rows;
	double scale = 1.f / (scale_x * scale_y);
	int dx, dy, sx, sy, i, k;
	for (dx = 0, k = 0; dx < b->cols; dx++)
	{
		double fsx1 = dx * scale_x, fsx2 = fsx1 + scale_x;
		int sx1 = (int)(fsx1 + 1.0 - 1e-6), sx2 = (int)(fsx2);
		sx1 = ccv_min(sx1, a->cols - 1);
		sx2 = ccv_min(sx2, a->cols - 1);

		if (sx1 > fsx1)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = (sx1 - 1) * ch;
			xofs[k++].alpha = (float)((sx1 - fsx1) * scale);
		}

		for (sx = sx1; sx < sx2; sx++)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx * ch;
			xofs[k++].alpha = (float)scale;
		}

		if (fsx2 - sx2 > 1e-3)
		{
			xofs[k].di = dx * ch;
			xofs[k].si = sx2 * ch;
			xofs[k++].alpha = (float)((fsx2 - sx2) * scale);
		}
	}
	int xofs_count = k;
	float* buf = (float*)alloca(b->cols * ch * sizeof(float));
	float* sum = (float*)alloca(b->cols * ch * sizeof(float));
	for (dx = 0; dx < b->cols * ch; dx++)
		buf[dx] = sum[dx] = 0;
	dy = 0;
#define for_block(__for_get, __for_set) \
	for (sy = 0; sy < a->rows; sy++) \
	{ \
		unsigned char* a_ptr = a->data.ptr + a->step * sy; \
		for (k = 0; k < xofs_count; k++) \
		{ \
			int dxn = xofs[k].di; \
			float alpha = xofs[k].alpha; \
			for (i = 0; i < ch; i++) \
				buf[dxn + i] += __for_get(a_ptr, xofs[k].si + i, 0) * alpha; \
		} \
		if ((dy + 1) * scale_y <= sy + 1 || sy == a->rows - 1) \
		{ \
			float beta = ccv_max(sy + 1 - (dy + 1) * scale_y, 0.f); \
			float beta1 = 1 - beta; \
			unsigned char* b_ptr = b->data.ptr + b->step * dy; \
			if (fabs(beta) < 1e-3) \
			{ \
				for (dx = 0; dx < b->cols * ch; dx++) \
				{ \
					__for_set(b_ptr, dx, sum[dx] + buf[dx], 0); \
					sum[dx] = buf[dx] = 0; \
				} \
			} else { \
				for (dx = 0; dx < b->cols * ch; dx++) \
				{ \
					__for_set(b_ptr, dx, sum[dx] + buf[dx] * beta1, 0); \
					sum[dx] = buf[dx] * beta; \
					buf[dx] = 0; \
				} \
			} \
			dy++; \
		} \
		else \
		{ \
			for(dx = 0; dx < b->cols * ch; dx++) \
			{ \
				sum[dx] += buf[dx]; \
				buf[dx] = 0; \
			} \
		} \
	}
	ccv_matrix_getter(a->type, ccv_matrix_setter, b->type, for_block);
#undef for_block
}

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_resample(%d,%d,%d)", rows, cols, type);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	btype = (btype == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(btype) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, rows, cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), btype, sig);
	ccv_cache_return(db, );
	if (a->rows == db->rows && a->cols == db->cols)
	{
		if (CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(db->type) && CCV_GET_DATA_TYPE(db->type) == CCV_GET_DATA_TYPE(a->type))
			memcpy(db->data.ptr, a->data.ptr, a->rows * a->step);
		else {
			/* format convert */
		}
		return;
	}
	switch (type)
	{
		case CCV_INTER_AREA:
			if (a->rows > db->rows && a->cols > db->cols)
			{
				/* using the fast alternative (fix point scale, 0x100 to avoid overflow) */
				if (CCV_GET_DATA_TYPE(a->type) == CCV_8U && CCV_GET_DATA_TYPE(db->type) == CCV_8U && a->rows * a->cols / (db->rows * db->cols) < 0x100)
					__ccv_resample_area_8u(a, db);
				else
					__ccv_resample_area(a, db);
				break;
			}
		case CCV_INTER_LINEAR:
			break;
		case CCV_INTER_CUBIC:
			break;
		case CCV_INTER_LACZOS:
			break;
	}
}

/* the following code is adopted from OpenCV cvPyrDown */
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)
{
	assert(src_x >= 0 && src_y >= 0);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sample_down(%d,%d)", src_x, src_y);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows / 2, a->cols / 2, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int ch = ccv_clamp(CCV_GET_CHANNEL_NUM(a->type), 1, 4);
	int cols0 = db->cols - 1 - src_x;
	int dy, sy = -2 + src_y, sx = src_x * ch, dx, k;
	int* tab = (int*)alloca((a->cols + src_x + 2) * ch * sizeof(int));
	for (dx = 0; dx < a->cols + src_x + 2; dx++)
		for (k = 0; k < ch; k++)
			tab[dx * ch + k] = ((dx >= a->cols) ? a->cols * 2 - 1 - dx : dx) * ch + k;
	unsigned char* buf = (unsigned char*)alloca(5 * db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int)));
	int bufstep = db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int));
	unsigned char* b_ptr = db->data.ptr;
	/* why is src_y * 4 in computing the offset of row?
	 * Essentially, it means sy - src_y but in a manner that doesn't result negative number.
	 * notice that we added src_y before when computing sy in the first place, however,
	 * it is not desirable to have that offset when we try to wrap it into our 5-row buffer (
	 * because in later rearrangement, we have no src_y to backup the arrangement). In
	 * such micro scope, we managed to stripe 5 addition into one shift and addition. */
#define for_block(boundary_x_handler, __for_get_a, __for_get, __for_set, __for_set_b) \
	for (dy = 0; dy < db->rows; dy++) \
	{ \
		for(; sy <= dy * 2 + 2 + src_y; sy++) \
		{ \
			unsigned char* row = buf + ((sy + src_y * 4 + 2) % 5) * bufstep; \
			int _sy = (sy < 0) ? -1 - sy : (sy >= a->rows) ? a->rows * 2 - 1 - sy : sy; \
			unsigned char* a_ptr = a->data.ptr + a->step * _sy; \
			for (k = 0; k < ch; k++) \
				__for_set(row, k, __for_get_a(a_ptr, sx + k, 0) * 10 + __for_get_a(a_ptr, ch + sx + k, 0) * 5 + __for_get_a(a_ptr, 2 * ch + sx + k, 0), 0); \
			for(dx = ch; dx < cols0 * ch; dx += ch) \
				for (k = 0; k < ch; k++) \
					__for_set(row, dx + k, __for_get_a(a_ptr, dx * 2 + sx + k, 0) * 6 + (__for_get_a(a_ptr, dx * 2 + sx + k - ch, 0) + __for_get_a(a_ptr, dx * 2 + sx + k + ch, 0)) * 4 + __for_get_a(a_ptr, dx * 2 + sx + k - ch * 2, 0) + __for_get_a(a_ptr, dx * 2 + sx + k + ch * 2, 0), 0); \
			boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b); \
		} \
		unsigned char* rows[5]; \
		for(k = 0; k < 5; k++) \
			rows[k] = buf + ((dy * 2 + k) % 5) * bufstep; \
		for(dx = 0; dx < db->cols * ch; dx++) \
			__for_set_b(b_ptr, dx, (__for_get(rows[2], dx, 0) * 6 + (__for_get(rows[1], dx, 0) + __for_get(rows[3], dx, 0)) * 4 + __for_get(rows[0], dx, 0) + __for_get(rows[4], dx, 0)) / 256, 0); \
		b_ptr += db->step; \
	}
	int no_8u_type = (a->type & CCV_8U) ? CCV_32S : a->type;
	/* here is the new technique to expand for loop with condition in manual way */
	if (src_x > 0)
	{
#define boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b) \
		for (dx = cols0 * ch; dx < db->cols * ch; dx += ch) \
			for (k = 0; k < ch; k++) \
				__for_set(row, dx + k, __for_get_a(a_ptr, tab[dx * 2 + sx + k], 0) * 6 + (__for_get_a(a_ptr, tab[dx * 2 + sx + k - ch], 0) + __for_get_a(a_ptr, tab[dx * 2 + sx + k + ch], 0)) * 4 + __for_get_a(a_ptr, tab[dx * 2 + sx + k - ch * 2], 0) + __for_get_a(a_ptr, tab[dx * 2 + sx + k + ch * 2], 0), 0);
		ccv_unswitch(boundary_x_handler, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef boundary_x_handler
	} else {
#define boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b) \
		for (k = 0; k < ch; k++) \
			__for_set(row, (db->cols - 1) * ch + k, __for_get_a(a_ptr, a->cols * ch + sx - ch + k, 0) * 10 + __for_get_a(a_ptr, (a->cols - 2) * ch + sx + k, 0) * 5 + __for_get_a(a_ptr, (a->cols - 3) * ch + sx + k, 0), 0);
		ccv_unswitch(boundary_x_handler, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef boundary_x_handler
	}
#undef for_block
}

void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y)
{
	assert(src_x >= 0 && src_y >= 0);
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_sample_up(%d,%d)", src_x, src_y);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows * 2, a->cols * 2, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int ch = ccv_clamp(CCV_GET_CHANNEL_NUM(a->type), 1, 4);
	int cols0 = a->cols - 1 - src_x;
	int y, x, sy = -1 + src_y, sx = src_x * ch, k;
	int* tab = (int*)alloca((a->cols + src_x + 2) * ch * sizeof(int));
	for (x = 0; x < a->cols + src_x + 2; x++)
		for (k = 0; k < ch; k++)
			tab[x * ch + k] = ((x >= a->cols) ? a->cols * 2 - 1 - x : x) * ch + k;
	unsigned char* buf = (unsigned char*)alloca(3 * db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int)));
	int bufstep = db->cols * ch * ccv_max(CCV_GET_DATA_TYPE_SIZE(db->type), sizeof(int));
	unsigned char* b_ptr = db->data.ptr;
	/* why src_y * 2: the same argument as in ccv_sample_down */
#define for_block(boundary_x_handler, __for_get_a, __for_get, __for_set, __for_set_b) \
	for (y = 0; y < a->rows; y++) \
	{ \
		for (; sy <= y + 1 + src_y; sy++) \
		{ \
			unsigned char* row = buf + ((sy + src_y * 2 + 1) % 3) * bufstep; \
			int _sy = (sy < 0) ? -1 - sy : (sy >= a->rows) ? a->rows * 2 - 1 - sy : sy; \
			unsigned char* a_ptr = a->data.ptr + a->step * _sy; \
			if (a->cols == 1) \
			{ \
				for (k = 0; k < ch; k++) \
				{ \
					__for_set(row, k, __for_get_a(a_ptr, k, 0) * 8, 0); \
					__for_set(row, k + ch, __for_get_a(a_ptr, k, 0) * 8, 0); \
				} \
				continue; \
			} \
			for (k = 0; k < ch; k++) \
			{ \
				__for_set(row, k, __for_get_a(a_ptr, k + sx, 0) * 6 + __for_get_a(a_ptr, k + sx + ch, 0) * 2, 0); \
				__for_set(row, k + ch, (__for_get_a(a_ptr, k + sx, 0) + __for_get_a(a_ptr, k + sx + ch, 0)) * 4, 0); \
			} \
			for (x = ch; x < cols0 * ch; x += ch) \
			{ \
				for (k = 0; k < ch; k++) \
				{ \
					__for_set(row, x * 2 + k, __for_get_a(a_ptr, x + sx - ch + k, 0) + __for_get_a(a_ptr, x + sx + k, 0) * 6 + __for_get_a(a_ptr, x + sx + ch + k, 0), 0); \
					__for_set(row, x * 2 + ch + k, (__for_get_a(a_ptr, x + sx + k, 0) + __for_get_a(a_ptr, x + sx + ch + k, 0)) * 4, 0); \
				} \
			} \
			boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b); \
		} \
		unsigned char* rows[3]; \
		for (k = 0; k < 3; k++) \
			rows[k] = buf + ((y + k) % 3) * bufstep; \
		for (x = 0; x < db->cols * ch; x++) \
		{ \
			__for_set_b(b_ptr, x, (__for_get(rows[0], x, 0) + __for_get(rows[1], x, 0) * 2 + __for_get(rows[2], x, 0)) / 32, 0); \
			__for_set_b(b_ptr + db->step, x, (__for_get(rows[1], x, 0) + __for_get(rows[2], x, 0)) / 16, 0); \
		} \
		b_ptr += 2 * db->step; \
	}
	int no_8u_type = (a->type & CCV_8U) ? CCV_32S : a->type;
	/* unswitch if condition in manual way */
	if (src_x > 0)
	{
#define boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b) \
		for (x = cols0 * ch; x < a->cols * ch; x += ch) \
			for (k = 0; k < ch; k++) \
			{ \
				__for_set(row, x * 2 + k, __for_get_a(a_ptr, tab[x + sx - ch + k], 0) + __for_get_a(a_ptr, tab[x + sx + k], 0) * 6 + __for_get_a(a_ptr, tab[x + sx + ch + k], 0), 0); \
				__for_set(row, x * 2 + ch + k, (__for_get_a(a_ptr, tab[x + sx + k], 0) + __for_get_a(a_ptr, tab[x + sx + ch + k], 0)) * 4, 0); \
			}
		ccv_unswitch(boundary_x_handler, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef boundary_x_handler
	} else {
#define boundary_x_handler(__for_get_a, __for_get, __for_set, __for_set_b) \
		for (k = 0; k < ch; k++) \
		{ \
			__for_set(row, (a->cols - 1) * 2 * ch + k, __for_get_a(a_ptr, (a->cols - 2) * ch + k, 0) + __for_get_a(a_ptr, (a->cols - 1) * ch + k, 0) * 7, 0); \
			__for_set(row, (a->cols - 1) * 2 * ch + ch + k, __for_get_a(a_ptr, (a->cols - 1) * ch + k, 0) * 4, 0); \
		}
		ccv_unswitch(boundary_x_handler, ccv_matrix_getter_a, a->type, ccv_matrix_getter, no_8u_type, ccv_matrix_setter, no_8u_type, ccv_matrix_setter_b, db->type, for_block);
#undef boundary_x_handler
	}
#undef for_block
}

void __ccv_flip_y_self(ccv_dense_matrix_t* a)
{
	int i;
	unsigned char* buffer = (unsigned char*)alloca(a->step);
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = a->data.ptr + (a->rows - 1) * a->step;
	for (i = 0; i < a->rows / 2; i++)
	{
		memcpy(buffer, a_ptr, a->step);
		memcpy(a_ptr, b_ptr, a->step);
		memcpy(b_ptr, buffer, a->step);
		a_ptr += a->step;
		b_ptr -= a->step;
	}
}

void __ccv_flip_x_self(ccv_dense_matrix_t* a)
{
	int i, j;
	int len = CCV_GET_DATA_TYPE_SIZE(a->type) * CCV_GET_CHANNEL_NUM(a->type);
	unsigned char* buffer = (unsigned char*)alloca(len);
	unsigned char* a_ptr = a->data.ptr;
	for (i = 0; i < a->rows; i++)
	{
		for (j = 0; j < a->cols / 2; j++)
		{
			memcpy(buffer, a_ptr + j * len, len);
			memcpy(a_ptr + j * len, a_ptr + (a->cols - 1 - j) * len, len);
			memcpy(a_ptr + (a->cols - 1 - j) * len, buffer, len);
		}
		a_ptr += a->step;
	}
}

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int type)
{
	uint64_t sig = a->sig;
	if (type & CCV_FLIP_Y)
		sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_flip_y", 10, sig, 0);
	if (type & CCV_FLIP_X)
		sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature("ccv_flip_x", 10, sig, 0);
	ccv_dense_matrix_t* db;
	if (b == 0)
	{
		db = a;
		if (!(a->sig == 0))
		{
			btype = CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type);
			sig = ccv_matrix_generate_signature((const char*)&btype, sizeof(int), sig, 0);
			a->sig = sig;
		}
	} else {
		btype = CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type);
		*b = db = ccv_dense_matrix_renew(*b, a->rows, a->cols, btype, btype, sig);
		ccv_cache_return(db, );
		memcpy(db->data.ptr, a->data.ptr, a->rows * a->step);
	}
	if (type & CCV_FLIP_Y)
		__ccv_flip_y_self(db);
	if (type & CCV_FLIP_X)
		__ccv_flip_x_self(db);
}

void ccv_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double sigma)
{
	char identifier[64];
	memset(identifier, 0, 64);
	snprintf(identifier, 64, "ccv_blur(%lf)", sigma);
	uint64_t sig = (a->sig == 0) ? 0 : ccv_matrix_generate_signature(identifier, 64, a->sig, 0);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_cache_return(db, );
	int fsz = ccv_max(1, (int)(4.0 * sigma + 1.0 - 1e-8)) * 2 + 1;
	int hfz = fsz / 2;
	unsigned char* buf = (unsigned char*)alloca(sizeof(double) * (fsz + ccv_max(a->rows, a->cols * CCV_GET_CHANNEL_NUM(a->type))));
	unsigned char* filter = (unsigned char*)alloca(sizeof(double) * fsz);
	double tw = 0;
	int i, j, k, ch = CCV_GET_CHANNEL_NUM(a->type);
	for (i = 0; i < fsz; i++)
		tw += ((double*)filter)[i] = exp(-((i - hfz) * (i - hfz)) / (2.0 * sigma * sigma));
	int no_8u_type = (db->type & CCV_8U) ? CCV_32S : db->type;
	if (no_8u_type & CCV_32S)
	{
		tw = 256.0 / tw;
		for (i = 0; i < fsz; i++)
			((int*)filter)[i] = (int)(((double*)filter)[i] * tw + 0.5);
	} else {
		tw = 1.0 / tw;
		for (i = 0; i < fsz; i++)
			ccv_set_value(db->type, filter, i, ((double*)filter)[i] * tw, 0);
	}
	/* horizontal */
	unsigned char* aptr = a->data.ptr;
	unsigned char* bptr = db->data.ptr;
#define for_block(__for_type, __for_set_b, __for_get_b, __for_set_a, __for_get_a) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < hfz; j++) \
			for (k = 0; k < ch; k++) \
				__for_set_b(buf, j * ch + k, __for_get_a(aptr, k, 0), 0); \
		for (j = 0; j < a->cols * ch; j++) \
			__for_set_b(buf, j + hfz * ch, __for_get_a(aptr, j, 0), 0); \
		for (j = a->cols; j < hfz + a->cols; j++) \
			for (k = 0; k < ch; k++) \
				__for_set_b(buf, j * ch + hfz * ch + k, __for_get_a(aptr, (a->cols - 1) * ch + k, 0), 0); \
		for (j = 0; j < a->cols * ch; j++) \
		{ \
			__for_type sum = 0; \
			for (k = 0; k < fsz; k++) \
				sum += __for_get_b(buf, k * ch + j, 0) * __for_get_b(filter, k, 0); \
			__for_set_b(buf, j, sum, 8); \
		} \
		for (j = 0; j < a->cols * ch; j++) \
			__for_set_a(bptr, j, __for_get_b(buf, j, 0), 0); \
		aptr += a->step; \
		bptr += db->step; \
	}
	ccv_matrix_typeof_setter_getter(no_8u_type, ccv_matrix_setter, db->type, ccv_matrix_getter, a->type, for_block);
#undef for_block
	/* vertical */
	bptr = db->data.ptr;
#define for_block(__for_type, __for_set_b, __for_get_b, __for_set_a, __for_get_a) \
	for (i = 0; i < a->cols * ch; i++) \
	{ \
		for (j = 0; j < hfz; j++) \
			__for_set_b(buf, j, __for_get_a(bptr, i, 0), 0); \
		for (j = 0; j < a->rows; j++) \
			__for_set_b(buf, j + hfz, __for_get_a(bptr + j * db->step, i, 0), 0); \
		for (j = a->rows; j < hfz + a->rows; j++) \
			__for_set_b(buf, j + hfz, __for_get_a(bptr + (a->rows - 1) * db->step, i, 0), 0); \
		for (j = 0; j < a->rows; j++) \
		{ \
			__for_type sum = 0; \
			for (k = 0; k < fsz; k++) \
				sum += __for_get_b(buf, k + j, 0) * __for_get_b(filter, k, 0); \
			__for_set_b(buf, j, sum, 8); \
		} \
		for (j = 0; j < a->rows; j++) \
			__for_set_a(bptr + j * db->step, i, __for_get_b(buf, j, 0), 0); \
	}
	ccv_matrix_typeof_setter_getter(no_8u_type, ccv_matrix_setter_getter, db->type, for_block);
#undef for_block
}
