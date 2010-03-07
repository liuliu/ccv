#include "ccv.h"

void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int dx, int dy)
{
	int sig[5];
	char identifier[64];
	memset(identifier, 0, 64);
	sprintf(identifier, "ccv_sobel(%d,%d)", dx, dy);
	ccv_matrix_generate_signature(identifier, 64, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(a->rows, a->cols, CCV_32S | CCV_C1, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
	}
	int i, j;
	unsigned char* a_ptr = a->data.ptr;
	unsigned char* b_ptr = db->data.ptr;
	ccv_dense_matrix_t* c = ccv_dense_matrix_new(a->rows, a->cols, CCV_32S | CCV_C1, NULL, NULL);
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
		for (j = 0; j < c->cols; j++)
			ccv_set_value(db->type, b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j]);
		b_ptr += db->step;
		c_ptr += c->cols;
		for (i = 1; i < c->rows - 1; i++)
		{
			for (j = 0; j < c->cols; j++)
				ccv_set_value(db->type, b_ptr, j, c_ptr[j + c->cols] + 2 * c_ptr[j] + c_ptr[j - c->cols]);
			b_ptr += db->step;
			c_ptr += c->cols;
		}
		for (j = 0; j < c->cols; j++)
			ccv_set_value(db->type, b_ptr, j, 2 * c_ptr[j] + c_ptr[j - c->cols]);
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
		for (i = 0; i < c->rows; i++)
		{
			ccv_set_value(db->type, b_ptr, 0, c_ptr[1] + 2 * c_ptr[0]);
			for (j = 1; j < c->cols - 1; j++)
				ccv_set_value(db->type, b_ptr, j, c_ptr[j + 1] + 2 * c_ptr[j] + c_ptr[j - 1]);
			ccv_set_value(db->type, b_ptr, c->cols - 1, c_ptr[c->cols - 1] + 2 * c_ptr[c->cols - 2]);
			b_ptr += db->step;
			c_ptr += c->cols;
		}
	}
	ccv_matrix_free(c);
}

/* the fast arctan function adopted from OpenCV */
void __ccv_atan2(float* x, float* y, float* angle, float* mag, int len)
{
	int i = 0;
	float scale = (float)(180 / 3.141592654);

	static const int iabsmask[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
	__m128 eps = _mm_set1_ps((float)1e-6), absmask = _mm_load_ps((const float*)iabsmask);
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
	
	for(; i < len; i++)
	{
		float xf = x[i], yf = y[i];
		float a, x2 = xf * xf, y2 = yf * yf;
		if(y2 <= x2)
			a = xf * yf / (x2 + 0.28f * y2 + (float)1e-6) + (float)(xf < 0 ? 3.141592654 : yf >= 0 ? 0 : 3.141592654 * 2);
		else
			a = (float)(yf >= 0 ? 3.141592654 * 0.5 : 3.141592654 * 1.5) - xf * yf / (y2 + 0.28f * x2 + (float)1e-6);
		angle[i] = a * scale;
		mag[i] = sqrt(x2 + y2);
	}
}

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int size)
{
	int border_size = size / 2;
	int sig[5];
	ccv_matrix_generate_signature("ccv_hog", 7, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(a->rows - border_size * 2, (a->cols - border_size * 2) * 8, CCV_32S | CCV_C1, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
	}
	ccv_dense_matrix_t* dx = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, NULL, sig);
	ccv_dense_matrix_t* dy = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, NULL, sig);
	ccv_sobel(a, &dx, 1, 0);
	ccv_sobel(a, &dy, 0, 1);
	ccv_dense_matrix_t* ag = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, NULL, sig);
	ccv_dense_matrix_t* mg = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, NULL, sig);
	__ccv_atan2(dx->data.fl, dy->data.fl, ag->data.fl, mg->data.fl, a->rows * a->cols);
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
	ccv_matrix_free(dx);
	ccv_matrix_free(dy);
	ccv_matrix_free(ag);
	ccv_matrix_free(mg);
}

/* area interpolation resample is adopted from OpenCV */

typedef struct {
	int si, di;
	float alpha;
} ccv_decimal_alpha;

void __ccv_resample_area(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b)
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
	for (sy = 0; sy < a->rows; sy++)
	{
		unsigned char* a_ptr = a->data.ptr + a->step * sy;
		for (k = 0; k < xofs_count; k++)
		{
			int dxn = xofs[k].di;
			float alpha = xofs[k].alpha;
			for (i = 0; i < ch; i++)
				buf[dxn + i] += ccv_get_value(a->type, a_ptr, xofs[k].si + i) * alpha;
		}
		if ((dy + 1) * scale_y <= sy + 1 || sy == a->rows - 1)
		{
			float beta = ccv_max(sy + 1 - (dy + 1) * scale_y, 0.f);
			float beta1 = 1 - beta;
			unsigned char* b_ptr = b->data.ptr + b->step * dy;
			if (fabs(beta) < 1e-3)
			{
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					ccv_set_value(b->type, b_ptr, dx, sum[dx] + buf[dx]);
					sum[dx] = buf[dx] = 0;
				}
			} else {
				for (dx = 0; dx < b->cols * ch; dx++)
				{
					ccv_set_value(b->type, b_ptr, dx, sum[dx] + buf[dx] * beta1);
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
				sum[dx] += buf[dx];
				buf[dx] = 0;
			}
		}
	}
}

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int rows, int cols, int type)
{
	int sig[5];
	char identifier[64];
	memset(identifier, 0, 64);
	sprintf(identifier, "ccv_resample(%d,%d,%d)", rows, cols, type);
	ccv_matrix_generate_signature(identifier, 64, sig, a->sig, NULL);
	ccv_dense_matrix_t* db;
	if (*b == NULL)
	{
		*b = db = ccv_dense_matrix_new(rows, cols, a->type, NULL, sig);
		if (db->type & CCV_GARBAGE)
		{
			db->type &= ~CCV_GARBAGE;
			return;
		}
	} else {
		db = *b;
		assert(db->rows == rows && db->cols == cols && db->type == a->type);
	}
	switch (type)
	{
		case CCV_INTER_AREA:
			if (a->rows > db->rows && a->cols > db->cols)
			{
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

void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
}

void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b)
{
}
