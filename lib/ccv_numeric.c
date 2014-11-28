#include "ccv.h"
#include "ccv_internal.h"
#include <complex.h>
#ifdef HAVE_FFTW3
#include <pthread.h>
#include <fftw3.h>
#else
#include "3rdparty/kissfft/kiss_fftndr.h"
#include "3rdparty/kissfft/kissf_fftndr.h"
#endif

const ccv_minimize_param_t ccv_minimize_default_params = {
	.interp = 0.1,
	.extrap = 3.0,
	.max_iter = 20,
	.ratio = 10.0,
	.sig = 0.1,
	.rho = 0.05,
};

void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b, int type)
{
}

void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type)
{
}

void ccv_eigen(ccv_dense_matrix_t* a, ccv_dense_matrix_t** vector, ccv_dense_matrix_t** lambda, int type, double epsilon)
{
	ccv_declare_derived_signature(vsig, a->sig != 0, ccv_sign_with_literal("ccv_eigen(vector)"), a->sig, CCV_EOF_SIGN);
	ccv_declare_derived_signature(lsig, a->sig != 0, ccv_sign_with_literal("ccv_eigen(lambda)"), a->sig, CCV_EOF_SIGN);
	assert(CCV_GET_CHANNEL(a->type) == 1);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_C1 : CCV_GET_DATA_TYPE(type) | CCV_C1;
	// as of now, this function only support real symmetric matrix
	ccv_dense_matrix_t* dvector = *vector = ccv_dense_matrix_renew(*vector, a->rows, a->cols, CCV_32F | CCV_64F | CCV_C1, type, vsig);
	ccv_dense_matrix_t* dlambda = *lambda = ccv_dense_matrix_renew(*lambda, 1, a->cols, CCV_32F | CCV_64F | CCV_C1, type, lsig);
	assert(CCV_GET_DATA_TYPE(dvector->type) == CCV_GET_DATA_TYPE(dlambda->type));
	ccv_object_return_if_cached(, dvector, dlambda);
	double* ja = (double*)ccmalloc(sizeof(double) * a->rows * a->cols);
	int i, j;
	unsigned char* aptr = a->data.u8;
	assert(a->rows > 0 && a->cols > 0);
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			ja[i * a->cols + j] = _for_get(aptr, j, 0); \
		aptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	ccv_zero(dvector);
	ccv_zero(dlambda);
	unsigned char* dvptr = dvector->data.u8;
#define for_block(_, _for_set) \
	for (i = 0; i < a->cols; i++) \
		_for_set(dvptr, i * a->cols + i, 1, 0);
	ccv_matrix_setter(dvector->type, for_block);
#undef for_block
	double accuracy = 0;
	for (i = 0; i < a->rows * a->cols; i++)
		accuracy += ja[i];
	accuracy = sqrt(2 * accuracy);
	int p, q;
	unsigned char* dlptr = dlambda->data.u8;
	int flag = 1;
	assert(a->rows == a->cols);
#define for_block(_, _for_set, _for_get) \
	do { \
		if (!flag) \
			accuracy = accuracy * 0.5; \
		flag = 0; \
		for (p = 0; p < a->rows; p++) \
		{ \
			for (q = p + 1; q < a->cols; q++) \
				if (fabs(ja[p * a->cols + q]) > accuracy) \
				{ \
					double x = -ja[p * a->cols + q]; \
					double y = (ja[q * a->cols + q] - ja[p * a->cols + p]) * 0.5; \
					double omega = (x == 0 && y == 0) ? 1 : x / sqrt(x * x + y * y); \
					if (y < 0) \
						omega = -omega; \
					double sn = 1.0 + sqrt(1.0 - omega * omega); \
					sn = omega / sqrt(2 * sn); \
					double cn = sqrt(1.0 - sn * sn); \
					double fpp = ja[p * a->cols + p]; \
					double fpq = ja[p * a->cols + q]; \
					double fqq = ja[q * a->cols + q]; \
					ja[p * a->cols + p] = fpp * cn * cn + fqq * sn * sn + fpq * omega; \
					ja[q * a->cols + q] = fpp * sn * sn + fqq * cn * cn - fpq * omega; \
					ja[p * a->cols + q] = ja[q * a->cols + p] = 0; \
					for (i = 0; i < a->cols; i++) \
						if (i != q && i != p) \
						{ \
							fpp = ja[p * a->cols + i]; \
							fqq = ja[q * a->cols + i]; \
							ja[p * a->cols + i] = fpp * cn + fqq * sn; \
							ja[q * a->cols + i] = -fpp * sn + fqq * cn; \
						} \
					for (i = 0; i < a->rows; i++) \
						if (i != q && i != p) \
						{ \
							fpp = ja[i * a->cols + p]; \
							fqq = ja[i * a->cols + q]; \
							ja[i * a->cols + p] = fpp * cn + fqq * sn; \
							ja[i * a->cols + q] = -fpp * sn + fqq * cn; \
						} \
					for (i = 0; i < a->cols; i++) \
					{ \
						fpp = _for_get(dvptr, p * a->cols + i, 0); \
						fqq = _for_get(dvptr, q * a->cols + i, 0); \
						_for_set(dvptr, p * a->cols + i, fpp * cn + fqq * sn, 0); \
						_for_set(dvptr, q * a->cols + i, -fpp * sn + fqq * cn, 0); \
					} \
					for (i = 0; i < a->cols; i++) \
						_for_set(dlptr, i, ja[i * a->cols + i], 0); \
					flag = 1; \
					break; \
				} \
			if (flag) \
				break; \
		} \
	} while (accuracy > epsilon);
	ccv_matrix_setter_getter(dvector->type, for_block);
#undef for_block
	ccfree(ja);
}

void ccv_minimize(ccv_dense_matrix_t* x, int length, double red, ccv_minimize_f func, ccv_minimize_param_t params, void* data)
{
	ccv_dense_matrix_t* df0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(df0);
	ccv_dense_matrix_t* df3 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(df3);
	ccv_dense_matrix_t* dF0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(dF0);
	ccv_dense_matrix_t* s = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(s);
	ccv_dense_matrix_t* x0 = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(x0);
	ccv_dense_matrix_t* xn = ccv_dense_matrix_new(x->rows, x->cols, x->type, 0, 0);
	ccv_zero(xn);
	
	double F0 = 0, f0 = 0, f1 = 0, f2 = 0, f3 = 0, f4 = 0;
	double x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	double d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;
	double A = 0, B = 0;
	int ls_failed = 0;

	int i, j, k;
	func(x, &f0, df0, data);
	d0 = 0;
	unsigned char* df0p = df0->data.u8;
	unsigned char* sp = s->data.u8;
	for (i = 0; i < x->rows; i++)
	{
		for (j = 0; j < x->cols; j++)
		{
			double ss = ccv_get_value(x->type, df0p, j);
			ccv_set_value(x->type, sp, j, -ss, 0);
			d0 += -ss * ss;
		}
		df0p += x->step;
		sp += x->step;
	}
	x3 = red / (1.0 - d0);
	int l = (length > 0) ? length : -length;
	int ls = (length > 0) ? 1 : 0;
	int eh = (length > 0) ? 0 : 1;
	for (k = 0; k < l;)
	{
		k += ls;
		memcpy(x0->data.u8, x->data.u8, x->rows * x->step);
		memcpy(dF0->data.u8, df0->data.u8, x->rows * x->step);
		F0 = f0;
		int m = ccv_min(params.max_iter, (length > 0) ? params.max_iter : l - k);
		for (;;)
		{
			x2 = 0;
			f3 = f2 = f0;
			d2 = d0;
			memcpy(df3->data.u8, df0->data.u8, x->rows * x->step);
			while (m > 0)
			{
				m--;
				k += eh;
				unsigned char* sp = s->data.u8;
				unsigned char* xp = x->data.u8;
				unsigned char* xnp = xn->data.u8;
				for (i = 0; i < x->rows; i++)
				{
					for (j = 0; j < x->cols; j++)
						ccv_set_value(x->type, xnp, j, x3 * ccv_get_value(x->type, sp, j) + ccv_get_value(x->type, xp, j), 0);
					sp += x->step;
					xp += x->step;
					xnp += x->step;
				}
				if (func(xn, &f3, df3, data) == 0)
					break;
				else
					x3 = (x2 + x3) * 0.5;
			}
			if (f3 < F0)
			{
				memcpy(x0->data.u8, xn->data.u8, x->rows * x->step);
				memcpy(dF0->data.u8, df3->data.u8, x->rows * x->step);
				F0 = f3;
			}
			d3 = 0;
			unsigned char* df3p = df3->data.u8;
			unsigned char* sp = s->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					d3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, sp, j);
				df3p += x->step;
				sp += x->step;
			}
			if ((d3 > params.sig * d0) || (f3 > f0 + x3 * params.rho * d0) || (m <= 0))
				break;
			x1 = x2; f1 = f2; d1 = d2;
			x2 = x3; f2 = f3; d2 = d3;
			A = 6.0 * (f1 - f2) + 3.0 * (d2 + d1) * (x2 - x1);
			B = 3.0 * (f2 - f1) - (2.0 * d1 + d2) * (x2 - x1);
			x3 = B * B - A * d1 * (x2 - x1);
			if (x3 < 0)
				x3 = x2 * params.extrap;
			else {
				x3 = x1 - d1 * (x2 - x1) * (x2 - x1) / (B + sqrt(x3));
				if (x3 < 0)
					x3 = x2 * params.extrap;
				else {
					if (x3 > x2 * params.extrap)
						x3 = x2 * params.extrap;
					else if (x3 < x2 + params.interp * (x2 - x1))
						x3 = x2 + params.interp * (x2 - x1);
				}
			}
		}
		while (((fabs(d3) > -params.sig * d0) || (f3 > f0 + x3 * params.rho * d0)) && (m > 0))
		{
			if ((d3 > 1e-8) || (f3 > f0 + x3 * params.rho * d0))
			{
				x4 = x3; f4 = f3; d4 = d3;
			} else {
				x2 = x3; f2 = f3; d2 = d3;
			}
			if (f4 > f0)
				x3 = x2 - (0.5 * d2 * (x4 - x2) * (x4 - x2)) / (f4 - f2 - d2 * (x4 - x2));
			else {
				A = 6.0 * (f2 - f4) / (x4 - x2) + 3.0 * (d4 + d2);
				B = 3.0 * (f4 - f2) - (2.0 * d2 + d4) * (x4 - x2);
				x3 = B * B - A * d2 * (x4 - x2) * (x4 - x2);
				x3 = (x3 < 0) ? (x2 + x4) * 0.5 : x2 + (sqrt(x3) - B) / A;
			}
			x3 = ccv_max(ccv_min(x3, x4 - params.interp * (x4 - x2)), x2 + params.interp * (x4 - x2));
			sp = s->data.u8;
			unsigned char* xp = x->data.u8;
			unsigned char* xnp = xn->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					ccv_set_value(x->type, xnp, j, x3 * ccv_get_value(x->type, sp, j) + ccv_get_value(x->type, xp, j), 0);
				sp += x->step;
				xp += x->step;
				xnp += x->step;
			}
			func(xn, &f3, df3, data);
			if (f3 < F0)
			{
				memcpy(x0->data.u8, xn->data.u8, x->rows * x->step);
				memcpy(dF0->data.u8, df3->data.u8, x->rows * x->step);
				F0 = f3;
			}
			m--;
			k += eh;
			d3 = 0;
			sp = s->data.u8;
			unsigned char* df3p = df3->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					d3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, sp, j);
				df3p += x->step;
				sp += x->step;
			}
		}
		if ((fabs(d3) < -params.sig * d0) && (f3 < f0 + x3 * params.rho * d0))
		{
			memcpy(x->data.u8, xn->data.u8, x->rows * x->step);
			f0 = f3;
			double df0_df3 = 0;
			double df3_df3 = 0;
			double df0_df0 = 0;
			unsigned char* df0p = df0->data.u8;
			unsigned char* df3p = df3->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					df0_df0 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, df0p, j);
					df0_df3 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, df3p, j);
					df3_df3 += ccv_get_value(x->type, df3p, j) * ccv_get_value(x->type, df3p, j);
				}
				df0p += x->step;
				df3p += x->step;
			}
			double slr = (df3_df3 - df0_df3) / df0_df0;
			df3p = df3->data.u8;
			unsigned char* sp = s->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
					ccv_set_value(x->type, sp, j, slr * ccv_get_value(x->type, sp, j) - ccv_get_value(x->type, df3p, j), 0);
				df3p += x->step;
				sp += x->step;
			}
			memcpy(df0->data.u8, df3->data.u8, x->rows * x->step);
			d3 = d0;
			d0 = 0;
			df0p = df0->data.u8;
			sp = s->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					d0 += ccv_get_value(x->type, df0p, j) * ccv_get_value(x->type, sp, j);
				}
				df0p += x->step;
				sp += x->step;
			}
			if (d0 > 0)
			{
				d0 = 0;
				df0p = df0->data.u8;
				sp = s->data.u8;
				for (i = 0; i < x->rows; i++)
				{
					for (j = 0; j < x->cols; j++)
					{
						double ss = ccv_get_value(x->type, df0p, j);
						ccv_set_value(x->type, sp, j, -ss, 0);
						d0 += -ss * ss;
					}
					df0p += x->step;
					sp += x->step;
				}
			}
			x3 = x3 * ccv_min(params.ratio, d3 / (d0 - 1e-8));
			ls_failed = 0;
		} else {
			memcpy(x->data.u8, x0->data.u8, x->rows * x->step);
			memcpy(df0->data.u8, dF0->data.u8, x->rows * x->step);
			f0 = F0;
			if (ls_failed)
				break;
			d0 = 0;
			unsigned char* df0p = df0->data.u8;
			unsigned char* sp = s->data.u8;
			for (i = 0; i < x->rows; i++)
			{
				for (j = 0; j < x->cols; j++)
				{
					double ss = ccv_get_value(x->type, df0p, j);
					ccv_set_value(x->type, sp, j, -ss, 0);
					d0 += -ss * ss;
				}
				df0p += x->step;
				sp += x->step;
			}
			x3 = red / (1.0 - d0);
			ls_failed = 1;
		}
	}
	ccv_matrix_free(s);
	ccv_matrix_free(x0);
	ccv_matrix_free(xn);
	ccv_matrix_free(dF0);
	ccv_matrix_free(df0);
	ccv_matrix_free(df3);
}

#ifdef HAVE_FFTW3
/* optimal FFT size table is adopted from OpenCV */
static const int _ccv_optimal_fft_size[] = {
	1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 
	50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 
	162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 
	384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 
	729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200, 
	1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875, 
	1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592, 
	2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840, 
	3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400, 
	5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290, 
	7500, 7680, 7776, 8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000, 
	10125, 10240, 10368, 10800, 10935, 11250, 11520, 11664, 12000, 12150, 12288, 12500, 
	12800, 12960, 13122, 13500, 13824, 14400, 14580, 15000, 15360, 15552, 15625, 16000, 
	16200, 16384, 16875, 17280, 17496, 18000, 18225, 18432, 18750, 19200, 19440, 19683, 
	20000, 20250, 20480, 20736, 21600, 21870, 22500, 23040, 23328, 24000, 24300, 24576, 
	25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800, 29160, 30000, 30375, 30720, 
	31104, 31250, 32000, 32400, 32768, 32805, 33750, 34560, 34992, 36000, 36450, 36864, 
	37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740, 45000, 46080, 
	46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000, 54675, 
	55296, 56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800, 
	65536, 65610, 67500, 69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760, 78125, 
	78732, 80000, 81000, 81920, 82944, 84375, 86400, 87480, 90000, 91125, 92160, 93312, 
	93750, 96000, 97200, 98304, 98415, 100000, 101250, 102400, 103680, 104976, 108000, 
	109350, 110592, 112500, 115200, 116640, 118098, 120000, 121500, 122880, 124416, 125000, 
	128000, 129600, 131072, 131220, 135000, 138240, 139968, 140625, 144000, 145800, 147456, 
	150000, 151875, 153600, 155520, 156250, 157464, 160000, 162000, 163840, 164025, 165888, 
	168750, 172800, 174960, 177147, 180000, 182250, 184320, 186624, 187500, 192000, 194400, 
	196608, 196830, 200000, 202500, 204800, 207360, 209952, 216000, 218700, 221184, 225000, 
	230400, 233280, 234375, 236196, 240000, 243000, 245760, 248832, 250000, 253125, 256000, 
	259200, 262144, 262440, 270000, 273375, 276480, 279936, 281250, 288000, 291600, 294912, 
	295245, 300000, 303750, 307200, 311040, 312500, 314928, 320000, 324000, 327680, 328050, 
	331776, 337500, 345600, 349920, 354294, 360000, 364500, 368640, 373248, 375000, 384000, 
	388800, 390625, 393216, 393660, 400000, 405000, 409600, 414720, 419904, 421875, 432000, 
	437400, 442368, 450000, 455625, 460800, 466560, 468750, 472392, 480000, 486000, 491520, 
	492075, 497664, 500000, 506250, 512000, 518400, 524288, 524880, 531441, 540000, 546750, 
	552960, 559872, 562500, 576000, 583200, 589824, 590490, 600000, 607500, 614400, 622080, 
	625000, 629856, 640000, 648000, 655360, 656100, 663552, 675000, 691200, 699840, 703125, 
	708588, 720000, 729000, 737280, 746496, 750000, 759375, 768000, 777600, 781250, 786432, 
	787320, 800000, 810000, 819200, 820125, 829440, 839808, 843750, 864000, 874800, 884736, 
	885735, 900000, 911250, 921600, 933120, 937500, 944784, 960000, 972000, 983040, 984150, 
	995328, 1000000, 1012500, 1024000, 1036800, 1048576, 1049760, 1062882, 1080000, 1093500, 
	1105920, 1119744, 1125000, 1152000, 1166400, 1171875, 1179648, 1180980, 1200000, 
	1215000, 1228800, 1244160, 1250000, 1259712, 1265625, 1280000, 1296000, 1310720, 
	1312200, 1327104, 1350000, 1366875, 1382400, 1399680, 1406250, 1417176, 1440000, 
	1458000, 1474560, 1476225, 1492992, 1500000, 1518750, 1536000, 1555200, 1562500, 
	1572864, 1574640, 1594323, 1600000, 1620000, 1638400, 1640250, 1658880, 1679616, 
	1687500, 1728000, 1749600, 1769472, 1771470, 1800000, 1822500, 1843200, 1866240, 
	1875000, 1889568, 1920000, 1944000, 1953125, 1966080, 1968300, 1990656, 2000000, 
	2025000, 2048000, 2073600, 2097152, 2099520, 2109375, 2125764, 2160000, 2187000, 
	2211840, 2239488, 2250000, 2278125, 2304000, 2332800, 2343750, 2359296, 2361960, 
	2400000, 2430000, 2457600, 2460375, 2488320, 2500000, 2519424, 2531250, 2560000, 
	2592000, 2621440, 2624400, 2654208, 2657205, 2700000, 2733750, 2764800, 2799360, 
	2812500, 2834352, 2880000, 2916000, 2949120, 2952450, 2985984, 3000000, 3037500, 
	3072000, 3110400, 3125000, 3145728, 3149280, 3188646, 3200000, 3240000, 3276800, 
	3280500, 3317760, 3359232, 3375000, 3456000, 3499200, 3515625, 3538944, 3542940, 
	3600000, 3645000, 3686400, 3732480, 3750000, 3779136, 3796875, 3840000, 3888000, 
	3906250, 3932160, 3936600, 3981312, 4000000, 4050000, 4096000, 4100625, 4147200, 
	4194304, 4199040, 4218750, 4251528, 4320000, 4374000, 4423680, 4428675, 4478976, 
	4500000, 4556250, 4608000, 4665600, 4687500, 4718592, 4723920, 4782969, 4800000, 
	4860000, 4915200, 4920750, 4976640, 5000000, 5038848, 5062500, 5120000, 5184000, 
	5242880, 5248800, 5308416, 5314410, 5400000, 5467500, 5529600, 5598720, 5625000, 
	5668704, 5760000, 5832000, 5859375, 5898240, 5904900, 5971968, 6000000, 6075000, 
	6144000, 6220800, 6250000, 6291456, 6298560, 6328125, 6377292, 6400000, 6480000, 
	6553600, 6561000, 6635520, 6718464, 6750000, 6834375, 6912000, 6998400, 7031250, 
	7077888, 7085880, 7200000, 7290000, 7372800, 7381125, 7464960, 7500000, 7558272, 
	7593750, 7680000, 7776000, 7812500, 7864320, 7873200, 7962624, 7971615, 8000000, 
	8100000, 8192000, 8201250, 8294400, 8388608, 8398080, 8437500, 8503056, 8640000, 
	8748000, 8847360, 8857350, 8957952, 9000000, 9112500, 9216000, 9331200, 9375000, 
	9437184, 9447840, 9565938, 9600000, 9720000, 9765625, 9830400, 9841500, 9953280, 
	10000000, 10077696, 10125000, 10240000, 10368000, 10485760, 10497600, 10546875, 10616832, 
	10628820, 10800000, 10935000, 11059200, 11197440, 11250000, 11337408, 11390625, 11520000, 
	11664000, 11718750, 11796480, 11809800, 11943936, 12000000, 12150000, 12288000, 12301875, 
	12441600, 12500000, 12582912, 12597120, 12656250, 12754584, 12800000, 12960000, 13107200, 
	13122000, 13271040, 13286025, 13436928, 13500000, 13668750, 13824000, 13996800, 14062500, 
	14155776, 14171760, 14400000, 14580000, 14745600, 14762250, 14929920, 15000000, 15116544, 
	15187500, 15360000, 15552000, 15625000, 15728640, 15746400, 15925248, 15943230, 16000000, 
	16200000, 16384000, 16402500, 16588800, 16777216, 16796160, 16875000, 17006112, 17280000, 
	17496000, 17578125, 17694720, 17714700, 17915904, 18000000, 18225000, 18432000, 18662400, 
	18750000, 18874368, 18895680, 18984375, 19131876, 19200000, 19440000, 19531250, 19660800, 
	19683000, 19906560, 20000000, 20155392, 20250000, 20480000, 20503125, 20736000, 20971520, 
	20995200, 21093750, 21233664, 21257640, 21600000, 21870000, 22118400, 22143375, 22394880, 
	22500000, 22674816, 22781250, 23040000, 23328000, 23437500, 23592960, 23619600, 23887872, 
	23914845, 24000000, 24300000, 24576000, 24603750, 24883200, 25000000, 25165824, 25194240, 
	25312500, 25509168, 25600000, 25920000, 26214400, 26244000, 26542080, 26572050, 26873856, 
	27000000, 27337500, 27648000, 27993600, 28125000, 28311552, 28343520, 28800000, 29160000, 
	29296875, 29491200, 29524500, 29859840, 30000000, 30233088, 30375000, 30720000, 31104000, 
	31250000, 31457280, 31492800, 31640625, 31850496, 31886460, 32000000, 32400000, 32768000, 
	32805000, 33177600, 33554432, 33592320, 33750000, 34012224, 34171875, 34560000, 34992000, 
	35156250, 35389440, 35429400, 35831808, 36000000, 36450000, 36864000, 36905625, 37324800, 
	37500000, 37748736, 37791360, 37968750, 38263752, 38400000, 38880000, 39062500, 39321600, 
	39366000, 39813120, 39858075, 40000000, 40310784, 40500000, 40960000, 41006250, 41472000, 
	41943040, 41990400, 42187500, 42467328, 42515280, 43200000, 43740000, 44236800, 44286750, 
	44789760, 45000000, 45349632, 45562500, 46080000, 46656000, 46875000, 47185920, 47239200, 
	47775744, 47829690, 48000000, 48600000, 48828125, 49152000, 49207500, 49766400, 50000000, 
	50331648, 50388480, 50625000, 51018336, 51200000, 51840000, 52428800, 52488000, 52734375, 
	53084160, 53144100, 53747712, 54000000, 54675000, 55296000, 55987200, 56250000, 56623104, 
	56687040, 56953125, 57600000, 58320000, 58593750, 58982400, 59049000, 59719680, 60000000, 
	60466176, 60750000, 61440000, 61509375, 62208000, 62500000, 62914560, 62985600, 63281250, 
	63700992, 63772920, 64000000, 64800000, 65536000, 65610000, 66355200, 66430125, 67108864, 
	67184640, 67500000, 68024448, 68343750, 69120000, 69984000, 70312500, 70778880, 70858800, 
	71663616, 72000000, 72900000, 73728000, 73811250, 74649600, 75000000, 75497472, 75582720, 
	75937500, 76527504, 76800000, 77760000, 78125000, 78643200, 78732000, 79626240, 79716150, 
	80000000, 80621568, 81000000, 81920000, 82012500, 82944000, 83886080, 83980800, 84375000, 
	84934656, 85030560, 86400000, 87480000, 87890625, 88473600, 88573500, 89579520, 90000000, 
	90699264, 91125000, 92160000, 93312000, 93750000, 94371840, 94478400, 94921875, 95551488, 
	95659380, 96000000, 97200000, 97656250, 98304000, 98415000, 99532800, 100000000, 
	100663296, 100776960, 101250000, 102036672, 102400000, 102515625, 103680000, 104857600, 
	104976000, 105468750, 106168320, 106288200, 107495424, 108000000, 109350000, 110592000, 
	110716875, 111974400, 112500000, 113246208, 113374080, 113906250, 115200000, 116640000, 
	117187500, 117964800, 118098000, 119439360, 119574225, 120000000, 120932352, 121500000, 
	122880000, 123018750, 124416000, 125000000, 125829120, 125971200, 126562500, 127401984, 
	127545840, 128000000, 129600000, 131072000, 131220000, 132710400, 132860250, 134217728, 
	134369280, 135000000, 136048896, 136687500, 138240000, 139968000, 140625000, 141557760, 
	141717600, 143327232, 144000000, 145800000, 146484375, 147456000, 147622500, 149299200, 
	150000000, 150994944, 151165440, 151875000, 153055008, 153600000, 155520000, 156250000, 
	157286400, 157464000, 158203125, 159252480, 159432300, 160000000, 161243136, 162000000, 
	163840000, 164025000, 165888000, 167772160, 167961600, 168750000, 169869312, 170061120, 
	170859375, 172800000, 174960000, 175781250, 176947200, 177147000, 179159040, 180000000, 
	181398528, 182250000, 184320000, 184528125, 186624000, 187500000, 188743680, 188956800, 
	189843750, 191102976, 191318760, 192000000, 194400000, 195312500, 196608000, 196830000, 
	199065600, 199290375, 200000000, 201326592, 201553920, 202500000, 204073344, 204800000, 
	205031250, 207360000, 209715200, 209952000, 210937500, 212336640, 212576400, 214990848, 
	216000000, 218700000, 221184000, 221433750, 223948800, 225000000, 226492416, 226748160, 
	227812500, 230400000, 233280000, 234375000, 235929600, 236196000, 238878720, 239148450, 
	240000000, 241864704, 243000000, 244140625, 245760000, 246037500, 248832000, 250000000, 
	251658240, 251942400, 253125000, 254803968, 255091680, 256000000, 259200000, 262144000, 
	262440000, 263671875, 265420800, 265720500, 268435456, 268738560, 270000000, 272097792, 
	273375000, 276480000, 279936000, 281250000, 283115520, 283435200, 284765625, 286654464, 
	288000000, 291600000, 292968750, 294912000, 295245000, 298598400, 300000000, 301989888, 
	302330880, 303750000, 306110016, 307200000, 307546875, 311040000, 312500000, 314572800, 
	314928000, 316406250, 318504960, 318864600, 320000000, 322486272, 324000000, 327680000, 
	328050000, 331776000, 332150625, 335544320, 335923200, 337500000, 339738624, 340122240, 
	341718750, 345600000, 349920000, 351562500, 353894400, 354294000, 358318080, 360000000, 
	362797056, 364500000, 368640000, 369056250, 373248000, 375000000, 377487360, 377913600, 
	379687500, 382205952, 382637520, 384000000, 388800000, 390625000, 393216000, 393660000, 
	398131200, 398580750, 400000000, 402653184, 403107840, 405000000, 408146688, 409600000, 
	410062500, 414720000, 419430400, 419904000, 421875000, 424673280, 425152800, 429981696, 
	432000000, 437400000, 439453125, 442368000, 442867500, 447897600, 450000000, 452984832, 
	453496320, 455625000, 460800000, 466560000, 468750000, 471859200, 472392000, 474609375, 
	477757440, 478296900, 480000000, 483729408, 486000000, 488281250, 491520000, 492075000, 
	497664000, 500000000, 503316480, 503884800, 506250000, 509607936, 510183360, 512000000, 
	512578125, 518400000, 524288000, 524880000, 527343750, 530841600, 531441000, 536870912, 
	537477120, 540000000, 544195584, 546750000, 552960000, 553584375, 559872000, 562500000, 
	566231040, 566870400, 569531250, 573308928, 576000000, 583200000, 585937500, 589824000, 
	590490000, 597196800, 597871125, 600000000, 603979776, 604661760, 607500000, 612220032, 
	614400000, 615093750, 622080000, 625000000, 629145600, 629856000, 632812500, 637009920, 
	637729200, 640000000, 644972544, 648000000, 655360000, 656100000, 663552000, 664301250, 
	671088640, 671846400, 675000000, 679477248, 680244480, 683437500, 691200000, 699840000, 
	703125000, 707788800, 708588000, 716636160, 720000000, 725594112, 729000000, 732421875, 
	737280000, 738112500, 746496000, 750000000, 754974720, 755827200, 759375000, 764411904, 
	765275040, 768000000, 777600000, 781250000, 786432000, 787320000, 791015625, 796262400, 
	797161500, 800000000, 805306368, 806215680, 810000000, 816293376, 819200000, 820125000, 
	829440000, 838860800, 839808000, 843750000, 849346560, 850305600, 854296875, 859963392, 
	864000000, 874800000, 878906250, 884736000, 885735000, 895795200, 900000000, 905969664, 
	906992640, 911250000, 921600000, 922640625, 933120000, 937500000, 943718400, 944784000, 
	949218750, 955514880, 956593800, 960000000, 967458816, 972000000, 976562500, 983040000, 
	984150000, 995328000, 996451875, 1000000000, 1006632960, 1007769600, 1012500000, 
	1019215872, 1020366720, 1024000000, 1025156250, 1036800000, 1048576000, 1049760000, 
	1054687500, 1061683200, 1062882000, 1073741824, 1074954240, 1080000000, 1088391168, 
	1093500000, 1105920000, 1107168750, 1119744000, 1125000000, 1132462080, 1133740800, 
	1139062500, 1146617856, 1152000000, 1166400000, 1171875000, 1179648000, 1180980000, 
	1194393600, 1195742250, 1200000000, 1207959552, 1209323520, 1215000000, 1220703125, 
	1224440064, 1228800000, 1230187500, 1244160000, 1250000000, 1258291200, 1259712000, 
	1265625000, 1274019840, 1275458400, 1280000000, 1289945088, 1296000000, 1310720000, 
	1312200000, 1318359375, 1327104000, 1328602500, 1342177280, 1343692800, 1350000000, 
	1358954496, 1360488960, 1366875000, 1382400000, 1399680000, 1406250000, 1415577600, 
	1417176000, 1423828125, 1433272320, 1440000000, 1451188224, 1458000000, 1464843750, 
	1474560000, 1476225000, 1492992000, 1500000000, 1509949440, 1511654400, 1518750000, 
	1528823808, 1530550080, 1536000000, 1537734375, 1555200000, 1562500000, 1572864000, 
	1574640000, 1582031250, 1592524800, 1594323000, 1600000000, 1610612736, 1612431360, 
	1620000000, 1632586752, 1638400000, 1640250000, 1658880000, 1660753125, 1677721600, 
	1679616000, 1687500000, 1698693120, 1700611200, 1708593750, 1719926784, 1728000000, 
	1749600000, 1757812500, 1769472000, 1771470000, 1791590400, 1800000000, 1811939328, 
	1813985280, 1822500000, 1843200000, 1845281250, 1866240000, 1875000000, 1887436800, 
	1889568000, 1898437500, 1911029760, 1913187600, 1920000000, 1934917632, 1944000000, 
	1953125000, 1966080000, 1968300000, 1990656000, 1992903750, 2000000000, 2013265920, 
	2015539200, 2025000000, 2038431744, 2040733440, 2048000000, 2050312500, 2073600000, 
	2097152000, 2099520000, 2109375000, 2123366400, 2125764000
};

static int _ccv_get_optimal_fft_size(int size)
{
	int a = 0, b = sizeof(_ccv_optimal_fft_size) / sizeof(_ccv_optimal_fft_size[0]) - 1;
    if((unsigned)size >= (unsigned)_ccv_optimal_fft_size[b])
		return -1;
	while(a < b)
	{
		int c = (a + b) >> 1;
		if(size <= _ccv_optimal_fft_size[c])
			b = c;
		else
			a = c + 1;
    }
    return _ccv_optimal_fft_size[b];
}

static pthread_mutex_t fftw_plan_mutex = PTHREAD_MUTEX_INITIALIZER;

static void _ccv_filter_fftw(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* d, int padding_pattern)
{
	int ch = CCV_GET_CHANNEL(a->type);
	int fft_type = (CCV_GET_DATA_TYPE(d->type) == CCV_8U || CCV_GET_DATA_TYPE(d->type) == CCV_32F) ? CCV_32F : CCV_64F;
	int rows = ccv_min(a->rows + b->rows - 1, _ccv_get_optimal_fft_size(b->rows * 3));
	int cols = ccv_min(a->cols + b->cols - 1, _ccv_get_optimal_fft_size(b->cols * 3));
	int cols_2c = 2 * (cols / 2 + 1);
	void* fftw_a;
	void* fftw_b;
	void* fftw_d;
	fftw_plan p, pinv;
	fftwf_plan pf, pinvf;
	if (fft_type == CCV_32F)
	{
		fftw_a = fftwf_malloc(rows * cols_2c * ch * sizeof(float));
		fftw_b = fftwf_malloc(rows * cols_2c * ch * sizeof(float));
		fftw_d = fftwf_malloc(rows * cols_2c * ch * sizeof(float));
		pthread_mutex_lock(&fftw_plan_mutex);
		if (ch == 1)
		{
			pf = fftwf_plan_dft_r2c_2d(rows, cols, 0, 0, FFTW_ESTIMATE);
			pinvf = fftwf_plan_dft_c2r_2d(rows, cols, 0, 0, FFTW_ESTIMATE);
		} else {
			int ndim[] = {rows, cols};
			pf = fftwf_plan_many_dft_r2c(2, ndim, ch, 0, 0, ch, 1, 0, 0, ch, 1, FFTW_ESTIMATE);
			pinvf = fftwf_plan_many_dft_c2r(2, ndim, ch, 0, 0, ch, 1, 0, 0, ch, 1, FFTW_ESTIMATE);
		}
		pthread_mutex_unlock(&fftw_plan_mutex);
	} else {
		fftw_a = fftw_malloc(rows * cols_2c * ch * sizeof(double));
		fftw_b = fftw_malloc(rows * cols_2c * ch * sizeof(double));
		fftw_d = fftw_malloc(rows * cols_2c * ch * sizeof(double));
		pthread_mutex_lock(&fftw_plan_mutex);
		if (ch == 1)
		{
			p = fftw_plan_dft_r2c_2d(rows, cols, 0, 0, FFTW_ESTIMATE);
			pinv = fftw_plan_dft_c2r_2d(rows, cols, 0, 0, FFTW_ESTIMATE);
		} else {
			int ndim[] = {rows, cols};
			p = fftw_plan_many_dft_r2c(2, ndim, ch, 0, 0, ch, 1, 0, 0, ch, 1, FFTW_ESTIMATE);
			pinv = fftw_plan_many_dft_c2r(2, ndim, ch, 0, 0, ch, 1, 0, 0, ch, 1, FFTW_ESTIMATE);
		}
		pthread_mutex_unlock(&fftw_plan_mutex);
	}
	memset(fftw_b, 0, rows * cols_2c * ch * CCV_GET_DATA_TYPE_SIZE(fft_type));

	int i, j, k;
	unsigned char* m_ptr = b->data.u8;
	// to flip matrix b is crucial, this problem only shows when I changed to a more sophisticated test case
#define for_block(_for_type, _for_get) \
	_for_type* fftw_ptr = (_for_type*)fftw_b + (b->rows - 1) * cols_2c * ch; \
	for (i = 0; i < b->rows; i++) \
	{ \
		for (j = 0; j < b->cols; j++) \
			for (k = 0; k < ch; k++) \
				fftw_ptr[(b->cols - 1 - j) * ch + k] = _for_get(m_ptr, j * ch + k, 0); \
		fftw_ptr -= cols_2c * ch; \
		m_ptr += b->step; \
	}
	ccv_matrix_typeof(fft_type, ccv_matrix_getter, b->type, for_block);
#undef for_block
	if (fft_type == CCV_32F)
		fftwf_execute_dft_r2c(pf, (float*)fftw_b, (fftwf_complex*)fftw_b);
	else
		fftw_execute_dft_r2c(p, (double*)fftw_b, (fftw_complex*)fftw_b);
	/* why a->cols + cols - 2 * (b->cols & ~1) ?
	 * what we really want is ceiling((a->cols - (b->cols & ~1)) / (cols - (b->cols & ~1)))
	 * in this case, we strip out paddings on the left/right, and compute how many tiles
	 * we need. It then be interpreted in the above integer division form */
	int tile_x = ccv_max(1, (a->cols + cols - 2 * (b->cols & ~1)) / (cols - (b->cols & ~1)));
	int tile_y = ccv_max(1, (a->rows + rows - 2 * (b->rows & ~1)) / (rows - (b->rows & ~1)));
	int brows2 = b->rows / 2;
	int bcols2 = b->cols / 2;
#define for_block(_for_type, _cpx_type, _for_set, _for_get) \
	_for_type scale = 1.0 / (rows * cols); \
	for (i = 0; i < tile_y; i++) \
		for (j = 0; j < tile_x; j++) \
		{ \
			int x, y; \
			memset(fftw_a, 0, rows * cols_2c * ch * sizeof(_for_type)); \
			int iy = ccv_min(i * (rows - (b->rows & ~1)), ccv_max(a->rows - rows, 0)); \
			int ix = ccv_min(j * (cols - (b->cols & ~1)), ccv_max(a->cols - cols, 0)); \
			_for_type* fftw_ptr = (_for_type*)fftw_a; \
			int end_y = ccv_min(rows, a->rows - iy); \
			int end_x = ccv_min(cols, a->cols - ix); \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(a, iy, ix, 0); \
			for (y = 0; y < end_y; y++) \
			{ \
				for (x = 0; x < end_x * ch; x++) \
					fftw_ptr[x] = _for_get(m_ptr, x, 0); \
				fftw_ptr += cols_2c * ch; \
				m_ptr += a->step; \
			} \
			_cpx_type* fftw_ac = (_cpx_type*)fftw_a; \
			_cpx_type* fftw_bc = (_cpx_type*)fftw_b; \
			_cpx_type* fftw_dc = (_cpx_type*)fftw_d; \
			fft_execute_dft_r2c((_for_type*)fftw_a, (_cpx_type*)fftw_ac); \
			for (x = 0; x < rows * ch * (cols / 2 + 1); x++) \
				fftw_dc[x] = (fftw_ac[x] * fftw_bc[x]) * scale; \
			fft_execute_dft_c2r((_cpx_type*)fftw_dc, (_for_type*)fftw_d); \
			fftw_ptr = (_for_type*)fftw_d + ((1 + (i > 0)) * brows2 * cols_2c + (1 + (j > 0)) * bcols2) * ch; \
			end_y = ccv_min(d->rows - (iy + (i > 0) * brows2), \
							(rows - (b->rows & ~1)) + (i == 0) * brows2); \
			end_x = ccv_min(d->cols - (ix + (j > 0) * bcols2), \
							(cols - (b->cols & ~1)) + (j == 0) * bcols2); \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2, ix + (j > 0) * bcols2, 0); \
			for (y = 0; y < end_y; y++) \
			{ \
				for (x = 0; x < end_x * ch; x++) \
					_for_set(m_ptr, x, fftw_ptr[x], 0); \
				m_ptr += d->step; \
				fftw_ptr += cols_2c * ch; \
			} \
			int end_tile_y, end_tile_x; \
			/* handle edge cases: */ \
			if (i + 1 == tile_y && end_y + iy + (i > 0) * brows2 < d->rows) \
			{ \
				end_tile_y = ccv_min(brows2, d->rows - (iy + (i > 0) * brows2 + end_y)); \
				fftw_ptr = (_for_type*)fftw_d + (1 + (j > 0)) * bcols2 * ch; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2 + end_y, ix + (j > 0) * bcols2, 0); \
				for (y = 0; y < end_tile_y; y++) \
				{ \
					for (x = 0; x < end_x * ch; x++) \
						_for_set(m_ptr, x, fftw_ptr[x], 0); \
					m_ptr += d->step; \
					fftw_ptr += cols_2c * ch; \
				} \
			} \
			if (j + 1 == tile_x && end_x + ix + (j > 0) * bcols2 < d->cols) \
			{ \
				end_tile_x = ccv_min(bcols2, d->cols - (ix + (j > 0) * bcols2 + end_x)); \
				fftw_ptr = (_for_type*)fftw_d + (1 + (i > 0)) * brows2 * cols_2c * ch; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2, ix + (j > 0) * bcols2 + end_x, 0); \
				for (y = 0; y < end_y; y++) \
				{ \
					for (x = 0; x < end_tile_x * ch; x++) \
						_for_set(m_ptr, x, fftw_ptr[x], 0); \
					m_ptr += d->step; \
					fftw_ptr += cols_2c * ch; \
				} \
			} \
			if (i + 1 == tile_y && end_y + iy + (i > 0) * brows2 < d->rows && \
				j + 1 == tile_x && end_x + ix + (j > 0) * bcols2 < d->cols) \
			{ \
				fftw_ptr = (_for_type*)fftw_d; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2 + end_y, ix + (j > 0) * bcols2 + end_x, 0); \
				for (y = 0; y < end_tile_y; y++) \
				{ \
					for (x = 0; x < end_tile_x * ch; x++) \
						_for_set(m_ptr, x, fftw_ptr[x], 0); \
					m_ptr += d->step; \
					fftw_ptr += cols_2c * ch; \
				} \
			} \
		}
	if (fft_type == CCV_32F)
	{
#define fft_execute_dft_r2c(r, c) fftwf_execute_dft_r2c(pf, r, c)
#define fft_execute_dft_c2r(c, r) fftwf_execute_dft_c2r(pinvf, c, r)
		ccv_matrix_setter(d->type, ccv_matrix_getter, a->type, for_block, float, fftwf_complex);
#undef fft_execute_dft_r2c
#undef fft_execute_dft_c2r
		pthread_mutex_lock(&fftw_plan_mutex);
		fftwf_destroy_plan(pf);
		fftwf_destroy_plan(pinvf);
		pthread_mutex_unlock(&fftw_plan_mutex);
		fftwf_free(fftw_a);
		fftwf_free(fftw_b);
		fftwf_free(fftw_d);
	} else {
#define fft_execute_dft_r2c(r, c) fftw_execute_dft_r2c(p, r, c)
#define fft_execute_dft_c2r(c, r) fftw_execute_dft_c2r(pinv, c, r)
		ccv_matrix_setter(d->type, ccv_matrix_getter, a->type, for_block, double, fftw_complex);
#undef fft_execute_dft_r2c
#undef fft_execute_dft_c2r
		pthread_mutex_lock(&fftw_plan_mutex);
		fftw_destroy_plan(p);
		fftw_destroy_plan(pinv);
		pthread_mutex_unlock(&fftw_plan_mutex);
		fftw_free(fftw_a);
		fftw_free(fftw_b);
		fftw_free(fftw_d);
	}
#undef for_block
}
#else
static void _ccv_filter_kissfft(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* d, int padding_pattern)
{
	int ch = CCV_GET_CHANNEL(a->type);
	int fft_type = (CCV_GET_DATA_TYPE(d->type) == CCV_8U || CCV_GET_DATA_TYPE(d->type) == CCV_32F) ? CCV_32F : CCV_64F;
	int rows = ((ccv_min(a->rows + b->rows - 1, kiss_fftr_next_fast_size_real(b->rows * 3)) + 1) >> 1) << 1;
	int cols = ((ccv_min(a->cols + b->cols - 1, kiss_fftr_next_fast_size_real(b->cols * 3)) + 1) >> 1) << 1;
	int ndim[] = {rows, cols};
	void* kiss_a;
	void* kiss_b;
	void* kiss_d;
	void* kiss_ac;
	void* kiss_bc;
	void* kiss_dc;
	kiss_fftndr_cfg p;
	kiss_fftndr_cfg pinv;
	kissf_fftndr_cfg pf;
	kissf_fftndr_cfg pinvf;
	if (fft_type == CCV_32F)
	{
		pf = kissf_fftndr_alloc(ndim, 2, 0, 0, 0);
		pinvf = kissf_fftndr_alloc(ndim, 2, 1, 0, 0);
		kiss_a = ccmalloc(rows * cols * ch * sizeof(kissf_fft_scalar));
		kiss_b = ccmalloc(rows * cols * ch * sizeof(kissf_fft_scalar));
		kiss_d = ccmalloc(rows * cols * ch * sizeof(kissf_fft_scalar));
		kiss_ac = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kissf_fft_cpx));
		kiss_bc = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kissf_fft_cpx));
		kiss_dc = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kissf_fft_cpx));
		memset(kiss_b, 0, rows * cols * ch * sizeof(kissf_fft_scalar));
	} else {
		p = kiss_fftndr_alloc(ndim, 2, 0, 0, 0);
		pinv = kiss_fftndr_alloc(ndim, 2, 1, 0, 0);
		kiss_a = ccmalloc(rows * cols * ch * sizeof(kiss_fft_scalar));
		kiss_b = ccmalloc(rows * cols * ch * sizeof(kiss_fft_scalar));
		kiss_d = ccmalloc(rows * cols * ch * sizeof(kiss_fft_scalar));
		kiss_ac = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kiss_fft_cpx));
		kiss_bc = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kiss_fft_cpx));
		kiss_dc = ccmalloc(rows * (cols / 2 + 1) * ch * sizeof(kiss_fft_cpx));
		memset(kiss_b, 0, rows * cols * ch * sizeof(kiss_fft_scalar));
	}
	int nch = rows * cols, nchc = rows * (cols / 2 + 1);
	int i, j, k;
	unsigned char* m_ptr = b->data.u8;
	// to flip matrix b is crucial, this problem only shows when I changed to a more sophisticated test case
#define for_block(_for_type, _for_get) \
	_for_type* kiss_ptr = (_for_type*)kiss_b + (b->rows - 1) * cols; \
	for (i = 0; i < b->rows; i++) \
	{ \
		for (j = 0; j < b->cols; j++) \
			for (k = 0; k < ch; k++) \
				kiss_ptr[k * nch + b->cols - 1 - j] = _for_get(m_ptr, j * ch + k, 0); \
		kiss_ptr -= cols; \
		m_ptr += b->step; \
	}
	ccv_matrix_typeof(fft_type, ccv_matrix_getter, b->type, for_block);
#undef for_block
	if (fft_type == CCV_32F)
		for (k = 0; k < ch; k++)
			kissf_fftndr(pf, (kissf_fft_scalar*)kiss_b + nch * k, (kissf_fft_cpx*)kiss_bc + nchc * k);
	else
		for (k = 0; k < ch; k++)
			kiss_fftndr(p, (kiss_fft_scalar*)kiss_b + nch * k, (kiss_fft_cpx*)kiss_bc + nchc * k);
	/* why a->cols + cols - 2 * (b->cols & ~1) ?
	 * what we really want is ceiling((a->cols - (b->cols & ~1)) / (cols - (b->cols & ~1)))
	 * in this case, we strip out paddings on the left/right, and compute how many tiles
	 * we need. It then be interpreted in the above integer division form */
	int tile_x = ccv_max(1, (a->cols + cols - 2 * (b->cols & ~1)) / (cols - (b->cols & ~1)));
	int tile_y = ccv_max(1, (a->rows + rows - 2 * (b->rows & ~1)) / (rows - (b->rows & ~1)));
	int brows2 = b->rows / 2;
	int bcols2 = b->cols / 2;
#define for_block(_for_type, _cpx_type, _for_set, _for_get) \
	_for_type scale = 1.0 / (rows * cols); \
	for (i = 0; i < tile_y; i++) \
		for (j = 0; j < tile_x; j++) \
		{ \
			int x, y; \
			memset(kiss_a, 0, rows * cols * ch * sizeof(_for_type)); \
			int iy = ccv_min(i * (rows - (b->rows & ~1)), ccv_max(a->rows - rows, 0)); \
			int ix = ccv_min(j * (cols - (b->cols & ~1)), ccv_max(a->cols - cols, 0)); \
			_for_type* kiss_ptr = (_for_type*)kiss_a; \
			int end_y = ccv_min(rows, a->rows - iy); \
			int end_x = ccv_min(cols, a->cols - ix); \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(a, iy, ix, 0); \
			for (y = 0; y < end_y; y++) \
			{ \
				for (x = 0; x < end_x; x++) \
					for (k = 0; k < ch; k++) \
						kiss_ptr[k * nch + x] = _for_get(m_ptr, x * ch + k, 0); \
				kiss_ptr += cols; \
				m_ptr += a->step; \
			} \
			for (k = 0; k < ch; k++) \
				fft_ndr((_for_type*)kiss_a + nch * k, (_cpx_type*)kiss_ac + nchc * k); \
			_cpx_type* fft_ac = (_cpx_type*)kiss_ac; \
			_cpx_type* fft_bc = (_cpx_type*)kiss_bc; \
			_cpx_type* fft_dc = (_cpx_type*)kiss_dc; \
			for (x = 0; x < rows * ch * (cols / 2 + 1); x++) \
			{ \
				fft_dc[x].r = (fft_ac[x].r * fft_bc[x].r - fft_ac[x].i * fft_bc[x].i) * scale; \
				fft_dc[x].i = (fft_ac[x].i * fft_bc[x].r + fft_ac[x].r * fft_bc[x].i) * scale; \
			} \
			for (k = 0; k < ch; k++) \
				fft_ndri((_cpx_type*)kiss_dc + nchc * k, (_for_type*)kiss_d + nch * k); \
			kiss_ptr = (_for_type*)kiss_d + (1 + (i > 0)) * brows2 * cols + (1 + (j > 0)) * bcols2; \
			end_y = ccv_min(d->rows - (iy + (i > 0) * brows2), \
							(rows - (b->rows & ~1)) + (i == 0) * brows2); \
			end_x = ccv_min(d->cols - (ix + (j > 0) * bcols2), \
							(cols - (b->cols & ~1)) + (j == 0) * bcols2); \
			m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2, ix + (j > 0) * bcols2, 0); \
			for (y = 0; y < end_y; y++) \
			{ \
				for (x = 0; x < end_x; x++) \
					for (k = 0; k < ch; k++) \
						_for_set(m_ptr, x * ch + k, kiss_ptr[k * nch + x], 0); \
				m_ptr += d->step; \
				kiss_ptr += cols; \
			} \
			int end_tile_y, end_tile_x; \
			/* handle edge cases: */ \
			if (i + 1 == tile_y && end_y + iy + (i > 0) * brows2 < d->rows) \
			{ \
				end_tile_y = ccv_min(brows2, d->rows - (iy + (i > 0) * brows2 + end_y)); \
				kiss_ptr = (_for_type*)kiss_d + (1 + (j > 0)) * bcols2; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2 + end_y, ix + (j > 0) * bcols2, 0); \
				for (y = 0; y < end_tile_y; y++) \
				{ \
					for (x = 0; x < end_x; x++) \
						for (k = 0; k < ch; k++) \
							_for_set(m_ptr, x * ch + k, kiss_ptr[k * nch + x], 0); \
					m_ptr += d->step; \
					kiss_ptr += cols; \
				} \
			} \
			if (j + 1 == tile_x && end_x + ix + (j > 0) * bcols2 < d->cols) \
			{ \
				end_tile_x = ccv_min(bcols2, d->cols - (ix + (j > 0) * bcols2 + end_x)); \
				kiss_ptr = (_for_type*)kiss_d + (1 + (i > 0)) * brows2 * cols; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2, ix + (j > 0) * bcols2 + end_x, 0); \
				for (y = 0; y < end_y; y++) \
				{ \
					for (x = 0; x < end_tile_x; x++) \
						for (k = 0; k < ch; k++) \
							_for_set(m_ptr, x * ch + k, kiss_ptr[k * nch + x], 0); \
					m_ptr += d->step; \
					kiss_ptr += cols; \
				} \
			} \
			if (i + 1 == tile_y && end_y + iy + (i > 0) * brows2 < d->rows && \
				j + 1 == tile_x && end_x + ix + (j > 0) * bcols2 < d->cols) \
			{ \
				kiss_ptr = (_for_type*)kiss_d; \
				m_ptr = (unsigned char*)ccv_get_dense_matrix_cell(d, iy + (i > 0) * brows2 + end_y, ix + (j > 0) * bcols2 + end_x, 0); \
				for (y = 0; y < end_tile_y; y++) \
				{ \
					for (x = 0; x < end_tile_x; x++) \
						for (k = 0; k < ch; k++) \
							_for_set(m_ptr, x * ch + k, kiss_ptr[k * nch + x], 0); \
					m_ptr += d->step; \
					kiss_ptr += cols; \
				} \
			} \
		}
	if (fft_type == CCV_32F)
	{
#define fft_ndr(r, c) kissf_fftndr(pf, r, c)
#define fft_ndri(c, r) kissf_fftndri(pinvf, c, r)
		ccv_matrix_setter(d->type, ccv_matrix_getter, a->type, for_block, kissf_fft_scalar, kissf_fft_cpx);
#undef fft_ndr
#undef fft_ndri
		kissf_fft_free(pf);
		kissf_fft_free(pinvf);
	} else {
#define fft_ndr(r, c) kiss_fftndr(p, r, c)
#define fft_ndri(c, r) kiss_fftndri(pinv, c, r)
		ccv_matrix_setter(d->type, ccv_matrix_getter, a->type, for_block, kiss_fft_scalar, kiss_fft_cpx);
#undef fft_ndr
#undef fft_ndri
		kiss_fft_free(p);
		kiss_fft_free(pinv);
	}
#undef for_block
	ccfree(kiss_dc);
	ccfree(kiss_bc);
	ccfree(kiss_ac);
	ccfree(kiss_d);
	ccfree(kiss_b);
	ccfree(kiss_a);
}
#endif

static void _ccv_filter_direct_8u(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t* d, int padding_pattern)
{
	int i, j, y, x, k;
	int nz = b->rows * b->cols;
	int* coeff = (int*)ccmalloc(nz * sizeof(int));
	int* cx = (int*)ccmalloc(nz * sizeof(int));
	int* cy = (int*)ccmalloc(nz * sizeof(int));
	int scale = 1 << 14;
	nz = 0;
	for (i = 0; i < b->rows; i++)
		for (j = 0; j < b->cols; j++)
		{
			coeff[nz] = (int)(ccv_get_dense_matrix_cell_value(b, i, j, 0) * scale + 0.5);
			if (coeff[nz] == 0)
				continue;
			cy[nz] = i;
			cx[nz] = j;
			nz++;
		}
	ccv_dense_matrix_t* pa = ccv_dense_matrix_new(a->rows + b->rows / 2 * 2, a->cols + b->cols / 2 * 2, CCV_8U | CCV_C1, 0, 0);
	/* the padding pattern is different from FFT: |aa{BORDER}|abcd|{BORDER}dd| */
	for (i = 0; i < pa->rows; i++)
		for (j = 0; j < pa->cols; j++)
			pa->data.u8[i * pa->step + j] = a->data.u8[ccv_clamp(i - b->rows / 2, 0, a->rows - 1) * a->step + ccv_clamp(j - b->cols / 2, 0, a->cols - 1)];
	unsigned char* m_ptr = d->data.u8;
	unsigned char* a_ptr = pa->data.u8;
	/* 0.5 denote the overhead for indexing x and y */
	if (nz < b->rows * b->cols * 0.75)
	{
		for (i = 0; i < d->rows; i++)
		{
			for (j = 0; j < d->cols; j++)
			{
				int z = 0;
				for (k = 0; k < nz; k++)
					z += a_ptr[cy[k] * pa->step + j + cx[k]] * coeff[k];
				m_ptr[j] = ccv_clamp(z >> 14, 0, 255);
			}
			m_ptr += d->step;
			a_ptr += pa->step;
		}
	} else {
		k = 0;
		for (i = 0; i < b->rows; i++)
			for (j = 0; j < b->cols; j++)
			{
				coeff[k] = (int)(ccv_get_dense_matrix_cell_value(b, i, j, 0) * scale + 0.5);
				k++;
			}
		for (i = 0; i < d->rows; i++)
		{
			for (j = 0; j < d->cols; j++)
			{
				int* c_ptr = coeff;
				int z = 0;
				for (y = 0; y < b->rows; y++)
				{
					int iyx = y * pa->step;
					for (x = 0; x < b->cols; x++)
					{
						z += a_ptr[iyx + j + x] * c_ptr[0];
						c_ptr++;
					}
				}
				m_ptr[j] = ccv_clamp(z >> 14, 0, 255);
			}
			m_ptr += d->step;
			a_ptr += pa->step;
		}
	}
	ccv_matrix_free(pa);
	ccfree(coeff);
	ccfree(cx);
	ccfree(cy);
}

void ccv_filter(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t** d, int type, int padding_pattern)
{
	ccv_declare_derived_signature(sig, a->sig != 0 && b->sig != 0, ccv_sign_with_literal("ccv_filter"), a->sig, b->sig, CCV_EOF_SIGN);
	type = (type == 0) ? CCV_GET_DATA_TYPE(a->type) | CCV_GET_CHANNEL(a->type) : CCV_GET_DATA_TYPE(type) | CCV_GET_CHANNEL(a->type);
	ccv_dense_matrix_t* dd = *d = ccv_dense_matrix_renew(*d, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_object_return_if_cached(, dd);

	/* 15 is the constant to indicate the high cost of FFT (even with O(nlog(m)) for
	 * integer image.
	 * NOTE: FFT has time complexity of O(nlog(n)), however, for convolution, it
	 * is not the case. Convolving one image (a) to a kernel (b), can be done by
	 * dividing image a to several blocks proportional to (b). Thus, one don't need
	 * to do FFT for the whole image. The image can be divided to n/m part, and
	 * the FFT itself is O(mlog(m)), so, the convolution process has time complexity
	 * of O(nlog(m)) */
	if ((b->rows * b->cols < (log((double)(b->rows * b->cols)) + 1) * 15) && (a->type & CCV_8U))
	{
		_ccv_filter_direct_8u(a, b, dd, padding_pattern);
	} else {
#ifdef HAVE_FFTW3
		_ccv_filter_fftw(a, b, dd, padding_pattern);
#else
		_ccv_filter_kissfft(a, b, dd, padding_pattern);
#endif
	}
}

void ccv_filter_kernel(ccv_dense_matrix_t* x, ccv_filter_kernel_f func, void* data)
{
	int i, j, k, ch = CCV_GET_CHANNEL(x->type);
	unsigned char* m_ptr = x->data.u8;
	double rows_2 = (x->rows - 1) * 0.5;
	double cols_2 = (x->cols - 1) * 0.5;
#define for_block(_, _for_set) \
	for (i = 0; i < x->rows; i++) \
	{ \
		for (j = 0; j < x->cols; j++) \
		{ \
			double result = func(j - cols_2, i - rows_2, data); \
			for (k = 0; k < ch; k++) \
				_for_set(m_ptr, j * ch + k, result, 0); \
		} \
		m_ptr += x->step; \
	}
	ccv_matrix_setter(x->type, for_block);
#undef for_block
	ccv_make_matrix_immutable(x);
}

void ccv_distance_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_dense_matrix_t** x, int x_type, ccv_dense_matrix_t** y, int y_type, double dx, double dy, double dxx, double dyy, int flag)
{
	assert(!(flag & CCV_L2_NORM) && (flag & CCV_GSEDT));
	ccv_declare_derived_signature(sig, a->sig != 0, ccv_sign_with_format(64, "ccv_distance_transform(%la,%la,%la,%la,%d)", dx, dy, dxx, dyy, flag), a->sig, CCV_EOF_SIGN);
	type = (CCV_GET_DATA_TYPE(type) == CCV_64F || CCV_GET_DATA_TYPE(a->type) == CCV_64F || CCV_GET_DATA_TYPE(a->type) == CCV_64S) ? CCV_GET_CHANNEL(a->type) | CCV_64F : CCV_GET_CHANNEL(a->type) | CCV_32F;
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_renew(*b, a->rows, a->cols, CCV_ALL_DATA_TYPE | CCV_GET_CHANNEL(a->type), type, sig);
	ccv_dense_matrix_t* mx = 0;
	ccv_dense_matrix_t* my = 0;
	if (x != 0)
	{
		ccv_declare_derived_signature(xsig, a->sig != 0, ccv_sign_with_format(64, "ccv_distance_transform_x(%la,%la,%la,%la,%d)", dx, dy, dxx, dyy, flag), a->sig, CCV_EOF_SIGN);
		mx = *x = ccv_dense_matrix_renew(*x, a->rows, a->cols, CCV_32S | CCV_C1, CCV_32S | CCV_C1, xsig);
	}
	if (y != 0)
	{
		ccv_declare_derived_signature(ysig, a->sig != 0, ccv_sign_with_format(64, "ccv_distance_transform_y(%la,%la,%la,%la,%d)", dx, dy, dxx, dyy, flag), a->sig, CCV_EOF_SIGN);
		my = *y = ccv_dense_matrix_renew(*y, a->rows, a->cols, CCV_32S | CCV_C1, CCV_32S | CCV_C1, ysig);
	}
	ccv_object_return_if_cached(, db, mx, my);
	ccv_revive_object_if_cached(db, mx, my);
	int i, j, k;
	unsigned char* a_ptr = a->data.u8;
	unsigned char* b_ptr = db->data.u8;
	int* v = (int*)alloca(sizeof(int) * ccv_max(db->rows, db->cols));
	unsigned char* c_ptr = (unsigned char*)alloca(CCV_GET_DATA_TYPE_SIZE(db->type) * db->rows);
	int* x_ptr = mx ? mx->data.i32 : 0;
	int* y_ptr = my ? my->data.i32 : 0;
#define for_block(_for_max, _for_type_b, _for_set_b, _for_get_b, _for_get_a) \
	_for_type_b _dx = dx, _dy = dy, _dxx = dxx, _dyy = dyy; \
	_for_type_b* z = (_for_type_b*)alloca(sizeof(_for_type_b) * (ccv_max(db->rows, db->cols) + 1)); \
	if (_dxx > 1e-6) \
	{ \
		for (i = 0; i < a->rows; i++) \
		{ \
			k = 0; \
			v[0] = 0; \
			z[0] = (_for_type_b)-_for_max; \
			z[1] = (_for_type_b)_for_max; \
			for (j = 1; j < a->cols; j++) \
			{ \
				_for_type_b s; \
				for (;;) \
				{ \
					assert(k >= 0 && k < ccv_max(db->rows, db->cols)); \
					s = ((SGN _for_get_a(a_ptr, j, 0) + _dxx * j * j - _dx * j) - (SGN _for_get_a(a_ptr, v[k], 0) + _dxx * v[k] * v[k] - _dx * v[k])) / (2.0 * _dxx * (j - v[k])); \
					if (s > z[k]) break; \
					--k; \
				} \
				++k; \
				assert(k >= 0 && k < ccv_max(db->rows, db->cols)); \
				v[k] = j; \
				z[k] = s; \
				z[k + 1] = (_for_type_b)_for_max; \
			} \
			assert(z[k + 1] >= a->cols - 1); \
			k = 0; \
			if (mx) \
			{ \
				for (j = 0; j < a->cols; j++) \
				{ \
					while (z[k + 1] < j) \
					{ \
						assert(k >= 0 && k < ccv_max(db->rows, db->cols) - 1); \
						++k; \
					} \
					_for_set_b(b_ptr, j, _dx * (j - v[k]) + _dxx * (j - v[k]) * (j - v[k]) SGN _for_get_a(a_ptr, v[k], 0), 0); \
					x_ptr[j] = j - v[k]; \
				} \
				x_ptr += mx->cols; \
			} else { \
				for (j = 0; j < a->cols; j++) \
				{ \
					while (z[k + 1] < j) \
					{ \
						assert(k >= 0 && k < ccv_max(db->rows, db->cols) - 1); \
						++k; \
					} \
					_for_set_b(b_ptr, j, _dx * (j - v[k]) + _dxx * (j - v[k]) * (j - v[k]) SGN _for_get_a(a_ptr, v[k], 0), 0); \
				} \
			} \
			a_ptr += a->step; \
			b_ptr += db->step; \
		} \
	} else { /* above algorithm cannot handle dxx == 0 properly, below is special casing for that */ \
		assert(mx == 0); \
		for (i = 0; i < a->rows; i++) \
		{ \
			for (j = 0; j < a->cols; j++) \
				_for_set_b(b_ptr, j, SGN _for_get_a(a_ptr, j, 0), 0); \
			for (j = 1; j < a->cols; j++) \
				_for_set_b(b_ptr, j, ccv_min(_for_get_b(b_ptr, j, 0), _for_get_b(b_ptr, j - 1, 0) + _dx), 0); \
			for (j = a->cols - 2; j >= 0; j--) \
				_for_set_b(b_ptr, j, ccv_min(_for_get_b(b_ptr, j, 0), _for_get_b(b_ptr, j + 1, 0) - _dx), 0); \
			a_ptr += a->step; \
			b_ptr += db->step; \
		} \
	} \
	b_ptr = db->data.u8; \
	if (_dyy > 1e-6) \
	{ \
		for (j = 0; j < db->cols; j++) \
		{ \
			for (i = 0; i < db->rows; i++) \
				_for_set_b(c_ptr, i, _for_get_b(b_ptr + i * db->step, j, 0), 0); \
			k = 0; \
			v[0] = 0; \
			z[0] = (_for_type_b)-_for_max; \
			z[1] = (_for_type_b)_for_max; \
			for (i = 1; i < db->rows; i++) \
			{ \
				_for_type_b s; \
				for (;;) \
				{ \
					assert(k >= 0 && k < ccv_max(db->rows, db->cols)); \
					s = ((_for_get_b(c_ptr, i, 0) + _dyy * i * i - _dy * i) - (_for_get_b(c_ptr, v[k], 0) + _dyy * v[k] * v[k] - _dy * v[k])) / (2.0 * _dyy * (i - v[k])); \
					if (s > z[k]) break; \
					--k; \
				} \
				++k; \
				assert(k >= 0 && k < ccv_max(db->rows, db->cols)); \
				v[k] = i; \
				z[k] = s; \
				z[k + 1] = (_for_type_b)_for_max; \
			} \
			assert(z[k + 1] >= db->rows - 1); \
			k = 0; \
			if (my) \
			{ \
				for (i = 0; i < db->rows; i++) \
				{ \
					while (z[k + 1] < i) \
					{ \
						assert(k >= 0 && k < ccv_max(db->rows, db->cols) - 1); \
						++k; \
					} \
					_for_set_b(b_ptr + i * db->step, j, _dy * (i - v[k]) + _dyy * (i - v[k]) * (i - v[k]) + _for_get_b(c_ptr, v[k], 0), 0); \
					y_ptr[i * my->cols] = i - v[k]; \
				} \
				++y_ptr; \
			} else { \
				for (i = 0; i < db->rows; i++) \
				{ \
					while (z[k + 1] < i) \
					{ \
						assert(k >= 0 && k < ccv_max(db->rows, db->cols) - 1); \
						++k; \
					} \
					_for_set_b(b_ptr + i * db->step, j, _dy * (i - v[k]) + _dyy * (i - v[k]) * (i - v[k]) + _for_get_b(c_ptr, v[k], 0), 0); \
				} \
			} \
		} \
	} else { \
		assert(my == 0); \
		for (j = 0; j < db->cols; j++) \
		{ \
			for (i = 1; i < db->rows; i++) \
				_for_set_b(b_ptr + i * db->step, j, ccv_min(_for_get_b(b_ptr + i * db->step, j, 0), _for_get_b(b_ptr + (i - 1) * db->step, j, 0) + _dy), 0); \
			for (i = db->rows - 2; i >= 0; i--) \
				_for_set_b(b_ptr + i * db->step, j, ccv_min(_for_get_b(b_ptr + i * db->step, j, 0), _for_get_b(b_ptr + (i + 1) * db->step, j, 0) - _dy), 0); \
		} \
	}
	if (flag & CCV_NEGATIVE)
	{
#define SGN -
		if (db->type & CCV_64F)
		{
			ccv_matrix_typeof_setter_getter(db->type, ccv_matrix_getter, a->type, for_block, DBL_MAX);
		} else {
			ccv_matrix_typeof_setter_getter(db->type, ccv_matrix_getter, a->type, for_block, FLT_MAX);
		}
#undef SGN
	} else {
#define SGN +
		if (db->type & CCV_64F)
		{
			ccv_matrix_typeof_setter_getter(db->type, ccv_matrix_getter, a->type, for_block, DBL_MAX);
		} else {
			ccv_matrix_typeof_setter_getter(db->type, ccv_matrix_getter, a->type, for_block, FLT_MAX);
		}
#undef SGN
	}
#undef for_block
}
