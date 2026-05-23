#include "ccv.h"
#include "ccv_internal.h"

static void _ccv_wavelet_blur_32f(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, int radius)
{
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	assert(CCV_GET_DATA_TYPE(b->type) == CCV_32F);
	assert(a->rows == b->rows && a->cols == b->cols);
	assert(CCV_GET_CHANNEL(a->type) == CCV_GET_CHANNEL(b->type));
	const int ch = CCV_GET_CHANNEL(a->type);
	const int a_step = a->step / sizeof(float);
	const int b_step = b->step / sizeof(float);
	const float weights[3][3] = {
		{0.0625f, 0.125f, 0.0625f},
		{0.125f, 0.25f, 0.125f},
		{0.0625f, 0.125f, 0.0625f},
	};
	int y, x, c, ky, kx;
	for (y = 0; y < a->rows; y++)
	{
		float* b_ptr = b->data.f32 + y * b_step;
		for (x = 0; x < a->cols; x++)
			for (c = 0; c < ch; c++)
			{
				float v = 0;
				for (ky = -1; ky <= 1; ky++)
				{
					const int yy = ccv_min(ccv_max(y + ky * radius, 0), a->rows - 1);
					const float* a_ptr = a->data.f32 + yy * a_step;
					for (kx = -1; kx <= 1; kx++)
					{
						const int xx = ccv_min(ccv_max(x + kx * radius, 0), a->cols - 1);
						v += a_ptr[xx * ch + c] * weights[ky + 1][kx + 1];
					}
				}
				b_ptr[x * ch + c] = v;
			}
	}
}

void ccv_wavelet_decompose(ccv_dense_matrix_t* a, ccv_dense_matrix_t** high, ccv_dense_matrix_t** low, int type)
{
	assert(CCV_GET_DATA_TYPE(a->type) == CCV_32F);
	ccv_declare_derived_signature(high_sig, a->sig != 0, ccv_sign_with_literal("ccv_wavelet_decompose(high)"), a->sig, CCV_EOF_SIGN);
	ccv_declare_derived_signature(low_sig, a->sig != 0, ccv_sign_with_literal("ccv_wavelet_decompose(low)"), a->sig, CCV_EOF_SIGN);
	const int ch = CCV_GET_CHANNEL(a->type);
	type = (type == 0) ? CCV_32F | ch : CCV_GET_DATA_TYPE(type) | ch;
	assert(CCV_GET_DATA_TYPE(type) == CCV_32F);
	ccv_dense_matrix_t* dhigh = *high = ccv_dense_matrix_renew(*high, a->rows, a->cols, CCV_32F | ch, type, high_sig);
	ccv_dense_matrix_t* dlow = *low = ccv_dense_matrix_renew(*low, a->rows, a->cols, CCV_32F | ch, type, low_sig);
	assert(dhigh && dlow);
	ccv_object_return_if_cached(, dhigh, dlow);
	ccv_revive_object_if_cached(dhigh, dlow);
	ccv_zero(dhigh);
	int y;
	const int row_size = a->cols * ch * sizeof(float);
	for (y = 0; y < a->rows; y++)
		memcpy(dlow->data.u8 + y * dlow->step, a->data.u8 + y * a->step, row_size);
	ccv_dense_matrix_t* blurred = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | ch, 0, 0);
	const int max_radius = ccv_max(1, ccv_min(a->rows, a->cols) / 8);
	const int requested_radius[5] = {1, 2, 4, 8, 16};
	const int high_step = dhigh->step / sizeof(float);
	const int low_step = dlow->step / sizeof(float);
	const int blurred_step = blurred->step / sizeof(float);
	int level, i;
	for (level = 0; level < 5; level++)
	{
		const int radius = ccv_min(requested_radius[level], max_radius);
		_ccv_wavelet_blur_32f(dlow, blurred, radius);
		for (y = 0; y < a->rows; y++)
		{
			float* high_ptr = dhigh->data.f32 + y * high_step;
			float* low_ptr = dlow->data.f32 + y * low_step;
			float* blurred_ptr = blurred->data.f32 + y * blurred_step;
			for (i = 0; i < a->cols * ch; i++)
			{
				high_ptr[i] += low_ptr[i] - blurred_ptr[i];
				low_ptr[i] = blurred_ptr[i];
			}
		}
	}
	ccv_matrix_free(blurred);
}
