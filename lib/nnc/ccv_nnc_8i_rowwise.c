#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include <float.h>
#include "ccv_nnc_8i_rowwise_packed_grids.inc"
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#elif defined(HAVE_MPS)
#include "mps/ccv_nnc_mps.h"
#endif

static int _ccv_nnc_8i_rowwise_x_group_size(const int format)
{
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			return 16;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			return 8;
		default:
			assert(0);
			return 0;
	}
}

static int _ccv_nnc_8i_rowwise_x_group_bits(const int format)
{
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			return 72;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			return 56;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			return 42;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			return 21;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			return 28;
		default:
			assert(0);
			return 0;
	}
}

static size_t _ccv_nnc_8i_rowwise_packed_scale_offset(const int format, const size_t input_length, const size_t row_length)
{
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	const size_t group_size = _ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t group_bits = _ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t payload_size = (row_count * groups_per_row * group_bits + 7) / 8;
	return (payload_size + 127) & -128;
}

CCV_WARN_UNUSED(size_t) ccv_nnc_8i_rowwise_x_data_size(const int format, const int datatype, const size_t input_length, const size_t row_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	const size_t scale_offset = _ccv_nnc_8i_rowwise_packed_scale_offset(format, input_length, row_length);
	return scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
}

static void _ccv_nnc_8i_rowwise_packed_write_bits(uint8_t* const data, const size_t bit_offset, const uint32_t value, const int bits)
{
	int i;
	for (i = 0; i < bits; i++)
		if (value & (1u << i))
			data[(bit_offset + i) >> 3] |= (uint8_t)(1u << ((bit_offset + i) & 7));
}

static uint32_t _ccv_nnc_8i_rowwise_packed_read_bits(const uint8_t* const data, const size_t bit_offset, const int bits)
{
	uint32_t value = 0;
	int i;
	for (i = 0; i < bits; i++)
		if (data[(bit_offset + i) >> 3] & (uint8_t)(1u << ((bit_offset + i) & 7)))
			value |= (1u << i);
	return value;
}

static double _ccv_nnc_8i_rowwise_packed_stored_scale(const double scale, const int datatype)
{
	if (datatype == CCV_16F)
	{
		const float scale_f = (float)scale;
		uint16_t scale_h;
		float stored_scale;
		ccv_float_to_half_precision(&scale_f, &scale_h, 1);
		ccv_half_precision_to_float(&scale_h, &stored_scale, 1);
		return stored_scale;
	} else if (datatype == CCV_16BF) {
		const float scale_f = (float)scale;
		uint16_t scale_bf;
		float stored_scale;
		ccv_float_to_bfloat(&scale_f, &scale_bf, 1);
		ccv_bfloat_to_float(&scale_bf, &stored_scale, 1);
		return stored_scale;
	} else if (datatype == CCV_32F)
		return (float)scale;
	return scale;
}

static void _ccv_nnc_8i_rowwise_packed_store_scale(uint8_t* const scales, const int datatype, const size_t i, const double scale)
{
	if (datatype == CCV_16F)
	{
		const float scale_f = (float)scale;
		ccv_float_to_half_precision(&scale_f, (uint16_t*)scales + i, 1);
	} else if (datatype == CCV_16BF) {
		const float scale_f = (float)scale;
		ccv_float_to_bfloat(&scale_f, (uint16_t*)scales + i, 1);
	} else if (datatype == CCV_32F)
		((float*)scales)[i] = (float)scale;
	else
		((double*)scales)[i] = scale;
}

static double _ccv_nnc_8i_rowwise_packed_load_scale(const uint8_t* const scales, const int datatype, const size_t i)
{
	if (datatype == CCV_16F)
	{
		float scale_f;
		ccv_half_precision_to_float((const uint16_t*)scales + i, &scale_f, 1);
		return scale_f;
	} else if (datatype == CCV_16BF) {
		float scale_f;
		ccv_bfloat_to_float((const uint16_t*)scales + i, &scale_f, 1);
		return scale_f;
	} else if (datatype == CCV_32F)
		return ((const float*)scales)[i];
	return ((const double*)scales)[i];
}

static void _ccv_nnc_8i_rowwise_packed_read_row(const void* const input, const int datatype, const size_t row_start, const size_t row_length, const size_t padded_row_length, double* const row)
{
	size_t j;
	if (datatype == CCV_16F)
	{
		const uint16_t* const f16 = (const uint16_t*)input + row_start;
		for (j = 0; j < row_length; j++)
		{
			float v;
			ccv_half_precision_to_float(f16 + j, &v, 1);
			row[j] = v;
		}
	} else if (datatype == CCV_16BF) {
		const uint16_t* const bf16 = (const uint16_t*)input + row_start;
		for (j = 0; j < row_length; j++)
		{
			float v;
			ccv_bfloat_to_float(bf16 + j, &v, 1);
			row[j] = v;
		}
	} else if (datatype == CCV_32F) {
		const float* const f32 = (const float*)input + row_start;
		for (j = 0; j < row_length; j++)
			row[j] = f32[j];
	} else {
		const double* const f64 = (const double*)input + row_start;
		for (j = 0; j < row_length; j++)
			row[j] = f64[j];
	}
	for (; j < padded_row_length; j++)
		row[j] = 0;
}

static void _ccv_nnc_8i_rowwise_packed_write_value(void* const output, const int datatype, const size_t j, const double v)
{
	if (datatype == CCV_16F)
	{
		const float v_f = (float)v;
		ccv_float_to_half_precision(&v_f, (uint16_t*)output + j, 1);
	} else if (datatype == CCV_16BF) {
		const float v_f = (float)v;
		ccv_float_to_bfloat(&v_f, (uint16_t*)output + j, 1);
	} else if (datatype == CCV_32F)
		((float*)output)[j] = (float)v;
	else
		((double*)output)[j] = v;
}

typedef struct {
	int q[16];
	int q8[16];
	int m;
	int b;
	int z;
	int scale;
	int grid[4];
	uint32_t signs;
} ccv_nnc_8i_rowwise_packed_group_t;

static void _ccv_nnc_8i_rowwise_packed_quant_q4(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int best_q[16] = {0};
	int best_q8[16] = {0};
	int best_m = 1, best_b = 0;
	int m, b, j;
	for (m = 1; m <= 16; m++)
		for (b = -8; b <= 7; b++)
		{
			if (-8 * m + b < -127 || 7 * m + b > 127)
				continue;
			double sse = 0;
			int q[16];
			int q8[16];
			for (j = 0; j < 16; j++)
			{
				q[j] = ccv_clamp((int)lrint((y[j] - b) / m), -8, 7);
				q8[j] = q[j] * m + b;
				const double d = q8[j] - y[j];
				sse += w[j] * d * d;
			}
			if (sse < best_sse)
			{
				best_sse = sse;
				best_m = m;
				best_b = b;
				memcpy(best_q, q, sizeof(best_q));
				memcpy(best_q8, q8, sizeof(best_q8));
			}
		}
	group->m = best_m;
	group->b = best_b;
	memcpy(group->q, best_q, sizeof(best_q));
	memcpy(group->q8, best_q8, sizeof(best_q8));
}

static void _ccv_nnc_8i_rowwise_packed_quant_q3(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int best_q[16] = {0};
	int best_q8[16] = {0};
	int best_m = 1, best_b = 0;
	int m, b, j;
	for (m = 1; m <= 32; m++)
		for (b = -8; b <= 6; b += 2)
		{
			if (-4 * m + b < -127 || 3 * m + b > 127)
				continue;
			double sse = 0;
			int q[16];
			int q8[16];
			for (j = 0; j < 16; j++)
			{
				q[j] = ccv_clamp((int)lrint((y[j] - b) / m), -4, 3);
				q8[j] = q[j] * m + b;
				const double d = q8[j] - y[j];
				sse += w[j] * d * d;
			}
			if (sse < best_sse)
			{
				best_sse = sse;
				best_m = m;
				best_b = b;
				memcpy(best_q, q, sizeof(best_q));
				memcpy(best_q8, q8, sizeof(best_q8));
			}
		}
	group->m = best_m;
	group->b = best_b;
	memcpy(group->q, best_q, sizeof(best_q));
	memcpy(group->q8, best_q8, sizeof(best_q8));
}

static void _ccv_nnc_8i_rowwise_packed_quant_q2(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int best_q[16] = {0};
	int best_q8[16] = {0};
	int best_m = 1, best_z = 0;
	int m, z, j;
	for (m = 1; m <= 64; m++)
		for (z = 0; z <= 120; z += 8)
		{
			if (3 * m - z > 127)
				continue;
			double sse = 0;
			int q[16];
			int q8[16];
			for (j = 0; j < 16; j++)
			{
				q[j] = ccv_clamp((int)lrint((y[j] + z) / m), 0, 3);
				q8[j] = q[j] * m - z;
				const double d = q8[j] - y[j];
				sse += w[j] * d * d;
			}
			if (sse < best_sse)
			{
				best_sse = sse;
				best_m = m;
				best_z = z;
				memcpy(best_q, q, sizeof(best_q));
				memcpy(best_q8, q8, sizeof(best_q8));
			}
		}
	group->m = best_m;
	group->z = best_z;
	memcpy(group->q, best_q, sizeof(best_q));
	memcpy(group->q8, best_q8, sizeof(best_q8));
}

static int _ccv_nnc_8i_rowwise_packed_iq2_value(const uint64_t* const grid, const int index, const int lane)
{
	const int v = (int)((grid[index] >> (lane * 8)) & 0xff);
	if (v == 8)
		return 1;
	if (v == 25)
		return 3;
	assert(v == 43);
	return 5;
}

#define CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE (1024)

static int ccv_nnc_8i_rowwise_packed_iq2s_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq2s_level[CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE][8];
static uint8_t ccv_nnc_8i_rowwise_packed_iq2s_scale_level[65][3];
static uint16_t ccv_nnc_8i_rowwise_packed_iq2s_scale_level2[65][3];
static uint8_t ccv_nnc_8i_rowwise_packed_iq2s_scaled_value[65][CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE][8];

static void _ccv_nnc_8i_rowwise_packed_iq2s_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2s_initialized)
		return;
	int index, j, scale;
	for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE; index++)
		for (j = 0; j < 8; j++)
		{
			const int v = _ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2s_grid, index, j);
			ccv_nnc_8i_rowwise_packed_iq2s_level[index][j] = (uint8_t)((v - 1) / 2);
		}
	for (scale = 1; scale <= 64; scale++)
	{
		for (j = 0; j < 3; j++)
		{
			const int v = ccv_min((1 + j * 2) * scale, 127);
			ccv_nnc_8i_rowwise_packed_iq2s_scale_level[scale][j] = (uint8_t)v;
			ccv_nnc_8i_rowwise_packed_iq2s_scale_level2[scale][j] = (uint16_t)(v * v);
		}
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE; index++)
			for (j = 0; j < 8; j++)
				ccv_nnc_8i_rowwise_packed_iq2s_scaled_value[scale][index][j] = ccv_nnc_8i_rowwise_packed_iq2s_scale_level[scale][ccv_nnc_8i_rowwise_packed_iq2s_level[index][j]];
	}
	ccv_nnc_8i_rowwise_packed_iq2s_initialized = 1;
}

static double _ccv_nnc_8i_rowwise_packed_iq2s_sse(const double* const ay, const double* const w, const int lane, const int scale, const int index)
{
	const uint8_t* const mag = ccv_nnc_8i_rowwise_packed_iq2s_scaled_value[scale][index];
	double d = (double)mag[0] - ay[lane];
	double sse = w[lane] * d * d;
	d = (double)mag[1] - ay[lane + 1];
	sse += w[lane + 1] * d * d;
	d = (double)mag[2] - ay[lane + 2];
	sse += w[lane + 2] * d * d;
	d = (double)mag[3] - ay[lane + 3];
	sse += w[lane + 3] * d * d;
	d = (double)mag[4] - ay[lane + 4];
	sse += w[lane + 4] * d * d;
	d = (double)mag[5] - ay[lane + 5];
	sse += w[lane + 5] * d * d;
	d = (double)mag[6] - ay[lane + 6];
	sse += w[lane + 6] * d * d;
	d = (double)mag[7] - ay[lane + 7];
	sse += w[lane + 7] * d * d;
	return sse;
}

static int _ccv_nnc_8i_rowwise_packed_iq3xxs_value(const int index, const int lane)
{
	const int v = (int)((ccv_nnc_8i_rowwise_packed_iq3xxs_grid[index] >> (lane * 8)) & 0xff);
	switch (v)
	{
		case 4: return 1;
		case 12: return 3;
		case 20: return 5;
		case 28: return 7;
		case 36: return 9;
		case 44: return 11;
		case 52: return 13;
		default:
			assert(v == 62);
			return 15;
	}
}

static int _ccv_nnc_8i_rowwise_packed_iq3s_value(const int index, const int lane)
{
	return (int)((ccv_nnc_8i_rowwise_packed_iq3s_grid[index] >> (lane * 8)) & 0xff);
}

#define CCV_NNC_8I_ROWWISE_PACKED_IQ3S_GRID_SIZE (512)
#define CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE (256)

static int ccv_nnc_8i_rowwise_packed_iq3s_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq3s_scaled_value[17][CCV_NNC_8I_ROWWISE_PACKED_IQ3S_GRID_SIZE][4];

static int ccv_nnc_8i_rowwise_packed_iq3xxs_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[17][CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE][4];

static void _ccv_nnc_8i_rowwise_packed_iq3s_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq3s_initialized)
		return;
	int index, j, scale;
	for (scale = 1; scale <= 16; scale++)
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3S_GRID_SIZE; index++)
			for (j = 0; j < 4; j++)
				ccv_nnc_8i_rowwise_packed_iq3s_scaled_value[scale][index][j] = (uint8_t)ccv_min(_ccv_nnc_8i_rowwise_packed_iq3s_value(index, j) * scale, 127);
	ccv_nnc_8i_rowwise_packed_iq3s_initialized = 1;
}

static void _ccv_nnc_8i_rowwise_packed_iq3xxs_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq3xxs_initialized)
		return;
	int index, j, scale;
	for (scale = 1; scale <= 16; scale++)
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE; index++)
			for (j = 0; j < 4; j++)
				ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[scale][index][j] = (uint8_t)ccv_min(_ccv_nnc_8i_rowwise_packed_iq3xxs_value(index, j) * scale, 127);
	ccv_nnc_8i_rowwise_packed_iq3xxs_initialized = 1;
}

static double _ccv_nnc_8i_rowwise_packed_iq3s_sse(const double* const ay, const double* const w, const int lane, const int scale, const int index)
{
	const uint8_t* const mag = ccv_nnc_8i_rowwise_packed_iq3s_scaled_value[scale][index];
	double d = (double)mag[0] - ay[lane];
	double sse = w[lane] * d * d;
	d = (double)mag[1] - ay[lane + 1];
	sse += w[lane + 1] * d * d;
	d = (double)mag[2] - ay[lane + 2];
	sse += w[lane + 2] * d * d;
	d = (double)mag[3] - ay[lane + 3];
	sse += w[lane + 3] * d * d;
	return sse;
}

static double _ccv_nnc_8i_rowwise_packed_iq3xxs_sse(const double* const ay, const double* const w, const int lane, const int scale, const int index)
{
	const uint8_t* const mag = ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[scale][index];
	double d = (double)mag[0] - ay[lane];
	double sse = w[lane] * d * d;
	d = (double)mag[1] - ay[lane + 1];
	sse += w[lane + 1] * d * d;
	d = (double)mag[2] - ay[lane + 2];
	sse += w[lane + 2] * d * d;
	d = (double)mag[3] - ay[lane + 3];
	sse += w[lane + 3] * d * d;
	return sse;
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq2_s(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	assert(ccv_nnc_8i_rowwise_packed_iq2s_initialized);
	double best_sse = DBL_MAX;
	int best_scale = 1;
	int best_grid[2] = {0};
	double ay[16];
	double wy[16];
	uint32_t signs = 0;
	int j;
	for (j = 0; j < 16; j++)
	{
		ay[j] = fabs(y[j]);
		wy[j] = w[j] * ay[j];
		if (y[j] < 0)
			signs |= (1u << j);
	}
	double sub_sse[2][65];
	int sub_grid[2][65];
	int sg, scale;
	for (sg = 0; sg < 2; sg++)
		for (scale = 1; scale <= 64; scale++)
		{
			sub_sse[sg][scale] = DBL_MAX;
			sub_grid[sg][scale] = 0;
		}
	for (sg = 0; sg < 2; sg++)
	{
		const int lane = sg * 8;
		double sum_y2 = 0;
		for (j = 0; j < 8; j++)
			sum_y2 += w[lane + j] * ay[lane + j] * ay[lane + j];
		int index;
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE; index++)
		{
			double sw[3] = {0};
			double swy[3] = {0};
			for (j = 0; j < 8; j++)
			{
				const int level = ccv_nnc_8i_rowwise_packed_iq2s_level[index][j];
				sw[level] += w[lane + j];
				swy[level] += wy[lane + j];
			}
			for (scale = 1; scale <= 64; scale++)
			{
				const double sse = sum_y2 +
					sw[0] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level2[scale][0] - 2 * swy[0] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level[scale][0] +
					sw[1] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level2[scale][1] - 2 * swy[1] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level[scale][1] +
					sw[2] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level2[scale][2] - 2 * swy[2] * (double)ccv_nnc_8i_rowwise_packed_iq2s_scale_level[scale][2];
				if (sub_sse[sg][scale] == DBL_MAX || sse <= sub_sse[sg][scale] + ccv_max(1., fabs(sub_sse[sg][scale])) * 1e-9)
				{
					const double exact_sse = _ccv_nnc_8i_rowwise_packed_iq2s_sse(ay, w, lane, scale, index);
					if (exact_sse < sub_sse[sg][scale])
					{
						sub_sse[sg][scale] = exact_sse;
						sub_grid[sg][scale] = index;
					}
				}
			}
		}
	}
	for (scale = 1; scale <= 64; scale++)
	{
		const double group_sse = sub_sse[0][scale] + sub_sse[1][scale];
		if (group_sse < best_sse)
		{
			best_sse = group_sse;
			best_scale = scale;
			best_grid[0] = sub_grid[0][scale];
			best_grid[1] = sub_grid[1][scale];
		}
	}
	group->scale = best_scale;
	group->signs = signs;
	memcpy(group->grid, best_grid, sizeof(best_grid));
	for (j = 0; j < 16; j++)
	{
		const int sg = j >> 3;
		const int lane = j & 7;
		const int mag = ccv_nnc_8i_rowwise_packed_iq2s_scaled_value[best_scale][best_grid[sg]][lane];
		group->q8[j] = (signs & (1u << j)) ? -mag : mag;
	}
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq2_xs(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	static const int scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
	double best_sse = DBL_MAX;
	int best_scale_code = 0;
	int best_grid = 0;
	int best_q8[16] = {0};
	uint32_t signs = 0;
	int j;
	for (j = 0; j < 8; j++)
		if (y[j] < 0)
			signs |= (1u << j);
	int scale_code;
	for (scale_code = 0; scale_code < 16; scale_code++)
	{
		const int scale = scales[scale_code];
		int index;
		for (index = 0; index < 512; index++)
		{
			double sse = 0;
			int q8[16] = {0};
			for (j = 0; j < 8; j++)
			{
				const int mag = ccv_min(_ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2xs_grid, index, j) * scale, 127);
				q8[j] = (signs & (1u << j)) ? -mag : mag;
				const double d = q8[j] - y[j];
				sse += w[j] * d * d;
			}
			if (sse < best_sse)
			{
				best_sse = sse;
				best_scale_code = scale_code;
				best_grid = index;
				memcpy(best_q8, q8, sizeof(best_q8));
			}
		}
	}
	group->scale = best_scale_code;
	group->grid[0] = best_grid;
	group->signs = signs;
	memcpy(group->q8, best_q8, sizeof(best_q8));
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq3_s(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	assert(ccv_nnc_8i_rowwise_packed_iq3s_initialized);
	double best_sse = DBL_MAX;
	int best_scale = 1;
	int best_grid[4] = {0};
	double ay[16];
	uint32_t signs = 0;
	int j;
	for (j = 0; j < 16; j++)
	{
		ay[j] = fabs(y[j]);
		if (y[j] < 0)
			signs |= (1u << j);
	}
	int scale;
	for (scale = 1; scale <= 16; scale++)
	{
		double group_sse = 0;
		int group_grid[4] = {0};
		int sg;
		for (sg = 0; sg < 4; sg++)
		{
			double best_sub_sse = DBL_MAX;
			int best_sub_grid = 0;
			const int lane = sg * 4;
			int index;
			for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3S_GRID_SIZE; index++)
			{
				const double sse = _ccv_nnc_8i_rowwise_packed_iq3s_sse(ay, w, lane, scale, index);
				if (sse < best_sub_sse)
				{
					best_sub_sse = sse;
					best_sub_grid = index;
				}
			}
			group_sse += best_sub_sse;
			group_grid[sg] = best_sub_grid;
		}
		if (group_sse < best_sse)
		{
			best_sse = group_sse;
			best_scale = scale;
			memcpy(best_grid, group_grid, sizeof(best_grid));
		}
	}
	group->scale = best_scale;
	group->signs = signs;
	memcpy(group->grid, best_grid, sizeof(best_grid));
	for (j = 0; j < 16; j++)
	{
		const int sg = j >> 2;
		const int lane = j & 3;
		const int mag = ccv_nnc_8i_rowwise_packed_iq3s_scaled_value[best_scale][best_grid[sg]][lane];
		group->q8[j] = (signs & (1u << j)) ? -mag : mag;
	}
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq3_xxs(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	assert(ccv_nnc_8i_rowwise_packed_iq3xxs_initialized);
	double best_sse = DBL_MAX;
	int best_scale = 1;
	int best_grid[2] = {0};
	double ay[8];
	uint32_t signs = 0;
	int j;
	for (j = 0; j < 8; j++)
	{
		ay[j] = fabs(y[j]);
		if (y[j] < 0)
			signs |= (1u << j);
	}
	int scale;
	for (scale = 1; scale <= 16; scale++)
	{
		double group_sse = 0;
		int group_grid[2] = {0};
		int sg;
		for (sg = 0; sg < 2; sg++)
		{
			double best_sub_sse = DBL_MAX;
			int best_sub_grid = 0;
			const int lane = sg * 4;
			int index;
			for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE; index++)
			{
				const double sse = _ccv_nnc_8i_rowwise_packed_iq3xxs_sse(ay, w, lane, scale, index);
				if (sse < best_sub_sse)
				{
					best_sub_sse = sse;
					best_sub_grid = index;
				}
			}
			group_sse += best_sub_sse;
			group_grid[sg] = best_sub_grid;
		}
		if (group_sse < best_sse)
		{
			best_sse = group_sse;
			best_scale = scale;
			memcpy(best_grid, group_grid, sizeof(best_grid));
		}
	}
	group->scale = best_scale;
	group->signs = signs;
	memcpy(group->grid, best_grid, sizeof(best_grid));
	memset(group->q8, 0, sizeof(group->q8));
	for (j = 0; j < 8; j++)
	{
		const int sg = j >> 2;
		const int lane = j & 3;
		const int mag = ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[best_scale][best_grid[sg]][lane];
		group->q8[j] = (signs & (1u << j)) ? -mag : mag;
	}
}

static void _ccv_nnc_8i_rowwise_packed_quant_group(const int format, const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			_ccv_nnc_8i_rowwise_packed_quant_q4(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
			_ccv_nnc_8i_rowwise_packed_quant_q3(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
			_ccv_nnc_8i_rowwise_packed_quant_q2(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			_ccv_nnc_8i_rowwise_packed_quant_iq2_s(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			_ccv_nnc_8i_rowwise_packed_quant_iq2_xs(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			_ccv_nnc_8i_rowwise_packed_quant_iq3_s(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			_ccv_nnc_8i_rowwise_packed_quant_iq3_xxs(y, w, group);
			break;
		default:
			assert(0);
	}
}

static void _ccv_nnc_8i_rowwise_packed_pack_group(uint8_t* const output, const size_t group_index, const int format, const ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	const size_t bit_offset = group_index * _ccv_nnc_8i_rowwise_x_group_bits(format);
	size_t bit = bit_offset;
	int j;
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			for (j = 0; j < 16; j++, bit += 4)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->q[j] + 8), 4);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->m - 1), 4);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 4, (uint32_t)(group->b + 8), 4);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
			for (j = 0; j < 16; j++, bit += 3)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->q[j] + 4), 3);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->m - 1), 5);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 5, (uint32_t)(group->b / 2 + 4), 3);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
			for (j = 0; j < 16; j++, bit += 2)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)group->q[j], 2);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->m - 1), 6);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 6, (uint32_t)(group->z >> 3), 4);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)group->grid[0], 10);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 10, (uint32_t)group->grid[1], 10);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 20, group->signs, 16);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 36, (uint32_t)(group->scale - 1), 6);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)group->grid[0], 9);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 9, group->signs, 8);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 17, (uint32_t)group->scale, 4);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			for (j = 0; j < 4; j++)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + j * 9, (uint32_t)group->grid[j], 9);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 36, group->signs, 16);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 52, (uint32_t)(group->scale - 1), 4);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)group->grid[0], 8);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 8, (uint32_t)group->grid[1], 8);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 16, group->signs, 8);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 24, (uint32_t)(group->scale - 1), 4);
			break;
		default:
			assert(0);
	}
}

static void _ccv_nnc_8i_rowwise_packed_decode_group(const uint8_t* const input, const size_t group_index, const int format, int* const q8)
{
	static const int q2_xs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
	const size_t bit_offset = group_index * _ccv_nnc_8i_rowwise_x_group_bits(format);
	size_t bit = bit_offset;
	int j;
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_Q4_K: {
			int q[16];
			for (j = 0; j < 16; j++, bit += 4)
				q[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 4) - 8;
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 4) + 1;
			const int b = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 4, 4) - 8;
			for (j = 0; j < 16; j++)
				q8[j] = q[j] * m + b;
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_Q3_K: {
			int q[16];
			for (j = 0; j < 16; j++, bit += 3)
				q[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 3) - 4;
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 5) + 1;
			const int b = ((int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 5, 3) - 4) << 1;
			for (j = 0; j < 16; j++)
				q8[j] = q[j] * m + b;
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_Q2_K: {
			int q[16];
			for (j = 0; j < 16; j++, bit += 2)
				q[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 2);
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 6) + 1;
			const int z = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 6, 4) << 3;
			for (j = 0; j < 16; j++)
				q8[j] = q[j] * m - z;
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S: {
			const int grid0 = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 10);
			const int grid1 = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 10, 10);
			const uint32_t signs = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 20, 16);
			const int scale = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 36, 6) + 1;
			for (j = 0; j < 8; j++)
			{
				const int mag0 = ccv_min(_ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2s_grid, grid0, j) * scale, 127);
				const int mag1 = ccv_min(_ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2s_grid, grid1, j) * scale, 127);
				q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
				q8[8 + j] = (signs & (1u << (8 + j))) ? -mag1 : mag1;
			}
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS: {
			const int grid0 = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 9);
			const uint32_t signs = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 9, 8);
			const int scale = q2_xs_scales[_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 17, 4)];
			for (j = 0; j < 8; j++)
			{
				const int mag = ccv_min(_ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2xs_grid, grid0, j) * scale, 127);
				q8[j] = (signs & (1u << j)) ? -mag : mag;
			}
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S: {
			int grid[4];
			for (j = 0; j < 4; j++)
				grid[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + j * 9, 9);
			const uint32_t signs = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 36, 16);
			const int scale = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 52, 4) + 1;
			int sg;
			for (sg = 0; sg < 4; sg++)
				for (j = 0; j < 4; j++)
				{
					const int lane = sg * 4 + j;
					const int mag = ccv_min(_ccv_nnc_8i_rowwise_packed_iq3s_value(grid[sg], j) * scale, 127);
					q8[lane] = (signs & (1u << lane)) ? -mag : mag;
				}
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS: {
			const int grid0 = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 8);
			const int grid1 = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 8, 8);
			const uint32_t signs = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 16, 8);
			const int scale = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 24, 4) + 1;
			for (j = 0; j < 4; j++)
			{
				const int mag0 = ccv_min(_ccv_nnc_8i_rowwise_packed_iq3xxs_value(grid0, j) * scale, 127);
				const int mag1 = ccv_min(_ccv_nnc_8i_rowwise_packed_iq3xxs_value(grid1, j) * scale, 127);
				q8[j] = (signs & (1u << j)) ? -mag0 : mag0;
				q8[4 + j] = (signs & (1u << (4 + j))) ? -mag1 : mag1;
			}
			break;
		}
		default:
			assert(0);
	}
}

CCV_WARN_UNUSED(size_t) ccv_nnc_quantize_8i_rowwise_x(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, const int format, const float* const imatrix, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	const size_t group_size = _ccv_nnc_8i_rowwise_x_group_size(format);
	const int group_bits = _ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t padded_row_length = groups_per_row * group_size;
	const size_t scale_offset = _ccv_nnc_8i_rowwise_packed_scale_offset(format, input_length, row_length);
	const size_t output_size = scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	assert(output_length >= output_size);
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			_ccv_nnc_8i_rowwise_packed_iq2s_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			_ccv_nnc_8i_rowwise_packed_iq3s_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			_ccv_nnc_8i_rowwise_packed_iq3xxs_init();
			break;
	}
	uint8_t* const u8 = (uint8_t*)output;
	uint8_t* const scales = u8 + scale_offset;
	memset(u8, 0, scale_offset);
	const size_t row_bits = groups_per_row * group_bits;
	size_t rows_per_chunk;
	switch (row_bits & 7)
	{
		case 0:
			rows_per_chunk = 1;
			break;
		case 4:
			rows_per_chunk = 2;
			break;
		case 2:
		case 6:
			rows_per_chunk = 4;
			break;
		default:
			rows_per_chunk = 8;
			break;
	}
	const size_t row_chunks = (row_count + rows_per_chunk - 1) / rows_per_chunk;
	parallel_for(chunk_idx, (int)row_chunks) {
		const size_t chunk_begin = (size_t)chunk_idx * rows_per_chunk;
		const size_t chunk_end = ccv_min(chunk_begin + rows_per_chunk, row_count);
		size_t i;
		for (i = chunk_begin; i < chunk_end; i++)
		{
			double* const row = (double*)ccmalloc(sizeof(double) * padded_row_length);
			double* const weights = (double*)ccmalloc(sizeof(double) * padded_row_length);
			int* const q8 = (int*)ccmalloc(sizeof(int) * padded_row_length);
			const size_t row_start = i * row_length;
			_ccv_nnc_8i_rowwise_packed_read_row(input, datatype, row_start, row_length, padded_row_length, row);
			double max_abs = 0;
			size_t j;
			for (j = 0; j < row_length; j++)
			{
				max_abs = ccv_max(max_abs, fabs(row[j]));
				weights[j] = imatrix ? ccv_max((double)imatrix[j], 0.) : 1.;
			}
			for (; j < padded_row_length; j++)
				weights[j] = 0;
			double scale = max_abs / 127.;
			double best_scale = 0;
			double best_sse = DBL_MAX;
			int k;
			for (k = 0; k < 8; k++)
			{
				const double stored_scale = _ccv_nnc_8i_rowwise_packed_stored_scale(scale, datatype);
				if (!(stored_scale > 0))
					break;
				size_t g;
				for (g = 0; g < groups_per_row; g++)
				{
					double y[16] = {0};
					double w[16] = {0};
					for (j = 0; j < group_size; j++)
					{
						y[j] = row[g * group_size + j] / stored_scale;
						w[j] = weights[g * group_size + j];
					}
					ccv_nnc_8i_rowwise_packed_group_t group;
					_ccv_nnc_8i_rowwise_packed_quant_group(format, y, w, &group);
					memcpy(q8 + g * group_size, group.q8, sizeof(int) * group_size);
				}
				double sse = 0;
				double sum_qx = 0;
				double sum_qq = 0;
				for (j = 0; j < row_length; j++)
				{
					const double d = row[j] - stored_scale * q8[j];
					sse += weights[j] * d * d;
					sum_qx += weights[j] * q8[j] * row[j];
					sum_qq += weights[j] * q8[j] * q8[j];
				}
				if (sse < best_sse)
				{
					best_sse = sse;
					best_scale = stored_scale;
				}
				if (!(sum_qq > 0) || !(sum_qx > 0))
					break;
				const double next_scale = sum_qx / sum_qq;
				if (_ccv_nnc_8i_rowwise_packed_stored_scale(next_scale, datatype) == stored_scale)
					break;
				scale = next_scale;
			}
			_ccv_nnc_8i_rowwise_packed_store_scale(scales, datatype, i, best_scale);
			const double final_scale = best_scale > 0 ? best_scale : 1;
			size_t g;
			for (g = 0; g < groups_per_row; g++)
			{
				double y[16] = {0};
				double w[16] = {0};
				for (j = 0; j < group_size; j++)
				{
					y[j] = row[g * group_size + j] / final_scale;
					w[j] = weights[g * group_size + j];
				}
				ccv_nnc_8i_rowwise_packed_group_t group;
				_ccv_nnc_8i_rowwise_packed_quant_group(format, y, w, &group);
				_ccv_nnc_8i_rowwise_packed_pack_group(u8, i * groups_per_row + g, format, &group);
			}
			ccfree(q8);
			ccfree(weights);
			ccfree(row);
		}
	} parallel_endfor
	return output_size;
}

void ccv_nnc_dequantize_8i_rowwise_x(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	assert(row_length > 0);
	assert(output_length % row_length == 0);
	const size_t row_count = output_length / row_length;
	const size_t group_size = _ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t scale_offset = _ccv_nnc_8i_rowwise_packed_scale_offset(format, output_length, row_length);
	assert(input_length >= scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype));
	const uint8_t* const u8 = (const uint8_t*)input;
	const uint8_t* const scales = u8 + scale_offset;
	parallel_for(i, (int)row_count) {
		const double scale = _ccv_nnc_8i_rowwise_packed_load_scale(scales, datatype, i);
		size_t g;
		for (g = 0; g < groups_per_row; g++)
		{
			int q8[16] = {0};
			_ccv_nnc_8i_rowwise_packed_decode_group(u8, (size_t)i * groups_per_row + g, format, q8);
			size_t j;
			for (j = 0; j < group_size; j++)
			{
				const size_t col = g * group_size + j;
				if (col < row_length)
					_ccv_nnc_8i_rowwise_packed_write_value(output, datatype, (size_t)i * row_length + col, scale * q8[j]);
			}
		}
	} parallel_endfor
}

static inline int _ccv_nnc_8i_rowwise_quantize(const double v, const double inv_scale)
{
	const int q = (int)lrint(v * inv_scale);
	return ccv_clamp(q, -127, 127);
}

static float _ccv_nnc_quantize_8i_rowwise_16f(const uint16_t* const row, const size_t row_length, int8_t* const q)
{
	size_t j;
	double max_abs = 0;
	for (j = 0; j < row_length; j++)
	{
		float v;
		ccv_half_precision_to_float(row + j, &v, 1);
		max_abs = ccv_max(max_abs, fabs(v));
	}
	if (max_abs == 0)
	{
		memset(q, 0, row_length);
		return 0;
	}
	double scale = max_abs / 127.;
	float best_scale = 0;
	double best_sse = DBL_MAX;
	int k;
	for (k = 0; k < 8; k++)
	{
		// Round with the scale that will actually be stored, then refit scale by least squares.
		const float scale_f = (float)scale;
		uint16_t scale_h;
		float stored_scale;
		ccv_float_to_half_precision(&scale_f, &scale_h, 1);
		ccv_half_precision_to_float(&scale_h, &stored_scale, 1);
		if (!(stored_scale > 0))
			break;
		const double inv_scale = 1. / stored_scale;
		double sum_qx = 0;
		double sum_qq = 0;
		double sse = 0;
		for (j = 0; j < row_length; j++)
		{
			float v_f;
			ccv_half_precision_to_float(row + j, &v_f, 1);
			const double v = v_f;
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += d * d;
			sum_qx += qj * v;
			sum_qq += qj * qj;
		}
		if (sse < best_sse)
		{
			best_sse = sse;
			best_scale = stored_scale;
		}
		if (!(sum_qq > 0) || !(sum_qx > 0))
			break;
		const double next_scale = sum_qx / sum_qq;
		const float next_scale_f = (float)next_scale;
		uint16_t next_scale_h;
		float next_stored_scale;
		ccv_float_to_half_precision(&next_scale_f, &next_scale_h, 1);
		ccv_half_precision_to_float(&next_scale_h, &next_stored_scale, 1);
		if (next_stored_scale == stored_scale)
			break;
		scale = next_scale;
	}
	if (!(best_scale > 0))
	{
		memset(q, 0, row_length);
		return 0;
	}
	const double inv_scale = 1. / best_scale;
	for (j = 0; j < row_length; j++)
	{
		float v;
		ccv_half_precision_to_float(row + j, &v, 1);
		q[j] = (int8_t)_ccv_nnc_8i_rowwise_quantize(v, inv_scale);
	}
	return best_scale;
}

static float _ccv_nnc_quantize_8i_rowwise_16bf(const uint16_t* const row, const size_t row_length, int8_t* const q)
{
	size_t j;
	double max_abs = 0;
	for (j = 0; j < row_length; j++)
	{
		float v;
		ccv_bfloat_to_float(row + j, &v, 1);
		max_abs = ccv_max(max_abs, fabs(v));
	}
	if (max_abs == 0)
	{
		memset(q, 0, row_length);
		return 0;
	}
	double scale = max_abs / 127.;
	float best_scale = 0;
	double best_sse = DBL_MAX;
	int k;
	for (k = 0; k < 8; k++)
	{
		const float scale_f = (float)scale;
		uint16_t scale_bf;
		float stored_scale;
		ccv_float_to_bfloat(&scale_f, &scale_bf, 1);
		ccv_bfloat_to_float(&scale_bf, &stored_scale, 1);
		if (!(stored_scale > 0))
			break;
		const double inv_scale = 1. / stored_scale;
		double sum_qx = 0;
		double sum_qq = 0;
		double sse = 0;
		for (j = 0; j < row_length; j++)
		{
			float v_f;
			ccv_bfloat_to_float(row + j, &v_f, 1);
			const double v = v_f;
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += d * d;
			sum_qx += qj * v;
			sum_qq += qj * qj;
		}
		if (sse < best_sse)
		{
			best_sse = sse;
			best_scale = stored_scale;
		}
		if (!(sum_qq > 0) || !(sum_qx > 0))
			break;
		const double next_scale = sum_qx / sum_qq;
		const float next_scale_f = (float)next_scale;
		uint16_t next_scale_bf;
		float next_stored_scale;
		ccv_float_to_bfloat(&next_scale_f, &next_scale_bf, 1);
		ccv_bfloat_to_float(&next_scale_bf, &next_stored_scale, 1);
		if (next_stored_scale == stored_scale)
			break;
		scale = next_scale;
	}
	if (!(best_scale > 0))
	{
		memset(q, 0, row_length);
		return 0;
	}
	const double inv_scale = 1. / best_scale;
	for (j = 0; j < row_length; j++)
	{
		float v;
		ccv_bfloat_to_float(row + j, &v, 1);
		q[j] = (int8_t)_ccv_nnc_8i_rowwise_quantize(v, inv_scale);
	}
	return best_scale;
}

static float _ccv_nnc_quantize_8i_rowwise_32f(const float* const row, const size_t row_length, int8_t* const q)
{
	size_t j;
	double max_abs = 0;
	for (j = 0; j < row_length; j++)
		max_abs = ccv_max(max_abs, fabs(row[j]));
	if (max_abs == 0)
	{
		memset(q, 0, row_length);
		return 0;
	}
	double scale = max_abs / 127.;
	float best_scale = 0;
	double best_sse = DBL_MAX;
	int k;
	for (k = 0; k < 8; k++)
	{
		const float stored_scale = (float)scale;
		if (!(stored_scale > 0))
			break;
		const double inv_scale = 1. / stored_scale;
		double sum_qx = 0;
		double sum_qq = 0;
		double sse = 0;
		for (j = 0; j < row_length; j++)
		{
			const double v = row[j];
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += d * d;
			sum_qx += qj * v;
			sum_qq += qj * qj;
		}
		if (sse < best_sse)
		{
			best_sse = sse;
			best_scale = stored_scale;
		}
		if (!(sum_qq > 0) || !(sum_qx > 0))
			break;
		const double next_scale = sum_qx / sum_qq;
		if ((float)next_scale == stored_scale)
			break;
		scale = next_scale;
	}
	if (!(best_scale > 0))
	{
		memset(q, 0, row_length);
		return 0;
	}
	const double inv_scale = 1. / best_scale;
	for (j = 0; j < row_length; j++)
		q[j] = (int8_t)_ccv_nnc_8i_rowwise_quantize(row[j], inv_scale);
	return best_scale;
}

static double _ccv_nnc_quantize_8i_rowwise_64f(const double* const row, const size_t row_length, int8_t* const q)
{
	size_t j;
	double max_abs = 0;
	for (j = 0; j < row_length; j++)
		max_abs = ccv_max(max_abs, fabs(row[j]));
	if (max_abs == 0)
	{
		memset(q, 0, row_length);
		return 0;
	}
	double scale = max_abs / 127.;
	double best_scale = 0;
	double best_sse = DBL_MAX;
	int k;
	for (k = 0; k < 8; k++)
	{
		const double stored_scale = scale;
		if (!(stored_scale > 0))
			break;
		const double inv_scale = 1. / stored_scale;
		double sum_qx = 0;
		double sum_qq = 0;
		double sse = 0;
		for (j = 0; j < row_length; j++)
		{
			const double v = row[j];
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += d * d;
			sum_qx += qj * v;
			sum_qq += qj * qj;
		}
		if (sse < best_sse)
		{
			best_sse = sse;
			best_scale = stored_scale;
		}
		if (!(sum_qq > 0) || !(sum_qx > 0))
			break;
		const double next_scale = sum_qx / sum_qq;
		if (next_scale == stored_scale)
			break;
		scale = next_scale;
	}
	if (!(best_scale > 0))
	{
		memset(q, 0, row_length);
		return 0;
	}
	const double inv_scale = 1. / best_scale;
	for (j = 0; j < row_length; j++)
		q[j] = (int8_t)_ccv_nnc_8i_rowwise_quantize(row[j], inv_scale);
	return best_scale;
}

CCV_WARN_UNUSED(size_t) ccv_nnc_quantize_8i_rowwise(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	const size_t scale_offset = (input_length + 127) & -128;
	const size_t scale_size = row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	assert(output_length >= scale_offset + scale_size);
	int8_t* const q = (int8_t*)output;
	uint8_t* const u8 = (uint8_t*)output;
	if (datatype == CCV_16F)
	{
		const uint16_t* const f16 = (const uint16_t*)input;
		uint16_t* const scales = (uint16_t*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float scale_f = _ccv_nnc_quantize_8i_rowwise_16f(f16 + row_start, row_length, q + row_start);
			ccv_float_to_half_precision(&scale_f, scales + i, 1);
		} parallel_endfor
	} else if (datatype == CCV_16BF) {
		const uint16_t* const bf16 = (const uint16_t*)input;
		uint16_t* const scales = (uint16_t*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float scale_f = _ccv_nnc_quantize_8i_rowwise_16bf(bf16 + row_start, row_length, q + row_start);
			ccv_float_to_bfloat(&scale_f, scales + i, 1);
		} parallel_endfor
	} else if (datatype == CCV_32F) {
		const float* const f32 = (const float*)input;
		float* const scales = (float*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			scales[i] = _ccv_nnc_quantize_8i_rowwise_32f(f32 + row_start, row_length, q + row_start);
		} parallel_endfor
	} else {
		assert(datatype == CCV_64F);
		const double* const f64 = (const double*)input;
		double* const scales = (double*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			scales[i] = _ccv_nnc_quantize_8i_rowwise_64f(f64 + row_start, row_length, q + row_start);
		} parallel_endfor
	}
	return scale_offset + scale_size;
}

void ccv_nnc_dequantize_8i_rowwise(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY || memory_type == CCV_TENSOR_GPU_MEMORY);
	assert(row_length > 0);
	assert(output_length % row_length == 0);
	if (memory_type != CCV_TENSOR_CPU_MEMORY)
	{
#ifdef HAVE_CUDA
		ccv_nnc_compat_dequantize_8i_rowwise(input, datatype, input_length, row_length, output, output_length, 0);
#elif defined(HAVE_MPS)
		assert(datatype != CCV_64F);
		ccv_nnc_mps_dequantize_8i_rowwise(input, datatype, input_length, row_length, output, output_length, 0);
#else
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
#endif
		return;
	}
	const size_t row_count = output_length / row_length;
	const size_t scale_offset = (output_length + 127) & -128;
	assert(input_length >= scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype));
	const int8_t* const q = (const int8_t*)input;
	const uint8_t* const u8 = (const uint8_t*)input;
	if (datatype == CCV_16F)
	{
		uint16_t* const f16 = (uint16_t*)output;
		const uint16_t* const scales = (const uint16_t*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			float scale_f;
			ccv_half_precision_to_float(scales + i, &scale_f, 1);
			size_t j;
			for (j = 0; j < row_length; j++)
			{
				const float v = q[row_start + j] * scale_f;
				ccv_float_to_half_precision(&v, f16 + row_start + j, 1);
			}
		} parallel_endfor
	} else if (datatype == CCV_16BF) {
		uint16_t* const bf16 = (uint16_t*)output;
		const uint16_t* const scales = (const uint16_t*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			float scale_f;
			ccv_bfloat_to_float(scales + i, &scale_f, 1);
			size_t j;
			for (j = 0; j < row_length; j++)
			{
				const float v = q[row_start + j] * scale_f;
				ccv_float_to_bfloat(&v, bf16 + row_start + j, 1);
			}
		} parallel_endfor
	} else if (datatype == CCV_32F) {
		float* const f32 = (float*)output;
		const float* const scales = (const float*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float scale = scales[i];
			size_t j;
			for (j = 0; j < row_length; j++)
				f32[row_start + j] = q[row_start + j] * scale;
		} parallel_endfor
	} else {
		assert(datatype == CCV_64F);
		double* const f64 = (double*)output;
		const double* const scales = (const double*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const double scale = scales[i];
			size_t j;
			for (j = 0; j < row_length; j++)
				f64[row_start + j] = q[row_start + j] * scale;
		} parallel_endfor
	}
}
