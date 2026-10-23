#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"
#include "ccv_nnc_easy.h"
#include <float.h>
#include <limits.h>
#if HAVE_ACCELERATE_FRAMEWORK
#include <Accelerate/Accelerate.h>
#elif HAVE_CBLAS
#include <cblas.h>
#endif
#include "ccv_nnc_8i_rowwise_packed_grids.inc"
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif
#ifdef HAVE_CUDA
#include "gpu/ccv_nnc_compat.h"
#elif defined(HAVE_MPS)
#include "mps/ccv_nnc_mps.h"
#endif

enum {
	CCV_NNC_8I_ROWWISE_X_REFINEMENT_STEPS = 8,
	CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES = 16,
	CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES = 24,
	CCV_NNC_8I_ROWWISE_X_FEEDBACK_ROWS = 32,
	CCV_NNC_8I_ROWWISE_X_CHOLESKY_PANEL = 64,
};

static size_t _ccv_nnc_8i_rowwise_packed_scale_offset(const int format, const size_t input_length, const size_t row_length)
{
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t group_bits = ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t payload_size = (row_count * groups_per_row * group_bits + 7) / 8;
	return (payload_size + 127) & -128;
}

CCV_WARN_UNUSED(size_t) ccv_nnc_8i_rowwise_x_data_size(const int format, const int datatype, const size_t input_length, const size_t row_length)
{
	if (row_length == 0 || input_length % row_length != 0 ||
		((format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256) && row_length % 256 != 0))
		return 0;
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

static inline int _ccv_nnc_8i_rowwise_packed_sign_extend(const uint32_t value, const int bits)
{
	const uint32_t sign = 1u << (bits - 1);
	return (value & sign) ? (int)value - (int)(1u << bits) : (int)value;
}

static inline int _ccv_nnc_8i_rowwise_packed_floor_div(const int numerator, const int denominator)
{
	assert(denominator > 0);
	return numerator >= 0 ? numerator / denominator : -((-numerator + denominator - 1) / denominator);
}

static inline int _ccv_nnc_8i_rowwise_packed_ceil_div(const int numerator, const int denominator)
{
	assert(denominator > 0);
	return numerator >= 0 ? (numerator + denominator - 1) / denominator : -((-numerator) / denominator);
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

static inline double _ccv_nnc_8i_rowwise_weight(const float* const imatrix, const size_t j)
{
	return imatrix ? ccv_max((double)imatrix[j], 0.) : 1.;
}

static inline int _ccv_nnc_8i_rowwise_imatrix_is_valid(const float* const imatrix, const size_t imatrix_length, const size_t row_length, const size_t row_count)
{
	if (!imatrix)
		return 1;
	if (imatrix_length < row_length || imatrix_length % row_length != 0)
		return 0;
	const size_t imatrix_slices = imatrix_length / row_length;
	return imatrix_slices > 0 && row_count % imatrix_slices == 0;
}

static inline const float* _ccv_nnc_8i_rowwise_imatrix_for_row(const float* const imatrix, const size_t imatrix_length, const size_t row_length, const size_t row_count, const size_t row_idx)
{
	if (!imatrix)
		return 0;
	const size_t imatrix_slices = imatrix_length / row_length;
	if (imatrix_slices == 1)
		return imatrix;
	const size_t rows_per_slice = row_count / imatrix_slices;
	return imatrix + (row_idx / rows_per_slice) * row_length;
}

typedef struct {
	int q[32];
	int q8[32];
	int m;
	int b;
	int z;
	int scale;
	int grid[4];
	uint32_t signs;
} ccv_nnc_8i_rowwise_packed_group_t;

static const int ccv_nnc_8i_rowwise_packed_iq1s_scales[8] = {1, 2, 3, 5, 7, 9, 12, 16};
static const int ccv_nnc_8i_rowwise_packed_iq1_powers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};

typedef struct {
	int lane;
	int trit[3];
	double cost[3];
} ccv_nnc_8i_rowwise_iq1_lane_t;

typedef struct {
	ccv_nnc_8i_rowwise_iq1_lane_t lanes[8];
	const int16_t* map;
	double best;
	int index;
} ccv_nnc_8i_rowwise_iq1_search_t;

static void _ccv_nnc_8i_rowwise_packed_iq1_search(ccv_nnc_8i_rowwise_iq1_search_t* const search, const int depth, const int code, const double cost)
{
	if (cost >= search->best)
		return;
	if (depth == 8)
	{
		if (search->map[code] >= 0)
		{
			search->best = cost;
			search->index = search->map[code];
		}
		return;
	}
	const ccv_nnc_8i_rowwise_iq1_lane_t* const lane = search->lanes + depth;
	int k;
	for (k = 0; k < 3; k++)
	{
		const double next = cost + lane->cost[k];
		if (next >= search->best)
			break;
		_ccv_nnc_8i_rowwise_packed_iq1_search(search, depth + 1, code + lane->trit[k] * ccv_nnc_8i_rowwise_packed_iq1_powers[lane->lane], next);
	}
}

static double _ccv_nnc_8i_rowwise_packed_iq1_nearest(const double* const y, const double* const w, const int m, const int shift, const int16_t* const map, int* const index)
{
	const int levels[3] = {ccv_max((-8 + shift) * m, -127), shift * m, ccv_min((8 + shift) * m, 127)};
	ccv_nnc_8i_rowwise_iq1_search_t search;
	search.map = map;
	double baseline = 0;
	int natural_code = 0;
	int j, k;
	for (j = 0; j < 8; j++)
	{
		ccv_nnc_8i_rowwise_iq1_lane_t* const lane = search.lanes + j;
		lane->lane = j;
		for (k = 0; k < 3; k++)
		{
			const double d = levels[k] - y[j];
			lane->trit[k] = k;
			lane->cost[k] = w[j] * d * d;
		}
		for (k = 1; k < 3; k++)
		{
			int t;
			for (t = k; t > 0 && lane->cost[t] < lane->cost[t - 1]; t--)
			{
				const double cost = lane->cost[t];
				lane->cost[t] = lane->cost[t - 1];
				lane->cost[t - 1] = cost;
				const int trit = lane->trit[t];
				lane->trit[t] = lane->trit[t - 1];
				lane->trit[t - 1] = trit;
			}
		}
		natural_code += lane->trit[0] * ccv_nnc_8i_rowwise_packed_iq1_powers[j];
		const double minimum = lane->cost[0];
		baseline += minimum;
		for (k = 0; k < 3; k++)
			lane->cost[k] -= minimum;
	}
	if (map[natural_code] >= 0)
	{
		*index = map[natural_code];
		return baseline;
	}
	search.best = DBL_MAX;
	search.index = -1;
	for (j = 0; j < 8; j++)
		for (k = 1; k < 3; k++)
		{
			const ccv_nnc_8i_rowwise_iq1_lane_t* const lane = search.lanes + j;
			const int code = natural_code + (lane->trit[k] - lane->trit[0]) * ccv_nnc_8i_rowwise_packed_iq1_powers[j];
			if (map[code] >= 0 && lane->cost[k] < search.best)
			{
				search.best = lane->cost[k];
				search.index = map[code];
			}
		}
	// Independent lane costs bound the exact search; put cheap changes last.
	for (j = 1; j < 8; j++)
		for (k = j; k > 0 && search.lanes[k].cost[1] > search.lanes[k - 1].cost[1]; k--)
		{
			const ccv_nnc_8i_rowwise_iq1_lane_t lane = search.lanes[k];
			search.lanes[k] = search.lanes[k - 1];
			search.lanes[k - 1] = lane;
		}
	_ccv_nnc_8i_rowwise_packed_iq1_search(&search, 0, 0, 0);
	assert(search.index >= 0);
	*index = search.index;
	return baseline + search.best;
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq1_m(const double* const y, const double* const w, const int16_t* const map, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int m, sg, j;
	for (m = 1; m <= 16; m++)
	{
		double sse = 0;
		int grid[2];
		uint32_t shifts = 0;
		for (sg = 0; sg < 2; sg++)
		{
			int index[2];
			const double plus = _ccv_nnc_8i_rowwise_packed_iq1_nearest(y + sg * 8, w + sg * 8, m, 1, map, index);
			const double minus = _ccv_nnc_8i_rowwise_packed_iq1_nearest(y + sg * 8, w + sg * 8, m, -1, map, index + 1);
			const int shift = minus < plus;
			sse += shift ? minus : plus;
			grid[sg] = index[shift];
			shifts |= (uint32_t)shift << sg;
		}
		if (sse < best_sse)
		{
			best_sse = sse;
			group->scale = m;
			group->signs = shifts;
			memcpy(group->grid, grid, sizeof(grid));
		}
	}
	for (sg = 0; sg < 2; sg++)
		for (j = 0; j < 8; j++)
		{
			const int g = (int)((ccv_nnc_8i_rowwise_packed_iq1_grid[group->grid[sg]] >> (2 * j)) & 3) - 1;
			const int shift = (group->signs & (1u << sg)) ? -1 : 1;
			group->q8[sg * 8 + j] = ccv_clamp(group->scale * (8 * g + shift), -127, 127);
		}
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq1_s(const double* const y, const double* const w, const int16_t* const map, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int sc, sg, j;
	for (sc = 0; sc < 8; sc++)
	{
		const int m = ccv_nnc_8i_rowwise_packed_iq1s_scales[sc];
		double costs[4][2];
		int indices[4][2];
		for (sg = 0; sg < 4; sg++)
			for (j = 0; j < 2; j++)
				costs[sg][j] = _ccv_nnc_8i_rowwise_packed_iq1_nearest(y + sg * 8, w + sg * 8, m, j ? -1 : 1, map, &indices[sg][j]);
		int mask;
		for (mask = 0; mask < 16; mask++)
		{
			if ((mask ^ (mask >> 1) ^ (mask >> 2) ^ (mask >> 3)) & 1)
				continue;
			double sse = 0;
			for (sg = 0; sg < 4; sg++)
				sse += costs[sg][(mask >> sg) & 1];
			if (sse < best_sse)
			{
				best_sse = sse;
				group->scale = sc;
				group->signs = (uint32_t)mask;
				for (sg = 0; sg < 4; sg++)
					group->grid[sg] = indices[sg][(mask >> sg) & 1];
			}
		}
	}
	const int m = ccv_nnc_8i_rowwise_packed_iq1s_scales[group->scale];
	for (sg = 0; sg < 4; sg++)
		for (j = 0; j < 8; j++)
		{
			const int g = (int)((ccv_nnc_8i_rowwise_packed_iq1_grid[group->grid[sg]] >> (2 * j)) & 3) - 1;
			const int shift = (group->signs & (1u << sg)) ? -1 : 1;
			group->q8[sg * 8 + j] = ccv_clamp(m * (8 * g + shift), -127, 127);
		}
}

static void _ccv_nnc_8i_rowwise_packed_nearest_candidates(const uint8_t* const grid, const int grid_size, const int lanes, const int levels, const int code_count, const int candidate_count, uint16_t* const candidates)
{
	assert(candidate_count <= CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES);
	assert(candidate_count <= grid_size);
	int code;
	for (code = 0; code < code_count; code++)
	{
		uint8_t target[8];
		int value = code;
		int j;
		for (j = 0; j < lanes; j++)
		{
			target[j] = (uint8_t)(value % levels);
			value /= levels;
		}
		int best_distance[CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES];
		int k;
		for (k = 0; k < candidate_count; k++)
		{
			best_distance[k] = INT_MAX;
			candidates[(size_t)code * candidate_count + k] = 0;
		}
		int index;
		for (index = 0; index < grid_size; index++)
		{
			int distance = 0;
			for (j = 0; j < lanes; j++)
			{
				const int d = (int)grid[(size_t)index * lanes + j] - target[j];
				distance += d * d;
			}
			if (distance >= best_distance[candidate_count - 1])
				continue;
			k = candidate_count - 1;
			while (k > 0 && distance < best_distance[k - 1])
			{
				best_distance[k] = best_distance[k - 1];
				candidates[(size_t)code * candidate_count + k] = candidates[(size_t)code * candidate_count + k - 1];
				--k;
			}
			best_distance[k] = distance;
			candidates[(size_t)code * candidate_count + k] = (uint16_t)index;
		}
	}
}

static void _ccv_nnc_8i_rowwise_packed_quant_q5(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int best_q[16] = {0};
	int best_q8[16] = {0};
	int best_m = 1, best_b = 0;
	int m, b, j;
	for (m = 1; m <= 8; m++)
		for (b = -16; b <= 15; b++)
		{
			if (-16 * m + b < -127 || 15 * m + b > 127)
				continue;
			double sse = 0;
			int q[16];
			int q8[16];
			for (j = 0; j < 16; j++)
			{
				q[j] = ccv_clamp((int)lrint((y[j] - b) / m), -16, 15);
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

static void _ccv_nnc_8i_rowwise_packed_quant_q6(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	double best_sse = DBL_MAX;
	int best_q[8] = {0};
	int best_q8[8] = {0};
	int best_m = 1, best_b = 0;
	int m, b, j;
	for (m = 1; m <= 4; m++)
		for (b = -2; b <= 1; b++)
		{
				const int qmin = ccv_max(-32, _ccv_nnc_8i_rowwise_packed_ceil_div(-127 - b, m));
				const int qmax = ccv_min(31, _ccv_nnc_8i_rowwise_packed_floor_div(127 - b, m));
			if (qmin > qmax)
				continue;
			double sse = 0;
			int q[8];
			int q8[8];
			for (j = 0; j < 8; j++)
			{
				q[j] = ccv_clamp((int)lrint((y[j] - b) / m), qmin, qmax);
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

static int _ccv_nnc_8i_rowwise_packed_iq2xxs_value(const int index, const int lane)
{
	const int v = (int)((ccv_nnc_8i_rowwise_packed_iq2xxs_grid[index] >> (lane * 2)) & 3);
	assert(v < 3);
	return 1 + v * 2;
}

enum {
	CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE = 256,
	CCV_NNC_8I_ROWWISE_PACKED_IQ2S_GRID_SIZE = 1024,
	CCV_NNC_8I_ROWWISE_PACKED_IQ2_CODE_COUNT = 6561,
};

static const int ccv_nnc_8i_rowwise_packed_iq2xxs_scales[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
static int ccv_nnc_8i_rowwise_packed_iq2_scale_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[33][3];
static int ccv_nnc_8i_rowwise_packed_iq2xxs_initialized = 0;
static int ccv_nnc_8i_rowwise_packed_iq2xxs_candidates_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq2xxs_level[CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE][8];
static uint8_t ccv_nnc_8i_rowwise_packed_iq2xxs_scaled_value[33][CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE][8];
static uint16_t ccv_nnc_8i_rowwise_packed_iq2xxs_candidates[CCV_NNC_8I_ROWWISE_PACKED_IQ2_CODE_COUNT][CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES];

static void _ccv_nnc_8i_rowwise_packed_iq2_scale_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2_scale_initialized)
		return;
	int scale, level;
	for (scale = 1; scale <= 32; scale++)
		for (level = 0; level < 3; level++)
			ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[scale][level] = (uint8_t)ccv_min((1 + level * 2) * scale, 127);
	ccv_nnc_8i_rowwise_packed_iq2_scale_initialized = 1;
}

static void _ccv_nnc_8i_rowwise_packed_iq2xxs_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2xxs_initialized)
		return;
	_ccv_nnc_8i_rowwise_packed_iq2_scale_init();
	int index, j, scale;
	for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE; index++)
		for (j = 0; j < 8; j++)
			ccv_nnc_8i_rowwise_packed_iq2xxs_level[index][j] = (uint8_t)((_ccv_nnc_8i_rowwise_packed_iq2xxs_value(index, j) - 1) / 2);
	for (scale = 1; scale <= 32; scale++)
	{
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE; index++)
			for (j = 0; j < 8; j++)
				ccv_nnc_8i_rowwise_packed_iq2xxs_scaled_value[scale][index][j] = ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[scale][ccv_nnc_8i_rowwise_packed_iq2xxs_level[index][j]];
	}
	ccv_nnc_8i_rowwise_packed_iq2xxs_initialized = 1;
}

static void _ccv_nnc_8i_rowwise_packed_iq2xxs_candidates_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2xxs_candidates_initialized)
		return;
	_ccv_nnc_8i_rowwise_packed_iq2xxs_init();
	_ccv_nnc_8i_rowwise_packed_nearest_candidates(&ccv_nnc_8i_rowwise_packed_iq2xxs_level[0][0], CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE, 8, 3, CCV_NNC_8I_ROWWISE_PACKED_IQ2_CODE_COUNT, CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES, &ccv_nnc_8i_rowwise_packed_iq2xxs_candidates[0][0]);
	ccv_nnc_8i_rowwise_packed_iq2xxs_candidates_initialized = 1;
}

enum {
	CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE = 512,
};

static int ccv_nnc_8i_rowwise_packed_iq2xs_initialized = 0;
static int ccv_nnc_8i_rowwise_packed_iq2xs_candidates_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq2xs_level[CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE][8];
static uint8_t ccv_nnc_8i_rowwise_packed_iq2xs_scaled_value[16][CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE][8];
static uint16_t ccv_nnc_8i_rowwise_packed_iq2xs_candidates[CCV_NNC_8I_ROWWISE_PACKED_IQ2_CODE_COUNT][CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES];

static void _ccv_nnc_8i_rowwise_packed_iq2xs_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2xs_initialized)
		return;
	_ccv_nnc_8i_rowwise_packed_iq2_scale_init();
	int index, j, scale_code;
	for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE; index++)
		for (j = 0; j < 8; j++)
			ccv_nnc_8i_rowwise_packed_iq2xs_level[index][j] = (uint8_t)((_ccv_nnc_8i_rowwise_packed_iq2_value(ccv_nnc_8i_rowwise_packed_iq2xs_grid, index, j) - 1) / 2);
	for (scale_code = 0; scale_code < 16; scale_code++)
	{
		const int scale = ccv_nnc_8i_rowwise_packed_iq2xxs_scales[scale_code];
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE; index++)
			for (j = 0; j < 8; j++)
				ccv_nnc_8i_rowwise_packed_iq2xs_scaled_value[scale_code][index][j] = ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[scale][ccv_nnc_8i_rowwise_packed_iq2xs_level[index][j]];
	}
	ccv_nnc_8i_rowwise_packed_iq2xs_initialized = 1;
}

static void _ccv_nnc_8i_rowwise_packed_iq2xs_candidates_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq2xs_candidates_initialized)
		return;
	_ccv_nnc_8i_rowwise_packed_iq2xs_init();
	_ccv_nnc_8i_rowwise_packed_nearest_candidates(&ccv_nnc_8i_rowwise_packed_iq2xs_level[0][0], CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE, 8, 3, CCV_NNC_8I_ROWWISE_PACKED_IQ2_CODE_COUNT, CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES, &ccv_nnc_8i_rowwise_packed_iq2xs_candidates[0][0]);
	ccv_nnc_8i_rowwise_packed_iq2xs_candidates_initialized = 1;
}

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
#define CCV_NNC_8I_ROWWISE_PACKED_IQ3_CODE_COUNT (4096)

static int ccv_nnc_8i_rowwise_packed_iq3s_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq3s_scaled_value[17][CCV_NNC_8I_ROWWISE_PACKED_IQ3S_GRID_SIZE][4];

static int ccv_nnc_8i_rowwise_packed_iq3xxs_initialized = 0;
static int ccv_nnc_8i_rowwise_packed_iq3xxs_candidates_initialized = 0;
static uint8_t ccv_nnc_8i_rowwise_packed_iq3xxs_level[CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE][4];
static uint8_t ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[17][CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE][4];
static uint16_t ccv_nnc_8i_rowwise_packed_iq3xxs_candidates[CCV_NNC_8I_ROWWISE_PACKED_IQ3_CODE_COUNT][CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES];

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
	for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE; index++)
		for (j = 0; j < 4; j++)
			ccv_nnc_8i_rowwise_packed_iq3xxs_level[index][j] = (uint8_t)((_ccv_nnc_8i_rowwise_packed_iq3xxs_value(index, j) - 1) / 2);
	for (scale = 1; scale <= 16; scale++)
		for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE; index++)
			for (j = 0; j < 4; j++)
				ccv_nnc_8i_rowwise_packed_iq3xxs_scaled_value[scale][index][j] = (uint8_t)ccv_min((1 + ccv_nnc_8i_rowwise_packed_iq3xxs_level[index][j] * 2) * scale, 127);
	ccv_nnc_8i_rowwise_packed_iq3xxs_initialized = 1;
}

static void _ccv_nnc_8i_rowwise_packed_iq3xxs_candidates_init(void)
{
	if (ccv_nnc_8i_rowwise_packed_iq3xxs_candidates_initialized)
		return;
	_ccv_nnc_8i_rowwise_packed_iq3xxs_init();
	_ccv_nnc_8i_rowwise_packed_nearest_candidates(&ccv_nnc_8i_rowwise_packed_iq3xxs_level[0][0], CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE, 4, 8, CCV_NNC_8I_ROWWISE_PACKED_IQ3_CODE_COUNT, CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES, &ccv_nnc_8i_rowwise_packed_iq3xxs_candidates[0][0]);
	ccv_nnc_8i_rowwise_packed_iq3xxs_candidates_initialized = 1;
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

static inline double _ccv_nnc_8i_rowwise_packed_iq2_cost(const double cost[8][3], const uint8_t* const levels)
{
	double sse = 0;
	int j;
	for (j = 0; j < 8; j++)
		sse += cost[j][levels[j]];
	return sse;
}

static inline double _ccv_nnc_8i_rowwise_packed_safe_lower_bound(const double value)
{
	return nextafter(value - (fabs(value) + 1) * (64 * DBL_EPSILON), -INFINITY);
}

static inline double _ccv_nnc_8i_rowwise_packed_iq2xxs_cost(const double cost[8][3], const double flipped_cost[8][3], const double flip_metric[8][3], const uint8_t* const levels, const uint8_t signs, const int flip_sign, int* const sign_index)
{
	int best_flip = -1;
	int j;
	if (flip_sign)
	{
		double best_flip_metric = DBL_MAX;
		for (j = 0; j < 8; j++)
		{
			const double metric = flip_metric[j][levels[j]];
			if (metric < best_flip_metric)
			{
				best_flip_metric = metric;
				best_flip = j;
			}
		}
	}
	double sse = 0;
	for (j = 0; j < 8; j++)
		sse += j == best_flip ? flipped_cost[j][levels[j]] : cost[j][levels[j]];
	*sign_index = (signs ^ (best_flip >= 0 ? (uint8_t)(1u << best_flip) : 0)) & 0x7f;
	return sse;
}

static void _ccv_nnc_8i_rowwise_packed_quant_iq2_xxs(const double* const y, const double* const w, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	assert(ccv_nnc_8i_rowwise_packed_iq2xxs_initialized);
	double best_sse = DBL_MAX;
	int best_scale_code = 0;
	int best_grid[4] = {0};
	int best_sign[4] = {0};
	int scale_code;
	for (scale_code = 0; scale_code < 16; scale_code++)
	{
		const int scale = ccv_nnc_8i_rowwise_packed_iq2xxs_scales[scale_code];
		double group_sse = 0;
		int group_grid[4] = {0};
		int group_sign[4] = {0};
		int sg;
		for (sg = 0; sg < 4; sg++)
		{
			double best_sub_sse = DBL_MAX;
			int best_sub_grid = 0;
			int best_sub_sign = 0;
			const int lane = sg * 8;
			uint8_t signs = 0;
			int negative_count = 0;
			double cost[8][3];
			double flipped_cost[8][3];
			double flip_metric[8][3];
			int j;
			for (j = 0; j < 8; j++)
			{
				if (y[lane + j] < 0)
				{
					signs |= (uint8_t)(1u << j);
					negative_count++;
				}
				int level;
				for (level = 0; level < 3; level++)
				{
					const int mag = ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[scale][level];
					const int q8 = (signs & (1u << j)) ? -mag : mag;
					double d = (double)q8 - y[lane + j];
					cost[j][level] = w[lane + j] * d * d;
					d = (double)-q8 - y[lane + j];
					flipped_cost[j][level] = w[lane + j] * d * d;
					flip_metric[j][level] = w[lane + j] * (double)mag * fabs(y[lane + j]);
				}
			}
			int target_code = 0;
			int code_multiplier = 1;
			uint8_t target_levels[8];
			double lower_bound_base = 0;
			double lower_bound_ratio = DBL_MAX;
			double lower_bound_flip = DBL_MAX;
			for (j = 0; j < 8; j++)
			{
				int target_level = 0;
				if (cost[j][1] < cost[j][target_level])
					target_level = 1;
				if (cost[j][2] < cost[j][target_level])
					target_level = 2;
				target_levels[j] = (uint8_t)target_level;
				target_code += target_level * code_multiplier;
				code_multiplier *= 3;
				lower_bound_base += cost[j][target_level];
				int level;
				for (level = 0; level < 3; level++)
				{
					if (level != target_level)
					{
						const int d = level - target_level;
						lower_bound_ratio = ccv_min(lower_bound_ratio, (cost[j][level] - cost[j][target_level]) / (double)(d * d));
					}
					lower_bound_flip = ccv_min(lower_bound_flip, flipped_cost[j][level] - cost[j][level]);
				}
			}
			int candidate;
			for (candidate = 0; candidate < CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES; candidate++)
			{
				const int index = ccv_nnc_8i_rowwise_packed_iq2xxs_candidates[target_code][candidate];
				const uint8_t* const levels = ccv_nnc_8i_rowwise_packed_iq2xxs_level[index];
				int sign_index;
				const double sse = _ccv_nnc_8i_rowwise_packed_iq2xxs_cost(cost, flipped_cost, flip_metric, levels, signs, negative_count & 1, &sign_index);
				if (sse < best_sub_sse || (sse == best_sub_sse && index < best_sub_grid))
				{
					best_sub_sse = sse;
					best_sub_grid = index;
					best_sub_sign = sign_index;
				}
			}
			const uint8_t* const cutoff_levels = ccv_nnc_8i_rowwise_packed_iq2xxs_level[ccv_nnc_8i_rowwise_packed_iq2xxs_candidates[target_code][CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES - 1]];
			int cutoff_distance = 0;
			for (j = 0; j < 8; j++)
			{
				const int d = (int)cutoff_levels[j] - target_levels[j];
				cutoff_distance += d * d;
			}
			const double safe_ratio = nextafter(lower_bound_ratio, 0);
			const double lower_bound = _ccv_nnc_8i_rowwise_packed_safe_lower_bound(lower_bound_base + safe_ratio * cutoff_distance + ((negative_count & 1) ? nextafter(lower_bound_flip, -INFINITY) : 0));
			if (!(best_sub_sse < lower_bound))
			{
				int index;
				for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XXS_GRID_SIZE; index++)
				{
					int sign_index;
					const double sse = _ccv_nnc_8i_rowwise_packed_iq2xxs_cost(cost, flipped_cost, flip_metric, ccv_nnc_8i_rowwise_packed_iq2xxs_level[index], signs, negative_count & 1, &sign_index);
					if (sse < best_sub_sse || (sse == best_sub_sse && index < best_sub_grid))
					{
						best_sub_sse = sse;
						best_sub_grid = index;
						best_sub_sign = sign_index;
					}
				}
			}
			group_sse += best_sub_sse;
			group_grid[sg] = best_sub_grid;
			group_sign[sg] = best_sub_sign;
		}
		if (group_sse < best_sse)
		{
			best_sse = group_sse;
			best_scale_code = scale_code;
			memcpy(best_grid, group_grid, sizeof(best_grid));
			memcpy(best_sign, group_sign, sizeof(best_sign));
		}
	}
	group->scale = best_scale_code;
	group->signs = 0;
	memcpy(group->grid, best_grid, sizeof(best_grid));
	int j;
	for (j = 0; j < 4; j++)
		group->signs |= (uint32_t)best_sign[j] << (j * 7);
	for (j = 0; j < 32; j++)
	{
		const int sg = j >> 3;
		const int lane = j & 7;
		const uint8_t signs = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[best_sign[sg]];
		const int mag = ccv_nnc_8i_rowwise_packed_iq2xxs_scaled_value[ccv_nnc_8i_rowwise_packed_iq2xxs_scales[best_scale_code]][best_grid[sg]][lane];
		group->q8[j] = (signs & (1u << lane)) ? -mag : mag;
	}
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
	assert(ccv_nnc_8i_rowwise_packed_iq2xs_initialized);
	double best_sse = DBL_MAX;
	int best_scale_code = 0;
	int best_grid = 0;
	uint32_t signs = 0;
	int j;
	for (j = 0; j < 8; j++)
		if (y[j] < 0)
			signs |= (1u << j);
	int scale_code;
	for (scale_code = 0; scale_code < 16; scale_code++)
	{
		double cost[8][3];
		for (j = 0; j < 8; j++)
		{
			int level;
			for (level = 0; level < 3; level++)
			{
				const int mag = ccv_nnc_8i_rowwise_packed_iq2xxs_scale_level[ccv_nnc_8i_rowwise_packed_iq2xxs_scales[scale_code]][level];
				const int q8 = (signs & (1u << j)) ? -mag : mag;
				const double d = (double)q8 - y[j];
				cost[j][level] = w[j] * d * d;
			}
		}
		int target_code = 0;
		int code_multiplier = 1;
		uint8_t target_levels[8];
		double lower_bound_base = 0;
		double lower_bound_ratio = DBL_MAX;
		for (j = 0; j < 8; j++)
		{
			int target_level = 0;
			if (cost[j][1] < cost[j][target_level])
				target_level = 1;
			if (cost[j][2] < cost[j][target_level])
				target_level = 2;
			target_levels[j] = (uint8_t)target_level;
			target_code += target_level * code_multiplier;
			code_multiplier *= 3;
			lower_bound_base += cost[j][target_level];
			int level;
			for (level = 0; level < 3; level++)
				if (level != target_level)
				{
					const int d = level - target_level;
					lower_bound_ratio = ccv_min(lower_bound_ratio, (cost[j][level] - cost[j][target_level]) / (double)(d * d));
				}
		}
		int candidate;
		for (candidate = 0; candidate < CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES; candidate++)
		{
			const int index = ccv_nnc_8i_rowwise_packed_iq2xs_candidates[target_code][candidate];
			const uint8_t* const levels = ccv_nnc_8i_rowwise_packed_iq2xs_level[index];
			const double sse = _ccv_nnc_8i_rowwise_packed_iq2_cost(cost, levels);
			if (sse < best_sse || (sse == best_sse && scale_code == best_scale_code && index < best_grid))
			{
				best_sse = sse;
				best_scale_code = scale_code;
				best_grid = index;
			}
		}
		const uint8_t* const cutoff_levels = ccv_nnc_8i_rowwise_packed_iq2xs_level[ccv_nnc_8i_rowwise_packed_iq2xs_candidates[target_code][CCV_NNC_8I_ROWWISE_IQ2_CANDIDATES - 1]];
		int cutoff_distance = 0;
		for (j = 0; j < 8; j++)
		{
			const int d = (int)cutoff_levels[j] - target_levels[j];
			cutoff_distance += d * d;
		}
		const double lower_bound = _ccv_nnc_8i_rowwise_packed_safe_lower_bound(lower_bound_base + nextafter(lower_bound_ratio, 0) * cutoff_distance);
		if (!(best_sse < lower_bound))
		{
			int index;
			for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ2XS_GRID_SIZE; index++)
			{
				const double sse = _ccv_nnc_8i_rowwise_packed_iq2_cost(cost, ccv_nnc_8i_rowwise_packed_iq2xs_level[index]);
				if (sse < best_sse || (sse == best_sse && scale_code == best_scale_code && index < best_grid))
				{
					best_sse = sse;
					best_scale_code = scale_code;
					best_grid = index;
				}
			}
		}
	}
	group->scale = best_scale_code;
	group->grid[0] = best_grid;
	group->signs = signs;
	memset(group->q8, 0, sizeof(group->q8));
	for (j = 0; j < 8; j++)
	{
		const int mag = ccv_nnc_8i_rowwise_packed_iq2xs_scaled_value[best_scale_code][best_grid][j];
		group->q8[j] = (signs & (1u << j)) ? -mag : mag;
	}
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
			double cost[4][8];
			for (j = 0; j < 4; j++)
			{
				int level;
				for (level = 0; level < 8; level++)
				{
					const int mag = ccv_min((1 + level * 2) * scale, 127);
					const double d = (double)mag - ay[lane + j];
					cost[j][level] = w[lane + j] * d * d;
				}
			}
			int target_code = 0;
			uint8_t target_levels[4];
			double lower_bound_base = 0;
			double lower_bound_ratio = DBL_MAX;
			for (j = 0; j < 4; j++)
			{
				int target_level = 0;
				int level;
				for (level = 1; level < 8; level++)
					if (cost[j][level] < cost[j][target_level])
						target_level = level;
				target_levels[j] = (uint8_t)target_level;
				target_code |= target_level << (j * 3);
				lower_bound_base += cost[j][target_level];
				for (level = 0; level < 8; level++)
					if (level != target_level)
					{
						const int d = level - target_level;
						lower_bound_ratio = ccv_min(lower_bound_ratio, (cost[j][level] - cost[j][target_level]) / (double)(d * d));
					}
			}
			int candidate;
			for (candidate = 0; candidate < CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES; candidate++)
			{
				const int index = ccv_nnc_8i_rowwise_packed_iq3xxs_candidates[target_code][candidate];
				const uint8_t* const levels = ccv_nnc_8i_rowwise_packed_iq3xxs_level[index];
				const double sse = cost[0][levels[0]] + cost[1][levels[1]] + cost[2][levels[2]] + cost[3][levels[3]];
				if (sse < best_sub_sse || (sse == best_sub_sse && index < best_sub_grid))
				{
					best_sub_sse = sse;
					best_sub_grid = index;
				}
			}
			const uint8_t* const cutoff_levels = ccv_nnc_8i_rowwise_packed_iq3xxs_level[ccv_nnc_8i_rowwise_packed_iq3xxs_candidates[target_code][CCV_NNC_8I_ROWWISE_IQ3_CANDIDATES - 1]];
			int cutoff_distance = 0;
			for (j = 0; j < 4; j++)
			{
				const int d = (int)cutoff_levels[j] - target_levels[j];
				cutoff_distance += d * d;
			}
			const double lower_bound = _ccv_nnc_8i_rowwise_packed_safe_lower_bound(lower_bound_base + nextafter(lower_bound_ratio, 0) * cutoff_distance);
			if (!(best_sub_sse < lower_bound))
			{
				int index;
				for (index = 0; index < CCV_NNC_8I_ROWWISE_PACKED_IQ3XXS_GRID_SIZE; index++)
				{
					const uint8_t* const levels = ccv_nnc_8i_rowwise_packed_iq3xxs_level[index];
					const double sse = cost[0][levels[0]] + cost[1][levels[1]] + cost[2][levels[2]] + cost[3][levels[3]];
					if (sse < best_sub_sse || (sse == best_sub_sse && index < best_sub_grid))
					{
						best_sub_sse = sse;
						best_sub_grid = index;
					}
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

static void _ccv_nnc_8i_rowwise_packed_quant_group(const int format, const double* const y, const double* const w, const int16_t* const iq1_map, ccv_nnc_8i_rowwise_packed_group_t* const group)
{
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
			_ccv_nnc_8i_rowwise_packed_quant_iq1_s(y, w, iq1_map, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
			_ccv_nnc_8i_rowwise_packed_quant_iq1_m(y, w, iq1_map, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
			_ccv_nnc_8i_rowwise_packed_quant_q5(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
			_ccv_nnc_8i_rowwise_packed_quant_q6(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q4_K:
			_ccv_nnc_8i_rowwise_packed_quant_q4(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q3_K:
			_ccv_nnc_8i_rowwise_packed_quant_q3(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q2_K:
			_ccv_nnc_8i_rowwise_packed_quant_q2(y, w, group);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			_ccv_nnc_8i_rowwise_packed_quant_iq2_xxs(y, w, group);
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
	const size_t bit_offset = group_index * ccv_nnc_8i_rowwise_x_group_bits(format);
	size_t bit = bit_offset;
	int j;
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_IQ1_S:
			for (j = 0; j < 4; j++)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + j * 11, (uint32_t)group->grid[j], 11);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 44, group->signs & 7, 3);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 47, (uint32_t)group->scale, 3);
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ1_M:
			for (j = 0; j < 2; j++)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + j * 12, (uint32_t)group->grid[j] | (((group->signs >> j) & 1) << 11), 12);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 24, (uint32_t)(group->scale - 1), 4);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q5_K:
			for (j = 0; j < 16; j++, bit += 5)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->q[j] + 16), 5);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->m - 1), 3);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 3, (uint32_t)(group->b + 16), 5);
			break;
		case CCV_NNC_QX_8I_ROWWISE_Q6_K:
			for (j = 0; j < 8; j++, bit += 6)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->q[j] & 0x3f), 6);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit, (uint32_t)(group->m - 1), 2);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 2, (uint32_t)(group->b & 3), 2);
			break;
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
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			for (j = 0; j < 4; j++)
				_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + j * 8, (uint32_t)group->grid[j], 8);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 32, group->signs, 28);
			_ccv_nnc_8i_rowwise_packed_write_bits(output, bit + 60, (uint32_t)group->scale, 4);
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
	const size_t bit_offset = group_index * ccv_nnc_8i_rowwise_x_group_bits(format);
	size_t bit = bit_offset;
	int j;
	switch (format)
	{
		case CCV_NNC_QX_8I_ROWWISE_IQ1_S: {
			uint32_t shifts = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 44, 3);
			shifts |= ((shifts ^ (shifts >> 1) ^ (shifts >> 2)) & 1) << 3;
			const int m = ccv_nnc_8i_rowwise_packed_iq1s_scales[_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 47, 3)];
			int sg;
			for (sg = 0; sg < 4; sg++)
			{
				const int index = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + sg * 11, 11);
				const int shift = (shifts & (1u << sg)) ? -1 : 1;
				for (j = 0; j < 8; j++)
				{
					const int g = (int)((ccv_nnc_8i_rowwise_packed_iq1_grid[index] >> (2 * j)) & 3) - 1;
					q8[sg * 8 + j] = ccv_clamp(m * (8 * g + shift), -127, 127);
				}
			}
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_IQ1_M: {
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 24, 4) + 1;
			int sg;
			for (sg = 0; sg < 2; sg++)
			{
				const uint32_t code = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + sg * 12, 12);
				const int shift = (code & 2048) ? -1 : 1;
				for (j = 0; j < 8; j++)
				{
					const int g = (int)((ccv_nnc_8i_rowwise_packed_iq1_grid[code & 2047] >> (2 * j)) & 3) - 1;
					q8[sg * 8 + j] = ccv_clamp(m * (8 * g + shift), -127, 127);
				}
			}
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_Q5_K: {
			int q[16];
			for (j = 0; j < 16; j++, bit += 5)
				q[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 5) - 16;
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 3) + 1;
			const int b = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 3, 5) - 16;
			for (j = 0; j < 16; j++)
				q8[j] = q[j] * m + b;
			break;
		}
		case CCV_NNC_QX_8I_ROWWISE_Q6_K: {
			int q[8];
			for (j = 0; j < 8; j++, bit += 6)
				q[j] = _ccv_nnc_8i_rowwise_packed_sign_extend(_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 6), 6);
			const int m = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit, 2) + 1;
			const int b = _ccv_nnc_8i_rowwise_packed_sign_extend(_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 2, 2), 2);
			for (j = 0; j < 8; j++)
				q8[j] = q[j] * m + b;
			break;
		}
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
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS: {
			int grid[4];
			for (j = 0; j < 4; j++)
				grid[j] = (int)_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + j * 8, 8);
			const uint32_t sign_codes = _ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 32, 28);
			const int scale = ccv_nnc_8i_rowwise_packed_iq2xxs_scales[_ccv_nnc_8i_rowwise_packed_read_bits(input, bit + 60, 4)];
			int sg;
			for (sg = 0; sg < 4; sg++)
			{
				const uint8_t signs = ccv_nnc_8i_rowwise_packed_iq2xxs_ksigns[(sign_codes >> (sg * 7)) & 0x7f];
				for (j = 0; j < 8; j++)
				{
					const int lane = sg * 8 + j;
					const int mag = ccv_min(_ccv_nnc_8i_rowwise_packed_iq2xxs_value(grid[sg], j) * scale, 127);
					q8[lane] = (signs & (1u << j)) ? -mag : mag;
				}
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

// Regular normalized H256, matching NAInt8MatMul activation staging. This is
// not the Sylvester WHT. Transform the existing floating row before fitting scales.
static void _ccv_nnc_8i_rowwise_hadamard_256(double* const row, const size_t row_length)
{
	size_t base;
	for (base = 0; base < row_length; base += 256)
	{
		int stride;
		for (stride = 1; stride < 256; stride *= 4)
		{
			int i, j;
			for (i = 0; i < 256; i += stride * 4)
				for (j = 0; j < stride; j++)
				{
					double* const v = row + base + i + j;
					const double a = v[0], b = v[stride], c = v[stride * 2], d = v[stride * 3];
					v[0] = a + b + c - d;
					v[stride] = a + b - c + d;
					v[stride * 2] = a - b + c + d;
					v[stride * 3] = -a + b + c + d;
				}
		}
		int i;
		for (i = 0; i < 256; i++)
			row[base + i] *= 1. / 16.;
	}
}

typedef struct {
	const float* hessian; // row_length x row_length Hessian of the current slice, e.g. input second moments.
	double* hessian64; // Double copy of hessian for BLAS products; 0 without BLAS.
	double ridge; // Added to the Hessian diagonal to keep it positive definite.
	double* u; // Feedback coefficients U[j][k] for lane j and lanes k in later groups.
	double* rbb; // Upper-triangular diagonal blocks R_bb of H = R R^T, one group_size^2 block per group.
	double* weights; // Conditional variances diag(D_b) = diag(R_bb R_bb^T), 0 for padded lanes.
} ccv_nnc_8i_rowwise_x_feedback_t;

// a[i][j] -= sum_{m in [k0, k1)} a[i][m] a[j][m] for i <= j < k0.
static void _ccv_nnc_8i_rowwise_x_rank_update(double* const a, const size_t n, const size_t k0, const size_t k1)
{
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, (int)k0, (int)(k1 - k0), -1, a + k0, (int)n, 1, a, (int)n);
#else
	const size_t w = k1 - k0;
	double* const panel = (double*)ccmalloc(sizeof(double) * k0 * w);
	size_t i;
	for (i = 0; i < k0; i++)
		memcpy(panel + i * w, a + i * n + k0, sizeof(double) * w);
	const size_t blocks = (k0 + 15) / 16;
#ifdef USE_DISPATCH
	dispatch_apply(blocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t block_idx) {
#else
	parallel_for(block_idx, (int)blocks) {
#endif
		const size_t row_end = ccv_min(((size_t)block_idx + 1) * 16, k0);
		size_t r, j, m;
		for (r = (size_t)block_idx * 16; r < row_end; r++)
		{
			const double* const pr = panel + r * w;
			for (j = r; j < k0; j++)
			{
				const double* const pj = panel + j * w;
				double v = 0;
				for (m = 0; m < w; m++)
					v += pr[m] * pj[m];
				a[r * n + j] -= v;
			}
		}
#ifdef USE_DISPATCH
	});
#else
	} parallel_endfor
#endif
	ccfree(panel);
#endif
}

// Factor H = R R^T with R upper triangular, in place on the upper triangle of a, starting from the
// last column so that column k only depends on lanes at or after k. Columns go in panels: factor the
// panel's diagonal block, solve each row above it on its contiguous panel segment, then one rank-k
// update of the leading block.
static int _ccv_nnc_8i_rowwise_x_reverse_cholesky(double* const a, const size_t n)
{
	size_t k1 = n;
	while (k1 > 0)
	{
		const size_t k0 = k1 > CCV_NNC_8I_ROWWISE_X_CHOLESKY_PANEL ? k1 - CCV_NNC_8I_ROWWISE_X_CHOLESKY_PANEL : 0;
		size_t k = k1;
		while (k-- > k0)
		{
			const double d = a[k * n + k];
			if (!(d > 0))
				return 0;
			const double rkk = sqrt(d);
			a[k * n + k] = rkk;
			size_t i, j;
			for (i = k0; i < k; i++)
				a[i * n + k] /= rkk;
			for (i = k0; i < k; i++)
			{
				const double ci = a[i * n + k];
				for (j = i; j < k; j++)
					a[i * n + j] -= ci * a[j * n + k];
			}
		}
		// R[i][m] = (H[i][m] - sum_{p in (m, k1)} R[i][p] R[m][p]) / R[m][m] for rows i above the panel.
		const size_t blocks = (k0 + 63) / 64;
#ifdef USE_DISPATCH
		dispatch_apply(blocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t block_idx) {
#else
		parallel_for(block_idx, (int)blocks) {
#endif
			const size_t row_end = ccv_min(((size_t)block_idx + 1) * 64, k0);
			size_t r;
			for (r = (size_t)block_idx * 64; r < row_end; r++)
			{
				double* const ar = a + r * n;
				size_t m = k1;
				while (m-- > k0)
				{
					const double* const am = a + m * n;
					double v = ar[m];
					size_t p;
					for (p = m + 1; p < k1; p++)
						v -= ar[p] * am[p];
					ar[m] = v / am[m];
				}
			}
#ifdef USE_DISPATCH
		});
#else
		} parallel_endfor
#endif
		if (k0 > 0)
			_ccv_nnc_8i_rowwise_x_rank_update(a, n, k0, k1);
		k1 = k0;
	}
	return 1;
}

// Block UDU^T of H = hessian + ridge I over quantization groups, U unit upper block-triangular. Walking
// the groups in order with targets t_k = x_k + sum_{j in earlier groups} e_j U[j][k], where e = x - x_hat,
// makes e H e^T = sum_b r_b D_b r_b^T for the rounding residuals r = t - x_hat. Each group then only rounds
// against D_b (the input variance conditioned on later groups) instead of the imatrix diagonal.
static void _ccv_nnc_8i_rowwise_x_feedback_init(ccv_nnc_8i_rowwise_x_feedback_t* const feedback, const float* const hessian, const size_t row_length, const size_t padded_row_length, const size_t group_size)
{
	double* const u = feedback->u;
	const size_t n = padded_row_length;
	size_t i, j;
	double mean_diag = 0;
	for (i = 0; i < row_length; i++)
		mean_diag += hessian[i * row_length + i];
	mean_diag /= (double)row_length;
	if (!(mean_diag > 0))
		mean_diag = 1;
	double ridge = mean_diag * 1e-6;
	int factored = 0;
	int attempt;
	for (attempt = 0; !factored && attempt < 8; attempt++)
	{
		if (attempt > 0)
			ridge *= 10;
		for (i = 0; i < n; i++)
		{
			for (j = i; j < n; j++)
				u[i * n + j] = (j < row_length) ? hessian[i * row_length + j] : 0;
			u[i * n + i] += i < row_length ? ridge : mean_diag;
		}
		factored = _ccv_nnc_8i_rowwise_x_reverse_cholesky(u, n);
	}
	if (!factored)
	{
		// Not positive definite even with a large ridge (non-finite input): drop the feedback.
		memset(u, 0, sizeof(double) * n * n);
		for (i = 0; i < n; i++)
			u[i * n + i] = sqrt(i < row_length ? ccv_max((double)hessian[i * row_length + i], 0.) + ridge : mean_diag);
	}
	feedback->hessian = hessian;
	feedback->ridge = ridge;
	if (feedback->hessian64)
		for (i = 0; i < row_length * row_length; i++)
			feedback->hessian64[i] = hessian[i];
	const size_t groups = n / group_size;
	size_t g;
	for (g = 0; g < groups; g++)
	{
		const size_t s = g * group_size;
		double* const rbb = feedback->rbb + g * group_size * group_size;
		size_t m, l;
		for (m = 0; m < group_size; m++)
		{
			double d = 0;
			for (l = 0; l < group_size; l++)
			{
				const double v = l >= m ? u[(s + m) * n + s + l] : 0;
				rbb[m * group_size + l] = v;
				d += v * v;
			}
			feedback->weights[s + m] = s + m < row_length ? d : 0;
		}
		if (s == 0)
			continue;
		double inv_data[32 * 32] = {0};
		double* const inv = inv_data;
		for (l = 0; l < group_size; l++)
		{
			inv[l * group_size + l] = 1. / rbb[l * group_size + l];
			m = l;
			while (m-- > 0)
			{
				double v = 0;
				size_t t;
				for (t = m + 1; t <= l; t++)
					v += rbb[m * group_size + t] * inv[t * group_size + l];
				inv[m * group_size + l] = -v / rbb[m * group_size + m];
			}
		}
		// U[j][group g] = R[j][group g] R_gg^-1 for every lane j in earlier groups.
		const size_t blocks = (s + 63) / 64;
#ifdef USE_DISPATCH
		dispatch_apply(blocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t block_idx) {
#else
		parallel_for(block_idx, (int)blocks) {
#endif
			const size_t row_end = ccv_min(((size_t)block_idx + 1) * 64, s);
			size_t r, a, b;
			for (r = (size_t)block_idx * 64; r < row_end; r++)
			{
				double* const ur = u + r * n + s;
				double t[32];
				for (b = 0; b < group_size; b++)
				{
					double v = 0;
					for (a = 0; a <= b; a++)
						v += ur[a] * inv[a * group_size + b];
					t[b] = v;
				}
				memcpy(ur, t, sizeof(double) * group_size);
			}
#ifdef USE_DISPATCH
		});
#else
		} parallel_endfor
#endif
	}
}

typedef struct {
	const void* input;
	int datatype;
	int format;
	int codec;
	int starts;
	size_t row_length;
	size_t row_count;
	const float* imatrix;
	size_t imatrix_length;
	size_t group_size;
	size_t groups_per_row;
	size_t padded_row_length;
	const int16_t* iq1_map;
	uint8_t* u8;
	uint8_t* scales;
} ccv_nnc_8i_rowwise_x_rows_t;

static double _ccv_nnc_8i_rowwise_x_start_scale(const double max_abs, const int start, const int starts, const int datatype)
{
	double scale = max_abs / 127.;
	if (starts == 3)
	{
		scale *= start == 0 ? 0.75 : start == 1 ? 1. : 1.25;
		if (datatype == CCV_16F && max_abs > 0 && _ccv_nnc_8i_rowwise_packed_stored_scale(scale, datatype) == 0)
			scale = 0x1p-24;
	}
	return scale;
}

static void _ccv_nnc_8i_rowwise_x_pack_row(const ccv_nnc_8i_rowwise_x_rows_t* const ctx, const size_t i, const double* const row, const double* const weights, const int has_best, const double best_scale, const ccv_nnc_8i_rowwise_packed_group_t* const best_groups)
{
	const size_t group_size = ctx->group_size;
	_ccv_nnc_8i_rowwise_packed_store_scale(ctx->scales, ctx->datatype, i, best_scale);
	size_t g, j;
	if (has_best)
	{
		for (g = 0; g < ctx->groups_per_row; g++)
			_ccv_nnc_8i_rowwise_packed_pack_group(ctx->u8, i * ctx->groups_per_row + g, ctx->codec, best_groups + g);
	} else {
		for (g = 0; g < ctx->groups_per_row; g++)
		{
			double y[32] = {0};
			double w[32] = {0};
			for (j = 0; j < group_size; j++)
			{
				y[j] = row[g * group_size + j];
				w[j] = weights[g * group_size + j];
			}
			ccv_nnc_8i_rowwise_packed_group_t group;
			_ccv_nnc_8i_rowwise_packed_quant_group(ctx->codec, y, w, ctx->iq1_map, &group);
			_ccv_nnc_8i_rowwise_packed_pack_group(ctx->u8, i * ctx->groups_per_row + g, ctx->codec, &group);
		}
	}
}

static void _ccv_nnc_8i_rowwise_x_quantize_rows(const ccv_nnc_8i_rowwise_x_rows_t* const ctx, const size_t row_begin, const size_t row_end)
{
	const size_t row_length = ctx->row_length;
	const size_t padded_row_length = ctx->padded_row_length;
	const size_t group_size = ctx->group_size;
	const size_t groups_per_row = ctx->groups_per_row;
	const int datatype = ctx->datatype;
	double* const row = (double*)ccmalloc(sizeof(double) * padded_row_length);
	double* const weights = (double*)ccmalloc(sizeof(double) * padded_row_length);
	ccv_nnc_8i_rowwise_packed_group_t* const groups = (ccv_nnc_8i_rowwise_packed_group_t*)ccmalloc(sizeof(ccv_nnc_8i_rowwise_packed_group_t) * groups_per_row);
	ccv_nnc_8i_rowwise_packed_group_t* const best_groups = (ccv_nnc_8i_rowwise_packed_group_t*)ccmalloc(sizeof(ccv_nnc_8i_rowwise_packed_group_t) * groups_per_row);
	size_t i;
	for (i = row_begin; i < row_end; i++)
	{
		const size_t row_start = i * row_length;
		const float* const row_imatrix = _ccv_nnc_8i_rowwise_imatrix_for_row(ctx->imatrix, ctx->imatrix_length, row_length, ctx->row_count, i);
		_ccv_nnc_8i_rowwise_packed_read_row(ctx->input, datatype, row_start, row_length, padded_row_length, row);
		if (ctx->format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256)
			_ccv_nnc_8i_rowwise_hadamard_256(row, row_length);
		double max_abs = 0;
		size_t j;
		for (j = 0; j < row_length; j++)
		{
			max_abs = ccv_max(max_abs, fabs(row[j]));
			weights[j] = _ccv_nnc_8i_rowwise_weight(row_imatrix, j);
		}
		for (; j < padded_row_length; j++)
			weights[j] = 0;
		double best_scale = 0;
		double best_sse = DBL_MAX;
		int has_best = 0;
		int start;
		for (start = 0; start < ctx->starts; start++)
		{
			double scale = _ccv_nnc_8i_rowwise_x_start_scale(max_abs, start, ctx->starts, datatype);
			int k;
			for (k = 0; k < CCV_NNC_8I_ROWWISE_X_REFINEMENT_STEPS; k++)
			{
				const double stored_scale = _ccv_nnc_8i_rowwise_packed_stored_scale(scale, datatype);
				if (!(stored_scale > 0))
					break;
				double sse = 0;
				double sum_qx = 0;
				double sum_qq = 0;
				size_t g;
				for (g = 0; g < groups_per_row; g++)
				{
					double y[32] = {0};
					double w[32] = {0};
					for (j = 0; j < group_size; j++)
					{
						y[j] = row[g * group_size + j] / stored_scale;
						w[j] = weights[g * group_size + j];
					}
					ccv_nnc_8i_rowwise_packed_group_t group;
					_ccv_nnc_8i_rowwise_packed_quant_group(ctx->codec, y, w, ctx->iq1_map, &group);
					groups[g] = group;
					const size_t group_start = g * group_size;
					const size_t group_end = ccv_min(group_start + group_size, row_length);
					for (j = group_start; j < group_end; j++)
					{
						const int q8 = group.q8[j - group_start];
						const double d = row[j] - stored_scale * q8;
						sse += weights[j] * d * d;
						sum_qx += weights[j] * q8 * row[j];
						sum_qq += weights[j] * q8 * q8;
					}
				}
				if (sse < best_sse)
				{
					best_sse = sse;
					best_scale = stored_scale;
					has_best = 1;
					memcpy(best_groups, groups, sizeof(ccv_nnc_8i_rowwise_packed_group_t) * groups_per_row);
				}
				if (!(sum_qq > 0) || !(sum_qx > 0))
					break;
				const double next_scale = sum_qx / sum_qq;
				if (_ccv_nnc_8i_rowwise_packed_stored_scale(next_scale, datatype) == stored_scale)
					break;
				scale = next_scale;
			}
		}
		_ccv_nnc_8i_rowwise_x_pack_row(ctx, i, row, weights, has_best, best_scale, best_groups);
	}
	ccfree(best_groups);
	ccfree(groups);
	ccfree(weights);
	ccfree(row);
}

// The same refinement as the imatrix path, with each pass adding group errors into later targets. Rows of a
// chunk run their passes in lockstep so every pass reads the feedback matrix once per chunk, not per row.
static void _ccv_nnc_8i_rowwise_x_quantize_rows_feedback(const ccv_nnc_8i_rowwise_x_rows_t* const ctx, const ccv_nnc_8i_rowwise_x_feedback_t* const fb, const size_t row_begin, const size_t row_end)
{
	enum { R = CCV_NNC_8I_ROWWISE_X_FEEDBACK_ROWS };
	const size_t rows = row_end - row_begin;
	assert(rows <= R);
	const size_t row_length = ctx->row_length;
	const size_t n = ctx->padded_row_length;
	const size_t group_size = ctx->group_size;
	const size_t groups_per_row = ctx->groups_per_row;
	double* const x = (double*)ccmalloc(sizeof(double) * rows * n);
	double* const target = (double*)ccmalloc(sizeof(double) * n * R); // Lane-major: target[l * R + r].
	double* const hx = (double*)ccmalloc(sizeof(double) * rows * row_length);
	ccv_nnc_8i_rowwise_packed_group_t* const groups = (ccv_nnc_8i_rowwise_packed_group_t*)ccmalloc(sizeof(ccv_nnc_8i_rowwise_packed_group_t) * rows * groups_per_row);
	ccv_nnc_8i_rowwise_packed_group_t* const best_groups = (ccv_nnc_8i_rowwise_packed_group_t*)ccmalloc(sizeof(ccv_nnc_8i_rowwise_packed_group_t) * rows * groups_per_row);
	double max_abs[R], xhx[R], scale[R], stored_scale[R], sse[R], sum_qx[R], best_sse[R], best_scale[R];
	int has_best[R], start[R], step[R], ready[R], done[R];
	double e[32][R];
	size_t r, j, l;
	for (r = 0; r < rows; r++)
	{
		double* const xr = x + r * n;
		_ccv_nnc_8i_rowwise_packed_read_row(ctx->input, ctx->datatype, (row_begin + r) * row_length, row_length, n, xr);
		max_abs[r] = 0;
		for (j = 0; j < row_length; j++)
			max_abs[r] = ccv_max(max_abs[r], fabs(xr[j]));
		xhx[r] = 0;
		best_sse[r] = DBL_MAX;
		best_scale[r] = 0;
		has_best[r] = 0;
		start[r] = 0;
		step[r] = 0;
		done[r] = 0;
		scale[r] = _ccv_nnc_8i_rowwise_x_start_scale(max_abs[r], 0, ctx->starts, ctx->datatype);
	}
	// hx = (hessian + ridge I) x, reading each Hessian row once for the chunk.
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
	// hx[r][j] = sum_l hessian[j][l] x[r][l] for the whole chunk in one product.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows, (int)row_length, (int)row_length, 1, x, (int)n, fb->hessian64, (int)row_length, 0, hx, (int)row_length);
	for (r = 0; r < rows; r++)
	{
		const double* const xr = x + r * n;
		for (j = 0; j < row_length; j++)
		{
			const double v = hx[r * row_length + j] + fb->ridge * xr[j];
			hx[r * row_length + j] = v;
			xhx[r] += xr[j] * v;
		}
	}
#else
	for (j = 0; j < row_length; j++)
	{
		const float* const cj = fb->hessian + j * row_length;
		for (r = 0; r < rows; r++)
		{
			const double* const xr = x + r * n;
			double v = fb->ridge * xr[j];
			for (l = 0; l < row_length; l++)
				v += cj[l] * xr[l];
			hx[r * row_length + j] = v;
			xhx[r] += xr[j] * v;
		}
	}
#endif
	for (;;)
	{
		int any = 0;
		for (r = 0; r < rows; r++)
		{
			ready[r] = 0;
			while (!done[r])
			{
				stored_scale[r] = _ccv_nnc_8i_rowwise_packed_stored_scale(scale[r], ctx->datatype);
				if (stored_scale[r] > 0)
				{
					ready[r] = 1;
					any = 1;
					break;
				}
				if (++start[r] >= ctx->starts)
					done[r] = 1;
				else {
					step[r] = 0;
					scale[r] = _ccv_nnc_8i_rowwise_x_start_scale(max_abs[r], start[r], ctx->starts, ctx->datatype);
				}
			}
			sse[r] = 0;
			sum_qx[r] = 0;
		}
		if (!any)
			break;
		for (l = 0; l < n; l++)
			for (r = 0; r < R; r++)
				target[l * R + r] = (r < rows && ready[r]) ? x[r * n + l] : 0;
		size_t g;
		for (g = 0; g < groups_per_row; g++)
		{
			const size_t group_start = g * group_size;
			const size_t group_end = ccv_min(group_start + group_size, row_length);
			const size_t next_group = group_start + group_size;
			memset(e, 0, sizeof(e));
			for (r = 0; r < rows; r++)
			{
				if (!ready[r])
					continue;
				double y[32] = {0};
				double w[32] = {0};
				for (j = 0; j < group_size; j++)
				{
					y[j] = target[(group_start + j) * R + r] / stored_scale[r];
					w[j] = fb->weights[group_start + j];
				}
				ccv_nnc_8i_rowwise_packed_group_t* const group = groups + r * groups_per_row + g;
				_ccv_nnc_8i_rowwise_packed_quant_group(ctx->codec, y, w, ctx->iq1_map, group);
				double res[32] = {0};
				for (j = group_start; j < group_end; j++)
				{
					const int q8 = group->q8[j - group_start];
					res[j - group_start] = target[j * R + r] - stored_scale[r] * q8;
					e[j - group_start][r] = x[r * n + j] - stored_scale[r] * q8;
					sum_qx[r] += q8 * hx[r * row_length + j];
				}
				// r_b D_b r_b^T = |r_b R_bb|^2.
				const double* const rbb = fb->rbb + g * group_size * group_size;
				size_t m;
				for (l = 0; l < group_size; l++)
				{
					double v = 0;
					for (m = 0; m <= l; m++)
						v += res[m] * rbb[m * group_size + l];
					sse[r] += v * v;
				}
			}
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
			// target[l][r] += sum_j U[group_start + j][l] e[j][r] for every later lane l, as one product.
			if (next_group < n)
				cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)(n - next_group), R, (int)(group_end - group_start), 1, fb->u + group_start * n + next_group, (int)n, &e[0][0], R, 1, target + next_group * R, R);
#else
			// Later lanes in tiles that stay in cache across this group's feedback rows.
			size_t l0;
			for (l0 = next_group; l0 < n; l0 += 64)
			{
				const size_t l1 = ccv_min(l0 + 64, n);
				for (j = 0; j < group_end - group_start; j++)
				{
					const double* const uj = fb->u + (group_start + j) * n;
					const double* const ej = e[j];
					for (l = l0; l < l1; l++)
					{
						const double u = uj[l];
						double* const t = target + l * R;
						for (r = 0; r < R; r++)
							t[r] += ej[r] * u;
					}
				}
			}
#endif
		}
		for (r = 0; r < rows; r++)
		{
			if (!ready[r])
				continue;
			// sse = e H e^T = x H x - 2 s q H x + s^2 q H q.
			const double s = stored_scale[r];
			const double sum_qq = (sse[r] - xhx[r] + 2 * s * sum_qx[r]) / (s * s);
			if (sse[r] < best_sse[r])
			{
				best_sse[r] = sse[r];
				best_scale[r] = s;
				has_best[r] = 1;
				memcpy(best_groups + r * groups_per_row, groups + r * groups_per_row, sizeof(ccv_nnc_8i_rowwise_packed_group_t) * groups_per_row);
			}
			int next_start = 1;
			if (sum_qq > 0 && sum_qx[r] > 0)
			{
				const double next_scale = sum_qx[r] / sum_qq;
				if (_ccv_nnc_8i_rowwise_packed_stored_scale(next_scale, ctx->datatype) != s && ++step[r] < CCV_NNC_8I_ROWWISE_X_REFINEMENT_STEPS)
				{
					scale[r] = next_scale;
					next_start = 0;
				}
			}
			if (next_start)
			{
				step[r] = 0;
				if (++start[r] >= ctx->starts)
					done[r] = 1;
				else
					scale[r] = _ccv_nnc_8i_rowwise_x_start_scale(max_abs[r], start[r], ctx->starts, ctx->datatype);
			}
		}
	}
	for (r = 0; r < rows; r++)
		_ccv_nnc_8i_rowwise_x_pack_row(ctx, row_begin + r, x + r * n, fb->weights, has_best[r], best_scale[r], best_groups + r * groups_per_row);
	ccfree(best_groups);
	ccfree(groups);
	ccfree(hx);
	ccfree(target);
	ccfree(x);
}

static size_t _ccv_nnc_quantize_8i_rowwise_x(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, const float* const imatrix, const size_t imatrix_length, const float* const hessian, const size_t hessian_length, void* output, const size_t output_length)
{
	const int codec = format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK;
	const size_t row_count = input_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const int group_bits = ccv_nnc_8i_rowwise_x_group_bits(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t padded_row_length = groups_per_row * group_size;
	const size_t scale_offset = _ccv_nnc_8i_rowwise_packed_scale_offset(format, input_length, row_length);
	const size_t output_size = scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype);
	if (output_length < output_size)
		return 0;
	int16_t* iq1_map = 0;
	const int starts = (codec == CCV_NNC_QX_8I_ROWWISE_IQ1_S || codec == CCV_NNC_QX_8I_ROWWISE_IQ1_M) ? 3 : 1;
	if (starts == 3)
	{
		iq1_map = (int16_t*)ccmalloc(sizeof(int16_t) * 6561);
		int i, j;
		for (i = 0; i < 6561; i++)
			iq1_map[i] = -1;
		for (i = 0; i < 2048; i++)
		{
			int code = 0;
			for (j = 0; j < 8; j++)
				code += ((ccv_nnc_8i_rowwise_packed_iq1_grid[i] >> (2 * j)) & 3) * ccv_nnc_8i_rowwise_packed_iq1_powers[j];
			iq1_map[code] = (int16_t)i;
		}
	}
	switch (codec)
	{
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XXS:
			_ccv_nnc_8i_rowwise_packed_iq2xxs_candidates_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_XS:
			_ccv_nnc_8i_rowwise_packed_iq2xs_candidates_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ2_S:
			_ccv_nnc_8i_rowwise_packed_iq2s_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_S:
			_ccv_nnc_8i_rowwise_packed_iq3s_init();
			break;
		case CCV_NNC_QX_8I_ROWWISE_IQ3_XXS:
			_ccv_nnc_8i_rowwise_packed_iq3xxs_candidates_init();
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
	rows_per_chunk = ccv_max(rows_per_chunk, (size_t)8);
	// Feedback chunks are a multiple of the packing chunk, so they stay byte aligned.
	if (hessian)
		rows_per_chunk = CCV_NNC_8I_ROWWISE_X_FEEDBACK_ROWS;
	const ccv_nnc_8i_rowwise_x_rows_t rows_data = {
		.input = input,
		.datatype = datatype,
		.format = format,
		.codec = codec,
		.starts = starts,
		.row_length = row_length,
		.row_count = row_count,
		.imatrix = imatrix,
		.imatrix_length = imatrix_length,
		.group_size = group_size,
		.groups_per_row = groups_per_row,
		.padded_row_length = padded_row_length,
		.iq1_map = iq1_map,
		.u8 = u8,
		.scales = scales,
	};
	const ccv_nnc_8i_rowwise_x_rows_t* const ctx = &rows_data;
	// Each Hessian slice is factored once and shared by its rows. Slices run one after another, so a
	// chunk that straddles two slices never writes its packed bytes concurrently.
	const size_t slices = hessian ? hessian_length / (row_length * row_length) : 1;
	const size_t rows_per_slice = row_count / slices;
	ccv_nnc_8i_rowwise_x_feedback_t feedback = {0};
	if (hessian)
	{
		feedback.u = (double*)ccmalloc(sizeof(double) * padded_row_length * padded_row_length);
		feedback.rbb = (double*)ccmalloc(sizeof(double) * padded_row_length * group_size);
		feedback.weights = (double*)ccmalloc(sizeof(double) * padded_row_length);
#if (defined HAVE_CBLAS || defined HAVE_ACCELERATE_FRAMEWORK)
		feedback.hessian64 = (double*)ccmalloc(sizeof(double) * row_length * row_length);
#endif
	}
	size_t slice;
	for (slice = 0; slice < slices; slice++)
	{
		const size_t row_begin = slice * rows_per_slice;
		const size_t row_end = row_begin + rows_per_slice;
		if (hessian)
			_ccv_nnc_8i_rowwise_x_feedback_init(&feedback, hessian + slice * row_length * row_length, row_length, padded_row_length, group_size);
		const ccv_nnc_8i_rowwise_x_feedback_t* const fb = hessian ? &feedback : 0;
		const size_t first_chunk = row_begin / rows_per_chunk;
		const size_t row_chunks = (row_end + rows_per_chunk - 1) / rows_per_chunk - first_chunk;
#ifdef USE_DISPATCH
		dispatch_apply(row_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t chunk_idx) {
#else
		parallel_for(chunk_idx, (int)row_chunks) {
#endif
			const size_t chunk_begin = ccv_max((first_chunk + (size_t)chunk_idx) * rows_per_chunk, row_begin);
			const size_t chunk_end = ccv_min((first_chunk + (size_t)chunk_idx + 1) * rows_per_chunk, row_end);
			if (fb)
				_ccv_nnc_8i_rowwise_x_quantize_rows_feedback(ctx, fb, chunk_begin, chunk_end);
			else
				_ccv_nnc_8i_rowwise_x_quantize_rows(ctx, chunk_begin, chunk_end);
#ifdef USE_DISPATCH
		});
#else
		} parallel_endfor
#endif
	}
	if (hessian)
	{
		ccfree(feedback.weights);
		ccfree(feedback.rbb);
		ccfree(feedback.u);
		if (feedback.hessian64)
			ccfree(feedback.hessian64);
	}
	ccfree(iq1_map);
	return output_size;
}

CCV_WARN_UNUSED(size_t) ccv_nnc_quantize_8i_rowwise_x(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, const int format, const float* const imatrix, const size_t imatrix_length, const float* const hessian, const size_t hessian_length, void* output, const size_t output_length)
{
	if (row_length == 0 || input_length % row_length != 0 || (imatrix && hessian) ||
		((format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256) && (row_length % 256 != 0 || hessian)))
		return 0;
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	const size_t row_count = input_length / row_length;
	if (!_ccv_nnc_8i_rowwise_imatrix_is_valid(imatrix, imatrix_length, row_length, row_count))
		return 0;
	if (hessian)
	{
		const size_t matrix_length = row_length * row_length;
		if (hessian_length < matrix_length || hessian_length % matrix_length != 0 || row_count % (hessian_length / matrix_length) != 0)
			return 0;
	}
	return _ccv_nnc_quantize_8i_rowwise_x(input, datatype, input_length, row_length, format, imatrix, imatrix_length, hessian, hessian_length, output, output_length);
}

void ccv_nnc_dequantize_8i_rowwise_x(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY || memory_type == CCV_TENSOR_GPU_MEMORY);
	assert(row_length > 0);
	assert(output_length % row_length == 0);
	assert(!(format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256) || row_length % 256 == 0);
	if (memory_type != CCV_TENSOR_CPU_MEMORY)
	{
#ifdef HAVE_CUDA
		ccv_nnc_compat_dequantize_8i_rowwise_x_fp(input, datatype, input_length, row_length, format, output, output_length, 0);
#elif defined(HAVE_MPS)
		assert(datatype != CCV_64F);
		ccv_nnc_mps_dequantize_8i_rowwise_x(input, datatype, input_length, row_length, format, output, output_length, 0);
#else
		assert(memory_type == CCV_TENSOR_CPU_MEMORY);
#endif
		return;
	}
	const size_t row_count = output_length / row_length;
	const size_t group_size = ccv_nnc_8i_rowwise_x_group_size(format);
	const size_t groups_per_row = (row_length + group_size - 1) / group_size;
	const size_t scale_offset = _ccv_nnc_8i_rowwise_packed_scale_offset(format, output_length, row_length);
	assert(input_length >= scale_offset + row_count * CCV_GET_DATA_TYPE_SIZE(datatype));
	const uint8_t* const u8 = (const uint8_t*)input;
	const uint8_t* const scales = u8 + scale_offset;
	parallel_for(i, (int)row_count) {
		const double scale = _ccv_nnc_8i_rowwise_packed_load_scale(scales, datatype, i);
		if (format & CCV_NNC_QX_8I_ROWWISE_HADAMARD_256)
		{
			// Transform decoded integers before scaling / rounding to the destination
			// precision. H256 is its own inverse and acts on independent blocks.
			double block[256];
			size_t base;
			for (base = 0; base < row_length; base += 256)
			{
				size_t g;
				for (g = 0; g < 256 / group_size; g++)
				{
					int q8[32] = {0};
					_ccv_nnc_8i_rowwise_packed_decode_group(u8, (size_t)i * groups_per_row + base / group_size + g, format & CCV_NNC_QX_8I_ROWWISE_FORMAT_MASK, q8);
					size_t j;
					for (j = 0; j < group_size; j++)
						block[g * group_size + j] = q8[j];
				}
				_ccv_nnc_8i_rowwise_hadamard_256(block, 256);
				size_t j;
				for (j = 0; j < 256; j++)
					_ccv_nnc_8i_rowwise_packed_write_value(output, datatype, (size_t)i * row_length + base + j, scale * block[j]);
			}
		} else {
			size_t g;
			for (g = 0; g < groups_per_row; g++)
			{
				int q8[32] = {0};
				_ccv_nnc_8i_rowwise_packed_decode_group(u8, (size_t)i * groups_per_row + g, format, q8);
				size_t j;
				for (j = 0; j < group_size; j++)
				{
					const size_t col = g * group_size + j;
					if (col < row_length)
						_ccv_nnc_8i_rowwise_packed_write_value(output, datatype, (size_t)i * row_length + col, scale * q8[j]);
				}
			}
		}
	} parallel_endfor
}

static inline int _ccv_nnc_8i_rowwise_quantize(const double v, const double inv_scale)
{
	const int q = (int)lrint(v * inv_scale);
	return ccv_clamp(q, -127, 127);
}

static float _ccv_nnc_quantize_8i_rowwise_16f(const uint16_t* const row, const size_t row_length, const float* const imatrix, int8_t* const q)
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
			const double w = _ccv_nnc_8i_rowwise_weight(imatrix, j);
			float v_f;
			ccv_half_precision_to_float(row + j, &v_f, 1);
			const double v = v_f;
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += w * d * d;
			sum_qx += w * qj * v;
			sum_qq += w * qj * qj;
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

static float _ccv_nnc_quantize_8i_rowwise_16bf(const uint16_t* const row, const size_t row_length, const float* const imatrix, int8_t* const q)
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
			const double w = _ccv_nnc_8i_rowwise_weight(imatrix, j);
			float v_f;
			ccv_bfloat_to_float(row + j, &v_f, 1);
			const double v = v_f;
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += w * d * d;
			sum_qx += w * qj * v;
			sum_qq += w * qj * qj;
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

static float _ccv_nnc_quantize_8i_rowwise_32f(const float* const row, const size_t row_length, const float* const imatrix, int8_t* const q)
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
			const double w = _ccv_nnc_8i_rowwise_weight(imatrix, j);
			const double v = row[j];
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += w * d * d;
			sum_qx += w * qj * v;
			sum_qq += w * qj * qj;
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

static double _ccv_nnc_quantize_8i_rowwise_64f(const double* const row, const size_t row_length, const float* const imatrix, int8_t* const q)
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
			const double w = _ccv_nnc_8i_rowwise_weight(imatrix, j);
			const double v = row[j];
			const int qj = _ccv_nnc_8i_rowwise_quantize(v, inv_scale);
			const double d = v - stored_scale * qj;
			sse += w * d * d;
			sum_qx += w * qj * v;
			sum_qq += w * qj * qj;
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

CCV_WARN_UNUSED(size_t) ccv_nnc_quantize_8i_rowwise(const void* input, const int datatype, const int memory_type, const size_t input_length, const size_t row_length, const float* const imatrix, const size_t imatrix_length, void* output, const size_t output_length)
{
	assert(datatype == CCV_16F || datatype == CCV_16BF || datatype == CCV_32F || datatype == CCV_64F);
	assert(memory_type == CCV_TENSOR_CPU_MEMORY);
	assert(row_length > 0);
	assert(input_length % row_length == 0);
	const size_t row_count = input_length / row_length;
	if (!_ccv_nnc_8i_rowwise_imatrix_is_valid(imatrix, imatrix_length, row_length, row_count))
		return 0;
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
			const float* const row_imatrix = _ccv_nnc_8i_rowwise_imatrix_for_row(imatrix, imatrix_length, row_length, row_count, (size_t)i);
			const float scale_f = _ccv_nnc_quantize_8i_rowwise_16f(f16 + row_start, row_length, row_imatrix, q + row_start);
			ccv_float_to_half_precision(&scale_f, scales + i, 1);
		} parallel_endfor
	} else if (datatype == CCV_16BF) {
		const uint16_t* const bf16 = (const uint16_t*)input;
		uint16_t* const scales = (uint16_t*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float* const row_imatrix = _ccv_nnc_8i_rowwise_imatrix_for_row(imatrix, imatrix_length, row_length, row_count, (size_t)i);
			const float scale_f = _ccv_nnc_quantize_8i_rowwise_16bf(bf16 + row_start, row_length, row_imatrix, q + row_start);
			ccv_float_to_bfloat(&scale_f, scales + i, 1);
		} parallel_endfor
	} else if (datatype == CCV_32F) {
		const float* const f32 = (const float*)input;
		float* const scales = (float*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float* const row_imatrix = _ccv_nnc_8i_rowwise_imatrix_for_row(imatrix, imatrix_length, row_length, row_count, (size_t)i);
			scales[i] = _ccv_nnc_quantize_8i_rowwise_32f(f32 + row_start, row_length, row_imatrix, q + row_start);
		} parallel_endfor
	} else {
		assert(datatype == CCV_64F);
		const double* const f64 = (const double*)input;
		double* const scales = (double*)(u8 + scale_offset);
		parallel_for(i, (int)row_count) {
			const size_t row_start = (size_t)i * row_length;
			const float* const row_imatrix = _ccv_nnc_8i_rowwise_imatrix_for_row(imatrix, imatrix_length, row_length, row_count, (size_t)i);
			scales[i] = _ccv_nnc_quantize_8i_rowwise_64f(f64 + row_start, row_length, row_imatrix, q + row_start);
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
