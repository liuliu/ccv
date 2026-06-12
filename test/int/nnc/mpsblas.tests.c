#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _mps_forward_na_gemm_a_value(const int row, const int k)
{
	return (float)(((row * 17 + k * 13) % 23) + 1) / 512.0f;
}

static float _mps_forward_na_gemm_b_value(const int col, const int k)
{
	return (float)(((col * 19 + k * 7) % 29) + 1) / 512.0f;
}

static float _mps_forward_na_gemm_signed_a_value(const int row, const int k)
{
	return (float)(((row * 31 + k * 17) % 257) - 128) / 128.0f;
}

static float _mps_forward_na_gemm_signed_b_value(const int col, const int k)
{
	return (float)(((col * 13 + k * 29) % 251) - 125) / 128.0f;
}

static float _mps_forward_na_gemm_bias_value(const int col)
{
	return (float)(((col * 5) % 17) - 8) / 256.0f;
}

static void _mps_forward_na_gemm_fill_half(ccv_float16_t* const data, const int rows, const int cols, const int for_a)
{
	float* const row_buffer = (float*)ccmalloc(sizeof(float) * cols);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
			row_buffer[j] = for_a ? _mps_forward_na_gemm_a_value(i, j) : _mps_forward_na_gemm_b_value(i, j);
		ccv_float_to_half_precision(row_buffer, (uint16_t*)data + (size_t)i * cols, cols);
	}
	ccfree(row_buffer);
}

static void _mps_forward_scaled_gemm_to_float(const int datatype, const void* const data, const int count, float* const values);

static float _mps_forward_na_gemm_round_value(const int datatype, const float value)
{
	if (datatype == CCV_16F)
	{
		uint16_t h;
		float f;
		ccv_float_to_half_precision(&value, &h, 1);
		ccv_half_precision_to_float(&h, &f, 1);
		return f;
	} else if (datatype == CCV_16BF) {
		uint16_t h;
		float f;
		ccv_float_to_bfloat(&value, &h, 1);
		ccv_bfloat_to_float(&h, &f, 1);
		return f;
	}
	return value;
}

static void _mps_forward_na_gemm_fill(const int datatype, void* const data, const int rows, const int cols, const int for_a)
{
	float* const values = (float*)ccmalloc(sizeof(float) * rows * cols);
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			values[i * cols + j] = for_a ? _mps_forward_na_gemm_a_value(i, j) : _mps_forward_na_gemm_b_value(i, j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, rows * cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, rows * cols);
	else
		memcpy(data, values, sizeof(float) * rows * cols);
	ccfree(values);
}

static void _mps_forward_na_gemm_fill_signed(const int datatype, void* const data, const int rows, const int cols, const int for_a)
{
	float* const values = (float*)ccmalloc(sizeof(float) * rows * cols);
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			values[i * cols + j] = for_a ? _mps_forward_na_gemm_signed_a_value(i, j) : _mps_forward_na_gemm_signed_b_value(i, j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, rows * cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, rows * cols);
	else
		memcpy(data, values, sizeof(float) * rows * cols);
	ccfree(values);
}

static void _mps_forward_na_gemm_fill_bias(const int datatype, void* const data, const int cols)
{
	float* const values = (float*)ccmalloc(sizeof(float) * cols);
	int j;
	for (j = 0; j < cols; j++)
		values[j] = _mps_forward_na_gemm_bias_value(j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, cols);
	else
		memcpy(data, values, sizeof(float) * cols);
	ccfree(values);
}

static float _mps_forward_na_gemm_expected(const int datatype, const int row, const int col, const int k_dim, const int use_bias)
{
	float sum = 0;
	int k;
	for (k = 0; k < k_dim; k++)
		sum += _mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_a_value(row, k)) *
			_mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_b_value(col, k));
	if (use_bias)
		sum += _mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_bias_value(col));
	return sum;
}

static float _mps_forward_na_gemm_expected_signed(const int datatype, const int row, const int col, const int k_dim, const int use_bias)
{
	float sum = 0;
	int k;
	for (k = 0; k < k_dim; k++)
		sum += _mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_signed_a_value(row, k)) *
			_mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_signed_b_value(col, k));
	if (use_bias)
		sum += _mps_forward_na_gemm_round_value(datatype, _mps_forward_na_gemm_bias_value(col));
	return sum;
}

static int _mps_forward_na_gemm_sample_indices(const int dim, const int boundary, const int include_large_m_boundary, int indices[8])
{
	const int candidates[] = {
		0, 1, boundary - 1, boundary,
		include_large_m_boundary ? 32767 : -1,
		include_large_m_boundary ? 32768 : -1,
		dim / 2, dim - 1,
	};
	int i, j;
	int count = 0;
	for (i = 0; i < 8; i++)
	{
		if (candidates[i] < 0 || candidates[i] >= dim)
			continue;
		for (j = 0; j < count; j++)
			if (indices[j] == candidates[i])
				break;
		if (j < count)
			continue;
		indices[count++] = candidates[i];
	}
	return count;
}

typedef struct {
	int row;
	int col;
	float actual;
	float expected;
	float max_abs;
	float max_rel;
} _mps_forward_na_gemm_mismatch_t;

static float _mps_forward_na_gemm_abs_tolerance(const int datatype)
{
	return datatype == CCV_16BF ? 2e-1f : 5e-2f;
}

static float _mps_forward_na_gemm_rel_tolerance(const int datatype)
{
	return datatype == CCV_16BF ? 5e-3f : 2e-3f;
}

static int _mps_forward_na_gemm_validate_shape_for_datatype(const int datatype, const int use_bias, const int m_dim, const int n_dim, const int k_dim, _mps_forward_na_gemm_mismatch_t* const mismatch)
{
	ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gw_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t sample_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { 1, 1, 0 },
	};
	ccv_nnc_tensor_param_t gsample_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { 1, 1, 0 },
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, gw_params, 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	_mps_forward_na_gemm_fill(datatype, ha->data.u8, m_dim, k_dim, 1);
	_mps_forward_na_gemm_fill(datatype, hw->data.u8, n_dim, k_dim, 0);
	if (use_bias)
		_mps_forward_na_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	else
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);

	int row_samples[8];
	int col_samples[8];
	const int row_sample_size = _mps_forward_na_gemm_sample_indices(m_dim, 128, 1, row_samples);
	const int col_sample_size = _mps_forward_na_gemm_sample_indices(n_dim, 64, 0, col_samples);
	ccv_nnc_tensor_t* const sample_h = ccv_nnc_tensor_new(0, sample_params, 0);
	int ok = 1;
	int i, j;
	for (i = 0; i < row_sample_size; i++)
		for (j = 0; j < col_sample_size; j++)
		{
			ccv_nnc_tensor_view_t* const bv = ccv_nnc_tensor_view_new(b, gsample_params, DIM_ALLOC(row_samples[i], col_samples[j]), DIM_ALLOC(n_dim, 1));
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)bv), TENSOR_LIST(sample_h), 0);
			mismatch->row = row_samples[i];
			mismatch->col = col_samples[j];
			_mps_forward_scaled_gemm_to_float(datatype, sample_h->data.u8, 1, &mismatch->actual);
			mismatch->expected = _mps_forward_na_gemm_expected(datatype, row_samples[i], col_samples[j], k_dim, use_bias);
			ccv_nnc_tensor_view_free(bv);
			const float abs_diff = fabsf(mismatch->actual - mismatch->expected);
			const float denom = fmaxf(fmaxf(fabsf(mismatch->actual), fabsf(mismatch->expected)), 1.0f);
			const float rel_diff = abs_diff / denom;
			if (abs_diff > mismatch->max_abs)
				mismatch->max_abs = abs_diff;
			if (rel_diff > mismatch->max_rel)
				mismatch->max_rel = rel_diff;
			if (abs_diff > _mps_forward_na_gemm_abs_tolerance(datatype) &&
				rel_diff > _mps_forward_na_gemm_rel_tolerance(datatype))
			{
				ok = 0;
				goto cleanup;
			}
		}

cleanup:
	ccv_nnc_tensor_free(sample_h);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	return ok;
}

static int _mps_forward_na_gemm_validate_full_shape_for_datatype(const int datatype, const int use_bias, const int signed_values, const int m_dim, const int n_dim, const int k_dim, _mps_forward_na_gemm_mismatch_t* const mismatch)
{
	ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gw_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, gw_params, 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, w_params, 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	if (signed_values)
	{
		_mps_forward_na_gemm_fill_signed(datatype, ha->data.u8, m_dim, k_dim, 1);
		_mps_forward_na_gemm_fill_signed(datatype, hw->data.u8, n_dim, k_dim, 0);
	} else {
		_mps_forward_na_gemm_fill(datatype, ha->data.u8, m_dim, k_dim, 1);
		_mps_forward_na_gemm_fill(datatype, hw->data.u8, n_dim, k_dim, 0);
	}
	if (use_bias)
		_mps_forward_na_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	else
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	float* const actual = (float*)ccmalloc(sizeof(float) * m_dim * n_dim);
	_mps_forward_scaled_gemm_to_float(datatype, hb->data.u8, m_dim * n_dim, actual);
	int ok = 1;
	int i, j;
	for (i = 0; i < m_dim; i++)
		for (j = 0; j < n_dim; j++)
		{
			const float expected = signed_values ? _mps_forward_na_gemm_expected_signed(datatype, i, j, k_dim, use_bias) : _mps_forward_na_gemm_expected(datatype, i, j, k_dim, use_bias);
			const float abs_diff = fabsf(actual[i * n_dim + j] - expected);
			const float denom = fmaxf(fmaxf(fabsf(actual[i * n_dim + j]), fabsf(expected)), 1.0f);
			const float rel_diff = abs_diff / denom;
			if (abs_diff > mismatch->max_abs)
			{
				mismatch->row = i;
				mismatch->col = j;
				mismatch->actual = actual[i * n_dim + j];
				mismatch->expected = expected;
				mismatch->max_abs = abs_diff;
			}
			if (rel_diff > mismatch->max_rel)
				mismatch->max_rel = rel_diff;
			if (abs_diff > _mps_forward_na_gemm_abs_tolerance(datatype) &&
				rel_diff > _mps_forward_na_gemm_rel_tolerance(datatype))
				ok = 0;
		}
	ccfree(actual);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	return ok;
}

static int _mps_forward_na_gemm_validate_shape(const int m_dim, const int n_dim, const int k_dim, _mps_forward_na_gemm_mismatch_t* const mismatch)
{
	return _mps_forward_na_gemm_validate_shape_for_datatype(CCV_16F, 0, m_dim, n_dim, k_dim, mismatch);
}

static int _mps_forward_na_gemm_validate_shape_with_bias(const int m_dim, const int n_dim, const int k_dim, _mps_forward_na_gemm_mismatch_t* const mismatch)
{
	return _mps_forward_na_gemm_validate_shape_for_datatype(CCV_16F, 1, m_dim, n_dim, k_dim, mismatch);
}

static float _mps_forward_ane_stream_lhs_value(const int row, const int k, const int variant)
{
	return (float)((((row * 31 + k * 17 + variant * 19) % 97) - 48)) / 64.0f;
}

static float _mps_forward_ane_stream_rhs_value(const int row, const int k, const int variant)
{
	return (float)((((row * 13 + k * 29 + variant * 23) % 89) - 44)) / 64.0f;
}

static void _mps_forward_ane_stream_fill_half(ccv_float16_t* const data, const int rows, const int cols, const int variant, const int for_lhs)
{
	float* const row_buffer = (float*)ccmalloc(sizeof(float) * cols);
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
			row_buffer[j] = for_lhs ? _mps_forward_ane_stream_lhs_value(i, j, variant) : _mps_forward_ane_stream_rhs_value(i, j, variant);
		ccv_float_to_half_precision(row_buffer, (uint16_t*)data + (size_t)i * cols, cols);
	}
	ccfree(row_buffer);
}

static int _mps_forward_ane_rowwise_gemm_stream_sync_validate(double* const max_abs_ref, double* const max_rel_ref)
{
	const int m_dim = 512;
	const int n_dim = 768;
	const int k_dim = 1024;
	const int writer_k = 4096;
	ccv_nnc_tensor_t* const hlhs_old = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, m_dim, writer_k), 0);
	ccv_nnc_tensor_t* const hrhs_old = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, k_dim, writer_k), 0);
	ccv_nnc_tensor_t* const hlhs_new = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, m_dim, writer_k), 0);
	ccv_nnc_tensor_t* const hrhs_new = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, k_dim, writer_k), 0);
	ccv_nnc_tensor_t* const hw_dense = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n_dim, k_dim), 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, n_dim, k_dim)), 0);
	ccv_nnc_tensor_t* const lhs = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, m_dim, writer_k), 0);
	ccv_nnc_tensor_t* const rhs = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, k_dim, writer_k), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, m_dim, k_dim), 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16F, n_dim, k_dim)), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const bref = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const hbref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, m_dim, n_dim), 0);
	ccv_nnc_stream_context_t* const stream_context = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	float* const actual = (float*)ccmalloc(sizeof(float) * m_dim * n_dim);
	float* const expected = (float*)ccmalloc(sizeof(float) * m_dim * n_dim);
	_mps_forward_ane_stream_fill_half(hlhs_old->data.f16, m_dim, writer_k, 0, 1);
	_mps_forward_ane_stream_fill_half(hrhs_old->data.f16, k_dim, writer_k, 0, 0);
	_mps_forward_ane_stream_fill_half(hlhs_new->data.f16, m_dim, writer_k, 1, 1);
	_mps_forward_ane_stream_fill_half(hrhs_new->data.f16, k_dim, writer_k, 1, 0);
	_mps_forward_na_gemm_fill_half(hw_dense->data.f16, n_dim, k_dim, 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(hw_dense->data.f16, CCV_16F, CCV_TENSOR_CPU_MEMORY, (size_t)n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
	{
		ccv_nnc_stream_context_free(stream_context);
		ccfree(expected);
		ccfree(actual);
		ccv_nnc_tensor_free(hbref);
		ccv_nnc_tensor_free(hb);
		ccv_nnc_tensor_free(bref);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(w);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(rhs);
		ccv_nnc_tensor_free(lhs);
		ccv_nnc_tensor_free(hwq);
		ccv_nnc_tensor_free(hw_dense);
		ccv_nnc_tensor_free(hrhs_new);
		ccv_nnc_tensor_free(hlhs_new);
		ccv_nnc_tensor_free(hrhs_old);
		ccv_nnc_tensor_free(hlhs_old);
		return -1;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hlhs_old, hrhs_old, hwq), TENSOR_LIST(lhs, rhs, w), stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(lhs, rhs), TENSOR_LIST(a), stream_context);
	ccv_nnc_synchronize_stream_context(stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), stream_context);
	ccv_nnc_synchronize_stream_context(stream_context);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hlhs_new, hrhs_new), TENSOR_LIST(lhs, rhs), stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(lhs, rhs), TENSOR_LIST(a), stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), stream_context);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), stream_context);
	ccv_nnc_synchronize_stream_context(stream_context);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hlhs_new, hrhs_new), TENSOR_LIST(lhs, rhs), stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(lhs, rhs), TENSOR_LIST(a), stream_context);
	ccv_nnc_synchronize_stream_context(stream_context);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(bref), stream_context);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(bref), TENSOR_LIST(hbref), stream_context);
	ccv_nnc_synchronize_stream_context(stream_context);

	_mps_forward_scaled_gemm_to_float(CCV_16F, hb->data.f16, m_dim * n_dim, actual);
	_mps_forward_scaled_gemm_to_float(CCV_16F, hbref->data.f16, m_dim * n_dim, expected);
	double max_abs = 0;
	double max_rel = 0;
	int i;
	for (i = 0; i < m_dim * n_dim; i++)
	{
		const double diff = fabs((double)actual[i] - (double)expected[i]);
		const double denom = ccv_max(1.0, ccv_max(fabs((double)actual[i]), fabs((double)expected[i])));
		max_abs = ccv_max(max_abs, diff);
		max_rel = ccv_max(max_rel, diff / denom);
	}
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	if (max_rel_ref)
		*max_rel_ref = max_rel;
	ccfree(expected);
	ccfree(actual);
	ccv_nnc_stream_context_free(stream_context);
	ccv_nnc_tensor_free(hbref);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bref);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(rhs);
	ccv_nnc_tensor_free(lhs);
	ccv_nnc_tensor_free(hwq);
	ccv_nnc_tensor_free(hw_dense);
	ccv_nnc_tensor_free(hrhs_new);
	ccv_nnc_tensor_free(hlhs_new);
	ccv_nnc_tensor_free(hrhs_old);
	ccv_nnc_tensor_free(hlhs_old);
	return 0;
}

static void _mps_forward_scaled_gemm_fill_matrix(const int datatype, void* const data, const int rows, const int cols, const int for_a)
{
	float* const values = (float*)ccmalloc(sizeof(float) * rows * cols);
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			values[i * cols + j] = for_a ? _mps_forward_na_gemm_a_value(i, j) : _mps_forward_na_gemm_b_value(i, j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, rows * cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, rows * cols);
	else
		memcpy(data, values, sizeof(float) * rows * cols);
	ccfree(values);
}

static void _mps_forward_scaled_gemm_fill_bias(const int datatype, void* const data, const int cols)
{
	float* const values = (float*)ccmalloc(sizeof(float) * cols);
	int j;
	for (j = 0; j < cols; j++)
		values[j] = _mps_forward_na_gemm_bias_value(j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, cols);
	else
		memcpy(data, values, sizeof(float) * cols);
	ccfree(values);
}

static void _mps_forward_scaled_gemm_to_float(const int datatype, const void* const data, const int count, float* const values)
{
	if (datatype == CCV_16F)
		ccv_half_precision_to_float((const uint16_t*)data, values, count);
	else if (datatype == CCV_16BF)
		ccv_bfloat_to_float((const uint16_t*)data, values, count);
	else
		memcpy(values, data, sizeof(float) * count);
}

static void _mps_sdpa_store_float_as_datatype(const int datatype, const float* const values, void* const data, const int count)
{
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, count);
	else
		memcpy(data, values, sizeof(float) * count);
}

static void _mps_sdpa_round_to_datatype(const int datatype, const float* const values, float* const rounded, const int count)
{
	void* const data = ccmalloc((size_t)CCV_GET_DATA_TYPE_SIZE(datatype) * count);
	_mps_sdpa_store_float_as_datatype(datatype, values, data, count);
	_mps_forward_scaled_gemm_to_float(datatype, data, count, rounded);
	ccfree(data);
}

static int _mps_dequantize_8i_rowwise_x_compare(const int datatype, const int format, const int rows, const int cols, const double tolerance, double* const max_abs_ref)
{
	ccv_nnc_tensor_param_t host_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { rows, cols, 0 },
	};
	ccv_nnc_tensor_param_t gpu_params = host_params;
	gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, host_params, 0);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_params, 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, gpu_params, 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_params, 0);
	_mps_forward_na_gemm_fill_signed(datatype, source->data.u8, rows, cols, 1);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise_x(source->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)rows * cols, cols, format, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	int status = 0;
	if (qsize != ccv_nnc_tensor_data_size_without_padding(q->info))
	{
		status = -1;
		goto cleanup;
	}
	ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, cols, format, expected->data.u8, (size_t)rows * cols);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
	ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatype, CCV_TENSOR_GPU_MEMORY, qsize, cols, format, gout->data.u8, (size_t)rows * cols);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = (float*)ccmalloc(sizeof(float) * (size_t)rows * cols);
	float* const actual_f32 = (float*)ccmalloc(sizeof(float) * (size_t)rows * cols);
	_mps_forward_scaled_gemm_to_float(datatype, expected->data.u8, rows * cols, expected_f32);
	_mps_forward_scaled_gemm_to_float(datatype, actual->data.u8, rows * cols, actual_f32);
	double max_abs = 0;
	int i;
	for (i = 0; i < rows * cols; i++)
		max_abs = ccv_max(max_abs, fabs((double)expected_f32[i] - (double)actual_f32[i]));
	if (max_abs > tolerance)
		status = 1;
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	ccfree(actual_f32);
	ccfree(expected_f32);
cleanup:
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	return status;
}

static int _mps_sdpa_attention_sinks_compare(const int datatype, const int force_generic, const int gpu_flags, const int B, const int R, const int C, const int Hq, const int Hk, const int D, const int is_causal, const int sink_count, const int sliding_window, const float tolerance, float* const max_abs_ref, float* const max_relative_ref, int* const max_idx_ref, float* const expected_ref, float* const actual_ref)
{
	const int q_count = B * R * Hq * D;
	const int kv_count = B * C * Hk * D;
	const float scale = 1.0 / sqrtf((float)D);
	const ccv_nnc_tensor_param_t q_ref_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = { B, R, Hq, D },
	};
	const ccv_nnc_tensor_param_t kv_ref_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = { B, C, Hk, D },
	};
	const ccv_nnc_tensor_param_t sink_ref_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_32F,
		.dim = { sink_count },
	};
	ccv_nnc_tensor_param_t q_input_params = q_ref_params;
	q_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t kv_input_params = kv_ref_params;
	kv_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t sink_input_params = sink_ref_params;
	sink_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t gpu_q_params = q_input_params;
	gpu_q_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_param_t gpu_kv_params = kv_input_params;
	gpu_kv_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_param_t gpu_sink_params = sink_input_params;
	gpu_sink_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const q_ref = ccv_nnc_tensor_new(0, q_ref_params, 0);
	ccv_nnc_tensor_t* const k_ref = ccv_nnc_tensor_new(0, kv_ref_params, 0);
	ccv_nnc_tensor_t* const v_ref = ccv_nnc_tensor_new(0, kv_ref_params, 0);
	ccv_nnc_tensor_t* const sinks_ref = ccv_nnc_tensor_new(0, sink_ref_params, 0);
	ccv_nnc_tensor_t* const q_input = ccv_nnc_tensor_new(0, q_input_params, 0);
	ccv_nnc_tensor_t* const k_input = ccv_nnc_tensor_new(0, kv_input_params, 0);
	ccv_nnc_tensor_t* const v_input = ccv_nnc_tensor_new(0, kv_input_params, 0);
	ccv_nnc_tensor_t* const sinks_input = ccv_nnc_tensor_new(0, sink_input_params, 0);
	for (int i = 0; i < q_count; ++i)
		q_ref->data.f32[i] = (float)(((i * 17 + R * 5 + is_causal * 11) % 97) - 48) / 256;
	for (int i = 0; i < kv_count; ++i)
	{
		k_ref->data.f32[i] = (float)(((i * 19 + C * 3) % 89) - 44) / 256;
		v_ref->data.f32[i] = (float)(((i * 23 + Hq * 7) % 101) - 50) / 128;
	}
	for (int i = 0; i < sink_count; ++i)
		sinks_ref->data.f32[i] = (float)(((i * 7 + D) % 13) - 6) / 32;
	_mps_sdpa_store_float_as_datatype(datatype, q_ref->data.f32, q_input->data.u8, q_count);
	_mps_sdpa_store_float_as_datatype(datatype, k_ref->data.f32, k_input->data.u8, kv_count);
	_mps_sdpa_store_float_as_datatype(datatype, v_ref->data.f32, v_input->data.u8, kv_count);
	_mps_sdpa_store_float_as_datatype(datatype, sinks_ref->data.f32, sinks_input->data.u8, sink_count);
	_mps_forward_scaled_gemm_to_float(datatype, q_input->data.u8, q_count, q_ref->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, k_input->data.u8, kv_count, k_ref->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, v_input->data.u8, kv_count, v_ref->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, sinks_input->data.u8, sink_count, sinks_ref->data.f32);

	ccv_nnc_tensor_t* const o_ref = ccv_nnc_tensor_new(0, q_ref_params, 0);
	ccv_nnc_cmd_t cpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
	cpu_cmd.info.scaled_dot_product_attention.attention_sinks = 1;
	cpu_cmd.info.scaled_dot_product_attention.sliding_window = sliding_window;
	if (ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_ref, k_ref, v_ref, NULL, NULL, NULL, NULL, NULL, sinks_ref), TENSOR_LIST(o_ref), 0) != CCV_NNC_EXEC_SUCCESS)
		return -1;

	ccv_nnc_tensor_t* const gpu_q = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const gpu_k = ccv_nnc_tensor_new(0, gpu_kv_params, 0);
	ccv_nnc_tensor_t* const gpu_v = ccv_nnc_tensor_new(0, gpu_kv_params, 0);
	ccv_nnc_tensor_t* const gpu_sinks = ccv_nnc_tensor_new(0, gpu_sink_params, 0);
	ccv_nnc_tensor_t* const gpu_o = ccv_nnc_tensor_new(0, gpu_q_params, 0);
	ccv_nnc_tensor_t* const o_actual = ccv_nnc_tensor_new(0, q_input_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input, sinks_input), TENSOR_LIST(gpu_q, gpu_k, gpu_v, gpu_sinks), 0);
	ccv_nnc_cmd_t gpu_cmd = cpu_cmd;
	gpu_cmd.info.scaled_dot_product_attention.flags = gpu_flags;
	const uint64_t old_flags = ccv_nnc_flags();
	if (force_generic)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	const int exec_status = ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q, gpu_k, gpu_v, NULL, NULL, NULL, NULL, NULL, gpu_sinks), TENSOR_LIST(gpu_o), 0);
	if (force_generic && !(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (exec_status != CCV_NNC_EXEC_SUCCESS)
		return -2;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o), TENSOR_LIST(o_actual), 0);

	float* const expected = (float*)ccmalloc(sizeof(float) * q_count);
	float* const actual = (float*)ccmalloc(sizeof(float) * q_count);
	_mps_sdpa_round_to_datatype(datatype, o_ref->data.f32, expected, q_count);
	_mps_forward_scaled_gemm_to_float(datatype, o_actual->data.u8, q_count, actual);
	float max_abs = 0;
	float max_relative = 0;
	int max_idx = 0;
	int status = 0;
	for (int i = 0; i < q_count; ++i)
	{
		if (!isfinite(actual[i]))
		{
			status = -3;
			max_idx = i;
			break;
		}
		const float abs_diff = fabsf(expected[i] - actual[i]);
		const float denom = fmaxf(fmaxf(fabsf(expected[i]), fabsf(actual[i])), 1.0f);
		const float relative = abs_diff / denom;
		if (relative > max_relative)
			max_relative = relative, max_abs = abs_diff, max_idx = i;
	}
	if (status == 0 && max_relative > tolerance)
		status = 1;
	*max_abs_ref = max_abs;
	*max_relative_ref = max_relative;
	*max_idx_ref = max_idx;
	*expected_ref = expected[max_idx];
	*actual_ref = actual[max_idx];

	ccfree(expected);
	ccfree(actual);
	ccv_nnc_tensor_free(o_actual);
	ccv_nnc_tensor_free(gpu_o);
	ccv_nnc_tensor_free(gpu_sinks);
	ccv_nnc_tensor_free(gpu_v);
	ccv_nnc_tensor_free(gpu_k);
	ccv_nnc_tensor_free(gpu_q);
	ccv_nnc_tensor_free(o_ref);
	ccv_nnc_tensor_free(sinks_input);
	ccv_nnc_tensor_free(v_input);
	ccv_nnc_tensor_free(k_input);
	ccv_nnc_tensor_free(q_input);
	ccv_nnc_tensor_free(sinks_ref);
	ccv_nnc_tensor_free(v_ref);
	ccv_nnc_tensor_free(k_ref);
	ccv_nnc_tensor_free(q_ref);
	return status;
}

static void _mps_forward_scaled_gemm_compare_rows(const int datatype, const void* const actual_data, const void* const expected_data, const int rows, const int cols, double* const max_abs_ref, double* const max_rel_ref)
{
	float* const actual_row = (float*)ccmalloc(sizeof(float) * cols);
	float* const expected_row = (float*)ccmalloc(sizeof(float) * cols);
	const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatype);
	const uint8_t* const actual_bytes = (const uint8_t*)actual_data;
	const uint8_t* const expected_bytes = (const uint8_t*)expected_data;
	double max_abs = 0;
	double max_rel = 0;
	int i, j;
	for (i = 0; i < rows; i++)
	{
		_mps_forward_scaled_gemm_to_float(datatype, actual_bytes + (size_t)i * cols * element_size, cols, actual_row);
		_mps_forward_scaled_gemm_to_float(datatype, expected_bytes + (size_t)i * cols * element_size, cols, expected_row);
		for (j = 0; j < cols; j++)
		{
			const double diff = fabs((double)actual_row[j] - (double)expected_row[j]);
			const double denom = ccv_max(1.0, ccv_max(fabs((double)actual_row[j]), fabs((double)expected_row[j])));
			max_abs = ccv_max(max_abs, diff);
			max_rel = ccv_max(max_rel, diff / denom);
		}
	}
	ccfree(expected_row);
	ccfree(actual_row);
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	if (max_rel_ref)
		*max_rel_ref = max_rel;
}

static void _mps_forward_scaled_gemm_quantized_reference(const int datatype, const void* const data, const int rows, const int cols, float* const values)
{
	ccv_nnc_tensor_param_t params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { rows, cols, 0 },
	};
	const ccv_nnc_tensor_param_t qparams = ccv_nnc_tensor_8i_rowwise(params);
	const size_t qsize = ccv_nnc_tensor_data_size_without_padding(qparams);
	uint8_t* const qdata = (uint8_t*)ccmalloc(qsize);
	const size_t encoded = ccv_nnc_quantize_8i_rowwise(data, datatype, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, qdata, qsize);
	void* dequantized = 0;
	if (datatype == CCV_16F || datatype == CCV_16BF)
		dequantized = ccmalloc(sizeof(uint16_t) * rows * cols);
	else
		dequantized = ccmalloc(sizeof(float) * rows * cols);
	ccv_nnc_dequantize_8i_rowwise(qdata, datatype, CCV_TENSOR_CPU_MEMORY, encoded, cols, dequantized, rows * cols);
	_mps_forward_scaled_gemm_to_float(datatype, dequantized, rows * cols, values);
	ccfree(dequantized);
	ccfree(qdata);
}

static void _mps_forward_scaled_gemm_reference(const float* const a, const float* const w, const float* const bias, const int m_dim, const int n_dim, const int k_dim, float* const out)
{
	int i, j, k;
	for (i = 0; i < m_dim; i++)
		for (j = 0; j < n_dim; j++)
		{
			float sum = bias ? bias[j] : 0;
			for (k = 0; k < k_dim; k++)
				sum += a[i * k_dim + k] * w[j * k_dim + k];
			out[i * n_dim + j] = sum;
		}
}

static float _mps_forward_scaled_gemm_a_batched_value(const int batch, const int row, const int k)
{
	return (float)(((batch * 11 + row * 17 + k * 13) % 41) - 20) / 256.0f;
}

static float _mps_forward_scaled_gemm_w_batched_value(const int batch, const int col, const int k)
{
	return (float)(((batch * 7 + col * 19 + k * 5) % 43) - 21) / 256.0f;
}

static float _mps_forward_scaled_gemm_bias_batched_value(const int batch, const int col)
{
	return (float)(((batch * 3 + col * 5) % 23) - 11) / 256.0f;
}

static void _mps_forward_scaled_gemm_fill_matrix_batched(const int datatype, void* const data, const int batch_dim, const int rows, const int cols, const int for_a)
{
	float* const values = (float*)ccmalloc(sizeof(float) * batch_dim * rows * cols);
	int b, i, j;
	for (b = 0; b < batch_dim; b++)
		for (i = 0; i < rows; i++)
			for (j = 0; j < cols; j++)
				values[((b * rows) + i) * cols + j] = for_a ? _mps_forward_scaled_gemm_a_batched_value(b, i, j) : _mps_forward_scaled_gemm_w_batched_value(b, i, j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, batch_dim * rows * cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, batch_dim * rows * cols);
	else
		memcpy(data, values, sizeof(float) * batch_dim * rows * cols);
	ccfree(values);
}

static void _mps_forward_scaled_gemm_fill_bias_batched(const int datatype, void* const data, const int batch_dim, const int cols)
{
	float* const values = (float*)ccmalloc(sizeof(float) * batch_dim * cols);
	int b, j;
	for (b = 0; b < batch_dim; b++)
		for (j = 0; j < cols; j++)
			values[b * cols + j] = _mps_forward_scaled_gemm_bias_batched_value(b, j);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, batch_dim * cols);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, batch_dim * cols);
	else
		memcpy(data, values, sizeof(float) * batch_dim * cols);
	ccfree(values);
}

static void _mps_forward_scaled_gemm_reference_batched(const float* const a, const float* const w, const float* const bias, const int batch_dim, const int w_batch_dim, const int bias_batch_dim, const int m_dim, const int n_dim, const int k_dim, float* const out)
{
	int b, i, j, k;
	for (b = 0; b < batch_dim; b++)
		for (i = 0; i < m_dim; i++)
			for (j = 0; j < n_dim; j++)
			{
				const int w_batch = (w_batch_dim > 1) ? b : 0;
				const int bias_batch = (bias_batch_dim > 1) ? b : 0;
				float sum = bias ? bias[bias_batch * n_dim + j] : 0;
				for (k = 0; k < k_dim; k++)
					sum += a[((b * m_dim) + i) * k_dim + k] * w[((w_batch * n_dim) + j) * k_dim + k];
				out[((b * m_dim) + i) * n_dim + j] = sum;
			}
}

static int _mps_forward_scaled_gemm_validate_shape(const int datatype, const int use_bias, const int m_dim, const int n_dim, const int k_dim, double* const max_abs_ref, double* const max_rel_ref)
{
	ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gw_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(gw_params), 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	_mps_forward_scaled_gemm_fill_matrix(datatype, ha->data.u8, m_dim, k_dim, 1);
	if (use_bias)
		_mps_forward_scaled_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
	void* const w_dense = ccmalloc(CCV_GET_DATA_TYPE_SIZE(datatype) * n_dim * k_dim);
	_mps_forward_scaled_gemm_fill_matrix(datatype, w_dense, n_dim, k_dim, 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(w_dense, datatype, CCV_TENSOR_CPU_MEMORY, n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
	{
		ccfree(w_dense);
		ccv_nnc_tensor_free(ha);
		ccv_nnc_tensor_free(hwq);
		if (hbias)
			ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(w);
		if (bias)
			ccv_nnc_tensor_free(bias);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(hb);
		return -1;
	}
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hbias), TENSOR_LIST(a, w, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq), TENSOR_LIST(a, w), 0);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	else
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);

	float* const a_ref = (float*)ccmalloc(sizeof(float) * m_dim * k_dim);
	float* const w_ref = (float*)ccmalloc(sizeof(float) * n_dim * k_dim);
	float* const bias_ref = use_bias ? (float*)ccmalloc(sizeof(float) * n_dim) : 0;
	float* const actual = (float*)ccmalloc(sizeof(float) * m_dim * n_dim);
	float* const expected = (float*)ccmalloc(sizeof(float) * m_dim * n_dim);
	_mps_forward_scaled_gemm_quantized_reference(datatype, ha->data.u8, m_dim, k_dim, a_ref);
	_mps_forward_scaled_gemm_quantized_reference(datatype, w_dense, n_dim, k_dim, w_ref);
	if (use_bias)
		_mps_forward_scaled_gemm_to_float(datatype, hbias->data.u8, n_dim, bias_ref);
	_mps_forward_scaled_gemm_to_float(datatype, hb->data.u8, m_dim * n_dim, actual);
	_mps_forward_scaled_gemm_reference(a_ref, w_ref, bias_ref, m_dim, n_dim, k_dim, expected);
	double max_abs = 0;
	double max_rel = 0;
	int i;
	for (i = 0; i < m_dim * n_dim; i++)
	{
		const double diff = fabs((double)actual[i] - (double)expected[i]);
		const double denom = ccv_max(1.0, ccv_max(fabs((double)actual[i]), fabs((double)expected[i])));
		max_abs = ccv_max(max_abs, diff);
		max_rel = ccv_max(max_rel, diff / denom);
	}
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	if (max_rel_ref)
		*max_rel_ref = max_rel;

	ccfree(expected);
	ccfree(actual);
	if (bias_ref)
		ccfree(bias_ref);
	ccfree(w_ref);
	ccfree(a_ref);
	ccfree(w_dense);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	return 0;
}

static int _mps_forward_scaled_gemm_validate(const int datatype, const int use_bias, double* const max_abs_ref, double* const max_rel_ref)
{
	return _mps_forward_scaled_gemm_validate_shape(datatype, use_bias, 257, 384, 128, max_abs_ref, max_rel_ref);
}

static int _mps_forward_scaled_gemm_validate_aligned_m(const int datatype, const int use_bias, double* const max_abs_ref, double* const max_rel_ref)
{
	return _mps_forward_scaled_gemm_validate_shape(datatype, use_bias, 384, 384, 128, max_abs_ref, max_rel_ref);
}

static int _mps_forward_scaled_gemm_validate_batched(const int datatype, const int use_bias, const int weight_batched, const int bias_batched, double* const max_abs_ref, double* const max_rel_ref)
{
	const int batch_dim = 2;
	const int m_dim = 129;
	const int n_dim = 384;
	const int k_dim = 128;
	ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gw_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { weight_batched ? batch_dim : n_dim, weight_batched ? n_dim : k_dim, weight_batched ? k_dim : 0, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { bias_batched ? batch_dim : n_dim, bias_batched ? n_dim : 0, 0, 0 },
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t w_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { weight_batched ? batch_dim : n_dim, weight_batched ? n_dim : k_dim, weight_batched ? k_dim : 0, 0 },
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { bias_batched ? batch_dim : n_dim, bias_batched ? n_dim : 0, 0, 0 },
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(w_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(gw_params), 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	_mps_forward_scaled_gemm_fill_matrix_batched(datatype, ha->data.u8, batch_dim, m_dim, k_dim, 1);
	if (use_bias)
	{
		if (bias_batched)
			_mps_forward_scaled_gemm_fill_bias_batched(datatype, hbias->data.u8, batch_dim, n_dim);
		else
			_mps_forward_scaled_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
	}
	const int w_batch_dim = weight_batched ? batch_dim : 1;
	void* const w_dense = ccmalloc(CCV_GET_DATA_TYPE_SIZE(datatype) * w_batch_dim * n_dim * k_dim);
	if (weight_batched)
		_mps_forward_scaled_gemm_fill_matrix_batched(datatype, w_dense, batch_dim, n_dim, k_dim, 0);
	else
		_mps_forward_scaled_gemm_fill_matrix(datatype, w_dense, n_dim, k_dim, 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(w_dense, datatype, CCV_TENSOR_CPU_MEMORY, w_batch_dim * n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
	{
		ccfree(w_dense);
		ccv_nnc_tensor_free(ha);
		ccv_nnc_tensor_free(hwq);
		if (hbias)
			ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(a);
		ccv_nnc_tensor_free(w);
		if (bias)
			ccv_nnc_tensor_free(bias);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(hb);
		return -1;
	}
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hbias), TENSOR_LIST(a, w, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq), TENSOR_LIST(a, w), 0);
	if (weight_batched)
	{
		if (use_bias)
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
		else
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	} else {
		if (use_bias)
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
		else
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);

	float* const a_ref = (float*)ccmalloc(sizeof(float) * batch_dim * m_dim * k_dim);
	float* const w_ref = (float*)ccmalloc(sizeof(float) * w_batch_dim * n_dim * k_dim);
	float* const bias_ref = use_bias ? (float*)ccmalloc(sizeof(float) * (bias_batched ? batch_dim : 1) * n_dim) : 0;
	float* const actual = (float*)ccmalloc(sizeof(float) * batch_dim * m_dim * n_dim);
	float* const expected = (float*)ccmalloc(sizeof(float) * batch_dim * m_dim * n_dim);
	_mps_forward_scaled_gemm_quantized_reference(datatype, ha->data.u8, batch_dim * m_dim, k_dim, a_ref);
	_mps_forward_scaled_gemm_quantized_reference(datatype, w_dense, w_batch_dim * n_dim, k_dim, w_ref);
	if (use_bias)
		_mps_forward_scaled_gemm_to_float(datatype, hbias->data.u8, (bias_batched ? batch_dim : 1) * n_dim, bias_ref);
	_mps_forward_scaled_gemm_to_float(datatype, hb->data.u8, batch_dim * m_dim * n_dim, actual);
	_mps_forward_scaled_gemm_reference_batched(a_ref, w_ref, bias_ref, batch_dim, w_batch_dim, bias_batched ? batch_dim : 1, m_dim, n_dim, k_dim, expected);
	double max_abs = 0;
	double max_rel = 0;
	int i;
	for (i = 0; i < batch_dim * m_dim * n_dim; i++)
	{
		const double diff = fabs((double)actual[i] - (double)expected[i]);
		const double denom = ccv_max(1.0, ccv_max(fabs((double)actual[i]), fabs((double)expected[i])));
		max_abs = ccv_max(max_abs, diff);
		max_rel = ccv_max(max_rel, diff / denom);
	}
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	if (max_rel_ref)
		*max_rel_ref = max_rel;

	ccfree(expected);
	ccfree(actual);
	if (bias_ref)
		ccfree(bias_ref);
	ccfree(w_ref);
	ccfree(a_ref);
	ccfree(w_dense);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(hb);
	return 0;
}

static int _mps_forward_scaled_gemm_compare_dense_format(const int datatype, const int use_bias, const int m_dim, const int n_dim, const int k_dim, const int format, double* const max_abs_ref, double* const max_rel_ref)
{
	ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gwq_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = ((datatype >> 12) & 0xff) | CCV_QX | (format ? CCV_NNC_QX_8I_ROWWISE_X : CCV_NNC_QX_8I_ROWWISE),
		.dim = { n_dim, k_dim, 0 },
		.reserved = format,
	};
	ccv_nnc_tensor_param_t gwd_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t a_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t wd_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hwd = ccv_nnc_tensor_new(0, wd_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, format ? ccv_nnc_tensor_8i_rowwise_x(wd_params, format) : ccv_nnc_tensor_8i_rowwise(wd_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const wq = ccv_nnc_tensor_new(0, gwq_params, 0);
	ccv_nnc_tensor_t* const wd = ccv_nnc_tensor_new(0, gwd_params, 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const bq = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const bd = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const hbq = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_t* const hbd = ccv_nnc_tensor_new(0, b_params, 0);
	_mps_forward_scaled_gemm_fill_matrix(datatype, ha->data.u8, m_dim, k_dim, 1);
	_mps_forward_scaled_gemm_fill_matrix(datatype, hwd->data.u8, n_dim, k_dim, 0);
	if (use_bias)
		_mps_forward_scaled_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
	const size_t qsize = format ? ccv_nnc_quantize_8i_rowwise_x(hwd->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n_dim * k_dim, k_dim, format, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info)) : ccv_nnc_quantize_8i_rowwise(hwd->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
		return -1;
	if (format)
		ccv_nnc_dequantize_8i_rowwise_x(hwq->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, k_dim, format, hwd->data.u8, n_dim * k_dim);
	if (use_bias)
	{
		if (format)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hwd, hbias), TENSOR_LIST(a, wq, wd, bias), 0);
		else
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hbias), TENSOR_LIST(a, wq, bias), 0);
	} else {
		if (format)
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hwd), TENSOR_LIST(a, wq, wd), 0);
		else
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq), TENSOR_LIST(a, wq), 0);
	}
	if (!format)
		ccv_nnc_dequantize_8i_rowwise(wq->data.u8, datatype, CCV_TENSOR_GPU_MEMORY, qsize, k_dim, wd->data.u8, n_dim * k_dim);
	if (use_bias) {
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, wq, bias), TENSOR_LIST(bq), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, wd, bias), TENSOR_LIST(bd), 0);
	} else {
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, wq), TENSOR_LIST(bq), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, wd), TENSOR_LIST(bd), 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(bq, bd), TENSOR_LIST(hbq, hbd), 0);
	_mps_forward_scaled_gemm_compare_rows(datatype, hbq->data.u8, hbd->data.u8, m_dim, n_dim, max_abs_ref, max_rel_ref);
	ccv_nnc_tensor_free(hbq);
	ccv_nnc_tensor_free(hbd);
	ccv_nnc_tensor_free(bq);
	ccv_nnc_tensor_free(bd);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(wq);
	ccv_nnc_tensor_free(wd);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hwd);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	if (bias)
		ccv_nnc_tensor_free(bias);
	return 0;
}

static int _mps_forward_scaled_gemm_compare_dense(const int datatype, const int use_bias, const int m_dim, const int n_dim, const int k_dim, double* const max_abs_ref, double* const max_rel_ref)
{
	return _mps_forward_scaled_gemm_compare_dense_format(datatype, use_bias, m_dim, n_dim, k_dim, 0, max_abs_ref, max_rel_ref);
}

static int _mps_forward_scaled_gemm_compare_dense_batched_padded_a_shape(const int datatype, const int use_bias, const int batch_dim, const int m_dim, const int n_dim, const int k_dim, const int padded_m_dim, double* const max_abs_ref, double* const max_rel_ref)
{
	ccv_nnc_tensor_param_t ga_storage_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, padded_m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t ga_view_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gwq_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = ((datatype >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_param_t ha_storage_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, padded_m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t ha_view_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t wd_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, k_dim, 0 },
	};
	ccv_nnc_tensor_param_t b_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { batch_dim, m_dim, n_dim, 0 },
	};
	ccv_nnc_tensor_param_t bias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { n_dim, 0 },
	};
	ccv_nnc_tensor_t* const ha_storage = ccv_nnc_tensor_new(0, ha_storage_params, 0);
	ccv_nnc_tensor_t* const hwd = ccv_nnc_tensor_new(0, wd_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(wd_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const a_storage = ccv_nnc_tensor_new(0, ga_storage_params, 0);
	ccv_nnc_tensor_t* const wq = ccv_nnc_tensor_new(0, gwq_params, 0);
	ccv_nnc_tensor_t* const bq = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const hbq = ccv_nnc_tensor_new(0, b_params, 0);
	ccv_nnc_tensor_view_t* const ha = ccv_nnc_tensor_view_new(ha_storage, ha_view_params, ccv_nnc_no_ofs, DIM_ALLOC(padded_m_dim * k_dim, k_dim, 1));
	ccv_nnc_tensor_view_t* const a = ccv_nnc_tensor_view_new(a_storage, ga_view_params, ccv_nnc_no_ofs, DIM_ALLOC(padded_m_dim * k_dim, k_dim, 1));
	float* const a_ref = (float*)ccmalloc(sizeof(float) * batch_dim * m_dim * k_dim);
	float* const w_ref = (float*)ccmalloc(sizeof(float) * n_dim * k_dim);
	float* const bias_ref = use_bias ? (float*)ccmalloc(sizeof(float) * n_dim) : 0;
	float* const out_ref = (float*)ccmalloc(sizeof(float) * batch_dim * m_dim * n_dim);
	int bch, i, j;
	for (bch = 0; bch < batch_dim; bch++)
		for (i = 0; i < padded_m_dim; i++)
			for (j = 0; j < k_dim; j++)
			{
				const int dst = ((bch * padded_m_dim) + i) * k_dim + j;
				float value = 0;
				if (i < m_dim)
				{
					value = _mps_forward_scaled_gemm_a_batched_value(bch, i, j);
					a_ref[((bch * m_dim) + i) * k_dim + j] = value;
				}
				if (datatype == CCV_16F)
					ccv_float_to_half_precision(&value, ((uint16_t*)ha_storage->data.u8) + dst, 1);
				else if (datatype == CCV_16BF)
					ccv_float_to_bfloat(&value, ((uint16_t*)ha_storage->data.u8) + dst, 1);
				else
					((float*)ha_storage->data.f32)[dst] = value;
			}
	_mps_forward_scaled_gemm_fill_matrix(datatype, hwd->data.u8, n_dim, k_dim, 0);
	_mps_forward_scaled_gemm_to_float(datatype, hwd->data.u8, n_dim * k_dim, w_ref);
	if (use_bias)
	{
		_mps_forward_scaled_gemm_fill_bias(datatype, hbias->data.u8, n_dim);
		_mps_forward_scaled_gemm_to_float(datatype, hbias->data.u8, n_dim, bias_ref);
	}
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(hwd->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
		return -1;
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_storage, hwq, hbias), TENSOR_LIST(a_storage, wq, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_storage, hwq), TENSOR_LIST(a_storage, wq), 0);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)a, wq, bias), TENSOR_LIST(bq), 0);
	else
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)a, wq), TENSOR_LIST(bq), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(bq), TENSOR_LIST(hbq), 0);
	_mps_forward_scaled_gemm_reference_batched(a_ref, w_ref, bias_ref, batch_dim, 1, use_bias ? 1 : 0, m_dim, n_dim, k_dim, out_ref);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(out_ref, (uint16_t*)ha_storage->data.u8, batch_dim * m_dim * n_dim);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(out_ref, (uint16_t*)ha_storage->data.u8, batch_dim * m_dim * n_dim);
	else
		memcpy(ha_storage->data.f32, out_ref, sizeof(float) * batch_dim * m_dim * n_dim);
	_mps_forward_scaled_gemm_compare_rows(datatype, hbq->data.u8, ha_storage->data.u8, batch_dim * m_dim, n_dim, max_abs_ref, max_rel_ref);
	ccfree(out_ref);
	if (bias_ref)
		ccfree(bias_ref);
	ccfree(w_ref);
	ccfree(a_ref);
	ccv_nnc_tensor_view_free(ha);
	ccv_nnc_tensor_view_free(a);
	ccv_nnc_tensor_free(ha_storage);
	ccv_nnc_tensor_free(hwd);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(a_storage);
	ccv_nnc_tensor_free(wq);
	ccv_nnc_tensor_free(bq);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(hbq);
	return 0;
}

static float _mps_segmented_scaled_gemm_a_value(const int row, const int k)
{
	return (float)(((row * 17 + k * 13) % 61) - 30) / 128.0f;
}

static float _mps_segmented_scaled_gemm_w_value(const int segment, const int col, const int k)
{
	return (float)(((segment * 23 + col * 11 + k * 7) % 67) - 33) / 256.0f;
}

static float _mps_segmented_scaled_gemm_bias_value(const int segment, const int col)
{
	return (float)(((segment * 5 + col * 3) % 29) - 14) / 256.0f;
}

static int _mps_segmented_scaled_gemm_validate_format(const int datatype, const int use_bias, const int force_fallback, const int format, double* const max_abs_ref, double* const max_rel_ref)
{
	const int total_m = 384;
	const int n_dim = 128;
	const int k_dim = 256;
	const int segments = format ? 4 : 3;
	const int counts_data_3[] = {129, 131, 124};
	const int indices_data_3[] = {1, 0, 2};
	const int counts_data_4[] = {129, 0, 131, 124};
	const int indices_data_4[] = {2, 1, 0, 3};
	const int* const counts_data = format ? counts_data_4 : counts_data_3;
	const int* const indices_data = format ? indices_data_4 : indices_data_3;
	const ccv_nnc_tensor_param_t ha_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { total_m, k_dim, 0 },
	};
	const ccv_nnc_tensor_param_t hwd_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { segments, n_dim, k_dim, 0 },
	};
	const ccv_nnc_tensor_param_t hbias_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { segments, n_dim, 0 },
	};
	const ccv_nnc_tensor_param_t ga_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { total_m, k_dim, 0 },
	};
	const ccv_nnc_tensor_param_t gw_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { segments, n_dim, k_dim, 0 },
	};
	const ccv_nnc_tensor_param_t gbias_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { segments, n_dim, 0 },
	};
	const ccv_nnc_tensor_param_t gb_params = {
		.type = CCV_TENSOR_GPU_MEMORY | 000,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { total_m, n_dim, 0 },
	};
	const ccv_nnc_tensor_param_t hb_params = {
		.type = CCV_TENSOR_CPU_MEMORY,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { total_m, n_dim, 0 },
	};
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ha_params, 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, segments), 0);
	ccv_nnc_tensor_t* const hwd = ccv_nnc_tensor_new(0, hwd_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, format ? ccv_nnc_tensor_8i_rowwise_x(hwd_params, format) : ccv_nnc_tensor_8i_rowwise(hwd_params), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, hbias_params, 0) : 0;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, segments), 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, format ? ccv_nnc_tensor_8i_rowwise_x(gw_params, format) : ccv_nnc_tensor_8i_rowwise(gw_params), 0);
	ccv_nnc_tensor_t* const bias = use_bias ? ccv_nnc_tensor_new(0, gbias_params, 0) : 0;
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, gb_params, 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, hb_params, 0);
	float* const a_values = (float*)ccmalloc(sizeof(float) * total_m * k_dim);
	float* const w_values = (float*)ccmalloc(sizeof(float) * segments * n_dim * k_dim);
	float* const bias_values = use_bias ? (float*)ccmalloc(sizeof(float) * segments * n_dim) : 0;
	int i, j, k;
	for (i = 0; i < total_m; i++)
		for (k = 0; k < k_dim; k++)
			a_values[i * k_dim + k] = _mps_segmented_scaled_gemm_a_value(i, k);
	for (i = 0; i < segments; i++)
		for (j = 0; j < n_dim; j++)
			for (k = 0; k < k_dim; k++)
				w_values[((i * n_dim) + j) * k_dim + k] = _mps_segmented_scaled_gemm_w_value(i, j, k);
	if (use_bias)
		for (i = 0; i < segments; i++)
			for (j = 0; j < n_dim; j++)
				bias_values[i * n_dim + j] = _mps_segmented_scaled_gemm_bias_value(i, j);
	if (datatype == CCV_16F)
	{
		ccv_float_to_half_precision(a_values, (uint16_t*)ha->data.u8, total_m * k_dim);
		ccv_float_to_half_precision(w_values, (uint16_t*)hwd->data.u8, segments * n_dim * k_dim);
		if (use_bias)
			ccv_float_to_half_precision(bias_values, (uint16_t*)hbias->data.u8, segments * n_dim);
	} else if (datatype == CCV_16BF) {
		ccv_float_to_bfloat(a_values, (uint16_t*)ha->data.u8, total_m * k_dim);
		ccv_float_to_bfloat(w_values, (uint16_t*)hwd->data.u8, segments * n_dim * k_dim);
		if (use_bias)
			ccv_float_to_bfloat(bias_values, (uint16_t*)hbias->data.u8, segments * n_dim);
	} else {
		memcpy(ha->data.f32, a_values, sizeof(float) * total_m * k_dim);
		memcpy(hwd->data.f32, w_values, sizeof(float) * segments * n_dim * k_dim);
		if (use_bias)
			memcpy(hbias->data.f32, bias_values, sizeof(float) * segments * n_dim);
	}
	memcpy(hindices->data.i32, indices_data, sizeof(int) * segments);
	memcpy(hcounts->data.i32, counts_data, sizeof(int) * segments);
	const size_t qsize = format ? ccv_nnc_quantize_8i_rowwise_x(hwd->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)segments * n_dim * k_dim, k_dim, format, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info)) : ccv_nnc_quantize_8i_rowwise(hwd->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)segments * n_dim * k_dim, k_dim, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	if (qsize != ccv_nnc_tensor_data_size_without_padding(hwq->info))
		return -1;
	if (format)
		ccv_nnc_dequantize_8i_rowwise_x(hwq->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, k_dim, format, hwd->data.u8, (size_t)segments * n_dim * k_dim);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hwq, hbias), TENSOR_LIST(a, indices, counts, w, bias), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hwq), TENSOR_LIST(a, indices, counts, w), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	if (force_fallback)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	else
		ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	if (force_fallback && !(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);

	float* const a_ref = (float*)ccmalloc(sizeof(float) * total_m * k_dim);
	float* const w_ref = (float*)ccmalloc(sizeof(float) * segments * n_dim * k_dim);
	float* const bias_ref = use_bias ? (float*)ccmalloc(sizeof(float) * segments * n_dim) : 0;
	float* const actual = (float*)ccmalloc(sizeof(float) * total_m * n_dim);
	ccv_nnc_tensor_t* const ha_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, total_m, k_dim), 0);
	ccv_nnc_tensor_t* const hw_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, segments, n_dim, k_dim), 0);
	ccv_nnc_tensor_t* const hbias_ref = use_bias ? ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, segments, n_dim), 0) : 0;
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, total_m, n_dim), 0);
	if (force_fallback)
		_mps_forward_scaled_gemm_to_float(datatype, ha->data.u8, total_m * k_dim, a_ref);
	else
		_mps_forward_scaled_gemm_quantized_reference(datatype, ha->data.u8, total_m, k_dim, a_ref);
	if (format)
		_mps_forward_scaled_gemm_to_float(datatype, hwd->data.u8, segments * n_dim * k_dim, w_ref);
	else
		_mps_forward_scaled_gemm_quantized_reference(datatype, hwd->data.u8, segments * n_dim, k_dim, w_ref);
	if (use_bias)
		_mps_forward_scaled_gemm_to_float(datatype, hbias->data.u8, segments * n_dim, bias_ref);
	_mps_forward_scaled_gemm_to_float(datatype, hb->data.u8, total_m * n_dim, actual);
	memcpy(ha_ref->data.f32, a_ref, sizeof(float) * total_m * k_dim);
	memcpy(hw_ref->data.f32, w_ref, sizeof(float) * segments * n_dim * k_dim);
	if (use_bias)
		memcpy(hbias_ref->data.f32, bias_ref, sizeof(float) * segments * n_dim);
	if (use_bias)
		ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_ref, hindices, hcounts, hw_ref, hbias_ref), TENSOR_LIST(bt), 0);
	else
		ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha_ref, hindices, hcounts, hw_ref), TENSOR_LIST(bt), 0);
	double max_abs = 0;
	double max_rel = 0;
	for (i = 0; i < total_m * n_dim; i++)
	{
		const double diff = fabs((double)actual[i] - (double)bt->data.f32[i]);
		const double denom = ccv_max(1.0, ccv_max(fabs((double)actual[i]), fabs((double)bt->data.f32[i])));
		max_abs = ccv_max(max_abs, diff);
		max_rel = ccv_max(max_rel, diff / denom);
	}
	if (max_abs_ref)
		*max_abs_ref = max_abs;
	if (max_rel_ref)
		*max_rel_ref = max_rel;
	ccv_nnc_tensor_free(bt);
	if (hbias_ref)
		ccv_nnc_tensor_free(hbias_ref);
	ccv_nnc_tensor_free(hw_ref);
	ccv_nnc_tensor_free(ha_ref);
	ccfree(actual);
	if (bias_ref)
		ccfree(bias_ref);
	ccfree(w_ref);
	ccfree(a_ref);
	ccfree(a_values);
	ccfree(w_values);
	if (bias_values)
		ccfree(bias_values);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(b);
	if (bias)
		ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hwq);
	ccv_nnc_tensor_free(hwd);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(ha);
	return 0;
}

static int _mps_segmented_scaled_gemm_validate(const int datatype, const int use_bias, const int force_fallback, double* const max_abs_ref, double* const max_rel_ref)
{
	return _mps_segmented_scaled_gemm_validate_format(datatype, use_bias, force_fallback, 0, max_abs_ref, max_rel_ref);
}

TEST_CASE("mps forward gemm with row-wise 8i weight NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_16F, 0, &max_abs, &max_rel), 0, "scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul should match row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_32F, 0, &max_abs, &max_rel), 0, "scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul should match row-wise quantized fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_16BF, 0, &max_abs, &max_rel), 0, "scaled GEMM validation should run");
	REQUIRE(max_rel < 5e-3, "quantized NAInt8MatMul should match row-wise quantized bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight and bias NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_16F, 1, &max_abs, &max_rel), 0, "scaled GEMM validation with bias should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul with bias should match row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_32F, 1, &max_abs, &max_rel), 0, "scaled GEMM validation with bias should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul with bias should match row-wise quantized fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate(CCV_16BF, 1, &max_abs, &max_rel), 0, "scaled GEMM validation with bias should run");
	REQUIRE(max_rel < 5e-3, "quantized NAInt8MatMul with bias should match row-wise quantized bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight NA aligned M")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_16F, 0, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul should match aligned-M row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_32F, 0, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul should match aligned-M row-wise quantized fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_16BF, 0, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation should run");
	REQUIRE(max_rel < 5e-3, "quantized NAInt8MatMul should match aligned-M row-wise quantized bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight and bias NA aligned M")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_16F, 1, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation with bias should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul with bias should match aligned-M row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_32F, 1, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation with bias should run");
	REQUIRE(max_rel < 2e-3, "quantized NAInt8MatMul with bias should match aligned-M row-wise quantized fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_aligned_m(CCV_16BF, 1, &max_abs, &max_rel), 0, "scaled GEMM aligned-M validation with bias should run");
	REQUIRE(max_rel < 5e-3, "quantized NAInt8MatMul with bias should match aligned-M row-wise quantized bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight NA small M")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
	double max_abs16f = 0;
	double max_rel16f = 0;
	const int status16f = _mps_forward_scaled_gemm_validate_shape(CCV_16F, 0, 16, 384, 1024, &max_abs16f, &max_rel16f);
	double max_abs16f_bias = 0;
	double max_rel16f_bias = 0;
	const int status16f_bias = _mps_forward_scaled_gemm_validate_shape(CCV_16F, 1, 6, 384, 1536, &max_abs16f_bias, &max_rel16f_bias);
	double max_abs16bf = 0;
	double max_rel16bf = 0;
	const int status16bf = _mps_forward_scaled_gemm_validate_shape(CCV_16BF, 0, 8, 384, 1024, &max_abs16bf, &max_rel16bf);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
	}
	REQUIRE_EQ(status16f, 0, "small-M scaled GEMM validation should run");
	REQUIRE(max_rel16f < 2e-3, "small-M NAInt8MatMul should match row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs16f, max_rel16f);
	REQUIRE_EQ(status16f_bias, 0, "small-M scaled GEMM validation with bias should run");
	REQUIRE(max_rel16f_bias < 2e-3, "small-M NAInt8MatMul with bias should match row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs16f_bias, max_rel16f_bias);
	REQUIRE_EQ(status16bf, 0, "small-M scaled GEMM bf16 validation should run");
	REQUIRE(max_rel16bf < 8e-3, "small-M NAInt8MatMul should match row-wise quantized bf16 reference, max_abs=%g max_rel=%g", max_abs16bf, max_rel16bf);
}

TEST_CASE("mps forward gemm with row-wise 8i weight ANE stream ordering")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_ane_rowwise_gemm_stream_sync_validate(&max_abs, &max_rel), 0, "ANE row-wise 8i stream-ordering validation should run");
	REQUIRE(max_rel < 2e-3, "ANE row-wise 8i GEMM should respect queued Metal writer work before quant/evaluate, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with loadM flag")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	double max_abs_scaled = 0;
	double max_rel_scaled = 0;
	const int status_scaled = _mps_forward_scaled_gemm_validate_shape(CCV_16F, 0, 257, 384, 128, &max_abs_scaled, &max_rel_scaled);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	double max_abs_generic = 0;
	double max_rel_generic = 0;
	const int status_generic = _mps_forward_scaled_gemm_compare_dense(CCV_16F, 0, 257, 384, 128, &max_abs_generic, &max_rel_generic);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
	REQUIRE_EQ(status_scaled, 0, "loadM scaled GEMM validation should run");
	REQUIRE(max_rel_scaled < 2e-3, "loadM scaled GEMM should match row-wise quantized fp16 reference, max_abs=%g max_rel=%g", max_abs_scaled, max_rel_scaled);
	REQUIRE_EQ(status_generic, 0, "loadM generic GEMM fallback validation should run");
	REQUIRE(max_rel_generic < 2e-3, "loadM generic GEMM fallback should match dense GPU fp16 reference, max_abs=%g max_rel=%g", max_abs_generic, max_rel_generic);
}

TEST_CASE("mps segmented gemm with row-wise 8i weight NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_16F, 0, 0, &max_abs, &max_rel), 0, "segmented row-wise 8i NA validation should run");
	REQUIRE(max_rel < 3e-3, "segmented row-wise 8i NA fp16 should match quantized reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_32F, 0, 0, &max_abs, &max_rel), 0, "segmented row-wise 8i NA validation should run");
	REQUIRE(max_rel < 3e-3, "segmented row-wise 8i NA fp32 should match quantized reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_16BF, 0, 0, &max_abs, &max_rel), 0, "segmented row-wise 8i NA validation should run");
	REQUIRE(max_rel < 6e-3, "segmented row-wise 8i NA bf16 should match quantized reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps segmented gemm with row-wise 8i-x weight NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	int i;
	for (i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
	{
		double max_abs = 0;
		double max_rel = 0;
		REQUIRE_EQ(_mps_segmented_scaled_gemm_validate_format(CCV_16F, 0, 0, formats[i], &max_abs, &max_rel), 0, "segmented row-wise 8i-x NA validation should run for format=%d", formats[i]);
		REQUIRE(max_rel < 3e-3, "segmented row-wise 8i-x fp16 should match selected-decode quantized reference for format=%d, max_abs=%g max_rel=%g", formats[i], max_abs, max_rel);
	}
}

TEST_CASE("mps segmented gemm with row-wise 8i weight and bias fallback dequantize")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_16F, 1, 1, &max_abs, &max_rel), 0, "segmented fallback row-wise 8i validation should run");
	REQUIRE(max_rel < 3e-3, "segmented fallback row-wise 8i fp16 should match dense-A reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_32F, 1, 1, &max_abs, &max_rel), 0, "segmented fallback row-wise 8i validation should run");
	REQUIRE(max_rel < 3e-3, "segmented fallback row-wise 8i fp32 should match dense-A reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_segmented_scaled_gemm_validate(CCV_16BF, 1, 1, &max_abs, &max_rel), 0, "segmented fallback row-wise 8i validation should run");
	REQUIRE(max_rel < 6e-3, "segmented fallback row-wise 8i bf16 should match dense-A reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight fallback dequantize")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	double max_abs = 0;
	double max_rel = 0;
	const int status16f = _mps_forward_scaled_gemm_compare_dense(CCV_16F, 0, 257, 384, 128, &max_abs, &max_rel);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE_EQ(status16f, 0, "fallback row-wise 8i GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "fallback row-wise 8i GEMM should match dense GPU fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);

	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	max_abs = 0;
	max_rel = 0;
	const int status32f = _mps_forward_scaled_gemm_compare_dense(CCV_32F, 0, 257, 384, 128, &max_abs, &max_rel);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE_EQ(status32f, 0, "fallback row-wise 8i GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "fallback row-wise 8i GEMM should match dense GPU fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);

	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	max_abs = 0;
	max_rel = 0;
	const int status16bf = _mps_forward_scaled_gemm_compare_dense(CCV_16BF, 0, 257, 384, 128, &max_abs, &max_rel);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE_EQ(status16bf, 0, "fallback row-wise 8i GEMM validation should run");
	REQUIRE(max_rel < 5e-3, "fallback row-wise 8i GEMM should match dense GPU bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with packed row-wise 8i-x weight fallback fp dequantize")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	};
	int status[sizeof(formats) / sizeof(formats[0])];
	double max_abs[sizeof(formats) / sizeof(formats[0])];
	double max_rel[sizeof(formats) / sizeof(formats[0])];
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	int i;
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		max_abs[i] = 0;
		max_rel[i] = 0;
		status[i] = _mps_forward_scaled_gemm_compare_dense_format(CCV_16F, 0, 257, 384, 128, formats[i], &max_abs[i], &max_rel[i]);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		REQUIRE_EQ(status[i], 0, "fallback packed row-wise 8i-x GEMM validation should run for format=%d", formats[i]);
		REQUIRE(max_rel[i] < 2e-3, "fallback packed row-wise 8i-x GEMM should match GPU fp16 reference for format=%d, max_abs=%g max_rel=%g", formats[i], max_abs[i], max_rel[i]);
	}
}

TEST_CASE("mps forward gemv with row-wise 8i weight scaled gemv")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 0, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV fp16 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV fp16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 0, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV fp32 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV fp32 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 0, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV bf16 validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV bf16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 0, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 fp16 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=2 fp16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 0, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 fp32 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=2 fp32 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 0, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 bf16 validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV M=2 bf16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 0, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 fp16 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=3 fp16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 0, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 fp32 validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=3 fp32 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 0, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 bf16 validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV M=3 bf16 should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemv with row-wise 8i weight and bias scaled gemv")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 1, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV fp16 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV fp16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 1, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV fp32 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV fp32 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 1, 1, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV bf16 with bias validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV bf16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 1, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 fp16 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=2 fp16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 1, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 fp32 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=2 fp32 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 1, 2, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=2 bf16 with bias validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV M=2 bf16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 1, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 fp16 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=3 fp16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 1, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 fp32 with bias validation should run");
	REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "row-wise 8i GEMV M=3 fp32 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 1, 3, 1024, 2560, &max_abs, &max_rel), 0, "row-wise 8i GEMV M=3 bf16 with bias validation should run");
	REQUIRE(max_abs < 4e-2 || max_rel < 1e-2, "row-wise 8i GEMV M=3 bf16 with bias should match dense GPU reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemv with packed row-wise 8i weight scaled gemv")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
	};
	int i;
	for (i = 0; i < sizeof(formats) / sizeof(formats[0]); i++)
	{
		double max_abs = 0;
		double max_rel = 0;
		REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_format(CCV_16F, 0, 1, 256, 256, formats[i], &max_abs, &max_rel), 0, "packed row-wise 8i GEMV fp16 validation should run for format=%d", formats[i]);
		REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "packed row-wise 8i GEMV fp16 should match dequantized dense GPU reference for format=%d, max_abs=%g max_rel=%g", formats[i], max_abs, max_rel);
		max_abs = 0;
		max_rel = 0;
		REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_format(CCV_16F, 1, 2, 256, 256, formats[i], &max_abs, &max_rel), 0, "packed row-wise 8i GEMV M=2 fp16 with bias validation should run for format=%d", formats[i]);
		REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "packed row-wise 8i GEMV M=2 fp16 with bias should match dequantized dense GPU reference for format=%d, max_abs=%g max_rel=%g", formats[i], max_abs, max_rel);
		max_abs = 0;
		max_rel = 0;
		REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_format(CCV_16F, 1, 3, 256, 256, formats[i], &max_abs, &max_rel), 0, "packed row-wise 8i GEMV M=3 fp16 with bias validation should run for format=%d", formats[i]);
		REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "packed row-wise 8i GEMV M=3 fp16 with bias should match dequantized dense GPU reference for format=%d, max_abs=%g max_rel=%g", formats[i], max_abs, max_rel);
	}
}

TEST_CASE("mps forward gemm with row-wise 8i weight and bias fallback dequantize")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	double max_abs = 0;
	double max_rel = 0;
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16F, 1, 257, 384, 128, &max_abs, &max_rel), 0, "fallback row-wise 8i GEMM with bias validation should run");
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE(max_rel < 2e-3, "fallback row-wise 8i GEMM with bias should match dense GPU fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_32F, 1, 257, 384, 128, &max_abs, &max_rel), 0, "fallback row-wise 8i GEMM with bias validation should run");
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE(max_rel < 2e-3, "fallback row-wise 8i GEMM with bias should match dense GPU fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 1, 257, 384, 128, &max_abs, &max_rel), 0, "fallback row-wise 8i GEMM with bias validation should run");
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	REQUIRE(max_rel < 5e-3, "fallback row-wise 8i GEMM with bias should match dense GPU bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward gemm with row-wise 8i weight and bias fallback dequantize large shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	static const int shapes[][3] = {
		{32, 3840, 3840},
		{32, 10240, 3840},
		{32, 3840, 10240},
	};
	int i;
	for (i = 0; i < (int)(sizeof(shapes) / sizeof(shapes[0])); i++)
	{
		double max_abs = 0;
		double max_rel = 0;
		REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense(CCV_16BF, 1, shapes[i][0], shapes[i][1], shapes[i][2], &max_abs, &max_rel), 0, "large fallback row-wise 8i GEMM with bias validation should run");
		REQUIRE(max_abs < 2e-2 || max_rel < 5e-3, "large fallback row-wise 8i GEMM with bias should match dense GPU bf16 reference for shape %d x %d x %d, max_abs=%g max_rel=%g", shapes[i][0], shapes[i][1], shapes[i][2], max_abs, max_rel);
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)) {
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
}

TEST_CASE("mps forward batched gemm with broadcast row-wise 8i weight NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_16F, 0, 0, 0, &max_abs, &max_rel), 0, "batched scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "batched quantized NAInt8MatMul should match broadcast-weight fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_32F, 0, 0, 0, &max_abs, &max_rel), 0, "batched scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "batched quantized NAInt8MatMul should match broadcast-weight fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_16BF, 0, 0, 0, &max_abs, &max_rel), 0, "batched scaled GEMM validation should run");
	REQUIRE(max_rel < 5e-3, "batched quantized NAInt8MatMul should match broadcast-weight bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward batched gemm with batched row-wise 8i weight and bias NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_16F, 1, 1, 1, &max_abs, &max_rel), 0, "batched scaled GEMM validation with batched weight and bias should run");
	REQUIRE(max_rel < 2e-3, "batched quantized NAInt8MatMul should match batched-weight fp16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_32F, 1, 1, 1, &max_abs, &max_rel), 0, "batched scaled GEMM validation with batched weight and bias should run");
	REQUIRE(max_rel < 2e-3, "batched quantized NAInt8MatMul should match batched-weight fp32 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_validate_batched(CCV_16BF, 1, 1, 1, &max_abs, &max_rel), 0, "batched scaled GEMM validation with batched weight and bias should run");
	REQUIRE(max_rel < 5e-3, "batched quantized NAInt8MatMul should match batched-weight bf16 reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

TEST_CASE("mps forward batched gemm with padded A view and broadcast row-wise 8i weight NA")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	double max_abs = 0;
	double max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_batched_padded_a_shape(CCV_16F, 0, 1, 512, 3072, 3072, 513, &max_abs, &max_rel), 0, "single-batch padded-A scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "single-batch padded-A scaled GEMM without bias should match dense reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_batched_padded_a_shape(CCV_16F, 0, 2, 512, 3072, 3072, 513, &max_abs, &max_rel), 0, "batched padded-A scaled GEMM validation should run");
	REQUIRE(max_rel < 2e-3, "batched padded-A scaled GEMM without bias should match dense reference, max_abs=%g max_rel=%g", max_abs, max_rel);
	max_abs = 0;
	max_rel = 0;
	REQUIRE_EQ(_mps_forward_scaled_gemm_compare_dense_batched_padded_a_shape(CCV_16F, 1, 2, 512, 3072, 3072, 513, &max_abs, &max_rel), 0, "batched padded-A scaled GEMM with bias validation should run");
	REQUIRE(max_rel < 2e-3, "batched padded-A scaled GEMM with bias should match dense reference, max_abs=%g max_rel=%g", max_abs, max_rel);
}

#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)
#define NA_GEMM_SHAPE_TEST(M, N, K) \
	TEST_CASE("mps forward gemm no bias NA shape " STRINGIFY(M) "x" STRINGIFY(N) "x" STRINGIFY(K)) \
	{ \
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)); \
		_mps_forward_na_gemm_mismatch_t mismatch = {}; \
		REQUIRE(_mps_forward_na_gemm_validate_shape(M, N, K, &mismatch), "sampled GEMM result should match reference for shape (%d, %d, %d) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", M, N, K, mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel); \
	}

#define NA_GEMM_BIAS_SHAPE_TEST(M, N, K) \
	TEST_CASE("mps forward gemm with bias NA shape " STRINGIFY(M) "x" STRINGIFY(N) "x" STRINGIFY(K)) \
	{ \
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)); \
		_mps_forward_na_gemm_mismatch_t mismatch = {}; \
		REQUIRE(_mps_forward_na_gemm_validate_shape_with_bias(M, N, K, &mismatch), "sampled GEMM result with bias should match reference for shape (%d, %d, %d) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", M, N, K, mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel); \
	}

#define NA_GEMM_BFLOAT_SHAPE_TEST(M, N, K) \
	TEST_CASE("mps forward gemm no bias bfloat NA shape " STRINGIFY(M) "x" STRINGIFY(N) "x" STRINGIFY(K)) \
	{ \
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)); \
		_mps_forward_na_gemm_mismatch_t mismatch = {}; \
		REQUIRE(_mps_forward_na_gemm_validate_shape_for_datatype(CCV_16BF, 0, M, N, K, &mismatch), "sampled bfloat GEMM result should match reference for shape (%d, %d, %d) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", M, N, K, mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel); \
	}

#define NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(M, N, K) \
	TEST_CASE("mps forward gemm with bias bfloat NA shape " STRINGIFY(M) "x" STRINGIFY(N) "x" STRINGIFY(K)) \
	{ \
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS)); \
		_mps_forward_na_gemm_mismatch_t mismatch = {}; \
		REQUIRE(_mps_forward_na_gemm_validate_shape_for_datatype(CCV_16BF, 1, M, N, K, &mismatch), "sampled bfloat GEMM result with bias should match reference for shape (%d, %d, %d) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", M, N, K, mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel); \
	}

TEST_CASE("mps forward gemm no bias NA full shape 6x1024x3072")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	_mps_forward_na_gemm_mismatch_t mismatch = {};
	REQUIRE(_mps_forward_na_gemm_validate_full_shape_for_datatype(CCV_16F, 0, 0, 6, 1024, 3072, &mismatch), "full GEMM result should match reference for shape (6, 1024, 3072) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel);
}

TEST_CASE("mps forward gemm no bias NA full signed shape 6x1024x3072")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	_mps_forward_na_gemm_mismatch_t mismatch = {};
	REQUIRE(_mps_forward_na_gemm_validate_full_shape_for_datatype(CCV_16F, 0, 1, 6, 1024, 3072, &mismatch), "full signed GEMM result should match reference for shape (6, 1024, 3072) at (%d, %d): %g vs %g, max_abs=%g max_rel=%g", mismatch.row, mismatch.col, mismatch.actual, mismatch.expected, mismatch.max_abs, mismatch.max_rel);
}

TEST_CASE("gemm no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm no transpose with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
		1, -1, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 + 1, 1 * 8 + 2 * 11 - 1, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 + 1, 3 * 8 + 4 * 11 - 1, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 + 1, 5 * 8 + 6 * 11 - 1, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 + 1, 7 * 8 + 8 * 11 - 1, 7 * 9 + 8 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm no transpose batch 2, no batch b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 7 + 3 * 10, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12,
		4 * 7 + 5 * 10, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12,
		6 * 7 + 7 * 10, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12,
		8 * 7 + 9 * 10, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm no transpose batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
		8, 9, 10,
		11, 12, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 8 + 3 * 11, 2 * 9 + 3 * 12, 2 * 10 + 3 * 13,
		4 * 8 + 5 * 11, 4 * 9 + 5 * 12, 4 * 10 + 5 * 13,
		6 * 8 + 7 * 11, 6 * 9 + 7 * 12, 6 * 10 + 7 * 13,
		8 * 8 + 9 * 11, 8 * 9 + 9 * 12, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose a batch 2, no batch b, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
		2, 4, 6, 8,
		3, 5, 7, 9,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	float dp[] = {
		-1, 0, 1,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 7 + 3 * 10 - 1, 2 * 8 + 3 * 11, 2 * 9 + 3 * 12 + 1,
		4 * 7 + 5 * 10 - 1, 4 * 8 + 5 * 11, 4 * 9 + 5 * 12 + 1,
		6 * 7 + 7 * 10 - 1, 6 * 8 + 7 * 11, 6 * 9 + 7 * 12 + 1,
		8 * 7 + 9 * 10 - 1, 8 * 8 + 9 * 11, 8 * 9 + 9 * 12 + 1,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm transpose a batch 2, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 3, 5, 7,
		2, 4, 6, 8,
		2, 4, 6, 8,
		3, 5, 7, 9,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		7, 8, 9,
		10, 11, 12,
		8, 9, 10,
		11, 12, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 8 + 3 * 11, 2 * 9 + 3 * 12, 2 * 10 + 3 * 13,
		4 * 8 + 5 * 11, 4 * 9 + 5 * 12, 4 * 10 + 5 * 13,
		6 * 8 + 7 * 11, 6 * 9 + 7 * 12, 6 * 10 + 7 * 13,
		8 * 8 + 9 * 11, 8 * 9 + 9 * 12, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("gemm transpose b batch 2, with bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
		80, 110,
		90, 120,
		10, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	float dp[] = {
		-1, 0, 1,
		2, 3, -4,
	};
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(dp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* gd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b, d), TENSOR_LIST(ga, gb, gd), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb, gd), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10 - 1, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12 + 1,
		3 * 7 + 4 * 10 - 1, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12 + 1,
		5 * 7 + 6 * 10 - 1, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12 + 1,
		7 * 7 + 8 * 10 - 1, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12 + 1,
		2 * 80 + 3 * 110 + 2, 2 * 90 + 3 * 120 + 3, 2 * 10 + 3 * 13 - 4,
		4 * 80 + 5 * 110 + 2, 4 * 90 + 5 * 120 + 3, 4 * 10 + 5 * 13 - 4,
		6 * 80 + 7 * 110 + 2, 6 * 90 + 7 * 120 + 3, 6 * 10 + 7 * 13 - 4,
		8 * 80 + 9 * 110 + 2, 8 * 90 + 9 * 120 + 3, 8 * 10 + 9 * 13 - 4,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
	ccv_nnc_tensor_free(gd);
}

TEST_CASE("gemm transpose b batch 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	float ap[] = {
		1, 2,
		3, 4,
		5, 6,
		7, 8,
		2, 3,
		4, 5,
		6, 7,
		8, 9
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		7, 10,
		8, 11,
		9, 12,
		80, 110,
		90, 120,
		10, 13,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* gc = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gb), TENSOR_LIST(gc), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gc), TENSOR_LIST(c), 0);
	float ctp[] = {
		1 * 7 + 2 * 10, 1 * 8 + 2 * 11, 1 * 9 + 2 * 12,
		3 * 7 + 4 * 10, 3 * 8 + 4 * 11, 3 * 9 + 4 * 12,
		5 * 7 + 6 * 10, 5 * 8 + 6 * 11, 5 * 9 + 6 * 12,
		7 * 7 + 8 * 10, 7 * 8 + 8 * 11, 7 * 9 + 8 * 12,
		2 * 80 + 3 * 110, 2 * 90 + 3 * 120, 2 * 10 + 3 * 13,
		4 * 80 + 5 * 110, 4 * 90 + 5 * 120, 4 * 10 + 5 * 13,
		6 * 80 + 7 * 110, 6 * 90 + 7 * 120, 6 * 10 + 7 * 13,
		8 * 80 + 9 * 110, 8 * 90 + 9 * 120, 8 * 10 + 9 * 13,
	};
	ccv_nnc_tensor_t ct = ccv_nnc_tensor(ctp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	REQUIRE_TENSOR_EQ(c, &ct, "result should be equal");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gc);
}

TEST_CASE("mps forward gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 5e-6, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps forward gemm in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 5e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps forward gemm in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 8e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps forward gemv in half precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps forward gemv in half precision with M=2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	for (i = 0; i < 2 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 2 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps forward gemv in bfloat precision, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_tensor_t* hbias2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw, hbias), TENSOR_LIST(ha2, hw2, hbias2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2, hbias2), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 8e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
	ccv_nnc_tensor_free(hbias2);
}

TEST_CASE("mps depalettize 5-bit half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	float lut_f32[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	uint16_t lut[32];
	ccv_float_to_half_precision(lut_f32, lut, 32);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 3072);
	int i;
	for (i = 0; i < 3072; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2112 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 3072, 5, 1024, compressed, 2112);
	REQUIRE_EQ(output_size, 2112, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2112 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 3072), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 1024, gv_tensor->data.u8, 3072);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 3072), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 3072, "GPU computed output should match CPU depalettize");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("mps depalettize 6-bit float precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	float lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 8192);
	for (i = 0; i < 8192; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (6144 + 2 * 64 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 8192, 6, 4096, compressed, 6144 + 2 * 64 * 4);
	REQUIRE_EQ(output_size, 6144 + 2 * 64 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (6144 + 2 * 64 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 8192), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 4096, gv_tensor->data.u8, 8192);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8192), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 8192, "GPU computed output should match CPU depalettize");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("mps depalettize 8-bit float precision with partial block")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	float lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 3072);
	for (i = 0; i < 3072; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (6144 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 3072, 8, 2048, compressed, 6144);
	REQUIRE(output_size <= 6144, "output size should fit the allocated buffer");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (6144 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 3072), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 2048, gv_tensor->data.u8, 3072);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 3072), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 3072, "GPU computed output should match CPU depalettize");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("mps dequantize row-wise 8i half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 17;
	const int cols = 64;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 13) % 41 - 20) / 32.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_float_to_half_precision(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, qsize, cols, expected->data.f16, rows * cols);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
	ccv_nnc_dequantize_8i_rowwise(gq->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, qsize, cols, gout->data.u8, rows * cols);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = (float*)ccmalloc(sizeof(float) * rows * cols);
	float* const actual_f32 = (float*)ccmalloc(sizeof(float) * rows * cols);
	ccv_half_precision_to_float((uint16_t*)expected->data.f16, expected_f32, rows * cols);
	ccv_half_precision_to_float((uint16_t*)actual->data.f16, actual_f32, rows * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, rows * cols, 1e-3, "GPU row-wise 8i dequantize should match CPU dequantize");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("mps dequantize row-wise 8i float precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 11;
	const int cols = 128;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = ((i * 17) % 53 - 26) / 64.0f;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, cols, expected->data.f32, rows * cols);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, rows, cols)), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
	ccv_nnc_dequantize_8i_rowwise(gq->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, qsize, cols, gout->data.u8, rows * cols);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	REQUIRE_ARRAY_EQ(float, expected->data.f32, actual->data.f32, rows * cols, "GPU row-wise 8i dequantize should match CPU dequantize");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("mps dequantize row-wise 8i bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 257;
	const int cols = 130;
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	int i;
	for (i = 0; i < rows * cols; i++)
		values[i] = ((i * 29) % 97 - 48) / 64.0f;
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, rows * cols);
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16BF, rows, cols)), 0);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16BF, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16BF, CCV_TENSOR_CPU_MEMORY, qsize, cols, expected->data.f16, rows * cols);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16BF, rows, cols)), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
	ccv_nnc_dequantize_8i_rowwise(gq->data.u8, CCV_16BF, CCV_TENSOR_GPU_MEMORY, qsize, cols, gout->data.u8, rows * cols);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
	float* const expected_f32 = (float*)ccmalloc(sizeof(float) * rows * cols);
	float* const actual_f32 = (float*)ccmalloc(sizeof(float) * rows * cols);
	ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, rows * cols);
	ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, rows * cols);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, rows * cols, 5e-3, "GPU row-wise 8i bf16 dequantize should match CPU dequantize");
	ccfree(actual_f32);
	ccfree(expected_f32);
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(source);
	ccfree(values);
}

TEST_CASE("mps dequantize row-wise 8i bfloat precision large shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int shapes[][2] = {
		{3840, 3840},
		{10240, 3840},
		{3840, 10240},
	};
	int s;
	for (s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
	{
		const int rows = shapes[s][0];
		const int cols = shapes[s][1];
		float* const values = ccmalloc(sizeof(float) * (size_t)rows * cols);
		int i;
		for (i = 0; i < rows * cols; i++)
			values[i] = ((i * 29) % 97 - 48) / 64.0f;
		ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
		ccv_float_to_bfloat(values, (uint16_t*)source->data.f16, rows * cols);
		ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16BF, rows, cols)), 0);
		const size_t qsize = ccv_nnc_quantize_8i_rowwise(source->data.f16, CCV_16BF, CCV_TENSOR_CPU_MEMORY, (size_t)rows * cols, cols, 0, 0, q->data.u8, ccv_nnc_tensor_data_size_without_padding(q->info));
		REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(q->info), "quantized row-wise 8i size should match");
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
		ccv_nnc_dequantize_8i_rowwise(q->data.u8, CCV_16BF, CCV_TENSOR_CPU_MEMORY, qsize, cols, expected->data.f16, (size_t)rows * cols);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 16BF, rows, cols)), 0);
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, rows, cols), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
		ccv_nnc_dequantize_8i_rowwise(gq->data.u8, CCV_16BF, CCV_TENSOR_GPU_MEMORY, qsize, cols, gout->data.u8, (size_t)rows * cols);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, rows, cols), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
		float* const expected_f32 = (float*)ccmalloc(sizeof(float) * (size_t)rows * cols);
		float* const actual_f32 = (float*)ccmalloc(sizeof(float) * (size_t)rows * cols);
		ccv_bfloat_to_float((uint16_t*)expected->data.f16, expected_f32, rows * cols);
		ccv_bfloat_to_float((uint16_t*)actual->data.f16, actual_f32, rows * cols);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_f32, actual_f32, rows * cols, 5e-3, "GPU row-wise 8i bf16 dequantize should match CPU dequantize on large shape");
		ccfree(actual_f32);
		ccfree(expected_f32);
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(q);
		ccv_nnc_tensor_free(source);
		ccfree(values);
	}
}

TEST_CASE("mps dequantize packed row-wise 8i-x half precision all formats")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q5_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q3_K,
		CCV_NNC_QX_8I_ROWWISE_Q2_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_S,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XS,
		CCV_NNC_QX_8I_ROWWISE_IQ3_S,
		CCV_NNC_QX_8I_ROWWISE_IQ3_XXS,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	};
	int i;
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		double max_abs = 0;
		const int status = _mps_dequantize_8i_rowwise_x_compare(CCV_16F, formats[i], 17, 130, 1e-3, &max_abs);
		REQUIRE_EQ(status, 0, "GPU packed row-wise 8i-x fp16 dequantize should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
	}
}

TEST_CASE("mps dequantize packed row-wise 8i-x float and bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	const int formats[] = {
		CCV_NNC_QX_8I_ROWWISE_Q4_K,
		CCV_NNC_QX_8I_ROWWISE_Q6_K,
		CCV_NNC_QX_8I_ROWWISE_IQ2_XXS,
	};
	int i;
	for (i = 0; i < (int)(sizeof(formats) / sizeof(formats[0])); i++)
	{
		double max_abs = 0;
		int status = _mps_dequantize_8i_rowwise_x_compare(CCV_32F, formats[i], 13, 131, 1e-5, &max_abs);
		REQUIRE_EQ(status, 0, "GPU packed row-wise 8i-x fp32 dequantize should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
		max_abs = 0;
		status = _mps_dequantize_8i_rowwise_x_compare(CCV_16BF, formats[i], 13, 131, 1e-2, &max_abs);
		REQUIRE_EQ(status, 0, "GPU packed row-wise 8i-x bf16 dequantize should match CPU for format=%d, max_abs=%g", formats[i], max_abs);
	}
}

TEST_CASE("mps forward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	for (i = 0; i < 64; i++)
		tb1->data.f32[i] = tb->data.f32[i];
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 5e-6, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
}

TEST_CASE("mps forward gemm no bias in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps forward gemm no bias in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, 10, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	for (i = 0; i < 10 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps forward gemv in half precision no bias, variant 1")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 128), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps forward gemv in half precision no bias with M=2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	for (i = 0; i < 2 * 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 128), 0);
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 2 * 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps forward gemv in half precision no bias, variant 2")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 128), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 128, 1), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 64, 1), 0);

	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	ccv_nnc_tensor_t* ha1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 128, 1), 0);
	for (i = 0; i < 128; i++)
		ha1->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 128; i++)
		ha->data.f32[i] = ha1->data.f32[i];
	ccv_nnc_tensor_t* hw2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 128), 0);
	ccv_nnc_tensor_t* ha2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 128, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha1, hw), TENSOR_LIST(ha2, hw2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha2, hw2), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(hw, ha), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, NO_TRANSPOSE), ccv_nnc_no_hint, 0, TENSOR_LIST(w, a), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 64, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(tb), 0);
	ccv_nnc_tensor_t* tb1 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tb), TENSOR_LIST(tb1), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb1->data.f32, hb->data.f32, 64, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ha1);
	ccv_nnc_tensor_free(tb1);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ha2);
	ccv_nnc_tensor_free(hw2);
}

TEST_CASE("mps handle permute")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 2, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 2, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 64), 0);
	int i;
	for (i = 0; i < 2 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 2 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(w), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(128, 2 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_tensor_t* hbt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, bt), TENSOR_LIST(hb, hbt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, hbt->data.f32, 2 * 10 * 64, 1e-5, "permute computed output should be numerically close to non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hbt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(a, w), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, wt, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched gemm with batch (2, 4) with bias and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 64;
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(a, w, bias), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)av, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(at, hw, hbias), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(hb, bt, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw), TENSOR_LIST(dat, tdw), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw), TENSOR_LIST(hda, hdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* wt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dwt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 64, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 64, 4, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 8 * 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(hw), TENSOR_LIST(wt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* wv = ccv_nnc_tensor_view_new(w, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dwv = ccv_nnc_tensor_view_new(dw, GPU_TENSOR_NHWC(000, 32F, 2, 4, 64, 128), ccv_nnc_no_ofs, DIM_ALLOC(64 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, (ccv_nnc_tensor_t*)wv), TENSOR_LIST((ccv_nnc_tensor_t*)dav, (ccv_nnc_tensor_t*)dwv, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(2, 3)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, wt), TENSOR_LIST(dat, dwt, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dwt), TENSOR_LIST(tdw), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(wv);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_view_free(dwv);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(wt);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(dwt);
	ccv_nnc_tensor_free(tda);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("generalized batched backward gemm with batch (2, 4) with bias and broadcast compare mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	// This is a particular batched gemm which treat every dimensions other than the last two as batching.
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 64), 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* da = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 64), 0);

	ccv_nnc_tensor_t* at = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* dat = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 10, 128), 0);
	ccv_nnc_tensor_t* tda = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 10, 4, 128), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 8 * 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 2 * 4 * 10 * 64; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(at), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hb), TENSOR_LIST(a, w, b), 0);
	ccv_nnc_tensor_view_t* av = ccv_nnc_tensor_view_new(a, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_tensor_view_t* dav = ccv_nnc_tensor_view_new(da, GPU_TENSOR_NHWC(000, 32F, 2, 4, 10, 128), ccv_nnc_no_ofs, DIM_ALLOC(10 * 4 * 128, 128, 4 * 128, 1));
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(b, (ccv_nnc_tensor_t*)av, w, dbias), TENSOR_LIST((ccv_nnc_tensor_t*)dav, dw, dbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hb, at, hw, hdbias), TENSOR_LIST(dat, tdw, tdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(da, dw, dbias), TENSOR_LIST(hda, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_TRANSPOSE_FORWARD(1, 2), ccv_nnc_no_hint, 0, TENSOR_LIST(dat), TENSOR_LIST(tda), 0);
	REQUIRE_TENSOR_EQ(hda, tda, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdw, tdw, "permute computed output should be the same as non-permute computed ones");
	REQUIRE_TENSOR_EQ(hdbias, tdbias, "permute computed output should be the same as non-permute computed ones");
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hda);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(da);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_view_free(av);
	ccv_nnc_tensor_view_free(dav);
	ccv_nnc_tensor_free(at);
	ccv_nnc_tensor_free(dat);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("ewdiv forward with reciprocal")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(0, ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewdiv forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWDIV_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	for (i = 0; i < 1000; i++)
		hb->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 0.01;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(a, b), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, b), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWDIV_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hb), TENSOR_LIST(ct), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_TENSOR_EQ(ct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("exp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWEXP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWEXP_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 100, 2e-3, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("softplus forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSOFTPLUS_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 101), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 101), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 101), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 101), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 101), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1010; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 40 - 20;
	ha->data.f32[0] = -100;
	ha->data.f32[1] = -40;
	ha->data.f32[2] = 0;
	ha->data.f32[3] = 40;
	ha->data.f32[4] = 100;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSOFTPLUS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 101, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewpow forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWPOW_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* c = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hc = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* ct = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 + 0.1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(c), 0);
	ccv_nnc_cmd_exec(CMD_EWPOW_FORWARD(3), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ct), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(hc), 0);
	REQUIRE_TENSOR_EQ(ct, hc, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hc);
	ccv_nnc_tensor_free(ct);
}

TEST_CASE("ewsin forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSIN_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSIN_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewcos forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWCOS_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWCOS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, bt->data.f32, hb->data.f32, 10 * 100, 1e-3, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewlog forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWLOG_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWLOG_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewsqrt forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWSQRT_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWSQRT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("ewabs forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_EWABS_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 5 + 0.0001;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_EWABS_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward with only max")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(NAN, 6), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("clamp forward with only min")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CLAMP_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 10, 100), 0);
	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	ccv_nnc_tensor_t* bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 10, 100), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	int i;
	for (i = 0; i < 1000; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 10 - 1;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_CLAMP_FORWARD(0, NAN), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(bt), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	REQUIRE_TENSOR_EQ(bt, hb, "GPU computed output should be the same as CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("compare set with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SET_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 11, 10, 9, 8), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 11, 10, 9, 8), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 11, 10, 9, 8), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(a), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(ga), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(ha), 0);
	REQUIRE_TENSOR_EQ(ha, ga, "format transform result should be the same");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(ga);
}

TEST_CASE("scaled dot product attention with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// Bypass error: variable-sized object may not be initialized
#define num_long_trials 7
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] =         {  32, 1, 1, 1,  32,   1,   3, 2, 1 };
		int R_candidates[num_trials] =         { 128, 4128, 4098, 4162, 128, 192,  61, 6, 2 };
		int C_candidates[num_trials] =         { 128, 4128, 4098, 4162, 128,  64,  49, 2, 1 };
		int Hq_candidates[num_trials] =        {   8, 32, 32, 32,  32,  16,  13, 3, 1 };
		int Hk_candidates[num_trials] =        {   8, 8, 8, 8,   8,  16,  13, 3, 1 };
		int D_candidates[num_trials] =         {  64, 32, 32, 32, 128, 128, 191, 4, 8 };
		int is_causal_candidates[num_trials] = {   0, 0, 0, 0,   1,   1,   0, 1, 0 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int Hq = Hq_candidates[trial];
		int Hk = Hk_candidates[trial];
		int D = D_candidates[trial];
		int is_causal = is_causal_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = (float)(i) / (float)(B * R * Hq * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = (float)(i) / (float)(B * C * Hk * D);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

		// Why it there 000 in the beginning of the argument list for GPU_TENSOR_NHWC?
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		if (is_causal)
		{
			ccv_nnc_tensor_t* const causal_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
			ccv_nnc_tensor_t* const gpu_causal_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, R, C), 0);
			for (int i = 0; i < R; i++)
				for (int j = 0; j < C; j++)
					causal_mask->data.f32[i * C + j] = 0;
			for (int i = 0; i < R - 1; i++)
				for (int j = ccv_max(i - R + C + 1, 0); j < C; j++)
					causal_mask->data.f32[i * C + j] = -FLT_MAX;
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(causal_mask), TENSOR_LIST(gpu_causal_mask), 0);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_causal_mask), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_tensor_free(gpu_causal_mask);
			ccv_nnc_tensor_free(causal_mask);
		} else {
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		}

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, B * R * Hq * D, 1e-3, "scaled dot product attention result should be the same");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention with masked NA mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 80;
	const int C = 64;
	const int H = 8;
	const int D = 128;
	const float scale = 1.0 / sqrt((float)D);
	const int q_count = B * R * H * D;
	const int kv_count = B * C * H * D;
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const causal_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const causal_mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 31);
	for (int i = 0; i < q_count; ++i)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < R; ++i)
		for (int j = 0; j < C; ++j)
		{
			mask_tensor->data.f32[i * C + j] = ((i * 13 + j * 7) % 11 == 0) ? -0.75 : (((i + j) % 5 == 0) ? 0.125 : 0);
			causal_mask_tensor->data.f32[i * C + j] = (j <= i + C - R) ? 0 : -FLT_MAX;
		}
	ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
	ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(mask_tensor->data.f32, (uint16_t*)mask_tensor_f16->data.f16, R * C);
	ccv_float_to_half_precision(causal_mask_tensor->data.f32, (uint16_t*)causal_mask_tensor_f16->data.f16, R * C);

	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const gpu_causal_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor_f16, causal_mask_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask_tensor, gpu_causal_mask_tensor), 0);
	for (int trial = 0; trial < 3; ++trial)
	{
		const int is_causal = trial == 1;
		ccv_nnc_tensor_t* const cpu_mask = trial == 2 ? 0 : mask_tensor;
		ccv_nnc_tensor_t* const gpu_mask = trial == 2 ? gpu_causal_mask_tensor : gpu_mask_tensor;
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		if (cpu_mask)
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, cpu_mask), TENSOR_LIST(o_tensor), 0);
		else
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, trial == 2 ? 0 : is_causal);
		gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask), TENSOR_LIST(gpu_o_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		const int count = B * R * H * D;
		float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < count; ++i)
		{
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
			const float relative_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]) / denom;
			if (relative_diff > max_relative_diff)
				max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 5e-2, "masked NA attention should match CPU reference for trial=%d (max relative diff %g at %d: %g vs %g)", trial, max_relative_diff, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);
		ccfree(gpu_f32);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(o_tensor);
	}
	ccv_nnc_tensor_free(gpu_causal_mask_tensor);
	ccv_nnc_tensor_free(gpu_mask_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(causal_mask_tensor_f16);
	ccv_nnc_tensor_free(mask_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(causal_mask_tensor);
	ccv_nnc_tensor_free(mask_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(q_tensor);
}

TEST_CASE("scaled dot product attention with generic causal and masked mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 2;
	const int R = 96;
	const int C = 80;
	const int H = 4;
	const int D = 64;
	const float scale = 1.0 / sqrt((float)D);
	const int q_count = B * R * H * D;
	const int kv_count = B * C * H * D;
	const int out_count = q_count;
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const triangular_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const broadcast_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const batched_mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, 1, R, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 67);
	for (int i = 0; i < q_count; ++i)
		q_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (int i = 0; i < kv_count; ++i)
		k_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (int i = 0; i < kv_count; ++i)
		v_tensor->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (int i = 0; i < R; ++i)
		for (int j = 0; j < C; ++j)
		{
			triangular_mask_tensor->data.f32[i * C + j] = (j <= i + C - R) ? 0 : -FLT_MAX;
			const int slot = (i * 17 + j * 13) % 29;
			broadcast_mask_tensor->data.f32[i * C + j] = (slot == 0) ? -FLT_MAX : ((slot == 1) ? -0.375 : ((slot == 2) ? 0.125 : 0));
		}
	for (int b = 0; b < B; ++b)
		for (int i = 0; i < R; ++i)
			for (int j = 0; j < C; ++j)
			{
				const int slot = (b * 37 + i * 11 + j * 5) % 31;
				batched_mask_tensor->data.f32[(b * R + i) * C + j] = (slot == 0) ? -FLT_MAX : ((slot == 1) ? -0.5 : ((slot == 2) ? 0.25 : 0));
			}

	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_triangular_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const gpu_broadcast_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const gpu_batched_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, 1, R, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, triangular_mask_tensor, broadcast_mask_tensor, batched_mask_tensor), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_triangular_mask_tensor, gpu_broadcast_mask_tensor, gpu_batched_mask_tensor), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	for (int trial = 0; trial < 5; ++trial)
	{
		const int is_causal = (trial == 0 || trial == 4);
		ccv_nnc_tensor_t* const cpu_mask = (trial == 1) ? triangular_mask_tensor : ((trial == 2 || trial == 4) ? broadcast_mask_tensor : (trial == 3 ? batched_mask_tensor : 0));
		ccv_nnc_tensor_t* const gpu_mask = (trial == 1) ? gpu_triangular_mask_tensor : ((trial == 2 || trial == 4) ? gpu_broadcast_mask_tensor : (trial == 3 ? gpu_batched_mask_tensor : 0));
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_t cpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, trial == 1 ? 1 : is_causal);
		if (trial == 1)
			ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);
		else if (cpu_mask)
			ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, cpu_mask), TENSOR_LIST(o_tensor), 0);
		else
			ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		if (gpu_mask)
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask), TENSOR_LIST(gpu_o_tensor), 0);
		else
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		float max_abs_diff = 0;
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < out_count; ++i)
		{
			const float abs_diff = fabsf(o_tensor->data.f32[i] - copy_of_gpu_o_tensor->data.f32[i]);
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(copy_of_gpu_o_tensor->data.f32[i])), 1.0f);
			const float relative_diff = abs_diff / denom;
			if (relative_diff > max_relative_diff)
			{
				max_abs_diff = abs_diff;
				max_relative_diff = relative_diff, max_diff_idx = i;
			}
		}
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(o_tensor);
		REQUIRE(max_relative_diff <= 2e-2, "generic MPS attention should match CPU reference for trial=%d (max abs %g relative %g at %d)", trial, max_abs_diff, max_relative_diff, max_diff_idx);
	}
	{
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
		ccv_nnc_tensor_t* const q_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const mask_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
		ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
		ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
		ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
		ccv_float_to_half_precision(broadcast_mask_tensor->data.f32, (uint16_t*)mask_tensor_f16->data.f16, R * C);
		ccv_half_precision_to_float((uint16_t*)q_tensor_f16->data.f16, q_tensor_ref->data.f32, q_count);
		ccv_half_precision_to_float((uint16_t*)k_tensor_f16->data.f16, k_tensor_ref->data.f32, kv_count);
		ccv_half_precision_to_float((uint16_t*)v_tensor_f16->data.f16, v_tensor_ref->data.f32, kv_count);
		ccv_half_precision_to_float((uint16_t*)mask_tensor_f16->data.f16, mask_tensor_ref->data.f32, R * C);
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_ref, k_tensor_ref, v_tensor_ref, mask_tensor_ref), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_float_to_half_precision(o_tensor->data.f32, (uint16_t*)o_tensor_f16->data.f16, out_count);
		ccv_half_precision_to_float((uint16_t*)o_tensor_f16->data.f16, o_tensor->data.f32, out_count);
		ccv_nnc_tensor_t* const gpu_q_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_mask_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor_f16), TENSOR_LIST(gpu_q_tensor_f16, gpu_k_tensor_f16, gpu_v_tensor_f16, gpu_mask_tensor_f16), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
		gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor_f16, gpu_k_tensor_f16, gpu_v_tensor_f16, gpu_mask_tensor_f16), TENSOR_LIST(gpu_o_tensor_f16), 0);
		if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * out_count);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor_f16->data.f16, gpu_f32, out_count);
		float max_abs_diff = 0;
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < out_count; ++i)
		{
			const float abs_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]);
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
			const float relative_diff = abs_diff / denom;
			if (relative_diff > max_relative_diff)
				max_abs_diff = abs_diff, max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 8e-2, "generic MPS attention should match CPU reference for broadcast masked 16F (max abs %g relative %g at %d)", max_abs_diff, max_relative_diff, max_diff_idx);
		ccfree(gpu_f32);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(gpu_o_tensor_f16);
		ccv_nnc_tensor_free(gpu_mask_tensor_f16);
		ccv_nnc_tensor_free(gpu_v_tensor_f16);
		ccv_nnc_tensor_free(gpu_k_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor_f16);
		ccv_nnc_tensor_free(o_tensor_f16);
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(mask_tensor_ref);
		ccv_nnc_tensor_free(v_tensor_ref);
		ccv_nnc_tensor_free(k_tensor_ref);
		ccv_nnc_tensor_free(q_tensor_ref);
		ccv_nnc_tensor_free(mask_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(q_tensor_f16);
	}
	ccv_nnc_tensor_free(gpu_batched_mask_tensor);
	ccv_nnc_tensor_free(gpu_broadcast_mask_tensor);
	ccv_nnc_tensor_free(gpu_triangular_mask_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(batched_mask_tensor);
	ccv_nnc_tensor_free(broadcast_mask_tensor);
	ccv_nnc_tensor_free(triangular_mask_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(q_tensor);
}

TEST_CASE("scaled dot product attention with generic varlen mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 4;
	const int Hq = 4;
	const int Hk = 2;
	const int D = 64;
	const int q_offsets[2][5] = {
		{ 0, 17, 80, 145, 223 },
		{ 0, 16, 81, 147, 226 },
	};
	const int kv_offsets[2][5] = {
		{ 0, 19, 83, 150, 229 },
		{ 0, 18, 82, 149, 229 },
	};
	const int max_seqlen_q[2] = { 78, 79 };
	const int max_seqlen_kv[2] = { 79, 80 };
	const uint64_t old_flags = ccv_nnc_flags();
	for (int trial = 0; trial < 2; ++trial)
	{
		const int is_causal = trial;
		const int total_q = q_offsets[trial][B];
		const int total_k = kv_offsets[trial][B];
		const float scale = 1.0 / sqrt((float)D);
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		for (int i = 0; i < total_q * Hq * D; ++i)
			q_tensor->data.f32[i] = (float)((i * 17) % 97 - 48) / 256;
		for (int i = 0; i < total_k * Hk * D; ++i)
		{
			k_tensor->data.f32[i] = (float)((i * 13) % 89 - 44) / 256;
			v_tensor->data.f32[i] = (float)((i * 19) % 101 - 50) / 128;
		}
		ccv_nnc_tensor_t* const o_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		for (int b = 0; b < B; ++b)
		{
			const int q_start = q_offsets[trial][b];
			const int k_start = kv_offsets[trial][b];
			const int R = q_offsets[trial][b + 1] - q_start;
			const int C = kv_offsets[trial][b + 1] - k_start;
			ccv_nnc_tensor_t* const q_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const o_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			memcpy(q_seq->data.f32, q_tensor->data.f32 + q_start * Hq * D, sizeof(float) * R * Hq * D);
			memcpy(k_seq->data.f32, k_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			memcpy(v_seq->data.f32, v_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_seq, k_seq, v_seq), TENSOR_LIST(o_seq), 0);
			memcpy(o_tensor_ref->data.f32 + q_start * Hq * D, o_seq->data.f32, sizeof(float) * R * Hq * D);
			ccv_nnc_tensor_free(q_seq);
			ccv_nnc_tensor_free(k_seq);
			ccv_nnc_tensor_free(v_seq);
			ccv_nnc_tensor_free(o_seq);
		}
		ccv_nnc_tensor_t* const q_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		ccv_nnc_tensor_t* const kv_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		for (int i = 0; i < B + 1; ++i)
		{
			q_seq_offsets->data.i32[i] = q_offsets[trial][i];
			kv_seq_offsets->data.i32[i] = kv_offsets[trial][i];
		}
		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.is_varlen = 1;
		cmd.info.scaled_dot_product_attention.max_seqlen_q = max_seqlen_q[trial];
		cmd.info.scaled_dot_product_attention.max_seqlen_kv = max_seqlen_kv[trial];
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_q_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_kv_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, q_seq_offsets, kv_seq_offsets), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_q_seq_offsets, gpu_kv_seq_offsets), 0);
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL, gpu_q_seq_offsets, gpu_kv_seq_offsets), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		float max_abs_diff = 0;
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < total_q * Hq * D; ++i)
		{
			const float abs_diff = fabsf(o_tensor_ref->data.f32[i] - copy_of_gpu_o_tensor->data.f32[i]);
			const float denom = fmaxf(fmaxf(fabsf(o_tensor_ref->data.f32[i]), fabsf(copy_of_gpu_o_tensor->data.f32[i])), 1.0f);
			const float relative_diff = abs_diff / denom;
			if (relative_diff > max_relative_diff)
				max_abs_diff = abs_diff, max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 2e-2, "generic varlen MPS attention should match CPU reference when causal=%d (max abs %g relative %g at %d)", is_causal, max_abs_diff, max_relative_diff, max_diff_idx);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const q_tensor_round = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_round = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_round = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, total_q * Hq * D);
		ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, total_k * Hk * D);
		ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, total_k * Hk * D);
		ccv_half_precision_to_float((uint16_t*)q_tensor_f16->data.f16, q_tensor_round->data.f32, total_q * Hq * D);
		ccv_half_precision_to_float((uint16_t*)k_tensor_f16->data.f16, k_tensor_round->data.f32, total_k * Hk * D);
		ccv_half_precision_to_float((uint16_t*)v_tensor_f16->data.f16, v_tensor_round->data.f32, total_k * Hk * D);
		ccv_nnc_tensor_t* const o_tensor_ref_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		for (int b = 0; b < B; ++b)
		{
			const int q_start = q_offsets[trial][b];
			const int k_start = kv_offsets[trial][b];
			const int R = q_offsets[trial][b + 1] - q_start;
			const int C = kv_offsets[trial][b + 1] - k_start;
			ccv_nnc_tensor_t* const q_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const o_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			memcpy(q_seq->data.f32, q_tensor_round->data.f32 + q_start * Hq * D, sizeof(float) * R * Hq * D);
			memcpy(k_seq->data.f32, k_tensor_round->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			memcpy(v_seq->data.f32, v_tensor_round->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_seq, k_seq, v_seq), TENSOR_LIST(o_seq), 0);
			memcpy(o_tensor_ref_f16->data.f32 + q_start * Hq * D, o_seq->data.f32, sizeof(float) * R * Hq * D);
			ccv_nnc_tensor_free(q_seq);
			ccv_nnc_tensor_free(k_seq);
			ccv_nnc_tensor_free(v_seq);
			ccv_nnc_tensor_free(o_seq);
		}
		ccv_nnc_tensor_t* const gpu_q_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor_f16, gpu_k_tensor_f16, gpu_v_tensor_f16), 0);
		ccv_nnc_cmd_t cmd_f16 = cmd;
		cmd_f16.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(cmd_f16, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor_f16, gpu_k_tensor_f16, gpu_v_tensor_f16, NULL, NULL, NULL, gpu_q_seq_offsets, gpu_kv_seq_offsets), TENSOR_LIST(gpu_o_tensor_f16, NULL), 0);
		if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * total_q * Hq * D);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor_f16->data.f16, gpu_f32, total_q * Hq * D);
		max_abs_diff = 0;
		max_relative_diff = 0;
		max_diff_idx = 0;
		for (int i = 0; i < total_q * Hq * D; ++i)
		{
			const float abs_diff = fabsf(o_tensor_ref_f16->data.f32[i] - gpu_f32[i]);
			const float denom = fmaxf(fmaxf(fabsf(o_tensor_ref_f16->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
			const float relative_diff = abs_diff / denom;
			if (relative_diff > max_relative_diff)
				max_abs_diff = abs_diff, max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 8e-2, "generic varlen 16F MPS attention should match CPU reference when causal=%d (max abs %g relative %g at %d)", is_causal, max_abs_diff, max_relative_diff, max_diff_idx);
		ccfree(gpu_f32);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(gpu_o_tensor_f16);
		ccv_nnc_tensor_free(gpu_v_tensor_f16);
		ccv_nnc_tensor_free(gpu_k_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor_f16);
		ccv_nnc_tensor_free(o_tensor_ref_f16);
		ccv_nnc_tensor_free(v_tensor_round);
		ccv_nnc_tensor_free(k_tensor_round);
		ccv_nnc_tensor_free(q_tensor_round);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_kv_seq_offsets);
		ccv_nnc_tensor_free(gpu_q_seq_offsets);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(kv_seq_offsets);
		ccv_nnc_tensor_free(q_seq_offsets);
		ccv_nnc_tensor_free(o_tensor_ref);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(q_tensor);
	}
}

TEST_CASE("scaled dot product attention with NA mps for odd sequence lengths")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int H = 16;
	const int D = 64;
	const int shapes[][2] = {
		{ 47, 47 },
		{ 161, 161 },
		{ 47, 41 },
		{ 161, 145 },
	};
	const float scale = 1.0 / sqrt((float)D);
	for (int shape_idx = 0; shape_idx < (int)(sizeof(shapes) / sizeof(shapes[0])); ++shape_idx)
	{
		const int R = shapes[shape_idx][0];
		const int C = shapes[shape_idx][1];
		const int q_count = B * R * H * D;
		const int kv_count = B * C * H * D;
		for (int is_causal = 0; is_causal <= 1; ++is_causal)
		{
			ccv_nnc_tensor_t* const q_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			for (int r = 0; r < R; ++r)
				for (int h = 0; h < H; ++h)
					for (int d = 0; d < D; ++d)
						q_tensor_ref->data.f32[((r * H + h) * D) + d] = (((r * 17 + h * 5 + d * 3 + is_causal) % 31) - 15) * (1.0 / 128);
			for (int c = 0; c < C; ++c)
				for (int h = 0; h < H; ++h)
					for (int d = 0; d < D; ++d)
					{
						k_tensor_ref->data.f32[((c * H + h) * D) + d] = (((c * 19 + h * 7 + d * 5 + shape_idx) % 31) - 15) * (1.0 / 128);
						v_tensor_ref->data.f32[((c * H + h) * D) + d] = (((c * 29 + h * 11 + d * 7 + 3) % 127) - 63) * (1.0 / 512);
					}
			ccv_float_to_half_precision(q_tensor_ref->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
			ccv_float_to_half_precision(k_tensor_ref->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
			ccv_float_to_half_precision(v_tensor_ref->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
			ccv_half_precision_to_float((uint16_t*)q_tensor_f16->data.f16, q_tensor_ref->data.f32, q_count);
			ccv_half_precision_to_float((uint16_t*)k_tensor_f16->data.f16, k_tensor_ref->data.f32, kv_count);
			ccv_half_precision_to_float((uint16_t*)v_tensor_f16->data.f16, v_tensor_ref->data.f32, kv_count);
			ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_ref, k_tensor_ref, v_tensor_ref), TENSOR_LIST(o_tensor), 0);
			ccv_nnc_tensor_t* const o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_float_to_half_precision(o_tensor->data.f32, (uint16_t*)o_tensor_f16->data.f16, q_count);
			ccv_half_precision_to_float((uint16_t*)o_tensor_f16->data.f16, o_tensor->data.f32, q_count);

			ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
			ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
			gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F;
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

			float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
			ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, q_count);
			float max_abs_diff = 0;
			float max_relative_diff = 0;
			int max_diff_idx = 0;
			for (int i = 0; i < q_count; ++i)
			{
				REQUIRE(isfinite(gpu_f32[i]), "NA attention output should stay finite for R=%d C=%d causal=%d at %d", R, C, is_causal, i);
				const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
				const float abs_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]);
				const float relative_diff = abs_diff / denom;
				if (relative_diff > max_relative_diff)
					max_relative_diff = relative_diff, max_abs_diff = abs_diff, max_diff_idx = i;
			}
			const int max_diff_d = max_diff_idx % D;
			const int max_diff_h = (max_diff_idx / D) % H;
			const int max_diff_r = (max_diff_idx / (H * D)) % R;
			const int max_diff_b = max_diff_idx / (R * H * D);
			REQUIRE(max_relative_diff <= 1e-4, "NA attention should match half-rounded CPU reference for R=%d C=%d causal=%d (max abs %g relative %g at b=%d r=%d h=%d d=%d idx=%d: CPU %g GPU %g)", R, C, is_causal, max_abs_diff, max_relative_diff, max_diff_b, max_diff_r, max_diff_h, max_diff_d, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);

			ccfree(gpu_f32);
			ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(o_tensor_f16);
			ccv_nnc_tensor_free(o_tensor);
			ccv_nnc_tensor_free(v_tensor_ref);
			ccv_nnc_tensor_free(k_tensor_ref);
			ccv_nnc_tensor_free(q_tensor_ref);
			ccv_nnc_tensor_free(q_tensor_f16);
			ccv_nnc_tensor_free(k_tensor_f16);
			ccv_nnc_tensor_free(v_tensor_f16);
		}
	}
}

TEST_CASE("scaled dot product attention with NA mps splitKV decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const struct {
		int R;
		int C;
		int Hq;
		int Hk;
		int D;
	} cases[] = {
		{ 1, 1536, 40, 4, 256 },
		{ 1, 2112, 8, 8, 256 },
		{ 1, 2113, 8, 8, 256 },
		{ 1, 8193, 40, 4, 256 },
		{ 2, 4096, 8, 8, 128 },
		{ 7, 4096, 8, 8, 128 },
		{ 7, 4097, 8, 8, 128 },
		{ 16, 4096, 8, 8, 128 },
		{ 16, 4096, 32, 8, 128 },
		{ 16, 4097, 32, 8, 128 },
		{ 32, 4096, 4, 4, 128 },
	};
	for (int case_idx = 0; case_idx < sizeof(cases) / sizeof(cases[0]); ++case_idx)
	{
		const int R = cases[case_idx].R;
		const int C = cases[case_idx].C;
		const int Hq = cases[case_idx].Hq;
		const int Hk = cases[case_idx].Hk;
		const int D = cases[case_idx].D;
		const int q_count = B * R * Hq * D;
		const int kv_count = B * C * Hk * D;
		const float scale = 1.0 / sqrt((float)D);
		for (int is_causal = 0; is_causal <= 1; ++is_causal)
		{
			ccv_nnc_tensor_t* const q_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
			ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
			for (int i = 0; i < q_count; ++i)
				q_tensor_ref->data.f32[i] = (float)((i * 17 + is_causal * 13 + R * 5) % 61 - 30) * (1.0 / 128);
			for (int i = 0; i < kv_count; ++i)
			{
				k_tensor_ref->data.f32[i] = (float)((i * 19 + R * 7) % 67 - 33) * (1.0 / 128);
				v_tensor_ref->data.f32[i] = (float)((i * 23 + C / 32) % 127 - 63) * (1.0 / 512);
			}
			ccv_float_to_half_precision(q_tensor_ref->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
			ccv_float_to_half_precision(k_tensor_ref->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
			ccv_float_to_half_precision(v_tensor_ref->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
			ccv_half_precision_to_float((uint16_t*)q_tensor_f16->data.f16, q_tensor_ref->data.f32, q_count);
			ccv_half_precision_to_float((uint16_t*)k_tensor_f16->data.f16, k_tensor_ref->data.f32, kv_count);
			ccv_half_precision_to_float((uint16_t*)v_tensor_f16->data.f16, v_tensor_ref->data.f32, kv_count);
			ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_ref, k_tensor_ref, v_tensor_ref), TENSOR_LIST(o_tensor), 0);
			ccv_nnc_tensor_t* const o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
			ccv_float_to_half_precision(o_tensor->data.f32, (uint16_t*)o_tensor_f16->data.f16, q_count);
			ccv_half_precision_to_float((uint16_t*)o_tensor_f16->data.f16, o_tensor->data.f32, q_count);

			ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
			ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
			ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
			ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
			ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
			ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

			float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
			ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, q_count);
			float max_abs_diff = 0;
			float max_relative_diff = 0;
			int max_diff_idx = 0;
			for (int i = 0; i < q_count; ++i)
			{
				REQUIRE(isfinite(gpu_f32[i]), "splitKV NA attention output should stay finite for R=%d C=%d Hq=%d Hk=%d D=%d causal=%d at %d", R, C, Hq, Hk, D, is_causal, i);
				const float abs_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]);
				const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
				const float relative_diff = abs_diff / denom;
				if (relative_diff > max_relative_diff)
					max_relative_diff = relative_diff, max_abs_diff = abs_diff, max_diff_idx = i;
			}
			REQUIRE(max_relative_diff <= 2e-3, "splitKV NA attention should match half-rounded CPU reference for R=%d C=%d Hq=%d Hk=%d D=%d causal=%d (max abs %g relative %g at %d: CPU %g GPU %g)", R, C, Hq, Hk, D, is_causal, max_abs_diff, max_relative_diff, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);

			ccfree(gpu_f32);
			ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(o_tensor_f16);
			ccv_nnc_tensor_free(o_tensor);
			ccv_nnc_tensor_free(v_tensor_ref);
			ccv_nnc_tensor_free(k_tensor_ref);
			ccv_nnc_tensor_free(q_tensor_ref);
			ccv_nnc_tensor_free(q_tensor_f16);
			ccv_nnc_tensor_free(k_tensor_f16);
			ccv_nnc_tensor_free(v_tensor_f16);
		}
	}
}

TEST_CASE("scaled dot product attention with BF16 R1 mps decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 1;
	const int C = 4097;
	const int Hq = 40;
	const int Hk = 4;
	const int D = 256;
	const int q_count = B * R * Hq * D;
	const int kv_count = B * C * Hk * D;
	const float scale = 1.0 / sqrt((float)D);
	for (int is_causal = 0; is_causal <= 1; ++is_causal)
	{
		ccv_nnc_tensor_t* const q_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const q_tensor_bf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_bf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_bf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		for (int i = 0; i < q_count; ++i)
			q_tensor_ref->data.f32[i] = (float)((i * 13 + is_causal * 7) % 71 - 35) * (1.0 / 128);
		for (int i = 0; i < kv_count; ++i)
		{
			k_tensor_ref->data.f32[i] = (float)((i * 17 + C / 16) % 79 - 39) * (1.0 / 128);
			v_tensor_ref->data.f32[i] = (float)((i * 29 + Hq) % 127 - 63) * (1.0 / 512);
		}
		ccv_float_to_bfloat(q_tensor_ref->data.f32, (uint16_t*)q_tensor_bf16->data.f16, q_count);
		ccv_float_to_bfloat(k_tensor_ref->data.f32, (uint16_t*)k_tensor_bf16->data.f16, kv_count);
		ccv_float_to_bfloat(v_tensor_ref->data.f32, (uint16_t*)v_tensor_bf16->data.f16, kv_count);
		ccv_bfloat_to_float((uint16_t*)q_tensor_bf16->data.f16, q_tensor_ref->data.f32, q_count);
		ccv_bfloat_to_float((uint16_t*)k_tensor_bf16->data.f16, k_tensor_ref->data.f32, kv_count);
		ccv_bfloat_to_float((uint16_t*)v_tensor_bf16->data.f16, v_tensor_ref->data.f32, kv_count);

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_ref, k_tensor_ref, v_tensor_ref), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const o_tensor_bf16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_float_to_bfloat(o_tensor->data.f32, (uint16_t*)o_tensor_bf16->data.f16, q_count);
		ccv_bfloat_to_float((uint16_t*)o_tensor_bf16->data.f16, o_tensor->data.f32, q_count);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_bf16, k_tensor_bf16, v_tensor_bf16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

		float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
		ccv_bfloat_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, q_count);
		float max_abs_diff = 0;
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < q_count; ++i)
		{
			REQUIRE(isfinite(gpu_f32[i]), "BF16 R1 attention output should stay finite for C=%d Hq=%d Hk=%d D=%d causal=%d at %d", C, Hq, Hk, D, is_causal, i);
			const float abs_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]);
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
			const float relative_diff = abs_diff / denom;
			if (relative_diff > max_relative_diff)
				max_relative_diff = relative_diff, max_abs_diff = abs_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 8e-3, "BF16 R1 attention should match bfloat-rounded CPU reference for C=%d Hq=%d Hk=%d D=%d causal=%d (max abs %g relative %g at %d: CPU %g GPU %g)", C, Hq, Hk, D, is_causal, max_abs_diff, max_relative_diff, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);

		ccfree(gpu_f32);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(o_tensor_bf16);
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(v_tensor_ref);
		ccv_nnc_tensor_free(k_tensor_ref);
		ccv_nnc_tensor_free(q_tensor_ref);
		ccv_nnc_tensor_free(q_tensor_bf16);
		ccv_nnc_tensor_free(k_tensor_bf16);
		ccv_nnc_tensor_free(v_tensor_bf16);
	}
}

TEST_CASE("scaled dot product attention with attention sinks on mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	float max_abs = 0;
	float max_relative = 0;
	float expected = 0;
	float actual = 0;
	int max_idx = 0;
	int status = _mps_sdpa_attention_sinks_compare(CCV_32F, 1, 0, 1, 17, 23, 4, 2, 64, 0, 1, 0, 2e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "generic global sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F, 1, 47, 41, 8, 8, 64, 0, 8, 0, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NA per-head sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F, 1, 17, 23, 8, 4, 64, 1, 8, 4, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NA sliding-window grouped-query per-head sink should match CPU reference (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 1, CCV_NNC_GEMM_16F, 1, 17, 23, 8, 4, 64, 1, 8, 4, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "generic MFA sliding-window grouped-query per-head sink should match CPU reference (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F, 1, 23, 23, 8, 8, 64, 1, 8, 1, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NA sliding-window size 1 per-head sink should match CPU reference (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F, 1, 17, 23, 8, 8, 64, 1, 8, 64, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NA sliding-window size covering C should match CPU reference (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F, 1, 7, 4096, 8, 8, 128, 1, 8, 0, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NA splitKV per-head sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, 0, 1, 1, 1536, 8, 4, 128, 1, 8, 0, 5e-3, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "R1 direct per-head sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, 0, 1, 1, 4097, 8, 4, 128, 1, 8, 0, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "R1 split per-head sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);

	max_abs = 0;
	max_relative = 0;
	expected = 0;
	actual = 0;
	max_idx = 0;
	status = _mps_sdpa_attention_sinks_compare(CCV_16F, 0, CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I, 1, 64, 64, 8, 8, 128, 0, 8, 0, 5e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(status, 0, "NAInt8 per-head sink should run and match CPU reference with attention sinks (status %d max abs %g relative %g at %d: CPU %g GPU %g)", status, max_abs, max_relative, max_idx, expected, actual);
}

TEST_CASE("scaled dot product attention with varlen NA mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 4;
	const int Hq = 4;
	const int Hk = 2;
	const int D = 64;
	const int q_offsets[2][5] = {
		{ 0, 17, 80, 145, 223 },
		{ 0, 16, 81, 147, 226 },
	};
	const int kv_offsets[2][5] = {
		{ 0, 19, 83, 150, 229 },
		{ 0, 18, 82, 149, 229 },
	};
	const int max_seqlen_q[2] = { 78, 79 };
	const int max_seqlen_kv[2] = { 79, 80 };
	for (int trial = 0; trial < 2; ++trial)
	{
		const int is_causal = trial;
		const int total_q = q_offsets[trial][B];
		const int total_k = kv_offsets[trial][B];
		const float scale = 1.0 / sqrt((float)D);
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_k, Hk, D), 0);
		for (int i = 0; i < total_q * Hq * D; ++i)
			q_tensor->data.f32[i] = (float)((i * 17) % 97 - 48) / 256;
		for (int i = 0; i < total_k * Hk * D; ++i)
		{
			k_tensor->data.f32[i] = (float)((i * 13) % 89 - 44) / 256;
			v_tensor->data.f32[i] = (float)((i * 19) % 101 - 50) / 128;
		}
		ccv_nnc_tensor_t* const o_tensor_ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		for (int b = 0; b < B; ++b)
		{
			const int q_start = q_offsets[trial][b];
			const int k_start = kv_offsets[trial][b];
			const int R = q_offsets[trial][b + 1] - q_start;
			const int C = kv_offsets[trial][b + 1] - k_start;
			ccv_nnc_tensor_t* const q_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			ccv_nnc_tensor_t* const k_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const v_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, C, Hk, D), 0);
			ccv_nnc_tensor_t* const o_seq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, R, Hq, D), 0);
			memcpy(q_seq->data.f32, q_tensor->data.f32 + q_start * Hq * D, sizeof(float) * R * Hq * D);
			memcpy(k_seq->data.f32, k_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			memcpy(v_seq->data.f32, v_tensor->data.f32 + k_start * Hk * D, sizeof(float) * C * Hk * D);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_seq, k_seq, v_seq), TENSOR_LIST(o_seq), 0);
			memcpy(o_tensor_ref->data.f32 + q_start * Hq * D, o_seq->data.f32, sizeof(float) * R * Hq * D);
			ccv_nnc_tensor_free(q_seq);
			ccv_nnc_tensor_free(k_seq);
			ccv_nnc_tensor_free(v_seq);
			ccv_nnc_tensor_free(o_seq);
		}
		ccv_nnc_tensor_t* const h_q_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		ccv_nnc_tensor_t* const h_kv_seq_offsets = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, B + 1), 0);
		for (int i = 0; i < B + 1; ++i)
		{
			h_q_seq_offsets->data.i32[i] = q_offsets[trial][i];
			h_kv_seq_offsets->data.i32[i] = kv_offsets[trial][i];
		}
		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.is_varlen = 1;
		cmd.info.scaled_dot_product_attention.max_seqlen_q = max_seqlen_q[trial];
		cmd.info.scaled_dot_product_attention.max_seqlen_kv = max_seqlen_kv[trial];
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, NULL, NULL, NULL, h_q_seq_offsets, h_kv_seq_offsets), TENSOR_LIST(o_tensor, NULL), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, o_tensor->data.f32, o_tensor_ref->data.f32, total_q * Hq * D, 1e-5, "varlen CPU computed output should match per-sequence CPU output when causal=%d", is_causal);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_k, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);
		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_k, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_q_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_kv_seq_offsets = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, B + 1), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, h_q_seq_offsets, h_kv_seq_offsets), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_q_seq_offsets, gpu_kv_seq_offsets), 0);
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL, gpu_q_seq_offsets, gpu_kv_seq_offsets), TENSOR_LIST(gpu_o_tensor, NULL), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, total_q * Hq * D, 5e-3, "varlen MPS computed output should match packed CPU output when causal=%d", is_causal);
		ccv_nnc_cmd_t int8_cmd = cmd;
		int8_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		ccv_nnc_tensor_t* const gpu_o_tensor_int8 = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(int8_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL, gpu_q_seq_offsets, gpu_kv_seq_offsets), TENSOR_LIST(gpu_o_tensor_int8, NULL), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_int8_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor_int8), TENSOR_LIST(copy_of_gpu_o_tensor_int8_f16), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_int8 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, total_q, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_int8_f16), TENSOR_LIST(copy_of_gpu_o_tensor_int8), 0);
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < total_q * Hq * D; ++i)
		{
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(copy_of_gpu_o_tensor_int8->data.f32[i])), 1.0f);
			const float relative_diff = fabsf(o_tensor->data.f32[i] - copy_of_gpu_o_tensor_int8->data.f32[i]) / denom;
			if (relative_diff > max_relative_diff)
				max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 5e-2, "varlen quantized MPS output should match packed CPU output when causal=%d (max relative diff %g at %d: %g vs %g)", is_causal, max_relative_diff, max_diff_idx, o_tensor->data.f32[max_diff_idx], copy_of_gpu_o_tensor_int8->data.f32[max_diff_idx]);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(o_tensor_ref);
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(h_q_seq_offsets);
		ccv_nnc_tensor_free(h_kv_seq_offsets);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_q_seq_offsets);
		ccv_nnc_tensor_free(gpu_kv_seq_offsets);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor_int8);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_int8_f16);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_int8);
	}
}

TEST_CASE("scaled dot product attention with quantized NA mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 128;
	const int C = 64;
	const int H = 24;
	const int Ds[] = { 64, 80, 128, 130, 160, 192, 224, 256 };
	const int datatypes[] = { CCV_16F, CCV_16BF, CCV_32F };
	const float tolerances[] = { 2e-2, 3e-2, 2e-2 };
	const char* datatype_names[] = { "16F", "16BF", "32F" };
	for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
	{
		const int D = Ds[d_idx];
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		const int q_count = B * R * H * D;
		const int kv_count = B * C * H * D;
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 11 + d_idx);
		for (int i = 0; i < q_count; ++i)
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		for (int i = 0; i < kv_count; ++i)
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		for (int i = 0; i < kv_count; ++i)
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_t cpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

		for (int datatype_idx = 0; datatype_idx < 3; ++datatype_idx)
		{
			const int datatype = datatypes[datatype_idx];
			ccv_nnc_tensor_t* q_input = q_tensor;
			ccv_nnc_tensor_t* k_input = k_tensor;
			ccv_nnc_tensor_t* v_input = v_tensor;
			ccv_nnc_tensor_t* copy_of_gpu_o_tensor = 0;
			ccv_nnc_tensor_t* gpu_q_tensor = 0;
			ccv_nnc_tensor_t* gpu_k_tensor = 0;
			ccv_nnc_tensor_t* gpu_v_tensor = 0;
			ccv_nnc_tensor_t* gpu_o_tensor = 0;
			if (datatype == CCV_16F)
			{
				ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			} else if (datatype == CCV_16BF) {
				ccv_float_to_bfloat(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_bfloat(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_bfloat(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, H, D), 0);
			} else {
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
			ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
			gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

			const int count = B * R * H * D;
			float* const cpu_f32 = (float*)ccmalloc(sizeof(float) * count);
			float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
			memcpy(cpu_f32, o_tensor->data.f32, sizeof(float) * count);
			if (datatype == CCV_16F)
				ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
			else if (datatype == CCV_16BF)
				ccv_bfloat_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
			else
				memcpy(gpu_f32, copy_of_gpu_o_tensor->data.f32, sizeof(float) * count);
			float max_relative_diff = 0;
			int max_diff_idx = 0;
			for (int i = 0; i < count; ++i)
			{
				const float denom = fmaxf(fmaxf(fabsf(cpu_f32[i]), fabsf(gpu_f32[i])), 1.0f);
				const float relative_diff = fabsf(cpu_f32[i] - gpu_f32[i]) / denom;
				if (relative_diff > max_relative_diff)
					max_relative_diff = relative_diff, max_diff_idx = i;
			}
			REQUIRE(max_relative_diff <= tolerances[datatype_idx], "quantized attention result should match CPU reference for dtype=%s D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], D, max_relative_diff, max_diff_idx, cpu_f32[max_diff_idx], gpu_f32[max_diff_idx]);

			ccfree(cpu_f32);
			ccfree(gpu_f32);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
		}
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
	}
}

TEST_CASE("scaled dot product attention with quantized NA mps and zero mask")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 128;
	const int C = 64;
	const int H = 16;
	const int D = 128;
	const float scale = 1.0 / sqrt((float)D);
	const int q_count = B * R * H * D;
	const int kv_count = B * C * H * D;
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const zero_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const causal_visible_zero_mask = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 23);
	for (int i = 0; i < q_count; ++i)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	memset(zero_mask->data.f16, 0, sizeof(uint16_t) * R * C);
	float* const causal_visible_zero_mask_f32 = (float*)ccmalloc(sizeof(float) * R * C);
	for (int i = 0; i < R; ++i)
		for (int j = 0; j < C; ++j)
			causal_visible_zero_mask_f32[i * C + j] = (j <= i + C - R) ? 0 : 1;
	ccv_float_to_half_precision(causal_visible_zero_mask_f32, (uint16_t*)causal_visible_zero_mask->data.f16, R * C);
	ccfree(causal_visible_zero_mask_f32);
	ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
	ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);

	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_zero_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const gpu_causal_visible_zero_mask = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, zero_mask, causal_visible_zero_mask), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_zero_mask, gpu_causal_visible_zero_mask), 0);
	for (int is_causal = 0; is_causal <= 1; ++is_causal)
	{
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_masked_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_masked_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		ccv_nnc_tensor_t* const gpu_mask_tensor = is_causal ? gpu_causal_visible_zero_mask : gpu_zero_mask;
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask_tensor), TENSOR_LIST(gpu_masked_o_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor, gpu_masked_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor, copy_of_gpu_masked_o_tensor), 0);

		const int count = B * R * H * D;
		float* const unmasked_f32 = (float*)ccmalloc(sizeof(float) * count);
		float* const masked_f32 = (float*)ccmalloc(sizeof(float) * count);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, unmasked_f32, count);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_masked_o_tensor->data.f16, masked_f32, count);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, masked_f32, unmasked_f32, count, 1e-3, "zero-mask quantized attention should match unmasked quantized attention when causal=%d", is_causal);
		ccfree(masked_f32);
		ccfree(unmasked_f32);
		ccv_nnc_tensor_free(copy_of_gpu_masked_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_masked_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
	}
	ccv_nnc_tensor_free(gpu_causal_visible_zero_mask);
	ccv_nnc_tensor_free(gpu_zero_mask);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(causal_visible_zero_mask);
	ccv_nnc_tensor_free(zero_mask);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("scaled dot product attention with quantized NA mps and numeric mask")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 64;
	const int C = 64;
	const int H = 8;
	const int D = 128;
	const float scale = 1.0 / sqrt((float)D);
	const int q_count = B * R * H * D;
	const int kv_count = B * C * H * D;
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const mask_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1, R, C), 0);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const mask_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 1, 1, R, C), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 29);
	for (int i = 0; i < q_count; ++i)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < kv_count; ++i)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	for (int i = 0; i < R; ++i)
		for (int j = 0; j < C; ++j)
			mask_tensor->data.f32[i * C + j] = ((i * 13 + j * 7) % 11 == 0) ? -0.75 : (((i + j) % 5 == 0) ? 0.125 : 0);
	ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
	ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(mask_tensor->data.f32, (uint16_t*)mask_tensor_f16->data.f16, R * C);

	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_mask_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, 1, R, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, mask_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask_tensor), 0);
	for (int is_causal = 0; is_causal <= 1; ++is_causal)
	{
		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, mask_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
		gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_mask_tensor), TENSOR_LIST(gpu_o_tensor), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		const int count = B * R * H * D;
		float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
		ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
		float max_relative_diff = 0;
		int max_diff_idx = 0;
		for (int i = 0; i < count; ++i)
		{
			const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
			const float relative_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]) / denom;
			if (relative_diff > max_relative_diff)
				max_relative_diff = relative_diff, max_diff_idx = i;
		}
		REQUIRE(max_relative_diff <= 5e-2, "numeric-mask quantized attention should match CPU reference when causal=%d (max relative diff %g at %d: %g vs %g)", is_causal, max_relative_diff, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);
		ccfree(gpu_f32);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(o_tensor);
	}
	ccv_nnc_tensor_free(gpu_mask_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(mask_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(mask_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(q_tensor);
}

TEST_CASE("scaled dot product attention with quantized causal NA mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int H = 8;
	const int Ds[] = { 64, 256 };
	const int is_causal = 1;
	const float tolerance = 4e-3;
	const float empty_row_tolerance = 1e-6;
	const int shapes[][2] = {
		{ 17, 64 }, // R remainder, R < C.
		{ 47, 41 }, // R and C remainders, R > C.
		{ 65, 37 }, // R > C with many empty causal rows.
		{ 79, 80 }, // R and C remainders, R < C by one.
		{ 47, 47 }, // R == C, remainder-only traversal.
		{ 65, 65 }, // R == C, full and remainder traversal.
		{ 83, 97 }, // Both R and C arbitrary, R < C.
		{ 97, 83 }, // Both R and C arbitrary, R > C.
		{ 128, 64 }, // Existing covered shape.
	};
	for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
	{
		const int D = Ds[d_idx];
		const float scale = 1.0 / sqrt((float)D);
		for (int shape_idx = 0; shape_idx < (int)(sizeof(shapes) / sizeof(shapes[0])); ++shape_idx)
		{
			const int R = shapes[shape_idx][0];
			const int C = shapes[shape_idx][1];
			const int q_count = B * R * H * D;
			const int kv_count = B * C * H * D;
			ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			// Zero Q/K makes causal attention average exactly the visible V columns.
			memset(q_tensor->data.f32, 0, sizeof(float) * q_count);
			memset(k_tensor->data.f32, 0, sizeof(float) * kv_count);
			memset(v_tensor->data.f32, 0, sizeof(float) * kv_count);
			for (int c = 0; c < C; ++c)
				for (int h = 0; h < H; ++h)
				{
					const int sentinel_d = (c * 17 + h * 13) % D;
					v_tensor->data.f32[(c * H + h) * D + sentinel_d] = 1;
				}
			ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
			ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
			ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
			ccv_half_precision_to_float((uint16_t*)q_tensor_f16->data.f16, q_tensor->data.f32, q_count);
			ccv_half_precision_to_float((uint16_t*)k_tensor_f16->data.f16, k_tensor->data.f32, kv_count);
			ccv_half_precision_to_float((uint16_t*)v_tensor_f16->data.f16, v_tensor->data.f32, kv_count);

			ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

			ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
			ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal);
			gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

			const int count = B * R * H * D;
			float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
			ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
			float max_relative_diff = 0;
			float max_abs_diff = 0;
			int max_diff_idx = 0;
			for (int i = 0; i < count; ++i)
			{
				REQUIRE(isfinite(gpu_f32[i]), "quantized causal attention output should stay finite for R=%d C=%d D=%d at %d", R, C, D, i);
				const int r = (i / (H * D)) % R;
				if (R > C && r < R - C)
					REQUIRE(fabsf(gpu_f32[i]) <= empty_row_tolerance, "quantized causal attention should zero empty leading row for R=%d C=%d D=%d at r=%d idx=%d: GPU %g", R, C, D, r, i, gpu_f32[i]);
				const float denom = fmaxf(fmaxf(fabsf(o_tensor->data.f32[i]), fabsf(gpu_f32[i])), 1.0f);
				const float abs_diff = fabsf(o_tensor->data.f32[i] - gpu_f32[i]);
				const float relative_diff = abs_diff / denom;
				if (relative_diff > max_relative_diff)
					max_relative_diff = relative_diff, max_abs_diff = abs_diff, max_diff_idx = i;
			}
			const int max_diff_d = max_diff_idx % D;
			const int max_diff_h = (max_diff_idx / D) % H;
			const int max_diff_r = (max_diff_idx / (H * D)) % R;
			REQUIRE(max_abs_diff <= tolerance, "quantized causal attention result should match CPU reference for R=%d C=%d D=%d (max abs %g relative %g at r=%d h=%d d=%d idx=%d: CPU %g GPU %g)", R, C, D, max_abs_diff, max_relative_diff, max_diff_r, max_diff_h, max_diff_d, max_diff_idx, o_tensor->data.f32[max_diff_idx], gpu_f32[max_diff_idx]);

			ccfree(gpu_f32);
			ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
			ccv_nnc_tensor_free(o_tensor);
			ccv_nnc_tensor_free(q_tensor);
			ccv_nnc_tensor_free(k_tensor);
			ccv_nnc_tensor_free(v_tensor);
			ccv_nnc_tensor_free(q_tensor_f16);
			ccv_nnc_tensor_free(k_tensor_f16);
			ccv_nnc_tensor_free(v_tensor_f16);
		}
	}
}

TEST_CASE("scaled dot product attention with quantized NA mps batched")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 3;
	const int R = 128;
	const int C = 128;
	const int H = 8;
	const int Ds[] = { 64, 128 };
	const int datatypes[] = { CCV_16F, CCV_16BF, CCV_32F };
	const float tolerances[] = { 2e-2, 3e-2, 2e-2 };
	const char* datatype_names[] = { "16F", "16BF", "32F" };
	for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
	{
		const int D = Ds[d_idx];
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		const int q_count = B * R * H * D;
		const int kv_count = B * C * H * D;
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 101 + d_idx);
		for (int i = 0; i < q_count; ++i)
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		for (int i = 0; i < kv_count; ++i)
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		for (int i = 0; i < kv_count; ++i)
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_cmd_t cpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

		for (int datatype_idx = 0; datatype_idx < 3; ++datatype_idx)
		{
			const int datatype = datatypes[datatype_idx];
			ccv_nnc_tensor_t* q_input = q_tensor;
			ccv_nnc_tensor_t* k_input = k_tensor;
			ccv_nnc_tensor_t* v_input = v_tensor;
			ccv_nnc_tensor_t* copy_of_gpu_o_tensor = 0;
			ccv_nnc_tensor_t* gpu_q_tensor = 0;
			ccv_nnc_tensor_t* gpu_k_tensor = 0;
			ccv_nnc_tensor_t* gpu_v_tensor = 0;
			ccv_nnc_tensor_t* gpu_o_tensor = 0;
			if (datatype == CCV_16F)
			{
				ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			} else if (datatype == CCV_16BF) {
				ccv_float_to_bfloat(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_bfloat(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_bfloat(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, H, D), 0);
			} else {
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
			ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
			gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
			ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

			const int count = B * R * H * D;
			float* const cpu_f32 = (float*)ccmalloc(sizeof(float) * count);
			float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
			memcpy(cpu_f32, o_tensor->data.f32, sizeof(float) * count);
			if (datatype == CCV_16F)
				ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
			else if (datatype == CCV_16BF)
				ccv_bfloat_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
			else
				memcpy(gpu_f32, copy_of_gpu_o_tensor->data.f32, sizeof(float) * count);
			float max_relative_diff = 0;
			int max_diff_idx = 0;
			for (int i = 0; i < count; ++i)
			{
				const float denom = fmaxf(fmaxf(fabsf(cpu_f32[i]), fabsf(gpu_f32[i])), 1.0f);
				const float relative_diff = fabsf(cpu_f32[i] - gpu_f32[i]) / denom;
				if (relative_diff > max_relative_diff)
					max_relative_diff = relative_diff, max_diff_idx = i;
			}
			REQUIRE(max_relative_diff <= tolerances[datatype_idx], "quantized batched attention result should match CPU reference for dtype=%s D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], D, max_relative_diff, max_diff_idx, cpu_f32[max_diff_idx], gpu_f32[max_diff_idx]);

			ccfree(cpu_f32);
			ccfree(gpu_f32);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
		}
		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
	}
}

TEST_CASE("scaled dot product attention with quantized NA mps for non-multiple-of-64 sequence")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 128;
	const int H = 24;
	const int Cs[] = { 130, 224 };
	const int Ds[] = { 128, 130, 224 };
	const int datatypes[] = { CCV_16F, CCV_16BF, CCV_32F };
	const float tolerances[] = { 4e-2, 5e-2, 4e-2 };
	const char* datatype_names[] = { "16F", "16BF", "32F" };
	for (int c_idx = 0; c_idx < (int)(sizeof(Cs) / sizeof(Cs[0])); ++c_idx)
	{
		const int C = Cs[c_idx];
		for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
		{
			const int D = Ds[d_idx];
			const float scale = 1.0 / sqrt((float)D);

			ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			const int q_count = B * R * H * D;
			const int kv_count = B * C * H * D;
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 211 + c_idx * 17 + d_idx);
			for (int i = 0; i < q_count; ++i)
				q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			for (int i = 0; i < kv_count; ++i)
				k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			for (int i = 0; i < kv_count; ++i)
				v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;

			ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_cmd_t cpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
			ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);

			for (int datatype_idx = 0; datatype_idx < 3; ++datatype_idx)
			{
				const int datatype = datatypes[datatype_idx];
				ccv_nnc_tensor_t* q_input = q_tensor;
				ccv_nnc_tensor_t* k_input = k_tensor;
				ccv_nnc_tensor_t* v_input = v_tensor;
				ccv_nnc_tensor_t* copy_of_gpu_o_tensor = 0;
				ccv_nnc_tensor_t* gpu_q_tensor = 0;
				ccv_nnc_tensor_t* gpu_k_tensor = 0;
				ccv_nnc_tensor_t* gpu_v_tensor = 0;
				ccv_nnc_tensor_t* gpu_o_tensor = 0;
				if (datatype == CCV_16F)
				{
					ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
					ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
					ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
					q_input = q_tensor_f16;
					k_input = k_tensor_f16;
					v_input = v_tensor_f16;
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
				} else if (datatype == CCV_16BF) {
					ccv_float_to_bfloat(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
					ccv_float_to_bfloat(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
					ccv_float_to_bfloat(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
					q_input = q_tensor_f16;
					k_input = k_tensor_f16;
					v_input = v_tensor_f16;
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, H, D), 0);
				} else {
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);
				ccv_nnc_cmd_t gpu_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
				gpu_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
				ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), TENSOR_LIST(gpu_o_tensor), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor), 0);

				const int count = B * R * H * D;
				float* const cpu_f32 = (float*)ccmalloc(sizeof(float) * count);
				float* const gpu_f32 = (float*)ccmalloc(sizeof(float) * count);
				memcpy(cpu_f32, o_tensor->data.f32, sizeof(float) * count);
				if (datatype == CCV_16F)
					ccv_half_precision_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
				else if (datatype == CCV_16BF)
					ccv_bfloat_to_float((uint16_t*)copy_of_gpu_o_tensor->data.f16, gpu_f32, count);
				else
					memcpy(gpu_f32, copy_of_gpu_o_tensor->data.f32, sizeof(float) * count);
				float max_relative_diff = 0;
				int max_diff_idx = 0;
				for (int i = 0; i < count; ++i)
				{
					const float denom = fmaxf(fmaxf(fabsf(cpu_f32[i]), fabsf(gpu_f32[i])), 1.0f);
					const float relative_diff = fabsf(cpu_f32[i] - gpu_f32[i]) / denom;
					if (relative_diff > max_relative_diff)
						max_relative_diff = relative_diff, max_diff_idx = i;
				}
				REQUIRE(max_relative_diff <= tolerances[datatype_idx], "quantized attention result should match CPU reference for dtype=%s C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], C, D, max_relative_diff, max_diff_idx, cpu_f32[max_diff_idx], gpu_f32[max_diff_idx]);

				ccfree(cpu_f32);
				ccfree(gpu_f32);
				ccv_nnc_tensor_free(gpu_o_tensor);
				ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
				ccv_nnc_tensor_free(gpu_q_tensor);
				ccv_nnc_tensor_free(gpu_k_tensor);
				ccv_nnc_tensor_free(gpu_v_tensor);
			}
			ccv_nnc_tensor_free(o_tensor);
			ccv_nnc_tensor_free(q_tensor);
			ccv_nnc_tensor_free(k_tensor);
			ccv_nnc_tensor_free(v_tensor);
			ccv_nnc_tensor_free(q_tensor_f16);
			ccv_nnc_tensor_free(k_tensor_f16);
			ccv_nnc_tensor_free(v_tensor_f16);
		}
	}
}

TEST_CASE("scaled dot product attention gradient with quantized NA mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int B = 2;
	const int R = 128;
	const int C = 128;
	const int H = 8;
	const int Ds[] = { 64, 80, 96, 128 };
	const int datatypes[] = { CCV_16F, CCV_16BF, CCV_32F };
	const char* datatype_names[] = { "16F", "16BF", "32F" };
	const float dq_tolerances[] = { 8e-2, 8e-2, 8e-2 };
	const float dk_tolerances[] = { 1e-1, 1e-1, 1e-1 };
	const float dv_tolerances[] = { 8e-2, 8e-2, 8e-2 };
	for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
	{
		const int D = Ds[d_idx];
		const int q_count = B * R * H * D;
		const int kv_count = B * C * H * D;
		const float scale = 1.0 / sqrt((float)D);
		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
		dsfmt_t dsfmt;
		dsfmt_init_gen_rand(&dsfmt, 181 + d_idx);
		for (int i = 0; i < q_count; ++i)
		{
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		}
		for (int i = 0; i < kv_count; ++i)
		{
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);

		for (int datatype_idx = 0; datatype_idx < 3; ++datatype_idx)
		{
			const int datatype = datatypes[datatype_idx];
			ccv_nnc_tensor_t* q_input = q_tensor;
			ccv_nnc_tensor_t* k_input = k_tensor;
			ccv_nnc_tensor_t* v_input = v_tensor;
			ccv_nnc_tensor_t* do_input = do_tensor;
			ccv_nnc_tensor_t* gpu_q_tensor = 0;
			ccv_nnc_tensor_t* gpu_k_tensor = 0;
			ccv_nnc_tensor_t* gpu_v_tensor = 0;
			ccv_nnc_tensor_t* gpu_do_tensor = 0;
			ccv_nnc_tensor_t* gpu_o_tensor = 0;
			ccv_nnc_tensor_t* gpu_dq_tensor = 0;
			ccv_nnc_tensor_t* gpu_dk_tensor = 0;
			ccv_nnc_tensor_t* gpu_dv_tensor = 0;
			ccv_nnc_tensor_t* copy_of_gpu_dq_tensor = 0;
			ccv_nnc_tensor_t* copy_of_gpu_dk_tensor = 0;
			ccv_nnc_tensor_t* copy_of_gpu_dv_tensor = 0;
			if (datatype == CCV_16F)
			{
				ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				ccv_float_to_half_precision(do_tensor->data.f32, (uint16_t*)do_tensor_f16->data.f16, q_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				do_input = do_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
				gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
				copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
				copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
				copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			} else if (datatype == CCV_16BF) {
				ccv_float_to_bfloat(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
				ccv_float_to_bfloat(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
				ccv_float_to_bfloat(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
				ccv_float_to_bfloat(do_tensor->data.f32, (uint16_t*)do_tensor_f16->data.f16, q_count);
				q_input = q_tensor_f16;
				k_input = k_tensor_f16;
				v_input = v_tensor_f16;
				do_input = do_tensor_f16;
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
				gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
				copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, H, D), 0);
				copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, H, D), 0);
				copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, H, D), 0);
			} else {
				gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
				gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
				copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
				copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
				copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			}
			ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, R), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input, do_input), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);
			ccv_nnc_cmd_t gpu_forw_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
			gpu_forw_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
			ccv_nnc_cmd_exec(gpu_forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);
			ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
			ccv_nnc_cmd_t gpu_back_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0);
			gpu_back_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
			gpu_back_cmd.info.scaled_dot_product_attention.deterministic = 0;
			ccv_nnc_cmd_exec(gpu_back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

			float* const dq_cpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
			float* const dk_cpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
			float* const dv_cpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
			float* const dq_gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
			float* const dk_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
			float* const dv_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
			memcpy(dq_cpu_f32, dq_tensor->data.f32, sizeof(float) * q_count);
			memcpy(dk_cpu_f32, dk_tensor->data.f32, sizeof(float) * kv_count);
			memcpy(dv_cpu_f32, dv_tensor->data.f32, sizeof(float) * kv_count);
			if (datatype == CCV_16F)
			{
				ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dq_tensor->data.f16, dq_gpu_f32, q_count);
				ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dk_tensor->data.f16, dk_gpu_f32, kv_count);
				ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dv_tensor->data.f16, dv_gpu_f32, kv_count);
			} else if (datatype == CCV_16BF) {
				ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dq_tensor->data.f16, dq_gpu_f32, q_count);
				ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dk_tensor->data.f16, dk_gpu_f32, kv_count);
				ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dv_tensor->data.f16, dv_gpu_f32, kv_count);
			} else {
				memcpy(dq_gpu_f32, copy_of_gpu_dq_tensor->data.f32, sizeof(float) * q_count);
				memcpy(dk_gpu_f32, copy_of_gpu_dk_tensor->data.f32, sizeof(float) * kv_count);
				memcpy(dv_gpu_f32, copy_of_gpu_dv_tensor->data.f32, sizeof(float) * kv_count);
			}
			float dq_max_relative_diff = 0;
			float dk_max_relative_diff = 0;
			float dv_max_relative_diff = 0;
			int dq_max_diff_idx = 0;
			int dk_max_diff_idx = 0;
			int dv_max_diff_idx = 0;
			for (int i = 0; i < q_count; ++i)
			{
				const float denom = fmaxf(fmaxf(fabsf(dq_cpu_f32[i]), fabsf(dq_gpu_f32[i])), 1.0f);
				const float relative_diff = fabsf(dq_cpu_f32[i] - dq_gpu_f32[i]) / denom;
				if (relative_diff > dq_max_relative_diff)
					dq_max_relative_diff = relative_diff, dq_max_diff_idx = i;
			}
			for (int i = 0; i < kv_count; ++i)
			{
				float denom = fmaxf(fmaxf(fabsf(dk_cpu_f32[i]), fabsf(dk_gpu_f32[i])), 1.0f);
				float relative_diff = fabsf(dk_cpu_f32[i] - dk_gpu_f32[i]) / denom;
				if (relative_diff > dk_max_relative_diff)
					dk_max_relative_diff = relative_diff, dk_max_diff_idx = i;
				denom = fmaxf(fmaxf(fabsf(dv_cpu_f32[i]), fabsf(dv_gpu_f32[i])), 1.0f);
				relative_diff = fabsf(dv_cpu_f32[i] - dv_gpu_f32[i]) / denom;
				if (relative_diff > dv_max_relative_diff)
					dv_max_relative_diff = relative_diff, dv_max_diff_idx = i;
			}
			REQUIRE(dq_max_relative_diff <= dq_tolerances[datatype_idx], "quantized attention dQ should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dq_max_relative_diff, dq_max_diff_idx, dq_cpu_f32[dq_max_diff_idx], dq_gpu_f32[dq_max_diff_idx]);
			REQUIRE(dk_max_relative_diff <= dk_tolerances[datatype_idx], "quantized attention dK should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dk_max_relative_diff, dk_max_diff_idx, dk_cpu_f32[dk_max_diff_idx], dk_gpu_f32[dk_max_diff_idx]);
			REQUIRE(dv_max_relative_diff <= dv_tolerances[datatype_idx], "quantized attention dV should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dv_max_relative_diff, dv_max_diff_idx, dv_cpu_f32[dv_max_diff_idx], dv_gpu_f32[dv_max_diff_idx]);

			ccfree(dq_cpu_f32);
			ccfree(dk_cpu_f32);
			ccfree(dv_cpu_f32);
			ccfree(dq_gpu_f32);
			ccfree(dk_gpu_f32);
			ccfree(dv_gpu_f32);
			ccv_nnc_tensor_free(gpu_q_tensor);
			ccv_nnc_tensor_free(gpu_k_tensor);
			ccv_nnc_tensor_free(gpu_v_tensor);
			ccv_nnc_tensor_free(gpu_do_tensor);
			ccv_nnc_tensor_free(gpu_o_tensor);
			ccv_nnc_tensor_free(gpu_dq_tensor);
			ccv_nnc_tensor_free(gpu_dk_tensor);
			ccv_nnc_tensor_free(gpu_dv_tensor);
			ccv_nnc_tensor_free(gpu_softmax_lse);
			ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
			ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
			ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		}

		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(do_tensor_f16);
	}
}

TEST_CASE("scaled dot product attention gradient with quantized NA mps for rectangular and edge sequence lengths")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	typedef struct {
		int R;
		int C;
	} qna_backward_shape_t;
	const int B = 1;
	const int H = 8;
	const int Ds[] = { 64, 128 };
	const qna_backward_shape_t shapes[] = {
		{ .R = 32, .C = 64 },
		{ .R = 40, .C = 72 },
		{ .R = 80, .C = 64 },
		{ .R = 96, .C = 88 },
		{ .R = 64, .C = 192 },
		{ .R = 144, .C = 64 },
	};
	const int datatypes[] = { CCV_16F, CCV_16BF, CCV_32F };
	const char* datatype_names[] = { "16F", "16BF", "32F" };
	const float dq_tolerances[] = { 8e-2, 8e-2, 8e-2 };
	const float dk_tolerances[] = { 1e-1, 1e-1, 1e-1 };
	const float dv_tolerances[] = { 8e-2, 8e-2, 8e-2 };
	for (int shape_idx = 0; shape_idx < (int)(sizeof(shapes) / sizeof(shapes[0])); ++shape_idx)
	{
		const int R = shapes[shape_idx].R;
		const int C = shapes[shape_idx].C;
		for (int d_idx = 0; d_idx < (int)(sizeof(Ds) / sizeof(Ds[0])); ++d_idx)
		{
			const int D = Ds[d_idx];
			const int q_count = B * R * H * D;
			const int kv_count = B * C * H * D;
			const float scale = 1.0 / sqrt((float)D);
			ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
			ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
			dsfmt_t dsfmt;
			dsfmt_init_gen_rand(&dsfmt, 281 + shape_idx * 17 + d_idx);
			for (int i = 0; i < q_count; ++i)
			{
				q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
				do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			}
			for (int i = 0; i < kv_count; ++i)
			{
				k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
				v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
			}
			ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);

			for (int datatype_idx = 0; datatype_idx < 3; ++datatype_idx)
			{
				const int datatype = datatypes[datatype_idx];
				ccv_nnc_tensor_t* q_input = q_tensor;
				ccv_nnc_tensor_t* k_input = k_tensor;
				ccv_nnc_tensor_t* v_input = v_tensor;
				ccv_nnc_tensor_t* do_input = do_tensor;
				ccv_nnc_tensor_t* gpu_q_tensor = 0;
				ccv_nnc_tensor_t* gpu_k_tensor = 0;
				ccv_nnc_tensor_t* gpu_v_tensor = 0;
				ccv_nnc_tensor_t* gpu_do_tensor = 0;
				ccv_nnc_tensor_t* gpu_o_tensor = 0;
				ccv_nnc_tensor_t* gpu_dq_tensor = 0;
				ccv_nnc_tensor_t* gpu_dk_tensor = 0;
				ccv_nnc_tensor_t* gpu_dv_tensor = 0;
				ccv_nnc_tensor_t* copy_of_gpu_dq_tensor = 0;
				ccv_nnc_tensor_t* copy_of_gpu_dk_tensor = 0;
				ccv_nnc_tensor_t* copy_of_gpu_dv_tensor = 0;
				if (datatype == CCV_16F)
				{
					ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
					ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
					ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
					ccv_float_to_half_precision(do_tensor->data.f32, (uint16_t*)do_tensor_f16->data.f16, q_count);
					q_input = q_tensor_f16;
					k_input = k_tensor_f16;
					v_input = v_tensor_f16;
					do_input = do_tensor_f16;
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
					gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
					copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
					copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
					copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
				} else if (datatype == CCV_16BF) {
					ccv_float_to_bfloat(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
					ccv_float_to_bfloat(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
					ccv_float_to_bfloat(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
					ccv_float_to_bfloat(do_tensor->data.f32, (uint16_t*)do_tensor_f16->data.f16, q_count);
					q_input = q_tensor_f16;
					k_input = k_tensor_f16;
					v_input = v_tensor_f16;
					do_input = do_tensor_f16;
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, H, D), 0);
					gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, H, D), 0);
					copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, H, D), 0);
					copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, H, D), 0);
					copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, H, D), 0);
				} else {
					gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
					gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
					copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
					copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
					copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
				}
				ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, R), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_input, k_input, v_input, do_input), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);
				ccv_nnc_cmd_t gpu_forw_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
				gpu_forw_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
				ccv_nnc_cmd_exec(gpu_forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);
				ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
				ccv_nnc_cmd_t gpu_back_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0);
				gpu_back_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
				gpu_back_cmd.info.scaled_dot_product_attention.deterministic = 0;
				ccv_nnc_cmd_exec(gpu_back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

				float* const dq_cpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
				float* const dk_cpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
				float* const dv_cpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
				float* const dq_gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
				float* const dk_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
				float* const dv_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
				memcpy(dq_cpu_f32, dq_tensor->data.f32, sizeof(float) * q_count);
				memcpy(dk_cpu_f32, dk_tensor->data.f32, sizeof(float) * kv_count);
				memcpy(dv_cpu_f32, dv_tensor->data.f32, sizeof(float) * kv_count);
				if (datatype == CCV_16F)
				{
					ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dq_tensor->data.f16, dq_gpu_f32, q_count);
					ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dk_tensor->data.f16, dk_gpu_f32, kv_count);
					ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dv_tensor->data.f16, dv_gpu_f32, kv_count);
				} else if (datatype == CCV_16BF) {
					ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dq_tensor->data.f16, dq_gpu_f32, q_count);
					ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dk_tensor->data.f16, dk_gpu_f32, kv_count);
					ccv_bfloat_to_float((uint16_t*)copy_of_gpu_dv_tensor->data.f16, dv_gpu_f32, kv_count);
				} else {
					memcpy(dq_gpu_f32, copy_of_gpu_dq_tensor->data.f32, sizeof(float) * q_count);
					memcpy(dk_gpu_f32, copy_of_gpu_dk_tensor->data.f32, sizeof(float) * kv_count);
					memcpy(dv_gpu_f32, copy_of_gpu_dv_tensor->data.f32, sizeof(float) * kv_count);
				}
				float dq_max_relative_diff = 0;
				float dk_max_relative_diff = 0;
				float dv_max_relative_diff = 0;
				int dq_max_diff_idx = 0;
				int dk_max_diff_idx = 0;
				int dv_max_diff_idx = 0;
				for (int i = 0; i < q_count; ++i)
				{
					const float denom = fmaxf(fmaxf(fabsf(dq_cpu_f32[i]), fabsf(dq_gpu_f32[i])), 1.0f);
					const float relative_diff = fabsf(dq_cpu_f32[i] - dq_gpu_f32[i]) / denom;
					if (relative_diff > dq_max_relative_diff)
						dq_max_relative_diff = relative_diff, dq_max_diff_idx = i;
				}
				for (int i = 0; i < kv_count; ++i)
				{
					float denom = fmaxf(fmaxf(fabsf(dk_cpu_f32[i]), fabsf(dk_gpu_f32[i])), 1.0f);
					float relative_diff = fabsf(dk_cpu_f32[i] - dk_gpu_f32[i]) / denom;
					if (relative_diff > dk_max_relative_diff)
						dk_max_relative_diff = relative_diff, dk_max_diff_idx = i;
					denom = fmaxf(fmaxf(fabsf(dv_cpu_f32[i]), fabsf(dv_gpu_f32[i])), 1.0f);
					relative_diff = fabsf(dv_cpu_f32[i] - dv_gpu_f32[i]) / denom;
					if (relative_diff > dv_max_relative_diff)
						dv_max_relative_diff = relative_diff, dv_max_diff_idx = i;
				}
				REQUIRE(dq_max_relative_diff <= dq_tolerances[datatype_idx], "quantized attention dQ should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dq_max_relative_diff, dq_max_diff_idx, dq_cpu_f32[dq_max_diff_idx], dq_gpu_f32[dq_max_diff_idx]);
				REQUIRE(dk_max_relative_diff <= dk_tolerances[datatype_idx], "quantized attention dK should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dk_max_relative_diff, dk_max_diff_idx, dk_cpu_f32[dk_max_diff_idx], dk_gpu_f32[dk_max_diff_idx]);
				REQUIRE(dv_max_relative_diff <= dv_tolerances[datatype_idx], "quantized attention dV should match CPU reference for dtype=%s R=%d C=%d D=%d (max relative diff %g at %d: %g vs %g)", datatype_names[datatype_idx], R, C, D, dv_max_relative_diff, dv_max_diff_idx, dv_cpu_f32[dv_max_diff_idx], dv_gpu_f32[dv_max_diff_idx]);

				ccfree(dq_cpu_f32);
				ccfree(dk_cpu_f32);
				ccfree(dv_cpu_f32);
				ccfree(dq_gpu_f32);
				ccfree(dk_gpu_f32);
				ccfree(dv_gpu_f32);
				ccv_nnc_tensor_free(gpu_q_tensor);
				ccv_nnc_tensor_free(gpu_k_tensor);
				ccv_nnc_tensor_free(gpu_v_tensor);
				ccv_nnc_tensor_free(gpu_do_tensor);
				ccv_nnc_tensor_free(gpu_o_tensor);
				ccv_nnc_tensor_free(gpu_dq_tensor);
				ccv_nnc_tensor_free(gpu_dk_tensor);
				ccv_nnc_tensor_free(gpu_dv_tensor);
				ccv_nnc_tensor_free(gpu_softmax_lse);
				ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
				ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
				ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
			}

			ccv_nnc_tensor_free(q_tensor);
			ccv_nnc_tensor_free(k_tensor);
			ccv_nnc_tensor_free(v_tensor);
			ccv_nnc_tensor_free(do_tensor);
			ccv_nnc_tensor_free(dq_tensor);
			ccv_nnc_tensor_free(dk_tensor);
			ccv_nnc_tensor_free(dv_tensor);
			ccv_nnc_tensor_free(q_tensor_f16);
			ccv_nnc_tensor_free(k_tensor_f16);
			ccv_nnc_tensor_free(v_tensor_f16);
			ccv_nnc_tensor_free(do_tensor_f16);
		}
	}
}

TEST_CASE("scaled dot product attention gradient with quantized NA mps on 1536 square surface")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int R = 1536;
	const int C = 1536;
	const int H = 24;
	const int D = 128;
	const int q_count = B * R * H * D;
	const int kv_count = B * C * H * D;
	const float scale = 1.0 / sqrt((float)D);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4177);
	for (int i = 0; i < q_count; ++i)
	{
		// Use a stronger shared Q / K signal on this surface so QK^T produces
		// sharper rows than the fully diffuse random-input case.
		const float q = 2.f * (dsfmt_genrand_open_close(&dsfmt) - 0.5f);
		q_tensor->data.f32[i] = q;
		k_tensor->data.f32[i] = q + 0.125f * (dsfmt_genrand_open_close(&dsfmt) - 0.5f);
		do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) - 0.5;
	}
	ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);

	ccv_float_to_half_precision(q_tensor->data.f32, (uint16_t*)q_tensor_f16->data.f16, q_count);
	ccv_float_to_half_precision(k_tensor->data.f32, (uint16_t*)k_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(v_tensor->data.f32, (uint16_t*)v_tensor_f16->data.f16, kv_count);
	ccv_float_to_half_precision(do_tensor->data.f32, (uint16_t*)do_tensor_f16->data.f16, q_count);

	ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, R), 0);
	ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);
	ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, H, D), 0);

	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);
	ccv_nnc_cmd_t gpu_forw_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0);
	gpu_forw_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
	ccv_nnc_cmd_exec(gpu_forw_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);
	ccv_nnc_cmd_exec(CMD_SET_FORWARD(0), ccv_nnc_no_hint, 0, TENSOR_LIST(), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
	ccv_nnc_cmd_t gpu_back_cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0);
	gpu_back_cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_16F | CCV_NNC_GEMM_8I;
	gpu_back_cmd.info.scaled_dot_product_attention.deterministic = 0;
	ccv_nnc_cmd_exec(gpu_back_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

	float* const dq_gpu_f32 = (float*)ccmalloc(sizeof(float) * q_count);
	float* const dk_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
	float* const dv_gpu_f32 = (float*)ccmalloc(sizeof(float) * kv_count);
	ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dq_tensor->data.f16, dq_gpu_f32, q_count);
	ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dk_tensor->data.f16, dk_gpu_f32, kv_count);
	ccv_half_precision_to_float((uint16_t*)copy_of_gpu_dv_tensor->data.f16, dv_gpu_f32, kv_count);

	float dq_max_relative_diff = 0;
	float dk_max_relative_diff = 0;
	float dv_max_relative_diff = 0;
	float dq_cpu_max_abs = 0;
	float dq_gpu_max_abs = 0;
	float dk_cpu_max_abs = 0;
	float dk_gpu_max_abs = 0;
	float dv_cpu_max_abs = 0;
	float dv_gpu_max_abs = 0;
	int dq_max_diff_idx = 0;
	int dk_max_diff_idx = 0;
	int dv_max_diff_idx = 0;
	for (int i = 0; i < q_count; ++i)
	{
		dq_cpu_max_abs = fmaxf(dq_cpu_max_abs, fabsf(dq_tensor->data.f32[i]));
		dq_gpu_max_abs = fmaxf(dq_gpu_max_abs, fabsf(dq_gpu_f32[i]));
		const float denom = fmaxf(fmaxf(fabsf(dq_tensor->data.f32[i]), fabsf(dq_gpu_f32[i])), 1.0f);
		const float relative_diff = fabsf(dq_tensor->data.f32[i] - dq_gpu_f32[i]) / denom;
		if (relative_diff > dq_max_relative_diff)
			dq_max_relative_diff = relative_diff, dq_max_diff_idx = i;
	}
	for (int i = 0; i < kv_count; ++i)
	{
		dk_cpu_max_abs = fmaxf(dk_cpu_max_abs, fabsf(dk_tensor->data.f32[i]));
		dk_gpu_max_abs = fmaxf(dk_gpu_max_abs, fabsf(dk_gpu_f32[i]));
		float denom = fmaxf(fmaxf(fabsf(dk_tensor->data.f32[i]), fabsf(dk_gpu_f32[i])), 1.0f);
		float relative_diff = fabsf(dk_tensor->data.f32[i] - dk_gpu_f32[i]) / denom;
		if (relative_diff > dk_max_relative_diff)
			dk_max_relative_diff = relative_diff, dk_max_diff_idx = i;
		dv_cpu_max_abs = fmaxf(dv_cpu_max_abs, fabsf(dv_tensor->data.f32[i]));
		dv_gpu_max_abs = fmaxf(dv_gpu_max_abs, fabsf(dv_gpu_f32[i]));
		denom = fmaxf(fmaxf(fabsf(dv_tensor->data.f32[i]), fabsf(dv_gpu_f32[i])), 1.0f);
		relative_diff = fabsf(dv_tensor->data.f32[i] - dv_gpu_f32[i]) / denom;
		if (relative_diff > dv_max_relative_diff)
			dv_max_relative_diff = relative_diff, dv_max_diff_idx = i;
	}
	REQUIRE(dq_gpu_max_abs >= dq_cpu_max_abs * 0.5f && dq_gpu_max_abs <= dq_cpu_max_abs * 2.0f,
		"quantized attention dQ magnitude should stay close to CPU reference on 1536 surface (cpu max abs %g gpu max abs %g)",
		dq_cpu_max_abs, dq_gpu_max_abs);
	REQUIRE(dk_gpu_max_abs >= dk_cpu_max_abs * 0.5f && dk_gpu_max_abs <= dk_cpu_max_abs * 2.0f,
		"quantized attention dK magnitude should stay close to CPU reference on 1536 surface (cpu max abs %g gpu max abs %g)",
		dk_cpu_max_abs, dk_gpu_max_abs);
	REQUIRE(dv_gpu_max_abs >= dv_cpu_max_abs * 0.5f && dv_gpu_max_abs <= dv_cpu_max_abs * 2.0f,
		"quantized attention dV magnitude should stay close to CPU reference on 1536 surface (cpu max abs %g gpu max abs %g)",
		dv_cpu_max_abs, dv_gpu_max_abs);
	REQUIRE(dq_max_relative_diff <= 8e-2, "quantized attention dQ should match CPU reference on 1536 surface (max relative diff %g at %d: %g vs %g)", dq_max_relative_diff, dq_max_diff_idx, dq_tensor->data.f32[dq_max_diff_idx], dq_gpu_f32[dq_max_diff_idx]);
	REQUIRE(dk_max_relative_diff <= 1e-1, "quantized attention dK should match CPU reference on 1536 surface (max relative diff %g at %d: %g vs %g)", dk_max_relative_diff, dk_max_diff_idx, dk_tensor->data.f32[dk_max_diff_idx], dk_gpu_f32[dk_max_diff_idx]);
	REQUIRE(dv_max_relative_diff <= 8e-2, "quantized attention dV should match CPU reference on 1536 surface (max relative diff %g at %d: %g vs %g)", dv_max_relative_diff, dv_max_diff_idx, dv_tensor->data.f32[dv_max_diff_idx], dv_gpu_f32[dv_max_diff_idx]);

	ccfree(dq_gpu_f32);
	ccfree(dk_gpu_f32);
	ccfree(dv_gpu_f32);
	ccv_nnc_tensor_free(gpu_q_tensor);
	ccv_nnc_tensor_free(gpu_k_tensor);
	ccv_nnc_tensor_free(gpu_v_tensor);
	ccv_nnc_tensor_free(gpu_do_tensor);
	ccv_nnc_tensor_free(gpu_o_tensor);
	ccv_nnc_tensor_free(gpu_dq_tensor);
	ccv_nnc_tensor_free(gpu_dk_tensor);
	ccv_nnc_tensor_free(gpu_dv_tensor);
	ccv_nnc_tensor_free(gpu_softmax_lse);
	ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
	ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
	ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
	ccv_nnc_tensor_free(q_tensor);
	ccv_nnc_tensor_free(k_tensor);
	ccv_nnc_tensor_free(v_tensor);
	ccv_nnc_tensor_free(do_tensor);
	ccv_nnc_tensor_free(dq_tensor);
	ccv_nnc_tensor_free(dk_tensor);
	ccv_nnc_tensor_free(dv_tensor);
	ccv_nnc_tensor_free(q_tensor_f16);
	ccv_nnc_tensor_free(k_tensor_f16);
	ccv_nnc_tensor_free(v_tensor_f16);
	ccv_nnc_tensor_free(do_tensor_f16);
}

TEST_CASE("scaled dot product attention with mps in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
#define num_long_trials 8
#define num_short_trials 4
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		const int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1, 32,   12, 16, 1, 2, 1 };
		const int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5, 160,  256, 128, 77, 77, 5 };
		const int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5, 128,  128, 128, 128, 128, 5 };
		const int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int Hk_candidates[num_trials] = {   8,  8, 4, 2, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int D_candidates[num_trials] = {  64, 40, 160, 192, 256, 128, 48, 96, 160, 192, 256, 128 };

		const int B = B_candidates[trial];
		const int R = R_candidates[trial];
		const int C = C_candidates[trial];
		const int Hq = Hq_candidates[trial];
		const int Hk = Hk_candidates[trial];
		const int D = D_candidates[trial];
		const int is_causal = 0;
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(o_tensor), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_o_tensor), TENSOR_LIST(copy_of_gpu_o_tensor_f16), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_o_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_o_tensor_f16), TENSOR_LIST(copy_of_gpu_o_tensor), 0);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_o_tensor->data.f32, o_tensor->data.f32, B * R * Hq * D, 8e-3, "scaled dot product attention result should be the same");

		ccv_nnc_tensor_free(o_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_o_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention + unify head with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_symbolic_graph_t* const sdp_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t q = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t k = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t v = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t w = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t bias = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 512), "bias");
	ccv_nnc_tensor_symbol_t c = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t r = ccv_nnc_tensor_symbol_new(sdp_symbolic_graph, CPU_TENSOR_NHWC(32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(sdp_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(q, k, v, NO_TENSOR_SYMBOL, w, bias), TENSOR_SYMBOL_LIST(r, NO_TENSOR_SYMBOL, c), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(sdp_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* sdp_graph = 0;
	ccv_nnc_tensor_arena_t* sdp_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* sdp_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(sdp_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(sdp_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(sdp_symbolic_graph), &sdp_graph, &sdp_tensor_arena, &sdp_graph_exec_arena);
	ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, q);
	ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, k);
	ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, v);
	ccv_nnc_tensor_t* const w_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, w);
	ccv_nnc_tensor_t* const bias_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, bias);
	dsfmt_t dsfmt;
	int i;
	dsfmt_init_gen_rand(&dsfmt, 1);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 32 * 8 * 128 * 64; i++)
		v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512 * 512; i++)
		w_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 512; i++)
		bias_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_symbolic_graph_t* const g_symbolic_graph = ccv_nnc_symbolic_graph_new();
	ccv_nnc_tensor_symbol_t gq = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "q");
	ccv_nnc_tensor_symbol_t gk = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "k");
	ccv_nnc_tensor_symbol_t gv = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "v");
	ccv_nnc_tensor_symbol_t gw = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 512, 512), "w");
	ccv_nnc_tensor_symbol_t gbias = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 512), "bias");
	ccv_nnc_tensor_symbol_t gc = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 8, 64), "c");
	ccv_nnc_tensor_symbol_t gr = ccv_nnc_tensor_symbol_new(g_symbolic_graph, GPU_TENSOR_NHWC(000, 32F, 32, 128, 512), "r");
	ccv_nnc_graph_exec_symbol_new(g_symbolic_graph, CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0), TENSOR_SYMBOL_LIST(gq, gk, gv, NO_TENSOR_SYMBOL, gw, gbias), TENSOR_SYMBOL_LIST(gr, NO_TENSOR_SYMBOL, gc), "scaled_dot_product_attention");
	ccv_nnc_graph_exec_symbol_autogen(g_symbolic_graph, 0, 0, CCV_NNC_AUTOGEN_ALL_EXECS | CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
	ccv_nnc_graph_t* g_graph = 0;
	ccv_nnc_tensor_arena_t* g_tensor_arena = 0;
	ccv_nnc_graph_exec_arena_t* g_graph_exec_arena = 0;
	ccv_nnc_symbolic_graph_compile(g_symbolic_graph, ccv_nnc_default_compile_params, 0, 0, 0, 0, SYMBOLIC_GRAPH_SOURCES(g_symbolic_graph), SYMBOLIC_GRAPH_DESTINATIONS(g_symbolic_graph), &g_graph, &g_tensor_arena, &g_graph_exec_arena);
	ccv_nnc_tensor_t* const gq_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gq);
	ccv_nnc_tensor_t* const gk_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gk);
	ccv_nnc_tensor_t* const gv_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gv);
	ccv_nnc_tensor_t* const gw_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gw);
	ccv_nnc_tensor_t* const gbias_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gbias);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, w_tensor, bias_tensor), TENSOR_LIST(gq_tensor, gk_tensor, gv_tensor, gw_tensor, gbias_tensor), 0);
	ccv_nnc_graph_run(sdp_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_graph_run(g_graph, 0, TRAVERSE_FULL, 0, 0);
	ccv_nnc_tensor_t* const r_tensor = ccv_nnc_tensor_from_symbol(sdp_tensor_arena, r);
	ccv_nnc_tensor_t* const gr_tensor = ccv_nnc_tensor_from_symbol(g_tensor_arena, gr);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 32, 128, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gr_tensor), TENSOR_LIST(hr), 0);
	float max_relative_diff = 0;
	int max_diff_idx = 0;
	for (i = 0; i < 32 * 128 * 512; i++)
	{
		const float denom = fmaxf(fmaxf(fabsf(r_tensor->data.f32[i]), fabsf(hr->data.f32[i])), 1.0f);
		const float relative_diff = fabsf(r_tensor->data.f32[i] - hr->data.f32[i]) / denom;
		if (relative_diff > max_relative_diff)
			max_relative_diff = relative_diff, max_diff_idx = i;
	}
	REQUIRE(max_relative_diff <= 2e-3, "graph computed result should match scaled dot product attention op result (max relative diff %g at %d: %g vs %g)", max_relative_diff, max_diff_idx, r_tensor->data.f32[max_diff_idx], hr->data.f32[max_diff_idx]);
	ccv_nnc_symbolic_graph_free(sdp_symbolic_graph);
	ccv_nnc_tensor_arena_free(sdp_tensor_arena);
	ccv_nnc_graph_exec_arena_free(sdp_graph_exec_arena);
	ccv_nnc_graph_free(sdp_graph);
	ccv_nnc_symbolic_graph_free(g_symbolic_graph);
	ccv_nnc_tensor_arena_free(g_tensor_arena);
	ccv_nnc_graph_exec_arena_free(g_graph_exec_arena);
	ccv_nnc_graph_free(g_graph);
	ccv_nnc_tensor_free(hr);
}

TEST_CASE("scaled dot product attention + row-wise 8i unify head with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 2;
	const int R = 16;
	const int H = 4;
	const int D = 32;
	const int K = H * D;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const hv = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const hw_dense = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, K, K), 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(32F, K, K)), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, K), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1);
	int i;
	for (i = 0; i < B * R * H * D; i++)
	{
		hq->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
		hk->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
		hv->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	}
	for (i = 0; i < K * K; i++)
		hw_dense->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (i = 0; i < K; i++)
		hbias->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	const size_t qsize = ccv_nnc_quantize_8i_rowwise(hw_dense->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, K * K, K, 0, 0, hwq->data.u8, ccv_nnc_tensor_data_size_without_padding(hwq->info));
	REQUIRE_EQ(qsize, ccv_nnc_tensor_data_size_without_padding(hwq->info), "row-wise 8i weight quantization should fit the output tensor");
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gk = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gv = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(GPU_TENSOR_NHWC(000, 32F, K, K)), 0);
	ccv_nnc_tensor_t* const gwd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, K, K), 0);
	ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, K), 0);
	ccv_nnc_tensor_t* const grq = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, K), 0);
	ccv_nnc_tensor_t* const grd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, K), 0);
	ccv_nnc_tensor_t* const gcq = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_tensor_t* const gcd = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hv, hwq, hbias), TENSOR_LIST(gq, gk, gv, gwq, gbias), 0);
	ccv_nnc_dequantize_8i_rowwise(gwq->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, qsize, K, gwd->data.u8, K * K);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(1.0 / 8, 0);
	cmd.info.scaled_dot_product_attention.flags = CCV_NNC_GEMM_8I;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gk, gv, NULL, gwq, gbias), TENSOR_LIST(grq, NULL, gcq), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gk, gv, NULL, gwd, gbias), TENSOR_LIST(grd, NULL, gcd), 0);
	ccv_nnc_tensor_t* const hrq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, K), 0);
	ccv_nnc_tensor_t* const hrd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, K), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(grq, grd), TENSOR_LIST(hrq, hrd), 0);
	float max_relative_diff = 0;
	int max_diff_idx = 0;
	for (i = 0; i < B * R * K; i++)
	{
		const float denom = fmaxf(fmaxf(fabsf(hrq->data.f32[i]), fabsf(hrd->data.f32[i])), 1.0f);
		const float relative_diff = fabsf(hrq->data.f32[i] - hrd->data.f32[i]) / denom;
		if (relative_diff > max_relative_diff)
			max_relative_diff = relative_diff, max_diff_idx = i;
	}
	REQUIRE(max_relative_diff <= 5e-2, "row-wise 8i unify head result should match dequantized weight result (max relative diff %g at %d: %g vs %g)", max_relative_diff, max_diff_idx, hrq->data.f32[max_diff_idx], hrd->data.f32[max_diff_idx]);
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hv);
	ccv_nnc_tensor_free(hw_dense);
	ccv_nnc_tensor_free(hwq);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(gk);
	ccv_nnc_tensor_free(gv);
	ccv_nnc_tensor_free(gwq);
	ccv_nnc_tensor_free(gwd);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(grq);
	ccv_nnc_tensor_free(grd);
	ccv_nnc_tensor_free(gcq);
	ccv_nnc_tensor_free(gcd);
	ccv_nnc_tensor_free(hrq);
	ccv_nnc_tensor_free(hrd);
}

TEST_CASE("scaled dot product attention gradient with mps")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
#define num_long_trials 2
#define num_short_trials 2
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		int B_candidates[num_trials] = {  32,   3, 2, 1 };
		int R_candidates[num_trials] = { 128,  61, 6, 2 };
		int C_candidates[num_trials] = { 128,  49, 2, 1 };
		int H_candidates[num_trials] = {   8,  13, 3, 1 };
		int D_candidates[num_trials] = {  64, 191, 4, 8 };

		int B = B_candidates[trial];
		int R = R_candidates[trial];
		int C = C_candidates[trial];
		int H = H_candidates[trial];
		int D = D_candidates[trial];
		float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);

		for (int i = 0; i < B * R * H * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * H * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * H * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		for (int i = 0; i < B * R * H * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, R, H, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, H, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, H, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dv_tensor->data.f32, dv_tensor->data.f32, B * C * H * D, 5e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * H * D, 5e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * H * D, 5e-3, "scaled dot product attention result should be the same");

		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention gradient with mps in half precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
#define num_long_trials 8
#define num_short_trials 4
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		const int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1, 32,   12, 16, 1, 2, 1 };
		const int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5, 160,  256, 128, 77, 77, 5 };
		const int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5, 128,  128, 128, 128, 128, 5 };
		const int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int D_candidates[num_trials] = {  64, 40, 160, 192, 256, 128, 48, 96, 160, 192, 256, 128 };

		const int B = B_candidates[trial];
		const int R = R_candidates[trial];
		const int C = C_candidates[trial];
		const int Hq = Hq_candidates[trial];
		const int Hk = Hq_candidates[trial];
		const int D = D_candidates[trial];
		const int is_causal = 0;
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		for (int i = 0; i < B * R * Hq * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.deterministic = 0;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * Hq * D, 1e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * Hk * D, 3e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dv_tensor->data.f32, dv_tensor->data.f32, B * C * Hk * D, 6e-3, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(do_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("scaled dot product attention gradient with mps in bfloat precision")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD, CCV_NNC_BACKEND_MPS));
#define num_long_trials 8
#define num_short_trials 4
#define num_trials (num_long_trials + num_short_trials)

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 10);
	for (int trial = 0; trial < num_trials; ++trial) {
		const int B_candidates[num_trials] = {  32,   12, 16, 1, 2, 1, 32,   12, 16, 1, 2, 1 };
		const int R_candidates[num_trials] = { 160,  256, 128, 77, 77, 5, 160,  256, 128, 77, 77, 5 };
		const int C_candidates[num_trials] = { 128,  128, 128, 128, 128, 5, 128,  128, 128, 128, 128, 5 };
		const int Hq_candidates[num_trials] = {   8,  8, 8, 8, 8, 32, 8,  8, 8, 8, 8, 32 };
		const int D_candidates[num_trials] = {  64, 40, 160, 192, 256, 128, 48, 96, 160, 192, 256, 128 };

		const int B = B_candidates[trial];
		const int R = R_candidates[trial];
		const int C = C_candidates[trial];
		const int Hq = Hq_candidates[trial];
		const int Hk = Hq_candidates[trial];
		const int D = D_candidates[trial];
		const int is_causal = 0;
		const float scale = 1.0 / sqrt((float)D);

		ccv_nnc_tensor_t* const q_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);

		for (int i = 0; i < B * R * Hq * D; ++i) {
			q_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			k_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		for (int i = 0; i < B * C * Hk * D; ++i) {
			v_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}

		ccv_nnc_tensor_t* const do_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		for (int i = 0; i < B * R * Hq * D; ++i) {
			do_tensor->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
		}
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(do_tensor, 0, 0, q_tensor, k_tensor, v_tensor), TENSOR_LIST(dq_tensor, dk_tensor, dv_tensor), 0);
		ccv_nnc_tensor_t* const q_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const k_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const v_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const do_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor, k_tensor, v_tensor, do_tensor), TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), 0);

		ccv_nnc_tensor_t* const gpu_q_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_k_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_v_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_o_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_do_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dq_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const gpu_dk_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const gpu_dv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16BF, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q_tensor_f16, k_tensor_f16, v_tensor_f16, do_tensor_f16), TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, gpu_do_tensor), 0);

		ccv_nnc_tensor_t* const gpu_softmax_lse = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, Hq, R), 0);
		ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ATTENTION_FORWARD(scale, is_causal), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, NULL, NULL, NULL), TENSOR_LIST(gpu_o_tensor, gpu_softmax_lse), 0);

		ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ATTENTION_BACKWARD(scale, is_causal);
		cmd.info.scaled_dot_product_attention.deterministic = 0;
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_do_tensor, 0, 0, gpu_q_tensor, gpu_k_tensor, gpu_v_tensor, 0, 0, 0, gpu_o_tensor, gpu_softmax_lse), TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor_f16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16BF, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu_dq_tensor, gpu_dk_tensor, gpu_dv_tensor), TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), 0);

		ccv_nnc_tensor_t* const copy_of_gpu_dq_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, R, Hq, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dk_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_tensor_t* const copy_of_gpu_dv_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, C, Hk, D), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(copy_of_gpu_dq_tensor_f16, copy_of_gpu_dk_tensor_f16, copy_of_gpu_dv_tensor_f16), TENSOR_LIST(copy_of_gpu_dq_tensor, copy_of_gpu_dk_tensor, copy_of_gpu_dv_tensor), 0);

		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dq_tensor->data.f32, dq_tensor->data.f32, B * R * Hq * D, 5e-3, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dk_tensor->data.f32, dk_tensor->data.f32, B * C * Hk * D, 1e-2, "scaled dot product attention result should be the same");
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, copy_of_gpu_dv_tensor->data.f32, dv_tensor->data.f32, B * C * Hk * D, 2e-2, "GPU computed output should be the same as CPU computed ones");

		ccv_nnc_tensor_free(do_tensor);
		ccv_nnc_tensor_free(gpu_do_tensor);
		ccv_nnc_tensor_free(gpu_o_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor_f16);
		ccv_nnc_tensor_free(copy_of_gpu_dq_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dk_tensor);
		ccv_nnc_tensor_free(copy_of_gpu_dv_tensor);
		ccv_nnc_tensor_free(q_tensor);
		ccv_nnc_tensor_free(k_tensor);
		ccv_nnc_tensor_free(v_tensor);
		ccv_nnc_tensor_free(q_tensor_f16);
		ccv_nnc_tensor_free(k_tensor_f16);
		ccv_nnc_tensor_free(v_tensor_f16);
		ccv_nnc_tensor_free(do_tensor_f16);
		ccv_nnc_tensor_free(gpu_q_tensor);
		ccv_nnc_tensor_free(gpu_k_tensor);
		ccv_nnc_tensor_free(gpu_v_tensor);
		ccv_nnc_tensor_free(dq_tensor);
		ccv_nnc_tensor_free(dk_tensor);
		ccv_nnc_tensor_free(dv_tensor);
		ccv_nnc_tensor_free(gpu_dq_tensor);
		ccv_nnc_tensor_free(gpu_dk_tensor);
		ccv_nnc_tensor_free(gpu_dv_tensor);
		ccv_nnc_tensor_free(gpu_softmax_lse);
	}
#undef num_long_trials
#undef num_short_trials
#undef num_trials
}

TEST_CASE("backward gemm with no transpose")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);

	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};

	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);

	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};

	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);

	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);

	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_t cmd = CMD_GEMM_BACKWARD();
	cmd.backend = CCV_NNC_BACKEND_MPS;
	cmd.algorithm = 1; // This is cblas.

	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(h, db, dbias), 0);

	ccv_nnc_tensor_t* const ch = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 4, 2), 0);
	ccv_nnc_tensor_t* const cdb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const cdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC( 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h, db, dbias), TENSOR_LIST(ch, cdb, cdbias), 0);

	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);

	REQUIRE_TENSOR_EQ(cdbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);

	REQUIRE_TENSOR_EQ(ch, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(cdb, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
}

TEST_CASE("backward gemm with transpose a")
{
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b")
{
		GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(0, 1), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}


TEST_CASE("backward gemm large data set")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, h), TENSOR_LIST(tb, tdw, tdbias, th), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb->data.f32, hb->data.f32, 10 * 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw->data.f32, hdw->data.f32, 64 * 128, 5e-3, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias->data.f32, hdbias->data.f32, 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th->data.f32, hh->data.f32, 10 * 128, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backward gemm no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hg), TENSOR_LIST(a, w, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, hdw, 0), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, dw, 0), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, h), TENSOR_LIST(tb, tdw, th), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb->data.f32, hb->data.f32, 10 * 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw->data.f32, hdw->data.f32, 64 * 128, 5e-3, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th->data.f32, hh->data.f32, 10 * 128, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdw);
}

TEST_CASE("backward gemm no h")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dw = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(0, hdw, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(0, dw, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, dw, dbias, 0), TENSOR_LIST(tb, tdw, tdbias, 0), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb->data.f32, hb->data.f32, 10 * 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdw->data.f32, hdw->data.f32, 64 * 128, 5e-3, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias->data.f32, hdbias->data.f32, 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dw);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdw);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(tdw);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backward gemm no dw")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0);
	ccv_nnc_tensor_t* a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64, 128), 0);
	ccv_nnc_tensor_t* bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* g = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 64), 0);
	ccv_nnc_tensor_t* dbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 64), 0);
	ccv_nnc_tensor_t* h = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 10, 128), 0);

	ccv_nnc_tensor_t* ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64, 128), 0);
	ccv_nnc_tensor_t* hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hg = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* hdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* hh = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	int i;
	for (i = 0; i < 64 * 128; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / (64 * 128);
	for (i = 0; i < 64; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 128; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < 10 * 64; i++)
		hg->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias, hg), TENSOR_LIST(a, w, bias, g), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(hg, ha, hw, 0), TENSOR_LIST(hh, 0, hdbias), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, w, 0), TENSOR_LIST(h, 0, dbias), 0);
	ccv_nnc_tensor_t* tb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 64), 0);
	ccv_nnc_tensor_t* tdbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 64), 0);
	ccv_nnc_tensor_t* th = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 10, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b, 0, dbias, h), TENSOR_LIST(tb, 0, tdbias, th), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tb->data.f32, hb->data.f32, 10 * 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, tdbias->data.f32, hdbias->data.f32, 64, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, th->data.f32, hh->data.f32, 10 * 128, 1e-5, "GPU computed output should be numerically close to CPU computed ones");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hg);
	ccv_nnc_tensor_free(hdbias);
	ccv_nnc_tensor_free(hh);
	ccv_nnc_tensor_free(tb);
	ccv_nnc_tensor_free(th);
	ccv_nnc_tensor_free(tdbias);
}

TEST_CASE("backwar gemm with no transpose batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 10 * 24 + 20 * 25 + 30 * 26,
		40 * 21 + 50 * 22 + 60 * 23, 40 * 24 + 50 * 25 + 60 * 26,
		70 * 21 + 80 * 22 + 90 * 23, 70 * 24 + 80 * 25 + 90 * 26,
		100 * 21 + 110 * 22 + 120 * 23, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
		212, 222, 232,
		242, 252, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose b batch 2, batched b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 1, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22, 26, 30,
		220, 260, 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 2, 1, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with transpose a and b batch 2, same b")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	ccv_nnc_tensor_t* const dbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 2), 0);
	ccv_nnc_tensor_t* const gdbias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2), TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb, gdbias), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb, gdbias), TENSOR_LIST(h, db, dbias), 0);
	float dbiastp[] = {
		22 + 220, 26 + 260, 30 + 300,
	};
	ccv_nnc_tensor_t dbiast = ccv_nnc_tensor(dbiastp, CPU_TENSOR_NHWC(32F, 3), 0);
	REQUIRE_TENSOR_EQ(dbias, &dbiast, "bias should be equal");
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 21 + 20 * 22 + 30 * 23, 40 * 21 + 50 * 22 + 60 * 23, 70 * 21 + 80 * 22 + 90 * 23, 100 * 21 + 110 * 22 + 120 * 23,
		10 * 24 + 20 * 25 + 30 * 26, 40 * 24 + 50 * 25 + 60 * 26, 70 * 24 + 80 * 25 + 90 * 26, 100 * 24 + 110 * 25 + 120 * 26,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19 + 10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20 + 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19 + 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20 + 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19 + 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20 + 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(dbias);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
	ccv_nnc_tensor_free(gdbias);
}

TEST_CASE("backward gemm with no transpose batch 2, batched b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 22, 23,
		24, 25, 26,
		212, 222, 232,
		242, 252, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 3), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 3 * 13 + 6 * 15 + 9 * 17 + 12 * 19,
		1 * 14 + 4 * 16 + 7 * 18 + 10 * 20, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 30 * 131 + 60 * 151 + 90 * 171 + 120 * 191,
		10 * 141 + 40 * 161 + 70 * 181 + 100 * 201, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 2, 3), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("backward gemm with transpose b batch 2, batched b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 14,
		15, 16,
		17, 18,
		19, 20,
		131, 141,
		151, 161,
		171, 181,
		191, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 2), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 1 * 24 + 2 * 25 + 3 * 26,
		4 * 21 + 5 * 22 + 6 * 23, 4 * 24 + 5 * 25 + 6 * 26,
		7 * 21 + 8 * 22 + 9 * 23, 7 * 24 + 8 * 25 + 9 * 26,
		10 * 21 + 11 * 22 + 12 * 23, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 10 * 242 + 20 * 252 + 30 * 262,
		40 * 212 + 50 * 222 + 60 * 232, 40 * 242 + 50 * 252 + 60 * 262,
		70 * 212 + 80 * 222 + 90 * 232, 70 * 242 + 80 * 252 + 90 * 262,
		100 * 212 + 110 * 222 + 120 * 232, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 4, 2), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("backward gemm with transpose a and b batch 2, batch b, no bias")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS) &&
		ccv_nnc_cmd_ok(CCV_NNC_GEMM_BACKWARD, CCV_NNC_BACKEND_MPS));
	float gp[] = {
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	};
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 4, 3), 0);
	float ap[] = {
		13, 15, 17, 19,
		14, 16, 18, 20,
		131, 151, 171, 191,
		141, 161, 181, 201,
	};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	float bp[] = {
		21, 24,
		22, 25,
		23, 26,
		212, 242,
		222, 252,
		232, 262,
	};
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(bp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const db = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gg = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 4, 3), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_tensor_t* const gh = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 2, 4), 0);
	ccv_nnc_tensor_t* const gdb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2, 3, 2), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g, a, b), TENSOR_LIST(gg, ga, gb), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_BACKWARD(TRANSPOSE(1, 2), TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(gg, ga, gb), TENSOR_LIST(gh, gdb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gh, gdb), TENSOR_LIST(h, db), 0);
	float htp[] = {
		1 * 21 + 2 * 22 + 3 * 23, 4 * 21 + 5 * 22 + 6 * 23, 7 * 21 + 8 * 22 + 9 * 23, 10 * 21 + 11 * 22 + 12 * 23,
		1 * 24 + 2 * 25 + 3 * 26, 4 * 24 + 5 * 25 + 6 * 26, 7 * 24 + 8 * 25 + 9 * 26, 10 * 24 + 11 * 25 + 12 * 26,
		10 * 212 + 20 * 222 + 30 * 232, 40 * 212 + 50 * 222 + 60 * 232, 70 * 212 + 80 * 222 + 90 * 232, 100 * 212 + 110 * 222 + 120 * 232,
		10 * 242 + 20 * 252 + 30 * 262, 40 * 242 + 50 * 252 + 60 * 262, 70 * 242 + 80 * 252 + 90 * 262, 100 * 242 + 110 * 252 + 120 * 262,
	};
	ccv_nnc_tensor_t ht = ccv_nnc_tensor(htp, CPU_TENSOR_NHWC(32F, 2, 2, 4), 0);
	REQUIRE_TENSOR_EQ(h, &ht, "h should be equal");
	float dbtp[] = {
		1 * 13 + 4 * 15 + 7 * 17 + 10 * 19, 1 * 14 + 4 * 16 + 7 * 18 + 10 * 20,
		2 * 13 + 5 * 15 + 8 * 17 + 11 * 19, 2 * 14 + 5 * 16 + 8 * 18 + 11 * 20,
		3 * 13 + 6 * 15 + 9 * 17 + 12 * 19, 3 * 14 + 6 * 16 + 9 * 18 + 12 * 20,
		10 * 131 + 40 * 151 + 70 * 171 + 100 * 191, 10 * 141 + 40 * 161 + 70 * 181 + 100 * 201,
		20 * 131 + 50 * 151 + 80 * 171 + 110 * 191, 20 * 141 + 50 * 161 + 80 * 181 + 110 * 201,
		30 * 131 + 60 * 151 + 90 * 171 + 120 * 191, 30 * 141 + 60 * 161 + 90 * 181 + 120 * 201,
	};
	ccv_nnc_tensor_t dbt = ccv_nnc_tensor(dbtp, CPU_TENSOR_NHWC(32F, 2, 3, 2), 0);
	REQUIRE_TENSOR_EQ(db, &dbt, "db should be equal");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(db);
	ccv_nnc_tensor_free(gg);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gh);
	ccv_nnc_tensor_free(gdb);
}

TEST_CASE("mps segmented gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 11);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 384, 256), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 1;
	hindices->data.i32[1] = 0;
	hindices->data.i32[2] = 2;
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hcounts->data.i32[0] = 129;
	hcounts->data.i32[1] = 131;
	hcounts->data.i32[2] = 124;
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 128, 256), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 384, 128), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 384, 128), 0);
	int i;
	for (i = 0; i < 3 * 128 * 256; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 256;
	for (i = 0; i < 384 * 256; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 384, 256), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3, 128, 256), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 384, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(a, indices, counts, w), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 384 * 128, 3e-4, "segmented GEMM result should match CPU reference");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(bt);
}

TEST_CASE("mps segmented gemm with bias in half precision, split-k")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 13);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 272, 4096), 0);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2), 0);
	hindices->data.i32[0] = 1;
	hindices->data.i32[1] = 0;
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2), 0);
	hcounts->data.i32[0] = 136;
	hcounts->data.i32[1] = 136;
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128, 4096), 0);
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 272, 128), 0);
	ccv_nnc_tensor_t* const hb16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 272, 128), 0);
	ccv_nnc_tensor_t* const bt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 272, 128), 0);
	int i;
	for (i = 0; i < 2 * 128 * 4096; i++)
		hw->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 4096;
	for (i = 0; i < 2 * 128; i++)
		hbias->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) / 128;
	for (i = 0; i < 272 * 4096; i++)
		ha->data.f32[i] = dsfmt_genrand_open_close(&dsfmt);
	ccv_nnc_tensor_t* const ha16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 272, 4096), 0);
	ccv_nnc_tensor_t* const hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 128, 4096), 0);
	ccv_nnc_tensor_t* const hbias16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, 2, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hw, hbias), TENSOR_LIST(ha16, hw16, hbias16), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 272, 4096), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const counts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 128, 4096), 0);
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 272, 128), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha16, hindices, hcounts, hw16, hbias16), TENSOR_LIST(a, indices, counts, w, bias), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w, bias), TENSOR_LIST(b), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(b), TENSOR_LIST(hb16), 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hb16), TENSOR_LIST(hb), 0);
	ccv_nnc_cmd_exec(CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2)), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hindices, hcounts, hw, hbias), TENSOR_LIST(bt), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hb->data.f32, bt->data.f32, 272 * 128, 2e-2, "half-precision segmented GEMM result should match CPU reference");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(bias);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(hb16);
	ccv_nnc_tensor_free(bt);
	ccv_nnc_tensor_free(ha16);
	ccv_nnc_tensor_free(hw16);
	ccv_nnc_tensor_free(hbias16);
}

// Derived from shapes.txt NA lines, assuming the call shape is C = A @ B^T.
NA_GEMM_SHAPE_TEST(306, 2048, 3840)
NA_GEMM_SHAPE_TEST(306, 4096, 3840)
NA_GEMM_SHAPE_TEST(306, 3840, 4096)
NA_GEMM_SHAPE_TEST(306, 15360, 3840)
NA_GEMM_SHAPE_TEST(306, 3840, 15360)
NA_GEMM_SHAPE_TEST(1024, 4096, 4096)
NA_GEMM_SHAPE_TEST(1024, 32, 4096)
NA_GEMM_SHAPE_TEST(1024, 16384, 4096)
NA_GEMM_SHAPE_TEST(1024, 4096, 16384)
NA_GEMM_SHAPE_TEST(1024, 2048, 2048)
NA_GEMM_SHAPE_TEST(1024, 32, 2048)
NA_GEMM_SHAPE_TEST(1024, 8192, 2048)
NA_GEMM_SHAPE_TEST(1024, 2048, 8192)
NA_GEMM_SHAPE_TEST(1, 2048, 256)
NA_GEMM_SHAPE_TEST(1, 2048, 2048)
NA_GEMM_SHAPE_TEST(1, 4096, 256)
NA_GEMM_SHAPE_TEST(1, 4096, 4096)
NA_GEMM_SHAPE_TEST(2, 2048, 2048)
NA_GEMM_SHAPE_TEST(2, 4096, 4096)
NA_GEMM_SHAPE_TEST(3, 4096, 4096)
NA_GEMM_SHAPE_TEST(4, 4096, 4096)
NA_GEMM_SHAPE_TEST(5, 4096, 4096)
NA_GEMM_SHAPE_TEST(6, 1024, 3072)
NA_GEMM_SHAPE_TEST(6, 4096, 4096)
NA_GEMM_SHAPE_TEST(7, 4096, 4096)
NA_GEMM_SHAPE_TEST(8, 4096, 4096)
NA_GEMM_SHAPE_TEST(16, 4096, 4096)
NA_GEMM_SHAPE_TEST(32, 4096, 4096)
NA_GEMM_SHAPE_TEST(48, 4096, 4096)
NA_GEMM_SHAPE_TEST(48, 4096, 15360)
NA_GEMM_SHAPE_TEST(16, 4096, 24576)
NA_GEMM_SHAPE_TEST(3, 4096, 32768)
NA_GEMM_SHAPE_TEST(6, 4096, 32768)
NA_GEMM_SHAPE_TEST(8, 4096, 32768)
NA_GEMM_SHAPE_TEST(16, 4096, 32768)
NA_GEMM_SHAPE_TEST(1024, 4096, 128)
NA_GEMM_SHAPE_TEST(257, 2048, 128)
NA_GEMM_SHAPE_TEST(33792, 4096, 4096)
NA_GEMM_SHAPE_TEST(33792, 32, 4096)
NA_GEMM_SHAPE_TEST(257, 2048, 2048)
NA_GEMM_SHAPE_TEST(257, 32, 2048)
NA_GEMM_SHAPE_TEST(33792, 2048, 4096)
NA_GEMM_SHAPE_TEST(33792, 4096, 2048)
NA_GEMM_SHAPE_TEST(33792, 16384, 4096)
NA_GEMM_SHAPE_TEST(33792, 4096, 16384)
NA_GEMM_SHAPE_TEST(257, 8192, 2048)
NA_GEMM_SHAPE_TEST(257, 2048, 8192)
NA_GEMM_SHAPE_TEST(33792, 128, 4096)
NA_GEMM_SHAPE_TEST(257, 128, 2048)
NA_GEMM_BIAS_SHAPE_TEST(306, 2048, 3840)
NA_GEMM_BIAS_SHAPE_TEST(306, 4096, 3840)
NA_GEMM_BIAS_SHAPE_TEST(306, 3840, 4096)
NA_GEMM_BIAS_SHAPE_TEST(306, 15360, 3840)
NA_GEMM_BIAS_SHAPE_TEST(306, 3840, 15360)
NA_GEMM_BIAS_SHAPE_TEST(1024, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(1024, 32, 4096)
NA_GEMM_BIAS_SHAPE_TEST(1024, 16384, 4096)
NA_GEMM_BIAS_SHAPE_TEST(1024, 4096, 16384)
NA_GEMM_BIAS_SHAPE_TEST(1024, 2048, 2048)
NA_GEMM_BIAS_SHAPE_TEST(1024, 32, 2048)
NA_GEMM_BIAS_SHAPE_TEST(1024, 8192, 2048)
NA_GEMM_BIAS_SHAPE_TEST(1024, 2048, 8192)
NA_GEMM_BIAS_SHAPE_TEST(1, 2048, 256)
NA_GEMM_BIAS_SHAPE_TEST(1, 2048, 2048)
NA_GEMM_BIAS_SHAPE_TEST(1, 4096, 256)
NA_GEMM_BIAS_SHAPE_TEST(1, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(2, 2048, 2048)
NA_GEMM_BIAS_SHAPE_TEST(2, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(3, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(4, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(5, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(6, 1024, 3072)
NA_GEMM_BIAS_SHAPE_TEST(6, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(7, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(8, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(3, 4096, 32768)
NA_GEMM_BIAS_SHAPE_TEST(16, 4096, 32768)
NA_GEMM_BIAS_SHAPE_TEST(32, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(48, 4096, 4096)
NA_GEMM_BFLOAT_SHAPE_TEST(3, 4096, 4096)
NA_GEMM_BFLOAT_SHAPE_TEST(5, 4096, 4096)
NA_GEMM_BFLOAT_SHAPE_TEST(6, 1024, 3072)
NA_GEMM_BFLOAT_SHAPE_TEST(7, 4096, 4096)
NA_GEMM_BFLOAT_SHAPE_TEST(8, 4096, 4096)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(3, 4096, 4096)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(5, 4096, 4096)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(6, 1024, 3072)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(7, 4096, 4096)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(8, 4096, 4096)
NA_GEMM_BFLOAT_SHAPE_TEST(48, 4096, 4096)
NA_GEMM_BFLOAT_BIAS_SHAPE_TEST(48, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(1024, 4096, 128)
NA_GEMM_BIAS_SHAPE_TEST(257, 2048, 128)
NA_GEMM_BIAS_SHAPE_TEST(33792, 4096, 4096)
NA_GEMM_BIAS_SHAPE_TEST(33792, 32, 4096)
NA_GEMM_BIAS_SHAPE_TEST(257, 2048, 2048)
NA_GEMM_BIAS_SHAPE_TEST(257, 32, 2048)
NA_GEMM_BIAS_SHAPE_TEST(33792, 2048, 4096)
NA_GEMM_BIAS_SHAPE_TEST(33792, 4096, 2048)
NA_GEMM_BIAS_SHAPE_TEST(33792, 16384, 4096)
NA_GEMM_BIAS_SHAPE_TEST(33792, 4096, 16384)
NA_GEMM_BIAS_SHAPE_TEST(257, 8192, 2048)
NA_GEMM_BIAS_SHAPE_TEST(257, 2048, 8192)
NA_GEMM_BIAS_SHAPE_TEST(33792, 128, 4096)
NA_GEMM_BIAS_SHAPE_TEST(257, 128, 2048)

TEST_CASE("scaled dot product arg partition cpu reference")
{
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2, 2), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 2), 0);
	ccv_nnc_tensor_t* const head_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 2), 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 3), 0);
	q->data.f32[0] = 1;
	q->data.f32[1] = 0;
	q->data.f32[2] = 0;
	q->data.f32[3] = 1;
	const float k_data[] = { 1, 1, 2, -1, -1, 3, 0, 0 };
	memcpy(k->data.f32, k_data, sizeof(k_data));
	head_w->data.f32[0] = 1;
	head_w->data.f32[1] = -2;
	ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(3, 1, 0, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, head_w), TENSOR_LIST(selected), 0);
	const int expected[] = { 1, 3, 0 };
	REQUIRE_ARRAY_EQ(int, selected->data.i32, expected, 3, "arg partition should apply relu before head weights and sort descending");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(head_w);
	ccv_nnc_tensor_free(selected);
}

TEST_CASE("scaled dot product arg partition causal compression cpu reference")
{
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1, 1), 0);
	ccv_nnc_tensor_t* const k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* const head_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 5, 1), 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 5, 3), 0);
	int i;
	for (i = 0; i < 5; i++)
		q->data.f32[i] = head_w->data.f32[i] = 1;
	k->data.f32[0] = 1;
	k->data.f32[1] = 2;
	k->data.f32[2] = 3;
	ccv_nnc_cmd_exec(CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(3, 1, 1, 4), ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, head_w), TENSOR_LIST(selected), 0);
	const int expected[] = {
		1, 0, -1,
		1, 0, -1,
		1, 0, -1,
		1, 0, -1,
		2, 1, 0,
	};
	REQUIRE_ARRAY_EQ(int, selected->data.i32, expected, 15, "causal compression should pad rows with too few visible compressed ids");
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(k);
	ccv_nnc_tensor_free(head_w);
	ccv_nnc_tensor_free(selected);
}

static void _mps_scaled_dot_product_arg_partition_fill_stable(ccv_nnc_tensor_t* const q, ccv_nnc_tensor_t* const k, ccv_nnc_tensor_t* const head_w)
{
	const int T = q->info.dim[0];
	const int H = q->info.dim[1];
	const int D = q->info.dim[2];
	const int C = k->info.dim[0];
	int t, h, d, c;
	for (t = 0; t < T; t++)
		for (h = 0; h < H; h++)
			for (d = 0; d < D; d++)
				q->data.f32[(t * H + h) * D + d] = (d == 0) ? (1.0f + (float)h / 1024.0f) : 0;
	for (c = 0; c < C; c++)
		for (d = 0; d < D; d++)
			k->data.f32[c * D + d] = (d == 0) ? (float)(c + 1) : 0;
	for (t = 0; t < T; t++)
		for (h = 0; h < H; h++)
			head_w->data.f32[t * H + h] = 1;
}

static void _mps_scaled_dot_product_arg_partition_from_float(const int datatype, const float* const source, ccv_nnc_tensor_t* const tensor, const int count)
{
	if (datatype == CCV_32F)
		memcpy(tensor->data.f32, source, sizeof(float) * count);
	else if (datatype == CCV_16F)
		ccv_float_to_half_precision(source, (uint16_t*)tensor->data.f16, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(source, (uint16_t*)tensor->data.f16, count);
	else
		assert(0);
}

static void _mps_scaled_dot_product_arg_partition_to_float(const int datatype, const ccv_nnc_tensor_t* const tensor, float* const destination, const int count)
{
	if (datatype == CCV_32F)
		memcpy(destination, tensor->data.f32, sizeof(float) * count);
	else if (datatype == CCV_16F)
		ccv_half_precision_to_float((uint16_t*)tensor->data.f16, destination, count);
	else if (datatype == CCV_16BF)
		ccv_bfloat_to_float((uint16_t*)tensor->data.f16, destination, count);
	else
		assert(0);
}

static int _mps_scaled_dot_product_arg_partition_compare(const int T, const int C, const int H, const int D, const int kth, const int is_causal, const int compression_ratio, const int datatype, const int force_graph, const int force_generic_mfa)
{
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H, D), 0);
	ccv_nnc_tensor_t* const hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, C, D), 0);
	ccv_nnc_tensor_t* const hhead_w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H), 0);
	ccv_nnc_tensor_t* const href = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, kth), 0);
	ccv_nnc_tensor_t* const hselected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, kth), 0);
	_mps_scaled_dot_product_arg_partition_fill_stable(hq, hk, hhead_w);
	ccv_nnc_tensor_param_t hq_input_params = CPU_TENSOR_NHWC(32F, T, H, D);
	hq_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t hk_input_params = CPU_TENSOR_NHWC(32F, C, D);
	hk_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t hhead_w_input_params = CPU_TENSOR_NHWC(32F, T, H);
	hhead_w_input_params.datatype = datatype;
	ccv_nnc_tensor_t* const hq_input = ccv_nnc_tensor_new(0, hq_input_params, 0);
	ccv_nnc_tensor_t* const hk_input = ccv_nnc_tensor_new(0, hk_input_params, 0);
	ccv_nnc_tensor_t* const hhead_w_input = ccv_nnc_tensor_new(0, hhead_w_input_params, 0);
	_mps_scaled_dot_product_arg_partition_from_float(datatype, hq->data.f32, hq_input, T * H * D);
	_mps_scaled_dot_product_arg_partition_from_float(datatype, hk->data.f32, hk_input, C * D);
	_mps_scaled_dot_product_arg_partition_from_float(datatype, hhead_w->data.f32, hhead_w_input, T * H);
	_mps_scaled_dot_product_arg_partition_to_float(datatype, hq_input, hq->data.f32, T * H * D);
	_mps_scaled_dot_product_arg_partition_to_float(datatype, hk_input, hk->data.f32, C * D);
	_mps_scaled_dot_product_arg_partition_to_float(datatype, hhead_w_input, hhead_w->data.f32, T * H);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(kth, 1, is_causal, compression_ratio);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hhead_w), TENSOR_LIST(href), 0);
	ccv_nnc_tensor_param_t q_params = GPU_TENSOR_NHWC(000, 32F, T, H, D);
	q_params.datatype = datatype;
	ccv_nnc_tensor_param_t k_params = GPU_TENSOR_NHWC(000, 32F, C, D);
	k_params.datatype = datatype;
	ccv_nnc_tensor_param_t head_w_params = GPU_TENSOR_NHWC(000, 32F, T, H);
	head_w_params.datatype = datatype;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const kg = ccv_nnc_tensor_new(0, k_params, 0);
	ccv_nnc_tensor_t* const head_w = ccv_nnc_tensor_new(0, head_w_params, 0);
	ccv_nnc_tensor_t* const selected = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, T, kth), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq_input, hk_input, hhead_w_input), TENSOR_LIST(q, kg, head_w), 0);
	const uint64_t old_flags = ccv_nnc_flags();
	if (force_graph)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	if (force_generic_mfa)
	{
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	}
	const int gpu_status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, kg, head_w), TENSOR_LIST(selected), 0);
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	else
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	int i;
	int mismatch = 0;
	if (gpu_status != CCV_NNC_EXEC_SUCCESS)
		mismatch = -1;
	else {
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(selected), TENSOR_LIST(hselected), 0);
		for (i = 0; i < T * kth && !mismatch; i++)
			if (hselected->data.i32[i] != href->data.i32[i])
				mismatch = i + 1;
	}
	ccv_nnc_tensor_free(hq);
	ccv_nnc_tensor_free(hk);
	ccv_nnc_tensor_free(hhead_w);
	ccv_nnc_tensor_free(hq_input);
	ccv_nnc_tensor_free(hk_input);
	ccv_nnc_tensor_free(hhead_w_input);
	ccv_nnc_tensor_free(href);
	ccv_nnc_tensor_free(hselected);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(kg);
	ccv_nnc_tensor_free(head_w);
	ccv_nnc_tensor_free(selected);
	return mismatch;
}

TEST_CASE("scaled dot product arg partition with MPSGraph")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(5, 6, 2, 4, 8, 1, 4, CCV_32F, 1, 0), 0, "MPSGraph selected ids should match CPU reference");
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(10, 2, 2, 4, 3, 1, 4, CCV_32F, 1, 0), 0, "MPSGraph selected ids should pad zero-visible causal compression rows");
}

TEST_CASE("scaled dot product arg partition with MFA DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(3, 8, 64, 128, 4, 1, 4, CCV_32F, 0, 0), 0, "MFA selected ids should match CPU reference");
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(160, 32, 64, 128, 8, 1, 4, CCV_32F, 0, 0), 0, "MFA selected ids should pad zero-visible causal compression rows");
}

TEST_CASE("scaled dot product arg partition with MFA FP16 DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(3, 8, 64, 128, 4, 1, 4, CCV_16F, 0, 0), 0, "MFA FP16 selected ids should match CPU reference");
}

TEST_CASE("scaled dot product arg partition with generic MFA FP16 DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(3, 8, 64, 128, 4, 1, 4, CCV_16F, 0, 1), 0, "generic MFA FP16 selected ids should match CPU reference");
	REQUIRE_EQ(_mps_scaled_dot_product_arg_partition_compare(160, 32, 64, 128, 8, 1, 4, CCV_16F, 0, 1), 0, "generic MFA FP16 selected ids should pad zero-visible causal compression rows");
}

TEST_CASE("scaled dot product arg partition with MFA BF16 DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int status = _mps_scaled_dot_product_arg_partition_compare(3, 8, 64, 128, 4, 1, 4, CCV_16BF, 0, 0);
	GUARD_ELSE_RETURN(status != -1);
	REQUIRE_EQ(status, 0, "MFA BF16 selected ids should match CPU reference");
}

TEST_CASE("sparse indexed attention cpu reference applies dense causal mask and sparse terminator")
{
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1, 1), 0);
	ccv_nnc_tensor_t* const dense_k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	ccv_nnc_tensor_t* const dense_v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1), 0);
	ccv_nnc_tensor_t* const sparse_k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* const sparse_v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 4, 3), 0);
	ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 4, 1, 1), 0);
	int i;
	for (i = 0; i < 4; i++)
		q->data.f32[i] = 1;
	const float dense_k_data[] = { 0, 1 };
	const float dense_v_data[] = { 10, 20 };
	const float sparse_k_data[] = { 2, 3, 4 };
	const float sparse_v_data[] = { 30, 40, 50 };
	const int indices_data[] = {
		1, -1, 0,
		2, 0, -1,
		-1, 0, 1,
		0, 1, 2,
	};
	memcpy(dense_k->data.f32, dense_k_data, sizeof(dense_k_data));
	memcpy(dense_v->data.f32, dense_v_data, sizeof(dense_v_data));
	memcpy(sparse_k->data.f32, sparse_k_data, sizeof(sparse_k_data));
	memcpy(sparse_v->data.f32, sparse_v_data, sizeof(sparse_v_data));
	memcpy(indices->data.i32, indices_data, sizeof(indices_data));
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(CMD_SPARSE_INDEXED_ATTENTION_FORWARD(1, 1, 0), ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense_k, dense_v, sparse_k, sparse_v, indices), TENSOR_LIST(out), 0), "sparse indexed attention CPU reference should run");
	float expected[4];
	for (i = 0; i < 4; i++)
	{
		float scores[5];
		float values[5];
		int count = 0;
		int dense_end = 2 - 4 + i + 1;
		dense_end = ccv_max(0, ccv_min(2, dense_end));
		int r;
		for (r = 0; r < dense_end; r++)
		{
			scores[count] = dense_k_data[r];
			values[count] = dense_v_data[r];
			++count;
		}
		for (r = 0; r < 3; r++)
		{
			const int idx = indices_data[i * 3 + r];
			if (idx < 0)
				break;
			scores[count] = sparse_k_data[idx];
			values[count] = sparse_v_data[idx];
			++count;
		}
		float maxval = -FLT_MAX;
		for (r = 0; r < count; r++)
			maxval = ccv_max(maxval, scores[r]);
		float sumval = 0;
		expected[i] = 0;
		for (r = 0; r < count; r++)
		{
			const float weight = expf(scores[r] - maxval);
			sumval += weight;
			expected[i] += values[r] * weight;
		}
		expected[i] /= sumval;
	}
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, out->data.f32, expected, 4, 1e-5, "CPU reference should apply causal dense rows while sparse ids terminate at first negative");
	ccv_nnc_tensor_free(out);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(sparse_v);
	ccv_nnc_tensor_free(sparse_k);
	ccv_nnc_tensor_free(dense_v);
	ccv_nnc_tensor_free(dense_k);
	ccv_nnc_tensor_free(q);
}

TEST_CASE("sparse indexed attention cpu reference applies attention sink to denominator only")
{
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 1), 0);
	ccv_nnc_tensor_t* const dense_k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const dense_v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const sparse_k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const sparse_v = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 1), 0);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2, 1), 0);
	ccv_nnc_tensor_t* const sinks = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1), 0);
	ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 1, 1), 0);
	q->data.f32[0] = q->data.f32[1] = 1;
	dense_k->data.f32[0] = 0;
	dense_v->data.f32[0] = 0;
	sparse_k->data.f32[0] = 0;
	sparse_v->data.f32[0] = 8;
	indices->data.i32[0] = 0;
	indices->data.i32[1] = -1;
	sinks->data.f32[0] = 0;
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(CMD_SPARSE_INDEXED_ATTENTION_FORWARD(1, 1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense_k, dense_v, sparse_k, sparse_v, indices, sinks), TENSOR_LIST(out), 0), "sparse indexed attention CPU reference with sinks should run");
	const float expected[] = { 4, 0 };
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, out->data.f32, expected, 2, 1e-5, "attention sink should add softmax mass without adding value mass");
	ccv_nnc_tensor_free(out);
	ccv_nnc_tensor_free(sinks);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(sparse_v);
	ccv_nnc_tensor_free(sparse_k);
	ccv_nnc_tensor_free(dense_v);
	ccv_nnc_tensor_free(dense_k);
	ccv_nnc_tensor_free(q);
}

static void _mps_sparse_indexed_attention_fill(const int T, const int H, const int D, const int dense_rows, const int sparse_rows, const int K, float* const q, float* const dense, float* const sparse, int* const indices, float* const sinks)
{
	int i, h, t, k;
	for (i = 0; i < T * H * D; i++)
		q[i] = (float)(((i * 17 + 11) % 97) - 48) / 256;
	for (i = 0; i < dense_rows * D; i++)
		dense[i] = (float)(((i * 19 + 7) % 89) - 44) / 256;
	for (i = 0; i < sparse_rows * D; i++)
		sparse[i] = (float)(((i * 23 + 5) % 101) - 50) / 256;
	if (K == 0)
	{
		for (t = 0; t < T; t++)
			indices[t] = -1;
	} else {
		for (t = 0; t < T; t++)
			for (k = 0; k < K; k++)
				indices[t * K + k] = (k == K - 1 && (t & 1)) ? -1 : (t * 17 + k * 13) % sparse_rows;
	}
	for (h = 0; h < H; h++)
		sinks[h] = (float)(((h * 7 + 3) % 17) - 8) / 32;
}

static int _mps_sparse_indexed_attention_compare(const int datatype, const int algorithm, const int T, const int D, const int dense_rows, const int sparse_rows, const int K, const int attention_sinks, const float tolerance, float* const max_abs_ref, float* const max_relative_ref, int* const max_idx_ref, float* const expected_ref, float* const actual_ref)
{
	const int H = 64;
	const int q_count = T * H * D;
	const int dense_count = dense_rows * D;
	const int sparse_count = sparse_rows * D;
	int status = 0;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H, D), 0);
	ccv_nnc_tensor_t* const hdense = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, dense_rows, D), 0);
	ccv_nnc_tensor_t* const hsparse = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, sparse_rows, D), 0);
	ccv_nnc_tensor_param_t hindices_params = K == 0 ? CPU_TENSOR_NHWC(32S, T) : CPU_TENSOR_NHWC(32S, T, K);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, hindices_params, 0);
	ccv_nnc_tensor_t* const hsinks = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, H), 0);
	ccv_nnc_tensor_t* const href = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H, D), 0);
	_mps_sparse_indexed_attention_fill(T, H, D, dense_rows, sparse_rows, K, hq->data.f32, hdense->data.f32, hsparse->data.f32, hindices->data.i32, hsinks->data.f32);
	ccv_nnc_tensor_param_t q_input_params = CPU_TENSOR_NHWC(32F, T, H, D);
	q_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t dense_input_params = CPU_TENSOR_NHWC(32F, dense_rows, D);
	dense_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t sparse_input_params = CPU_TENSOR_NHWC(32F, sparse_rows, D);
	sparse_input_params.datatype = datatype;
	ccv_nnc_tensor_param_t sinks_input_params = CPU_TENSOR_NHWC(32F, H);
	sinks_input_params.datatype = datatype;
	ccv_nnc_tensor_t* const hq_input = ccv_nnc_tensor_new(0, q_input_params, 0);
	ccv_nnc_tensor_t* const hdense_input = ccv_nnc_tensor_new(0, dense_input_params, 0);
	ccv_nnc_tensor_t* const hsparse_input = ccv_nnc_tensor_new(0, sparse_input_params, 0);
	ccv_nnc_tensor_t* const hsinks_input = ccv_nnc_tensor_new(0, sinks_input_params, 0);
	_mps_sdpa_store_float_as_datatype(datatype, hq->data.f32, hq_input->data.u8, q_count);
	_mps_sdpa_store_float_as_datatype(datatype, hdense->data.f32, hdense_input->data.u8, dense_count);
	_mps_sdpa_store_float_as_datatype(datatype, hsparse->data.f32, hsparse_input->data.u8, sparse_count);
	_mps_sdpa_store_float_as_datatype(datatype, hsinks->data.f32, hsinks_input->data.u8, H);
	_mps_forward_scaled_gemm_to_float(datatype, hq_input->data.u8, q_count, hq->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, hdense_input->data.u8, dense_count, hdense->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, hsparse_input->data.u8, sparse_count, hsparse->data.f32);
	_mps_forward_scaled_gemm_to_float(datatype, hsinks_input->data.u8, H, hsinks->data.f32);
	ccv_nnc_cmd_t cpu_cmd = CMD_SPARSE_INDEXED_ATTENTION_FORWARD(1.0f / sqrtf((float)D), 1, attention_sinks);
	const int cpu_status = attention_sinks ?
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hdense, hdense, hsparse, hsparse, hindices, hsinks), TENSOR_LIST(href), 0) :
		ccv_nnc_cmd_exec(cpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hdense, hdense, hsparse, hsparse, hindices), TENSOR_LIST(href), 0);
	if (cpu_status != CCV_NNC_EXEC_SUCCESS)
	{
		status = -1;
		goto cleanup_host;
	}
	ccv_nnc_tensor_param_t q_params = GPU_TENSOR_NHWC(000, 32F, T, H, D);
	q_params.datatype = datatype;
	ccv_nnc_tensor_param_t dense_params = GPU_TENSOR_NHWC(000, 32F, dense_rows, D);
	dense_params.datatype = datatype;
	ccv_nnc_tensor_param_t sparse_params = GPU_TENSOR_NHWC(000, 32F, sparse_rows, D);
	sparse_params.datatype = datatype;
	ccv_nnc_tensor_param_t sinks_params = GPU_TENSOR_NHWC(000, 32F, H);
	sinks_params.datatype = datatype;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const dense = ccv_nnc_tensor_new(0, dense_params, 0);
	ccv_nnc_tensor_t* const sparse = ccv_nnc_tensor_new(0, sparse_params, 0);
	ccv_nnc_tensor_param_t indices_params = K == 0 ? GPU_TENSOR_NHWC(000, 32S, T) : GPU_TENSOR_NHWC(000, 32S, T, K);
	ccv_nnc_tensor_t* const indices = ccv_nnc_tensor_new(0, indices_params, 0);
	ccv_nnc_tensor_t* const sinks = ccv_nnc_tensor_new(0, sinks_params, 0);
	ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, q_params, 0);
	ccv_nnc_tensor_t* const hactual = ccv_nnc_tensor_new(0, q_input_params, 0);
	if (attention_sinks)
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq_input, hdense_input, hsparse_input, hindices, hsinks_input), TENSOR_LIST(q, dense, sparse, indices, sinks), 0);
	else
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq_input, hdense_input, hsparse_input, hindices), TENSOR_LIST(q, dense, sparse, indices), 0);
	ccv_nnc_cmd_t gpu_cmd = CMD_SPARSE_INDEXED_ATTENTION_FORWARD(1.0f / sqrtf((float)D), 1, attention_sinks);
	gpu_cmd.algorithm = algorithm;
	const int gpu_status = attention_sinks ?
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices, sinks), TENSOR_LIST(out), 0) :
		ccv_nnc_cmd_exec(gpu_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, dense, dense, sparse, sparse, indices), TENSOR_LIST(out), 0);
	if (gpu_status != CCV_NNC_EXEC_SUCCESS)
	{
		status = -2;
		goto cleanup_device;
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(out), TENSOR_LIST(hactual), 0);
	float* const expected = (float*)ccmalloc(sizeof(float) * q_count);
	float* const actual = (float*)ccmalloc(sizeof(float) * q_count);
	_mps_sdpa_round_to_datatype(datatype, href->data.f32, expected, q_count);
	_mps_forward_scaled_gemm_to_float(datatype, hactual->data.u8, q_count, actual);
	float max_abs = 0;
	float max_relative = 0;
	int max_idx = 0;
	int i;
	for (i = 0; i < q_count; i++)
	{
		if (!isfinite(actual[i]))
		{
			status = -3;
			max_idx = i;
			break;
		}
		const float abs_diff = fabsf(expected[i] - actual[i]);
		const float denom = fmaxf(fmaxf(fabsf(expected[i]), fabsf(actual[i])), 1.0f);
		const float relative = abs_diff / denom;
		if (relative > max_relative)
			max_relative = relative, max_abs = abs_diff, max_idx = i;
	}
	if (status == 0 && max_relative > tolerance)
		status = 1;
	*max_abs_ref = max_abs;
	*max_relative_ref = max_relative;
	*max_idx_ref = max_idx;
	*expected_ref = expected[max_idx];
	*actual_ref = actual[max_idx];
	ccfree(actual);
	ccfree(expected);
cleanup_device:
	ccv_nnc_tensor_free(hactual);
	ccv_nnc_tensor_free(out);
	ccv_nnc_tensor_free(sinks);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(sparse);
	ccv_nnc_tensor_free(dense);
	ccv_nnc_tensor_free(q);
cleanup_host:
	ccv_nnc_tensor_free(hsinks_input);
	ccv_nnc_tensor_free(hsparse_input);
	ccv_nnc_tensor_free(hdense_input);
	ccv_nnc_tensor_free(hq_input);
	ccv_nnc_tensor_free(href);
	ccv_nnc_tensor_free(hsinks);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(hsparse);
	ccv_nnc_tensor_free(hdense);
	ccv_nnc_tensor_free(hq);
	return status;
}

TEST_CASE("sparse indexed attention with MFA FP16 DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	float max_abs, max_relative, expected, actual;
	int max_idx;
	const int sparse_status = _mps_sparse_indexed_attention_compare(CCV_16F, -1, 5, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	GUARD_ELSE_RETURN(sparse_status != -2);
	REQUIRE_EQ(sparse_status, 0, "MFA FP16 sparse indexed attention should match CPU reference");
	const int dense_status = _mps_sparse_indexed_attention_compare(CCV_16F, -1, 5, 512, 37, 1, 0, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(dense_status, 0, "MFA FP16 dense-only sparse indexed attention should match CPU reference");
	const int threadgroup16_status = _mps_sparse_indexed_attention_compare(CCV_16F, 0, 5, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(threadgroup16_status, 0, "MFA FP16 sparse indexed attention threadgroup16 should match CPU reference");
	const int threadgroup24_status = _mps_sparse_indexed_attention_compare(CCV_16F, 1, 5, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(threadgroup24_status, 0, "MFA FP16 sparse indexed attention threadgroup24 should match CPU reference");
	const int threadgroup64_status = _mps_sparse_indexed_attention_compare(CCV_16F, 3, 5, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(threadgroup64_status, 0, "MFA FP16 sparse indexed attention threadgroup64 should match CPU reference");
	const int threadgroup64_t1_status = _mps_sparse_indexed_attention_compare(CCV_16F, 3, 1, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(threadgroup64_t1_status, 0, "MFA FP16 sparse indexed attention threadgroup64 T=1 should match CPU reference");
	const int threadgroup64_d128_status = _mps_sparse_indexed_attention_compare(CCV_16F, 4, 5, 128, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE_EQ(threadgroup64_d128_status, 0, "MFA FP16 sparse indexed attention threadgroup64 D=128 should match CPU reference");
	const int generic_status = _mps_sparse_indexed_attention_compare(CCV_16F, 5, 5, 512, 4, 8, 4, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE(generic_status == 0, "generic MFA FP16 sparse indexed attention should match CPU reference (status=%d max abs %g relative %g at %d: CPU %g GPU %g)", generic_status, max_abs, max_relative, max_idx, expected, actual);
	const int generic_fp32_status = _mps_sparse_indexed_attention_compare(CCV_32F, 5, 5, 128, 7, 9, 3, 1, 1e-4, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE(generic_fp32_status == 0, "generic MFA FP32 sparse indexed attention should match CPU reference (status=%d max abs %g relative %g at %d: CPU %g GPU %g)", generic_fp32_status, max_abs, max_relative, max_idx, expected, actual);
	const int generic_t9_status = _mps_sparse_indexed_attention_compare(CCV_16F, 5, 9, 128, 7, 9, 3, 1, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE(generic_t9_status == 0, "generic MFA FP16 sparse indexed attention T=9 should match CPU reference (status=%d max abs %g relative %g at %d: CPU %g GPU %g)", generic_t9_status, max_abs, max_relative, max_idx, expected, actual);
	const int generic_no_sink_status = _mps_sparse_indexed_attention_compare(CCV_16F, 5, 5, 96, 6, 7, 2, 0, 1e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	REQUIRE(generic_no_sink_status == 0, "generic MFA FP16 sparse indexed attention without sinks should match CPU reference (status=%d max abs %g relative %g at %d: CPU %g GPU %g)", generic_no_sink_status, max_abs, max_relative, max_idx, expected, actual);
}

TEST_CASE("sparse indexed attention with MFA BF16 DS4-native shape")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SPARSE_INDEXED_ATTENTION_FORWARD, CCV_NNC_BACKEND_MPS));
	float max_abs, max_relative, expected, actual;
	int max_idx;
	const int sparse_status = _mps_sparse_indexed_attention_compare(CCV_16BF, -1, 5, 512, 4, 8, 4, 1, 5e-2, &max_abs, &max_relative, &max_idx, &expected, &actual);
	GUARD_ELSE_RETURN(sparse_status != -2);
	REQUIRE_EQ(sparse_status, 0, "MFA BF16 sparse indexed attention should match CPU reference");
}

#include "case_main.h"
