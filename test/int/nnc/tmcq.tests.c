#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include "3rdparty/dsfmt/dSFMT.h"
#include "3rdparty/sqlite3/sqlite3.h"
#include "nnc/ccv_nnc_trellis_builtin_table.inc"
#include "nnc/ccv_nnc_codebook.h"
#include "trellis_table_builtin.inc"
#include "trellis_table_alt.inc"

TEST_SETUP()
{
	ccv_nnc_init();
}

static const int tmcq_formats[9] = {
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, CCV_NNC_QX_8I_ROWWISE_TMCQ_N20,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, CCV_NNC_QX_8I_ROWWISE_TMCQ_N26,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N28, CCV_NNC_QX_8I_ROWWISE_TMCQ_N30, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32,
};

// Independent reference of the code value: x = cw * 0x83DCD12D (mod 2^32), q8 = T[2 (x >> 23) + parity],
// parity = t & 1 for weight t of a pair. The table itself is pinned by its SHA-256 below.
static int _test_tmcq_q8(const unsigned cw, const int parity)
{
	const uint32_t x = (uint32_t)((uint64_t)cw * 0x83DCD12Dull);
	return ccv_nnc_trellis_builtin_table[2 * (x >> 23) + parity];
}

static int _test_tmcq_bit(const uint8_t* const data, const size_t bit)
{
	return (data[bit / 8] >> (bit % 8)) & 1;
}

// Per weight, straight from the definition: cw = sum_j bit[(E - 15 + j) mod T] << j at E = E_{t | 1} with
// E_t = (t + 1) n / 16, then q8 = T[2 (x >> 23) + (t & 1)].
static void _test_tmcq_decode(const uint8_t* const payload, const int rows, const int cols, const int n, int* const q8)
{
	const int T = 16 * n;
	const int tile_cols = cols / 16;
	int g, t, j;
	for (g = 0; g < rows / 16 * tile_cols; g++)
	{
		const int tr = g / tile_cols, tc = g % tile_cols;
		for (t = 0; t < 256; t++)
		{
			const int end = ((t | 1) + 1) * n / 16;
			unsigned cw = 0;
			for (j = 0; j < 15; j++)
			{
				const int p = ((end - 15 + j) % T + T) % T;
				cw |= (unsigned)_test_tmcq_bit(payload, (size_t)g * T + p) << j;
			}
			q8[(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = _test_tmcq_q8(cw, t & 1);
		}
	}
}

static size_t _test_tmcq_scale_offset(const int rows, const int cols, const int n)
{
	const size_t payload = ((size_t)rows / 16 * (cols / 16) * 16 * n + 7) / 8;
	return (payload + 127) & -128;
}

static double _test_tmcq_load_scale(const uint8_t* const scales, const int datatype, const int i)
{
	float v;
	if (datatype == CCV_16F)
	{
		ccv_half_precision_to_float((const uint16_t*)scales + i, &v, 1);
		return v;
	} else if (datatype == CCV_16BF) {
		ccv_bfloat_to_float((const uint16_t*)scales + i, &v, 1);
		return v;
	} else if (datatype == CCV_32F)
		return ((const float*)scales)[i];
	return ((const double*)scales)[i];
}

static void _test_tmcq_to_double(const void* const data, const int datatype, const int count, double* const out)
{
	int i;
	for (i = 0; i < count; i++)
	{
		float v;
		if (datatype == CCV_16F)
			ccv_half_precision_to_float((const uint16_t*)data + i, &v, 1), out[i] = v;
		else if (datatype == CCV_16BF)
			ccv_bfloat_to_float((const uint16_t*)data + i, &v, 1), out[i] = v;
		else if (datatype == CCV_32F)
			out[i] = ((const float*)data)[i];
		else
			out[i] = ((const double*)data)[i];
	}
}

static void _test_tmcq_from_float(const float* const values, const int datatype, const int count, void* const out)
{
	int i;
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)out, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)out, count);
	else if (datatype == CCV_32F)
		memcpy(out, values, sizeof(float) * count);
	else
		for (i = 0; i < count; i++)
			((double*)out)[i] = values[i];
}

TEST_CASE("TMCQ code values follow the built-in pair codebook definition")
{
	// k = (cw * 0x83DCD12D) >> 23, read by hand off ccv_nnc_trellis_builtin_table.inc (16 entries per line, T[2k] and T[2k + 1]).
	REQUIRE_EQ(_test_tmcq_q8(0, 0), 27, "zero state is pair 0, first weight");
	REQUIRE_EQ(_test_tmcq_q8(0, 1), -27, "zero state is pair 0, second weight");
	// 1 * 0x83DCD12D >> 23 = 0x107 = 263 -> T[526], T[527].
	REQUIRE_EQ(_test_tmcq_q8(1, 0), -11, "unit state is pair 263");
	REQUIRE_EQ(_test_tmcq_q8(1, 1), -52, "unit state is pair 263");
	// 0x7fff * 0x83DCD12D = 0x41EDE4B9AED3 -> low 32 bits 0xE4B9AED3 >> 23 = 0x1C9 = 457 -> T[914], T[915].
	REQUIRE_EQ(_test_tmcq_q8(0x7fff, 0), 21, "all-ones state is pair 457");
	REQUIRE_EQ(_test_tmcq_q8(0x7fff, 1), -1, "all-ones state is pair 457");
	// 0x4000 * 0x83DCD12D = 0x20F7344B4000 -> 0x344B4000 >> 23 = 0x68 = 104 -> T[208], T[209].
	REQUIRE_EQ(_test_tmcq_q8(0x4000, 0), -10, "top-bit state is pair 104");
	REQUIRE_EQ(_test_tmcq_q8(0x4000, 1), -4, "top-bit state is pair 104");
	int s, min_q = 127, max_q = -127;
	for (s = 0; s < 32768; s++)
	{
		const int q0 = _test_tmcq_q8(s, 0), q1 = _test_tmcq_q8(s, 1);
		min_q = ccv_min(min_q, ccv_min(q0, q1));
		max_q = ccv_max(max_q, ccv_max(q0, q1));
	}
	REQUIRE_EQ(min_q, -127, "smallest reachable code");
	REQUIRE_EQ(max_q, 114, "largest reachable code");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tmcq_rate(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16), 16, "lowest rate");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tmcq_rate(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32), 32, "highest rate");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, 22, "codec ids follow the slots reserved for upstream formats (13 - 21)");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32, 30, "codec ids are contiguous");
}

TEST_CASE("TMCQ CPU decode of fixed payloads writes tiles back in row-major order")
{
	enum { rows = 32, cols = 512, count = rows * cols };
	int f;
	for (f = 0; f < 9; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
		const size_t size = scale_offset + rows * sizeof(float);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, count, cols), size, "size is the tile payload plus aligned row scales");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, rows, cols), tmcq_formats[f])), size, "tensor allocation agrees");
		uint8_t* const packed = cccalloc(size, 1);
		int i;
		// A structured payload: tile g's bytes are (g * 37 + byte * 11) ^ 0x5a.
		const size_t tile_bytes = 2 * n;
		for (i = 0; i < (int)(scale_offset); i++)
			packed[i] = i < rows / 16 * cols / 16 * (int)tile_bytes ? (uint8_t)(((i / tile_bytes) * 37 + (i % tile_bytes) * 11) ^ 0x5a) : 0;
		for (i = 0; i < rows; i++)
			((float*)(packed + scale_offset))[i] = (i % 2 ? -1 : 1) * (i + 1) / 64.f;
		int* const q8 = ccmalloc(sizeof(int) * count);
		_test_tmcq_decode(packed, rows, cols, n, q8);
		float* const expected = ccmalloc(sizeof(float) * count);
		float* const actual = ccmalloc(sizeof(float) * count);
		for (i = 0; i < count; i++)
			expected[i] = (float)((double)((float*)(packed + scale_offset))[i / cols] * q8[i]);
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], actual, count);
		REQUIRE_ARRAY_EQ(float, expected, actual, count, "tail-biting windows, codebook and tile order match for n = %d", n);
		// All-zero tiles are pair 0 everywhere: 27 on even columns, -27 on odd ones.
		memset(packed, 0, scale_offset);
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], actual, count);
		for (i = 0; i < count; i++)
			expected[i] = (float)((double)((float*)(packed + scale_offset))[i / cols] * (i % 2 ? -27 : 27));
		REQUIRE_ARRAY_EQ(float, expected, actual, count, "zero payload is the pair-0 code");
		ccfree(actual);
		ccfree(expected);
		ccfree(q8);
		ccfree(packed);
	}
}

TEST_CASE("TMCQ CPU decode of random payloads matches the definition bit for bit in every precision")
{
	enum { rows = 48, cols = 768, count = rows * cols };
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1013);
	int f, d, i;
	int* const q8 = ccmalloc(sizeof(int) * count);
	uint8_t* const decoded = ccmalloc(sizeof(double) * count);
	uint8_t* const expected = ccmalloc(sizeof(double) * count);
	for (f = 0; f < 9; f++)
	{
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
		for (d = 0; d < 4; d++)
		{
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], datatypes[d], count, cols);
			REQUIRE_EQ(size, scale_offset + rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "trailing scales use the source precision");
			uint8_t* const packed = ccmalloc(size);
			for (i = 0; i < (int)scale_offset; i++)
				packed[i] = (uint8_t)(dsfmt_genrand_uint32(&dsfmt) & 0xff);
			float scale_f[rows];
			for (i = 0; i < rows; i++)
				scale_f[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
			_test_tmcq_from_float(scale_f, datatypes[d], rows, packed + scale_offset);
			_test_tmcq_decode(packed, rows, cols, n, q8);
			for (i = 0; i < count; i++)
			{
				const double v = _test_tmcq_load_scale(packed + scale_offset, datatypes[d], i / cols) * q8[i];
				const float vf = (float)v;
				if (datatypes[d] == CCV_64F)
					((double*)expected)[i] = v;
				else
					_test_tmcq_from_float(&vf, datatypes[d], 1, expected + i * CCV_GET_DATA_TYPE_SIZE(datatypes[d]));
			}
			ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], decoded, count);
			REQUIRE_ARRAY_EQ(uint8_t, expected, decoded, count * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "random payload decodes exactly for n = %d, datatype %d", n, d);
			ccfree(packed);
		}
	}
	ccfree(expected);
	ccfree(decoded);
	ccfree(q8);
}

static void _test_tmcq_values(dsfmt_t* const dsfmt, float* const values, const int rows, const int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		const double row_scale = 0.02 * (1 + i % 5);
		for (j = 0; j < cols; j++)
		{
			// Sum of uniforms, a rough Gaussian, plus a few outliers.
			double v = 0;
			int k;
			for (k = 0; k < 4; k++)
				v += dsfmt_genrand_open_close(dsfmt) - 0.5;
			if ((i * 131 + j * 17) % 251 == 0)
				v *= 6;
			values[i * cols + j] = (float)(v * row_scale);
		}
	}
}

static double _test_tmcq_sse(const float* const x, const float* const xhat, const int count)
{
	double sse = 0;
	int i;
	for (i = 0; i < count; i++)
	{
		const double d = (double)x[i] - xhat[i];
		sse += d * d;
	}
	return sse;
}

// Nearest baseline: walk each tile once pair by pair and greedily take the fresh bits whose code pair is
// closest to the target pair (starting from state 0), keep the decoded codes, and give each row its
// least-squares scale.
static void _test_tmcq_nearest(const float* const x, const int rows, const int cols, const int n, float* const xhat)
{
	const int T = 16 * n;
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	uint8_t* const payload = cccalloc(scale_offset, 1);
	double cb = 0;
	int s;
	for (s = 0; s < 32768; s++)
		cb += (double)_test_tmcq_q8(s, 0) * _test_tmcq_q8(s, 0) + (double)_test_tmcq_q8(s, 1) * _test_tmcq_q8(s, 1);
	cb = sqrt(cb / 65536);
	double* const scale = ccmalloc(sizeof(double) * rows);
	int i, j, g, t;
	for (i = 0; i < rows; i++)
	{
		double sum = 0;
		for (j = 0; j < cols; j++)
			sum += (double)x[i * cols + j] * x[i * cols + j];
		scale[i] = sqrt(sum / cols) / cb;
	}
	const int tile_cols = cols / 16;
	for (g = 0; g < rows / 16 * tile_cols; g++)
	{
		const int tr = g / tile_cols, tc = g % tile_cols;
		unsigned state = 0;
		int prev_end = 0;
		for (t = 0; t < 256; t += 2)
		{
			// Weights t and t + 1 share a tile row.
			const int r = tr * 16 + t / 16;
			const double y0 = scale[r] > 0 ? x[r * cols + tc * 16 + t % 16] / scale[r] : 0;
			const double y1 = scale[r] > 0 ? x[r * cols + tc * 16 + t % 16 + 1] / scale[r] : 0;
			const int end = (t + 2) * n / 16, f = end - prev_end;
			unsigned best = 0;
			double best_d = DBL_MAX;
			unsigned h;
			for (h = 0; h < (1u << f); h++)
			{
				const unsigned next = (state >> f) | (h << (15 - f));
				const double d0 = y0 - _test_tmcq_q8(next, 0), d1 = y1 - _test_tmcq_q8(next, 1);
				const double d = d0 * d0 + d1 * d1;
				if (d < best_d)
					best_d = d, best = next;
			}
			state = best;
			int p;
			for (p = prev_end; p < end; p++)
				if ((state >> (14 - (end - 1 - p))) & 1)
					payload[((size_t)g * T + p) / 8] |= 1 << (((size_t)g * T + p) % 8);
			prev_end = end;
		}
	}
	int* const q8 = ccmalloc(sizeof(int) * rows * cols);
	_test_tmcq_decode(payload, rows, cols, n, q8);
	for (i = 0; i < rows; i++)
	{
		double num = 0, den = 0;
		for (j = 0; j < cols; j++)
		{
			num += (double)q8[i * cols + j] * x[i * cols + j];
			den += (double)q8[i * cols + j] * q8[i * cols + j];
		}
		const double sr = den > 0 ? num / den : 0;
		for (j = 0; j < cols; j++)
			xhat[i * cols + j] = (float)(sr * q8[i * cols + j]);
	}
	ccfree(q8);
	ccfree(scale);
	ccfree(payload);
}

TEST_CASE("TMCQ quantize then dequantize stays within the rate's error bound")
{
	enum { rows = 32, cols = 512, count = rows * cols };
	// Upper bounds on sum (x - x_hat)^2 / sum x^2 for rates n = 16 .. 32 (n / 16 bits per weight).
	const double bounds[9] = {0.30, 0.27, 0.235, 0.21, 0.185, 0.165, 0.15, 0.135, 0.125};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2027);
	float* const values = ccmalloc(sizeof(float) * count);
	float* const decoded = ccmalloc(sizeof(float) * count);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	double energy = 0;
	int i, f;
	for (i = 0; i < count; i++)
		energy += (double)values[i] * values[i];
	double previous = DBL_MAX;
	for (f = 0; f < 9; f++)
	{
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size + 16);
		memset(packed, 0xa5, size + 16);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, tmcq_formats[f], 0, 0, 0, 0, packed, size - 1), 0, "short output is rejected");
		for (i = 0; i < size + 16; i++)
			REQUIRE_EQ(packed[i], 0xa5, "rejected output stays unchanged");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, tmcq_formats[f], 0, 0, 0, 0, packed, size), size, "quantizer returns exact size");
		for (i = 0; i < 16; i++)
			REQUIRE_EQ(packed[size + i], 0xa5, "quantizer preserves output guard bytes");
		ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], decoded, count);
		const double relative = _test_tmcq_sse(values, decoded, count) / energy;
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
		float* const nearest = ccmalloc(sizeof(float) * count);
		_test_tmcq_nearest(values, rows, cols, n, nearest);
		const double nearest_relative = _test_tmcq_sse(values, nearest, count) / energy;
		ccfree(nearest);
		printf("\n\tTMCQ n = %d relative error %.6f (greedy nearest %.6f)", n, relative, nearest_relative);
		REQUIRE(relative < bounds[f], "n = %d relative error %g within %g", n, relative, bounds[f]);
		REQUIRE(relative < nearest_relative, "trellis beats the greedy nearest walk for n = %d", n);
		REQUIRE(relative < previous, "error falls as the rate rises");
		previous = relative;
		ccfree(packed);
	}
	ccfree(decoded);
	ccfree(values);
}

TEST_CASE("TMCQ quantizes every precision with identical codes and source-precision scales")
{
	enum { rows = 32, cols = 256, count = rows * cols };
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	float values[count];
	int i, d;
	// Multiples of 1/64 below 4 in magnitude are exact in every precision, so all four see the same input.
	for (i = 0; i < count; i++)
		values[i] = i < cols ? 0 : (float)(((i * 7919) % 255) - 127) / 64;
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20;
	const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	uint8_t* const reference = ccmalloc(scale_offset);
	int* const q8 = ccmalloc(sizeof(int) * count);
	for (d = 0; d < 4; d++)
	{
		const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
		uint8_t* const source = ccmalloc(element_size * count);
		uint8_t* const decoded = ccmalloc(element_size * count);
		_test_tmcq_from_float(values, datatypes[d], count, source);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatypes[d], count, cols);
		REQUIRE_EQ(size, scale_offset + rows * element_size, "size includes source-precision scales");
		uint8_t* const packed = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "quantizes datatype %d", d);
		if (d == 0)
			memcpy(reference, packed, scale_offset);
		else
			REQUIRE_ARRAY_EQ(uint8_t, reference, packed, scale_offset, "codes are fitted in double and match across precisions");
		ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, format, decoded, count);
		_test_tmcq_decode(packed, rows, cols, n, q8);
		double* const out = ccmalloc(sizeof(double) * count);
		_test_tmcq_to_double(decoded, datatypes[d], count, out);
		double err = 0, energy = 0;
		for (i = 0; i < count; i++)
		{
			const double scale = _test_tmcq_load_scale(packed + scale_offset, datatypes[d], i / cols);
			const double v = scale * q8[i];
			const double expected = datatypes[d] == CCV_64F ? v : datatypes[d] == CCV_32F ? (float)v : out[i];
			if (datatypes[d] == CCV_64F || datatypes[d] == CCV_32F)
				REQUIRE_EQ(out[i], expected, "decoded value is stored scale times code");
			err += (out[i] - values[i]) * (out[i] - values[i]);
			energy += (double)values[i] * values[i];
		}
		for (i = 0; i < cols; i++)
			REQUIRE_EQ(out[i], 0, "zero row decodes to exact zero");
		REQUIRE(err / energy < 0.2, "datatype %d error %g is bounded", d, err / energy);
		ccfree(out);
		ccfree(packed);
		ccfree(decoded);
		ccfree(source);
	}
	ccfree(q8);
	ccfree(reference);
}

TEST_CASE("TMCQ rejects invalid shapes and H256")
{
	enum { cols = 256 };
	static float values[64 * 512];
	static uint8_t output[64 * 512];
	int i;
	for (i = 0; i < 64 * 512; i++)
		values[i] = sinf(i * 0.1f);
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	// Rows not a multiple of 16, K not a multiple of 16 (whole 16 x 16 tiles; K % 256 is not needed).
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 15 * cols, cols), 0, "15 rows have no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 8, 8), 0, "K = 8 has no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 264, 264), 0, "K = 264 has no size");
	REQUIRE(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 128, 128) > 0, "K = 128 is valid");
	REQUIRE(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * 272, 272) > 0, "K = 272 is valid");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, CCV_32F, 16 * cols, cols), 0, "H256 has no size");
	REQUIRE(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * cols, cols) > 0, "16 x 256 is valid");
	memset(output, 0xa5, sizeof(output));
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 15 * cols, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "15 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 24 * cols, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "24 rows must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * 8, 8, format, 0, 0, 0, 0, output, sizeof(output)), 0, "K = 8 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * 392, 392, format, 0, 0, 0, 0, output, sizeof(output)), 0, "K = 392 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * cols, cols, format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, 0, 0, 0, 0, output, sizeof(output)), 0, "H256 must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 16 * cols, cols, format, 0, 0, 0, 0, output, ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, 16 * cols, cols) - 1), 0, "short output must fail");
	for (i = 0; i < (int)sizeof(output); i++)
		REQUIRE_EQ(output[i], 0xa5, "rejected calls leave the output untouched");
}

TEST_CASE("TMCQ and FTQ quantizers return 0 with an imatrix, a Hessian or a codebook slot, FTQ also for input other than FP16")
{
	enum { rows = 16, cols = 256, count = rows * cols };
	static float values[count];
	static uint16_t half[count];
	static float hessian[cols * cols];
	static uint8_t output[count * 4];
	float imatrix[cols];
	int i, f;
	for (i = 0; i < count; i++)
		values[i] = sinf(i * 0.1f) * 0.05f;
	ccv_float_to_half_precision(values, half, count);
	for (i = 0; i < cols; i++)
		imatrix[i] = 1, hessian[i * cols + i] = 1;
	const int formats[2] = {CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, CCV_NNC_QX_8I_ROWWISE_FTQ_N20};
	for (f = 0; f < 2; f++)
	{
		const int format = formats[f];
		const int datatype = f ? CCV_16F : CCV_32F;
		const void* const input = f ? (const void*)half : (const void*)values;
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatype, count, cols);
		REQUIRE(size > 0 && size <= sizeof(output), "16 x 256 has a size");
		memset(output, 0xa5, sizeof(output));
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(input, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format, imatrix, cols, 0, 0, output, size), 0, "an imatrix must fail");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(input, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, hessian, cols * cols, output, size), 0, "a Hessian must fail");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(input, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format | (1 << CCV_NNC_CODEBOOK_SLOT_SHIFT), 0, 0, 0, 0, output, size), 0, "a codebook slot must fail");
		if (f)
		{
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "FTQ from 32F input must fail");
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count - cols, cols, format, 0, 0, 0, 0, output, sizeof(output)), 0, "FTQ with 15 rows must fail");
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256, 0, 0, 0, 0, output, sizeof(output)), 0, "FTQ with H256 must fail");
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, output, size - 1), 0, "FTQ with a short output must fail");
		}
		for (i = 0; i < (int)sizeof(output); i++)
			REQUIRE_EQ(output[i], 0xa5, "rejected calls leave the output untouched");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(input, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, output, size), size, "the unweighted call quantizes");
	}
}

TEST_CASE("TMCQ metadata and payload survive tensor persistence")
{
	enum { rows = 32, cols = 256 };
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i, f;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = sinf(i * 0.37f) * (1 + i % 3);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	for (f = 0; f < 9; f += 4)
	{
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq_formats[f]), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
		REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], CCV_32F, rows * cols, cols), "tensor size matches");
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, tmcq_formats[f], 0, 0, 0, 0, packed->data.u8, size), size, "allocated tensor quantizes");
		char name[32];
		snprintf(name, sizeof(name), "tmcq_%d", f);
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "TMCQ tensor writes");
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ tensor reads");
		REQUIRE_EQ(restored->info.datatype, packed->info.datatype, "source precision persists");
		REQUIRE_EQ(restored->info.reserved, tmcq_formats[f], "codec persists");
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "size persists");
		REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "payload and trailing scales persist");
		float a[rows * cols], b[rows * cols];
		ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], a, rows * cols);
		ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, restored->info.reserved, b, rows * cols);
		REQUIRE_ARRAY_EQ(float, a, b, rows * cols, "restored tensor decodes the same");
		ccv_nnc_tensor_free(restored);
		ccv_nnc_tensor_free(packed);
	}
	sqlite3_close(db);
	ccv_nnc_tensor_free(source);
}

// Pair j covers weights 2j and 2j + 1, its window ends at E_j = (j + 1) n / 8 and it shifts in
// f_j = E_j - E_{j-1} fresh bits (E_{-1} = 0).
static int _test_trellis_fresh(const int j, const int n)
{
	return (j + 1) * n / 8 - j * n / 8;
}

// Straightforward pair Viterbi over one tile of W-bit states in processing order j' = 0..127,
// j = (j' + rotation) mod 128. start < 0: free ends. bp is 128 x 2^(W - 2) indexed by j', each up to f <= 11 bits (n88).
// Returns the end state.
static int _test_trellis_pass(const int window, const double* const y, const double* const w, const int n, const int rotation, const int start, uint16_t* const bp, double* const cost_out)
{
	const int states = 1 << window, stride = 1 << (window - 2);
	double* cost = ccmalloc(sizeof(double) * states);
	double* next = ccmalloc(sizeof(double) * states);
	int* const best = ccmalloc(sizeof(int) * stride);
	int s, step, low;
	for (s = 0; s < states; s++)
		cost[s] = start < 0 || s == start ? 0 : INFINITY;
	for (step = 0; step < 128; step++)
	{
		const int j = (step + rotation) % 128;
		const int f = _test_trellis_fresh(j, n);
		for (low = 0; low < (1 << (window - f)); low++)
		{
			int h;
			best[low] = low << f;
			for (h = 1; h < (1 << f); h++)
				if (cost[(low << f) | h] < cost[best[low]])
					best[low] = (low << f) | h;
			bp[step * stride + low] = best[low] & ((1 << f) - 1);
		}
		for (s = 0; s < states; s++)
		{
			const double d0 = y[2 * j] - _test_tmcq_q8(s, 0);
			const double d1 = y[2 * j + 1] - _test_tmcq_q8(s, 1);
			next[s] = cost[best[s & ((1 << (window - f)) - 1)]] + (w[2 * j] * (d0 * d0) + w[2 * j + 1] * (d1 * d1));
		}
		double* const swap = cost;
		cost = next;
		next = swap;
	}
	int end = start;
	if (start < 0)
		for (s = 1, end = 0; s < states; s++)
			if (cost[s] < cost[end])
				end = s;
	*cost_out = cost[end];
	ccfree(best);
	ccfree(next);
	ccfree(cost);
	return end;
}

static int _test_trellis_trace(const int window, const uint16_t* const bp, const int n, const int rotation, const int last, const int first, int s, int* const states)
{
	const int stride = 1 << (window - 2);
	int step;
	for (step = last; step >= first; step--)
	{
		const int j = (step + rotation) % 128;
		const int f = _test_trellis_fresh(j, n);
		if (states)
			states[j] = s;
		const int low = s & ((1 << (window - f)) - 1);
		s = (low << f) | bp[step * stride + low];
	}
	return s;
}

// Tail-biting tile fit: mid = 0 is the plain two-pass (anchor s_127, the wrap-around), mid = 1 the mid
// anchor (s_63, the window ending at E_63 = 8n).
static double _test_trellis_fit_tile(const int window, const double* const y, const double* const w, const int n, const int mid, int* const states)
{
	uint16_t* const bp = ccmalloc(sizeof(uint16_t) * (128 << (window - 2)));
	double cost;
	int s_star = _test_trellis_pass(window, y, w, n, 0, -1, bp, &cost);
	if (mid)
		s_star = _test_trellis_trace(window, bp, n, 0, 127, 64, s_star, 0);
	const int rotation = mid ? 64 : 0;
	_test_trellis_pass(window, y, w, n, rotation, s_star, bp, &cost);
	const int s = _test_trellis_trace(window, bp, n, rotation, 127, 0, s_star, states);
	assert(s == s_star);
	(void)s;
	ccfree(bp);
	return cost;
}

static size_t _test_ftq_scale_offset(const int rows, const int cols, const int n)
{
	return (32 + (size_t)rows / 16 * (cols / 16) * 2 * n + 127) & -128;
}

// Unweighted trellis encoder written out from the format, with either anchor. W = 15 writes the TMCQ 32F output
// (code from bit 0, float row scales); W = 16 the FTQ blob (CCVFTQSR header, code from bit 256, FP16 row scales).
// Row scale s = rms(row) / cb_rms, every pair weighted by s^2. Returns the summed Viterbi tile costs.
static double _test_trellis_encode(const float* const x, const int rows, const int cols, const int n, const int window, const int mid, uint8_t* const output)
{
	const int T = 16 * n;
	const int ftq = window == 16;
	const size_t base = ftq ? 256 : 0;
	const size_t scale_offset = ftq ? _test_ftq_scale_offset(rows, cols, n) : _test_tmcq_scale_offset(rows, cols, n);
	memset(output, 0, scale_offset);
	if (ftq)
	{
		const uint32_t header[6] = { rows, cols, T, 0, 9u << 8, 0 };
		memcpy(output, "CCVFTQSR", 8);
		memcpy(output + 8, header, sizeof(header));
	}
	double cb = 0;
	int s, i, j, k, tr, tc, t;
	for (s = 0; s < (1 << window); s++)
		cb += (double)_test_tmcq_q8(s, 0) * _test_tmcq_q8(s, 0) + (double)_test_tmcq_q8(s, 1) * _test_tmcq_q8(s, 1);
	cb = sqrt(cb / (2 << window));
	double total = 0;
	for (tr = 0; tr < rows / 16; tr++)
	{
		double scale[16];
		for (i = 0; i < 16; i++)
		{
			double sum = 0;
			for (k = 0; k < cols; k++)
				sum += (double)x[(tr * 16 + i) * cols + k] * x[(tr * 16 + i) * cols + k];
			const double sr = sqrt(sum / cols) / cb;
			scale[i] = sr > 0 && isfinite(sr) ? sr : 0;
			const int r = tr * 16 + i;
			if (ftq)
			{
				const float sf = (float)scale[i];
				ccv_float_to_half_precision(&sf, (uint16_t*)(output + scale_offset) + r, 1);
			} else
				((float*)(output + scale_offset))[r] = (float)scale[i];
		}
		for (tc = 0; tc < cols / 16; tc++)
		{
			double y[256], w[256];
			for (i = 0; i < 16; i++)
				for (j = 0; j < 16; j++)
				{
					y[i * 16 + j] = scale[i] > 0 ? x[(tr * 16 + i) * cols + tc * 16 + j] / scale[i] : 0;
					w[i * 16 + j] = scale[i] * scale[i];
				}
			int states[128];
			total += _test_trellis_fit_tile(window, y, w, n, mid, states);
			const size_t g = (size_t)tr * (cols / 16) + tc;
			int prev_end = 0;
			for (t = 0; t < 128; t++)
			{
				const int end = (t + 1) * n / 8;
				int p;
				for (p = prev_end; p < end; p++)
					if ((states[t] >> (window - 1 - (end - 1 - p))) & 1)
						output[(base + g * T + p) / 8] |= 1 << ((base + g * T + p) % 8);
				prev_end = end;
			}
		}
	}
	return total;
}

TEST_CASE("TMCQ and FTQ mid anchor matches an independent encoder and costs no more than the plain two-pass fit")
{
	enum { rows = 32, cols = 256, count = rows * cols };
	static const struct {
		int format;
		int window;
	} cases[7] = {
		{CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, 15},
		{CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, 15},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N20, 16},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N56, 16},
		// Above 4 bpw: 9 or 10 fresh bits a pair (n66 / n78), 11 (n88).
		{CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (66 - 8) / 2, 16},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (78 - 8) / 2, 16},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (88 - 8) / 2, 16},
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4242);
	float* const values = ccmalloc(sizeof(float) * count);
	uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	// FP16-exact values, so the FTQ (FP16 input) and TMCQ (32F input) cases see the same weights.
	ccv_float_to_half_precision(values, half, count);
	ccv_half_precision_to_float(half, values, count);
	int c;
	for (c = 0; c < 7; c++)
	{
		const int format = cases[c].format;
		const int ftq = cases[c].window == 16;
		const int n = ftq ? ccv_nnc_8i_rowwise_x_ftq_rate(format) : ccv_nnc_8i_rowwise_x_tmcq_rate(format);
		const int datatype = ftq ? CCV_16F : CCV_32F;
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatype, count, cols);
		uint8_t* const packed = ccmalloc(size);
		uint8_t* const mid = ccmalloc(size);
		uint8_t* const plain = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(ftq ? (const void*)half : (const void*)values, datatype, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "quantization succeeds");
		const double mid_cost = _test_trellis_encode(values, rows, cols, n, cases[c].window, 1, mid);
		const double plain_cost = _test_trellis_encode(values, rows, cols, n, cases[c].window, 0, plain);
		REQUIRE_ARRAY_EQ(uint8_t, packed, mid, size, "library output matches the independent encoder for %s n = %d", ftq ? "FTQ" : "TMCQ", n);
		printf("\n\t%s n = %d trellis cost mid %.6g plain %.6g (%+.3f%%)", ftq ? "FTQ" : "TMCQ", n, mid_cost, plain_cost, (mid_cost / plain_cost - 1) * 100);
		REQUIRE(mid_cost <= plain_cost, "mid anchor trellis cost %g is no more than plain %g for n = %d", mid_cost, plain_cost, n);
		ccfree(plain);
		ccfree(mid);
		ccfree(packed);
	}
	ccfree(half);
	ccfree(values);
}

// FIPS 180-4 SHA-256, for the golden payload digests below.
static void _test_sha256(const uint8_t* const data, const size_t length, char hex[65])
{
	static const uint32_t k[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
	};
	uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
	const size_t padded = (length + 9 + 63) / 64 * 64;
	uint8_t* const m = cccalloc(padded, 1);
	memcpy(m, data, length);
	m[length] = 0x80;
	const uint64_t bits = (uint64_t)length * 8;
	int i;
	for (i = 0; i < 8; i++)
		m[padded - 1 - i] = (uint8_t)(bits >> (8 * i));
	size_t c;
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
	for (c = 0; c < padded; c += 64)
	{
		uint32_t w[64];
		for (i = 0; i < 16; i++)
			w[i] = ((uint32_t)m[c + 4 * i] << 24) | ((uint32_t)m[c + 4 * i + 1] << 16) | ((uint32_t)m[c + 4 * i + 2] << 8) | m[c + 4 * i + 3];
		for (i = 16; i < 64; i++)
		{
			const uint32_t s0 = ROTR(w[i - 15], 7) ^ ROTR(w[i - 15], 18) ^ (w[i - 15] >> 3);
			const uint32_t s1 = ROTR(w[i - 2], 17) ^ ROTR(w[i - 2], 19) ^ (w[i - 2] >> 10);
			w[i] = w[i - 16] + s0 + w[i - 7] + s1;
		}
		uint32_t a = h[0], b = h[1], cc = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
		for (i = 0; i < 64; i++)
		{
			const uint32_t t1 = hh + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
			const uint32_t t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
			hh = g, g = f, f = e, e = d + t1, d = cc, cc = b, b = a, a = t1 + t2;
		}
		h[0] += a, h[1] += b, h[2] += cc, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
	}
#undef ROTR
	for (i = 0; i < 8; i++)
		snprintf(hex + i * 8, 9, "%08x", h[i]);
	ccfree(m);
}

// Portable generator for the golden inputs: xorshift32 from 0x12345678, u = (x >> 8) - 2^23, v = u / 2^23.
static uint32_t _test_tmcq_xorshift(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return *state = x;
}

static float _test_tmcq_golden_uniform(uint32_t* const state)
{
	return (float)((int32_t)(_test_tmcq_xorshift(state) >> 8) - 8388608) / 8388608.f;
}

TEST_CASE("built-in and alternate tables match their frozen SHA-256")
{
	REQUIRE_EQ(sizeof(ccv_nnc_trellis_builtin_table), 1024, "512 pairs of int8");
	char digest[65];
	_test_sha256((const uint8_t*)ccv_nnc_trellis_builtin_table, sizeof(ccv_nnc_trellis_builtin_table), digest);
	REQUIRE(strcmp(digest, "830ff43f1222371bb131cfd15d12f26a3881e4ba96584f1484345cb0f1605a3a") == 0, "built-in table digest %s", digest);
	REQUIRE_EQ(sizeof(test_trellis_table_alt), 16 + 1024, "the alternate asset holds 512 pairs of int8");
	_test_sha256(test_trellis_table_alt + 16, 1024, digest);
	REQUIRE(strcmp(digest, "d58cf36d15b31e5b121334f704b15ad99bc39837ac5c096f0a37b7632763f34c") == 0, "alternate table digest %s", digest);
}

TEST_CASE("TMCQ encoder golden vectors")
{
	// Input: W[16][256] = uniform() (float32) from one xorshift32 stream seeded 0x12345678, little-endian digest
	// 2457d29c78ab867fd09946a2ca8b1c18baf38290b88174728302da8bdb2354e7. TMCQ encodes W (32F output: code, zero
	// padding to 128 bytes, float row scales). The digests cover the whole output, which the independent encoder above reproduces byte for byte.
	enum { rows = 16, cols = 256, count = rows * cols };
	static const struct {
		int format;
		const char* digest;
	} cases[3] = {
		{CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, "cff83612a4db5220f0ae3d87a4e228b0caa7072d06dd9a0d64853a4821447a0b"},
		{CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, "4a254f285455ef355284e5988afc9813ebf0e066c7cb99f9f93fa89d9e22a35c"},
		{CCV_NNC_QX_8I_ROWWISE_TMCQ_N32, "bcea82123cef5d13e5de5f15cb49c93e252161f70bd8b89bdb483c2431f469ee"},
	};
	char abc[65];
	_test_sha256((const uint8_t*)"abc", 3, abc);
	REQUIRE(strcmp(abc, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0, "SHA-256 self check");
	uint32_t state = 0x12345678;
	float* const values = ccmalloc(sizeof(float) * count);
	int i, c;
	for (i = 0; i < count; i++)
		values[i] = _test_tmcq_golden_uniform(&state);
	char digest[65];
	_test_sha256((const uint8_t*)values, sizeof(float) * count, digest);
	REQUIRE(strcmp(digest, "2457d29c78ab867fd09946a2ca8b1c18baf38290b88174728302da8bdb2354e7") == 0, "golden input digest %s", digest);
	for (c = 0; c < 3; c++)
	{
		const int format = cases[c].format;
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "golden input quantizes");
		_test_sha256(packed, size, digest);
		printf("\n\tTMCQ golden n = %d size %zu sha256 %s", n, size, digest);
		REQUIRE(strcmp(digest, cases[c].digest) == 0, "golden digest for n = %d", n);
		uint8_t* const independent = ccmalloc(size);
		_test_trellis_encode(values, rows, cols, n, 15, 1, independent);
		REQUIRE_ARRAY_EQ(uint8_t, packed, independent, size, "independent encoder reproduces the golden output for n = %d", n);
		ccfree(independent);
		ccfree(packed);
	}
	ccfree(values);
}

TEST_CASE("FTQ encoder golden vectors")
{
	// The TMCQ golden input W rounded to FP16; FTQ encodes it into the whole CCVFTQSR blob (header, code, zero padding
	// to 128 bytes, FP16 row scales). The digests cover the whole output, which the independent encoder above
	// reproduces byte for byte.
	enum { rows = 16, cols = 256, count = rows * cols };
	static const struct {
		int format;
		const char* digest;
	} cases[4] = {
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N16, "7881fcfd31389285cfbcce24b607bc0718d3b2a4b78a3af6f76dad19a64153e4"},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N32, "4996f179e57448b9acd3bf311b408ec11d2a722c29b4ab31c701e0aaba63bd52"},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N48, "9c294d6526c5bfd599637e7ae126797d399bb3123e68802dce3d91b5e431cdb0"},
		{CCV_NNC_QX_8I_ROWWISE_FTQ_N64, "feb3595ff4fd4d974dbcc23cbd2c89c843b60975698bd8b5b45b54daada0ba98"},
	};
	uint32_t state = 0x12345678;
	float* const values = ccmalloc(sizeof(float) * count);
	float* const rounded = ccmalloc(sizeof(float) * count);
	uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
	int i, c;
	for (i = 0; i < count; i++)
		values[i] = _test_tmcq_golden_uniform(&state);
	char digest[65];
	_test_sha256((const uint8_t*)values, sizeof(float) * count, digest);
	REQUIRE(strcmp(digest, "2457d29c78ab867fd09946a2ca8b1c18baf38290b88174728302da8bdb2354e7") == 0, "golden input digest %s", digest);
	ccv_float_to_half_precision(values, half, count);
	ccv_half_precision_to_float(half, rounded, count);
	for (c = 0; c < 4; c++)
	{
		const int format = cases[c].format;
		const int n = ccv_nnc_8i_rowwise_x_ftq_rate(format);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, count, cols);
		uint8_t* const packed = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "golden input quantizes");
		_test_sha256(packed, size, digest);
		printf("\n\tFTQ golden n = %d size %zu sha256 %s", n, size, digest);
		REQUIRE(strcmp(digest, cases[c].digest) == 0, "golden digest for FTQ n = %d", n);
		uint8_t* const independent = ccmalloc(size);
		_test_trellis_encode(rounded, rows, cols, n, 16, 1, independent);
		REQUIRE_ARRAY_EQ(uint8_t, packed, independent, size, "independent encoder reproduces the golden output for FTQ n = %d", n);
		ccfree(independent);
		ccfree(packed);
	}
	ccfree(half);
	ccfree(rounded);
	ccfree(values);
}



// _test_tmcq_decode with a pair table of 2^log2_entries pairs: k = (cw * 0x83DCD12D mod 2^32) >> (32 - log2_entries).
static void _test_tmcq_decode_table(const uint8_t* const payload, const int rows, const int cols, const int n, const int8_t* const table, const int log2_entries, int* const q8)
{
	const int T = 16 * n;
	const int tile_cols = cols / 16;
	int g, t, j;
	for (g = 0; g < rows / 16 * tile_cols; g++)
	{
		const int tr = g / tile_cols, tc = g % tile_cols;
		for (t = 0; t < 256; t++)
		{
			const int end = ((t | 1) + 1) * n / 16;
			unsigned cw = 0;
			for (j = 0; j < 15; j++)
			{
				const int p = ((end - 15 + j) % T + T) % T;
				cw |= (unsigned)_test_tmcq_bit(payload, (size_t)g * T + p) << j;
			}
			const uint32_t x = (uint32_t)((uint64_t)cw * 0x83DCD12Dull);
			q8[(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = table[2 * (x >> (32 - log2_entries)) + (t & 1)];
		}
	}
}

static void _test_store_u32(uint8_t* const p, const uint32_t v)
{
	p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8), p[2] = (uint8_t)(v >> 16), p[3] = (uint8_t)(v >> 24);
}

TEST_CASE("codebook container matches the built-in table asset byte for byte")
{
	const int slot = ccv_nnc_codebook_register_serialized(test_trellis_table_builtin, sizeof(test_trellis_table_builtin));
	REQUIRE(slot >= 1, "the built-in table registers, slot %d", slot);
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(test_trellis_table_builtin, sizeof(test_trellis_table_builtin)), slot, "the same content takes the same slot");
	int entries = 0;
	uint32_t crc = 0;
	const int8_t* const table = ccv_nnc_codebook_get(slot, &entries, &crc);
	REQUIRE(table, "the slot has its table");
	REQUIRE_EQ(entries, 512, "the built-in table has 512 pairs");
	REQUIRE_EQ(crc, 0x4b7574dcu, "table crc32 matches zlib.crc32 of the asset data");
	char digest[65];
	_test_sha256((const uint8_t*)table, 1024, digest);
	REQUIRE(strcmp(digest, "830ff43f1222371bb131cfd15d12f26a3881e4ba96584f1484345cb0f1605a3a") == 0, "table digest %s", digest);
	uint8_t serialized[16 + 4096];
	REQUIRE_EQ(ccv_nnc_codebook_serialize(slot, serialized, sizeof(serialized)), sizeof(test_trellis_table_builtin), "serialized size");
	REQUIRE_EQ(memcmp(serialized, test_trellis_table_builtin, sizeof(test_trellis_table_builtin)), 0, "ccv serializes exactly like the asset container");
	REQUIRE_EQ(ccv_nnc_codebook_serialize(slot, serialized, sizeof(test_trellis_table_builtin) - 1), 0, "short buffer is refused");
	// The built-in table is slot 0: the same 1024 bytes and zlib crc32 as the asset.
	const int8_t* const builtin = ccv_nnc_codebook_get(0, &entries, &crc);
	REQUIRE(builtin && memcmp(builtin, ccv_nnc_trellis_builtin_table, sizeof(ccv_nnc_trellis_builtin_table)) == 0, "slot 0 is the built-in table");
	REQUIRE(memcmp(builtin, test_trellis_table_builtin + 16, 1024) == 0, "slot 0 is the asset table");
	REQUIRE_EQ(entries, 512, "the built-in table has 512 pairs");
	REQUIRE_EQ(crc, 0x4b7574dcu, "built-in table crc32");
	// The alternate table registers as a file codebook with its bytes and crc32.
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	REQUIRE(slot_alt >= 1, "the alternate table registers, slot %d", slot_alt);
	const int8_t* const alt = ccv_nnc_codebook_get(slot_alt, &entries, &crc);
	REQUIRE(alt && memcmp(alt, test_trellis_table_alt + 16, 1024) == 0, "the slot holds the alternate table");
	REQUIRE_EQ(entries, 512, "the alternate table has 512 pairs");
	REQUIRE_EQ(crc, 0xfd995f08u, "alternate table crc32");
	REQUIRE(ccv_nnc_codebook_get(255, 0, 0) == 0, "unregistered slots have no table");
	// Corrupt containers.
	uint8_t bad[sizeof(test_trellis_table_builtin)];
	memcpy(bad, test_trellis_table_builtin, sizeof(bad));
	bad[0] = 'X';
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(bad, sizeof(bad)), -1, "wrong magic fails");
	memcpy(bad, test_trellis_table_builtin, sizeof(bad));
	bad[100] ^= 1;
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(bad, sizeof(bad)), -1, "wrong crc fails");
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(test_trellis_table_builtin, sizeof(test_trellis_table_builtin) - 1), -1, "truncated container fails");
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(test_trellis_table_builtin, 15), -1, "truncated header fails");
	memcpy(bad, test_trellis_table_builtin, sizeof(bad));
	_test_store_u32(bad + 8, 1024);
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(bad, sizeof(bad)), -1, "entries that do not match the size fail");
	memcpy(bad, test_trellis_table_builtin, sizeof(bad));
	_test_store_u32(bad + 8, 513);
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(bad, 16 + 1026), -1, "entries other than 512 / 1024 / 2048 fail");
	memcpy(bad, test_trellis_table_builtin, sizeof(bad));
	bad[16 + 7] = 0x80;
	_test_store_u32(bad + 12, ccv_nnc_codebook_crc32(bad + 16, 1024));
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(bad, sizeof(bad)), -1, "-128 fails even with a matching crc");
}

static int _test_set_stored_data(sqlite3* const db, const char* const name, const void* const data, const size_t size)
{
	sqlite3_stmt* stmt = 0;
	if (sqlite3_prepare_v2(db, "UPDATE tensors SET data=$data WHERE name=$name", -1, &stmt, 0) != SQLITE_OK)
		return -1;
	sqlite3_bind_blob(stmt, 1, data, (int)size, SQLITE_TRANSIENT);
	sqlite3_bind_text(stmt, 2, name, -1, 0);
	const int result = sqlite3_step(stmt) == SQLITE_DONE ? 0 : -1;
	sqlite3_finalize(stmt);
	return result;
}

static int _test_set_stored_reserved(sqlite3* const db, const char* const name, const int reserved, const int datatype)
{
	sqlite3_stmt* stmt = 0;
	if (sqlite3_prepare_v2(db, "UPDATE tensors SET datatype=$datatype WHERE name=$name", -1, &stmt, 0) != SQLITE_OK)
		return -1;
	sqlite3_bind_int64(stmt, 1, ((sqlite_int64)reserved << 32) | (unsigned int)datatype);
	sqlite3_bind_text(stmt, 2, name, -1, 0);
	const int result = sqlite3_step(stmt) == SQLITE_DONE ? 0 : -1;
	sqlite3_finalize(stmt);
	return result;
}

TEST_CASE("TMCQ weights with a codebook slot decode with it, and only the built-in table or an inline codebook goes into a checkpoint")
{
	enum { rows = 32, cols = 256, n = 24 };
	// A nonzero slot (reserved bits 16 - 23) is a table registered in this process; 0 is the built-in table.
	const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	REQUIRE(ccv_nnc_8i_rowwise_x_is_tmcq(tmcq | (5 << 16)), "TMCQ with a slot is TMCQ");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq | (5 << 16)), n, "the slot keeps the rate");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_kernel_format(tmcq | (5 << 16)), tmcq, "kernels see TMCQ without the slot");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq | (5 << 16), CCV_32F, rows * cols, cols), ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_32F, rows * cols, cols), "a slot keeps the TMCQ size");
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = sinf(i * 0.37f) * (1 + i % 3);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, tmcq, 0, 0, 0, 0, packed->data.u8, size), size, "TMCQ quantizes");
	uint8_t* const scratch = ccmalloc(size);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, tmcq | (1 << 16), 0, 0, 0, 0, scratch, size), 0, "ccv does not fit TMCQ with a slot");
	ccfree(scratch);
	// Codebooks: the alternate 512-pair table and a synthetic 2048-pair table, decoded by slot like the definition.
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	int8_t big[4096];
	for (i = 0; i < 4096; i++)
		big[i] = (int8_t)((i * 37 + 11) % 255 - 127);
	const int slot_big = ccv_nnc_codebook_register(big, 2048);
	REQUIRE(slot_alt >= 1 && slot_big >= 1 && slot_big != slot_alt, "both register");
	const int slots[2] = { slot_alt, slot_big };
	float* const a = ccmalloc(sizeof(float) * rows * cols);
	float* const builtin = ccmalloc(sizeof(float) * rows * cols);
	int* const q8 = ccmalloc(sizeof(int) * rows * cols);
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq, builtin, rows * cols), 0, "decodes with the built-in table");
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	int c;
	for (c = 0; c < 2; c++)
	{
		const int format = ccv_nnc_codebook_with_slot(tmcq, slots[c]);
		int entries = 0;
		const int8_t* const table = ccv_nnc_codebook_get(slots[c], &entries, 0);
		_test_tmcq_decode_table(packed->data.u8, rows, cols, n, table, ccv_nnc_codebook_log2_entries(entries), q8);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, format, a, rows * cols), 0, "decodes with codebook %d", c);
		const uint8_t* const scales = packed->data.u8 + _test_tmcq_scale_offset(rows, cols, n);
		int same_as_builtin = 1;
		for (i = 0; i < rows * cols; i++)
		{
			REQUIRE_EQ_WITH_TOLERANCE(a[i], (float)(_test_tmcq_load_scale(scales, CCV_32F, i / cols) * q8[i]), 0, "CPU decode with codebook %d matches the reference at %d", c, i);
			same_as_builtin = same_as_builtin && a[i] == builtin[i];
		}
		REQUIRE(!same_as_builtin, "codebook %d does not decode like the built-in table", c);
		// A process slot means nothing in a file: the write fails.
		ccv_nnc_tensor_t* const external = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, format), 0);
		memcpy(external->data.u8, packed->data.u8, size);
		REQUIRE_EQ(ccv_nnc_tensor_write(external, db, "slot", 0), CCV_IO_ERROR, "a weight naming a process slot is not written");
		ccv_nnc_tensor_free(external);
	}
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, ccv_nnc_codebook_with_slot(tmcq, 250), a, rows * cols), -1, "an unregistered slot fails to decode");
	// Files without a codebook: TMCQ, reserved == codec, reads back unchanged.
	REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "old", 0), CCV_IO_FINAL, "TMCQ writes");
	ccv_nnc_tensor_t* restored = 0;
	REQUIRE_EQ(ccv_nnc_tensor_read(db, "old", 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ reads");
	REQUIRE_EQ(restored->info.reserved, tmcq, "the TMCQ reserved field is unchanged");
	REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "TMCQ payload is unchanged");
	ccv_nnc_tensor_free(restored);
	// A stored weight that names a file-level codebook id (1 - 254) fails, data read and metadata read alike, rather
	// than decoding with the built-in table.
	static const int ids[] = { 1, 3, 254 };
	for (i = 0; i < (int)(sizeof(ids) / sizeof(ids[0])); i++)
	{
		REQUIRE_EQ(_test_set_stored_reserved(db, "old", ccv_nnc_codebook_with_slot(tmcq, ids[i]), packed->info.datatype), 0, "reserved updates");
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, "old", 0, 0, 0, &restored), CCV_IO_ERROR, "file codebook id %d fails", ids[i]);
		REQUIRE(restored == 0, "id %d leaves no tensor", ids[i]);
		REQUIRE_EQ(ccv_nnc_tensor_read(db, "old", 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_ERROR, "file codebook id %d metadata fails", ids[i]);
	}
	sqlite3_close(db);
	ccfree(q8);
	ccfree(builtin);
	ccfree(a);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(source);
}

TEST_CASE("TMCQ weights with an inline codebook survive the checkpoint without a codebook tensor and decode like the table by slot")
{
	enum { rows = 32, cols = 256, n = 24 };
	const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	// The layout: slot 255 and bits 24 - 25 = log2 P - 8; 128 + 2 P bytes before the payload.
	static const int entries_of[3] = { 512, 1024, 2048 };
	int e;
	for (e = 0; e < 3; e++)
	{
		const int format = ccv_nnc_8i_rowwise_x_with_inline_codebook(tmcq, entries_of[e]);
		REQUIRE_EQ(format, tmcq | (0xff << 16) | ((e + 1) << 24), "inline format bits for P %d", entries_of[e]);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_inline_codebook_size(format), 128 + 2 * entries_of[e], "prefix size for P %d", entries_of[e]);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, rows * cols, cols), 128 + 2 * entries_of[e] + ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_32F, rows * cols, cols), "the public size counts the prefix");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_kernel_format(format), tmcq, "kernels see TMCQ alone");
	}
	REQUIRE(!ccv_nnc_8i_rowwise_x_codebook_is_valid(tmcq | (1 << 24)), "inline bits need slot 255");
	REQUIRE(!ccv_nnc_8i_rowwise_x_codebook_is_valid(tmcq | (0xff << 16)), "slot 255 needs inline bits");
	REQUIRE(!ccv_nnc_8i_rowwise_x_codebook_is_valid(CCV_NNC_QX_8I_ROWWISE_Q4_K | (0xff << 16) | (1 << 24)), "only TMCQ and FTQ take an inline codebook");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq | (1 << 24), CCV_32F, rows * cols, cols), 0, "an invalid codebook has no size");
	ccv_nnc_tensor_t* const source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, cols), 0);
	int i;
	for (i = 0; i < rows * cols; i++)
		source->data.f32[i] = sinf(i * 0.29f) * (1 + i % 5);
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, tmcq, 0, 0, 0, 0, packed->data.u8, size), size, "TMCQ quantizes");
	uint8_t* const scratch = ccmalloc(size + 4096 + 128);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(source->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, rows * cols, cols, ccv_nnc_8i_rowwise_x_with_inline_codebook(tmcq, 512), 0, 0, 0, 0, scratch, size + 4096 + 128), 0, "ccv does not fit TMCQ with an inline codebook");
	ccfree(scratch);
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	int8_t big[4096];
	for (i = 0; i < 4096; i++)
		big[i] = (int8_t)((i * 53 + 7) % 255 - 127);
	const int slot_big = ccv_nnc_codebook_register(big, 2048);
	REQUIRE(slot_alt >= 1 && slot_big >= 1 && slot_big < CCV_NNC_CODEBOOK_INLINE_SLOT, "both register below the inline slot");
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	const int slots[2] = { slot_alt, slot_big };
	float* const by_slot = ccmalloc(sizeof(float) * rows * cols);
	float* const by_inline = ccmalloc(sizeof(float) * rows * cols);
	int c;
	for (c = 0; c < 2; c++)
	{
		int entries = 0;
		ccv_nnc_codebook_get(slots[c], &entries, 0);
		const int format = ccv_nnc_8i_rowwise_x_with_inline_codebook(tmcq, entries);
		const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(format);
		ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, format), 0);
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(weight->info), prefix + size, "the tensor holds the prefix and the payload");
		memset(weight->data.u8, 0, prefix);
		REQUIRE_EQ(ccv_nnc_codebook_serialize(slots[c], weight->data.u8, prefix), 16 + 2 * entries, "the container serializes");
		REQUIRE_EQ(memcmp(weight->data.u8, "CCVCBOOK", 8), 0, "the container starts with CCVCBOOK");
		memcpy(weight->data.u8 + prefix, packed->data.u8, size);
		char name[32];
		snprintf(name, sizeof(name), "inline%d", c);
		REQUIRE_EQ(ccv_nnc_tensor_write(weight, db, name, 0), CCV_IO_FINAL, "TMCQ with an inline codebook writes");
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ with an inline codebook reads without a codebook tensor");
		REQUIRE_EQ(restored->info.reserved, format, "the inline bits and slot 255 are unchanged");
		REQUIRE_EQ(memcmp(restored->data.u8, weight->data.u8, prefix + size), 0, "the data is unchanged");
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, prefix + size, cols, restored->info.reserved, by_inline, rows * cols), 0, "decodes inline");
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, ccv_nnc_codebook_with_slot(tmcq, slots[c]), by_slot, rows * cols), 0, "decodes by slot");
		REQUIRE_EQ(memcmp(by_inline, by_slot, sizeof(float) * rows * cols), 0, "table %d inline decodes like it by slot", c);
		ccv_nnc_tensor_free(restored);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_FINAL, "metadata reads");
		REQUIRE_EQ(restored->info.reserved, format, "metadata keeps the inline format");
		ccv_nnc_tensor_free(restored);
		// Caller params or an existing tensor must name the same inline codebook (the table sets the size).
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, &weight->info, &restored), CCV_IO_FINAL, "reads into matching caller params");
		REQUIRE_EQ(memcmp(restored->data.u8, weight->data.u8, prefix + size), 0, "the data is unchanged");
		ccv_nnc_tensor_free(restored);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, &packed->info, &restored), CCV_IO_ERROR, "caller params without the inline codebook fail");
		REQUIRE(restored == 0, "no tensor on failure");
		ccv_nnc_tensor_t* const existing = ccv_nnc_tensor_new(0, packed->info, 0);
		restored = existing;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "an existing tensor without the inline codebook fails");
		ccv_nnc_tensor_free(existing);
		// A stored table with a bad value, crc32 or padding fails the read itself, before any backend sees it.
		static const size_t corrupt_at[3] = { 16 + 5, 12, 16 + 1024 + 3 };
		int k;
		for (k = 0; k < 3; k++)
		{
			uint8_t* const bad = (uint8_t*)ccmalloc(prefix + size);
			memcpy(bad, weight->data.u8, prefix + size);
			bad[corrupt_at[k] < prefix ? corrupt_at[k] : 16] ^= 1;
			REQUIRE_EQ(_test_set_stored_data(db, name, bad, prefix + size), 0, "data updates");
			restored = 0;
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "a corrupt inline codebook fails the read (byte %d)", (int)corrupt_at[k]);
			REQUIRE(restored == 0, "no tensor on failure");
			ccfree(bad);
		}
		REQUIRE_EQ(_test_set_stored_data(db, name, weight->data.u8, prefix + size), 0, "data restores");
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "the restored row reads");
		ccv_nnc_tensor_free(restored);
		// The same row with half the marker fails, whichever half.
		REQUIRE_EQ(_test_set_stored_reserved(db, name, format & ~(0xff << 16), weight->info.datatype), 0, "reserved updates");
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "inline bits without slot 255 fail");
		REQUIRE_EQ(_test_set_stored_reserved(db, name, format & ~(0x3 << 24), weight->info.datatype), 0, "reserved updates");
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "slot 255 without inline bits fails");
		REQUIRE(restored == 0, "no tensor on failure");
		ccv_nnc_tensor_free(weight);
	}
	sqlite3_close(db);
	ccfree(by_inline);
	ccfree(by_slot);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(source);
}

// K % 16: the tiled codecs (TMCQ, FTQ, variable-rate FTQ) cut the weight into (rows / 16) x (K / 16) tiles of 16 n
// bits, so any K % 16 == 0 is a whole tile grid. The cases below cover K % 256 (sizes must not change), the MoE
// down projection K = 1152 (72 tiles, 4.5 groups of 256), and K = 272 / 48 / 16 (an odd tile count per tile row).

static size_t _test_tiled_align128(const size_t x)
{
	return (x + 127) & ~(size_t)127;
}

static const int tiled_shapes[][2] = {
	{ 16, 256 }, { 32, 512 }, { 48, 768 }, { 16, 4352 }, { 2304, 1152 }, { 32, 1152 }, { 16, 272 }, { 48, 144 }, { 16, 16 }, { 64, 48 },
};

TEST_CASE("tiled row-wise-x sizes follow the 16 x 16 tile grid for K % 16 and are unchanged for K % 256")
{
	int s, f;
	for (s = 0; s < (int)(sizeof(tiled_shapes) / sizeof(tiled_shapes[0])); s++)
	{
		const int rows = tiled_shapes[s][0], cols = tiled_shapes[s][1];
		const size_t count = (size_t)rows * cols;
		const size_t tiles = (size_t)rows / 16 * (cols / 16);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tile_count(rows, cols), (uint64_t)tiles, "%d x %d tile count", rows, cols);
		// TMCQ_N16 .. N32, FTQ_N16 .. N88, FTQ_VR_N16 .. N88.
		for (f = 0; f < 9 + 37 + 37; f++)
		{
			const int vr = f >= 46, ftq = f >= 9 && !vr;
			const int format = f < 9 ? tmcq_formats[f] : ftq ? CCV_NNC_QX_8I_ROWWISE_FTQ_N16 + (f - 9) : CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 + (f - 46);
			const int n = f < 9 ? ccv_nnc_8i_rowwise_x_tmcq_rate(format) : ftq ? ccv_nnc_8i_rowwise_x_ftq_rate(format) : ccv_nnc_8i_rowwise_x_ftq_vr_rate(format);
			const int datatype = f < 9 ? CCV_32F : CCV_16F;
			const size_t code = tiles * 2 * n;
			const size_t header = vr ? 32 + 4 * ((size_t)rows / 16 + 1) : ftq ? 32 : 0;
			const size_t expected = _test_tiled_align128(header + code) + (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatype);
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatype, count, cols);
			REQUIRE_EQ(size, expected, "format %d %d x %d is the tile grid plus the header and aligned scales", format, rows, cols);
			ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
			params.datatype = datatype;
			params = ccv_nnc_tensor_8i_rowwise_x(params, format);
			REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(params), expected, "format %d %d x %d tensor allocation agrees", format, rows, cols);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_payload_bits(format, rows, cols), (uint64_t)code * 8, "format %d %d x %d payload bits", format, rows, cols);
			// The group formula of the row-wise codecs, rows x ceil(K / 256) tiles' worth of bits, is what the sizes
			// were before: byte for byte the same for K % 256, and 1 / 9 too large at K = 1152.
			const size_t row_wise = _test_tiled_align128(header + ((size_t)rows * ((cols + 255) / 256) * 16 * n + 7) / 8) + (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatype);
			if (cols % 256 == 0)
			{
				REQUIRE_EQ(size, row_wise, "format %d %d x %d keeps its K %% 256 size", format, rows, cols);
			} else {
				REQUIRE(size < row_wise, "format %d %d x %d is smaller than the per-row group count", format, rows, cols);
			}
			if (ftq)
			{
				const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
				REQUIRE_EQ(schedule.tile_count, (uint64_t)tiles, "FTQ n%d %d x %d schedule tiles", n, rows, cols);
				REQUIRE_EQ(schedule.code_bits, (uint64_t)code * 8, "FTQ n%d %d x %d schedule bits", n, rows, cols);
				REQUIRE_EQ(schedule.scale_offset, (uint64_t)_test_tiled_align128(32 + code), "FTQ n%d %d x %d schedule scale offset", n, rows, cols);
			}
		}
	}
	// The MoE down projection of 128 experts of 2304 x 1152 stacked along the rows, at TMCQ_N16: 2304 / 16 x 72 tiles
	// of 32 bytes per expert, no padding between experts.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_16F, (size_t)128 * 2304 * 1152, 1152), (size_t)128 * 144 * 72 * 32 + 128 * 2304 * 2, "128 stacked 2304 x 1152 experts");
	// K not a multiple of 16 or rows not a multiple of 16 have no size.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, CCV_16F, 24 * 1152, 1152), 0, "TMCQ 24 rows have no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_N20, CCV_16F, 16 * 1160, 1160), 0, "FTQ K = 1160 has no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20, CCV_16F, 24 * 1152, 1152), 0, "FTQ_VR 24 rows have no size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, CCV_16F, 16 * 1144, 1144), 0, "TMCQ K = 1144 has no size");
}

static const int tiled_decode_shapes[][2] = { { 32, 1152 }, { 16, 272 }, { 48, 144 }, { 16, 16 }, { 32, 48 } };

TEST_CASE("TMCQ CPU decode of random payloads matches the definition for K % 16 in every precision")
{
	const int datatypes[4] = {CCV_16F, CCV_16BF, CCV_32F, CCV_64F};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1153);
	int s, f, d, i;
	for (s = 0; s < (int)(sizeof(tiled_decode_shapes) / sizeof(tiled_decode_shapes[0])); s++)
	{
		const int rows = tiled_decode_shapes[s][0], cols = tiled_decode_shapes[s][1], count = rows * cols;
		int* const q8 = ccmalloc(sizeof(int) * count);
		uint8_t* const decoded = ccmalloc(sizeof(double) * count);
		uint8_t* const expected = ccmalloc(sizeof(double) * count);
		for (f = 0; f < 9; f++)
		{
			const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(tmcq_formats[f]);
			const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
			d = (s + f) % 4;
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq_formats[f], datatypes[d], count, cols);
			REQUIRE_EQ(size, scale_offset + rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "%d x %d n = %d size", rows, cols, n);
			uint8_t* const packed = ccmalloc(size);
			for (i = 0; i < (int)scale_offset; i++)
				packed[i] = (uint8_t)(dsfmt_genrand_uint32(&dsfmt) & 0xff);
			float* const scale_f = ccmalloc(sizeof(float) * rows);
			for (i = 0; i < rows; i++)
				scale_f[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 8;
			_test_tmcq_from_float(scale_f, datatypes[d], rows, packed + scale_offset);
			ccfree(scale_f);
			_test_tmcq_decode(packed, rows, cols, n, q8);
			for (i = 0; i < count; i++)
			{
				const double v = _test_tmcq_load_scale(packed + scale_offset, datatypes[d], i / cols) * q8[i];
				const float vf = (float)v;
				if (datatypes[d] == CCV_64F)
					((double*)expected)[i] = v;
				else
					_test_tmcq_from_float(&vf, datatypes[d], 1, expected + i * CCV_GET_DATA_TYPE_SIZE(datatypes[d]));
			}
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed, datatypes[d], CCV_TENSOR_CPU_MEMORY, size, cols, tmcq_formats[f], decoded, count), 0, "%d x %d n = %d decodes", rows, cols, n);
			REQUIRE_ARRAY_EQ(uint8_t, expected, decoded, count * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "random payload decodes exactly for %d x %d n = %d, datatype %d", rows, cols, n, d);
			ccfree(packed);
		}
		ccfree(expected);
		ccfree(decoded);
		ccfree(q8);
	}
}

TEST_CASE("TMCQ quantizes K = 1152 and K = 272 into payloads that decode as the definition, within the rate's error bound")
{
	// The bounds of the K = 512 test at n = 16, 24, 32.
	const int formats[3] = { CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 };
	const double bounds[3] = { 0.30, 0.185, 0.125 };
	static const int shapes[][2] = { { 32, 1152 }, { 16, 272 } };
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1152);
	int s, f, i;
	for (s = 0; s < 2; s++)
	{
		const int rows = shapes[s][0], cols = shapes[s][1], count = rows * cols;
		float* const values = ccmalloc(sizeof(float) * count);
		float* const decoded = ccmalloc(sizeof(float) * count);
		int* const q8 = ccmalloc(sizeof(int) * count);
		_test_tmcq_values(&dsfmt, values, rows, cols);
		double energy = 0;
		for (i = 0; i < count; i++)
			energy += (double)values[i] * values[i];
		for (f = 0; f < 3; f++)
		{
			const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(formats[f]);
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_32F, count, cols);
			uint8_t* const packed = ccmalloc(size + 16);
			memset(packed, 0xa5, size + 16);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, formats[f], 0, 0, 0, 0, packed, size), size, "%d x %d n = %d quantizer returns the exact size", rows, cols, n);
			for (i = 0; i < 16; i++)
				REQUIRE_EQ(packed[size + i], 0xa5, "quantizer preserves output guard bytes");
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], decoded, count), 0, "decodes");
			_test_tmcq_decode(packed, rows, cols, n, q8);
			const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
			int mismatches = 0;
			for (i = 0; i < count; i++)
				mismatches += decoded[i] != (float)((double)((float*)(packed + scale_offset))[i / cols] * q8[i]);
			REQUIRE_EQ(mismatches, 0, "%d x %d n = %d decode matches the definition", rows, cols, n);
			const double relative = _test_tmcq_sse(values, decoded, count) / energy;
			REQUIRE(relative < bounds[f], "%d x %d n = %d relative error %g within %g", rows, cols, n, relative, bounds[f]);
			ccfree(packed);
		}
		ccfree(q8);
		ccfree(decoded);
		ccfree(values);
	}
}

static void _test_ftq_ref_tile(const uint8_t* const blob, const uint64_t start, const int T, int8_t* const q8)
{
	int j, b;
	for (j = 0; j < 128; j++)
	{
		const int end = (j + 1) * T / 128;
		uint32_t state = 0;
		for (b = 0; b < 16; b++)
			state |= (uint32_t)_test_tmcq_bit(blob, start + (uint64_t)((end + T - 16 + b) % T)) << b;
		const uint32_t k = (state * 0x83DCD12Du) >> 23;
		q8[2 * j] = ccv_nnc_trellis_builtin_table[2 * k];
		q8[2 * j + 1] = ccv_nnc_trellis_builtin_table[2 * k + 1];
	}
}

// The FP16 values an FTQ / FTQ_VR blob decodes to: tile row tr has rates[tr] (FTQ: n for all) and starts at bit
// row_bits[tr] of the blob.
static void _test_ftq_ref_decode(const uint8_t* const blob, const int rows, const int cols, const int* const rates, const uint64_t* const row_bits, const size_t scale_offset, uint16_t* const out)
{
	const int strips = cols / 16;
	float* const scales = ccmalloc(sizeof(float) * rows);
	ccv_half_precision_to_float((const uint16_t*)(blob + scale_offset), scales, rows);
	int tr, tc, t;
	for (tr = 0; tr < rows / 16; tr++)
		for (tc = 0; tc < strips; tc++)
		{
			int8_t q8[256];
			_test_ftq_ref_tile(blob, row_bits[tr] + (uint64_t)tc * 16 * rates[tr], 16 * rates[tr], q8);
			for (t = 0; t < 256; t++)
			{
				const int row = tr * 16 + t / 16;
				const float v = (float)q8[t] * scales[row];
				ccv_float_to_half_precision(&v, out + (size_t)row * cols + tc * 16 + t % 16, 1);
			}
		}
	ccfree(scales);
}

TEST_CASE("FTQ CPU decode of random blobs matches the reference for K % 16")
{
	static const int rates[] = { 16, 18, 20, 36, 50, 64, 66, 72, 86, 88 };
	uint32_t seed = 0x1152u;
	int s, r, i;
	for (s = 0; s < (int)(sizeof(tiled_decode_shapes) / sizeof(tiled_decode_shapes[0])); s++)
		for (r = 0; r < (int)(sizeof(rates) / sizeof(rates[0])); r++)
		{
			const int rows = tiled_decode_shapes[s][0], cols = tiled_decode_shapes[s][1], n = rates[r];
			const int format = CCV_NNC_QX_8I_ROWWISE_FTQ_N16 + (n - 16) / 2;
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, (size_t)rows * cols, cols);
			const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
			REQUIRE_EQ(size, (size_t)schedule.scale_offset + 2 * rows, "%d x %d n%d size", rows, cols, n);
			uint8_t* const blob = ccmalloc(size);
			memset(blob, 0, size);
			memcpy(blob, "CCVFTQSR", 8);
			const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, schedule.high_bits, 0, 9u << 8, 0 };
			memcpy(blob + 8, header, sizeof(header));
			for (i = 0; i < (int)(schedule.code_bits / 8); i++)
			{
				seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5;
				blob[32 + i] = (uint8_t)(seed >> 24);
			}
			float* const scales = ccmalloc(sizeof(float) * rows);
			for (i = 0; i < rows; i++)
				scales[i] = 0.001f * (1 + (i * 7) % 13);
			ccv_float_to_half_precision(scales, (uint16_t*)(blob + schedule.scale_offset), rows);
			ccfree(scales);
			int* const row_rates = ccmalloc(sizeof(int) * (rows / 16));
			uint64_t* const row_bits = ccmalloc(sizeof(uint64_t) * (rows / 16));
			for (i = 0; i < rows / 16; i++)
				row_rates[i] = n, row_bits[i] = 256 + (uint64_t)i * (cols / 16) * 16 * n;
			uint16_t* const expected = ccmalloc(sizeof(uint16_t) * rows * cols);
			uint16_t* const actual = ccmalloc(sizeof(uint16_t) * rows * cols);
			_test_ftq_ref_decode(blob, rows, cols, row_rates, row_bits, schedule.scale_offset, expected);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, (size_t)rows * cols), 0, "%d x %d n%d decodes", rows, cols, n);
			REQUIRE_ARRAY_EQ(uint16_t, expected, actual, rows * cols, "FTQ %d x %d n%d decode matches the reference", rows, cols, n);
			// One byte short of the blob is rejected.
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size - 1, cols, format, actual, (size_t)rows * cols), -1, "%d x %d n%d short blob", rows, cols, n);
			ccfree(actual);
			ccfree(expected);
			ccfree(row_bits);
			ccfree(row_rates);
			ccfree(blob);
		}
}

// The pair of an L-bit FTQ window (CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT): x = state + (state >> 16), h = x *
// 0x83DCD12D; table: the built-in pairs (fold 0) or a quadrant-folded base table (fold 1).
static void _test_ftq_wide_ref_tile(const uint8_t* const blob, const uint64_t start, const int T, const int L, const int8_t* const table, const int fold, int8_t* const q8)
{
	int j, b;
	for (j = 0; j < 128; j++)
	{
		const int end = (j + 1) * T / 128;
		uint32_t state = 0;
		for (b = 0; b < L; b++)
			state |= (uint32_t)_test_tmcq_bit(blob, start + (uint64_t)((end + T - L + b) % T)) << b;
		const uint32_t h = (state + (state >> 16)) * 0x83DCD12Du;
		if (fold)
		{
			const uint32_t k = (h >> 16) & 511;
			q8[2 * j] = (h & 0x8000u) ? -table[2 * k] : table[2 * k];
			q8[2 * j + 1] = (h & 0x80000000u) ? -table[2 * k + 1] : table[2 * k + 1];
		} else {
			const uint32_t k = h >> 23;
			q8[2 * j] = table[2 * k];
			q8[2 * j + 1] = table[2 * k + 1];
		}
	}
}

TEST_CASE("FTQ CPU decode of random blobs with an 18, 20 and 22-bit window matches the reference")
{
	// Three words a lane up to n48, four above (n64, n88: tiles of 1408 bits).
	static const int rates[] = { 16, 32, 48, 64, 88 };
	static const int shapes[][2] = { { 32, 1152 }, { 16, 272 } };
	int8_t base[1024];
	int i;
	for (i = 0; i < 1024; i++)
		base[i] = (int8_t)((i * 37 + 11) % 120);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(folded > 0, "the folded table registers");
	uint32_t folded_crc = 0;
	ccv_nnc_codebook_get(folded, 0, &folded_crc);
	uint32_t seed = 0x2b0fu;
	int w, f, s, r;
	for (w = 1; w <= 3; w++)
		for (f = 0; f < 2; f++)
			for (s = 0; s < 2; s++)
				for (r = 0; r < 5; r++)
				{
					const int rows = shapes[s][0], cols = shapes[s][1], n = rates[r], L = 16 + 2 * w;
					const int narrow = ccv_nnc_codebook_with_slot(CCV_NNC_QX_8I_ROWWISE_FTQ_N16 + (n - 16) / 2, f ? folded : 0);
					const int format = narrow | (w << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
					REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_window(format), w, "the format names window %d", w);
					const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, (size_t)rows * cols, cols);
					const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
					uint8_t* const blob = ccmalloc(size);
					memset(blob, 0, size);
					memcpy(blob, "CCVFTQSR", 8);
					const uint32_t field = (f ? 1u | (1u << 16) : 0u) | (9u << 8) | ((uint32_t)w << 30);
					uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, schedule.high_bits, 0, field, f ? folded_crc : 0 };
					memcpy(blob + 8, header, sizeof(header));
					for (i = 0; i < (int)(schedule.code_bits / 8); i++)
					{
						seed ^= seed << 13, seed ^= seed >> 17, seed ^= seed << 5;
						blob[32 + i] = (uint8_t)(seed >> 24);
					}
					float* const scales = ccmalloc(sizeof(float) * rows);
					for (i = 0; i < rows; i++)
						scales[i] = 0.001f * (1 + (i * 5) % 11);
					ccv_float_to_half_precision(scales, (uint16_t*)(blob + schedule.scale_offset), rows);
					// The reference scales by the stored FP16 scales.
					ccv_half_precision_to_float((const uint16_t*)(blob + schedule.scale_offset), scales, rows);
					uint16_t* const expected = ccmalloc(sizeof(uint16_t) * rows * cols);
					uint16_t* const actual = ccmalloc(sizeof(uint16_t) * rows * cols);
					int tr, tc, t;
					for (tr = 0; tr < rows / 16; tr++)
						for (tc = 0; tc < cols / 16; tc++)
						{
							int8_t q8[256];
							_test_ftq_wide_ref_tile(blob, 256 + ((uint64_t)tr * (cols / 16) + tc) * 16 * n, 16 * n, L, f ? base : ccv_nnc_trellis_builtin_table, f, q8);
							for (t = 0; t < 256; t++)
							{
								const int row = tr * 16 + t / 16;
								const float v = (float)q8[t] * scales[row];
								ccv_float_to_half_precision(&v, expected + (size_t)row * cols + tc * 16 + t % 16, 1);
							}
						}
					ccfree(scales);
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, (size_t)rows * cols), 0, "L%d %s %d x %d n%d decodes", L, f ? "folded" : "built-in", rows, cols, n);
					REQUIRE_ARRAY_EQ(uint16_t, expected, actual, rows * cols, "L%d %s %d x %d n%d decode matches the reference", L, f ? "folded" : "built-in", rows, cols, n);
					REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(blob, size, format, rows, cols), 0, "L%d n%d passes the load check", L, n);
					// The header and the format must name the same window: a 16-bit reader rejects the blob, and so
					// does any other window.
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, narrow, actual, (size_t)rows * cols), -1, "L%d blob with a 16-bit format", L);
					REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(blob, size, narrow, rows, cols), -1, "L%d blob fails the 16-bit load check", L);
					const int other = narrow | ((w % 3 + 1) << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, other, actual, (size_t)rows * cols), -1, "L%d blob with another window", L);
					header[4] = field & ~(3u << 30);
					memcpy(blob + 8, header, sizeof(header));
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, (size_t)rows * cols), -1, "a 16-bit header with an L%d format", L);
					header[4] = field | (1u << 18);
					memcpy(blob + 8, header, sizeof(header));
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, (size_t)rows * cols), -1, "table bits 18 - 29 stay zero");
					ccfree(actual);
					ccfree(expected);
					ccfree(blob);
				}
	// Only FTQ and FTQ-VR take a window; the CPU encoder does not search one.
	float values[256 * 16];
	for (i = 0; i < 256 * 16; i++)
		values[i] = 0.01f * ((i * 13) % 17 - 8);
	uint16_t half[256 * 16];
	ccv_float_to_half_precision(values, half, 256 * 16);
	const int wide = CCV_NNC_QX_8I_ROWWISE_FTQ_N32 | (2 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
	const size_t packed_size = ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_N32, CCV_16F, 256 * 16, 256);
	uint8_t* const packed = ccmalloc(packed_size);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, 256 * 16, 256, CCV_NNC_QX_8I_ROWWISE_FTQ_N32, 0, 0, 0, 0, packed, packed_size), packed_size, "the CPU encoder fits a 16-bit window");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, 256 * 16, 256, wide, 0, 0, 0, 0, packed, packed_size), 0, "the CPU encoder does not fit a 20-bit window");
	ccfree(packed);
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_codebook_is_valid(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 | (1 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT)), 0, "TMCQ has no window");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_codebook_is_valid(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N32 | (3 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT)), 1, "FTQ-VR may take a window");
	const size_t q4_size = ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_Q4_K, CCV_16F, 256 * 16, 256);
	uint8_t* const q4 = ccmalloc(q4_size);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, 256 * 16, 256, CCV_NNC_QX_8I_ROWWISE_Q4_K | (1 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT), 0, 0, 0, 0, q4, q4_size), 0, "a window on another codec does not quantize");
	ccfree(q4);
	const int vr_wide = CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N24 | (1 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
	const int vr_rates[] = { 16, 32 };
	const size_t vr_size = ccv_nnc_8i_rowwise_x_data_size(vr_wide, CCV_16F, 256 * 16, 256);
	uint8_t* const vr = ccmalloc(vr_size);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, 256 * 16, 256, vr_wide, vr_rates, vr, vr_size), 0, "the CPU encoder does not fit a wide FTQ-VR window");
	ccfree(vr);
}

// A random wide-window blob of format (static-rate n or variable-rate with tile row rates row_rates), built-in table,
// and its reference decode. The blob is ccmalloc'ed, *size its size.
static uint8_t* _test_ftq_wide_blob(const int format, const int rows, const int cols, const int* const row_rates, uint32_t* const seed, size_t* const size, uint16_t* const expected)
{
	const int vr = ccv_nnc_8i_rowwise_x_is_ftq_vr(format);
	const int w = ccv_nnc_8i_rowwise_x_ftq_window(format), L = 16 + 2 * w;
	const int R = rows / 16, C = cols / 16;
	*size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, (size_t)rows * cols, cols);
	uint8_t* const blob = ccmalloc(*size);
	memset(blob, 0, *size);
	const int nm = vr ? ccv_nnc_8i_rowwise_x_ftq_vr_rate(format) : ccv_nnc_8i_rowwise_x_ftq_rate(format);
	uint64_t level_mask = 0;
	int i, r;
	for (r = 0; r < R; r++)
		level_mask |= (uint64_t)1 << ((row_rates[r] - 8) / 2);
	memcpy(blob, vr ? "CCVFTQVR" : "CCVFTQSR", 8);
	// Level mask bits 32 - 40 (n72 ... n88) go to bits 18 - 26 of the table word, next to the window in bits 30 - 31.
	const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, 16u * nm, vr ? (uint32_t)level_mask : 0, (9u << 8) | (vr ? (uint32_t)((level_mask >> 32) << 18) : 0) | ((uint32_t)w << 30), 0 };
	memcpy(blob + 8, header, sizeof(header));
	// Code start (bytes) and each tile row's start (bits).
	const size_t code = vr ? 32 + 4 * (size_t)(R + 1) : 32;
	uint64_t* const row_bits = ccmalloc(sizeof(uint64_t) * R);
	uint32_t off = 0;
	for (r = 0; r < R; r++)
	{
		if (vr)
			memcpy(blob + 32 + 4 * r, &off, 4);
		row_bits[r] = (uint64_t)code * 8 + (uint64_t)off * 32;
		off += (uint32_t)(C * row_rates[r] / 2);
	}
	if (vr)
		memcpy(blob + 32 + 4 * R, &off, 4);
	for (i = 0; i < (int)off * 4; i++)
	{
		*seed ^= *seed << 13, *seed ^= *seed >> 17, *seed ^= *seed << 5;
		blob[code + i] = (uint8_t)(*seed >> 24);
	}
	const size_t scale_offset = *size - 2 * (size_t)rows;
	float* const scales = ccmalloc(sizeof(float) * rows);
	for (i = 0; i < rows; i++)
		scales[i] = 0.001f * (1 + (i * 7) % 13);
	ccv_float_to_half_precision(scales, (uint16_t*)(blob + scale_offset), rows);
	ccv_half_precision_to_float((const uint16_t*)(blob + scale_offset), scales, rows);
	int tc, t;
	for (r = 0; r < R; r++)
		for (tc = 0; tc < C; tc++)
		{
			int8_t q8[256];
			_test_ftq_wide_ref_tile(blob, row_bits[r] + (uint64_t)tc * 16 * row_rates[r], 16 * row_rates[r], L, ccv_nnc_trellis_builtin_table, 0, q8);
			for (t = 0; t < 256; t++)
			{
				const int row = r * 16 + t / 16;
				const float v = (float)q8[t] * scales[row];
				ccv_float_to_half_precision(&v, expected + (size_t)row * cols + tc * 16 + t % 16, 1);
			}
		}
	ccfree(scales);
	ccfree(row_bits);
	return blob;
}

TEST_CASE("FTQ-VR CPU decode of random blobs with an 18, 20 and 22-bit window matches the reference")
{
	enum { rows = 64, cols = 272 };
	// Tile row rates of mean 24 (8 wraps the first windows of rows 0 and 1, see CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16), and of
	// mean 70 with n72 and n88, whose level mask bits 32 and 40 sit in table word bits 18 and 26 next to the window.
	static const int row_rates[2][4] = { { 8, 48, 16, 24 }, { 72, 48, 88, 72 } };
	static const int means[2] = { 24, 70 };
	uint32_t seed = 0x51edu;
	uint16_t* const expected = ccmalloc(sizeof(uint16_t) * rows * cols);
	uint16_t* const actual = ccmalloc(sizeof(uint16_t) * rows * cols);
	int w, v;
	for (v = 0; v < 2; v++)
		for (w = 1; w <= 3; w++)
		{
			const int narrow = CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 + (means[v] - 16) / 2;
			const int format = narrow | (w << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
			size_t size;
			uint8_t* const blob = _test_ftq_wide_blob(format, rows, cols, row_rates[v], &seed, &size, expected);
			uint64_t mask = 0;
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows, cols, &mask), 0, "L%d FTQ-VR blob passes the check", 16 + 2 * w);
			uint64_t want = 0;
			int r;
			for (r = 0; r < 4; r++)
				want |= (uint64_t)1 << ((row_rates[v][r] - 8) / 2);
			REQUIRE_EQ(mask, want, "L%d FTQ-VR nm%d level mask", 16 + 2 * w, means[v]);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, (size_t)rows * cols), 0, "L%d FTQ-VR decodes", 16 + 2 * w);
			REQUIRE_ARRAY_EQ(uint16_t, expected, actual, rows * cols, "L%d FTQ-VR decode matches the reference", 16 + 2 * w);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, narrow, rows, cols, 0), -1, "L%d FTQ-VR blob fails the 16-bit check", 16 + 2 * w);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, narrow, actual, (size_t)rows * cols), -1, "L%d FTQ-VR blob with a 16-bit format", 16 + 2 * w);
			ccfree(blob);
		}
	ccfree(actual);
	ccfree(expected);
}

static int _test_xor_encode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, void* const context, void* const encoded, size_t* const encoded_size, ccv_nnc_tensor_param_t* const params, unsigned int* const identifier)
{
	size_t i;
	for (i = 0; i < data_size; i++)
		((uint8_t*)encoded)[i] = ((const uint8_t*)data)[i] ^ 0x5a;
	*encoded_size = data_size;
	*identifier = 7;
	return 1;
}

static int _test_xor_decode(const void* const data, const size_t data_size, const int datatype, const int* const dimensions, const int dimension_count, const unsigned int identifier, void* const context, const ccv_nnc_tensor_param_t tensor_params, ccv_nnc_tensor_t** const tensor_out, void* const decoded, size_t* const decoded_size)
{
	if (identifier != 7)
		return 0;
	if (!*tensor_out)
		*tensor_out = ccv_nnc_tensor_new(0, tensor_params, 0);
	uint8_t* const output = decoded ? (uint8_t*)decoded : tensor_out[0]->data.u8;
	size_t i;
	const size_t size = ccv_min(*decoded_size, data_size);
	for (i = 0; i < size; i++)
		output[i] = ((const uint8_t*)data)[i] ^ 0x5a;
	*decoded_size = size;
	return 1;
}

TEST_CASE("wide-window FTQ is stored raw only, and a split keeps an FTQ header in the first part")
{
	enum { rows = 32, cols = 256 };
	static const int rates[] = { 32, 32 };
	uint32_t seed = 0x3c5du;
	uint16_t* const expected = ccmalloc(sizeof(uint16_t) * rows * cols);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	const ccv_nnc_tensor_io_option_t options = { .decode = _test_xor_decode, .encode = _test_xor_encode };
	// An identifier codec: a 16-bit FTQ weight round-trips, a wide one is not written (a reader has no header to check).
	int w;
	for (w = 0; w < 2; w++)
	{
		const int format = CCV_NNC_QX_8I_ROWWISE_FTQ_N32 | (w << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
		size_t size;
		uint8_t* const blob = _test_ftq_wide_blob(format, rows, cols, rates, &seed, &size, expected);
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, rows, cols), format), 0);
		memcpy(packed->data.u8, blob, size);
		if (w) {
			REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "encoded", &options), CCV_IO_ERROR, "a wide window is not written through an identifier codec");
		} else {
			REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "encoded", &options), CCV_IO_FINAL, "a 16-bit window writes through an identifier codec");
			ccv_nnc_tensor_t* restored = 0;
			REQUIRE_EQ(ccv_nnc_tensor_read(db, "encoded", &options, 0, 0, &restored), CCV_IO_FINAL, "and reads back");
			REQUIRE_EQ(restored->info.reserved, format, "with its format");
			REQUIRE_EQ(memcmp(restored->data.u8, blob, size), 0, "and its payload");
			ccv_nnc_tensor_free(restored);
		}
		ccv_nnc_tensor_free(packed);
		ccfree(blob);
	}
	// Split tensors: the header must be in the first part. With a 1000-byte limit the parts are 500 bytes and a wide
	// SR and VR blob round-trip; with a 60-byte limit the first part (30 bytes) cannot hold the header.
	sqlite3_limit(db, SQLITE_LIMIT_LENGTH, 1000);
	int v;
	for (v = 0; v < 2; v++)
	{
		static const int vr_rates[] = { 16, 32 };
		const int format = (v ? CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N24 : CCV_NNC_QX_8I_ROWWISE_FTQ_N32) | (3 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
		size_t size;
		uint8_t* const blob = _test_ftq_wide_blob(format, rows, cols, v ? vr_rates : rates, &seed, &size, expected);
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, rows, cols), format), 0);
		memcpy(packed->data.u8, blob, size);
		REQUIRE(size > 1000, "the blob is split");
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "split", 0), CCV_IO_FINAL, "a split wide blob writes");
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, "split", 0, 0, 0, &restored), CCV_IO_FINAL, "and reads back");
		REQUIRE_EQ(restored->info.reserved, format, "with its window");
		REQUIRE_EQ(memcmp(restored->data.u8, blob, size), 0, "and its payload");
		ccv_nnc_tensor_free(restored);
		ccv_nnc_tensor_free(packed);
		ccfree(blob);
	}
	sqlite3_limit(db, SQLITE_LIMIT_LENGTH, 60);
	{
		size_t size;
		uint8_t* const blob = _test_ftq_wide_blob(CCV_NNC_QX_8I_ROWWISE_FTQ_N32, rows, cols, rates, &seed, &size, expected);
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, rows, cols), CCV_NNC_QX_8I_ROWWISE_FTQ_N32), 0);
		memcpy(packed->data.u8, blob, size);
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "tiny", 0), CCV_IO_ERROR, "a first part shorter than the FTQ header is not written");
		ccv_nnc_tensor_free(packed);
		ccfree(blob);
	}
	sqlite3_close(db);
	ccfree(expected);
}

TEST_CASE("wide-window FTQ and FTQ-VR survive tensor persistence and a read names the stored window")
{
	enum { rows = 32, cols = 256 };
	static const int row_rates[] = { 16, 32 };
	uint32_t seed = 0x7a31u;
	uint16_t* const expected = ccmalloc(sizeof(uint16_t) * rows * cols);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	int v;
	for (v = 0; v < 2; v++)
	{
		const int narrow = v ? CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N24 : CCV_NNC_QX_8I_ROWWISE_FTQ_N32;
		const int format = narrow | (2 << CCV_NNC_QX_8I_ROWWISE_FTQ_WINDOW_SHIFT);
		const int fixed_rates[] = { 32, 32 };
		size_t size;
		uint8_t* const blob = _test_ftq_wide_blob(format, rows, cols, v ? row_rates : fixed_rates, &seed, &size, expected);
		const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, rows, cols), format);
		const ccv_nnc_tensor_param_t narrow_params = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(16F, rows, cols), narrow);
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, params, 0);
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(packed->info), size, "tensor size matches the blob");
		memcpy(packed->data.u8, blob, size);
		const char* const name = v ? "ftq_vr_wide" : "ftq_wide";
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "%s writes", name);
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s reads", name);
		REQUIRE_EQ(restored->info.reserved, format, "%s keeps its window", name);
		REQUIRE_EQ(memcmp(restored->data.u8, blob, size), 0, "%s payload persists", name);
		ccv_nnc_tensor_free(restored);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_FINAL, "%s metadata reads", name);
		REQUIRE_EQ(restored->info.reserved, format, "%s metadata keeps its window", name);
		ccv_nnc_tensor_free(restored);
		// Params or a tensor naming another window fail the read.
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, &narrow_params, &restored), CCV_IO_ERROR, "%s into 16-bit params", name);
		REQUIRE(restored == 0, "no tensor is left");
		restored = ccv_nnc_tensor_new(0, narrow_params, 0);
		ccv_nnc_tensor_t* const narrow_tensor = restored;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s into a 16-bit tensor", name);
		ccv_nnc_tensor_free(narrow_tensor);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, &params, &restored), CCV_IO_FINAL, "%s into params of its window", name);
		ccv_nnc_tensor_free(restored);
		// A header naming another window than the stored format fails the read, metadata only too.
		uint32_t field;
		memcpy(&field, blob + 24, 4);
		const uint32_t narrow_field = field & ~(3u << 30);
		memcpy(blob + 24, &narrow_field, 4);
		REQUIRE_EQ(_test_set_stored_data(db, name, blob, size), 0, "a 16-bit header is stored");
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_ERROR, "%s with a 16-bit header, metadata", name);
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s with a 16-bit header", name);
		// The writer checks the header against the format both ways.
		memcpy(packed->data.u8, blob, size);
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_ERROR, "a 16-bit header does not write as %s", name);
		memcpy(blob + 24, &field, 4);
		ccv_nnc_tensor_t* const narrow_packed = ccv_nnc_tensor_new(0, narrow_params, 0);
		memcpy(narrow_packed->data.u8, blob, size);
		REQUIRE_EQ(ccv_nnc_tensor_write(narrow_packed, db, name, 0), CCV_IO_ERROR, "a wide header does not write with a 16-bit format");
		ccv_nnc_tensor_free(narrow_packed);
		ccv_nnc_tensor_free(packed);
		ccfree(blob);
	}
	sqlite3_close(db);
	ccfree(expected);
}

TEST_CASE("FTQ and FTQ_VR quantize K = 1152 and K = 272 into blobs that decode as the reference")
{
	static const int shapes[][2] = { { 64, 1152 }, { 48, 272 } };
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 272);
	int s, v, i;
	for (s = 0; s < 2; s++)
		for (v = 0; v < 3; v++)
		{
			// v = 0: static-rate FTQ_N20; 1: FTQ_VR_N20 at 20 everywhere; 2: FTQ_VR_N20 with tile row rates 12, 28, 20, ...
			const int rows = shapes[s][0], cols = shapes[s][1], count = rows * cols, R = rows / 16, C = cols / 16;
			const int format = v ? CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20 : CCV_NNC_QX_8I_ROWWISE_FTQ_N20;
			float* const values = ccmalloc(sizeof(float) * count);
			uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
			_test_tmcq_values(&dsfmt, values, rows, cols);
			ccv_float_to_half_precision(values, half, count);
			int* const rates = ccmalloc(sizeof(int) * R);
			for (i = 0; i < R; i++)
				rates[i] = v == 2 ? (i % 3 == 0 ? 12 : i % 3 == 1 ? 28 : 20) : 20;
			if (v == 2 && R % 3 == 1)
				rates[R - 1] = 20;
			else if (v == 2 && R % 3 == 2)
				rates[R - 2] = rates[R - 1] = 20;
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, count, cols);
			uint8_t* const blob = ccmalloc(size);
			const size_t written = v ? ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, rates, blob, size) : ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, blob, size);
			REQUIRE_EQ(written, size, "%d x %d variant %d writes the exact size", rows, cols, v);
			uint64_t* const row_bits = ccmalloc(sizeof(uint64_t) * R);
			size_t scale_offset;
			if (v)
			{
				REQUIRE(memcmp(blob, "CCVFTQVR", 8) == 0, "variable-rate magic");
				uint32_t word = 0;
				for (i = 0; i <= R; i++)
				{
					uint32_t off;
					memcpy(&off, blob + 32 + 4 * i, 4);
					REQUIRE_EQ(off, word, "%d x %d tile row %d offset is the sum of C n_r / 2", rows, cols, i);
					if (i < R)
						row_bits[i] = ((uint64_t)8 + R + 1 + off) * 32, word += (uint32_t)(C * rates[i] / 2);
				}
				scale_offset = _test_tiled_align128(32 + 4 * (R + 1) + (size_t)R * C * 2 * 20);
			} else {
				REQUIRE(memcmp(blob, "CCVFTQSR", 8) == 0, "static-rate magic");
				for (i = 0; i < R; i++)
					row_bits[i] = 256 + (uint64_t)i * C * 16 * 20;
				scale_offset = _test_tiled_align128(32 + (size_t)R * C * 2 * 20);
			}
			REQUIRE_EQ(size, scale_offset + 2 * rows, "%d x %d variant %d scales follow the code", rows, cols, v);
			uint16_t* const expected = ccmalloc(sizeof(uint16_t) * count);
			uint16_t* const actual = ccmalloc(sizeof(uint16_t) * count);
			_test_ftq_ref_decode(blob, rows, cols, rates, row_bits, scale_offset, expected);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, count), 0, "%d x %d variant %d decodes", rows, cols, v);
			REQUIRE_ARRAY_EQ(uint16_t, expected, actual, count, "%d x %d variant %d decode matches the reference", rows, cols, v);
			// The fit is close: relative error of the FP16 decode under the n = 20 TMCQ bound.
			double sse = 0, energy = 0;
			for (i = 0; i < count; i++)
			{
				float x;
				ccv_half_precision_to_float(actual + i, &x, 1);
				sse += ((double)x - values[i]) * ((double)x - values[i]);
				energy += (double)values[i] * values[i];
			}
			REQUIRE(sse / energy < 0.30, "%d x %d variant %d relative error %g", rows, cols, v, sse / energy);
			ccfree(actual);
			ccfree(expected);
			ccfree(row_bits);
			ccfree(blob);
			ccfree(rates);
			ccfree(half);
			ccfree(values);
		}
}

// Quantizes FP16 `half` of `params` (any number of leading dimensions) into a tiled `format`; FTQ-VR takes the tile
// row rates `rates` (0 for its budget everywhere).
static ccv_nnc_tensor_t* _test_tiled_quantize(const ccv_nnc_tensor_param_t params, const int format, const uint16_t* const half, const int* const rates)
{
	ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params, format), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
	const int nd = ccv_nnc_tensor_nd(params.dim);
	const size_t count = ccv_nnc_tensor_count(params), cols = params.dim[nd - 1];
	size_t written;
	if (ccv_nnc_8i_rowwise_x_is_ftq_vr(format))
	{
		const size_t R = count / cols / 16;
		int* const uniform = ccmalloc(sizeof(int) * R);
		size_t i;
		for (i = 0; i < R; i++)
			uniform[i] = rates ? rates[i] : ccv_nnc_8i_rowwise_x_ftq_vr_rate(format);
		written = ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, uniform, packed->data.u8, size);
		ccfree(uniform);
	} else
		written = ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed->data.u8, size);
	if (written != size)
	{
		ccv_nnc_tensor_free(packed);
		return 0;
	}
	return packed;
}

TEST_CASE("TMCQ, FTQ and FTQ_VR at K = 1152 and K = 272 survive tensor persistence, also with experts, and FTQ_VR naming a process slot is not written")
{
	// 2D tensors and a 3D tensor of 2 experts (its rows are the rows of both experts).
	static const int shapes[][3] = { { 0, 32, 1152 }, { 0, 48, 272 }, { 2, 16, 1152 }, { 2, 32, 272 } };
	const int formats[] = { CCV_NNC_QX_8I_ROWWISE_TMCQ_N24, CCV_NNC_QX_8I_ROWWISE_FTQ_N20, CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20, CCV_NNC_QX_8I_ROWWISE_FTQ_N88 };
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1152);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	// The alternate 512-pair table in a process slot: an FTQ_VR blob fitted with it decodes by slot, but a slot means
	// nothing in a file, so such a weight is not written (a checkpoint holds the built-in table or an inline one).
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	REQUIRE(slot_alt >= 1, "the alternate table registers");
	uint32_t crc_alt = 0;
	REQUIRE(ccv_nnc_codebook_get(slot_alt, 0, &crc_alt) != 0, "the alternate table has a crc");
	int s, f, external;
	for (s = 0; s < 4; s++)
		for (f = 0; f < 4; f++)
			for (external = 0; external < (f == 2 ? 2 : 1); external++)
			{
				const int experts = shapes[s][0], rows = shapes[s][1], cols = shapes[s][2];
				const int count = (experts ? experts : 1) * rows * cols;
				const ccv_nnc_tensor_param_t params = experts ? CPU_TENSOR_NHWC(16F, experts, rows, cols) : CPU_TENSOR_NHWC(16F, rows, cols);
				float* const values = ccmalloc(sizeof(float) * count);
				uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
				_test_tmcq_values(&dsfmt, values, count / cols, cols);
				ccv_float_to_half_precision(values, half, count);
				ccv_nnc_tensor_t* const packed = _test_tiled_quantize(params, formats[f], half, 0);
				REQUIRE(packed, "shape %d format %x quantizes", s, formats[f]);
				const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
				REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(formats[f], CCV_16F, count, cols), "shape %d format %x tensor size matches", s, formats[f]);
				char name[48];
				snprintf(name, sizeof(name), "w_%d_%d_%d", s, f, external);
				if (external)
				{
					// The header table field (id 3, 2^9 pairs, plain) and its crc; the payload now decodes with that table.
					const uint32_t field = 3u | (9u << 8);
					memcpy(packed->data.u8 + 24, &field, 4);
					memcpy(packed->data.u8 + 28, &crc_alt, 4);
					const int slot_format = ccv_nnc_codebook_with_slot(formats[f], slot_alt);
					uint16_t* const a = ccmalloc(sizeof(uint16_t) * count);
					uint16_t* const b = ccmalloc(sizeof(uint16_t) * count);
					REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, slot_format, a, count), 0, "%s decodes by slot", name);
					REQUIRE(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, formats[f], b, count) != 0, "%s naming another table does not decode with the built-in one", name);
					packed->info.reserved = slot_format;
					REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_ERROR, "%s naming a process slot is not written", name);
					ccfree(b);
					ccfree(a);
					ccv_nnc_tensor_free(packed);
					ccfree(half);
					ccfree(values);
					continue;
				}
				REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "%s writes", name);
				ccv_nnc_tensor_t* restored = 0;
				REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s reads", name);
				const int expected_reserved = formats[f];
				REQUIRE_EQ(restored->info.reserved, expected_reserved, "%s codec", name);
				REQUIRE_EQ(restored->info.datatype, packed->info.datatype, "%s source precision persists", name);
				REQUIRE_EQ(ccv_nnc_tensor_nd(restored->info.dim), experts ? 3 : 2, "%s keeps its dimensions", name);
				REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "%s size persists", name);
				REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "%s payload, header and scales persist", name);
				uint16_t* const a = ccmalloc(sizeof(uint16_t) * count);
				uint16_t* const b = ccmalloc(sizeof(uint16_t) * count);
				REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, expected_reserved, a, count), 0, "%s source decodes", name);
				REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, restored->info.reserved, b, count), 0, "%s restored decodes", name);
				REQUIRE_ARRAY_EQ(uint16_t, a, b, count, "%s restored tensor decodes the same", name);
				ccfree(b);
				ccfree(a);
				ccv_nnc_tensor_free(restored);
				ccv_nnc_tensor_free(packed);
				ccfree(half);
				ccfree(values);
			}
	sqlite3_close(db);
}

TEST_CASE("FTQ and FTQ_VR with an inline codebook read without a codebook tensor, register a folded table from the blob header, and fail on a mismatch")
{
	enum { rows = 32, cols = 1152 };
	const int formats[] = { CCV_NNC_QX_8I_ROWWISE_FTQ_N20, CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20 };
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2048);
	// The alternate table, and its pairs made non-negative and registered folded: the two structures the header names.
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	REQUIRE(slot_alt >= 1, "the alternate table registers");
	int entries = 0;
	uint32_t crc_alt = 0;
	const int8_t* const alt_pairs = ccv_nnc_codebook_get(slot_alt, &entries, &crc_alt);
	REQUIRE(alt_pairs && entries == 512, "the alternate table has 512 pairs");
	int8_t folded[1024];
	int i;
	for (i = 0; i < 1024; i++)
		folded[i] = alt_pairs[i] < 0 ? -alt_pairs[i] : alt_pairs[i];
	const int slot_fold = ccv_nnc_codebook_register_with_flags(folded, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(slot_fold >= 1 && ccv_nnc_codebook_flags(slot_fold) == CCV_NNC_CODEBOOK_FOLD, "the folded table registers folded");
	uint32_t crc_fold = 0;
	ccv_nnc_codebook_get(slot_fold, 0, &crc_fold);
	const int table_slots[2] = { slot_alt, slot_fold };
	const uint32_t table_crcs[2] = { crc_alt, crc_fold };
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	float* const values = ccmalloc(sizeof(float) * rows * cols);
	uint16_t* const half = ccmalloc(sizeof(uint16_t) * rows * cols);
	uint16_t* const by_slot = ccmalloc(sizeof(uint16_t) * rows * cols);
	uint16_t* const by_inline = ccmalloc(sizeof(uint16_t) * rows * cols);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	ccv_float_to_half_precision(values, half, rows * cols);
	int f, t;
	for (f = 0; f < 2; f++)
	{
		REQUIRE(ccv_nnc_8i_rowwise_x_codebook_is_valid(ccv_nnc_8i_rowwise_x_with_inline_codebook(formats[f], 512)), "format %x takes an inline codebook", formats[f]);
		for (t = 0; t < 2; t++)
		{
			ccv_nnc_tensor_t* const packed = _test_tiled_quantize(CPU_TENSOR_NHWC(16F, rows, cols), formats[f], half, 0);
			REQUIRE(packed, "format %x quantizes", formats[f]);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
			// The header keeps the id the blob was fitted with (3 here; it is not looked up), 2^9 pairs and the
			// structure (bit 16: folded), and the table's crc.
			const uint32_t field = 3u | (9u << 8) | ((uint32_t)t << 16);
			memcpy(packed->data.u8 + 24, &field, 4);
			memcpy(packed->data.u8 + 28, &table_crcs[t], 4);
			const int slot_format = ccv_nnc_codebook_with_slot(formats[f], table_slots[t]);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, slot_format, by_slot, rows * cols), 0, "format %x table %d decodes by slot", formats[f], t);
			const int format = ccv_nnc_8i_rowwise_x_with_inline_codebook(formats[f], 512);
			const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(format);
			REQUIRE_EQ(prefix, 1152, "the prefix of 512 pairs is 1152 bytes");
			ccv_nnc_tensor_param_t params = packed->info;
			params.reserved = format;
			ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, params, 0);
			REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(weight->info), prefix + size, "the tensor holds the prefix and the blob");
			memset(weight->data.u8, 0, prefix);
			// A folded slot serializes its 512 pairs; the structure stays in the blob header.
			REQUIRE_EQ(ccv_nnc_codebook_serialize(table_slots[t], weight->data.u8, prefix), 16 + 1024, "the container serializes");
			memcpy(weight->data.u8 + prefix, packed->data.u8, size);
			char name[32];
			snprintf(name, sizeof(name), "ftq_inline_%d_%d", f, t);
			REQUIRE_EQ(ccv_nnc_tensor_write(weight, db, name, 0), CCV_IO_FINAL, "%s writes", name);
			ccv_nnc_tensor_t* restored = 0;
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s reads without a codebook tensor", name);
			REQUIRE_EQ(restored->info.reserved, format, "%s keeps the inline bits and slot 255", name);
			REQUIRE_EQ(memcmp(restored->data.u8, weight->data.u8, prefix + size), 0, "%s data is unchanged", name);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, prefix + size, cols, restored->info.reserved, by_inline, rows * cols), 0, "%s decodes inline", name);
			REQUIRE_ARRAY_EQ(uint16_t, by_inline, by_slot, rows * cols, "%s inline decodes like its table by slot", name);
			// The structure decides the slot: the inline table resolves to the slot registered with the same flags.
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_resolve_inline(restored->data.u8, prefix + size, format) & CCV_NNC_CODEBOOK_SLOT_MASK, table_slots[t] << CCV_NNC_CODEBOOK_SLOT_SHIFT, "%s resolves to the table's slot", name);
			ccv_nnc_tensor_free(restored);
			// A header crc that is not the inline table's fails the read and the decode.
			uint8_t* const bad = ccmalloc(prefix + size);
			memcpy(bad, weight->data.u8, prefix + size);
			memcpy(bad + prefix + 28, &table_crcs[1 - t], 4);
			REQUIRE_EQ(_test_set_stored_data(db, name, bad, prefix + size), 0, "data updates");
			restored = 0;
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s with another table's crc fails", name);
			REQUIRE(restored == 0, "no tensor on failure");
			REQUIRE(ccv_nnc_dequantize_8i_rowwise_x(bad, CCV_16F, CCV_TENSOR_CPU_MEMORY, prefix + size, cols, format, by_inline, rows * cols) != 0, "%s with another table's crc does not decode", name);
			// The header names a 1024-pair table.
			memcpy(bad, weight->data.u8, prefix + size);
			const uint32_t log2_field = field + (1u << 8);
			memcpy(bad + prefix + 24, &log2_field, 4);
			REQUIRE_EQ(_test_set_stored_data(db, name, bad, prefix + size), 0, "data updates");
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s with another P in its header fails", name);
			// Structure 2 does not exist.
			memcpy(bad, weight->data.u8, prefix + size);
			const uint32_t structure_field = (field & ~(3u << 16)) | (2u << 16);
			memcpy(bad + prefix + 24, &structure_field, 4);
			REQUIRE_EQ(_test_set_stored_data(db, name, bad, prefix + size), 0, "data updates");
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s with structure 2 fails", name);
			// A corrupt inline container (one table byte changed, the crc kept).
			memcpy(bad, weight->data.u8, prefix + size);
			bad[16] ^= 1;
			REQUIRE_EQ(_test_set_stored_data(db, name, bad, prefix + size), 0, "data updates");
			REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s with a corrupt inline table fails", name);
			REQUIRE(restored == 0, "no tensor on failure");
			ccfree(bad);
			ccv_nnc_tensor_free(weight);
			ccv_nnc_tensor_free(packed);
		}
	}
	sqlite3_close(db);
	ccfree(by_inline);
	ccfree(by_slot);
	ccfree(half);
	ccfree(values);
}

TEST_CASE("codebook containers take the CCVCBOOK magic only, registered or inline")
{
	uint8_t legacy[sizeof(test_trellis_table_alt)];
	memcpy(legacy, test_trellis_table_alt, sizeof(legacy));
	REQUIRE_EQ(memcmp(legacy, "CCVCBOOK", 8), 0, "the asset has the CCVCBOOK magic");
	const int slot = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	REQUIRE(slot >= 1, "the CCVCBOOK container registers");
	memcpy(legacy, "CCVCB001", 8);
	REQUIRE_EQ(ccv_nnc_codebook_register_serialized(legacy, sizeof(legacy)), -1, "the old CCVCB001 magic fails");
	const int format = ccv_nnc_8i_rowwise_x_with_inline_codebook(CCV_NNC_QX_8I_ROWWISE_FTQ_N20, 512);
	uint8_t prefix[1152];
	memset(prefix, 0, sizeof(prefix));
	memcpy(prefix, test_trellis_table_alt, sizeof(test_trellis_table_alt));
	REQUIRE_EQ(ccv_nnc_codebook_resolve_inline(prefix, format) & CCV_NNC_CODEBOOK_SLOT_MASK, slot << CCV_NNC_CODEBOOK_SLOT_SHIFT, "the CCVCBOOK magic resolves inline");
	memcpy(prefix, "CCVCB001", 8);
	REQUIRE_EQ(ccv_nnc_codebook_resolve_inline(prefix, format), -1, "the old magic does not resolve inline");
}

TEST_CASE("FTQ_VR under budget pads the code with zeros, and ftq_vr_check rejects bad shapes, padding and offsets")
{
	static const int shapes[][2] = { { 48, 272 }, { 64, 1152 } };
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 2304);
	// nm20 with tile rows 12, 16, 12, ...; nm72 with 48, 88, 48, ... (across 64: n48 is bit 20 of word 3, n88 bit 40 of
	// the level mask, bit 26 of the table word).
	static const int means[2] = { 20, 72 }, low[2] = { 12, 48 }, high[2] = { 16, 88 };
	int c, i;
	for (c = 0; c < 4; c++)
	{
		const int v = c / 2, s = c % 2, nm = means[v];
		const int format = CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 + (nm - 16) / 2;
		const int rows = shapes[s][0], cols = shapes[s][1], count = rows * cols, R = rows / 16, C = cols / 16;
		float* const values = ccmalloc(sizeof(float) * count);
		uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
		_test_tmcq_values(&dsfmt, values, rows, cols);
		ccv_float_to_half_precision(values, half, count);
		// Tile row rates low, high, low, ...: sum(n_r) < nm R, thus the code ends before the budget.
		int* const rates = ccmalloc(sizeof(int) * R);
		int sum = 0;
		for (i = 0; i < R; i++)
			sum += rates[i] = i % 2 ? high[v] : low[v];
		REQUIRE(sum < R * nm, "under budget");
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, count, cols);
		uint8_t* const blob = ccmalloc(size);
		memset(blob, 0xa5, size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, rates, blob, size), size, "%d x %d under budget writes the exact size", rows, cols);
		const size_t code_start = 32 + 4 * (R + 1);
		const size_t code_end = code_start + (size_t)C * sum / 2 * 4;
		const size_t scale_offset = _test_tiled_align128(code_start + (size_t)R * C * 2 * nm);
		REQUIRE_EQ(size, scale_offset + 2 * rows, "%d x %d scales follow the budget", rows, cols);
		uint32_t last;
		memcpy(&last, blob + 32 + 4 * R, 4);
		REQUIRE_EQ((size_t)last, (size_t)C * sum / 2, "%d x %d the last offset is sum C n_r / 2", rows, cols);
		size_t p, nonzero = 0;
		for (p = code_end; p < scale_offset; p++)
			nonzero += blob[p] != 0;
		REQUIRE_EQ((int)nonzero, 0, "%d x %d padding from the code end to the scales is zero", rows, cols);
		REQUIRE(scale_offset - code_end >= (size_t)C * (R * nm - sum) * 2, "%d x %d the unused budget is padding", rows, cols);
		uint64_t level_mask = 0;
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows, cols, &level_mask), 0, "%d x %d nm%d under budget passes the check", rows, cols, nm);
		const uint64_t want_mask = ((uint64_t)1 << ((low[v] - 8) / 2)) | (R > 1 ? (uint64_t)1 << ((high[v] - 8) / 2) : 0);
		REQUIRE_EQ(level_mask, want_mask, "%d x %d level mask has %d and %d", rows, cols, low[v], high[v]);
		// The encoder splits the mask as QuantStar's split_level_mask: bits 0 - 31 in word 3, 32 - 40 in table word bits 18 - 26.
		uint32_t header[6];
		memcpy(header, blob + 8, sizeof(header));
		REQUIRE_EQ(header[3], (uint32_t)want_mask, "%d x %d nm%d header word 3", rows, cols, nm);
		REQUIRE_EQ(header[4], (9u << 8) | (uint32_t)((want_mask >> 32) << 18), "%d x %d nm%d table word", rows, cols, nm);
		uint64_t* const row_bits = ccmalloc(sizeof(uint64_t) * R);
		uint32_t word = 0;
		for (i = 0; i < R; i++)
			row_bits[i] = ((uint64_t)8 + R + 1 + word) * 32, word += (uint32_t)(C * rates[i] / 2);
		uint16_t* const expected = ccmalloc(sizeof(uint16_t) * count);
		uint16_t* const actual = ccmalloc(sizeof(uint16_t) * count);
		_test_ftq_ref_decode(blob, rows, cols, rates, row_bits, scale_offset, expected);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, count), 0, "%d x %d under budget decodes", rows, cols);
		REQUIRE_ARRAY_EQ(uint16_t, expected, actual, count, "%d x %d under budget decode matches the reference", rows, cols);
		// Shapes off the 16 x 16 grid, or not the header's.
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows, cols - 8, 0), -1, "K %% 16 != 0 fails");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows, 264, 0), -1, "K = 264 fails");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows, cols + 16, 0), -1, "a K other than the header's fails");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size, format, rows - 8, cols, 0), -1, "rows %% 16 != 0 fails");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(blob, size - 1, format, rows, cols, 0), -1, "a short blob fails");
		// A nonzero padding byte, the first and the last.
		uint8_t* const bad = ccmalloc(size);
		const size_t pads[2] = { code_end, scale_offset - 1 };
		for (i = 0; i < 2; i++)
		{
			memcpy(bad, blob, size);
			bad[pads[i]] = 1;
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(bad, size, format, rows, cols, 0), -1, "%d x %d nonzero padding at %zu fails", rows, cols, pads[i]);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(bad, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, count), -1, "%d x %d decode of nonzero padding fails", rows, cols);
		}
		// Offsets: a word difference that is not a multiple of C (off[1] + 1), one that is C n / 2 with n = n_0 + 2 (18 or
		// 50, not a level of the mask), a decreasing pair, and off[0] != 0.
		uint32_t off1;
		memcpy(&off1, blob + 36, 4);
		const uint32_t offsets[][2] = { { 1, off1 + 1 }, { 1, off1 + C }, { 0, 1 } };
		for (i = 0; i < 3; i++)
		{
			memcpy(bad, blob, size);
			memcpy(bad + 32 + 4 * offsets[i][0], &offsets[i][1], 4);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(bad, size, format, rows, cols, 0), -1, "%d x %d offset case %d fails", rows, cols, i);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(bad, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, actual, count), -1, "%d x %d decode of offset case %d fails", rows, cols, i);
		}
		if (R > 1)
		{
			memcpy(bad, blob, size);
			uint32_t off2;
			memcpy(&off2, bad + 40, 4);
			const uint32_t over = off2 + 1; // off[1] past off[2]
			memcpy(bad + 36, &over, 4);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(bad, size, format, rows, cols, 0), -1, "%d x %d decreasing offsets fail", rows, cols);
		}
		ccfree(bad);
		ccfree(actual);
		ccfree(expected);
		ccfree(row_bits);
		ccfree(blob);
		ccfree(rates);
		ccfree(half);
		ccfree(values);
	}
}

#include "case_main.h"
