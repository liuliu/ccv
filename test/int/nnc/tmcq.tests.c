#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
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

TEST_CASE("TMCQ quantizer returns 0 with an imatrix, a Hessian or a codebook slot")
{
	enum { rows = 16, cols = 256, count = rows * cols };
	static float values[count];
	static float hessian[cols * cols];
	static uint8_t output[count * 4];
	float imatrix[cols];
	int i;
	for (i = 0; i < count; i++)
		values[i] = sinf(i * 0.1f) * 0.05f;
	for (i = 0; i < cols; i++)
		imatrix[i] = 1, hessian[i * cols + i] = 1;
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20;
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, count, cols);
	REQUIRE(size > 0 && size <= sizeof(output), "16 x 256 has a size");
	memset(output, 0xa5, sizeof(output));
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, imatrix, cols, 0, 0, output, size), 0, "an imatrix must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, hessian, cols * cols, output, size), 0, "a Hessian must fail");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format | (1 << CCV_NNC_CODEBOOK_SLOT_SHIFT), 0, 0, 0, 0, output, size), 0, "a codebook slot must fail");
	for (i = 0; i < (int)sizeof(output); i++)
		REQUIRE_EQ(output[i], 0xa5, "rejected calls leave the output untouched");
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, output, size), size, "the unweighted call quantizes");
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
// j = (j' + rotation) mod 128. start < 0: free ends. bp is 128 x 2^(W - 2) indexed by j'. Returns the end state.
static int _test_trellis_pass(const int window, const double* const y, const double* const w, const int n, const int rotation, const int start, uint8_t* const bp, double* const cost_out)
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

static int _test_trellis_trace(const int window, const uint8_t* const bp, const int n, const int rotation, const int last, const int first, int s, int* const states)
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
	uint8_t* const bp = ccmalloc(128 << (window - 2));
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

// Unweighted trellis encoder written out from the format, with either anchor: the TMCQ 32F output (W = 15, code from
// bit 0, float row scales). Row scale s = rms(row) / cb_rms, every pair weighted by s^2. Returns the summed Viterbi tile costs.
static double _test_trellis_encode(const float* const x, const int rows, const int cols, const int n, const int window, const int mid, uint8_t* const output)
{
	const int T = 16 * n;
	const size_t scale_offset = _test_tmcq_scale_offset(rows, cols, n);
	memset(output, 0, scale_offset);
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
			((float*)(output + scale_offset))[tr * 16 + i] = (float)scale[i];
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
						output[(g * T + p) / 8] |= 1 << ((g * T + p) % 8);
				prev_end = end;
			}
		}
	}
	return total;
}

TEST_CASE("TMCQ mid anchor matches an independent encoder and costs no more than the plain two-pass fit")
{
	enum { rows = 32, cols = 256, count = rows * cols };
	static const int formats[2] = {
		CCV_NNC_QX_8I_ROWWISE_TMCQ_N16,
		CCV_NNC_QX_8I_ROWWISE_TMCQ_N24,
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 4242);
	float* const values = ccmalloc(sizeof(float) * count);
	_test_tmcq_values(&dsfmt, values, rows, cols);
	int c;
	for (c = 0; c < 2; c++)
	{
		const int format = formats[c];
		const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_32F, count, cols);
		uint8_t* const packed = ccmalloc(size);
		uint8_t* const mid = ccmalloc(size);
		uint8_t* const plain = ccmalloc(size);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed, size), size, "quantization succeeds");
		const double mid_cost = _test_trellis_encode(values, rows, cols, n, 15, 1, mid);
		const double plain_cost = _test_trellis_encode(values, rows, cols, n, 15, 0, plain);
		REQUIRE_ARRAY_EQ(uint8_t, packed, mid, size, "library output matches the independent encoder for n = %d", n);
		printf("\n\tTMCQ n = %d trellis cost mid %.6g plain %.6g (%+.3f%%)", n, mid_cost, plain_cost, (mid_cost / plain_cost - 1) * 100);
		REQUIRE(mid_cost <= plain_cost, "mid anchor trellis cost %g is no more than plain %g for n = %d", mid_cost, plain_cost, n);
		ccfree(plain);
		ccfree(mid);
		ccfree(packed);
	}
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

// Writes the bytes of a serialized codebook (valid or not) as __ccv_codebook__/<id>, the way the exporter stores it.
static int _test_write_codebook_bytes(sqlite3* const db, const int id, const uint8_t* const bytes, const int size)
{
	ccv_nnc_tensor_t* const tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(8U, size), 0);
	memcpy(tensor->data.u8, bytes, size);
	char name[64];
	snprintf(name, sizeof(name), CCV_NNC_CODEBOOK_TENSOR_PREFIX "%d", id);
	const int result = ccv_nnc_tensor_write(tensor, db, name, 0);
	ccv_nnc_tensor_free(tensor);
	return result;
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

TEST_CASE("TMCQ weights with a codebook id read their codebook from the checkpoint, TMCQ files without one read as before")
{
	enum { rows = 32, cols = 256, n = 24 };
	// A nonzero codebook id (reserved bits 16 - 23) is the file's codebook; 0 is the built-in table.
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
	// Codebooks: the alternate 512-pair table and a synthetic 2048-pair table.
	const int slot_alt = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	int8_t big[4096];
	for (i = 0; i < 4096; i++)
		big[i] = (int8_t)((i * 37 + 11) % 255 - 127);
	const int slot_big = ccv_nnc_codebook_register(big, 2048);
	REQUIRE(slot_alt >= 1 && slot_big >= 1 && slot_big != slot_alt, "both register");
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	// The old file layout: TMCQ, reserved == codec, reads back unchanged.
	REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, "old", 0), CCV_IO_FINAL, "TMCQ writes");
	ccv_nnc_tensor_t* restored = 0;
	REQUIRE_EQ(ccv_nnc_tensor_read(db, "old", 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ reads");
	REQUIRE_EQ(restored->info.reserved, tmcq, "the TMCQ reserved field is unchanged");
	REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "TMCQ payload is unchanged");
	ccv_nnc_tensor_free(restored);
	REQUIRE_EQ(ccv_nnc_tensor_write_codebook(db, slot_alt, 3), CCV_IO_FINAL, "the alternate table writes as id 3");
	REQUIRE_EQ(ccv_nnc_tensor_write_codebook(db, slot_big, 7), CCV_IO_FINAL, "the big table writes as id 7");
	REQUIRE_EQ(ccv_nnc_tensor_write_codebook(db, 0, 1), CCV_IO_ERROR, "the built-in table is never written");
	REQUIRE_EQ(ccv_nnc_tensor_read_codebook(db, 3), slot_alt, "id 3 reads back as the alternate slot");
	REQUIRE_EQ(ccv_nnc_tensor_read_codebook(db, 4), -1, "a missing id fails");
	REQUIRE_EQ(ccv_nnc_tensor_read_codebook(db, 0), -1, "id 0 is not a file codebook");
	const int ids[2] = { 3, 7 };
	const int slots[2] = { slot_alt, slot_big };
	int c;
	float* const a = ccmalloc(sizeof(float) * rows * cols);
	float* const builtin = ccmalloc(sizeof(float) * rows * cols);
	int* const q8 = ccmalloc(sizeof(int) * rows * cols);
	ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq, builtin, rows * cols);
	for (c = 0; c < 2; c++)
	{
		ccv_nnc_tensor_t* const external = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq | (ids[c] << 16)), 0);
		memcpy(external->data.u8, packed->data.u8, size);
		char name[32];
		snprintf(name, sizeof(name), "w%d", c);
		REQUIRE_EQ(ccv_nnc_tensor_write(external, db, name, 0), CCV_IO_FINAL, "TMCQ writes with its file id");
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "TMCQ with a file id reads");
		REQUIRE_EQ(restored->info.reserved, tmcq | (slots[c] << 16), "the file id becomes the process slot");
		REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "TMCQ payload with a file id is unchanged");
		int entries = 0;
		const int8_t* const table = ccv_nnc_codebook_get(ccv_nnc_8i_rowwise_x_codebook_slot(restored->info.reserved), &entries, 0);
		_test_tmcq_decode_table(packed->data.u8, rows, cols, n, table, ccv_nnc_codebook_log2_entries(entries), q8);
		ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, size, cols, restored->info.reserved, a, rows * cols);
		const uint8_t* const scales = packed->data.u8 + _test_tmcq_scale_offset(rows, cols, n);
		int same_as_builtin = 1;
		for (i = 0; i < rows * cols; i++)
		{
			REQUIRE_EQ_WITH_TOLERANCE(a[i], (float)(_test_tmcq_load_scale(scales, CCV_32F, i / cols) * q8[i]), 0, "CPU decode with codebook %d matches the reference at %d", c, i);
			same_as_builtin = same_as_builtin && a[i] == builtin[i];
		}
		REQUIRE(!same_as_builtin, "codebook %d does not decode like the built-in table", c);
		ccv_nnc_tensor_free(restored);
		// A metadata read maps the id too, and so does a read into caller params that still hold the file id.
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_FINAL, "metadata reads");
		REQUIRE_EQ(restored->info.reserved, tmcq | (slots[c] << 16), "metadata carries the process slot");
		ccv_nnc_tensor_free(restored);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, &external->info, &restored), CCV_IO_FINAL, "read into caller params");
		REQUIRE_EQ(restored->info.reserved, tmcq | (slots[c] << 16), "caller params get the process slot");
		ccv_nnc_tensor_free(restored);
		ccv_nnc_tensor_free(external);
	}
	// A second file with the same table under another id resolves to the same slot.
	sqlite3* db2 = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db2), SQLITE_OK, "second database opens");
	REQUIRE_EQ(_test_write_codebook_bytes(db2, 9, test_trellis_table_alt, sizeof(test_trellis_table_alt)), CCV_IO_FINAL, "codebook bytes write");
	REQUIRE_EQ(ccv_nnc_tensor_read_codebook(db2, 9), slot_alt, "another file, another id, the same slot");
	// Inconsistent or broken files fail instead of decoding with the built-in table.
	struct {
		const char* name;
		int reserved;
	} invalid[] = {
		{ "missing", tmcq | (5 << 16) },
		{ "bad_magic", tmcq | (11 << 16) },
		{ "bad_crc", tmcq | (12 << 16) },
		{ "bad_entries", tmcq | (13 << 16) },
		{ "truncated", tmcq | (14 << 16) },
	};
	uint8_t bad[sizeof(test_trellis_table_alt)];
	memcpy(bad, test_trellis_table_alt, sizeof(bad));
	bad[3] = 'X';
	REQUIRE_EQ(_test_write_codebook_bytes(db2, 11, bad, sizeof(bad)), CCV_IO_FINAL, "codebook bytes write");
	memcpy(bad, test_trellis_table_alt, sizeof(bad));
	bad[200] ^= 4;
	REQUIRE_EQ(_test_write_codebook_bytes(db2, 12, bad, sizeof(bad)), CCV_IO_FINAL, "codebook bytes write");
	memcpy(bad, test_trellis_table_alt, sizeof(bad));
	_test_store_u32(bad + 8, 1000);
	REQUIRE_EQ(_test_write_codebook_bytes(db2, 13, bad, sizeof(bad)), CCV_IO_FINAL, "codebook bytes write");
	REQUIRE_EQ(_test_write_codebook_bytes(db2, 14, test_trellis_table_alt, sizeof(test_trellis_table_alt) - 2), CCV_IO_FINAL, "codebook bytes write");
	for (i = 0; i < (int)(sizeof(invalid) / sizeof(invalid[0])); i++)
	{
		ccv_nnc_tensor_param_t params = ccv_nnc_tensor_8i_rowwise_x(source->info, tmcq);
		params.reserved = invalid[i].reserved;
		ccv_nnc_tensor_t* const weight = ccv_nnc_tensor_new(0, params, 0);
		memcpy(weight->data.u8, packed->data.u8, size);
		REQUIRE_EQ(ccv_nnc_tensor_write(weight, db2, invalid[i].name, 0), CCV_IO_FINAL, "writes");
		ccv_nnc_tensor_free(weight);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db2, invalid[i].name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s fails", invalid[i].name);
		REQUIRE(restored == 0, "%s leaves no tensor", invalid[i].name);
		REQUIRE_EQ(ccv_nnc_tensor_read(db2, invalid[i].name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_ERROR, "%s metadata fails", invalid[i].name);
	}
	sqlite3_close(db2);
	sqlite3_close(db);
	ccfree(q8);
	ccfree(builtin);
	ccfree(a);
	ccv_nnc_tensor_free(packed);
	ccv_nnc_tensor_free(source);
}

// K % 16: TMCQ cuts the weight into (rows / 16) x (K / 16) tiles of 16 n
// bits, so any K % 16 == 0 is a whole tile grid. The cases below cover K % 256 (sizes must not change), the MoE
// down projection K = 1152 (72 tiles, 4.5 groups of 256), and K = 272 / 48 / 16 (an odd tile count per tile row).

static size_t _test_tiled_align128(const size_t x)
{
	return (x + 127) & ~(size_t)127;
}

static const int tiled_shapes[][2] = {
	{ 16, 256 }, { 32, 512 }, { 48, 768 }, { 16, 4352 }, { 2304, 1152 }, { 32, 1152 }, { 16, 272 }, { 48, 144 }, { 16, 16 }, { 64, 48 },
};

TEST_CASE("TMCQ sizes follow the 16 x 16 tile grid for K % 16 and are unchanged for K % 256")
{
	int s, f;
	for (s = 0; s < (int)(sizeof(tiled_shapes) / sizeof(tiled_shapes[0])); s++)
	{
		const int rows = tiled_shapes[s][0], cols = tiled_shapes[s][1];
		const size_t count = (size_t)rows * cols;
		const size_t tiles = (size_t)rows / 16 * (cols / 16);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_tile_count(rows, cols), (uint64_t)tiles, "%d x %d tile count", rows, cols);
		// TMCQ_N16 .. N32.
		for (f = 0; f < 9; f++)
		{
			const int format = tmcq_formats[f];
			const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
			const int datatype = CCV_32F;
			const size_t code = tiles * 2 * n;
			const size_t expected = _test_tiled_align128(code) + (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatype);
			const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, datatype, count, cols);
			REQUIRE_EQ(size, expected, "format %d %d x %d is the tile grid plus aligned scales", format, rows, cols);
			ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
			params.datatype = datatype;
			params = ccv_nnc_tensor_8i_rowwise_x(params, format);
			REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(params), expected, "format %d %d x %d tensor allocation agrees", format, rows, cols);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_payload_bits(format, rows, cols), (uint64_t)code * 8, "format %d %d x %d payload bits", format, rows, cols);
			// The group formula of the row-wise codecs, rows x ceil(K / 256) tiles' worth of bits, is what the sizes
			// were before: byte for byte the same for K % 256, and 1 / 9 too large at K = 1152.
			const size_t row_wise = _test_tiled_align128(((size_t)rows * ((cols + 255) / 256) * 16 * n + 7) / 8) + (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatype);
			if (cols % 256 == 0)
			{
				REQUIRE_EQ(size, row_wise, "format %d %d x %d keeps its K %% 256 size", format, rows, cols);
			} else {
				REQUIRE(size < row_wise, "format %d %d x %d is smaller than the per-row group count", format, rows, cols);
			}
		}
	}
	// The MoE down projection of 128 experts of 2304 x 1152 stacked along the rows, at TMCQ_N16: 2304 / 16 x 72 tiles
	// of 32 bytes per expert, no padding between experts.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_16F, (size_t)128 * 2304 * 1152, 1152), (size_t)128 * 144 * 72 * 32 + 128 * 2304 * 2, "128 stacked 2304 x 1152 experts");
	// K not a multiple of 16 or rows not a multiple of 16 have no size.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, CCV_16F, 24 * 1152, 1152), 0, "TMCQ 24 rows have no size");
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

TEST_CASE("TMCQ at K = 1152 and K = 272 survives tensor persistence, also with experts")
{
	// 2D tensors and a 3D tensor of 2 experts (its rows are the rows of both experts).
	static const int shapes[][3] = { { 0, 32, 1152 }, { 0, 48, 272 }, { 2, 16, 1152 }, { 2, 32, 272 } };
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 1152);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	int s;
	for (s = 0; s < 4; s++)
	{
		const int experts = shapes[s][0], rows = shapes[s][1], cols = shapes[s][2];
		const int count = (experts ? experts : 1) * rows * cols;
		const ccv_nnc_tensor_param_t params = experts ? CPU_TENSOR_NHWC(16F, experts, rows, cols) : CPU_TENSOR_NHWC(16F, rows, cols);
		float* const values = ccmalloc(sizeof(float) * count);
		uint16_t* const half = ccmalloc(sizeof(uint16_t) * count);
		_test_tmcq_values(&dsfmt, values, count / cols, cols);
		ccv_float_to_half_precision(values, half, count);
		ccv_nnc_tensor_t* const packed = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params, format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(packed->info);
		REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, count, cols), "shape %d tensor size matches", s);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, packed->data.u8, size), size, "shape %d quantizes", s);
		char name[48];
		snprintf(name, sizeof(name), "w_%d", s);
		REQUIRE_EQ(ccv_nnc_tensor_write(packed, db, name, 0), CCV_IO_FINAL, "%s writes", name);
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s reads", name);
		REQUIRE_EQ(restored->info.reserved, format, "%s codec persists", name);
		REQUIRE_EQ(restored->info.datatype, packed->info.datatype, "%s source precision persists", name);
		REQUIRE_EQ(ccv_nnc_tensor_nd(restored->info.dim), experts ? 3 : 2, "%s keeps its dimensions", name);
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(restored->info), size, "%s size persists", name);
		REQUIRE_EQ(memcmp(restored->data.u8, packed->data.u8, size), 0, "%s payload and scales persist", name);
		uint16_t* const a = ccmalloc(sizeof(uint16_t) * count);
		uint16_t* const b = ccmalloc(sizeof(uint16_t) * count);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(packed->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, a, count), 0, "%s source decodes", name);
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

#include "case_main.h"
