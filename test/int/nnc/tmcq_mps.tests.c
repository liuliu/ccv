#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <nnc/ccv_nnc_tmcq_table.inc>

// Metal TMCQ tests: the int8 and fp decode kernels, the selected (MoE expert) int8 decode, the TMCQ
// GEMV (M = 1..3) and the decode-then-GEMM paths. Every expected value comes from tmcq_ref_q8 /
// tmcq_ref_decode below, written from the TMCQ v1.2 spec at the byte level (only the frozen pair
// table ccv_nnc_tmcq_p2_table is shared), independent of the Metal helpers in
// lib/nnc/mfa/kernels/TMCQDecodeSource.hpp. Any bit string is a valid
// TMCQ payload, so payloads are random bytes.
//
// Set TMCQ_CPU_REFERENCE=1 once the CPU TMCQ decode (ccv_nnc_dequantize_8i_rowwise_x on CPU memory)
// is in the build, and the fp decode test also checks the GPU against it.

TEST_SETUP()
{
	ccv_nnc_init();
}

static uint32_t tmcq_rng_next(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

static int tmcq_rate_of(const int format)
{
	return 16 + 2 * (format - CCV_NNC_QX_8I_ROWWISE_TMCQ_N16);
}

// Code `half` (0 for an even weight, 1 for an odd one) of the pair that codeword cw selects:
// entry (cw * 0x83DCD12D) >> 23 of the 512-pair table.
static int tmcq_ref_q8(const uint32_t cw, const int half)
{
	const uint32_t k = (cw * 0x83DCD12Du) >> 23;
	return ccv_nnc_tmcq_p2_table[2 * k + half];
}

// Weight t (row t / 16, column t % 16) of tile `tile`: the 15-bit LSB-first window of weight t | 1,
// ending at E_{t|1} = (((t | 1) + 1) n) >> 4 and wrapping inside the tile's T = 16 n bits, then half
// t & 1 of its pair.
static int tmcq_ref_decode(const uint8_t* const payload, const size_t tile, const int n, const int t)
{
	const size_t T = (size_t)16 * n;
	const size_t base = tile * T;
	const size_t end = ((size_t)((t | 1) + 1) * n) >> 4;
	uint32_t cw = 0;
	int j;
	for (j = 0; j < 15; j++)
	{
		const size_t p = base + (end + T - 15 + j) % T;
		cw |= (uint32_t)((payload[p >> 3] >> (p & 7)) & 1) << j;
	}
	return tmcq_ref_q8(cw, t & 1);
}

// q8 of the whole [rows, cols] weight, row-major.
static void tmcq_ref_decode_matrix(const uint8_t* const payload, const int rows, const int cols, const int n, int8_t* const q8)
{
	const int tiles_per_row = cols / 16;
	int tr, tc, t;
	for (tr = 0; tr < rows / 16; tr++)
		for (tc = 0; tc < tiles_per_row; tc++)
			for (t = 0; t < 256; t++)
				q8[(size_t)(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = (int8_t)tmcq_ref_decode(payload, (size_t)tr * tiles_per_row + tc, n, t);
}

static size_t tmcq_payload_bytes(const int rows, const int cols, const int n)
{
	return ((size_t)rows * (cols / 256) * 16 * n + 7) / 8;
}

static size_t tmcq_scale_offset(const int rows, const int cols, const int n)
{
	return (tmcq_payload_bytes(rows, cols, n) + 127) & ~(size_t)127;
}

// Round to nearest even, so the expected fp outputs do not depend on the host conversion helpers.
static uint16_t tmcq_float_to_half_rne(const float value)
{
	uint32_t x;
	memcpy(&x, &value, sizeof(x));
	const uint32_t sign = (x >> 16) & 0x8000;
	uint32_t ax = x & 0x7fffffffu;
	if (ax >= 0x7f800000u)
		return (uint16_t)(sign | 0x7c00 | (ax > 0x7f800000u ? 0x200 : 0));
	if (ax >= 0x477ff000u)
		return (uint16_t)(sign | 0x7c00);
	if (ax < 0x38800000u)
	{
		float af;
		memcpy(&af, &ax, sizeof(af));
		return (uint16_t)(sign | (uint32_t)lrintf(af * 16777216.0f));
	}
	ax += 0xfffu + ((ax >> 13) & 1);
	ax -= 0x38000000u;
	return (uint16_t)(sign | (ax >> 13));
}

static uint16_t tmcq_float_to_bfloat_rne(const float value)
{
	uint32_t x;
	memcpy(&x, &value, sizeof(x));
	return (uint16_t)((x + 0x7fffu + ((x >> 16) & 1)) >> 16);
}

static float tmcq_half_to_float(const uint16_t h)
{
	float f;
	ccv_half_precision_to_float(&h, &f, 1);
	return f;
}

static float tmcq_bfloat_to_float(const uint16_t h)
{
	const uint32_t u = (uint32_t)h << 16;
	float f;
	memcpy(&f, &u, sizeof(f));
	return f;
}

static float tmcq_load_value(const int datatype, const void* const data, const size_t i)
{
	if (datatype == CCV_16F)
		return tmcq_half_to_float(((const uint16_t*)data)[i]);
	if (datatype == CCV_16BF)
		return tmcq_bfloat_to_float(((const uint16_t*)data)[i]);
	return ((const float*)data)[i];
}

// Stores a value already representable in the datatype (or rounds to nearest even).
static void tmcq_store_value(const int datatype, void* const data, const size_t i, const float value)
{
	if (datatype == CCV_16F)
		((uint16_t*)data)[i] = tmcq_float_to_half_rne(value);
	else if (datatype == CCV_16BF)
		((uint16_t*)data)[i] = tmcq_float_to_bfloat_rne(value);
	else
		((float*)data)[i] = value;
}

// A random normal scale in [2^-10, 2) with random mantissa bits of the datatype, so q8 * scale
// stays normal and exercises the output rounding of 16F / 16BF.
static float tmcq_random_scale(const int datatype, uint32_t* const state)
{
	const uint32_t r = tmcq_rng_next(state);
	const int exponent = -10 + (int)(r % 11);
	const float mantissa = 1.0f + (float)((r >> 8) & 0xffff) / 65536.0f;
	const float value = ldexpf(mantissa, exponent);
	if (datatype == CCV_16F)
		return tmcq_half_to_float(tmcq_float_to_half_rne(value));
	if (datatype == CCV_16BF)
		return tmcq_bfloat_to_float(tmcq_float_to_bfloat_rne(value));
	return value;
}

// Small dyadic scales: q8 * scale is exact in every datatype, so GEMM references only differ by
// accumulation.
static float tmcq_exact_scale(uint32_t* const state)
{
	return (float)(1 + tmcq_rng_next(state) % 5) / 4096.0f;
}

// Fills a host TMCQ tensor: random payload bytes, zero padding up to the scales, then the scales.
static void tmcq_fill_weights(ccv_nnc_tensor_t* const q, const int datatype, const int rows, const int cols, const int n, const int exact_scales, uint32_t seed, float* const scales)
{
	const size_t payload = tmcq_payload_bytes(rows, cols, n);
	const size_t scale_offset = tmcq_scale_offset(rows, cols, n);
	uint8_t* const data = q->data.u8;
	size_t i;
	for (i = 0; i < payload; i++)
		data[i] = (uint8_t)(tmcq_rng_next(&seed) >> 24);
	for (; i < scale_offset; i++)
		data[i] = 0;
	int r;
	for (r = 0; r < rows; r++)
	{
		scales[r] = exact_scales ? tmcq_exact_scale(&seed) : tmcq_random_scale(datatype, &seed);
		tmcq_store_value(datatype, data + scale_offset, r, scales[r]);
	}
}

static ccv_nnc_tensor_param_t tmcq_params(const int type, const int datatype, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { rows, cols, 0 },
	};
	return params;
}

static const int tmcq_formats[] = {
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N16,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N18,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N20,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N22,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N24,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N26,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N28,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N30,
	CCV_NNC_QX_8I_ROWWISE_TMCQ_N32,
};

static const int tmcq_datatypes[] = { CCV_32F, CCV_16F, CCV_16BF };

TEST_CASE("tmcq reference decode fixed vectors")
{
	// CPU-only checks of the reference itself (no GPU needed). All-zero codewords: cw = 0 selects pair
	// 0 of the frozen table, (54, -52): even weights take 54, odd weights -52.
	uint8_t zeros[128] = {0}; // tile 1 at n = 32 is bytes 64 .. 127
	REQUIRE_EQ(tmcq_ref_decode(zeros, 0, 16, 0), 54, "zero payload decodes even weights to pair 0's first code");
	REQUIRE_EQ(tmcq_ref_decode(zeros, 1, 32, 255), -52, "zero payload decodes odd weights to pair 0's second code");
	// All-ones windows give cw = 0x7fff, pair (0x7fff * 0x83DCD12D) >> 23, recomputed here.
	const uint32_t k = (0x7fffu * 0x83DCD12Du) >> 23;
	uint8_t ones[64];
	memset(ones, 0xff, sizeof(ones));
	REQUIRE_EQ(tmcq_ref_decode(ones, 0, 22, 16), ccv_nnc_tmcq_p2_table[2 * k], "all-ones payload decodes an even weight to pair k's first code");
	REQUIRE_EQ(tmcq_ref_decode(ones, 0, 22, 17), ccv_nnc_tmcq_p2_table[2 * k + 1], "all-ones payload decodes an odd weight to pair k's second code");
	// Weights 0 and 1 share weight 1's window. With n = 16 it ends at E_1 = 2, so it takes bits 243..255
	// of the tile's end, then bits 0 and 1: bit 1 is the codeword's top bit (1 << 14), bit 0 is 1 << 13.
	uint8_t second[32] = {0};
	second[0] = 2;
	REQUIRE_EQ(tmcq_ref_decode(second, 0, 16, 0), tmcq_ref_q8(1u << 14, 0), "the newest bit is the codeword's top bit");
	REQUIRE_EQ(tmcq_ref_decode(second, 0, 16, 1), tmcq_ref_q8(1u << 14, 1), "the odd weight reads the same window");
	uint8_t first[32] = {0};
	first[0] = 1;
	REQUIRE_EQ(tmcq_ref_decode(first, 0, 16, 0), tmcq_ref_q8(1u << 13, 0), "bit 0 is the window's second newest bit");
	// Only the last bit of the tile (bit 255) set: the shared window of weights 0 and 1 holds it at
	// position 12 (tail-biting wrap inside the tile).
	uint8_t last[32] = {0};
	last[31] = 0x80;
	REQUIRE_EQ(tmcq_ref_decode(last, 0, 16, 0), tmcq_ref_q8(1u << 12, 0), "the window wraps inside the tile");
	REQUIRE_EQ(tmcq_ref_decode(last, 0, 16, 1), tmcq_ref_q8(1u << 12, 1), "the wrapped window is shared by the pair");
}

// The GPU tests call MPS-only entry points (the int8 decode wrapper and the selected decode).
#ifdef HAVE_MPS

TEST_CASE("mps tmcq int8 decode matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 48;
	const int cols = 768;
	int f, d;
	for (f = 0; f < (int)(sizeof(tmcq_formats) / sizeof(tmcq_formats[0])); f++)
		for (d = 0; d < (int)(sizeof(tmcq_datatypes) / sizeof(tmcq_datatypes[0])); d++)
		{
			const int format = tmcq_formats[f];
			const int datatype = tmcq_datatypes[d];
			const int n = tmcq_rate_of(format);
			const size_t scale_size = CCV_GET_DATA_TYPE_SIZE(datatype);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, rows, cols), format), 0);
			float* const scales = (float*)ccmalloc(sizeof(float) * rows);
			tmcq_fill_weights(hq, datatype, rows, cols, n, 0, 0x1234u + f * 31 + d, scales);
			const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
			const size_t out_bytes = out_scale_offset + rows * scale_size;
			ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
			ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
			ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, datatype, ccv_nnc_tensor_data_size_without_padding(hq->info), cols, format, gout->data.u8, (size_t)rows * cols, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
			int8_t* const expected = (int8_t*)ccmalloc((size_t)rows * cols);
			tmcq_ref_decode_matrix(hq->data.u8, rows, cols, n, expected);
			size_t mismatches = 0, first = (size_t)-1;
			size_t i;
			for (i = 0; i < (size_t)rows * cols; i++)
				if ((int8_t)hout->data.u8[i] != expected[i])
				{
					if (first == (size_t)-1)
						first = i;
					mismatches++;
				}
			REQUIRE_EQ(mismatches, 0, "TMCQ n=%d datatype=%x int8 decode: %zu mismatches, first at row %zu col %zu", n, datatype, mismatches, first / cols, first % cols);
			REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + tmcq_scale_offset(rows, cols, n), rows * scale_size), 0, "TMCQ n=%d datatype=%x row scales are copied", n, datatype);
			ccfree(expected);
			ccfree(scales);
			ccv_nnc_tensor_free(hout);
			ccv_nnc_tensor_free(gout);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
		}
}

TEST_CASE("mps tmcq int8 decode of 7168-wide rows matches the reference bit for bit at every rate")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 32;
	const int cols = 7168;
	const int datatype = CCV_16F;
	int f;
	for (f = 0; f < (int)(sizeof(tmcq_formats) / sizeof(tmcq_formats[0])); f++)
	{
		const int format = tmcq_formats[f];
		const int n = tmcq_rate_of(format);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, rows, cols), format), 0);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		tmcq_fill_weights(hq, datatype, rows, cols, n, 0, 0x7168u + f, scales);
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, datatype, ccv_nnc_tensor_data_size_without_padding(hq->info), cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		int8_t* const expected = (int8_t*)ccmalloc((size_t)rows * cols);
		tmcq_ref_decode_matrix(hq->data.u8, rows, cols, n, expected);
		size_t mismatches = 0, i;
		for (i = 0; i < (size_t)rows * cols; i++)
			mismatches += (int8_t)hout->data.u8[i] != expected[i];
		REQUIRE_EQ(mismatches, 0, "TMCQ n=%d 7168-wide int8 decode: %zu mismatches", n, mismatches);
		REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + tmcq_scale_offset(rows, cols, n), rows * 2), 0, "TMCQ n=%d row scales are copied", n);
		ccfree(expected);
		ccfree(scales);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

TEST_CASE("mps tmcq fp decode matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int check_cpu = getenv("TMCQ_CPU_REFERENCE") && atoi(getenv("TMCQ_CPU_REFERENCE")) != 0;
	const int rows = 32;
	const int cols = 512;
	int f, d;
	for (f = 0; f < (int)(sizeof(tmcq_formats) / sizeof(tmcq_formats[0])); f++)
		for (d = 0; d < (int)(sizeof(tmcq_datatypes) / sizeof(tmcq_datatypes[0])); d++)
		{
			const int format = tmcq_formats[f];
			const int datatype = tmcq_datatypes[d];
			const int n = tmcq_rate_of(format);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, rows, cols), 0);
			ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), 0);
			ccv_nnc_tensor_t* const href = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), 0);
			float* const scales = (float*)ccmalloc(sizeof(float) * rows);
			tmcq_fill_weights(hq, datatype, rows, cols, n, 0, 0xbeefu + f * 131 + d, scales);
			const size_t qsize = ccv_nnc_tensor_data_size_without_padding(hq->info);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
			ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatype, CCV_TENSOR_GPU_MEMORY, qsize, cols, format, gout->data.u8, (size_t)rows * cols);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
			int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
			tmcq_ref_decode_matrix(hq->data.u8, rows, cols, n, q8);
			const size_t element_size = CCV_GET_DATA_TYPE_SIZE(datatype);
			size_t mismatches = 0, first = (size_t)-1;
			size_t i;
			for (i = 0; i < (size_t)rows * cols; i++)
			{
				// q8 * scale is exact in float for 16F / 16BF scales and rounds once for 32F, like the
				// CPU decode's double product rounded to the output type.
				const float value = (float)((double)scales[i / cols] * q8[i]);
				uint8_t expected[4];
				tmcq_store_value(datatype, expected, 0, value);
				if (memcmp(expected, hout->data.u8 + i * element_size, element_size) != 0)
				{
					if (first == (size_t)-1)
						first = i;
					mismatches++;
				}
			}
			REQUIRE_EQ(mismatches, 0, "TMCQ n=%d datatype=%x fp decode: %zu mismatches, first at row %zu col %zu (GPU %g, expected %g)", n, datatype, mismatches, first == (size_t)-1 ? 0 : first / cols, first == (size_t)-1 ? 0 : first % cols, first == (size_t)-1 ? 0.0 : (double)tmcq_load_value(datatype, hout->data.u8, first), first == (size_t)-1 ? 0.0 : (double)scales[first / cols] * q8[first]);
			if (check_cpu)
			{
				ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, qsize, cols, format, href->data.u8, (size_t)rows * cols);
				if (datatype == CCV_32F)
				{
					REQUIRE_EQ(memcmp(href->data.u8, hout->data.u8, (size_t)rows * cols * element_size), 0, "TMCQ n=%d datatype=%x fp decode matches the CPU decode bit for bit", n, datatype);
				} else {
					// The shared CPU writer (_ccv_nnc_8i_rowwise_packed_write_value, used by every packed
					// format) narrows through ccv_float_to_half_precision / ccv_float_to_bfloat, which round
					// toward zero, while the GPU rounds to nearest like the spec reference above. So for 16F /
					// 16BF the CPU decode may sit one unit in the last place below the GPU in magnitude.
					size_t far = 0, below = 0, j;
					for (j = 0; j < (size_t)rows * cols; j++)
					{
						const uint16_t c = ((const uint16_t*)href->data.u8)[j];
						const uint16_t g = ((const uint16_t*)hout->data.u8)[j];
						if (c == g)
							continue;
						if ((c & 0x8000) == (g & 0x8000) && (uint16_t)(g & 0x7fff) == (uint16_t)((c & 0x7fff) + 1))
							below++;
						else
							far++;
					}
					REQUIRE_EQ(far, 0, "TMCQ n=%d datatype=%x fp decode is within the CPU decode's round-toward-zero ulp (%zu elements one ulp below)", n, datatype, below);
				}
			}
			ccfree(q8);
			ccfree(scales);
			ccv_nnc_tensor_free(href);
			ccv_nnc_tensor_free(hout);
			ccv_nnc_tensor_free(gout);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
		}
}

TEST_CASE("mps tmcq selected expert int8 decode matches the reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int expert_count = 3;
	const int rows_per_expert = 32;
	const int cols = 512;
	const int rows = expert_count * rows_per_expert;
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N22;
	const int datatype = CCV_16F;
	const int n = tmcq_rate_of(format);
	const size_t scale_size = 2;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, rows, cols), format), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, rows, cols), format), 0);
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	tmcq_fill_weights(hq, datatype, rows, cols, n, 0, 0x5eedu, scales);
	// Bins route to experts 2 and 1; expert 0's bin is empty, so its rows must stay untouched.
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	ccv_nnc_tensor_t* const hcounts = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3), 0);
	hindices->data.i32[0] = 2;
	hindices->data.i32[1] = 0;
	hindices->data.i32[2] = 1;
	hcounts->data.i32[0] = 3;
	hcounts->data.i32[1] = 0;
	hcounts->data.i32[2] = 5;
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	ccv_nnc_tensor_t* const gcounts = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 3), 0);
	const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
	const size_t out_bytes = out_scale_offset + rows * scale_size;
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	memset(hout->data.u8, 0x5a, out_bytes);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hindices, hcounts, hout), TENSOR_LIST(gq, gindices, gcounts, gout), 0);
	const ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params = {
		.data_type = 16, // MTL::DataTypeHalf
		.format = (uint32_t)format,
		.row_length = cols,
		.rows_per_expert = rows_per_expert,
		.expert_count = expert_count,
		.bincount = 3,
	};
	ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
	mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(params));
	mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[6] = {
		(mtl_buffer_t*)gq->data.u8,
		(mtl_buffer_t*)gindices->data.u8,
		(mtl_buffer_t*)gcounts->data.u8,
		(mtl_buffer_t*)gout->data.u8,
		scratch,
		NULL,
	};
	size_t tensor_offsets[5] = {
		(size_t)gq->dataof,
		(size_t)gindices->dataof,
		(size_t)gcounts->dataof,
		(size_t)gout->dataof,
		0,
	};
	ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(context, params, command_batch, tensors, tensor_offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
	int8_t* const expected = (int8_t*)ccmalloc((size_t)rows * cols);
	tmcq_ref_decode_matrix(hq->data.u8, rows, cols, n, expected);
	int e;
	for (e = 0; e < expert_count; e++)
	{
		const size_t base = (size_t)e * rows_per_expert * cols;
		size_t mismatches = 0;
		size_t i;
		for (i = 0; i < (size_t)rows_per_expert * cols; i++)
		{
			const uint8_t want = e == 0 ? 0x5a : (uint8_t)expected[base + i];
			if (hout->data.u8[base + i] != want)
				mismatches++;
		}
		REQUIRE_EQ(mismatches, 0, "expert %d: %zu mismatches (expert 0 is not selected and must keep the sentinel)", e, mismatches);
		const uint8_t* const scale_out = hout->data.u8 + out_scale_offset + (size_t)e * rows_per_expert * scale_size;
		const uint8_t* const scale_in = hq->data.u8 + tmcq_scale_offset(rows, cols, n) + (size_t)e * rows_per_expert * scale_size;
		if (e == 0)
		{
			for (i = 0; i < rows_per_expert * scale_size; i++)
				REQUIRE_EQ(scale_out[i], 0x5a, "expert 0 scales stay untouched");
		} else
			REQUIRE_EQ(memcmp(scale_out, scale_in, rows_per_expert * scale_size), 0, "expert %d scales are copied", e);
	}
	ccfree(expected);
	ccfree(scales);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(hcounts);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
}

static float tmcq_activation_value(const int row, const int k)
{
	return (float)(((row * 37 + k * 11) % 33) - 16) / 128.0f;
}

static float tmcq_bias_value(const int col)
{
	return (float)((col * 7) % 13 - 6) / 64.0f;
}

// Runs B = A W^T + bias through the MPS GEMM command with TMCQ weights [n_dim, k_dim], and the same
// GEMM with the reference-decoded dense weights, and compares both against a CPU double reference.
// max_rel uses the mpsblas metric |x - y| / max(1, |x|, |y|).
static int tmcq_gemm_compare(const int datatype, const int format, const int use_bias, const int m_dim, const int n_dim, const int k_dim, double* const max_rel_cpu, double* const max_rel_dense)
{
	const int n = tmcq_rate_of(format);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, n_dim, k_dim), format), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, n_dim, k_dim), format), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, n_dim, k_dim), 0);
	ccv_nnc_tensor_t* const gw = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, n_dim, k_dim), 0);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, m_dim, k_dim), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, m_dim, k_dim), 0);
	ccv_nnc_tensor_param_t bias_params = tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, n_dim, 0);
	bias_params.dim[1] = 0;
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	bias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const gbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const gbq = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const gbd = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_GPU_MEMORY, datatype, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const hbq = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, m_dim, n_dim), 0);
	ccv_nnc_tensor_t* const hbd = ccv_nnc_tensor_new(0, tmcq_params(CCV_TENSOR_CPU_MEMORY, datatype, m_dim, n_dim), 0);
	float* const scales = (float*)ccmalloc(sizeof(float) * n_dim);
	tmcq_fill_weights(hq, datatype, n_dim, k_dim, n, 1, 0xc0ffeeu + format * 7 + m_dim, scales);
	int8_t* const q8 = (int8_t*)ccmalloc((size_t)n_dim * k_dim);
	tmcq_ref_decode_matrix(hq->data.u8, n_dim, k_dim, n, q8);
	int i, j, k;
	for (i = 0; i < n_dim; i++)
		for (k = 0; k < k_dim; k++)
			tmcq_store_value(datatype, hw->data.u8, (size_t)i * k_dim + k, scales[i] * q8[(size_t)i * k_dim + k]);
	for (i = 0; i < m_dim; i++)
		for (k = 0; k < k_dim; k++)
			tmcq_store_value(datatype, ha->data.u8, (size_t)i * k_dim + k, tmcq_activation_value(i, k));
	if (use_bias)
		for (j = 0; j < n_dim; j++)
			tmcq_store_value(datatype, hbias->data.u8, j, tmcq_bias_value(j));
	if (use_bias)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hw, ha, hbias), TENSOR_LIST(gq, gw, ga, gbias), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gbq), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gw, gbias), TENSOR_LIST(gbd), 0);
	} else {
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hw, ha), TENSOR_LIST(gq, gw, ga), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gbq), 0);
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gw), TENSOR_LIST(gbd), 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gbq, gbd), TENSOR_LIST(hbq, hbd), 0);
	double cpu = 0, dense = 0;
	for (i = 0; i < m_dim; i++)
		for (j = 0; j < n_dim; j++)
		{
			double ref = use_bias ? tmcq_load_value(datatype, hbias->data.u8, j) : 0;
			for (k = 0; k < k_dim; k++)
				ref += (double)tmcq_load_value(datatype, ha->data.u8, (size_t)i * k_dim + k) * (double)scales[j] * q8[(size_t)j * k_dim + k];
			const double actual = tmcq_load_value(datatype, hbq->data.u8, (size_t)i * n_dim + j);
			const double dense_value = tmcq_load_value(datatype, hbd->data.u8, (size_t)i * n_dim + j);
			cpu = ccv_max(cpu, fabs(actual - ref) / ccv_max(1.0, ccv_max(fabs(actual), fabs(ref))));
			dense = ccv_max(dense, fabs(actual - dense_value) / ccv_max(1.0, ccv_max(fabs(actual), fabs(dense_value))));
		}
	*max_rel_cpu = cpu;
	*max_rel_dense = dense;
	ccfree(q8);
	ccfree(scales);
	ccv_nnc_tensor_free(hbd);
	ccv_nnc_tensor_free(hbq);
	ccv_nnc_tensor_free(gbd);
	ccv_nnc_tensor_free(gbq);
	if (gbias)
		ccv_nnc_tensor_free(gbias);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(gw);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
	return 0;
}

static double tmcq_tolerance(const int datatype)
{
	return datatype == CCV_32F ? 1e-4 : (datatype == CCV_16F ? 4e-3 : 2e-2);
}

TEST_CASE("mps tmcq gemv for 1 to 3 rows matches the reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// N = 1040 is a multiple of 16 but not of 256: only TMCQ takes the GEMV kernel at that width.
	const int formats[] = { CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 };
	int f, d, m, bias;
	for (f = 0; f < 3; f++)
		for (d = 0; d < (int)(sizeof(tmcq_datatypes) / sizeof(tmcq_datatypes[0])); d++)
			for (m = 1; m <= 3; m++)
				for (bias = 0; bias < 2; bias++)
				{
					double max_rel_cpu = 0, max_rel_dense = 0;
					tmcq_gemm_compare(tmcq_datatypes[d], formats[f], bias, m, 1040, 2048, &max_rel_cpu, &max_rel_dense);
					REQUIRE(max_rel_cpu < tmcq_tolerance(tmcq_datatypes[d]), "TMCQ n=%d datatype=%x M=%d bias=%d GEMV vs CPU max_rel=%g", tmcq_rate_of(formats[f]), tmcq_datatypes[d], m, bias, max_rel_cpu);
					REQUIRE(max_rel_dense < tmcq_tolerance(tmcq_datatypes[d]), "TMCQ n=%d datatype=%x M=%d bias=%d GEMV vs dense GPU max_rel=%g", tmcq_rate_of(formats[f]), tmcq_datatypes[d], m, bias, max_rel_dense);
				}
}

TEST_CASE("mps tmcq gemv covers K from 256 to 7168 and N from 16 to 4096")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// The GEMV splits a tile row's K / 16 tiles over 2 streams in each of 4 SIMD groups. TMCQ GEMM
	// requires K % 256 == 0, so K / 16 is a multiple of 16 and every stream runs the same number of
	// steps: K = 256 is the shortest (2 steps per stream), K = 7168 the DeepSeek hidden width.
	const int formats[] = { CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 };
	const int datatypes[] = { CCV_16F, CCV_32F };
	const int ks[] = { 256, 512, 4096, 7168 };
	const int ns[] = { 16, 1040, 4096 };
	int f, d, ki, ni, m;
	for (f = 0; f < 3; f++)
		for (d = 0; d < 2; d++)
			for (ki = 0; ki < 4; ki++)
				for (ni = 0; ni < 3; ni++)
					for (m = 1; m <= 3; m++)
					{
						const int bias = (ki + ni + m) & 1;
						double max_rel_cpu = 0, max_rel_dense = 0;
						tmcq_gemm_compare(datatypes[d], formats[f], bias, m, ns[ni], ks[ki], &max_rel_cpu, &max_rel_dense);
						REQUIRE(max_rel_cpu < tmcq_tolerance(datatypes[d]), "TMCQ n=%d datatype=%x M=%d N=%d K=%d bias=%d GEMV vs CPU max_rel=%g", tmcq_rate_of(formats[f]), datatypes[d], m, ns[ni], ks[ki], bias, max_rel_cpu);
						REQUIRE(max_rel_dense < tmcq_tolerance(datatypes[d]), "TMCQ n=%d datatype=%x M=%d N=%d K=%d bias=%d GEMV vs dense GPU max_rel=%g", tmcq_rate_of(formats[f]), datatypes[d], m, ns[ni], ks[ki], bias, max_rel_dense);
					}
	// Every rate at the widest K, so each n's window offsets k_j run across many tiles per row.
	int format;
	for (format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N16; format <= CCV_NNC_QX_8I_ROWWISE_TMCQ_N32; format++)
		for (m = 1; m <= 3; m++)
		{
			double max_rel_cpu = 0, max_rel_dense = 0;
			tmcq_gemm_compare(CCV_16F, format, m == 2, m, 1040, 7168, &max_rel_cpu, &max_rel_dense);
			REQUIRE(max_rel_cpu < tmcq_tolerance(CCV_16F), "TMCQ n=%d M=%d K=7168 GEMV vs CPU max_rel=%g", tmcq_rate_of(format), m, max_rel_cpu);
			REQUIRE(max_rel_dense < tmcq_tolerance(CCV_16F), "TMCQ n=%d M=%d K=7168 GEMV vs dense GPU max_rel=%g", tmcq_rate_of(format), m, max_rel_dense);
		}
}

TEST_CASE("mps tmcq gemm prefill covers K 256 and 7168 with partial row blocks")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	// M = 16 and 64 fill whole row blocks; M = 65 leaves one row in the last block. Same three routing
	// passes as the prefill case above.
	const int ms[] = { 16, 64, 65 };
	const int ks[] = { 256, 7168 };
	const int formats[] = { CCV_NNC_QX_8I_ROWWISE_TMCQ_N16, CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 };
	int pass, f, ki, mi;
	for (pass = 0; pass < 3; pass++)
	{
		if (pass >= 1)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
		if (pass >= 2)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		for (f = 0; f < 3; f++)
			for (ki = 0; ki < 2; ki++)
				for (mi = 0; mi < 3; mi++)
				{
					double max_rel_cpu = 0, max_rel_dense = 0;
					tmcq_gemm_compare(CCV_16F, formats[f], mi & 1, ms[mi], 528, ks[ki], &max_rel_cpu, &max_rel_dense);
					// Passes 0 and 1 run the int8 scaled GEMM, which quantizes each activation row to int8:
					// that error grows like sqrt(K) against the max(1, |x|) metric (measured about 6e-4 at
					// K = 256 and 4e-3 at K = 7168), so the bound is the K = 1024 case's 2e-3 scaled by
					// sqrt(K / 1024). Pass 2 decodes to fp16 and must equal the dense GEMM exactly.
					const double dense_bound = pass < 2 ? 2e-3 * ccv_max(1.0, sqrt(ks[ki] / 1024.0)) : 0;
					REQUIRE(max_rel_dense <= dense_bound, "TMCQ pass=%d n=%d M=%d K=%d GEMM vs dense GPU max_rel=%g bound=%g", pass, tmcq_rate_of(formats[f]), ms[mi], ks[ki], max_rel_dense, dense_bound);
					REQUIRE(max_rel_cpu < 1e-2, "TMCQ pass=%d n=%d M=%d K=%d GEMM vs CPU max_rel=%g", pass, tmcq_rate_of(formats[f]), ms[mi], ks[ki], max_rel_cpu);
				}
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
}

TEST_CASE("mps tmcq gemm decodes then multiplies for prefill rows")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	// Pass 0: default routing (ANE, ANE + GPU split, NA or Int8MatMul scaled GEMM). Pass 1: no ANE.
	// Pass 2: no ANE and no neural accelerators (Int8MatMul, or the fp decode fallback).
	int pass, d, mi;
	const int ms[] = { 64, 512 };
	const int datatypes[] = { CCV_16F, CCV_32F };
	for (pass = 0; pass < 3; pass++)
	{
		if (pass >= 1)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
		if (pass >= 2)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		for (d = 0; d < 2; d++)
			for (mi = 0; mi < 2; mi++)
			{
				double max_rel_cpu = 0, max_rel_dense = 0;
				tmcq_gemm_compare(datatypes[d], CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, mi, ms[mi], 528, 1024, &max_rel_cpu, &max_rel_dense);
				REQUIRE(max_rel_dense < 2e-3, "TMCQ pass=%d datatype=%x M=%d GEMM vs dense GPU max_rel=%g", pass, datatypes[d], ms[mi], max_rel_dense);
				REQUIRE(max_rel_cpu < 1e-2, "TMCQ pass=%d datatype=%x M=%d GEMM vs CPU max_rel=%g", pass, datatypes[d], ms[mi], max_rel_cpu);
			}
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
}

TEST_CASE("mps tmcq index select is rejected")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS));
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N22;
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_CPU_MEMORY, CCV_16F, 32, 256), format), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(tmcq_params(CCV_TENSOR_GPU_MEMORY, CCV_16F, 32, 256), format), 0);
	float scales[32];
	tmcq_fill_weights(hq, CCV_16F, 32, 256, tmcq_rate_of(format), 1, 7, scales);
	ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 2), 0);
	hindices->data.i32[0] = 3;
	hindices->data.i32[1] = 17;
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 256), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hindices), TENSOR_LIST(gq, gindices), 0);
	const int status = ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0);
	REQUIRE_EQ(status, CCV_NNC_EXEC_INVALID, "TMCQ tables do not support IndexSelect");
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(hindices);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
}

#endif

#include "case_main.h"
