#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <nnc/ccv_nnc_codebook.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nnc/ccv_nnc_trellis_builtin_table.inc>
#include "trellis_table_alt.inc"
#include "3rdparty/sqlite3/sqlite3.h"

// FTQ (CCV_NNC_QX_8I_ROWWISE_FTQ_N16). The reference below lays tiles out by adding up tile lengths and decodes a
// tile from the spec (128 pairs, pair j reads the 16-bit LSB-first window ending at floor((j + 1) T / 128), wrapping
// inside the tile), independently of the codec and the kernels. Any bit string is a valid code, so codes are random
// bytes.

TEST_SETUP()
{
	ccv_nnc_init();
}

static uint32_t ftq_rng_next(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

static ccv_nnc_tensor_param_t ftq_params(const int type, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = CCV_16F,
		.dim = { rows, cols, 0 },
	};
	return params;
}

static int ftq_format(const int n)
{
	return CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (n - 8) / 2;
}

// Start bit and length of every tile (tile row major), by summing lengths: every tile has high (16 n) bits.
static void ftq_ref_layout(const int rows, const int cols, const uint32_t high, const uint64_t drop, uint64_t* const starts, uint32_t* const lengths)
{
	(void)drop;
	const uint64_t count = (uint64_t)rows / 16 * (cols / 16);
	uint64_t bit = 0, t;
	for (t = 0; t < count; t++)
	{
		starts[t] = bit;
		lengths[t] = high;
		bit += high;
	}
}

static void ftq_ref_decode_matrix(const uint8_t* const code, const int rows, const int cols, const uint32_t high, const uint64_t drop, const int8_t* const table, const int log2_entries, int8_t* const q8)
{
	const uint64_t count = (uint64_t)rows * cols / 256;
	uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
	uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
	ftq_ref_layout(rows, cols, high, drop, starts, lengths);
	const int strips = cols / 16;
	uint64_t t;
	for (t = 0; t < count; t++)
	{
		const uint64_t start = starts[t];
		const uint64_t T = lengths[t];
		int j, b;
		for (j = 0; j < 128; j++)
		{
			const uint64_t end = (uint64_t)(j + 1) * T / 128;
			uint32_t state = 0;
			for (b = 0; b < 16; b++)
			{
				const uint64_t p = start + (end + T - 16 + b) % T;
				state |= (uint32_t)((code[p >> 3] >> (p & 7)) & 1) << b;
			}
			const uint32_t k = (state * 0x83DCD12Du) >> (32 - log2_entries);
			const int row = (int)(t / strips) * 16 + j / 8;
			const int col = (int)(t % strips) * 16 + 2 * (j % 8);
			q8[(size_t)row * cols + col] = table[2 * k];
			q8[(size_t)row * cols + col + 1] = table[2 * k + 1];
		}
	}
	ccfree(lengths);
	ccfree(starts);
}

// A CCVFTQSR blob: header ("CCVFTQSR", rows, cols, high_bits, drop_count, table, table_crc), random code bits,
// random FP16 scales.
static void ftq_fill(uint8_t* const data, const size_t size, const int format, const int rows, const int cols, const uint32_t table, const uint32_t table_crc, uint32_t seed)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	memset(data, 0, size);
	memcpy(data, "CCVFTQSR", 8);
	const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, schedule.high_bits, (uint32_t)schedule.drop_count, table, table_crc };
	memcpy(data + 8, header, sizeof(header));
	size_t i;
	for (i = 0; i < (schedule.code_bits + 7) / 8; i++)
		data[32 + i] = (uint8_t)(ftq_rng_next(&seed) >> 24);
	if (schedule.code_bits % 8)
		data[32 + schedule.code_bits / 8] &= (uint8_t)((1u << (schedule.code_bits % 8)) - 1);
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	int r;
	for (r = 0; r < rows; r++)
		scales[r] = (float)((ftq_rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
	ccv_float_to_half_precision(scales, (uint16_t*)(data + schedule.scale_offset), rows);
	ccfree(scales);
}

static const uint32_t ftq_builtin_table = 9u << 8; // id 0 (the built-in table), 2^9 pairs

// (rows, cols, n): small grids (32 x 256 n16, 272 x 256 n16, a single tile row 16 x 256 n16 and 16 x 4352 n32),
// odd word counts per tile (n18, n = 2 * 9 words), n24, the real 0.8B / 4B projection sizes at n20, and rates 36 - 48.
static const int ftq_cases[][3] = {
	{ 32, 256, 16 }, { 272, 256, 16 }, { 16, 256, 16 }, { 16, 4352, 32 }, { 48, 768, 18 }, { 64, 1024, 24 }, { 528, 1024, 20 }, { 3584, 1024, 20 }, { 9216, 2560, 20 },
	// Rates above TMCQ's (to 3 bpw, high_bits to 768).
	{ 528, 1024, 36 }, { 64, 1024, 40 }, { 9216, 2560, 36 }, { 9216, 2560, 40 }, { 9216, 2560, 48 }, { 16, 4352, 48 },
};

TEST_CASE("ftq schedule has full word-aligned tiles and a header outside the TMCQ budget")
{
	int c;
	for (c = 0; c < (int)(sizeof(ftq_cases) / sizeof(ftq_cases[0])); c++)
	{
		const int rows = ftq_cases[c][0], cols = ftq_cases[c][1], n = ftq_cases[c][2];
		const int format = ftq_format(n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, (size_t)rows * cols, cols);
		const uint64_t tiles = (uint64_t)rows / 16 * (cols / 16);
		REQUIRE_EQ(schedule.high_bits, 16u * n, "case %d high_bits is 16 n", c);
		REQUIRE_EQ((int)schedule.drop_count, 0, "case %d drops no bit", c);
		REQUIRE_EQ(schedule.code_bits, tiles * 16 * n, "case %d code bits", c);
		REQUIRE_EQ(schedule.scale_offset, (32 + tiles * 2 * n + 127) & ~(uint64_t)127, "case %d scale offset", c);
		REQUIRE_EQ(size, (size_t)schedule.scale_offset + 2 * rows, "case %d size", c);
		if (n <= 32)
		{
			// The header is outside the TMCQ_N<n> budget: 32 bytes plus the padding to the next 128.
			const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N16 + (n - 16) / 2;
			const size_t tmcq_size = ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_16F, (size_t)rows * cols, cols);
			REQUIRE_EQ(size - tmcq_size, (size_t)(((32 + tiles * 2 * n + 127) & ~(uint64_t)127) - ((tiles * 2 * n + 127) & ~(uint64_t)127)), "case %d size is TMCQ plus the header", c);
		}
		const uint64_t count = schedule.tile_count;
		uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
		uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
		ftq_ref_layout(rows, cols, schedule.high_bits, schedule.drop_count, starts, lengths);
		REQUIRE_EQ(starts[count - 1] + lengths[count - 1], schedule.code_bits, "case %d tiles fill the code", c);
		uint64_t t, misaligned = 0;
		for (t = 0; t < count; t++)
			misaligned += starts[t] % 32 != 0;
		REQUIRE_EQ((int)misaligned, 0, "case %d every tile starts on a 32-bit word", c);
		ccfree(lengths);
		ccfree(starts);
	}
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_N20, CCV_32F, 16 * 256, 256), 0, "FTQ is FP16 only");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST, 32, "the static-rate FTQ block starts at 32");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_LAST, 95, "and ends at 95");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST, 96, "the variable-rate FTQ block starts at 96");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_LAST, 159, "and ends at 159");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_N16, 36, "FTQ codec ids are 32 + (n - 8) / 2");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_N48, 52, "FTQ codec ids are 32 + (n - 8) / 2");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST) && !ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_N16 - 1), "n8 .. n14 are not open");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_N48 + 1) && !ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_LAST), "n50 .. n134 are not open");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_tiled(CCV_NNC_QX_8I_ROWWISE_TMCQ_N32 + 1), "code 31 is empty");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_rate(CCV_NNC_QX_8I_ROWWISE_FTQ_N48), 48, "n48");
}

TEST_CASE("cpu ftq decode matches the reference with the built-in and a registered table")
{
	// A registered 1024-pair table (slot from the registry, header id 1, log2 10, its crc32).
	int8_t* const big = (int8_t*)ccmalloc(2048);
	uint32_t seed = 0xc0ffeeu;
	int i;
	for (i = 0; i < 2048; i++)
		big[i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
	const int slot = ccv_nnc_codebook_register(big, 1024);
	REQUIRE(slot >= 1, "the 1024-pair table registers");
	uint32_t big_crc;
	ccv_nnc_codebook_get(slot, 0, &big_crc);
	static const int cases[][3] = { { 32, 256, 16 }, { 272, 256, 16 }, { 48, 768, 18 }, { 64, 1024, 24 }, { 528, 1024, 20 }, { 528, 1024, 40 }, { 9216, 2560, 48 } };
	int c, which;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
		for (which = 0; which < 2; which++)
		{
			const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
			const int format = ftq_format(n) | (which ? (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT) : 0);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
			ftq_fill(hq->data.u8, size, format, rows, cols, which ? (1u | (10u << 8)) : ftq_builtin_table, which ? big_crc : 0, 0x1700u + c);
			const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
			int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
			ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, which ? big : ccv_nnc_trellis_builtin_table, which ? 10 : 9, q8);
			float* const scales = (float*)ccmalloc(sizeof(float) * rows);
			ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols);
			size_t k, mismatches = 0;
			for (k = 0; k < (size_t)rows * cols; k++)
			{
				const float v = (float)q8[k] * scales[k / cols];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				mismatches += out[k] != e;
			}
			REQUIRE_EQ((int)mismatches, 0, "FTQ %dx%d n%d %s table CPU decode", rows, cols, n, which ? "registered" : "built-in");
			ccfree(out);
			ccfree(scales);
			ccfree(q8);
			ccv_nnc_tensor_free(hq);
		}
	ccfree(big);
}

// One rejection case: a valid FTQ blob (528 x 1024 n20, a registered 1024-pair table unless table is 0), with
// `mutate` applied, must make the CPU decode return -1.
typedef void (*ftq_mutation_f)(uint8_t* const data, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const schedule, int* const format, const int slot);

static void ftq_bad_magic(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[7] = '6'; }
static void ftq_bad_rows(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[8] ^= 16; }
static void ftq_bad_cols(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[12] ^= 1; }
static void ftq_bad_high(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[16] ^= 1; }
static void ftq_bad_drop(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[20] ^= 1; }
static void ftq_bad_reserved(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[26] = 1; }
static void ftq_bad_log2(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[25] = 11; }
static void ftq_bad_crc(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { d[28] ^= 1; }
static void ftq_id_without_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f = ccv_nnc_codebook_with_slot(*f, 0); }
static void ftq_unregistered_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f = ccv_nnc_codebook_with_slot(*f, 250); }
static void ftq_id0_with_crc(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	*f = ccv_nnc_codebook_with_slot(*f, 0);
	const uint32_t t[2] = { 9u << 8, 1 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_id0_log2_10(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	*f = ccv_nnc_codebook_with_slot(*f, 0);
	const uint32_t t[2] = { 10u << 8, 0 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_id0_with_slot(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	const uint32_t t[2] = { 9u << 8, 0 };
	memcpy(d + 24, t, sizeof(t));
}
static void ftq_negative_scale(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	d[s->scale_offset + 1] |= 0x80;
}
static void ftq_infinite_scale(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot)
{
	d[s->scale_offset + 2] = 0x00;
	d[s->scale_offset + 3] = 0x7c;
}
static void ftq_hadamard(uint8_t* const d, const ccv_nnc_8i_rowwise_x_ftq_schedule_t* const s, int* const f, const int slot) { *f |= CCV_NNC_QX_8I_ROWWISE_HADAMARD_256; }

TEST_CASE("cpu ftq decode rejects every broken container, also without asserts")
{
	int8_t* const big = (int8_t*)ccmalloc(2048);
	uint32_t seed = 0xbadu;
	int i;
	for (i = 0; i < 2048; i++)
		big[i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
	const int slot = ccv_nnc_codebook_register(big, 1024);
	REQUIRE(slot >= 1, "the table registers");
	uint32_t crc;
	ccv_nnc_codebook_get(slot, 0, &crc);
	const int rows = 528, cols = 1024, n = 20;
	const int base = ftq_format(n) | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(base, rows, cols);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(base, CCV_16F, (size_t)rows * cols, cols);
	uint8_t* const blob = (uint8_t*)ccmalloc(size);
	uint8_t* const work = (uint8_t*)ccmalloc(size);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	ftq_fill(blob, size, base, rows, cols, 1u | (10u << 8), crc, 0x4242u);
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, base, out, (size_t)rows * cols), 0, "the valid container decodes");
	static const struct {
		const char* name;
		ftq_mutation_f mutate;
	} cases[] = {
		{ "magic", ftq_bad_magic }, { "rows", ftq_bad_rows }, { "cols", ftq_bad_cols }, { "high_bits", ftq_bad_high },
		{ "drop_count", ftq_bad_drop }, { "table bits 16 - 31", ftq_bad_reserved }, { "log2 entries", ftq_bad_log2 },
		{ "table crc", ftq_bad_crc }, { "a table id without a slot", ftq_id_without_slot }, { "an unregistered slot", ftq_unregistered_slot },
		{ "id 0 with a crc", ftq_id0_with_crc }, { "id 0 with log2 10", ftq_id0_log2_10 }, { "id 0 with a slot", ftq_id0_with_slot },
		{ "a negative scale", ftq_negative_scale }, { "an infinite scale", ftq_infinite_scale },
		{ "H256", ftq_hadamard },
	};
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		memcpy(work, blob, size);
		int format = base;
		cases[c].mutate(work, &schedule, &format, slot);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(work, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), -1, "FTQ with a broken %s is rejected", cases[c].name);
		// The load-time check runs the same container checks.
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(work, size, format, rows, cols), -1, "FTQ with a broken %s fails the load check", cases[c].name);
	}
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(blob, size, base, rows, cols), 0, "the valid container passes the load check");
	// The code is always whole bytes (16 n bits a tile) and the padding after it must be zero; the FP16-only
	// rule is checked on the same small container.
	{
		const int r2 = 16, c2 = 256, n2 = 16;
		const int f2 = ftq_format(n2);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t s2 = ccv_nnc_8i_rowwise_x_ftq_schedule(f2, r2, c2);
		const size_t z2 = ccv_nnc_8i_rowwise_x_data_size(f2, CCV_16F, (size_t)r2 * c2, c2);
		REQUIRE_EQ(s2.code_bits % 8, 0, "the code is whole bytes");
		REQUIRE(32 + s2.code_bits / 8 < s2.scale_offset, "padding follows the code");
		uint8_t* const b2 = (uint8_t*)ccmalloc(z2);
		ftq_fill(b2, z2, f2, r2, c2, ftq_builtin_table, 0, 7);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_16F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), 0, "the small container decodes");
		b2[32 + s2.code_bits / 8] = 1;
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_16F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), -1, "FTQ with non-zero padding is rejected");
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(b2, z2, f2, r2, c2), -1, "FTQ with non-zero padding fails the load check");
		b2[32 + s2.code_bits / 8] = 0;
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_load_check(b2, z2, f2, r2, c2), 0, "the small container passes the load check");
		// Not FP16 output and a short input are rejected too.
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(b2, CCV_32F, CCV_TENSOR_CPU_MEMORY, z2, c2, f2, out, (size_t)r2 * c2), -1, "FTQ decodes to FP16 only");
		ccfree(b2);
	}
	ccfree(out);
	ccfree(work);
	ccfree(blob);
	ccfree(big);
}

TEST_CASE("tmcq takes a registered codebook slot and other codecs reject one, also without asserts")
{
	int8_t* const table = (int8_t*)ccmalloc(1024);
	int i;
	for (i = 0; i < 1024; i++)
		table[i] = (int8_t)((i * 31) % 255 - 127);
	const int slot = ccv_nnc_codebook_register(table, 512);
	REQUIRE(slot >= 1, "the table registers");
	const int rows = 32, cols = 256;
	const int tmcq = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20;
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(tmcq, CCV_16F, (size_t)rows * cols, cols);
	REQUIRE(size > 0, "TMCQ has a size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_16F, (size_t)rows * cols, cols), size, "TMCQ with a slot has the same size");
	uint8_t* const blob = (uint8_t*)cccalloc(size, 1);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq, out, (size_t)rows * cols), 0, "TMCQ decodes");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), 0, "TMCQ with a slot decodes");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, tmcq | (250 << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), -1, "TMCQ with an unregistered slot is rejected");
	REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, CCV_NNC_QX_8I_ROWWISE_Q4_K | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), out, (size_t)rows * cols), -1, "a codebook slot on a non-trellis codec is rejected");
#ifdef HAVE_MPS
	if (ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		// Allocated with a valid format, then given an unregistered slot, or a slot on a codec that takes none.
		const int bad[2] = { tmcq | (250 << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_NNC_QX_8I_ROWWISE_Q4_K | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT) };
		const int good[2] = { tmcq | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT), CCV_NNC_QX_8I_ROWWISE_Q4_K };
		int b;
		for (b = 0; b < 2; b++)
		{
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), good[b]), 0);
			gq->info.reserved = bad[b];
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, rows), 0);
			REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "the GPU GEMM rejects case %d", b);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(gq);
		}
	}
#endif
	ccfree(out);
	ccfree(blob);
	ccfree(table);
}

#ifdef HAVE_MPS

TEST_CASE("mps ftq int8 decode matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int c;
	for (c = 0; c < (int)(sizeof(ftq_cases) / sizeof(ftq_cases[0])); c++)
	{
		const int rows = ftq_cases[c][0], cols = ftq_cases[c][1], n = ftq_cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		ftq_fill(hq->data.u8, size, format, rows, cols, ftq_builtin_table, 0, 0x5170u + c);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d codes", rows, cols, n);
		REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ %dx%d n%d scales", rows, cols, n);
		ccfree(q8);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

static float ftq_activation_value(const int row, const int k)
{
	return (float)(((row * 37 + k * 11) % 33) - 16) / 128.0f;
}

static float ftq_bias_value(const int col)
{
	return (float)((col * 7) % 13 - 6) / 64.0f;
}

// Resident 8I_ROWWISE weights from the reference codes and the blob's scales.
static ccv_nnc_tensor_t* ftq_rowwise_reference(const ccv_nnc_tensor_t* const hq, const int format, const int rows, const int cols)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
	const size_t r_scale = ((size_t)rows * cols + 127) & ~(size_t)127;
	memset(hr->data.u8, 0, r_scale);
	ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, (int8_t*)hr->data.u8);
	memcpy(hr->data.u8 + r_scale, hq->data.u8 + schedule.scale_offset, rows * 2);
	return hr;
}

TEST_CASE("mps ftq gemm from 4 rows equals the gemm on its decoded 8i rowwise weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	static const int cases[][3] = { { 528, 1024, 22 }, { 272, 256, 16 }, { 9216, 2560, 20 }, { 528, 1024, 36 }, { 9216, 2560, 40 }, { 9216, 2560, 48 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		ftq_fill(hq->data.u8, ccv_nnc_tensor_data_size_without_padding(hq->info), format, rows, cols, ftq_builtin_table, 0, 0x9a9au + c);
		ccv_nnc_tensor_t* const hr = ftq_rowwise_reference(hq, format, rows, cols);
		ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hr, hbias), TENSOR_LIST(gq, gr, gbias), 0);
		// Pass 0: default routing; pass 1: no ANE. From 4 rows FTQ decodes and takes the quantized scaled GEMM
		// like TMCQ; resident 8I_ROWWISE keeps its GEMV up to 8 rows, so its reference runs on at least 9 rows
		// (independent in the integer GEMM) and the first M rows are compared.
		static const int ms[] = { 4, 8, 9, 64 };
		int pass, mi, with_bias, i, k;
		for (pass = 0; pass < 2; pass++)
		{
			if (pass == 1)
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
			{
				const int m = ms[mi];
				const int mr = m < 9 ? 9 : m;
				ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
				for (i = 0; i < mr; i++)
					for (k = 0; k < cols; k++)
					{
						const float v = ftq_activation_value(i, k);
						ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
					}
				ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
				memcpy(ham->data.u8, ha->data.u8, (size_t)m * cols * 2);
				ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
				ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
				ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
				ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
				ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
				ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
				for (with_bias = 0; with_bias < 2; with_bias++)
				{
					int r1, r2;
					if (with_bias)
					{
						r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0);
						r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0);
					} else {
						r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb1), 0);
						r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr), TENSOR_LIST(gb2), 0);
					}
					REQUIRE_EQ(r1, CCV_NNC_EXEC_SUCCESS, "FTQ GEMM case=%d pass=%d M=%d bias=%d runs", c, pass, m, with_bias);
					REQUIRE_EQ(r2, CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM case=%d pass=%d M=%d bias=%d runs", c, pass, m, with_bias);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
					REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)m * rows * 2), 0, "FTQ GEMM case=%d pass=%d M=%d bias=%d equals the 8I_ROWWISE GEMM", c, pass, m, with_bias);
				}
				ccv_nnc_tensor_free(hb2);
				ccv_nnc_tensor_free(hb1);
				ccv_nnc_tensor_free(gb2);
				ccv_nnc_tensor_free(gb1);
				ccv_nnc_tensor_free(ga);
				ccv_nnc_tensor_free(gar);
				ccv_nnc_tensor_free(ham);
				ccv_nnc_tensor_free(ha);
			}
		}
		if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(gr);
		ccv_nnc_tensor_free(hr);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

TEST_CASE("mps ftq fused gemv for 1 to 3 rows matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// One-hot activation rows make every output fp16(fp32(q8 * scale) [+ bias]) whatever the summation
	// order. Rates from n16 to n48, including an odd word count per tile (n46, 23 words), the
	// largest tiles (16 x 4352 n48) and the 4B size.
	static const int cases[][3] = { { 64, 1024, 24 }, { 528, 1024, 20 }, { 272, 256, 16 }, { 32, 256, 16 }, { 16, 4352, 32 }, { 64, 1024, 40 }, { 16, 4352, 48 }, { 9216, 2560, 36 }, { 9216, 2560, 48 }, { 272, 512, 46 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		ftq_fill(hq->data.u8, ccv_nnc_tensor_data_size_without_padding(hq->info), format, rows, cols, ftq_builtin_table, 0, 0x77u + c);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hbias), TENSOR_LIST(gq, gbias), 0);
		int m;
		for (m = 1; m <= 3; m++)
		{
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
			int c0, mismatches = 0;
			for (c0 = 0; c0 < cols; c0 += m)
			{
				const int with_bias = (c0 / m) & 1;
				memset(ha->data.u8, 0, (size_t)m * cols * 2);
				int i;
				for (i = 0; i < m; i++)
				{
					const float one = 1;
					ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
				int status;
				if (with_bias)
					status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0);
				else
					status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
				REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "FTQ GEMV M=%d runs", m);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
				for (i = 0; i < m; i++)
					for (j = 0; j < rows; j++)
					{
						float v = (float)q8[(size_t)j * cols + (c0 + i) % cols] * scales[j];
						if (with_bias)
							v += bias[j];
						uint16_t e;
						ccv_float_to_half_precision(&v, &e, 1);
						mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != e;
					}
			}
			REQUIRE_EQ(mismatches, 0, "FTQ %dx%d n%d GEMV M=%d one-hot outputs", rows, cols, n, m);
			ccv_nnc_tensor_free(hb);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(ha);
		}
		ccfree(bias);
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
}

TEST_CASE("ftq quantize round trip: CPU, GPU int8 and fused GEMV decodes agree and stay within the error bound")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// 48 rows (three tile rows) x 512 columns of rough Gaussians with row scales 0.02 .. 0.10 and a few outliers.
	enum { rows = 48, cols = 512, count = rows * cols };
	static const int rates[4] = { 16, 20, 32, 48 };
	// Upper bounds on sum (x - x_hat)^2 / sum x^2 per rate.
	static const double bounds[4] = { 0.24, 0.18, 0.09, 0.045 };
	float* const values = (float*)ccmalloc(sizeof(float) * count);
	uint16_t* const half = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	uint32_t state = 0x6a09e667u;
	int i, j, r;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
		{
			double v = 0;
			int k;
			for (k = 0; k < 4; k++)
				v += (ftq_rng_next(&state) >> 8) / 16777216.0 - 0.5;
			if ((i * 131 + j * 17) % 251 == 0)
				v *= 6;
			values[i * cols + j] = (float)(v * 0.02 * (1 + i % 5));
		}
	ccv_float_to_half_precision(values, half, count);
	ccv_half_precision_to_float(half, values, count);
	double energy = 0;
	for (i = 0; i < count; i++)
		energy += (double)values[i] * values[i];
	double previous = DBL_MAX;
	for (r = 0; r < 4; r++)
	{
		const int n = rates[r];
		const int format = ftq_format(n);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, hq->data.u8, size), size, "FTQ n%d quantizes to the blob size", n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		uint32_t header[6];
		memcpy(header, hq->data.u8 + 8, sizeof(header));
		REQUIRE(memcmp(hq->data.u8, "CCVFTQSR", 8) == 0 && header[0] == rows && header[1] == cols && header[2] == 16 * n && header[3] == 0 && header[4] == ftq_builtin_table && header[5] == 0, "FTQ n%d header", n);
		// CPU decode equals stored scale times the reference codes, bit for bit.
		int8_t* const q8 = (int8_t*)ccmalloc(count);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, count), 0, "FTQ n%d CPU decode accepts the blob", n);
		int mismatches = 0;
		double err = 0;
		for (i = 0; i < count; i++)
		{
			const float v = (float)q8[i] * scales[i / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			mismatches += out[i] != e;
			float o;
			ccv_half_precision_to_float(out + i, &o, 1);
			err += ((double)o - values[i]) * ((double)o - values[i]);
		}
		REQUIRE_EQ(mismatches, 0, "FTQ n%d CPU decode is scale times the reference codes", n);
		const double relative = err / energy;
		printf("\n\tFTQ n = %d relative error %.6f", n, relative);
		REQUIRE(relative < bounds[r], "FTQ n%d relative error %g within %g", n, relative, bounds[r]);
		REQUIRE(relative < previous, "error falls as the rate rises");
		previous = relative;
		// GPU int8 decode.
		const size_t out_scale_offset = ((size_t)count + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, count, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, count), 0, "FTQ n%d GPU int8 codes", n);
		REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ n%d GPU int8 scales", n);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		// Fused GEMV for 1 to 3 rows with one-hot activations: every output is fp16(q8 * scale).
		int m;
		for (m = 1; m <= 3; m++)
		{
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
			int c0;
			mismatches = 0;
			for (c0 = 0; c0 < cols; c0 += m)
			{
				memset(ha->data.u8, 0, (size_t)m * cols * 2);
				for (i = 0; i < m; i++)
				{
					const float one = 1;
					ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
				}
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_SUCCESS, "FTQ GEMV M=%d runs", m);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
				for (i = 0; i < m; i++)
					for (j = 0; j < rows; j++)
						mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != out[(size_t)j * cols + (c0 + i) % cols];
			}
			REQUIRE_EQ(mismatches, 0, "FTQ n%d GEMV M=%d one-hot outputs equal the CPU decode", n, m);
			ccv_nnc_tensor_free(hb);
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(ha);
		}
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	ccfree(out);
	ccfree(half);
	ccfree(values);
}

TEST_CASE("mps ftq rejects an unregistered codebook and index select")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 32, cols = 256;
	// A slot nothing registered: the dispatch has no table to bind.
	const int slot = 250;
	REQUIRE(ccv_nnc_codebook_get(slot, 0, 0) == 0, "slot 250 is not registered");
	const int format = CCV_NNC_QX_8I_ROWWISE_FTQ_N20;
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT)), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, rows), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "an unregistered codebook slot is rejected");
	ccv_nnc_tensor_t* const gp = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
	if (ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
		hindices->data.i32[0] = 3;
		ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, cols), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hindices), TENSOR_LIST(gindices), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gp, gindices), TENSOR_LIST(gout), 0), CCV_NNC_EXEC_INVALID, "FTQ tables do not support IndexSelect");
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gindices);
		ccv_nnc_tensor_free(hindices);
	}
	ccv_nnc_tensor_free(gp);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gq);
}

// An external pair table for FTQ: its pairs (2 * 2^log2 int8), the registry slot, and the header fields that name it
// (header id 1, log2 entries, crc32). expand > 0 replicates the built-in table (entry i = T[i >> expand]); 0 is a random table.
typedef struct {
	int8_t* pairs;
	int log2;
	int slot;
	uint32_t table;
	uint32_t crc;
} ftq_table_t;

static ftq_table_t ftq_table_new(const int log2, const int expand, uint32_t seed)
{
	ftq_table_t t;
	const int entries = 1 << log2;
	t.pairs = (int8_t*)ccmalloc(2 * entries);
	int i;
	for (i = 0; i < entries; i++)
		if (expand > 0)
		{
			t.pairs[2 * i] = ccv_nnc_trellis_builtin_table[2 * (i >> expand)];
			t.pairs[2 * i + 1] = ccv_nnc_trellis_builtin_table[2 * (i >> expand) + 1];
		} else {
			t.pairs[2 * i] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
			t.pairs[2 * i + 1] = (int8_t)((int)(ftq_rng_next(&seed) % 255) - 127);
		}
	t.log2 = log2;
	t.slot = ccv_nnc_codebook_register(t.pairs, entries);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | ((uint32_t)log2 << 8);
	return t;
}

// One-hot GEMV outputs (M = 1 .. 3) against fp16(fp32(q8 * scale) [+ bias]); returns the number of mismatches.
static int ftq_onehot_mismatches(ccv_nnc_tensor_t* const gq, ccv_nnc_tensor_t* const gbias, const int8_t* const q8, const float* const scales, const float* const bias, const int rows, const int cols, const int m)
{
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
	int c0, i, j, mismatches = 0;
	for (c0 = 0; c0 < cols; c0 += m)
	{
		const int with_bias = (c0 / m) & 1;
		memset(ha->data.u8, 0, (size_t)m * cols * 2);
		for (i = 0; i < m; i++)
		{
			const float one = 1;
			ccv_float_to_half_precision(&one, (uint16_t*)ha->data.u8 + (size_t)i * cols + (c0 + i) % cols, 1);
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
		const int status = with_bias ?
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0) :
			ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
		if (status != CCV_NNC_EXEC_SUCCESS)
		{
			mismatches = -1;
			break;
		}
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
		for (i = 0; i < m; i++)
			for (j = 0; j < rows; j++)
			{
				float v = (float)q8[(size_t)j * cols + (c0 + i) % cols] * scales[j];
				if (with_bias)
					v += bias[j];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				mismatches += ((uint16_t*)hb->data.u8)[(size_t)i * rows + j] != e;
			}
	}
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ha);
	return mismatches;
}

TEST_CASE("mps ftq with external codebooks matches the reference and the built-in table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	// The built-in table replicated to 1024 / 2048 pairs (decodes as the built-in table), then random 512 / 1024 / 2048 tables.
	ftq_table_t tables[5];
	tables[0] = ftq_table_new(10, 1, 0);
	tables[1] = ftq_table_new(11, 2, 0);
	tables[2] = ftq_table_new(9, 0, 0x1234u);
	tables[3] = ftq_table_new(10, 0, 0x5678u);
	tables[4] = ftq_table_new(11, 0, 0x9abcu);
	int ti;
	for (ti = 0; ti < 5; ti++)
		REQUIRE(tables[ti].slot >= 1, "table %d registers", ti);
	// (rows, cols, n, one-hot GEMV): small grids, the 4B size, and rates up to 3 bpw.
	static const int cases[][4] = { { 64, 1024, 24, 1 }, { 528, 1024, 20, 1 }, { 272, 256, 16, 1 }, { 9216, 2560, 20, 0 }, { 64, 1024, 46, 1 }, { 272, 512, 48, 1 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2], onehot = cases[c][3];
		const int base = ftq_format(n);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(base, rows, cols);
		// The built-in table weights (slot 0) share the code with every table case.
		ccv_nnc_tensor_t* const h0 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), base), 0);
		ccv_nnc_tensor_t* const g0 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), base), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(h0->info);
		ftq_fill(h0->data.u8, size, base, rows, cols, ftq_builtin_table, 0, 0xe0e0u + c);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(h0), TENSOR_LIST(g0), 0);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(h0->data.u8 + schedule.scale_offset), scales, rows);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		for (ti = 0; ti < 5; ti++)
		{
			const ftq_table_t t = tables[ti];
			const int format = base | (t.slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
			memcpy(hq->data.u8, h0->data.u8, size);
			const uint32_t header_table[2] = { t.table, t.crc };
			memcpy(hq->data.u8 + 24, header_table, sizeof(header_table));
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
			ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, t.pairs, t.log2, q8);
			// GPU int8 decode.
			ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
			REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d table %d GPU codes", rows, cols, n, ti);
			// CPU decode with the same table.
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols);
			size_t k, cpu_mismatches = 0;
			for (k = 0; k < (size_t)rows * cols; k++)
			{
				const float v = (float)q8[k] * scales[k / cols];
				uint16_t e;
				ccv_float_to_half_precision(&v, &e, 1);
				cpu_mismatches += out[k] != e;
			}
			ccfree(out);
			REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ %dx%d n%d table %d CPU decode", rows, cols, n, ti);
			// Fused GEMV, one-hot.
			int m;
			if (onehot)
				for (m = 1; m <= 3; m++)
					REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ %dx%d n%d table %d GEMV M=%d one-hot", rows, cols, n, ti, m);
			// GEMM from 4 rows against resident 8I_ROWWISE of the reference codes (at least 9 rows), and for the
			// replicated built-in tables every M against the built-in table bit for bit.
			ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
			memset(hr->data.u8, 0, out_scale_offset);
			memcpy(hr->data.u8, q8, (size_t)rows * cols);
			memcpy(hr->data.u8 + out_scale_offset, h0->data.u8 + schedule.scale_offset, rows * 2);
			ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
			static const int ms[] = { 1, 2, 3, 4, 8, 9, 64 };
			int pass, mi, i;
			for (pass = 0; pass < 2; pass++)
			{
				if (pass == 1)
					ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
				for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
				{
					const int mm = ms[mi];
					const int mr = mm < 9 ? 9 : mm;
					ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
					for (i = 0; i < mr; i++)
						for (k = 0; k < (size_t)cols; k++)
						{
							const float v = ftq_activation_value(i, (int)k);
							ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
						}
					ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, cols), 0);
					memcpy(ham->data.u8, ha->data.u8, (size_t)mm * cols * 2);
					ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
					ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, cols), 0);
					ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const gb0 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
					ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const hb0 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
					ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0), CCV_NNC_EXEC_SUCCESS, "FTQ table %d GEMM M=%d runs", ti, mm);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, g0, gbias), TENSOR_LIST(gb0), 0), CCV_NNC_EXEC_SUCCESS, "FTQ built-in GEMM M=%d runs", mm);
					REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0), CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM M=%d runs", mr);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb0, gb2), TENSOR_LIST(hb1, hb0, hb2), 0);
					if (mm >= 4)
						REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)mm * rows * 2), 0, "FTQ %dx%d n%d table %d pass %d GEMM M=%d equals 8I_ROWWISE", rows, cols, n, ti, pass, mm);
					if (ti < 2)
						REQUIRE_EQ(memcmp(hb1->data.u8, hb0->data.u8, (size_t)mm * rows * 2), 0, "FTQ %dx%d n%d replicated table %d pass %d M=%d equals the built-in table", rows, cols, n, ti, pass, mm);
					ccv_nnc_tensor_free(hb2);
					ccv_nnc_tensor_free(hb0);
					ccv_nnc_tensor_free(hb1);
					ccv_nnc_tensor_free(gb2);
					ccv_nnc_tensor_free(gb0);
					ccv_nnc_tensor_free(gb1);
					ccv_nnc_tensor_free(ga);
					ccv_nnc_tensor_free(gar);
					ccv_nnc_tensor_free(ham);
					ccv_nnc_tensor_free(ha);
				}
			}
			if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
				ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
			ccv_nnc_tensor_free(gr);
			ccv_nnc_tensor_free(hr);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
		}
		ccfree(q8);
		ccfree(bias);
		ccfree(scales);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(g0);
		ccv_nnc_tensor_free(h0);
	}
	for (ti = 0; ti < 5; ti++)
		ccfree(tables[ti].pairs);
}


// Quadrant-folded tables (FTQ table bits 16 - 17 = 1, CCV_NNC_CODEBOOK_FOLD). The reference expands the 512 base pairs
// to the explicit 2048-pair table (pair i: base pair i & 511, the first value negated by bit 9 of i, the second by bit
// 10) and looks it up with the index i = ((h >> 16) & 511) | bit 15 of h << 9 | bit 31 of h << 10, h = state *
// 0x83DCD12D, written independently of the codec.
static void ftq_ref_decode_fold(const uint8_t* const code, const int rows, const int cols, const uint32_t high, const uint64_t drop, const int8_t* const base, int8_t* const q8)
{
	int8_t expanded[4096];
	int i;
	for (i = 0; i < 2048; i++)
	{
		const int a = base[2 * (i & 511)], b = base[2 * (i & 511) + 1];
		expanded[2 * i] = (int8_t)((i >> 9) & 1 ? -a : a);
		expanded[2 * i + 1] = (int8_t)((i >> 10) & 1 ? -b : b);
	}
	const uint64_t count = (uint64_t)rows * cols / 256;
	uint64_t* const starts = (uint64_t*)ccmalloc(sizeof(uint64_t) * count);
	uint32_t* const lengths = (uint32_t*)ccmalloc(sizeof(uint32_t) * count);
	ftq_ref_layout(rows, cols, high, drop, starts, lengths);
	const int strips = cols / 16;
	uint64_t t;
	for (t = 0; t < count; t++)
	{
		const uint64_t start = starts[t];
		const uint64_t T = lengths[t];
		int j, b;
		for (j = 0; j < 128; j++)
		{
			const uint64_t end = (uint64_t)(j + 1) * T / 128;
			uint32_t state = 0;
			for (b = 0; b < 16; b++)
			{
				const uint64_t p = start + (end + T - 16 + b) % T;
				state |= (uint32_t)((code[p >> 3] >> (p & 7)) & 1) << b;
			}
			const uint32_t h = state * 0x83DCD12Du;
			const uint32_t k = ((h >> 16) & 511) | (((h >> 15) & 1) << 9) | ((h >> 31) << 10);
			const int row = (int)(t / strips) * 16 + j / 8;
			const int col = (int)(t % strips) * 16 + 2 * (j % 8);
			q8[(size_t)row * cols + col] = expanded[2 * k];
			q8[(size_t)row * cols + col + 1] = expanded[2 * k + 1];
		}
	}
	ccfree(lengths);
	ccfree(starts);
}

static int8_t* ftq_fold_base_new(uint32_t seed)
{
	int8_t* const base = (int8_t*)ccmalloc(1024);
	int i;
	for (i = 0; i < 1024; i++)
		base[i] = (int8_t)(ftq_rng_next(&seed) % 128);
	return base;
}

TEST_CASE("codebook registry keeps folded and plain tables apart")
{
	int8_t* const base = ftq_fold_base_new(0x0f01u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(plain >= 1 && folded >= 1 && plain != folded, "the same content plain and folded are two slots");
	REQUIRE_EQ(ccv_nnc_codebook_flags(plain), 0, "plain slot has no flags");
	REQUIRE_EQ(ccv_nnc_codebook_flags(folded), CCV_NNC_CODEBOOK_FOLD, "folded slot is folded");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(plain), folded, "the folded slot of the plain one");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(folded), -1, "a folded slot does not fold again");
	REQUIRE_EQ(ccv_nnc_codebook_fold_slot(0), -1, "the built-in table does not fold");
	REQUIRE_EQ(ccv_nnc_codebook_flags(0), 0, "built-in has no flags");
	REQUIRE_EQ(ccv_nnc_codebook_flags(251), -1, "unregistered slot");
	int8_t negative[1024];
	memcpy(negative, base, 1024);
	negative[5] = -3;
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(negative, 512, CCV_NNC_CODEBOOK_FOLD), -1, "a folded table has no negative value");
	REQUIRE(ccv_nnc_codebook_fold_slot(ccv_nnc_codebook_register(negative, 512)) == -1, "a table with a negative value does not fold");
	int8_t big[2048];
	memset(big, 1, sizeof(big));
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(big, 1024, CCV_NNC_CODEBOOK_FOLD), -1, "a folded table has 512 pairs");
	REQUIRE_EQ(ccv_nnc_codebook_register_with_flags(base, 512, 2), -1, "unknown flags");
	ccfree(base);
}

TEST_CASE("ftq quadrant-folded tables match the explicit 2048-pair reference on CPU, GPU, GEMV and GEMM")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	int8_t* const base = ftq_fold_base_new(0x0f17u);
	const int slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(slot >= 1, "folded table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(slot, 0, &crc);
	const uint32_t table_field = 1u | (9u << 8) | (1u << 16);
	// (rows, cols, n, one-hot GEMV): small grids, the 4B size at n20, n40 and n48, and n46.
	static const int cases[][4] = { { 64, 1024, 24, 1 }, { 272, 256, 16, 1 }, { 9216, 2560, 20, 0 }, { 9216, 2560, 40, 0 }, { 64, 1024, 46, 1 }, { 272, 512, 48, 1 }, { 9216, 2560, 48, 0 } };
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2], onehot = cases[c][3];
		const int format = ftq_format(n) | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
		ftq_fill(hq->data.u8, size, format, rows, cols, table_field, crc, 0xf0f0u + c);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_fold(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
		size_t k, negatives = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
			negatives += q8[k] < 0;
		REQUIRE(negatives > (size_t)rows * cols / 4, "the reference flips signs (%zu negative)", negatives);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
		// CPU decode.
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), 0, "FTQ fold CPU decode runs");
		size_t cpu_mismatches = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
		{
			const float v = (float)q8[k] * scales[k / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			cpu_mismatches += out[k] != e;
		}
		ccfree(out);
		REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ fold %dx%d n%d CPU decode", rows, cols, n);
		// GPU int8 decode.
		const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
		const size_t out_bytes = out_scale_offset + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ fold %dx%d n%d GPU codes", rows, cols, n);
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		int j;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
		int m;
		if (onehot)
			for (m = 1; m <= 3; m++)
				REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ fold %dx%d n%d GEMV M=%d one-hot", rows, cols, n, m);
		// GEMV (M = 1 .. 3, the 4B shapes) and GEMM from 4 rows against resident 8I_ROWWISE of the reference codes;
		// GEMV rows sum in another order, so the GEMV is compared within FP16 rounding.
		ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
		memset(hr->data.u8, 0, out_scale_offset);
		memcpy(hr->data.u8, q8, (size_t)rows * cols);
		memcpy(hr->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2);
		ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
		static const int ms[] = { 1, 3, 4, 8, 9, 64 };
		int pass, mi, i;
		for (pass = 0; pass < 2; pass++)
		{
			if (pass == 1)
				ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
			{
				const int mm = ms[mi];
				const int mr = mm < 9 ? 9 : mm;
				ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
				for (i = 0; i < mr; i++)
					for (k = 0; k < (size_t)cols; k++)
					{
						const float v = ftq_activation_value(i, (int)k);
						ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
					}
				ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, cols), 0);
				memcpy(ham->data.u8, ha->data.u8, (size_t)mm * cols * 2);
				ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
				ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, cols), 0);
				ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
				ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
				ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
				ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0), CCV_NNC_EXEC_SUCCESS, "FTQ fold GEMM M=%d runs", mm);
				REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0), CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM M=%d runs", mr);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
				if (mm >= 4) {
					REQUIRE_EQ(memcmp(hb1->data.u8, hb2->data.u8, (size_t)mm * rows * 2), 0, "FTQ fold %dx%d n%d pass %d GEMM M=%d equals 8I_ROWWISE", rows, cols, n, pass, mm);
				} else {
					// The fused GEMV against an FP64 sum of the reference codes: within two FP16 roundings of the
					// result plus FP32 summation error (bounded by the sum of |terms|).
					float* const f1 = (float*)ccmalloc(sizeof(float) * mm * rows);
					ccv_half_precision_to_float((uint16_t*)hb1->data.u8, f1, (size_t)mm * rows);
					float* const fa = (float*)ccmalloc(sizeof(float) * mm * cols);
					ccv_half_precision_to_float((uint16_t*)ham->data.u8, fa, (size_t)mm * cols);
					int bad = 0;
					for (i = 0; i < mm && !bad; i++)
						for (j = 0; j < rows && !bad; j++)
						{
							double sum = 0, magnitude = 0;
							for (k = 0; k < (size_t)cols; k++)
							{
								const double term = (double)q8[(size_t)j * cols + k] * fa[(size_t)i * cols + k];
								sum += term;
								magnitude += fabs(term);
							}
							const double expected = sum * scales[j] + bias[j];
							const double err = fabs((double)f1[(size_t)i * rows + j] - expected);
							if (err > fabs(expected) * (2.0 / 2048) + magnitude * scales[j] * 1e-6 + 1e-6)
								bad = 1;
						}
					ccfree(fa);
					ccfree(f1);
					REQUIRE(!bad, "FTQ fold %dx%d n%d GEMV M=%d within rounding of the FP64 sum", rows, cols, n, mm);
				}
				ccv_nnc_tensor_free(hb2);
				ccv_nnc_tensor_free(hb1);
				ccv_nnc_tensor_free(gb2);
				ccv_nnc_tensor_free(gb1);
				ccv_nnc_tensor_free(ga);
				ccv_nnc_tensor_free(gar);
				ccv_nnc_tensor_free(ham);
				ccv_nnc_tensor_free(ha);
			}
		}
		if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
			ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
		ccfree(bias);
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(gr);
		ccv_nnc_tensor_free(hr);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	ccfree(base);
}

TEST_CASE("cpu ftq decode rejects a fold field that does not match the table, also without asserts")
{
	int8_t* const base = ftq_fold_base_new(0x0f23u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	uint32_t crc = 0;
	ccv_nnc_codebook_get(folded, 0, &crc);
	const int rows = 64, cols = 1024;
	const int base_format = ftq_format(24);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(base_format, CCV_16F, (size_t)rows * cols, cols);
	uint8_t* const data = (uint8_t*)ccmalloc(size);
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	// (slot, table field, expected): good folded, the fold bit with a plain slot, a plain field with a folded slot,
	// fold with log2 10, fold with id 0, structure 2, bits 18 - 31.
	const struct { int slot; uint32_t field; int ok; const char* what; } cases[] = {
		{ folded, 1u | (9u << 8) | (1u << 16), 0, "folded" },
		{ plain, 1u | (9u << 8) | (1u << 16), -1, "fold bit, plain slot" },
		{ folded, 1u | (9u << 8), -1, "plain field, folded slot" },
		{ folded, 1u | (10u << 8) | (1u << 16), -1, "fold with log2 10" },
		{ folded, (9u << 8) | (1u << 16), -1, "fold with id 0" },
		{ folded, 1u | (9u << 8) | (2u << 16), -1, "structure 2" },
		{ folded, 1u | (9u << 8) | (1u << 16) | (1u << 18), -1, "bit 18" },
		{ folded, 1u | (9u << 8) | (1u << 16) | (1u << 31), -1, "bit 31" },
	};
	int i;
	for (i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++)
	{
		const int format = base_format | (cases[i].slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
		ftq_fill(data, size, format, rows, cols, cases[i].field, crc, 0x77u);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(data, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), cases[i].ok, "FTQ %s", cases[i].what);
	}
	// A folded slot names a table for FTQ only.
	const int tmcq_format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N20 | (folded << CCV_NNC_CODEBOOK_SLOT_SHIFT);
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(tmcq_format, CCV_16F, (size_t)rows * cols, cols) > 0 ? ccv_nnc_dequantize_8i_rowwise_x(data, CCV_16F, CCV_TENSOR_CPU_MEMORY, ccv_nnc_8i_rowwise_x_data_size(tmcq_format, CCV_16F, (size_t)rows * cols, cols), cols, tmcq_format, out, (size_t)rows * cols) : -1, -1, "TMCQ with a folded slot");
	ccfree(out);
	ccfree(data);
	ccfree(base);
}

TEST_CASE("an ftq checkpoint with a folded table inline reads into the folded slot and other schedules fail to load")
{
	int8_t* const base = ftq_fold_base_new(0x0f31u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	REQUIRE(plain >= 1, "base table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(plain, 0, &crc);
	const int rows = 64, cols = 1024;
	// The base table inline; the header keeps id 4 (not looked up) and names the structure.
	const int file_format = ccv_nnc_8i_rowwise_x_with_inline_codebook(ftq_format(24), 512);
	const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(file_format);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	// header: 0 as filled, 1 a non-zero drop_count, 2 high_bits 16 n - 1 (schedules FTQ rejects).
	const struct { const char* name; uint32_t field; int folded; int header; } tensors[] = {
		{ "w_fold", 4u | (9u << 8) | (1u << 16), 1, 0 },
		{ "w_plain", 4u | (9u << 8), 0, 0 },
		{ "w_bad_structure", 4u | (9u << 8) | (3u << 16), -1, 0 },
		{ "w_drop", 4u | (9u << 8), -1, 1 },
		{ "w_short_high", 4u | (9u << 8), -1, 2 },
	};
	const int tensor_count = (int)(sizeof(tensors) / sizeof(tensors[0]));
	int i;
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), file_format), 0);
		memset(t->data.u8, 0, prefix);
		REQUIRE_EQ(ccv_nnc_codebook_serialize(plain, t->data.u8, prefix), 16 + 1024, "the base table serializes");
		uint8_t* const blob = t->data.u8 + prefix;
		ftq_fill(blob, ccv_nnc_tensor_data_size_without_padding(t->info) - prefix, ftq_format(24), rows, cols, tensors[i].field, crc, 0x99u + i);
		if (tensors[i].header)
		{
			uint32_t v;
			memcpy(&v, blob + (tensors[i].header == 1 ? 20 : 16), 4);
			v = tensors[i].header == 1 ? 1 : v - 1;
			memcpy(blob + (tensors[i].header == 1 ? 20 : 16), &v, 4);
		}
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, tensors[i].name, 0), CCV_IO_FINAL, "%s writes", tensors[i].name);
		ccv_nnc_tensor_free(t);
	}
	for (i = 0; i < tensor_count; i++)
	{
		ccv_nnc_tensor_t* restored = 0;
		const int result = ccv_nnc_tensor_read(db, tensors[i].name, 0, 0, 0, &restored);
		if (tensors[i].folded < 0)
		{
			REQUIRE(result != CCV_IO_FINAL && !restored, "%s is rejected", tensors[i].name);
			continue;
		}
		REQUIRE_EQ(result, CCV_IO_FINAL, "%s reads", tensors[i].name);
		REQUIRE_EQ(restored->info.reserved, file_format, "%s keeps the inline bits", tensors[i].name);
		const int resolved = ccv_nnc_8i_rowwise_x_resolve_inline(restored->data.u8, ccv_nnc_tensor_data_size_without_padding(restored->info), restored->info.reserved);
		REQUIRE(resolved >= 0, "%s resolves", tensors[i].name);
		const int slot = ccv_nnc_codebook_slot_of(resolved);
		REQUIRE_EQ(ccv_nnc_codebook_flags(slot), tensors[i].folded ? CCV_NNC_CODEBOOK_FOLD : 0, "%s slot flags", tensors[i].name);
		if (!tensors[i].folded)
			REQUIRE_EQ(slot, plain, "%s takes the plain slot", tensors[i].name);
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, ccv_nnc_tensor_data_size_without_padding(restored->info), cols, restored->info.reserved, out, (size_t)rows * cols), 0, "%s decodes", tensors[i].name);
		ccfree(out);
		ccv_nnc_tensor_free(restored);
	}
	sqlite3_close(db);
	ccfree(base);
}

// Tile-row boundaries of the int8 decode, which runs in blocks of 2 neighbouring tiles x 16 rows: 1, 3, 13 and 17 tile
// rows (16 x 256, 48 x 256, 208 x 768, 272 x 512), at n16, n22, n36, n46 and n48.
// GPU int8 decode, CPU decode and the one-hot fused GEMV (M = 1 .. 3) must equal
// the reference bit for bit, for the built-in table and a quadrant-folded one.
TEST_CASE("mps ftq decode is exact across odd tile-row counts with the built-in and a folded table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0xb0d7u);
	const int fold_slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(fold_slot >= 1, "folded table registers");
	uint32_t fold_crc = 0;
	ccv_nnc_codebook_get(fold_slot, 0, &fold_crc);
	static const int shapes[][2] = { { 16, 256 }, { 48, 256 }, { 208, 768 }, { 272, 512 } };
	static const int rates[] = { 16, 22, 36, 46, 48 };
	int s, r, folded;
	for (s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
		for (r = 0; r < (int)(sizeof(rates) / sizeof(rates[0])); r++)
			for (folded = 0; folded < 2; folded++)
			{
				const int rows = shapes[s][0], cols = shapes[s][1], n = rates[r];
				const int format = ftq_format(n) | ((folded ? fold_slot : 0) << CCV_NNC_CODEBOOK_SLOT_SHIFT);
				const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
				ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
				const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
				ftq_fill(hq->data.u8, size, format, rows, cols, folded ? 1u | (9u << 8) | (1u << 16) : ftq_builtin_table, folded ? fold_crc : 0, 0xb0d0u + s * 16 + r * 2 + folded);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
				int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
				if (folded)
					ftq_ref_decode_fold(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
				else
					ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
				float* const scales = (float*)ccmalloc(sizeof(float) * rows);
				ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
				// CPU decode.
				uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
				REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), 0, "FTQ %dx%d n%d fold %d CPU decode runs", rows, cols, n, folded);
				size_t k, cpu_mismatches = 0;
				for (k = 0; k < (size_t)rows * cols; k++)
				{
					const float v = (float)q8[k] * scales[k / cols];
					uint16_t e;
					ccv_float_to_half_precision(&v, &e, 1);
					cpu_mismatches += out[k] != e;
				}
				ccfree(out);
				REQUIRE_EQ((int)cpu_mismatches, 0, "FTQ %dx%d n%d fold %d CPU decode", rows, cols, n, folded);
				// GPU int8 decode, codes and scales.
				const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
				const size_t out_bytes = out_scale_offset + rows * 2;
				ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
				ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
				ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
				REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "FTQ %dx%d n%d fold %d GPU codes", rows, cols, n, folded);
				REQUIRE_EQ(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2), 0, "FTQ %dx%d n%d fold %d GPU scales", rows, cols, n, folded);
				// One-hot fused GEMV.
				ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
				bias_params.dim[1] = 0;
				ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
				bias_params.type = CCV_TENSOR_GPU_MEMORY;
				ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
				float* const bias = (float*)ccmalloc(sizeof(float) * rows);
				int j, m;
				for (j = 0; j < rows; j++)
				{
					const float v = ftq_bias_value(j);
					ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
				}
				ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
				for (m = 1; m <= 3; m++)
					REQUIRE_EQ(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m), 0, "FTQ %dx%d n%d fold %d GEMV M=%d one-hot", rows, cols, n, folded, m);
				ccfree(bias);
				ccv_nnc_tensor_free(gbias);
				ccv_nnc_tensor_free(hbias);
				ccv_nnc_tensor_free(hout);
				ccv_nnc_tensor_free(gout);
				ccfree(scales);
				ccfree(q8);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(hq);
			}
	ccfree(base);
}

// Variable-rate FTQ (CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16). Tile row rates follow the r6a level set: nm + {-12, -8, -4, 0,
// +4, +12} by tile row mod 6, clamped to 8 ... 48 (every prefix of the offsets sums to at most nm per tile row).
static int ftq_vr_format(const int nm)
{
	return CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST + (nm - 8) / 2;
}

static void ftq_vr_rates(const int tile_rows, const int nm, int* const rates)
{
	static const int deltas[6] = { -12, -8, -4, 0, 4, 12 };
	int r;
	for (r = 0; r < tile_rows; r++)
	{
		const int n = nm + deltas[r % 6];
		rates[r] = n < 8 ? 8 : n > 48 ? 48 : n;
	}
}

// A CCVFTQVR blob at the given tile row rates: header, offsets, random code words, zero padding, random FP16 scales.
static void ftq_vr_fill(uint8_t* const data, const size_t size, const int format, const int rows, const int cols, const int* const rates, const uint32_t table, const uint32_t table_crc, uint32_t seed)
{
	const int tile_rows = rows / 16, tiles = cols / 16;
	memset(data, 0, size);
	memcpy(data, "CCVFTQVR", 8);
	uint32_t mask = 0;
	int r;
	for (r = 0; r < tile_rows; r++)
		mask |= 1u << ((rates[r] - 8) / 2);
	const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, 16u * ccv_nnc_8i_rowwise_x_ftq_vr_rate(format), mask, table, table_crc };
	memcpy(data + 8, header, sizeof(header));
	uint32_t* const off = (uint32_t*)(data + 32);
	off[0] = 0;
	for (r = 0; r < tile_rows; r++)
		off[r + 1] = off[r] + (uint32_t)(tiles * rates[r] / 2);
	uint32_t* const code = off + tile_rows + 1;
	uint32_t w;
	for (w = 0; w < off[tile_rows]; w++)
		code[w] = ftq_rng_next(&seed);
	const size_t scale_offset = size - 2 * rows;
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	for (r = 0; r < rows; r++)
		scales[r] = (float)((ftq_rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
	ccv_float_to_half_precision(scales, (uint16_t*)(data + scale_offset), rows);
	ccfree(scales);
}

// The reference decode of a CCVFTQVR blob: tile row r is static-rate FTQ at rate n_r from word off[r] of the code.
static void ftq_vr_ref_decode(const uint8_t* const data, const int rows, const int cols, const int8_t* const table, const int folded, int8_t* const q8)
{
	const int tile_rows = rows / 16, tiles = cols / 16;
	const uint32_t* const off = (const uint32_t*)(data + 32);
	const uint8_t* const code = (const uint8_t*)(off + tile_rows + 1);
	int r;
	for (r = 0; r < tile_rows; r++)
	{
		const uint32_t high = 16u * (2u * (off[r + 1] - off[r]) / tiles);
		if (folded)
			ftq_ref_decode_fold(code + 4 * (size_t)off[r], 16, cols, high, 0, table, q8 + (size_t)r * 16 * cols);
		else
			ftq_ref_decode_matrix(code + 4 * (size_t)off[r], 16, cols, high, 0, table, 9, q8 + (size_t)r * 16 * cols);
	}
}

// CPU decode, GPU int8 decode (codes and scales) and, with gemv, the one-hot fused GEMV for M = 1 .. 3 against the
// reference codes q8, bit for bit. Returns the CPU decode in out and the number of failed checks (each printed).
#define FTQ_VR_EXPECT(cond, ...) do { if (!(cond)) { printf("\n\tFAILED: "); printf(__VA_ARGS__); failures++; } } while (0)
static int ftq_vr_decodes(ccv_nnc_tensor_t* const hq, ccv_nnc_tensor_t* const gq, const int format, const int8_t* const q8, const int rows, const int cols, const int gemv, uint16_t* const out, const char* const what)
{
	int failures = 0;
	const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
	const size_t scale_offset = size - 2 * rows;
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + scale_offset), scales, rows);
	FTQ_VR_EXPECT(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols) == 0, "%s CPU decode runs", what);
	size_t k, mismatches = 0;
	for (k = 0; k < (size_t)rows * cols; k++)
	{
		const float v = (float)q8[k] * scales[k / cols];
		uint16_t e;
		ccv_float_to_half_precision(&v, &e, 1);
		mismatches += out[k] != e;
	}
	FTQ_VR_EXPECT(mismatches == 0, "%s CPU decode, %zu mismatches", what, mismatches);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
	const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
	const size_t out_bytes = out_scale_offset + rows * 2;
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
	FTQ_VR_EXPECT(memcmp(hout->data.u8, q8, (size_t)rows * cols) == 0, "%s GPU int8 codes", what);
	FTQ_VR_EXPECT(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + scale_offset, rows * 2) == 0, "%s GPU int8 scales", what);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gout);
	if (gemv)
	{
		ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
		bias_params.dim[1] = 0;
		ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
		bias_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
		float* const bias = (float*)ccmalloc(sizeof(float) * rows);
		int j, m;
		for (j = 0; j < rows; j++)
		{
			const float v = ftq_bias_value(j);
			ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
		}
		ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
		for (m = 1; m <= 3; m++)
		{
			const int gemv_mismatches = ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m);
			FTQ_VR_EXPECT(gemv_mismatches == 0, "%s GEMV M=%d one-hot, %d mismatches", what, m, gemv_mismatches);
		}
		ccfree(bias);
		ccv_nnc_tensor_free(gbias);
		ccv_nnc_tensor_free(hbias);
	}
	ccfree(scales);
	return failures;
}

TEST_CASE("ftq_vr enum block, sizes and the rate formula")
{
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16, 100, "FTQ_VR codec ids are 96 + (nm - 8) / 2");
	REQUIRE_EQ(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N48, 116, "FTQ_VR codec ids are 96 + (nm - 8) / 2");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_rate(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20), 20, "nm20");
	REQUIRE(!ccv_nnc_8i_rowwise_x_is_ftq_vr(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST) && !ccv_nnc_8i_rowwise_x_is_ftq_vr(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N48 + 1) && !ccv_nnc_8i_rowwise_x_is_ftq(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20), "only the open variable-rate codes");
	// 48 x 512: R = 3, C = 32, scale_offset = align128(32 + 16 + 3 * 32 * 2 * 20) = 3968.
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20, CCV_16F, 48 * 512, 512), 3968 + 96, "blob size");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_data_size(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N20, CCV_32F, 48 * 512, 512), 0, "FP16 only");
}

// Only n (FTQ) and nm (FTQ_VR) 16 .. 48 are open: the other codes of both blocks neither quantize, decode nor load, and a
// variable-rate tile row rate above 48 does not quantize.
TEST_CASE("ftq and ftq_vr codes above n48 do not quantize, decode or load")
{
	enum { rows = 16, cols = 256, count = rows * cols };
	static uint16_t half[count];
	static uint8_t blob[count * 4];
	static uint16_t out[count];
	static const int rates[] = { 50, 64, 88, 134 };
	int i, vr;
	for (vr = 0; vr < 2; vr++)
		for (i = 0; i < (int)(sizeof(rates) / sizeof(rates[0])); i++)
		{
			const int n = rates[i];
			const int format = vr ? ftq_vr_format(n) : ftq_format(n);
			REQUIRE(!ccv_nnc_8i_rowwise_x_is_tiled(format) && !ccv_nnc_8i_rowwise_x_codebook_is_valid(format), "%s n%d is not a format", vr ? "FTQ_VR" : "FTQ", n);
			const int row_rates[1] = { n };
			const size_t written = vr ? ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, row_rates, blob, sizeof(blob)) : ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, blob, sizeof(blob));
			REQUIRE_EQ(written, 0, "%s n%d does not quantize", vr ? "FTQ_VR" : "FTQ", n);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(blob, CCV_16F, CCV_TENSOR_CPU_MEMORY, sizeof(blob), cols, format, out, count), -1, "%s n%d does not decode", vr ? "FTQ_VR" : "FTQ", n);
		}
	// FTQ_VR_N48 takes tile row rates up to 48.
	const int rates_over[1] = { 50 };
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N48, rates_over, blob, sizeof(blob)), 0, "a tile row at n50 does not quantize");
	// A checkpoint of an FTQ_N48 / FTQ_VR_N48 weight whose stored format is moved to n50 fails to load.
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	for (vr = 0; vr < 2; vr++)
	{
		const int format = vr ? CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N48 : CCV_NNC_QX_8I_ROWWISE_FTQ_N48;
		ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(t->info);
		const int row_rates[1] = { 48 };
		REQUIRE_EQ(vr ? ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, row_rates, t->data.u8, size) : ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, 0, 0, 0, t->data.u8, size), size, "n48 quantizes");
		const char* const name = vr ? "vr" : "sr";
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "%s writes", name);
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s n48 loads", name);
		ccv_nnc_tensor_free(restored);
		sqlite3_stmt* stmt = 0;
		REQUIRE_EQ(sqlite3_prepare_v2(db, "UPDATE tensors SET datatype=$datatype WHERE name=$name", -1, &stmt, 0), SQLITE_OK, "update prepares");
		const int moved = format + 1;
		sqlite3_bind_int64(stmt, 1, ((sqlite_int64)moved << 32) | (unsigned int)t->info.datatype);
		sqlite3_bind_text(stmt, 2, name, -1, 0);
		REQUIRE_EQ(sqlite3_step(stmt), SQLITE_DONE, "the stored format moves to n50");
		sqlite3_finalize(stmt);
		restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_ERROR, "%s n50 does not load", name);
		REQUIRE(restored == 0, "%s leaves no tensor", name);
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, CCV_NNC_TENSOR_READ_METADATA_ONLY, 0, &restored), CCV_IO_ERROR, "%s n50 metadata does not load", name);
		ccv_nnc_tensor_free(t);
	}
	sqlite3_close(db);
}

TEST_CASE("mps ftq_vr decode matches the reference on CPU, GPU int8 and fused GEMV across odd tile-row counts with the built-in and a folded table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0x7a11u);
	const int fold_slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(fold_slot >= 1, "folded table registers");
	uint32_t fold_crc = 0;
	ccv_nnc_codebook_get(fold_slot, 0, &fold_crc);
	// 1, 3, 13 and 17 tile rows; 1024 x 1024 has 2^16 row segments (one block per thread folded), 4096 x 4096 2^20 (two).
	static const int shapes[][3] = { { 16, 256, 1 }, { 48, 256, 1 }, { 208, 768, 1 }, { 272, 512, 1 }, { 1024, 1024, 0 }, { 4096, 4096, 0 } };
	static const int means[] = { 20, 32 };
	int s, m, folded;
	for (s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
		for (m = 0; m < 2; m++)
			for (folded = 0; folded < 2; folded++)
			{
				const int rows = shapes[s][0], cols = shapes[s][1], nm = means[m];
				if (rows > 1024 && nm != 20)
					continue;
				const int format = ftq_vr_format(nm) | ((folded ? fold_slot : 0) << CCV_NNC_CODEBOOK_SLOT_SHIFT);
				int* const rates = (int*)ccmalloc(sizeof(int) * (rows / 16));
				ftq_vr_rates(rows / 16, nm, rates);
				ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
				const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
				ftq_vr_fill(hq->data.u8, size, format, rows, cols, rates, folded ? 1u | (9u << 8) | (1u << 16) : ftq_builtin_table, folded ? fold_crc : 0, 0x7a00u + s * 8 + m * 2 + folded);
				REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(hq->data.u8, size, format, rows, cols, 0), 0, "the filled blob passes the checks");
				int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
				ftq_vr_ref_decode(hq->data.u8, rows, cols, folded ? base : ccv_nnc_trellis_builtin_table, folded, q8);
				uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
				char what[128];
				snprintf(what, sizeof(what), "FTQ_VR %dx%d nm%d fold %d", rows, cols, nm, folded);
				REQUIRE_EQ(ftq_vr_decodes(hq, gq, format, q8, rows, cols, shapes[s][2], out, what), 0, "%s decodes bit for bit", what);
				ccfree(out);
				ccfree(q8);
				ccfree(rates);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(hq);
			}
	ccfree(base);
}

TEST_CASE("ftq_vr quantize round trip at r6a rates: CPU, GPU int8 and fused GEMV agree and stay within the error bound")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int shapes[][2] = { { 16, 256 }, { 48, 512 }, { 208, 256 }, { 272, 256 } };
	static const int means[] = { 20, 32 };
	// Upper bounds on sum (x - x_hat)^2 / sum x^2 per shape and mean rate (one tile row of 16 x 256 is all n = 8).
	static const double bounds[4][2] = { { 0.46, 0.17 }, { 0.35, 0.13 }, { 0.26, 0.095 }, { 0.25, 0.093 } };
	int s, m;
	for (s = 0; s < 4; s++)
		for (m = 0; m < 2; m++)
		{
			const int rows = shapes[s][0], cols = shapes[s][1], nm = means[m], count = rows * cols;
			const int format = ftq_vr_format(nm);
			int* const rates = (int*)ccmalloc(sizeof(int) * (rows / 16));
			ftq_vr_rates(rows / 16, nm, rates);
			float* const values = (float*)ccmalloc(sizeof(float) * count);
			uint16_t* const half = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
			uint32_t state = 0x3c6ef372u + s * 2 + m;
			int i, j;
			for (i = 0; i < rows; i++)
				for (j = 0; j < cols; j++)
				{
					double v = 0;
					int k;
					for (k = 0; k < 4; k++)
						v += (ftq_rng_next(&state) >> 8) / 16777216.0 - 0.5;
					values[i * cols + j] = (float)(v * 0.02 * (1 + i % 5));
				}
			ccv_float_to_half_precision(values, half, count);
			ccv_half_precision_to_float(half, values, count);
			ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, rates, hq->data.u8, size), size, "FTQ_VR %dx%d nm%d quantizes", rows, cols, nm);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(hq->data.u8, size, format, rows, cols, 0), 0, "the encoded blob passes the checks");
			const uint32_t* const off = (const uint32_t*)(hq->data.u8 + 32);
			for (i = 0; i < rows / 16; i++)
				REQUIRE_EQ((int)(2 * (off[i + 1] - off[i]) / (cols / 16)), rates[i], "tile row %d has its rate", i);
			int8_t* const q8 = (int8_t*)ccmalloc(count);
			ftq_vr_ref_decode(hq->data.u8, rows, cols, ccv_nnc_trellis_builtin_table, 0, q8);
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
			char what[128];
			snprintf(what, sizeof(what), "FTQ_VR round trip %dx%d nm%d", rows, cols, nm);
			REQUIRE_EQ(ftq_vr_decodes(hq, gq, format, q8, rows, cols, 1, out, what), 0, "%s decodes bit for bit", what);
			double err = 0, energy = 0;
			for (i = 0; i < count; i++)
			{
				float o;
				ccv_half_precision_to_float(out + i, &o, 1);
				err += ((double)o - values[i]) * ((double)o - values[i]);
				energy += (double)values[i] * values[i];
			}
			printf("\n\tFTQ_VR %dx%d nm %d relative error %.6f", rows, cols, nm, err / energy);
			REQUIRE(err / energy < bounds[s][m], "%s relative error %g within %g", what, err / energy, bounds[s][m]);
			// The same rate everywhere is static-rate FTQ at nm: the code words are identical.
			for (i = 0; i < rows / 16; i++)
				rates[i] = nm;
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, 0, hq->data.u8, size), size, "one rate quantizes");
			const size_t static_size = ccv_nnc_8i_rowwise_x_data_size(ftq_format(nm), CCV_16F, count, cols);
			uint8_t* const fixed = (uint8_t*)ccmalloc(static_size);
			REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, ftq_format(nm), 0, 0, 0, 0, fixed, static_size), static_size, "static rate quantizes");
			REQUIRE_EQ(memcmp(hq->data.u8 + 32 + 4 * (rows / 16 + 1), fixed + 32, (size_t)count / 256 * 2 * nm), 0, "one rate gives the static-rate code");
			REQUIRE_EQ(memcmp(hq->data.u8 + size - 2 * rows, fixed + static_size - 2 * rows, 2 * rows), 0, "and its scales");
			ccfree(fixed);
			ccfree(out);
			ccfree(q8);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
			ccfree(half);
			ccfree(values);
			ccfree(rates);
		}
}

TEST_CASE("ftq_vr quantize rejects rates outside the format")
{
	enum { rows = 48, cols = 256, count = rows * cols };
	static uint16_t half[count];
	static uint8_t output[count * 4];
	const int format = ftq_vr_format(20);
	const size_t size = ccv_nnc_8i_rowwise_x_data_size(format, CCV_16F, count, cols);
	static const int bad[][3] = { { 20, 20, 21 }, { 6, 20, 20 }, { 50, 8, 8 }, { 24, 24, 20 } }; // odd, below 8, above 48, over budget
	int i;
	memset(output, 0xa5, sizeof(output));
	for (i = 0; i < 4; i++)
		REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, bad[i], output, size), 0, "rates %d must fail", i);
	static uint16_t big[112 * 256];
	static const int levels[7] = { 8, 10, 12, 14, 16, 18, 20 }; // 7 levels
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(big, CCV_16F, CCV_TENSOR_CPU_MEMORY, 112 * 256, 256, format, levels, output, sizeof(output)), 0, "7 levels must fail");
	for (i = 0; i < (int)sizeof(output); i++)
		REQUIRE_EQ(output[i], 0xa5, "rejected calls leave the output untouched");
	static const int ok[3] = { 8, 20, 32 };
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x_ftq_vr(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, cols, format, ok, output, size), size, "8, 20, 32 quantizes");
}

// Each rule of CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N16 broken once: the checkpoint fails to load and the CPU decode returns -1.
TEST_CASE("an ftq_vr checkpoint loads only when every rule of the format holds")
{
	// off = 0, 128, 320, 448 words; the budget is R C nm / 2 = 480.
	enum { rows = 48, cols = 256, tile_rows = 3, tiles = 16 };
	const int format = ftq_vr_format(20);
	static const int rates[tile_rows] = { 16, 24, 16 };
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	enum {
		VALID, BAD_MAGIC, BAD_ROWS, BAD_COLS, BAD_MEAN, RATE_NOT_IN_MASK, SEVEN_LEVELS, DECREASING, OVER_BUDGET,
		NOT_MULTIPLE_OF_C, ZERO_RATE, NONZERO_PADDING, BAD_TABLE, NONZERO_FIRST_OFFSET, LEVEL_ABOVE_48, CASES
	};
	static const char* const names[CASES] = {
		"valid", "bad magic", "rows mismatch", "cols mismatch", "16 nm mismatch", "a rate not in the level mask",
		"more than 6 levels", "decreasing offsets", "off[R] over the budget", "a delta not a multiple of C",
		"a delta of 0 (n_r = 0)", "non-zero padding", "id 0 with a crc", "off[0] != 0", "a level above 48 in the mask",
	};
	int c;
	for (c = 0; c < CASES; c++)
	{
		ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(t->info);
		ftq_vr_fill(t->data.u8, size, format, rows, cols, rates, ftq_builtin_table, 0, 0x5eedu + c);
		uint8_t* const d = t->data.u8;
		uint32_t* const header = (uint32_t*)(d + 8);
		uint32_t* const off = (uint32_t*)(d + 32);
		const size_t code_end = 32 + 4 * (tile_rows + 1) + 4 * (size_t)off[tile_rows];
		switch (c)
		{
			case BAD_MAGIC: d[7] = 'X'; break;
			case BAD_ROWS: header[0] = rows + 16; break;
			case BAD_COLS: header[1] = cols + 256; break;
			case BAD_MEAN: header[2] = 16 * 22; break;
			case RATE_NOT_IN_MASK: header[3] &= ~(1u << ((24 - 8) / 2)); break; // tile row 1 has n = 24
			case SEVEN_LEVELS: header[3] |= 0x7fu; break; // 8 ... 20 and 24: 8 levels
			case DECREASING: off[2] = off[1] - tiles; break;
			case OVER_BUDGET: header[3] |= 1u << ((48 - 8) / 2); off[3] = off[2] + tiles * 48 / 2; break; // n_2 = 48, 88 > 60
			case NOT_MULTIPLE_OF_C: off[1] += tiles / 2; break; // deltas 136 and 184, multiples of C / 2 only
			case ZERO_RATE: off[1] = 0; break; // tile row 0 has delta 0
			case NONZERO_PADDING: d[code_end] = 1; break;
			case BAD_TABLE: header[5] = 1; break;
			case NONZERO_FIRST_OFFSET: off[0] = tiles; break;
			case LEVEL_ABOVE_48: header[3] |= 1u << ((50 - 8) / 2); break; // no tile row takes it
		}
		char name[32];
		snprintf(name, sizeof(name), "w%d", c);
		REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(d, size, format, rows, cols, 0), c == VALID ? 0 : -1, "%s: check", names[c]);
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(d, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols), c == VALID ? 0 : -1, "%s: CPU decode", names[c]);
		ccfree(out);
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "%s writes", names[c]);
		ccv_nnc_tensor_free(t);
		ccv_nnc_tensor_t* restored = 0;
		const int result = ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored);
		if (c == VALID)
		{
			REQUIRE_EQ(result, CCV_IO_FINAL, "%s loads", names[c]);
			ccv_nnc_tensor_free(restored);
		} else
			REQUIRE(result != CCV_IO_FINAL, "%s fails to load", names[c]);
	}
	sqlite3_close(db);
}

// A 48 x 256 static-rate (n20) or variable-rate (nm20, tile rows at 16, 24, 16) CPU tensor, filled with the built-in
// table, and its blob size.
static ccv_nnc_tensor_t* ftq_rule_tensor_new(const int vr, const uint32_t seed, size_t* const size)
{
	static const int rates[3] = { 16, 24, 16 };
	const int rows = 48, cols = 256;
	const int format = vr ? ftq_vr_format(20) : ftq_format(20);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
	*size = ccv_nnc_tensor_data_size_without_padding(t->info);
	if (vr)
		ftq_vr_fill(t->data.u8, *size, format, rows, cols, rates, ftq_builtin_table, 0, seed);
	else
		ftq_fill(t->data.u8, *size, format, rows, cols, ftq_builtin_table, 0, seed);
	return t;
}

static int ftq_rule_check(const int vr, const uint8_t* const data, const size_t length)
{
	return vr ? ccv_nnc_8i_rowwise_x_ftq_vr_check(data, length, ftq_vr_format(20), 48, 256, 0) : ccv_nnc_8i_rowwise_x_ftq_load_check(data, length, ftq_format(20), 48, 256);
}

// Row scales (CCV_NNC_QX_8I_ROWWISE_FTQ_N16, FTQ_VR_N16; QuantStar ftq-spec-20261117, II and VII.5): every FP16 row
// scale is finite and not negative; -0 passes as 0, so do +0 and the smallest subnormal. A NaN, +Inf or -Inf scale (or
// a negative one) fails the blob check, the CPU decode and the checkpoint load, for static and variable rate alike.
TEST_CASE("ftq and ftq_vr reject a non-finite row scale in the check, the cpu decode and the checkpoint load")
{
	static const struct { const char* name; uint16_t h; int ok; } scales[] = {
		{ "the filled scale", 0, 1 }, { "+0", 0x0000, 1 }, { "-0", 0x8000, 1 }, { "the smallest subnormal", 0x0001, 1 },
		{ "the largest finite", 0x7bff, 1 }, { "NaN", 0x7e00, 0 }, { "a NaN with payload", 0x7c01, 0 }, { "-NaN", 0xfe00, 0 },
		{ "+Inf", 0x7c00, 0 }, { "-Inf", 0xfc00, 0 }, { "-1", 0xbc00, 0 },
	};
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	int vr, c;
	for (vr = 0; vr < 2; vr++)
		for (c = 0; c < (int)(sizeof(scales) / sizeof(scales[0])); c++)
		{
			size_t size;
			ccv_nnc_tensor_t* const t = ftq_rule_tensor_new(vr, 0x5ca1u + c, &size);
			const char* const kind = vr ? "ftq_vr" : "ftq";
			// Row 17 (tile row 1, the n24 one for variable rate); its 2 bytes are at scale_offset + 34.
			if (c > 0)
				memcpy(t->data.u8 + size - 2 * 48 + 2 * 17, &scales[c].h, 2);
			REQUIRE_EQ(ftq_rule_check(vr, t->data.u8, size), scales[c].ok ? 0 : -1, "%s %s: check", kind, scales[c].name);
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * 48 * 256);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(t->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, 256, t->info.reserved, out, 48 * 256), scales[c].ok ? 0 : -1, "%s %s: CPU decode", kind, scales[c].name);
			ccfree(out);
			char name[32];
			snprintf(name, sizeof(name), "s%d_%d", vr, c);
			REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "%s %s writes", kind, scales[c].name);
			ccv_nnc_tensor_free(t);
			ccv_nnc_tensor_t* restored = 0;
			const int result = ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored);
			if (scales[c].ok)
			{
				REQUIRE_EQ(result, CCV_IO_FINAL, "%s %s loads", kind, scales[c].name);
				ccv_nnc_tensor_free(restored);
			} else
				REQUIRE(result != CCV_IO_FINAL && !restored, "%s %s fails to load", kind, scales[c].name);
		}
	sqlite3_close(db);
}

// Blob length: the stored blob of an FTQ / FTQ-VR checkpoint must be exactly the size its shape and rate give
// (QuantStar: the total length matches the header). One byte more or less fails the load and the blob checks. The CPU
// decode takes a buffer (possibly a padded tensor) and still decodes from one of at least that size.
TEST_CASE("ftq and ftq_vr checkpoints load only when the stored blob is exactly the size of the format")
{
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	int vr, d;
	for (vr = 0; vr < 2; vr++)
	{
		const char* const kind = vr ? "ftq_vr" : "ftq";
		size_t size;
		ccv_nnc_tensor_t* const t = ftq_rule_tensor_new(vr, 0x1e9u + vr, &size);
		REQUIRE_EQ(size, ccv_nnc_8i_rowwise_x_data_size(t->info.reserved, CCV_16F, 48 * 256, 256), "%s blob size is the writer's", kind);
		uint8_t* const longer = (uint8_t*)ccmalloc(size + 1);
		memcpy(longer, t->data.u8, size);
		longer[size] = 0;
		REQUIRE_EQ(ftq_rule_check(vr, t->data.u8, size), 0, "%s exact size passes the check", kind);
		REQUIRE_EQ(ftq_rule_check(vr, longer, size + 1), -1, "%s one byte more fails the check", kind);
		REQUIRE_EQ(ftq_rule_check(vr, t->data.u8, size - 1), -1, "%s one byte less fails the check", kind);
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * 48 * 256);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(longer, CCV_16F, CCV_TENSOR_CPU_MEMORY, size + 1, 256, t->info.reserved, out, 48 * 256), 0, "%s CPU decode takes a larger buffer", kind);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(t->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size - 1, 256, t->info.reserved, out, 48 * 256), -1, "%s CPU decode rejects a short buffer", kind);
		ccfree(out);
		// -1, 0, +1 bytes: write the valid tensor, then replace its stored blob.
		for (d = -1; d <= 1; d++)
		{
			char name[32];
			snprintf(name, sizeof(name), "l%d_%d", vr, d + 1);
			REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "%s writes", kind);
			sqlite3_stmt* update = 0;
			REQUIRE_EQ(sqlite3_prepare_v2(db, "UPDATE tensors SET data=?1 WHERE name=?2", -1, &update, 0), SQLITE_OK, "update prepares");
			sqlite3_bind_blob(update, 1, longer, (int)(size + d), SQLITE_TRANSIENT);
			sqlite3_bind_text(update, 2, name, -1, SQLITE_TRANSIENT);
			REQUIRE_EQ(sqlite3_step(update), SQLITE_DONE, "%s blob of %d bytes more is stored", kind, d);
			sqlite3_finalize(update);
			ccv_nnc_tensor_t* restored = 0;
			const int result = ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored);
			if (d == 0)
			{
				REQUIRE_EQ(result, CCV_IO_FINAL, "%s exact blob loads", kind);
				REQUIRE(memcmp(restored->data.u8, t->data.u8, size) == 0, "%s exact blob reads back", kind);
				ccv_nnc_tensor_free(restored);
			} else
				REQUIRE(result != CCV_IO_FINAL && !restored, "%s blob of %d bytes more fails to load", kind, d);
		}
		ccfree(longer);
		ccv_nnc_tensor_free(t);
	}
	sqlite3_close(db);
}

// The whole static-rate container is checked when a checkpoint loads, as for variable rate: header shape and schedule,
// the table and its CRC, the zero padding after the code and the row scales. Each broken blob replaces the stored data
// of a valid tensor and fails to load; the valid one loads.
TEST_CASE("an ftq checkpoint loads only when its whole container holds")
{
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	size_t size;
	ccv_nnc_tensor_t* const t = ftq_rule_tensor_new(0, 0xc0ffu, &size);
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(t->info.reserved, 48, 256);
	// Byte offset and value written there; offset -1 is the valid blob.
	const struct { const char* name; int offset; uint8_t value; } cases[] = {
		{ "the valid blob", -1, 0 }, { "rows", 8, 47 }, { "cols", 12, 255 }, { "table", 24, 1 }, { "table crc", 28, 1 },
		{ "padding", 32 + (int)(schedule.code_bits / 8), 1 }, { "the last padding byte", (int)schedule.scale_offset - 1, 0x80 },
	};
	REQUIRE(32 + schedule.code_bits / 8 < schedule.scale_offset, "padding follows the code");
	uint8_t* const work = (uint8_t*)ccmalloc(size);
	int c;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		memcpy(work, t->data.u8, size);
		if (cases[c].offset >= 0)
			work[cases[c].offset] = cases[c].value;
		REQUIRE_EQ(ftq_rule_check(0, work, size), cases[c].offset < 0 ? 0 : -1, "ftq with %s: the load check", cases[c].name);
		char name[32];
		snprintf(name, sizeof(name), "c%d", c);
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "ftq writes");
		sqlite3_stmt* update = 0;
		REQUIRE_EQ(sqlite3_prepare_v2(db, "UPDATE tensors SET data=?1 WHERE name=?2", -1, &update, 0), SQLITE_OK, "update prepares");
		sqlite3_bind_blob(update, 1, work, (int)size, SQLITE_TRANSIENT);
		sqlite3_bind_text(update, 2, name, -1, SQLITE_TRANSIENT);
		REQUIRE_EQ(sqlite3_step(update), SQLITE_DONE, "ftq with %s is stored", cases[c].name);
		sqlite3_finalize(update);
		ccv_nnc_tensor_t* restored = 0;
		const int result = ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored);
		if (cases[c].offset < 0)
		{
			REQUIRE_EQ(result, CCV_IO_FINAL, "ftq with %s loads", cases[c].name);
			REQUIRE(memcmp(restored->data.u8, work, size) == 0, "ftq with %s reads back", cases[c].name);
			ccv_nnc_tensor_free(restored);
		} else
			REQUIRE(result != CCV_IO_FINAL && !restored, "ftq with %s fails to load", cases[c].name);
	}
	// An FP32 base is the same blob with the same FP16 scales; it writes and reads back too.
	ccv_nnc_tensor_param_t f32_params = t->info;
	f32_params.datatype = ((CCV_32F >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X;
	REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(f32_params), size, "the FP32 base takes the same blob");
	ccv_nnc_tensor_t* const t32 = ccv_nnc_tensor_new(0, f32_params, 0);
	memcpy(t32->data.u8, t->data.u8, size);
	REQUIRE_EQ(ccv_nnc_tensor_write(t32, db, "f32", 0), CCV_IO_FINAL, "the FP32 base writes");
	ccv_nnc_tensor_t* restored = 0;
	REQUIRE_EQ(ccv_nnc_tensor_read(db, "f32", 0, 0, 0, &restored), CCV_IO_FINAL, "the FP32 base loads");
	REQUIRE_EQ(restored->info.datatype, f32_params.datatype, "the FP32 base keeps its datatype");
	REQUIRE(memcmp(restored->data.u8, t->data.u8, size) == 0, "the FP32 base reads back");
	ccv_nnc_tensor_free(restored);
	ccv_nnc_tensor_free(t32);
	ccfree(work);
	ccv_nnc_tensor_free(t);
	sqlite3_close(db);
}

// A folded-table blob (slot with CCV_NNC_CODEBOOK_FOLD, table field 1 | 9 << 8 | 1 << 16) at rate n against the explicit
// 2048-pair reference: CPU decode, GPU int8 codes, the fused GEMV for M = 1 .. 3 (one-hot and bit for bit with onehot,
// else within FP16 rounding of an FP64 sum) and the GEMM for M = 4, 8 and 64 equal to the GEMM on resident 8I_ROWWISE of
// the reference codes, with and without the ANE path. Returns the number of failed checks (each printed).
static int ftq_fold_rate_checks(const int slot, const uint32_t crc, const int8_t* const base, const int rows, const int cols, const int n, const int onehot, const uint32_t seed)
{
	int failures = 0;
	const uint64_t old_flags = ccv_nnc_flags();
	const int format = ftq_format(n) | (slot << CCV_NNC_CODEBOOK_SLOT_SHIFT);
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), format), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(hq->info);
	ftq_fill(hq->data.u8, size, format, rows, cols, 1u | (9u << 8) | (1u << 16), crc, seed);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
	int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
	ftq_ref_decode_fold(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + schedule.scale_offset), scales, rows);
	// CPU decode.
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	FTQ_VR_EXPECT(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols) == 0, "fold %dx%d n%d CPU decode runs", rows, cols, n);
	size_t k, mismatches = 0;
	for (k = 0; k < (size_t)rows * cols; k++)
	{
		const float v = (float)q8[k] * scales[k / cols];
		uint16_t e;
		ccv_float_to_half_precision(&v, &e, 1);
		mismatches += out[k] != e;
	}
	ccfree(out);
	FTQ_VR_EXPECT(mismatches == 0, "fold %dx%d n%d CPU decode, %zu mismatches", rows, cols, n, mismatches);
	// GPU int8 decode.
	const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
	const size_t out_bytes = out_scale_offset + rows * 2;
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
	FTQ_VR_EXPECT(memcmp(hout->data.u8, q8, (size_t)rows * cols) == 0, "fold %dx%d n%d GPU codes", rows, cols, n);
	FTQ_VR_EXPECT(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2) == 0, "fold %dx%d n%d GPU scales", rows, cols, n);
	ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
	bias_params.dim[1] = 0;
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
	bias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	float* const bias = (float*)ccmalloc(sizeof(float) * rows);
	int i, j, m;
	for (j = 0; j < rows; j++)
	{
		const float v = ftq_bias_value(j);
		ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
	}
	ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
	if (onehot)
		for (m = 1; m <= 3; m++)
			FTQ_VR_EXPECT(ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m) == 0, "fold %dx%d n%d GEMV M=%d one-hot", rows, cols, n, m);
	ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
	memset(hr->data.u8, 0, out_scale_offset);
	memcpy(hr->data.u8, q8, (size_t)rows * cols);
	memcpy(hr->data.u8 + out_scale_offset, hq->data.u8 + schedule.scale_offset, rows * 2);
	ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
	// M = 1 .. 3 without onehot: the fused GEMV within rounding of an FP64 sum; M = 4, 8, 64: the GEMM equals the GEMM on
	// the resident reference codes, both passes. The reference GEMM takes at least 9 rows (the first M are the same), so
	// it never runs the 8I_ROWWISE vector kernel, which sums in another order.
	static const int ms[] = { 1, 2, 3, 4, 8, 64 };
	int pass, mi;
	for (pass = 0; pass < 2; pass++)
	{
		if (pass == 1)
			ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ANE);
		for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
		{
			const int mm = ms[mi];
			if (mm < 4 && (onehot || pass == 1))
				continue;
			const int mr = mm < 9 ? 9 : mm;
			ccv_nnc_tensor_t* const har = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
			for (i = 0; i < mr; i++)
				for (k = 0; k < (size_t)cols; k++)
				{
					const float v = ftq_activation_value(i, (int)k);
					ccv_float_to_half_precision(&v, (uint16_t*)har->data.u8 + (size_t)i * cols + k, 1);
				}
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, cols), 0);
			memcpy(ha->data.u8, har->data.u8, (size_t)mm * cols * 2);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, cols), 0);
			ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
			ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mm, rows), 0);
			ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
			ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mm, rows), 0);
			ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, har), TENSOR_LIST(ga, gar), 0);
			FTQ_VR_EXPECT(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0) == CCV_NNC_EXEC_SUCCESS, "fold %dx%d n%d GEMM M=%d runs", rows, cols, n, mm);
			if (mm >= 4)
			{
				FTQ_VR_EXPECT(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0) == CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM M=%d runs", mr);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
				FTQ_VR_EXPECT(memcmp(hb1->data.u8, hb2->data.u8, (size_t)mm * rows * 2) == 0, "fold %dx%d n%d pass %d GEMM M=%d equals 8I_ROWWISE", rows, cols, n, pass, mm);
			} else {
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1), TENSOR_LIST(hb1), 0);
				float* const f1 = (float*)ccmalloc(sizeof(float) * mm * rows);
				ccv_half_precision_to_float((uint16_t*)hb1->data.u8, f1, (size_t)mm * rows);
				float* const fa = (float*)ccmalloc(sizeof(float) * mm * cols);
				ccv_half_precision_to_float((uint16_t*)ha->data.u8, fa, (size_t)mm * cols);
				int bad = 0;
				for (i = 0; i < mm && !bad; i++)
					for (j = 0; j < rows && !bad; j++)
					{
						double sum = 0, magnitude = 0;
						for (k = 0; k < (size_t)cols; k++)
						{
							const double term = (double)q8[(size_t)j * cols + k] * fa[(size_t)i * cols + k];
							sum += term;
							magnitude += fabs(term);
						}
						const double expected = sum * scales[j] + bias[j];
						if (fabs((double)f1[(size_t)i * rows + j] - expected) > fabs(expected) * (2.0 / 2048) + magnitude * scales[j] * 1e-6 + 1e-6)
							bad = 1;
					}
				ccfree(fa);
				ccfree(f1);
				FTQ_VR_EXPECT(!bad, "fold %dx%d n%d GEMV M=%d within rounding of the FP64 sum", rows, cols, n, mm);
			}
			ccv_nnc_tensor_free(hb2);
			ccv_nnc_tensor_free(hb1);
			ccv_nnc_tensor_free(gb2);
			ccv_nnc_tensor_free(gb1);
			ccv_nnc_tensor_free(gar);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(ha);
			ccv_nnc_tensor_free(har);
		}
	}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_ANE))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_ANE);
	ccfree(bias);
	ccfree(scales);
	ccfree(q8);
	ccv_nnc_tensor_free(gr);
	ccv_nnc_tensor_free(hr);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
	return failures;
}

TEST_CASE("ftq with a folded table is exact at every open static rate, and at the 4B size for n16, n20 and n24")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0x0f41u);
	const int slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(slot >= 1, "folded table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(slot, 0, &crc);
	int n;
	// 3 tile rows x 32 tiles, every rate n = 16, 18, .., 48, one-hot GEMV.
	for (n = 16; n <= 48; n += 2)
		REQUIRE_EQ(ftq_fold_rate_checks(slot, crc, base, 48, 512, n, 1, 0x4100u + n), 0, "fold 48x512 n%d", n);
	static const int large[] = { 16, 20, 24 };
	int i;
	for (i = 0; i < 3; i++)
		REQUIRE_EQ(ftq_fold_rate_checks(slot, crc, base, 9216, 2560, large[i], 0, 0x4200u + large[i]), 0, "fold 9216x2560 n%d", large[i]);
	ccfree(base);
}

TEST_CASE("an ftq checkpoint with the folded table inline loads and decodes like the reference at n16 and n20")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0x0f51u);
	const int plain = ccv_nnc_codebook_register(base, 512);
	REQUIRE(plain >= 1, "base table registers");
	uint32_t crc = 0;
	ccv_nnc_codebook_get(plain, 0, &crc);
	sqlite3* db = 0;
	REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
	const int rows = 64, cols = 1024;
	static const int rates[] = { 16, 20 };
	int i;
	for (i = 0; i < 2; i++)
	{
		const int n = rates[i];
		// The base table inline; the header names id 1 (not looked up) and the quadrant fold.
		const int file_format = ccv_nnc_8i_rowwise_x_with_inline_codebook(ftq_format(n), 512);
		ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), file_format), 0);
		const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(file_format);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(t->info) - prefix;
		memset(t->data.u8, 0, prefix);
		REQUIRE_EQ(ccv_nnc_codebook_serialize(plain, t->data.u8, prefix), 16 + 1024, "the base table serializes");
		ftq_fill(t->data.u8 + prefix, size, ftq_format(n), rows, cols, 1u | (9u << 8) | (1u << 16), crc, 0x5100u + n);
		char name[16];
		snprintf(name, sizeof(name), "w_n%d", n);
		REQUIRE_EQ(ccv_nnc_tensor_write(t, db, name, 0), CCV_IO_FINAL, "%s writes", name);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(ftq_format(n), rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_fold(t->data.u8 + prefix + 32, rows, cols, schedule.high_bits, schedule.drop_count, base, q8);
		float* const scales = (float*)ccmalloc(sizeof(float) * rows);
		ccv_half_precision_to_float((uint16_t*)(t->data.u8 + prefix + schedule.scale_offset), scales, rows);
		ccv_nnc_tensor_free(t);
		ccv_nnc_tensor_t* restored = 0;
		REQUIRE_EQ(ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored), CCV_IO_FINAL, "%s reads", name);
		REQUIRE_EQ(restored->info.reserved, file_format, "%s keeps the inline bits", name);
		const int format = restored->info.reserved;
		const size_t rsize = ccv_nnc_tensor_data_size_without_padding(restored->info);
		const int slot = ccv_nnc_codebook_slot_of(ccv_nnc_8i_rowwise_x_resolve_inline(restored->data.u8, rsize, format));
		REQUIRE_EQ(ccv_nnc_codebook_flags(slot), CCV_NNC_CODEBOOK_FOLD, "%s takes a folded slot", name);
		REQUIRE_EQ(ccv_nnc_codebook_fold_slot(plain), slot, "%s takes the folded slot of the inline table", name);
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(restored->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, rsize, cols, format, out, (size_t)rows * cols), 0, "%s decodes", name);
		size_t k, mismatches = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
		{
			const float v = (float)q8[k] * scales[k / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			mismatches += out[k] != e;
		}
		REQUIRE_EQ((int)mismatches, 0, "%s CPU decode equals the reference", name);
		// The loaded weights on the GPU: int8 codes equal the reference.
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols), format), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(restored), TENSOR_LIST(gq), 0);
		const size_t out_bytes = (((size_t)rows * cols + 127) & ~(size_t)127) + rows * 2;
		ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
		ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, rsize, cols, format, gout->data.u8, (size_t)rows * cols, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)rows * cols), 0, "%s GPU codes equal the reference", name);
		ccv_nnc_tensor_free(hout);
		ccv_nnc_tensor_free(gout);
		ccv_nnc_tensor_free(gq);
		ccfree(out);
		ccfree(scales);
		ccfree(q8);
		ccv_nnc_tensor_free(restored);
	}
	sqlite3_close(db);
	ccfree(base);
}

// FTQ and FTQ_VR at K % 16 (rows % 16): K = 1152 (72 tiles a tile row), 272 (17, odd), 16 and 48 (1 and 3), 96 (6:
// the variable-rate blocks of 2 T = 4 tiles do not tile it), 288 (18), 784 (49). The blobs are laid out on the tile grid
// here, so the Metal paths run before the CPU codec takes these shapes (its tensor constructor and container checks
// take K % 256 until then); every CPU comparison is skipped, and counted, while the CPU size is not the tile grid's.

// FP16 FTQ weights [rows, cols]; see tmcq_x_params in tmcq_mps.tests.c.
static ccv_nnc_tensor_param_t ftq_x_params(const int type, const int rows, const int cols, const int format)
{
	if (cols % 256 == 0)
		return ccv_nnc_tensor_8i_rowwise_x(ftq_params(type, rows, cols), format);
	ccv_nnc_tensor_param_t params = ftq_params(type, rows, cols);
	params.datatype = ((CCV_16F >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X;
	params.reserved = format;
	return params;
}

// The blob size on the tile grid: static rate n has the 32-byte header and (rows / 16) (cols / 16) tiles of 2 n bytes;
// variable rate nm the header, the rows / 16 + 1 word offsets and at most that many tiles at 2 nm bytes; then the FP16
// scales at the next 128 bytes.
static size_t ftq_k16_size(const int format, const int rows, const int cols)
{
	const size_t tiles = (size_t)(rows / 16) * (cols / 16);
	const size_t code = ccv_nnc_8i_rowwise_x_is_ftq_vr(format) ?
		32 + 4 * (size_t)(rows / 16 + 1) + tiles * 2 * ccv_nnc_8i_rowwise_x_ftq_vr_rate(format) :
		32 + tiles * 2 * ccv_nnc_8i_rowwise_x_ftq_rate(format);
	return ((code + 127) & ~(size_t)127) + 2 * (size_t)rows;
}

static int ftq_k16_cpu_takes(const int format, const int rows, const int cols)
{
	return ccv_nnc_8i_rowwise_x_data_size(ccv_nnc_8i_rowwise_x_codec(format), CCV_16F, (size_t)rows * cols, cols) == ftq_k16_size(format, rows, cols);
}

// The static-rate schedule (ccv_nnc_ftq_schedule.inc) counts the tiles of the grid: the Metal descriptors check it
// against their own tile count, so the static-rate shapes wait for it.
static int ftq_k16_schedule_takes(const int format, const int rows, const int cols)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	return schedule.tile_count == (uint64_t)(rows / 16) * (cols / 16) && schedule.scale_offset + 2 * (size_t)rows == ftq_k16_size(format, rows, cols);
}

// GPU int8 decode (codes and scales), the one-hot fused GEMV for M = 1 .. 3 when gemv, the GEMM from 4 rows against
// the GEMM on the reference codes as resident 8I_ROWWISE when gemm, and the CPU decode when the CPU takes the shape;
// all bit for bit. Returns the number of failed checks (each printed); *cpu_skipped counts the skipped CPU decodes.
static int ftq_k16_checks(ccv_nnc_tensor_t* const hq, ccv_nnc_tensor_t* const gq, const int format, const int8_t* const q8, const int rows, const int cols, const int gemv, const int gemm, int* const cpu_skipped, const char* const what)
{
	int failures = 0;
	const size_t size = ftq_k16_size(format, rows, cols);
	const size_t scale_offset = size - 2 * (size_t)rows;
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + scale_offset), scales, rows);
	if (ftq_k16_cpu_takes(format, rows, cols))
	{
		uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
		FTQ_VR_EXPECT(ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols) == 0, "%s CPU decode runs", what);
		size_t k, mismatches = 0;
		for (k = 0; k < (size_t)rows * cols; k++)
		{
			const float v = (float)q8[k] * scales[k / cols];
			uint16_t e;
			ccv_float_to_half_precision(&v, &e, 1);
			mismatches += out[k] != e;
		}
		FTQ_VR_EXPECT(mismatches == 0, "%s CPU decode, %zu mismatches", what, mismatches);
		ccfree(out);
	} else
		++*cpu_skipped;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
	const size_t out_scale_offset = ((size_t)rows * cols + 127) & ~(size_t)127;
	const size_t out_bytes = out_scale_offset + rows * 2;
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
	size_t k, code_mismatches = 0, first = (size_t)-1;
	for (k = 0; k < (size_t)rows * cols; k++)
		if ((int8_t)hout->data.u8[k] != q8[k])
		{
			if (first == (size_t)-1)
				first = k;
			code_mismatches++;
		}
	FTQ_VR_EXPECT(code_mismatches == 0, "%s GPU int8 codes, %zu mismatches, first at row %zu col %zu", what, code_mismatches, first == (size_t)-1 ? 0 : first / cols, first == (size_t)-1 ? 0 : first % cols);
	FTQ_VR_EXPECT(memcmp(hout->data.u8 + out_scale_offset, hq->data.u8 + scale_offset, rows * 2) == 0, "%s GPU int8 scales", what);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
	bias_params.dim[1] = 0;
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
	bias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	float* const bias = (float*)ccmalloc(sizeof(float) * rows);
	int j, m;
	for (j = 0; j < rows; j++)
	{
		const float v = ftq_bias_value(j);
		ccv_float_to_half_precision(&v, (uint16_t*)hbias->data.u8 + j, 1);
	}
	ccv_half_precision_to_float((uint16_t*)hbias->data.u8, bias, rows);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
	if (gemv)
		for (m = 1; m <= 3; m++)
		{
			const int gemv_mismatches = ftq_onehot_mismatches(gq, gbias, q8, scales, bias, rows, cols, m);
			FTQ_VR_EXPECT(gemv_mismatches == 0, "%s GEMV M=%d one-hot, %d mismatches (-1: did not run)", what, m, gemv_mismatches);
		}
	if (gemm)
	{
		// Resident 8I_ROWWISE weights from the reference codes and the blob's scales; it keeps its GEMV up to 8 rows,
		// so it runs on at least 9 rows (independent in the integer GEMM) and the first M rows are compared.
		ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols)), 0);
		const size_t r_scale = ((size_t)rows * cols + 127) & ~(size_t)127;
		memset(hr->data.u8, 0, r_scale);
		memcpy(hr->data.u8, q8, (size_t)rows * cols);
		memcpy(hr->data.u8 + r_scale, hq->data.u8 + scale_offset, rows * 2);
		ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(ftq_params(CCV_TENSOR_GPU_MEMORY, rows, cols)), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
		static const int ms[] = { 4, 9, 65 };
		int mi, with_bias, i;
		for (mi = 0; mi < 3; mi++)
		{
			const int m = ms[mi];
			const int mr = m < 9 ? 9 : m;
			ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, cols), 0);
			for (i = 0; i < mr; i++)
				for (k = 0; k < (size_t)cols; k++)
				{
					const float v = ftq_activation_value(i, (int)k);
					ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
				}
			ccv_nnc_tensor_t* const ham = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
			memcpy(ham->data.u8, ha->data.u8, (size_t)m * cols * 2);
			ccv_nnc_tensor_t* const gar = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, cols), 0);
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
			ccv_nnc_tensor_t* const gb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const gb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, mr, rows), 0);
			ccv_nnc_tensor_t* const hb1 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
			ccv_nnc_tensor_t* const hb2 = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, mr, rows), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ham, ha), TENSOR_LIST(ga, gar), 0);
			for (with_bias = 0; with_bias < 2; with_bias++)
			{
				int r1, r2;
				if (with_bias)
				{
					r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb1), 0);
					r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr, gbias), TENSOR_LIST(gb2), 0);
				} else {
					r1 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb1), 0);
					r2 = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(gar, gr), TENSOR_LIST(gb2), 0);
				}
				FTQ_VR_EXPECT(r1 == CCV_NNC_EXEC_SUCCESS && r2 == CCV_NNC_EXEC_SUCCESS, "%s GEMM M=%d bias=%d runs (%d, %d)", what, m, with_bias, r1, r2);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb1, gb2), TENSOR_LIST(hb1, hb2), 0);
				FTQ_VR_EXPECT(memcmp(hb1->data.u8, hb2->data.u8, (size_t)m * rows * 2) == 0, "%s GEMM M=%d bias=%d equals the 8I_ROWWISE GEMM", what, m, with_bias);
			}
			ccv_nnc_tensor_free(hb2);
			ccv_nnc_tensor_free(hb1);
			ccv_nnc_tensor_free(gb2);
			ccv_nnc_tensor_free(gb1);
			ccv_nnc_tensor_free(ga);
			ccv_nnc_tensor_free(gar);
			ccv_nnc_tensor_free(ham);
			ccv_nnc_tensor_free(ha);
		}
		ccv_nnc_tensor_free(gr);
		ccv_nnc_tensor_free(hr);
	}
	ccfree(bias);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(hbias);
	ccfree(scales);
	return failures;
}

TEST_CASE("mps ftq static rate is exact at K % 16 on int8 decode, fused gemv and gemm, at n16, n18, n32, n46 and n48")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// (rows, cols, n, gemv, gemm).
	static const int cases[][5] = {
		{ 2304, 1152, 16, 0, 1 }, { 2304, 1152, 48, 0, 1 }, { 64, 1152, 18, 1, 0 },
		{ 48, 272, 16, 1, 1 }, { 48, 272, 18, 1, 0 }, { 48, 272, 48, 1, 1 }, { 48, 272, 46, 1, 0 },
		{ 32, 16, 16, 1, 1 }, { 16, 48, 32, 1, 0 }, { 32, 96, 48, 1, 0 }, { 32, 288, 46, 1, 1 }, { 16, 784, 18, 1, 0 },
	};
	int c, cpu_skipped = 0, schedule_skipped = 0;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
	{
		const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2];
		const int format = ftq_format(n);
		if (!ftq_k16_schedule_takes(format, rows, cols))
		{
			schedule_skipped++;
			continue;
		}
		ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ftq_x_params(CCV_TENSOR_CPU_MEMORY, rows, cols, format), 0);
		ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ftq_x_params(CCV_TENSOR_GPU_MEMORY, rows, cols, format), 0);
		const size_t size = ftq_k16_size(format, rows, cols);
		REQUIRE(ccv_nnc_tensor_data_size_without_padding(hq->info) >= size, "FTQ %dx%d n%d: the tensor holds the tile grid", rows, cols, n);
		ftq_fill(hq->data.u8, size, format, rows, cols, ftq_builtin_table, 0, 0x16f0u + c);
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
		int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
		ftq_ref_decode_matrix(hq->data.u8 + 32, rows, cols, schedule.high_bits, schedule.drop_count, ccv_nnc_trellis_builtin_table, 9, q8);
		char what[128];
		snprintf(what, sizeof(what), "FTQ %dx%d n%d", rows, cols, n);
		REQUIRE_EQ(ftq_k16_checks(hq, gq, format, q8, rows, cols, cases[c][3], cases[c][4], &cpu_skipped, what), 0, "%s decodes bit for bit", what);
		ccfree(q8);
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	if (schedule_skipped)
		printf("[ftq k16] the static-rate schedule does not count K %% 16 tiles yet: %d shapes skipped\n", schedule_skipped);
	if (cpu_skipped)
		printf("[ftq k16] the CPU codec does not take K %% 16 yet: %d CPU decodes skipped\n", cpu_skipped);
}

TEST_CASE("mps ftq_vr is exact at K % 16 on int8 decode, fused gemv and gemm, with the built-in and a folded table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	int8_t* const base = ftq_fold_base_new(0x7a16u);
	const int fold_slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	REQUIRE(fold_slot >= 1, "folded table registers");
	uint32_t fold_crc = 0;
	ccv_nnc_codebook_get(fold_slot, 0, &fold_crc);
	// (rows, cols, gemv, gemm). The folded table decodes one block a thread below 2^20 row segments, two from there
	// (16384 x 1040: 65 tiles a row, 2^20 + 16384 segments); the built-in table always two.
	static const int shapes[][4] = {
		{ 2304, 1152, 0, 1 }, { 48, 272, 1, 1 }, { 16, 16, 1, 1 }, { 32, 48, 1, 0 }, { 16, 96, 1, 0 },
		{ 32, 288, 1, 1 }, { 48, 784, 1, 0 }, { 16384, 1040, 0, 0 },
	};
	static const int means[] = { 20, 32 };
	int s, m, folded, cpu_skipped = 0;
	for (s = 0; s < (int)(sizeof(shapes) / sizeof(shapes[0])); s++)
		for (m = 0; m < 2; m++)
			for (folded = 0; folded < 2; folded++)
			{
				const int rows = shapes[s][0], cols = shapes[s][1], nm = means[m];
				if (rows > 1024 && nm != 20)
					continue;
				const int format = ftq_vr_format(nm) | ((folded ? fold_slot : 0) << CCV_NNC_CODEBOOK_SLOT_SHIFT);
				int* const rates = (int*)ccmalloc(sizeof(int) * (rows / 16));
				ftq_vr_rates(rows / 16, nm, rates);
				ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ftq_x_params(CCV_TENSOR_CPU_MEMORY, rows, cols, format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ftq_x_params(CCV_TENSOR_GPU_MEMORY, rows, cols, format), 0);
				const size_t size = ftq_k16_size(format, rows, cols);
				REQUIRE(ccv_nnc_tensor_data_size_without_padding(hq->info) >= size, "FTQ_VR %dx%d nm%d: the tensor holds the tile grid", rows, cols, nm);
				ftq_vr_fill(hq->data.u8, size, format, rows, cols, rates, folded ? 1u | (9u << 8) | (1u << 16) : ftq_builtin_table, folded ? fold_crc : 0, 0x7a16u + s * 8 + m * 2 + folded);
				if (ftq_k16_cpu_takes(format, rows, cols))
				{
					REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(hq->data.u8, size, format, rows, cols, 0), 0, "the filled blob passes the checks");
				}
				int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * cols);
				ftq_vr_ref_decode(hq->data.u8, rows, cols, folded ? base : ccv_nnc_trellis_builtin_table, folded, q8);
				char what[128];
				snprintf(what, sizeof(what), "FTQ_VR %dx%d nm%d fold %d", rows, cols, nm, folded);
				REQUIRE_EQ(ftq_k16_checks(hq, gq, format, q8, rows, cols, shapes[s][2], shapes[s][3] && !folded, &cpu_skipped, what), 0, "%s decodes bit for bit", what);
				ccfree(q8);
				ccfree(rates);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(hq);
			}
	ccfree(base);
	if (cpu_skipped)
		printf("[ftq_vr k16] the CPU codec does not take K %% 16 yet: %d CPU decodes and container checks skipped\n", cpu_skipped);
}

// Inline codebooks (CCV_NNC_QX_8I_ROWWISE_CODEBOOK_INLINE_MASK) on Metal: the same FTQ / FTQ_VR blob once naming its
// table's registered slot and once stored with its table inline (the serialized table, zeros up to 128 + 2 P bytes,
// then the blob) and read back from a checkpoint. Moved to the GPU, the two run every Metal path bit for bit alike. The
// blob header keeps id 1, which is never looked up.
// ccv_nnc_mps.h declares these for Objective-C only (the buffer is an id<MTLBuffer>).
extern int ccv_nnc_mps_8i_rowwise_x_resolve_inline(void* buffer, const off_t offset, const int format);
extern const ccv_nnc_tensor_t* ccv_nnc_mps_8i_rowwise_x_inline_view(const ccv_nnc_tensor_t* const tensor, ccv_nnc_tensor_t* const view);

typedef struct {
	int slot; // The table the blob decodes with (folded for the folded table).
	int entries;
	uint32_t field; // The header table field: id 1, log2 P, structure.
	uint32_t crc;
} ftq_inline_table_t;

static ftq_inline_table_t ftq_inline_table_of(const int slot)
{
	ftq_inline_table_t t = { .slot = slot };
	ccv_nnc_codebook_get(slot, &t.entries, &t.crc);
	const int folded = !!(ccv_nnc_codebook_flags(slot) & CCV_NNC_CODEBOOK_FOLD);
	t.field = 1u | ((uint32_t)ccv_nnc_codebook_log2_entries(t.entries) << 8) | ((uint32_t)folded << 16);
	return t;
}

// The tables: the built-in table's content registered again, random 512, 1024 and 2048-pair tables, and a folded table.
static void ftq_inline_tables(ftq_inline_table_t* const tables)
{
	const int builtin = ccv_nnc_codebook_register(ccv_nnc_trellis_builtin_table, 512);
	tables[0] = ftq_inline_table_of(builtin);
	int i;
	for (i = 0; i < 3; i++)
	{
		ftq_table_t t = ftq_table_new(9 + i, 0, 0x1a11eu + i);
		tables[1 + i] = ftq_inline_table_of(t.slot);
		ccfree(t.pairs);
	}
	int8_t* const base = ftq_fold_base_new(0x1a1f0u);
	const int folded = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	tables[4] = ftq_inline_table_of(folded);
	ccfree(base);
}

// The inline form of an FTQ / FTQ_VR weight of base format `format` (no slot) whose blob is blob: params of `type`.
static ccv_nnc_tensor_t* ftq_inline_tensor_new(const int type, const int rows, const int cols, const int format, const int slot, const uint8_t* const blob, const size_t blob_size)
{
	int entries = 0;
	ccv_nnc_codebook_get(slot, &entries, 0);
	const int inline_format = ccv_nnc_8i_rowwise_x_with_inline_codebook(format, entries);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), inline_format), 0);
	const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(inline_format);
	assert(ccv_nnc_tensor_data_size_without_padding(t->info) == prefix + blob_size);
	memset(t->data.u8, 0, prefix);
	const size_t serialized = ccv_nnc_codebook_serialize(slot, t->data.u8, prefix);
	assert(serialized == 16 + 2 * (size_t)entries);
	(void)serialized;
	memcpy(t->data.u8 + prefix, blob, blob_size);
	if (CCV_TENSOR_GET_MEMORY(type) == CCV_TENSOR_CPU_MEMORY)
		return t;
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(type, rows, cols), inline_format), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(t), TENSOR_LIST(g), 0);
	ccv_nnc_tensor_free(t);
	return g;
}

static ccv_nnc_tensor_t* ftq_inline_to_gpu(const ccv_nnc_tensor_t* const host)
{
	ccv_nnc_tensor_param_t params = host->info;
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)host), TENSOR_LIST(g), 0);
	return g;
}

// GEMM_FORWARD of m rows of fixed activations against gq (with or without bias) into out (m x rows FP16). Returns the
// exec status.
static int ftq_inline_gemm(ccv_nnc_tensor_t* const gq, ccv_nnc_tensor_t* const gbias, const int m, const int rows, const int cols, uint16_t* const out)
{
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, cols), 0);
	int i, k;
	for (i = 0; i < m; i++)
		for (k = 0; k < cols; k++)
		{
			const float v = ftq_activation_value(i * 3 + 1, k);
			ccv_float_to_half_precision(&v, (uint16_t*)ha->data.u8 + (size_t)i * cols + k, 1);
		}
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, m, rows), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_CPU_MEMORY, m, rows), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
	const int status = gbias ?
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0) :
		ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
	if (status == CCV_NNC_EXEC_SUCCESS && out)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
		memcpy(out, hb->data.u8, sizeof(uint16_t) * m * rows);
	}
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(ha);
	return status;
}

// The Metal int8 decode (codes and scales) of a GPU weight of data size `size` and format `format`, out_bytes bytes.
static void ftq_inline_int8(ccv_nnc_tensor_t* const gq, const size_t size, const int format, const int rows, const int cols, uint8_t* const out, const size_t out_bytes)
{
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, CCV_16F, size, cols, format, gout->data.u8, (size_t)rows * cols, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
	memcpy(out, hout->data.u8, out_bytes);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gout);
}

// Writes t to name in db and reads it back on the CPU; NULL if either fails.
static ccv_nnc_tensor_t* ftq_inline_round_trip(sqlite3* const db, const char* const name, ccv_nnc_tensor_t* const t)
{
	if (ccv_nnc_tensor_write(t, db, name, 0) != CCV_IO_FINAL)
		return 0;
	ccv_nnc_tensor_t* restored = 0;
	if (ccv_nnc_tensor_read(db, name, 0, 0, 0, &restored) != CCV_IO_FINAL)
		return 0;
	return restored;
}

TEST_CASE("mps ftq and ftq_vr with an inline codebook equal the registered slot bit for bit on int8 decode, fused gemv and gemm")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	ftq_inline_table_t tables[5];
	ftq_inline_tables(tables);
	int ti;
	for (ti = 0; ti < 5; ti++)
		REQUIRE(tables[ti].slot >= 1, "table %d registers", ti);
	REQUIRE(ccv_nnc_codebook_flags(tables[4].slot) == CCV_NNC_CODEBOOK_FOLD, "table 4 is folded");
	// (rows, cols, static rate n or FTQ_VR mean nm, vr): FTQ at n20, n36 and n48; FTQ_VR at nm20 (rates 8 - 32), nm40
	// and nm48 (tile row rates up to 48).
	static const int cases[][4] = { { 64, 1024, 20, 0 }, { 272, 512, 48, 0 }, { 48, 768, 36, 0 }, { 208, 768, 20, 1 }, { 96, 512, 40, 1 }, { 48, 1024, 48, 1 } };
	int c, compared = 0;
	for (c = 0; c < (int)(sizeof(cases) / sizeof(cases[0])); c++)
		for (ti = 0; ti < 5; ti++)
		{
			const int rows = cases[c][0], cols = cases[c][1], n = cases[c][2], vr = cases[c][3];
			// FTQ_VR runs the built-in table's content, a random 512-pair table and the folded table.
			if (vr && (ti == 2 || ti == 3))
				continue;
			const ftq_inline_table_t t = tables[ti];
			const int base = vr ? ftq_vr_format(n) : ftq_format(n);
			// The slot form names the table's registered slot.
			const int file_format = ccv_nnc_codebook_with_slot(base, t.slot);
			ccv_nnc_tensor_t* const hfile = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), file_format), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(hfile->info);
			const uint32_t seed = 0x1a1700u + c * 16 + ti;
			int* rates = 0;
			if (vr)
			{
				rates = (int*)ccmalloc(sizeof(int) * (rows / 16));
				ftq_vr_rates(rows / 16, n, rates);
				ftq_vr_fill(hfile->data.u8, size, base, rows, cols, rates, t.field, t.crc, seed);
			} else
				ftq_fill(hfile->data.u8, size, base, rows, cols, t.field, t.crc, seed);
			ccv_nnc_tensor_t* const hinline = ftq_inline_tensor_new(CCV_TENSOR_CPU_MEMORY, rows, cols, base, t.slot, hfile->data.u8, size);
			const int inline_format = hinline->info.reserved;
			sqlite3* inline_db = 0;
			REQUIRE_EQ(sqlite3_open(":memory:", &inline_db), SQLITE_OK, "in-memory database opens");
			char what[96];
			snprintf(what, sizeof(what), "%s %dx%d n%d table %d", vr ? "FTQ_VR" : "FTQ", rows, cols, n, ti);
			// A process slot means nothing in a file: only the inline form goes into the checkpoint.
			REQUIRE_EQ(ccv_nnc_tensor_write(hfile, inline_db, "slot", 0), CCV_IO_ERROR, "%s: the slot form is not written", what);
			ccv_nnc_tensor_t* const rfile = hfile;
			ccv_nnc_tensor_t* const rinline = ftq_inline_round_trip(inline_db, "w", hinline);
			REQUIRE(rinline, "%s: the inline form writes and reads", what);
			REQUIRE_EQ(rinline->info.reserved, inline_format, "%s: the inline form keeps the inline bits and slot 255", what);
			const int file_slot_format = rfile->info.reserved;
			const size_t inline_size = ccv_nnc_tensor_data_size_without_padding(rinline->info);
			REQUIRE_EQ(inline_size, ccv_nnc_8i_rowwise_x_inline_codebook_size(inline_format) + size, "%s: the inline form is the prefix and the blob", what);
			ccv_nnc_tensor_t* const gfile = ftq_inline_to_gpu(rfile);
			ccv_nnc_tensor_t* const ginline = ftq_inline_to_gpu(rinline);
			// The Metal view of the inline weight: the data after the prefix, the table's slot.
			ccv_nnc_tensor_t view;
			const ccv_nnc_tensor_t* const v = ccv_nnc_mps_8i_rowwise_x_inline_view(ginline, &view);
			REQUIRE(v == &view, "%s: the inline weight has a view", what);
			REQUIRE_EQ(view.info.reserved, file_slot_format, "%s: its view names the table's slot", what);
			REQUIRE_EQ((size_t)view.dataof, ccv_nnc_8i_rowwise_x_inline_codebook_size(inline_format), "%s: its view starts after the prefix", what);
			REQUIRE_EQ(ccv_nnc_mps_8i_rowwise_x_resolve_inline(ginline->data.u8, 0, inline_format), file_slot_format, "%s: the buffer resolves to the same slot", what);
			// int8 decode.
			const size_t out_bytes = (((size_t)rows * cols + 127) & ~(size_t)127) + rows * 2;
			uint8_t* const codes_file = (uint8_t*)ccmalloc(out_bytes);
			uint8_t* const codes_inline = (uint8_t*)ccmalloc(out_bytes);
			ftq_inline_int8(gfile, size, file_slot_format, rows, cols, codes_file, out_bytes);
			ftq_inline_int8(ginline, inline_size, inline_format, rows, cols, codes_inline, out_bytes);
			REQUIRE_EQ(memcmp(codes_file, codes_inline, out_bytes), 0, "%s: int8 codes and scales are the same", what);
			// CPU decode of the inline form.
			uint16_t* const cpu_file = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			uint16_t* const cpu_inline = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(rfile->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, file_slot_format, cpu_file, (size_t)rows * cols), 0, "%s: the slot form decodes on the CPU", what);
			REQUIRE_EQ(ccv_nnc_dequantize_8i_rowwise_x(rinline->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, inline_size, cols, inline_format, cpu_inline, (size_t)rows * cols), 0, "%s: the inline form decodes on the CPU", what);
			REQUIRE_EQ(memcmp(cpu_file, cpu_inline, sizeof(uint16_t) * rows * cols), 0, "%s: CPU decodes are the same", what);
			ccfree(cpu_inline);
			ccfree(cpu_file);
			// GEMM_FORWARD: the fused GEMV for 1 to 3 rows, from 4 rows the decode and the quantized GEMM; with a bias
			// for odd M.
			ccv_nnc_tensor_param_t bias_params = ftq_params(CCV_TENSOR_CPU_MEMORY, rows, 0);
			bias_params.dim[1] = 0;
			ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
			bias_params.type = CCV_TENSOR_GPU_MEMORY;
			ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
			int j;
			for (j = 0; j < rows; j++)
			{
				const float b = ftq_bias_value(j);
				ccv_float_to_half_precision(&b, (uint16_t*)hbias->data.u8 + j, 1);
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hbias), TENSOR_LIST(gbias), 0);
			static const int ms[] = { 1, 2, 3, 4, 9, 64 };
			int mi;
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
			{
				const int m = ms[mi];
				uint16_t* const out_file = (uint16_t*)ccmalloc(sizeof(uint16_t) * m * rows);
				uint16_t* const out_inline = (uint16_t*)ccmalloc(sizeof(uint16_t) * m * rows);
				REQUIRE_EQ(ftq_inline_gemm(gfile, (m & 1) ? gbias : 0, m, rows, cols, out_file), CCV_NNC_EXEC_SUCCESS, "%s: slot form GEMM M=%d runs", what, m);
				REQUIRE_EQ(ftq_inline_gemm(ginline, (m & 1) ? gbias : 0, m, rows, cols, out_inline), CCV_NNC_EXEC_SUCCESS, "%s: inline form GEMM M=%d runs", what, m);
				REQUIRE_EQ(memcmp(out_file, out_inline, sizeof(uint16_t) * m * rows), 0, "%s: GEMM M=%d outputs are the same", what, m);
				ccfree(out_inline);
				ccfree(out_file);
			}
			compared++;
			ccv_nnc_tensor_free(gbias);
			ccv_nnc_tensor_free(hbias);
			ccfree(codes_inline);
			ccfree(codes_file);
			ccv_nnc_tensor_free(ginline);
			ccv_nnc_tensor_free(gfile);
			ccv_nnc_tensor_free(rinline);
			sqlite3_close(inline_db);
			ccv_nnc_tensor_free(hinline);
			ccv_nnc_tensor_free(hfile);
			if (rates)
				ccfree(rates);
		}
	REQUIRE_EQ(compared, 24, "every case and table ran");
}

// A file holding the data of t, padded to its GPU size, as an on-demand file-backed GPU tensor (pread path).
static ccv_nnc_tensor_t* ftq_inline_file_backed(const ccv_nnc_tensor_t* const t, char path[64])
{
	strcpy(path, "/tmp/ccv-ftq-inline-XXXXXX");
	const int fd = mkstemp(path);
	if (fd < 0)
		return 0;
	const size_t data_size = ccv_nnc_tensor_data_size_without_padding(t->info);
	size_t written = 0;
	while (written < data_size)
	{
		const ssize_t result = write(fd, t->data.u8 + written, data_size - written);
		if (result <= 0)
			break;
		written += (size_t)result;
	}
	ccv_nnc_tensor_param_t params = t->info;
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	const int ok = written == data_size && ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(params)) == 0;
	close(fd);
	return ok ? ccv_nnc_tensor_new_from_file(params, path, 0, CCV_NNC_TENSOR_MEMORY_MAP_ON_DEMAND) : 0;
}

TEST_CASE("mps ftq and ftq_vr reject an inline codebook that the blob header does not name, never looking the header id up")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	enum { rows = 64, cols = 512 };
	ftq_inline_table_t tables[5];
	ftq_inline_tables(tables);
	// A registered table whose process slot is the header id of the "decoy id" mutation, with its crc and P: a reader
	// that looked the header id up would find it and accept the blob.
	ftq_table_t decoy = ftq_table_new(9, 0, 0xdec1u);
	REQUIRE(decoy.slot >= 1 && decoy.slot < 255, "the decoy registers");
	// A plain 512-pair table with negative values (not a valid folded table) and a 1024-pair table.
	const ftq_inline_table_t plain = tables[1], wide = tables[2], folded = tables[4];
	enum {
		GOOD, ID_0, DECOY_ID, LOG2, CRC, OTHER_CRC, STRUCTURE_2, STRUCTURE_3, FOLD_NEGATIVE, FOLD_1024, HIGH_BITS, LEGACY_MAGIC, BLOB_MAGIC, MUTATIONS,
	};
	static const char* const names[MUTATIONS] = {
		"good", "header id 0", "header id and crc of a registered decoy", "header log2 P + 1", "header crc + 1", "header crc of another table", "structure 2", "structure 3",
		"structure 1 on a table with negative values", "structure 1 on a 1024-pair table", "table word bit 18", "prefix magic CCVCB001", "blob magic of the other FTQ",
	};
	int vr, mu, checked = 0;
	for (vr = 0; vr < 2; vr++)
		for (mu = 0; mu < MUTATIONS; mu++)
		{
			const ftq_inline_table_t t = mu == FOLD_NEGATIVE ? plain : mu == FOLD_1024 ? wide : mu == STRUCTURE_2 || mu == STRUCTURE_3 ? folded : (mu & 1) ? folded : plain;
			const int base = vr ? ftq_vr_format(20) : ftq_format(20);
			ccv_nnc_tensor_t* const blob = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), base), 0);
			const size_t size = ccv_nnc_tensor_data_size_without_padding(blob->info);
			uint32_t field = t.field, crc = t.crc;
			switch (mu)
			{
				case ID_0: field &= ~0xffu; break;
				case DECOY_ID: field = (uint32_t)decoy.slot | (9u << 8); crc = decoy.crc; break;
				case LOG2: field += 1u << 8; break;
				case CRC: crc += 1; break;
				case OTHER_CRC: crc = t.slot == plain.slot ? folded.crc : plain.crc; break;
				case STRUCTURE_2: field = (field & ~(3u << 16)) | (2u << 16); break;
				case STRUCTURE_3: field |= 3u << 16; break;
				case FOLD_NEGATIVE: case FOLD_1024: field |= 1u << 16; break;
				case HIGH_BITS: field |= 1u << 18; break;
			}
			int rates[rows / 16];
			if (vr)
			{
				ftq_vr_rates(rows / 16, 20, rates);
				ftq_vr_fill(blob->data.u8, size, base, rows, cols, rates, field, crc, 0x1a1e00u + mu);
			} else
				ftq_fill(blob->data.u8, size, base, rows, cols, field, crc, 0x1a1e00u + mu);
			if (mu == BLOB_MAGIC)
				memcpy(blob->data.u8, vr ? "CCVFTQSR" : "CCVFTQVR", 8);
			ccv_nnc_tensor_t* const hq = ftq_inline_tensor_new(CCV_TENSOR_CPU_MEMORY, rows, cols, base, t.slot, blob->data.u8, size);
			if (mu == LEGACY_MAGIC)
				memcpy(hq->data.u8, "CCVCB001", 8);
			const int format = hq->info.reserved;
			const size_t data_size = ccv_nnc_tensor_data_size_without_padding(hq->info);
			char what[128];
			snprintf(what, sizeof(what), "%s inline %s", vr ? "FTQ_VR" : "FTQ", names[mu]);
			const int good = mu == GOOD;
			ccv_nnc_tensor_t* const gq = ftq_inline_to_gpu(hq);
			// The resolver that the int8 decode takes (it asserts on -1), the view every command takes, the CPU decode.
			const int resolved = ccv_nnc_mps_8i_rowwise_x_resolve_inline(gq->data.u8, 0, format);
			REQUIRE(good ? resolved >= 0 && ccv_nnc_codebook_slot_of(resolved) == t.slot : resolved == -1, "%s: the buffer %s", what, good ? "resolves to the table's slot" : "does not resolve");
			ccv_nnc_tensor_t view;
			REQUIRE((ccv_nnc_mps_8i_rowwise_x_inline_view(gq, &view) != 0) == good, "%s: the view %s", what, good ? "exists" : "is NULL");
			uint16_t* const cpu = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
			REQUIRE((ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, CCV_16F, CCV_TENSOR_CPU_MEMORY, data_size, cols, format, cpu, (size_t)rows * cols) == 0) == good, "%s: the CPU decode %s", what, good ? "runs" : "fails");
			ccfree(cpu);
			// GEMM_FORWARD through the fused GEMV and through the decode + quantized GEMM.
			uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * 64 * rows);
			uint16_t* const out_file_backed = (uint16_t*)ccmalloc(sizeof(uint16_t) * 64 * rows);
			REQUIRE_EQ(ftq_inline_gemm(gq, 0, 1, rows, cols, out), good ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID, "%s: GEMV %s", what, good ? "runs" : "is rejected");
			REQUIRE_EQ(ftq_inline_gemm(gq, 0, 64, rows, cols, 0), good ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID, "%s: GEMM %s", what, good ? "runs" : "is rejected");
			// The same bytes on demand from a file, read with pread (twice: the second takes the cached slot).
			char path[64];
			ccv_nnc_tensor_t* const fq = ftq_inline_file_backed(hq, path);
			REQUIRE(fq, "%s: maps on demand", what);
			int pass;
			for (pass = 0; pass < 2; pass++)
			{
				REQUIRE_EQ(ftq_inline_gemm(fq, 0, 1, rows, cols, out_file_backed), good ? CCV_NNC_EXEC_SUCCESS : CCV_NNC_EXEC_INVALID, "%s: file-backed GEMV pass %d %s", what, pass, good ? "runs" : "is rejected");
				if (good)
					REQUIRE_EQ(memcmp(out, out_file_backed, sizeof(uint16_t) * rows), 0, "%s: file-backed GEMV pass %d equals the GEMV in memory", what, pass);
			}
			ccv_nnc_tensor_free(fq);
			unlink(path);
			ccfree(out_file_backed);
			ccfree(out);
			// A checkpoint: the read fails unless the blob is good.
			sqlite3* db = 0;
			REQUIRE_EQ(sqlite3_open(":memory:", &db), SQLITE_OK, "in-memory database opens");
			REQUIRE_EQ(ccv_nnc_tensor_write(hq, db, "w", 0), CCV_IO_FINAL, "%s writes", what);
			ccv_nnc_tensor_t* restored = 0;
			REQUIRE_EQ(ccv_nnc_tensor_read(db, "w", 0, 0, 0, &restored), good ? CCV_IO_FINAL : CCV_IO_ERROR, "%s: the read %s", what, good ? "succeeds" : "fails");
			if (restored)
				ccv_nnc_tensor_free(restored);
			sqlite3_close(db);
			checked++;
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
			ccv_nnc_tensor_free(blob);
		}
	REQUIRE_EQ(checked, 2 * MUTATIONS, "every mutation ran");
	// IndexSelect and the dense SwiGLU take no FTQ weight, also not an inline one.
	{
		ccv_nnc_tensor_t* const blob = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(ftq_params(CCV_TENSOR_CPU_MEMORY, rows, cols), ftq_format(20)), 0);
		const size_t size = ccv_nnc_tensor_data_size_without_padding(blob->info);
		ftq_fill(blob->data.u8, size, ftq_format(20), rows, cols, plain.field, plain.crc, 0x1a1effu);
		ccv_nnc_tensor_t* const gq = ftq_inline_tensor_new(CCV_TENSOR_GPU_MEMORY, rows, cols, ftq_format(20), plain.slot, blob->data.u8, size);
		if (ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
		{
			ccv_nnc_tensor_t* const hindices = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1), 0);
			hindices->data.i32[0] = 3;
			ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1), 0);
			ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 1, cols), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hindices), TENSOR_LIST(gindices), 0);
			REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gq, gindices), TENSOR_LIST(gout), 0), CCV_NNC_EXEC_INVALID, "an inline FTQ weight does not support IndexSelect");
			ccv_nnc_tensor_free(gout);
			ccv_nnc_tensor_free(gindices);
			ccv_nnc_tensor_free(hindices);
		}
		if (ccv_nnc_cmd_ok(CCV_NNC_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS))
		{
			ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, cols), 0);
			ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, ftq_params(CCV_TENSOR_GPU_MEMORY, 1, rows), 0);
			ccv_nnc_cmd_t swiglu = CMD_SWIGLU_FORWARD(10);
			swiglu.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "the dense SwiGLU takes no inline FTQ weight");
			ccv_nnc_tensor_free(gb);
			ccv_nnc_tensor_free(ga);
		}
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(blob);
	}
	ccfree(decoy.pairs);
}

#endif

#include "case_main.h"
