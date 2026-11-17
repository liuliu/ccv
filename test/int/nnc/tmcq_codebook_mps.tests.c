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
#include <nnc/ccv_nnc_trellis_builtin_table.inc>
#include "trellis_table_alt.inc"

// External pair codebooks (ccv_nnc_codebook.h) on the Metal TMCQ paths. A weight names its codebook slot in
// bits 16 - 23 of its format (reserved field). Two kinds of tests:
// 1. A table that repeats each pair of a 512-pair table rep = P / 512 times selects, with shift 32 - log2 P, exactly
//    that table's pair of every state, so every GPU path must give the same bits with that slot as with slot 0. Slot
//    0 is the built-in table, so the tests repeat it.
// 2. A random table (P = 512, 1024, 2048): the GPU int8 decode, the TMCQ fp decode, the CPU decode and the one-hot
//    fused GEMV must match a reference decode written here from the format definition with that table.

TEST_SETUP()
{
	ccv_nnc_init();
}

static uint32_t rng_next(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

typedef struct {
	const int8_t* pairs;
	int shift;
} ref_table_t;

static ref_table_t ref_table(const int slot)
{
	int entries = 0;
	ref_table_t table;
	table.pairs = ccv_nnc_codebook_get(slot, &entries, 0);
	table.shift = 32 - ccv_nnc_codebook_log2_entries(entries);
	return table;
}

static int ref_q8(const ref_table_t table, const uint32_t state, const int half)
{
	return table.pairs[2 * ((state * 0x83DCD12Du) >> table.shift) + half];
}

static int register_repeated(const int8_t* const base, const int entries)
{
	int8_t* const pairs = (int8_t*)ccmalloc((size_t)entries * 2);
	const int rep = entries / 512;
	int i;
	for (i = 0; i < entries; i++)
	{
		pairs[2 * i] = base[2 * (i / rep)];
		pairs[2 * i + 1] = base[2 * (i / rep) + 1];
	}
	const int slot = ccv_nnc_codebook_register(pairs, entries);
	ccfree(pairs);
	return slot;
}

static int register_random(const int entries, uint32_t seed)
{
	int8_t* const pairs = (int8_t*)ccmalloc((size_t)entries * 2);
	int i;
	for (i = 0; i < entries * 2; i++)
		pairs[i] = (int8_t)((int)(rng_next(&seed) % 255) - 127);
	const int slot = ccv_nnc_codebook_register(pairs, entries);
	ccfree(pairs);
	return slot;
}

static ccv_nnc_tensor_param_t params_of(const int type, const int datatype, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { rows, cols, 0 },
	};
	return params;
}

// TMCQ: tile g holds 16 n bits from bit 16 n g; weight t reads the 15-bit window of weight t | 1, ending at
// ((t | 1) + 1) n / 16 and wrapping in the tile, then half t & 1 of its pair.
static void tmcq_ref_decode(const uint8_t* const payload, const int rows, const int cols, const int n, const ref_table_t table, int8_t* const q8)
{
	const int strips = cols / 16;
	const size_t T = (size_t)16 * n;
	int tr, tc, t, j;
	for (tr = 0; tr < rows / 16; tr++)
		for (tc = 0; tc < strips; tc++)
			for (t = 0; t < 256; t++)
			{
				const size_t base = ((size_t)tr * strips + tc) * T;
				const size_t end = ((size_t)((t | 1) + 1) * n) >> 4;
				uint32_t cw = 0;
				for (j = 0; j < 15; j++)
				{
					const size_t p = base + (end + T - 15 + j) % T;
					cw |= (uint32_t)((payload[p >> 3] >> (p & 7)) & 1) << j;
				}
				q8[(size_t)(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = (int8_t)ref_q8(table, cw, t & 1);
			}
}

// Random TMCQ payload and FP16 / FP32 row scales.
static void tmcq_fill(ccv_nnc_tensor_t* const q, const int datatype, const int rows, const int cols, const int n, uint32_t seed)
{
	const size_t payload = ((size_t)rows * (cols / 256) * 16 * n + 7) / 8;
	const size_t scale_offset = (payload + 127) & ~(size_t)127;
	memset(q->data.u8, 0, ccv_nnc_tensor_data_size_without_padding(q->info));
	size_t i;
	for (i = 0; i < payload; i++)
		q->data.u8[i] = (uint8_t)(rng_next(&seed) >> 24);
	int r;
	for (r = 0; r < rows; r++)
	{
		const float s = (float)((rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
		if (datatype == CCV_32F)
			((float*)(q->data.u8 + scale_offset))[r] = s;
		else
			ccv_float_to_half_precision((float*)&s, (uint16_t*)(q->data.u8 + scale_offset) + r, 1);
	}
}

typedef struct {
	int format; // Codec only.
	int rows, cols;
	int datatype;
} weight_case_t;

static int rate_of(const int format)
{
	return ccv_nnc_8i_rowwise_x_tmcq_rate(format);
}

// The format of a case naming codebook slot `slot`.
static int format_with_slot(const int format, const int slot)
{
	return ccv_nnc_codebook_with_slot(format, slot);
}

// CPU and GPU weights of one case, filled with the same bytes, naming codebook slot `slot`.
static void make_weights(const weight_case_t w, const int slot, const uint32_t seed, ccv_nnc_tensor_t** const hq, ccv_nnc_tensor_t** const gq)
{
	const int format = format_with_slot(w.format, slot);
	*hq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, w.rows, w.cols), format), 0);
	*gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(params_of(CCV_TENSOR_GPU_MEMORY, w.datatype, w.rows, w.cols), format), 0);
	tmcq_fill(*hq, w.datatype, w.rows, w.cols, rate_of(w.format), seed);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(*hq), TENSOR_LIST(*gq), 0);
}

static void ref_decode(const weight_case_t w, const ccv_nnc_tensor_t* const hq, const ref_table_t table, int8_t* const q8)
{
	tmcq_ref_decode(hq->data.u8, w.rows, w.cols, rate_of(w.format), table, q8);
}

static float activation_value(const int row, const int k)
{
	return (float)(((row * 37 + k * 11) % 33) - 16) / 128.0f;
}

// GPU GEMM B = A W^T (+ bias) for m activation rows; returns the exec status and the output bytes in *out.
static int run_gemm(const weight_case_t w, ccv_nnc_tensor_t* const gq, const int m, const int with_bias, uint8_t** const out)
{
	const size_t esize = CCV_GET_DATA_TYPE_SIZE(w.datatype);
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, m, w.cols), 0);
	int i, k;
	for (i = 0; i < m; i++)
		for (k = 0; k < w.cols; k++)
		{
			const float v = activation_value(i, k);
			if (w.datatype == CCV_32F)
				ha->data.f32[(size_t)i * w.cols + k] = v;
			else
				ccv_float_to_half_precision((float*)&v, (uint16_t*)ha->data.u8 + (size_t)i * w.cols + k, 1);
		}
	ccv_nnc_tensor_param_t bias_params = params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, w.rows, 0);
	bias_params.dim[1] = 0;
	ccv_nnc_tensor_t* const hbias = ccv_nnc_tensor_new(0, bias_params, 0);
	for (i = 0; i < w.rows; i++)
	{
		const float v = (float)((i * 7) % 13 - 6) / 64.0f;
		if (w.datatype == CCV_32F)
			hbias->data.f32[i] = v;
		else
			ccv_float_to_half_precision((float*)&v, (uint16_t*)hbias->data.u8 + i, 1);
	}
	bias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const gbias = ccv_nnc_tensor_new(0, bias_params, 0);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_GPU_MEMORY, w.datatype, m, w.cols), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_GPU_MEMORY, w.datatype, m, w.rows), 0);
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, m, w.rows), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hbias), TENSOR_LIST(ga, gbias), 0);
	int status;
	if (with_bias)
		status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq, gbias), TENSOR_LIST(gb), 0);
	else
		status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
	*out = (uint8_t*)ccmalloc((size_t)m * w.rows * esize);
	memcpy(*out, hb->data.u8, (size_t)m * w.rows * esize);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gbias);
	ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(ha);
	return status;
}

static const weight_case_t weight_cases[] = {
	{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N20, 64, 512, CCV_16F },
	{ CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, 48, 768, CCV_32F },
};

TEST_CASE("mps tmcq with a repeated slot-0 codebook equals slot 0 bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int entries[] = { 1024, 2048 };
	// 1 - 3 rows: the fused GEMV; 4 - 512: the int8 decode and the GEMM (or the ANE).
	static const int ms[] = { 1, 2, 3, 4, 9, 64, 512 };
	int e, c, mi, with_bias;
	for (e = 0; e < 2; e++)
	{
		const int builtin_slot = register_repeated(ccv_nnc_trellis_builtin_table, entries[e]);
		REQUIRE(builtin_slot > 0, "registers the repeated built-in table of %d pairs", entries[e]);
		for (c = 0; c < (int)(sizeof(weight_cases) / sizeof(weight_cases[0])); c++)
		{
			const weight_case_t w = weight_cases[c];
			const int slot = builtin_slot;
			ccv_nnc_tensor_t *h0, *g0, *h1, *g1;
			make_weights(w, 0, 0x5150u + c, &h0, &g0);
			make_weights(w, slot, 0x5150u + c, &h1, &g1);
			for (mi = 0; mi < (int)(sizeof(ms) / sizeof(ms[0])); mi++)
				for (with_bias = 0; with_bias < 2; with_bias++)
				{
					uint8_t *b0, *b1;
					const int s0 = run_gemm(w, g0, ms[mi], with_bias, &b0);
					const int s1 = run_gemm(w, g1, ms[mi], with_bias, &b1);
					REQUIRE_EQ(s0, CCV_NNC_EXEC_SUCCESS, "built-in table GEMM runs (format %d, M %d)", w.format, ms[mi]);
					REQUIRE_EQ(s1, CCV_NNC_EXEC_SUCCESS, "codebook GEMM runs (format %d, P %d, M %d)", w.format, entries[e], ms[mi]);
					REQUIRE_EQ(memcmp(b0, b1, (size_t)ms[mi] * w.rows * CCV_GET_DATA_TYPE_SIZE(w.datatype)), 0, "format %d %dx%d P %d M %d bias %d: codebook output equals the built-in output", w.format, w.rows, w.cols, entries[e], ms[mi], with_bias);
					ccfree(b0);
					ccfree(b1);
				}
			ccv_nnc_tensor_free(g1);
			ccv_nnc_tensor_free(h1);
			ccv_nnc_tensor_free(g0);
			ccv_nnc_tensor_free(h0);
		}
	}
}

TEST_CASE("mps tmcq decode with a random codebook matches the reference bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int entries[] = { 512, 1024, 2048 };
	int e, c;
	for (e = 0; e < 3; e++)
	{
		const int slot = register_random(entries[e], 0x9e3779b9u + e);
		REQUIRE(slot > 0, "registers a random table of %d pairs", entries[e]);
		const ref_table_t table = ref_table(slot);
		for (c = 0; c < (int)(sizeof(weight_cases) / sizeof(weight_cases[0])); c++)
		{
			const weight_case_t w = weight_cases[c];
			const int format = format_with_slot(w.format, slot);
			ccv_nnc_tensor_t *hq, *gq;
			make_weights(w, slot, 0x7777u + 13 * c + e, &hq, &gq);
			int8_t* const q8 = (int8_t*)ccmalloc((size_t)w.rows * w.cols);
			ref_decode(w, hq, table, q8);
			const size_t qsize = ccv_nnc_tensor_data_size_without_padding(hq->info);
			const size_t esize = CCV_GET_DATA_TYPE_SIZE(w.datatype);
			// GPU int8 decode (the path the GEMM and the ANE take from 4 rows).
			const size_t out_scale_offset = ((size_t)w.rows * w.cols + 127) & ~(size_t)127;
			const size_t out_bytes = out_scale_offset + w.rows * esize;
			ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
			ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
			ccv_nnc_mps_dequantize_8i_rowwise_x_to_8i_rowwise(gq->data.u8, w.datatype, qsize, w.cols, format, gout->data.u8, (size_t)w.rows * w.cols, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
			REQUIRE_EQ(memcmp(hout->data.u8, q8, (size_t)w.rows * w.cols), 0, "format %d P %d: GPU int8 decode matches the reference", w.format, entries[e]);
			// CPU decode, FP32 output for TMCQ FP32 scales (exact q8 * scale), FP16 otherwise.
			float* const scales = (float*)ccmalloc(sizeof(float) * w.rows);
			const size_t scale_offset = ((((size_t)w.rows * (w.cols / 256) * 16 * rate_of(w.format) + 7) / 8 + 127) & ~(size_t)127);
			int r;
			for (r = 0; r < w.rows; r++)
				scales[r] = w.datatype == CCV_32F ? ((float*)(hq->data.u8 + scale_offset))[r] : 0;
			if (w.datatype != CCV_32F)
				ccv_half_precision_to_float((uint16_t*)(hq->data.u8 + scale_offset), scales, w.rows);
			ccv_nnc_tensor_t* const hcpu = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, w.rows, w.cols), 0);
			ccv_nnc_dequantize_8i_rowwise_x(hq->data.u8, w.datatype, CCV_TENSOR_CPU_MEMORY, qsize, w.cols, format, hcpu->data.u8, (size_t)w.rows * w.cols);
			if (w.datatype == CCV_32F)
			{
				size_t i, mismatches = 0;
				for (i = 0; i < (size_t)w.rows * w.cols; i++)
					mismatches += hcpu->data.f32[i] != (float)((double)scales[i / w.cols] * q8[i]);
				REQUIRE_EQ(mismatches, 0, "format %d P %d: CPU decode matches the reference", w.format, entries[e]);
				// GPU fp decode of TMCQ.
				ccv_nnc_tensor_t* const gfp = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_GPU_MEMORY, w.datatype, w.rows, w.cols), 0);
				ccv_nnc_tensor_t* const hfp = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, w.rows, w.cols), 0);
				ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, w.datatype, CCV_TENSOR_GPU_MEMORY, qsize, w.cols, format, gfp->data.u8, (size_t)w.rows * w.cols);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gfp), TENSOR_LIST(hfp), 0);
				REQUIRE_EQ(memcmp(hfp->data.u8, hcpu->data.u8, (size_t)w.rows * w.cols * esize), 0, "format %d P %d: GPU fp decode matches the CPU decode", w.format, entries[e]);
				ccv_nnc_tensor_free(hfp);
				ccv_nnc_tensor_free(gfp);
			} else {
				// The CPU writer narrows to FP16 toward zero; compare its codes through the scale instead.
				size_t i, mismatches = 0;
				for (i = 0; i < (size_t)w.rows * w.cols; i++)
				{
					float expected = (float)((double)scales[i / w.cols] * q8[i]);
					float got;
					ccv_half_precision_to_float((uint16_t*)hcpu->data.u8 + i, &got, 1);
					mismatches += fabsf(got - expected) > fabsf(expected) * 2e-3f;
				}
				REQUIRE_EQ(mismatches, 0, "format %d P %d: CPU decode matches the reference within FP16 narrowing", w.format, entries[e]);
			}
			ccv_nnc_tensor_free(hcpu);
			// One-hot fused GEMV: every output is fp(q8 * scale) whatever the summation order.
			if (w.datatype == CCV_16F)
			{
				int m;
				for (m = 1; m <= 3; m++)
				{
					ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, CCV_16F, m, w.cols), 0);
					ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_GPU_MEMORY, CCV_16F, m, w.cols), 0);
					ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_GPU_MEMORY, CCV_16F, m, w.rows), 0);
					ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, params_of(CCV_TENSOR_CPU_MEMORY, CCV_16F, m, w.rows), 0);
					int c0, mismatches = 0;
					// Every 7th column keeps the run short; the columns cover every tile column and lane offset mod 16.
					for (c0 = 0; c0 < w.cols; c0 += 7 * m)
					{
						memset(ha->data.u8, 0, (size_t)m * w.cols * 2);
						int i, j;
						const float one = 1;
						for (i = 0; i < m; i++)
							ccv_float_to_half_precision((float*)&one, (uint16_t*)ha->data.u8 + (size_t)i * w.cols + (c0 + i) % w.cols, 1);
						ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha), TENSOR_LIST(ga), 0);
						const int status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0);
						REQUIRE_EQ(status, CCV_NNC_EXEC_SUCCESS, "format %d P %d GEMV M=%d runs", w.format, entries[e], m);
						ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
						for (i = 0; i < m; i++)
							for (j = 0; j < w.rows; j++)
							{
								float v = (float)q8[(size_t)j * w.cols + (c0 + i) % w.cols] * scales[j];
								uint16_t expected;
								ccv_float_to_half_precision(&v, &expected, 1);
								// ccv_float_to_half_precision rounds toward zero; the GPU rounds to nearest. q8 * scale
								// is exact in float, so the two can differ by one unit only.
								const int d = (int)((uint16_t*)hb->data.u8)[(size_t)i * w.rows + j] - (int)expected;
								mismatches += d < -1 || d > 1;
							}
					}
					REQUIRE_EQ(mismatches, 0, "format %d %dx%d P %d GEMV M=%d one-hot outputs", w.format, w.rows, w.cols, entries[e], m);
					ccv_nnc_tensor_free(hb);
					ccv_nnc_tensor_free(gb);
					ccv_nnc_tensor_free(ga);
					ccv_nnc_tensor_free(ha);
				}
			}
			// From 9 rows the GEMM runs on the int8 decode with the codebook: it must equal the GEMM on resident
			// 8I_ROWWISE weights holding the reference codes and the same scales (which take the GEMM from 9 rows too).
			{
				ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(params_of(CCV_TENSOR_CPU_MEMORY, w.datatype, w.rows, w.cols)), 0);
				ccv_nnc_tensor_t* const gr = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(params_of(CCV_TENSOR_GPU_MEMORY, w.datatype, w.rows, w.cols)), 0);
				memset(hr->data.u8, 0, out_scale_offset);
				memcpy(hr->data.u8, q8, (size_t)w.rows * w.cols);
				memcpy(hr->data.u8 + out_scale_offset, hq->data.u8 + scale_offset, w.rows * esize);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hr), TENSOR_LIST(gr), 0);
				static const int ms[] = { 9, 64 };
				int mi, with_bias;
				for (mi = 0; mi < 2; mi++)
					for (with_bias = 0; with_bias < 2; with_bias++)
					{
						uint8_t *b0, *b1;
						REQUIRE_EQ(run_gemm(w, gr, ms[mi], with_bias, &b0), CCV_NNC_EXEC_SUCCESS, "8I_ROWWISE GEMM runs");
						REQUIRE_EQ(run_gemm(w, gq, ms[mi], with_bias, &b1), CCV_NNC_EXEC_SUCCESS, "codebook GEMM runs");
						REQUIRE_EQ(memcmp(b0, b1, (size_t)ms[mi] * w.rows * esize), 0, "format %d P %d M %d bias %d: codebook GEMM equals the GEMM on the reference codes", w.format, entries[e], ms[mi], with_bias);
						ccfree(b0);
						ccfree(b1);
					}
				ccv_nnc_tensor_free(gr);
				ccv_nnc_tensor_free(hr);
			}
			ccfree(scales);
			ccv_nnc_tensor_free(hout);
			ccv_nnc_tensor_free(gout);
			ccfree(q8);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(hq);
		}
	}
}

TEST_CASE("mps gemm rejects an unregistered codebook slot")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const weight_case_t w = weight_cases[0];
	ccv_nnc_tensor_t *hq, *gq;
	make_weights(w, 250, 0x1u, &hq, &gq);
	int m;
	for (m = 1; m <= 9; m += 8)
	{
		uint8_t* out;
		const int status = run_gemm(w, gq, m, 0, &out);
		REQUIRE_EQ(status, CCV_NNC_EXEC_INVALID, "slot 250 is not registered (M %d)", m);
		ccfree(out);
	}
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
}

#include "case_main.h"
