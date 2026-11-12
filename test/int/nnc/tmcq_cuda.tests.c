#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <math.h>
#include "nnc/ccv_nnc_tmcq_table.inc"

TEST_SETUP()
{
	ccv_nnc_init();
}

#ifdef HAVE_CUDA
extern void ccv_nnc_compat_dequantize_8i_rowwise_x(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context);
extern void ccv_nnc_compat_dequantize_8i_rowwise_x_fp(const void* input, const int datatype, const size_t input_length, const size_t row_length, const int format, void* output, const size_t output_length, ccv_nnc_stream_context_t* const stream_context);
extern int ccv_nnc_compat_8i_rowwise_gemm_supported(const int datatype, const void* const a, const void* const w, const void* const bias, const void* const b, const int m, const int n, const int k, size_t* const workspace_size);
#endif

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

// The v1.2 code of a 15-bit window for weight parity `parity`: T[2 ((cw * 0x83DCD12D) >> 23) + parity].
static int _tmcq_code(const uint32_t cw, const int parity)
{
	return ccv_nnc_tmcq_p2_table[2 * ((cw * 0x83DCD12Du) >> 23) + parity];
}

// An independent decode, bit by bit and per weight, of the codec definition (see CCV_NNC_QX_8I_ROWWISE_TMCQ_N16):
// weight t of tile g reads the 15-bit LSB-first window ending at E_{t | 1} with E_t = (t + 1) n / 16 (exclusive),
// wrapping inside the 16 n bits of the tile, and its code is T[2 ((cw * 0x83DCD12D) >> 23) + (t & 1)].
static void _tmcq_reference_codes(const uint8_t* const payload, const int rows, const int k, const int n, int8_t* const codes)
{
	const size_t tile_bits = (size_t)16 * n;
	const int tiles_per_row = k / 16;
	size_t g;
	int t, j;
	for (g = 0; g < (size_t)(rows / 16) * tiles_per_row; g++)
		for (t = 0; t < 256; t++)
		{
			const size_t end = (size_t)(((t | 1) + 1) * n) >> 4;
			uint32_t cw = 0;
			for (j = 0; j < 15; j++)
			{
				const size_t bit = g * tile_bits + (end + tile_bits - 15 + j) % tile_bits;
				cw |= (uint32_t)((payload[bit >> 3] >> (bit & 7)) & 1) << j;
			}
			const size_t tr = g / tiles_per_row;
			const size_t tc = g - tr * tiles_per_row;
			codes[(tr * 16 + t / 16) * k + tc * 16 + t % 16] = (int8_t)_tmcq_code(cw, t & 1);
		}
}

static double _tmcq_load(const int datatype, const void* const data, const size_t i)
{
	float f;
	if (datatype == CCV_16F)
	{
		ccv_half_precision_to_float((const uint16_t*)data + i, &f, 1);
		return f;
	} else if (datatype == CCV_16BF) {
		ccv_bfloat_to_float((const uint16_t*)data + i, &f, 1);
		return f;
	} else if (datatype == CCV_32F)
		return ((const float*)data)[i];
	return ((const double*)data)[i];
}

static void _tmcq_store(const int datatype, void* const data, const size_t i, const double v)
{
	// The same conversions as the CPU row-wise decode (_ccv_nnc_8i_rowwise_packed_write_value).
	const float f = (float)v;
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(&f, (uint16_t*)data + i, 1);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(&f, (uint16_t*)data + i, 1);
	else if (datatype == CCV_32F)
		((float*)data)[i] = f;
	else
		((double*)data)[i] = v;
}

// Fills a TMCQ payload with random bits (every bit string is a valid payload) and random row scales of the datatype.
// Returns the scale offset.
static size_t _tmcq_fill(ccv_nnc_tensor_t* const tensor, const int datatype, const int rows, const double scale_range, dsfmt_t* const dsfmt)
{
	const size_t size = ccv_nnc_tensor_data_size_without_padding(tensor->info);
	const size_t scale_offset = size - (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatype);
	size_t i;
	memset(tensor->data.u8, 0, size);
	for (i = 0; i < scale_offset; i++)
		tensor->data.u8[i] = (uint8_t)(dsfmt_genrand_uint32(dsfmt) >> 24);
	for (i = 0; i < (size_t)rows; i++)
	{
		// Mostly positive scales of varied magnitude, some negative, and an exact zero.
		double s = scale_range * (0.05 + dsfmt_genrand_open_close(dsfmt));
		if (i % 5 == 3)
			s = -s;
		if (i == 7)
			s = 0;
		_tmcq_store(datatype, tensor->data.u8 + scale_offset, i, s);
	}
	return scale_offset;
}

TEST_CASE("TMCQ reference decode reads the expected codes of a fixed payload")
{
	// Codes read by hand off ccv_nnc_tmcq_table.inc (16 entries per line). All-zero bits are pair 0 = (54, -52) for
	// every weight pair; all-one bits are cw = 0x7fff, k = 0xE4B9AED3 >> 23 = 457 = (44, -3).
	const int n = 16;
	uint8_t payload[16 * 16 * 2] = {0};
	int8_t codes[16 * 16];
	_tmcq_reference_codes(payload, 16, 16, n, codes);
	int i;
	for (i = 0; i < 256; i++)
		REQUIRE_EQ(codes[i], i % 2 ? -52 : 54, "zero bits decode to pair 0");
	memset(payload, 0xff, sizeof(payload));
	_tmcq_reference_codes(payload, 16, 16, n, codes);
	for (i = 0; i < 256; i++)
		REQUIRE_EQ(codes[i], i % 2 ? -3 : 44, "one bits decode to pair 457");
	// A single set bit at position 0 of a n = 16 tile: pair j reads the window ending at E_{2j+1} = 2j + 2, i.e.
	// bits [2j - 13, 2j + 1] (mod 256), which holds bit 0 as bit 13 - 2j for 2j + 1 <= 14.
	memset(payload, 0, sizeof(payload));
	payload[0] = 1;
	_tmcq_reference_codes(payload, 16, 16, n, codes);
	// cw = 0x2000 -> k = 308 = (0, 21); cw = 0x2 -> k = 15 = (9, -4); past weight 13 the window is clear.
	REQUIRE_EQ(codes[0], 0, "pair 0 reads cw 0x2000");
	REQUIRE_EQ(codes[1], 21, "pair 0 reads cw 0x2000");
	REQUIRE_EQ(codes[12], 9, "pair 6 reads cw 0x2");
	REQUIRE_EQ(codes[13], -4, "pair 6 reads cw 0x2");
	REQUIRE_EQ(codes[14], 54, "pair 7 reads cw 0");
	REQUIRE_EQ(codes[15], -52, "pair 7 reads cw 0");
	for (i = 0; i < 256; i++)
	{
		const uint32_t cw = (i | 1) <= 14 ? 1u << (14 - (i | 1)) : 0;
		REQUIRE_EQ(codes[i], _tmcq_code(cw, i & 1), "a single set bit lands in the windows that end after it");
	}
}

TEST_CASE("TMCQ CUDA dequantization matches the CPU decode bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
#ifdef HAVE_CUDA
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF, CCV_64F};
	const int shapes[][2] = {
		{16, 256},
		{48, 768},
		{32, 4096},
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 13);
	int d, f, s, unaligned;
	for (f = 0; f < sizeof(tmcq_formats) / sizeof(tmcq_formats[0]); f++)
		for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
			for (s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
				for (unaligned = 0; unaligned < 2; unaligned++)
				{
					if (unaligned && s != 1)
						continue;
					const int format = tmcq_formats[f];
					const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
					const int rows = shapes[s][0];
					const int k = shapes[s][1];
					const size_t count = (size_t)rows * k;
					ccv_nnc_tensor_param_t host_params = CPU_TENSOR_NHWC(32F, rows, k);
					host_params.datatype = datatypes[d];
					ccv_nnc_tensor_param_t gpu_params = host_params;
					gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
					ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
					ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
					const size_t qsize = ccv_nnc_tensor_data_size_without_padding(q->info);
					REQUIRE_EQ(qsize, ((((size_t)rows * (k / 256) * 16 * n + 7) / 8 + 127) & ~(size_t)127) + (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]), "TMCQ payload size follows the tile layout");
					const size_t scale_offset = _tmcq_fill(q, datatypes[d], rows, 0.01, &dsfmt);
					// Expected: the reference codes times the row scales, narrowed like the CPU decode.
					int8_t* const codes = (int8_t*)ccmalloc(count);
					_tmcq_reference_codes(q->data.u8, rows, k, n, codes);
					ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_params, 0);
					size_t i;
					for (i = 0; i < count; i++)
						_tmcq_store(datatypes[d], expected->data.u8, i, _tmcq_load(datatypes[d], q->data.u8 + scale_offset, i / k) * codes[i]);
					// The unaligned case writes one element past a 16-byte boundary, through the scalar stores.
					ccv_nnc_tensor_param_t out_params = gpu_params;
					out_params.dim[0] = rows + 1;
					ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, out_params, 0);
					uint8_t* const gout_data = gout->data.u8 + (unaligned ? CCV_GET_DATA_TYPE_SIZE(datatypes[d]) : 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
					ccv_nnc_compat_dequantize_8i_rowwise_x_fp(gq->data.u8, datatypes[d], qsize, k, format, gout_data, count, 0);
					ccv_nnc_tensor_t gview = ccv_nnc_tensor(gout_data, gpu_params, 0);
					ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_params, 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&gview), TENSOR_LIST(actual), 0);
					const int equal = memcmp(expected->data.u8, actual->data.u8, count * CCV_GET_DATA_TYPE_SIZE(datatypes[d])) == 0;
					REQUIRE(equal, "TMCQ fp dequantization should match the CPU decode bit for bit: n=%d datatype=%x rows=%d k=%d unaligned=%d", n, datatypes[d], rows, k, unaligned);
					// The public entry point dispatches to the same kernel.
					ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], CCV_TENSOR_GPU_MEMORY, qsize, k, format, gout->data.u8, count);
					gview = ccv_nnc_tensor(gout->data.u8, gpu_params, 0);
					ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&gview), TENSOR_LIST(actual), 0);
					REQUIRE(memcmp(expected->data.u8, actual->data.u8, count * CCV_GET_DATA_TYPE_SIZE(datatypes[d])) == 0, "public GPU dequantization should match: n=%d datatype=%x", n, datatypes[d]);
					ccfree(codes);
					ccv_nnc_tensor_free(actual);
					ccv_nnc_tensor_free(gout);
					ccv_nnc_tensor_free(expected);
					ccv_nnc_tensor_free(gq);
					ccv_nnc_tensor_free(q);
				}
#endif
}

TEST_CASE("TMCQ CUDA int8 row-wise conversion matches the CPU codes and copies the scales")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
#ifdef HAVE_CUDA
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF, CCV_64F};
	const int shapes[][2] = {
		{16, 256},
		{64, 512},
		{32, 4096},
		{2048, 256},
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 17);
	int d, f, s;
	for (f = 0; f < sizeof(tmcq_formats) / sizeof(tmcq_formats[0]); f++)
		for (s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
		{
			const int format = tmcq_formats[f];
			const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
			const int rows = shapes[s][0];
			const int k = shapes[s][1];
			const size_t count = (size_t)rows * k;
			d = (f + s) % 4;
			ccv_nnc_tensor_param_t host_params = CPU_TENSOR_NHWC(32F, rows, k);
			host_params.datatype = datatypes[d];
			ccv_nnc_tensor_param_t gpu_params = host_params;
			gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
			ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
			const size_t qsize = ccv_nnc_tensor_data_size_without_padding(q->info);
			const size_t scale_offset = _tmcq_fill(q, datatypes[d], rows, 0.01, &dsfmt);
			int8_t* const codes = (int8_t*)ccmalloc(count);
			_tmcq_reference_codes(q->data.u8, rows, k, n, codes);
			ccv_nnc_tensor_t* const g8 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(gpu_params), 0);
			ccv_nnc_tensor_t* const h8 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(host_params), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
			ccv_nnc_compat_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], qsize, k, format, g8->data.u8, count, 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g8), TENSOR_LIST(h8), 0);
			REQUIRE(memcmp(h8->data.u8, codes, count) == 0, "TMCQ int8 codes should match the CPU decode: n=%d rows=%d k=%d", n, rows, k);
			const size_t scale_size = (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
			REQUIRE(memcmp(h8->data.u8 + ((count + 127) & ~(size_t)127), q->data.u8 + scale_offset, scale_size) == 0, "TMCQ int8 conversion copies the row scales: n=%d datatype=%x", n, datatypes[d]);
			// An output 8 bytes off a 16-byte boundary takes the 8-byte stores.
			ccv_nnc_tensor_param_t big_params = gpu_params;
			big_params.dim[0] = rows + 16;
			ccv_nnc_tensor_t* const g8_big = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(big_params), 0);
			ccv_nnc_compat_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], qsize, k, format, g8_big->data.u8 + 8, count, 0);
			ccv_nnc_tensor_t g8_view = ccv_nnc_tensor(g8_big->data.u8 + 8, ccv_nnc_tensor_8i_rowwise(gpu_params), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&g8_view), TENSOR_LIST(h8), 0);
			REQUIRE(memcmp(h8->data.u8, codes, count) == 0, "unaligned TMCQ int8 codes should match the CPU decode: n=%d rows=%d k=%d", n, rows, k);
			ccv_nnc_tensor_free(g8_big);
			ccfree(codes);
			ccv_nnc_tensor_free(h8);
			ccv_nnc_tensor_free(g8);
			ccv_nnc_tensor_free(gq);
			ccv_nnc_tensor_free(q);
		}
#endif
}

static void _tmcq_to_float(const int datatype, const void* const data, const size_t count, float* const values)
{
	if (datatype == CCV_16F)
		ccv_half_precision_to_float((const uint16_t*)data, values, count);
	else if (datatype == CCV_16BF)
		ccv_bfloat_to_float((const uint16_t*)data, values, count);
	else
		memcpy(values, data, sizeof(float) * count);
}

static void _tmcq_from_float(const int datatype, const float* const values, const size_t count, void* const data)
{
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)data, count);
	else if (datatype == CCV_16BF)
		ccv_float_to_bfloat(values, (uint16_t*)data, count);
	else
		memcpy(data, values, sizeof(float) * count);
}

// Runs a[m, k] x w[n, k]^T (+ bias[n]) on the GPU with random TMCQ weights through the GEMM command, and returns the
// largest difference to a double precision CPU reference, relative to max(1, |reference|, |actual|). Like the row-wise
// int8 GEMM tests, the reference follows the GPU path: the row-wise int8 path multiplies the exact codes times the
// row scales, and above 4 rows quantizes the activations row-wise to int8 (scale = max |a| / 127); the dequantizing
// fallback multiplies the weights decoded into the datatype. If fp_rel_out is set, it also returns the largest
// difference to the plain FP GEMM of the activations and the decoded weights, without the activation quantization.
static double _tmcq_gemm_max_rel_diff(const int datatype, const int format, const int use_bias, const int misaligned_bias, const int m, const int n, const int k, dsfmt_t* const dsfmt, int* const fast_path_out, double* const fp_rel_out)
{
	ccv_nnc_tensor_param_t a_params = CPU_TENSOR_NHWC(32F, m, k);
	a_params.datatype = datatype;
	ccv_nnc_tensor_param_t w_params = CPU_TENSOR_NHWC(32F, n, k);
	w_params.datatype = datatype;
	ccv_nnc_tensor_param_t bias_params = CPU_TENSOR_NHWC(32F, n);
	bias_params.datatype = datatype;
	ccv_nnc_tensor_param_t b_params = CPU_TENSOR_NHWC(32F, m, n);
	b_params.datatype = datatype;
	ccv_nnc_tensor_t* const ha = ccv_nnc_tensor_new(0, a_params, 0);
	ccv_nnc_tensor_t* const hwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(w_params, format), 0);
	ccv_nnc_tensor_t* const hbias = use_bias ? ccv_nnc_tensor_new(0, bias_params, 0) : 0;
	ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, b_params, 0);
	float* const a_f = (float*)ccmalloc(sizeof(float) * m * k);
	float* const w_f = (float*)ccmalloc(sizeof(float) * n * k);
	float* const bias_f = (float*)ccmalloc(sizeof(float) * n);
	float* const b_f = (float*)ccmalloc(sizeof(float) * m * n);
	int i, j, l;
	for (i = 0; i < m * k; i++)
		a_f[i] = (float)(dsfmt_genrand_open_close(dsfmt) * 2 - 1) * (0.25 + (i / k) % 7);
	for (i = 0; i < n; i++)
		bias_f[i] = (float)(dsfmt_genrand_open_close(dsfmt) * 2 - 1);
	_tmcq_from_float(datatype, a_f, m * k, ha->data.u8);
	_tmcq_to_float(datatype, ha->data.u8, m * k, a_f);
	if (use_bias)
	{
		_tmcq_from_float(datatype, bias_f, n, hbias->data.u8);
		_tmcq_to_float(datatype, hbias->data.u8, n, bias_f);
	}
	// Random codes are about N(0, 37^2), thus scales of ~1 / 4096 keep the outputs around 1.
	const size_t scale_offset = _tmcq_fill(hwq, datatype, n, 1.0 / 4096, dsfmt);
	const int rate = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
	int8_t* const codes = (int8_t*)ccmalloc((size_t)n * k);
	_tmcq_reference_codes(hwq->data.u8, n, k, rate, codes);
	ccv_nnc_tensor_param_t ga_params = a_params;
	ga_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gw_params = w_params;
	gw_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gbias_params = bias_params;
	gbias_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_param_t gb_params = b_params;
	gb_params.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ga_params, 0);
	ccv_nnc_tensor_t* const gwq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gw_params, format), 0);
	// A bias off its 16-byte alignment keeps the CUTLASS epilogue out, thus more than 4 rows take the fallback.
	ccv_nnc_tensor_param_t gbias_storage_params = gbias_params;
	gbias_storage_params.dim[0] = n + 8;
	ccv_nnc_tensor_t* const gbias_storage = use_bias ? ccv_nnc_tensor_new(0, gbias_storage_params, 0) : 0;
	ccv_nnc_tensor_t gbias_view;
	if (use_bias)
		gbias_view = ccv_nnc_tensor(gbias_storage->data.u8 + (misaligned_bias ? CCV_GET_DATA_TYPE_SIZE(datatype) : 0), gbias_params, 0);
	ccv_nnc_tensor_t* const gbias = use_bias ? &gbias_view : 0;
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, gb_params, 0);
#ifdef HAVE_CUDA
	size_t workspace_size;
	const int fast_path = ccv_nnc_compat_8i_rowwise_gemm_supported(datatype, ga->data.u8, 0, gbias ? gbias->data.u8 : 0, gb->data.u8, m, n, k, &workspace_size);
#else
	const int fast_path = 0;
#endif
	*fast_path_out = fast_path;
	for (i = 0; i < n; i++)
	{
		const double scale = _tmcq_load(datatype, hwq->data.u8 + scale_offset, i);
		for (j = 0; j < k; j++)
		{
			if (fast_path)
				w_f[i * k + j] = (float)(scale * codes[(size_t)i * k + j]);
			else {
				// The fallback decodes into the datatype (truncating like the CPU decode) before cuBLAS.
				uint8_t v[8];
				_tmcq_store(datatype, v, 0, scale * codes[(size_t)i * k + j]);
				w_f[i * k + j] = (float)_tmcq_load(datatype, v, 0);
			}
		}
	}
	float* const a_fp = fp_rel_out ? (float*)ccmalloc(sizeof(float) * m * k) : 0;
	if (a_fp)
		memcpy(a_fp, a_f, sizeof(float) * m * k);
	if (fast_path && m > 4)
		for (i = 0; i < m; i++)
		{
			float max_abs = 0;
			for (j = 0; j < k; j++)
				max_abs = ccv_max(max_abs, fabsf(a_f[i * k + j]));
			const float scale = max_abs > 0 ? max_abs / 127.f : 1.f / 127.f;
			const float inv_scale = max_abs > 0 ? 127.f / max_abs : 127.f;
			for (j = 0; j < k; j++)
				a_f[i * k + j] = ccv_clamp(rintf(a_f[i * k + j] * inv_scale), -127, 127) * scale;
		}
	int status;
	if (use_bias)
	{
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq, hbias), TENSOR_LIST(ga, gwq, gbias), 0);
		status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gwq, gbias), TENSOR_LIST(gb), 0);
	} else {
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ha, hwq), TENSOR_LIST(ga, gwq), 0);
		status = ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gwq), TENSOR_LIST(gb), 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(hb), 0);
	double max_rel = status == CCV_NNC_EXEC_SUCCESS ? 0 : HUGE_VAL;
	double fp_rel = max_rel;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
		{
			double expected = use_bias ? bias_f[j] : 0;
			double expected_fp = expected;
			for (l = 0; l < k; l++)
				expected += (double)a_f[(size_t)i * k + l] * w_f[(size_t)j * k + l];
			const double actual = _tmcq_load(datatype, hb->data.u8, (size_t)i * n + j);
			const double rel = fabs(actual - expected) / ccv_max(1.0, ccv_max(fabs(actual), fabs(expected)));
			if (!(rel <= max_rel)) // Also catches NaN.
				max_rel = isnan(rel) ? HUGE_VAL : rel;
			if (a_fp)
			{
				for (l = 0; l < k; l++)
					expected_fp += (double)a_fp[(size_t)i * k + l] * w_f[(size_t)j * k + l];
				const double rel_fp = fabs(actual - expected_fp) / ccv_max(1.0, ccv_max(fabs(actual), fabs(expected_fp)));
				if (!(rel_fp <= fp_rel))
					fp_rel = isnan(rel_fp) ? HUGE_VAL : rel_fp;
			}
		}
	if (fp_rel_out)
		*fp_rel_out = fp_rel;
	if (a_fp)
		ccfree(a_fp);
	ccfree(codes);
	ccfree(a_f);
	ccfree(w_f);
	ccfree(bias_f);
	ccfree(b_f);
	ccv_nnc_tensor_free(ha);
	ccv_nnc_tensor_free(hwq);
	if (hbias)
		ccv_nnc_tensor_free(hbias);
	ccv_nnc_tensor_free(hb);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(gwq);
	if (gbias_storage)
		ccv_nnc_tensor_free(gbias_storage);
	ccv_nnc_tensor_free(gb);
	return max_rel;
}

TEST_CASE("cublas forward gemm with TMCQ weights matches CPU across rates, row counts and precisions")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	// Same tolerances as the row-wise int8 GEMM tests.
	const double tolerances[] = {1e-4, 1e-3, 5e-3};
	const double fallback_tolerances[] = {1e-4, 2e-2, 5e-2};
	// M = 1 and 4 run the int8 GEMV, 64 and 512 the int8 CUTLASS GEMM.
	const int shapes[][3] = {
		{1, 2048, 4096},
		{4, 1024, 2048},
		{64, 384, 768},
		{512, 512, 1024},
	};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 29);
	int d, s;
	for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
		for (s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
		{
			const int f = (d * 4 + s) % (sizeof(tmcq_formats) / sizeof(tmcq_formats[0]));
			const int use_bias = (d + s) % 2;
			int fast_path;
			const double max_rel = _tmcq_gemm_max_rel_diff(datatypes[d], tmcq_formats[f], use_bias, 0, shapes[s][0], shapes[s][1], shapes[s][2], &dsfmt, &fast_path, 0);
			const double tolerance = fast_path ? tolerances[d] : fallback_tolerances[d];
			if (max_rel > tolerance)
				printf("format %d, datatype %x, bias %d, m %d, n %d, k %d, fast path %d: max relative difference %g\n", tmcq_formats[f], datatypes[d], use_bias, shapes[s][0], shapes[s][1], shapes[s][2], fast_path, max_rel);
			REQUIRE(max_rel <= tolerance, "TMCQ GEMM should match the CPU reference");
		}
}

TEST_CASE("cublas forward gemm with TMCQ weights falls back to the dequantized weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	const double fallback_tolerances[] = {1e-4, 2e-2, 5e-2};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 31);
	int d;
	for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
	{
		int fast_path;
		const double max_rel = _tmcq_gemm_max_rel_diff(datatypes[d], tmcq_formats[d * 3 + 1], 1, 1, 64, 256, 512, &dsfmt, &fast_path, 0);
		REQUIRE(!fast_path, "a misaligned bias takes the dequantizing fallback");
		REQUIRE(max_rel <= fallback_tolerances[d], "TMCQ fallback GEMM should match the CPU reference: datatype=%x max_rel=%g", datatypes[d], max_rel);
	}
}

TEST_CASE("cublas forward gemm with TMCQ weights runs the CUTLASS int8 GEMM from M = 5 to 2048")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// CCV_REQUIRE_CUTLASS_TESTS=1 turns every skip of this case into a failure, so a build that is meant to cover the
	// CUTLASS path cannot pass silently.
	const char* const require_env = getenv("CCV_REQUIRE_CUTLASS_TESTS");
	const int require = require_env && require_env[0] == '1';
#ifndef HAVE_CUDA_SM80
	printf("SKIP: TMCQ CUTLASS int8 GEMM test: built without HAVE_CUDA_SM80 (configure --enable-sm80), thus more than 4 rows take the dequantizing cuBLAS fallback\n");
	REQUIRE(!require, "CCV_REQUIRE_CUTLASS_TESTS=1 but the build has no HAVE_CUDA_SM80");
#else
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF};
	const double tolerances[] = {1e-4, 1e-3, 5e-3};
	// Each M picks a different CUTLASS tile in _ccv_nnc_8i_rowwise_cutlass_gemm_dispatch: 16 x 128 (n >= 8192), 64 x 64,
	// 128 x 128 and 256 x 128.
	const int shapes[][3] = {
		{5, 8192, 512},
		{64, 384, 768},
		{512, 512, 1024},
		{2048, 256, 1024},
	};
	size_t workspace_size;
	// 16-byte aligned dummies: only the device and the shapes decide.
	if (!ccv_nnc_compat_8i_rowwise_gemm_supported(CCV_32F, 0, 0, 0, 0, shapes[0][0], shapes[0][1], shapes[0][2], &workspace_size))
	{
		printf("SKIP: TMCQ CUTLASS int8 GEMM test: the int8 GEMM is not supported on this device (needs sm_80 or newer)\n");
		REQUIRE(!require, "CCV_REQUIRE_CUTLASS_TESTS=1 but the device does not support the int8 GEMM");
		return;
	}
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 43);
	int d, s;
	for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
		for (s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++)
		{
			const int f = (d * 3 + s * 2) % (sizeof(tmcq_formats) / sizeof(tmcq_formats[0]));
			const int use_bias = (d + s) % 2;
			int fast_path;
			double fp_rel;
			const double max_rel = _tmcq_gemm_max_rel_diff(datatypes[d], tmcq_formats[f], use_bias, 0, shapes[s][0], shapes[s][1], shapes[s][2], &dsfmt, &fast_path, &fp_rel);
			printf("TMCQ CUTLASS GEMM: format %d, datatype %x, bias %d, m %d, n %d, k %d: max relative difference %g to the int8 activation reference, %g to the plain FP GEMM\n", tmcq_formats[f], datatypes[d], use_bias, shapes[s][0], shapes[s][1], shapes[s][2], max_rel, fp_rel);
			REQUIRE(fast_path, "TMCQ weights take the int8 GEMM: m=%d", shapes[s][0]);
			REQUIRE(max_rel <= tolerances[d], "TMCQ CUTLASS GEMM should match the CPU reference: datatype=%x m=%d max_rel=%g", datatypes[d], shapes[s][0], max_rel);
			// Only the CUTLASS path quantizes the activations to int8: in FP32, its error to the plain FP GEMM (the
			// cuBLAS fallback's result) is far above the error to the int8 activation reference, thus the fallback
			// cannot have produced this output.
			if (datatypes[d] == CCV_32F)
				REQUIRE(fp_rel > 10 * ccv_max(max_rel, tolerances[d]), "the output carries the int8 activation quantization of the CUTLASS path: m=%d max_rel=%g fp_rel=%g", shapes[s][0], max_rel, fp_rel);
		}
#endif
}

TEST_CASE("TMCQ CUDA dequantization matches ccv_nnc_dequantize_8i_rowwise_x on CPU bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	const int datatypes[] = {CCV_32F, CCV_16F, CCV_16BF, CCV_64F};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 37);
	int d, f, source;
	for (source = 0; source < 2; source++)
		for (f = 0; f < sizeof(tmcq_formats) / sizeof(tmcq_formats[0]); f++)
			for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
			{
				// The encoder is slow, thus it runs on a few (rate, datatype) pairs; random payloads cover every pair.
				if (source == 1 && (f + d) % 3 != 0)
					continue;
				const int format = tmcq_formats[f];
				const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(format);
				const int rows = source ? 32 : 48;
				const int k = source ? 256 : 768;
				const size_t count = (size_t)rows * k;
				ccv_nnc_tensor_param_t host_params = CPU_TENSOR_NHWC(32F, rows, k);
				host_params.datatype = datatypes[d];
				ccv_nnc_tensor_param_t gpu_params = host_params;
				gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
				ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(host_params, format), 0);
				ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(gpu_params, format), 0);
				const size_t qsize = ccv_nnc_tensor_data_size_without_padding(q->info);
				if (source)
				{
					// Payload and scales from the CPU encoder.
					ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, host_params, 0);
					size_t i;
					for (i = 0; i < count; i++)
						_tmcq_store(datatypes[d], w->data.u8, i, (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * (0.5 + (i / k) % 5));
					REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(w->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, count, k, format, 0, 0, 0, 0, q->data.u8, qsize), qsize, "TMCQ quantization fills the payload: n=%d", n);
					ccv_nnc_tensor_free(w);
				} else
					_tmcq_fill(q, datatypes[d], rows, 0.01, &dsfmt);
				ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, host_params, 0);
				ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, datatypes[d], CCV_TENSOR_CPU_MEMORY, qsize, k, format, expected->data.u8, count);
				// The CPU decode agrees with this test's independent reference as well.
				const size_t scale_offset = qsize - (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
				int8_t* const codes = (int8_t*)ccmalloc(count);
				_tmcq_reference_codes(q->data.u8, rows, k, n, codes);
				ccv_nnc_tensor_t* const reference = ccv_nnc_tensor_new(0, host_params, 0);
				size_t i;
				for (i = 0; i < count; i++)
					_tmcq_store(datatypes[d], reference->data.u8, i, _tmcq_load(datatypes[d], q->data.u8 + scale_offset, i / k) * codes[i]);
				const size_t bytes = count * CCV_GET_DATA_TYPE_SIZE(datatypes[d]);
				REQUIRE(memcmp(expected->data.u8, reference->data.u8, bytes) == 0, "CPU TMCQ decode should match the reference: n=%d datatype=%x source=%d", n, datatypes[d], source);
				ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, gpu_params, 0);
				ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, host_params, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(q), TENSOR_LIST(gq), 0);
				ccv_nnc_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], CCV_TENSOR_GPU_MEMORY, qsize, k, format, gout->data.u8, count);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(actual), 0);
				REQUIRE(memcmp(expected->data.u8, actual->data.u8, bytes) == 0, "GPU TMCQ decode should match the CPU API bit for bit: n=%d datatype=%x source=%d", n, datatypes[d], source);
				// The int8 conversion holds the CPU codes: the CPU decode with FP32 scales of 1.
				ccv_nnc_tensor_t* const g8 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(gpu_params), 0);
				ccv_nnc_tensor_t* const h8 = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(host_params), 0);
				ccv_nnc_compat_dequantize_8i_rowwise_x(gq->data.u8, datatypes[d], qsize, k, format, g8->data.u8, count, 0);
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(g8), TENSOR_LIST(h8), 0);
				uint8_t* const unit = (uint8_t*)ccmalloc(scale_offset + sizeof(float) * rows);
				memcpy(unit, q->data.u8, scale_offset);
				for (i = 0; i < rows; i++)
					((float*)(unit + scale_offset))[i] = 1;
				float* const cpu_codes = (float*)ccmalloc(sizeof(float) * count);
				ccv_nnc_dequantize_8i_rowwise_x(unit, CCV_32F, CCV_TENSOR_CPU_MEMORY, scale_offset + sizeof(float) * rows, k, format, cpu_codes, count);
				int codes_equal = 1;
				for (i = 0; i < count && codes_equal; i++)
					codes_equal = cpu_codes[i] == (float)(int8_t)h8->data.u8[i];
				REQUIRE(codes_equal, "GPU TMCQ int8 codes should match the CPU API: n=%d source=%d", n, source);
				REQUIRE(memcmp(h8->data.u8 + ((count + 127) & ~(size_t)127), q->data.u8 + scale_offset, (size_t)rows * CCV_GET_DATA_TYPE_SIZE(datatypes[d])) == 0, "GPU TMCQ int8 conversion copies the CPU scales");
				ccfree(cpu_codes);
				ccfree(unit);
				ccfree(codes);
				ccv_nnc_tensor_free(h8);
				ccv_nnc_tensor_free(g8);
				ccv_nnc_tensor_free(actual);
				ccv_nnc_tensor_free(gout);
				ccv_nnc_tensor_free(reference);
				ccv_nnc_tensor_free(expected);
				ccv_nnc_tensor_free(gq);
				ccv_nnc_tensor_free(q);
			}
}

TEST_CASE("cublas forward gemm with CPU-quantized TMCQ weights matches the CPU dequantized GEMM")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_GPU_CUBLAS));
	// M = 1 runs the int8 GEMV on the converted weights; the reference multiplies the CPU dequantized FP32 weights.
	const int m = 1, n = 64, k = 512;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 41);
	const int format = CCV_NNC_QX_8I_ROWWISE_TMCQ_N24;
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, n, k), 0);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, k), 0);
	int i, j;
	for (i = 0; i < n * k; i++)
		w->data.f32[i] = (dsfmt_genrand_open_close(&dsfmt) * 2 - 1) * 0.1;
	for (i = 0; i < m * k; i++)
		a->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
	ccv_nnc_tensor_t* const q = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, n, k), format), 0);
	const size_t qsize = ccv_nnc_tensor_data_size_without_padding(q->info);
	REQUIRE_EQ(ccv_nnc_quantize_8i_rowwise_x(w->data.f32, CCV_32F, CCV_TENSOR_CPU_MEMORY, n * k, k, format, 0, 0, 0, 0, q->data.u8, qsize), qsize, "TMCQ quantization fills the payload");
	ccv_nnc_dequantize_8i_rowwise_x(q->data.u8, CCV_32F, CCV_TENSOR_CPU_MEMORY, qsize, k, format, w->data.f32, n * k);
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, m, k), 0);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(GPU_TENSOR_NHWC(000, 32F, n, k), format), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, m, n), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, m, n), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(a, q), TENSOR_LIST(ga, gq), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gq), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_SUCCESS, "TMCQ GEMV runs");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gb), TENSOR_LIST(b), 0);
	double max_rel = 0;
	for (j = 0; j < n; j++)
	{
		double expected = 0;
		for (i = 0; i < k; i++)
			expected += (double)a->data.f32[i] * w->data.f32[j * k + i];
		max_rel = ccv_max(max_rel, fabs(b->data.f32[j] - expected) / ccv_max(1.0, fabs(expected)));
	}
	REQUIRE(max_rel <= 1e-4, "TMCQ GEMV on CPU-quantized weights should match: %g", max_rel);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(ga);
	ccv_nnc_tensor_free(q);
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(w);
}

TEST_CASE("CUDA index select rejects TMCQ embeddings")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	ccv_nnc_tensor_t* const ga = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(GPU_TENSOR_NHWC(000, 16F, 32, 256), CCV_NNC_QX_8I_ROWWISE_TMCQ_N16), 0);
	ccv_nnc_tensor_t* const gindices = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 2), 0);
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 2, 256), 0);
	ccv_nnc_cmd_t cmd = CMD_INDEX_SELECT_FORWARD();
	cmd.backend = CCV_NNC_BACKEND_GPU_REF;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices), TENSOR_LIST(gb), 0), CCV_NNC_EXEC_INVALID, "a TMCQ row cannot be gathered on its own");
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(ga);
}

#include "case_main.h"
