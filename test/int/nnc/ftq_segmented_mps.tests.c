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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Static-rate FTQ (CCV_NNC_QX_8I_ROWWISE_FTQ_N16) on the routed MoE paths of the Metal backend. An [E, N, K] FTQ weight
// is one CCVFTQSR blob of E N rows: the 32-byte header, expert e's (N / 16) (K / 16) tiles of 16 n bits from byte
// 32 + e (N / 16) (K / 16) 2n, and one FP16 scale per row from the blob's scale offset, whatever the base datatype (FP16
// or FP32). Every reference decodes the blob, or 16-row slices of it, with the CPU codec as an FP16 tensor whose scales
// are 1 (so the FP16 outputs are the int8 codes), and takes the FP16 row scales from the blob. Any bit string is a valid
// code, so codes are random bytes.
// Paths: the selected-expert int8 decode (bit for bit), segmented GEMM through the fused FTQ GEMV (one row per bin) and
// through the W8A8 segmented scaled GEMM, segmented SwiGLU through the packed SwiGLU kernel (gate and up in one pass,
// to rounding with two FTQ GEMVs and the SwiGLU of their outputs) and through the W8A8 segmented scaled SwiGLU (both
// W8A8 kernels decoding the tiles in registers, N % 64 and K % 64, bit for bit with the selected-expert decode into int8
// rows that they replace, or taking that expansion), MoE weights streaming of FTQ banks, and the rejections.

TEST_SETUP()
{
	ccv_nnc_init();
}

extern uint8_t ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_mfa_context_t* context);

static uint32_t rng_next(uint32_t* const state)
{
	uint32_t x = *state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

static int ftq_format(const int n)
{
	return CCV_NNC_QX_8I_ROWWISE_FTQ_BLOCK_FIRST + (n - 8) / 2;
}

// The pair table of a blob: the codebook slot (0: the built-in table) and the header's table field and crc32.
typedef struct {
	int slot;
	uint32_t table;
	uint32_t crc;
} ftq_table_t;

static const ftq_table_t ftq_builtin = { 0, 9u << 8, 0 };

// A random 1024-pair table (header id 1, log2 10).
static ftq_table_t ftq_external_table(void)
{
	static ftq_table_t t;
	if (t.slot)
		return t;
	int8_t* const pairs = (int8_t*)ccmalloc(2048);
	uint32_t seed = 0x7e57ab1eu;
	int i;
	for (i = 0; i < 2048; i++)
		pairs[i] = (int8_t)((int)(rng_next(&seed) % 255) - 127);
	t.slot = ccv_nnc_codebook_register(pairs, 1024);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | (10u << 8);
	ccfree(pairs);
	return t;
}

// An external table of 2048 pairs (header id 1, log2 11), whose threadgroup copy takes more than one pass of 256
// threads (the 1024-pair table takes exactly one).
static ftq_table_t ftq_external11_table(void)
{
	static ftq_table_t t;
	if (t.slot)
		return t;
	int8_t* const pairs = (int8_t*)ccmalloc(4096);
	uint32_t seed = 0x7e57ab11u;
	int i;
	for (i = 0; i < 4096; i++)
		pairs[i] = (int8_t)((int)(rng_next(&seed) % 255) - 127);
	t.slot = ccv_nnc_codebook_register(pairs, 2048);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | (11u << 8);
	ccfree(pairs);
	return t;
}

// A quadrant-folded table of 512 non-negative base pairs (header id 1, log2 9, structure 1).
static ftq_table_t ftq_folded_table(void)
{
	static ftq_table_t t;
	if (t.slot)
		return t;
	int8_t* const base = (int8_t*)ccmalloc(1024);
	uint32_t seed = 0xf01dedu;
	int i;
	for (i = 0; i < 1024; i++)
		base[i] = (int8_t)(rng_next(&seed) % 128);
	t.slot = ccv_nnc_codebook_register_with_flags(base, 512, CCV_NNC_CODEBOOK_FOLD);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | (9u << 8) | (1u << 16);
	ccfree(base);
	return t;
}

static ccv_nnc_tensor_param_t ftq_param(const int type, const int datatype, const int e, const int n_rows, const int k, const int format)
{
	const ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
		.dim = { e, n_rows, k },
	};
	return ccv_nnc_tensor_8i_rowwise_x(params, format);
}

// A CCVFTQSR blob of rows x cols: header, random code, random FP16 scales in [1e-4, 0.02).
static void ftq_fill(uint8_t* const data, const size_t size, const int format, const int rows, const int cols, const ftq_table_t* const table, uint32_t seed)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	memset(data, 0, size);
	memcpy(data, "CCVFTQSR", 8);
	const uint32_t header[6] = { (uint32_t)rows, (uint32_t)cols, schedule.high_bits, (uint32_t)schedule.drop_count, table->table, table->crc };
	memcpy(data + 8, header, sizeof(header));
	const size_t code_bytes = schedule.code_bits / 8;
	size_t i;
	for (i = 0; i + 4 <= code_bytes; i += 4)
	{
		const uint32_t v = rng_next(&seed);
		memcpy(data + 32 + i, &v, 4);
	}
	for (; i < code_bytes; i++)
		data[32 + i] = (uint8_t)(rng_next(&seed) >> 24);
	float* const scales = (float*)ccmalloc(sizeof(float) * rows);
	int r;
	for (r = 0; r < rows; r++)
		scales[r] = (float)((rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
	ccv_float_to_half_precision(scales, (uint16_t*)(data + schedule.scale_offset), rows);
	ccfree(scales);
}

// An [e, n, k] FTQ weight of base datatype `datatype` on the CPU (*host) and the GPU.
static ccv_nnc_tensor_t* ftq_weight_new(const int datatype, const int e, const int n_rows, const int k, const int n, const ftq_table_t* const table, const uint32_t seed, ccv_nnc_tensor_t** const host)
{
	const int format = ccv_nnc_codebook_with_slot(ftq_format(n), table->slot);
	ccv_nnc_tensor_t* const hq = ccv_nnc_tensor_new(0, ftq_param(CCV_TENSOR_CPU_MEMORY, datatype, e, n_rows, k, format), 0);
	ftq_fill(hq->data.u8, ccv_nnc_tensor_data_size_without_padding(hq->info), format, e * n_rows, k, table, seed);
	ccv_nnc_tensor_t* const gq = ccv_nnc_tensor_new(0, ftq_param(CCV_TENSOR_GPU_MEMORY | 000, datatype, e, n_rows, k, format), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(gq), 0);
	*host = hq;
	return gq;
}

// The int8 codes of the first `rows` rows of an FTQ blob from the CPU codec: the blob (rows must be the header's, a
// whole tile row multiple) decoded as FP16 with every scale 1.
static void ftq_cpu_codes(const uint8_t* const blob, const int format, const int rows, const int cols, int8_t* const q8)
{
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, cols);
	const size_t size = schedule.scale_offset + 2 * (size_t)rows;
	uint8_t* const copy = (uint8_t*)ccmalloc(size);
	memcpy(copy, blob, schedule.scale_offset);
	uint16_t* const scales = (uint16_t*)(copy + schedule.scale_offset);
	int r;
	for (r = 0; r < rows; r++)
		scales[r] = 0x3c00; // 1.0
	uint16_t* const out = (uint16_t*)ccmalloc(sizeof(uint16_t) * rows * cols);
	const int status = ccv_nnc_dequantize_8i_rowwise_x(copy, CCV_16F, CCV_TENSOR_CPU_MEMORY, size, cols, format, out, (size_t)rows * cols);
	assert(status == 0);
	(void)status;
	float* const values = (float*)ccmalloc(sizeof(float) * rows * cols);
	ccv_half_precision_to_float(out, values, (size_t)rows * cols);
	size_t i;
	for (i = 0; i < (size_t)rows * cols; i++)
		q8[i] = (int8_t)values[i];
	ccfree(values);
	ccfree(out);
	ccfree(copy);
}

// Decoded weights of an [e, n, k] FTQ blob, by 16-row tile rows on demand: tile row tr of expert x is the CPU decode of a
// 16 x k blob with the same header (rows 16) holding that tile row's code (tile rows are contiguous in the code).
typedef struct {
	const uint8_t* blob;
	int format;
	int e, n_rows, k, n;
	uint64_t scale_offset;
	int8_t** tile_rows;
} ftq_ref_t;

static ftq_ref_t ftq_ref_new(const ccv_nnc_tensor_t* const host)
{
	const int format = host->info.reserved;
	ftq_ref_t ref = {
		.blob = host->data.u8,
		.format = format,
		.e = host->info.dim[0],
		.n_rows = host->info.dim[1],
		.k = host->info.dim[2],
		.n = ccv_nnc_8i_rowwise_x_ftq_rate(format),
	};
	ref.scale_offset = ccv_nnc_8i_rowwise_x_ftq_schedule(format, (uint64_t)ref.e * ref.n_rows, ref.k).scale_offset;
	ref.tile_rows = (int8_t**)cccalloc((size_t)ref.e * (ref.n_rows / 16), sizeof(int8_t*));
	return ref;
}

static void ftq_ref_free(ftq_ref_t* const ref)
{
	size_t i;
	for (i = 0; i < (size_t)ref->e * (ref->n_rows / 16); i++)
		if (ref->tile_rows[i])
			ccfree(ref->tile_rows[i]);
	ccfree(ref->tile_rows);
}

// Codes of row `row` of expert `x`.
static const int8_t* ftq_ref_row(ftq_ref_t* const ref, const int x, const int row)
{
	const size_t tr = (size_t)x * (ref->n_rows / 16) + row / 16;
	if (!ref->tile_rows[tr])
	{
		const size_t tile_row_bytes = (size_t)(ref->k / 16) * 2 * ref->n;
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(ref->format, 16, ref->k);
		uint8_t* const slice = (uint8_t*)cccalloc(schedule.scale_offset, 1);
		memcpy(slice, ref->blob, 32);
		const uint32_t rows = 16;
		memcpy(slice + 8, &rows, 4);
		memcpy(slice + 32, ref->blob + 32 + tr * tile_row_bytes, tile_row_bytes);
		ref->tile_rows[tr] = (int8_t*)ccmalloc((size_t)16 * ref->k);
		ftq_cpu_codes(slice, ref->format, 16, ref->k, ref->tile_rows[tr]);
		ccfree(slice);
	}
	return ref->tile_rows[tr] + (size_t)(row % 16) * ref->k;
}

static float ftq_ref_scale(const ftq_ref_t* const ref, const int x, const int row)
{
	float scale;
	ccv_half_precision_to_float((const uint16_t*)(ref->blob + ref->scale_offset) + (size_t)x * ref->n_rows + row, &scale, 1);
	return scale;
}

// Values exactly representable in the io datatype.
static void io_round(const int datatype, float* const values, const size_t count)
{
	if (datatype != CCV_16F)
		return;
	uint16_t* const h = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	ccv_float_to_half_precision(values, h, count);
	ccv_half_precision_to_float(h, values, count);
	ccfree(h);
}

static ccv_nnc_tensor_t* gpu_from_floats(const int datatype, const float* const values, const int rows, const int cols)
{
	ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
	params.datatype = datatype;
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, params, 0);
	if (datatype == CCV_16F)
		ccv_float_to_half_precision(values, (uint16_t*)host->data.u8, (size_t)rows * cols);
	else
		memcpy(host->data.f32, values, sizeof(float) * rows * cols);
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const gpu = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(gpu), 0);
	ccv_nnc_tensor_free(host);
	return gpu;
}

static void floats_from_gpu(ccv_nnc_tensor_t* const gpu, float* const values)
{
	ccv_nnc_tensor_param_t params = gpu->info;
	params.type = CCV_TENSOR_CPU_MEMORY;
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu), TENSOR_LIST(host), 0);
	const size_t count = ccv_nnc_tensor_count(params);
	if (params.datatype == CCV_16F)
		ccv_half_precision_to_float((uint16_t*)host->data.u8, values, count);
	else
		memcpy(values, host->data.f32, sizeof(float) * count);
	ccv_nnc_tensor_free(host);
}

static ccv_nnc_tensor_t* gpu_ints(const int* const values, const int count)
{
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, count), 0);
	memcpy(host->data.i32, values, sizeof(int) * count);
	ccv_nnc_tensor_t* const gpu = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, count), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(gpu), 0);
	ccv_nnc_tensor_free(host);
	return gpu;
}

static void set_no_neural_accelerators(const int on, uint64_t* const old_flags)
{
	*old_flags = ccv_nnc_flags();
	if (on)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

static void restore_flags(const uint64_t old_flags)
{
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

// A routed batch: `bins` bins (expert indices[b], counts[b] rows, possibly 0) over `rows` routed rows, each routed row
// with its own activation row (or one broadcast row).
typedef struct {
	int bins;
	int rows;
	int* indices;
	int* counts;
	int* row_expert; // The expert of every routed row.
} routing_t;

static void routing_free(routing_t* const r)
{
	ccfree(r->indices);
	ccfree(r->counts);
	ccfree(r->row_expert);
}

static void routing_finish(routing_t* const r)
{
	int b, i, row = 0;
	r->rows = 0;
	for (b = 0; b < r->bins; b++)
		r->rows += r->counts[b];
	r->row_expert = (int*)ccmalloc(sizeof(int) * ccv_max(r->rows, 1));
	for (b = 0; b < r->bins; b++)
		for (i = 0; i < r->counts[b]; i++)
			r->row_expert[row++] = r->indices[b];
}

// Explicit bins.
static routing_t routing_of(const int bins, const int* const indices, const int* const counts)
{
	routing_t r = { .bins = bins };
	r.indices = (int*)ccmalloc(sizeof(int) * bins);
	r.counts = (int*)ccmalloc(sizeof(int) * bins);
	memcpy(r.indices, indices, sizeof(int) * bins);
	memcpy(r.counts, counts, sizeof(int) * bins);
	routing_finish(&r);
	return r;
}

// tokens tokens routed to top_k distinct experts each out of `experts`: one bin per used expert in a shuffled order,
// then an empty bin.
static routing_t routing_top_k(const int tokens, const int top_k, const int experts, uint32_t seed)
{
	int* const counts = (int*)cccalloc(experts, sizeof(int));
	int t, j, i;
	for (t = 0; t < tokens; t++)
	{
		int chosen[64];
		for (j = 0; j < top_k; j++)
		{
			int x, again;
			do {
				x = (int)(rng_next(&seed) % experts);
				again = 0;
				for (i = 0; i < j; i++)
					again |= chosen[i] == x;
			} while (again);
			chosen[j] = x;
			counts[x]++;
		}
	}
	int* const order = (int*)ccmalloc(sizeof(int) * experts);
	for (i = 0; i < experts; i++)
		order[i] = i;
	for (i = experts - 1; i > 0; i--)
	{
		j = (int)(rng_next(&seed) % (i + 1));
		const int tmp = order[i];
		order[i] = order[j];
		order[j] = tmp;
	}
	routing_t r = { .bins = 0 };
	r.indices = (int*)ccmalloc(sizeof(int) * (experts + 1));
	r.counts = (int*)ccmalloc(sizeof(int) * (experts + 1));
	int empty = -1;
	for (i = 0; i < experts; i++)
		if (counts[order[i]])
		{
			r.indices[r.bins] = order[i];
			r.counts[r.bins++] = counts[order[i]];
		} else if (empty < 0)
			empty = order[i];
	r.indices[r.bins] = empty >= 0 ? empty : order[0];
	r.counts[r.bins++] = 0;
	ccfree(order);
	ccfree(counts);
	routing_finish(&r);
	return r;
}

// Activation rows in [-0.5, 0.5), representable in the io type.
static float* activations_new(const int datatype, const int rows, const int k, uint32_t seed)
{
	float* const a = (float*)ccmalloc(sizeof(float) * rows * k);
	size_t i;
	for (i = 0; i < (size_t)rows * k; i++)
		a[i] = (float)((int)(rng_next(&seed) % 1024) - 512) / 1024.0f;
	io_round(datatype, a, (size_t)rows * k);
	return a;
}

// One output of x = activation row `a` against weight row `row` of expert `x`, and its tolerance. w8a8 emulates the
// W8A8 kernels: the activation row is quantized per row to int8 (scale max |a| / 127 in the io type, codes rint(a 127 /
// max |a|)) and the integer products are exact; otherwise (the fused GEMV) the int8 codes take the activations as they
// are. The tolerance covers float accumulation (relative to the sum of |terms|), for FP16 the output's rounding (also of
// the SwiGLU's FP16 gate and up), and one code either way for an activation within 1e-4 of a rounding tie (qa_tie: the
// kernel without neural accelerators divides 127 / max |a| to within an ulp, which moves such ties).
static double ref_dot(ftq_ref_t* const ref, const int x, const int row, const float* const a, const int8_t* const qa, const uint8_t* const qa_tie, const float a_scale, const int datatype, double* const tol)
{
	const int8_t* const q = ftq_ref_row(ref, x, row);
	const double scale = ftq_ref_scale(ref, x, row);
	double sum = 0, mag = 0;
	int i;
	if (qa)
	{
		int64_t isum = 0, imag = 0, ties = 0;
		for (i = 0; i < ref->k; i++)
		{
			isum += (int64_t)q[i] * qa[i];
			imag += llabs((int64_t)q[i] * qa[i]);
			ties += qa_tie[i] ? abs(q[i]) : 0;
		}
		sum = (double)isum * a_scale * scale;
		mag = (double)imag * a_scale * scale;
		// A tie that moves changes the sum by its weight code; a quarter more keeps the float rounding of the rest apart.
		*tol = 1.25 * (double)ties * a_scale * scale;
	} else {
		for (i = 0; i < ref->k; i++)
		{
			sum += (double)q[i] * a[i];
			mag += fabs((double)q[i] * a[i]);
		}
		sum *= scale;
		mag *= scale;
	}
	// FP16 rounds to 2^-11 relative, or 2^-24 apart in the subnormal range.
	*tol = (qa ? *tol : 0) + 2e-6 * mag + 2e-6 * fabs(sum) + (datatype == CCV_16F ? fabs(sum) / 1024 + 0x1p-24 : 0) + 1e-9;
	return sum;
}

// The W8A8 kernels' per-row int8 activation codes and scale, and which codes sit at a rounding tie.
static float quantize_row(const float* const a, const int k, const int datatype, int8_t* const qa, uint8_t* const qa_tie)
{
	float max_abs = 0;
	int i;
	for (i = 0; i < k; i++)
		max_abs = ccv_max(max_abs, fabsf(a[i]));
	const float inv_scale = max_abs > 0 ? 127.0f / max_abs : 127.0f;
	for (i = 0; i < k; i++)
	{
		const float scaled = a[i] * inv_scale;
		const int v = (int)rintf(scaled);
		qa[i] = (int8_t)ccv_clamp(v, -127, 127);
		qa_tie[i] = fabsf(scaled - floorf(scaled) - 0.5f) < 1e-4f;
	}
	float scale = max_abs > 0 ? max_abs / 127.0f : 1.0f / 127.0f;
	io_round(datatype, &scale, 1);
	return scale;
}

// The rows to check: all of them up to 64, else the first and last row of every bin and every 37th row.
static int check_row(const routing_t* const r, const int row)
{
	if (r->rows <= 64 || row % 37 == 0)
		return 1;
	int b, start = 0;
	for (b = 0; b < r->bins; b++)
	{
		if (r->counts[b] && (row == start || row == start + r->counts[b] - 1))
			return 1;
		start += r->counts[b];
	}
	return 0;
}

// The output columns to check: all of them up to 256, else 4 tile rows (the first two, one in the middle, the last).
static int check_col(const int n_rows, const int col)
{
	if (n_rows <= 256)
		return 1;
	const int tr = col / 16, last = n_rows / 16 - 1;
	return tr == 0 || tr == 1 || tr == last / 2 || tr == last;
}

static int run_segmented_gemm(const int datatype, const routing_t* const r, const float* const a, const int k, const int n_rows, ccv_nnc_tensor_t* const w, ccv_nnc_tensor_t* const bias, const int no_na, float* const out)
{
	ccv_nnc_tensor_t* const ga = gpu_from_floats(datatype, a, r->rows, k);
	ccv_nnc_tensor_t* const indices = gpu_ints(r->indices, r->bins);
	ccv_nnc_tensor_t* const counts = gpu_ints(r->counts, r->bins);
	ccv_nnc_tensor_param_t bp = GPU_TENSOR_NHWC(000, 32F, r->rows, n_rows);
	bp.datatype = datatype;
	ccv_nnc_tensor_t* const gb = ccv_nnc_tensor_new(0, bp, 0);
	ccv_nnc_cmd_t cmd = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	cmd.backend = CCV_NNC_BACKEND_MPS;
	uint64_t old_flags;
	set_no_neural_accelerators(no_na, &old_flags);
	const int status = bias ?
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, w, bias), TENSOR_LIST(gb), 0) :
		ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, w), TENSOR_LIST(gb), 0);
	restore_flags(old_flags);
	if (status == CCV_NNC_EXEC_SUCCESS && out)
		floats_from_gpu(gb, out);
	ccv_nnc_tensor_free(gb);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ga);
	return status;
}

static const float swiglu_clamp = 10;

// a has input_rows rows (r->rows, or 1 broadcast to every routed row).
static int run_segmented_swiglu_cmd(ccv_nnc_cmd_t cmd, const int datatype, const routing_t* const r, const float* const a, const int input_rows, const float* const route, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, const int no_na, float* const out);

static int run_segmented_swiglu(const int datatype, const routing_t* const r, const float* const a, const int input_rows, const float* const route, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, const int no_na, float* const out)
{
	return run_segmented_swiglu_cmd(CMD_SEGMENTED_SWIGLU_FORWARD(swiglu_clamp), datatype, r, a, input_rows, route, k, n_rows, gate, up, no_na, out);
}

static int run_segmented_swiglu_cmd(ccv_nnc_cmd_t cmd, const int datatype, const routing_t* const r, const float* const a, const int input_rows, const float* const route, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, const int no_na, float* const out)
{
	ccv_nnc_tensor_t* const ga = gpu_from_floats(datatype, a, input_rows, k);
	ccv_nnc_tensor_t* const indices = gpu_ints(r->indices, r->bins);
	ccv_nnc_tensor_t* const counts = gpu_ints(r->counts, r->bins);
	ccv_nnc_tensor_t* const groute = gpu_from_floats(datatype, route, r->rows, 1);
	ccv_nnc_tensor_param_t op = GPU_TENSOR_NHWC(000, 32F, r->rows, n_rows);
	op.datatype = datatype;
	ccv_nnc_tensor_t* const go = ccv_nnc_tensor_new(0, op, 0);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	uint64_t old_flags;
	set_no_neural_accelerators(no_na, &old_flags);
	const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, gate, up, groute), TENSOR_LIST(go), 0);
	restore_flags(old_flags);
	if (status == CCV_NNC_EXEC_SUCCESS && out)
		floats_from_gpu(go, out);
	ccv_nnc_tensor_free(go);
	ccv_nnc_tensor_free(groute);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ga);
	return status;
}

// Mismatches of a segmented GEMM output against the reference (checked rows and columns); *worst is the largest
// |error| / tolerance.
static int check_segmented_gemm(ftq_ref_t* const ref, const routing_t* const r, const float* const a, const int datatype, const int w8a8, const float* const out, double* const worst)
{
	const int k = ref->k, n_rows = ref->n_rows;
	int8_t* const qa = (int8_t*)ccmalloc(k);
	uint8_t* const qa_tie = (uint8_t*)ccmalloc(k);
	int mismatches = 0, row, col;
	*worst = 0;
	for (row = 0; row < r->rows; row++)
	{
		if (!check_row(r, row))
			continue;
		const float* const ar = a + (size_t)row * k;
		const float a_scale = w8a8 ? quantize_row(ar, k, datatype, qa, qa_tie) : 0;
		for (col = 0; col < n_rows; col++)
		{
			if (!check_col(n_rows, col))
				continue;
			double tol;
			const double want = ref_dot(ref, r->row_expert[row], col, ar, w8a8 ? qa : 0, qa_tie, a_scale, datatype, &tol);
			const double err = fabs((double)out[(size_t)row * n_rows + col] - want);
			if (!(err <= tol))
			{
				if (mismatches < 4)
					printf("  gemm mismatch row %d (expert %d) col %d: %g vs %g (tol %g)\n", row, r->row_expert[row], col, out[(size_t)row * n_rows + col], want, tol);
				mismatches++;
			}
			*worst = ccv_max(*worst, err / tol);
		}
	}
	ccfree(qa_tie);
	ccfree(qa);
	return mismatches;
}

static int check_segmented_swiglu(ftq_ref_t* const gate, ftq_ref_t* const up, const routing_t* const r, const float* const a, const int input_rows, const float* const route, const int datatype, const int w8a8, const float* const out, double* const worst)
{
	const int k = gate->k, n_rows = gate->n_rows;
	int8_t* const qa = (int8_t*)ccmalloc(k);
	uint8_t* const qa_tie = (uint8_t*)ccmalloc(k);
	int mismatches = 0, row, col;
	*worst = 0;
	for (row = 0; row < r->rows; row++)
	{
		if (!check_row(r, row))
			continue;
		const float* const ar = a + (size_t)(input_rows == 1 ? 0 : row) * k;
		const float a_scale = w8a8 ? quantize_row(ar, k, datatype, qa, qa_tie) : 0;
		for (col = 0; col < n_rows; col++)
		{
			if (!check_col(n_rows, col))
				continue;
			double gate_tol, up_tol;
			const double g = ref_dot(gate, r->row_expert[row], col, ar, w8a8 ? qa : 0, qa_tie, a_scale, datatype, &gate_tol);
			const double u = ref_dot(up, r->row_expert[row], col, ar, w8a8 ? qa : 0, qa_tie, a_scale, datatype, &up_tol);
			// SwiGLU with the clamp: up into [-clamp, clamp], gate to at most clamp; both are 1-Lipschitz, and so
			// (with |d(g sigmoid(g)) / dg| <= 1.1) is the tolerance's propagation.
			const double gc = ccv_min(g, swiglu_clamp), uc = ccv_min(ccv_max(u, -swiglu_clamp), swiglu_clamp);
			const double swish = gc / (1 + exp(-gc));
			const double want = route[row] * uc * swish;
			const double tol = fabs(route[row]) * (fabs(swish) * up_tol + (fabs(uc) + up_tol) * 1.1 * gate_tol) +
				(datatype == CCV_16F ? fabs(want) / 1024 + 0x1p-24 : 1e-6 * fabs(want)) + 1e-9;
			const double err = fabs((double)out[(size_t)row * n_rows + col] - want);
			if (!(err <= tol))
			{
				if (mismatches < 4)
					printf("  swiglu mismatch row %d (expert %d) col %d: %g vs %g (tol %g)\n", row, r->row_expert[row], col, out[(size_t)row * n_rows + col], want, tol);
				mismatches++;
			}
			*worst = ccv_max(*worst, err / tol);
		}
	}
	ccfree(qa_tie);
	ccfree(qa);
	return mismatches;
}

static float* route_weights_new(const int datatype, const int rows)
{
	float* const route = (float*)ccmalloc(sizeof(float) * ccv_max(rows, 1));
	int i;
	for (i = 0; i < rows; i++)
		route[i] = (float)(1 + (i * 7) % 13) / 16.0f;
	io_round(datatype, route, rows);
	return route;
}

static const char* dtype_name(const int datatype)
{
	return datatype == CCV_16F ? "fp16" : "fp32";
}

TEST_CASE("ftq container: fp32 base keeps the fp16 blob's size and the scale size is 2")
{
	static const int rates[] = { 16, 34, 46, 48 };
	static const int shapes[][3] = { { 1, 32, 16 }, { 3, 48, 272 }, { 8, 64, 1152 }, { 128, 3072, 2304 } };
	int i, s;
	for (i = 0; i < 4; i++)
		for (s = 0; s < 4; s++)
		{
			const int format = ftq_format(rates[i]);
			const ccv_nnc_tensor_param_t h = ftq_param(CCV_TENSOR_CPU_MEMORY, CCV_16F, shapes[s][0], shapes[s][1], shapes[s][2], format);
			const ccv_nnc_tensor_param_t f = ftq_param(CCV_TENSOR_CPU_MEMORY, CCV_32F, shapes[s][0], shapes[s][1], shapes[s][2], format);
			const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, (uint64_t)shapes[s][0] * shapes[s][1], shapes[s][2]);
			const size_t want = schedule.scale_offset + 2 * (size_t)shapes[s][0] * shapes[s][1];
			REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(h), want, "FP16 FTQ n%d [%d, %d, %d] is the schedule's blob", rates[i], shapes[s][0], shapes[s][1], shapes[s][2]);
			REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(f), want, "FP32 FTQ n%d [%d, %d, %d] is the same blob", rates[i], shapes[s][0], shapes[s][1], shapes[s][2]);
			REQUIRE_EQ(ccv_nnc_8i_rowwise_x_scale_size(format, CCV_32F), 2, "FTQ scales are FP16");
		}
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_scale_size(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, CCV_32F), 4, "TMCQ scales take the base datatype");
	REQUIRE_EQ(ccv_nnc_8i_rowwise_x_scale_size(CCV_NNC_QX_8I_ROWWISE_Q4_K, CCV_16F), 2, "Q4_K FP16 scales");
	// Other codecs keep their sizes: 8I_ROWWISE_X TMCQ FP32 scales after the payload.
	const ccv_nnc_tensor_param_t t = ccv_nnc_tensor_8i_rowwise_x(CPU_TENSOR_NHWC(32F, 64, 512), CCV_NNC_QX_8I_ROWWISE_TMCQ_N22);
	REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(t), (((size_t)64 * 512 * 22 / 128 + 127) & ~(size_t)127) + 64 * 4, "TMCQ FP32 size is unchanged");
}

// The selected-expert int8 decode: experts 6, 0 and 3 of 7 (bin 3 routes expert 2 with no rows) decode to the bank's
// int8 rows bit for bit with the CPU decode of the whole (7 x 32) x K blob; their scales are the blob's FP16 scales
// (FP16 output) or those as FP32; every other expert keeps the sentinel.
static int selected_decode_case(const int datatype, const int k, const int n, const ftq_table_t* const table, const uint32_t seed)
{
	const int experts = 7, n_rows = 32, rows = experts * n_rows;
	ccv_nnc_tensor_t* hq;
	ccv_nnc_tensor_t* const gq = ftq_weight_new(datatype, experts, n_rows, k, n, table, seed, &hq);
	const int format = hq->info.reserved;
	int8_t* const q8 = (int8_t*)ccmalloc((size_t)rows * k);
	ftq_cpu_codes(hq->data.u8, format, rows, k, q8);
	const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(format, rows, k);
	static const int indices[4] = { 6, 0, 3, 2 };
	static const int counts[4] = { 2, 1, 5, 0 };
	ccv_nnc_tensor_t* const gindices = gpu_ints(indices, 4);
	ccv_nnc_tensor_t* const gcounts = gpu_ints(counts, 4);
	const size_t scale_size = datatype == CCV_16F ? 2 : 4;
	const size_t out_scale_offset = ((size_t)rows * k + 127) & ~(size_t)127;
	const size_t out_bytes = out_scale_offset + rows * scale_size;
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, (int)((out_bytes + 3) / 4)), 0);
	memset(hout->data.u8, 0x5a, out_bytes);
	ccv_nnc_tensor_t* const gout = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, (int)((out_bytes + 3) / 4)), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hout), TENSOR_LIST(gout), 0);
	ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
	uint32_t mfa_format = (uint32_t)ccv_nnc_8i_rowwise_x_kernel_format(format);
	mtl_buffer_t* codebook = 0;
	int mismatches = 0;
	if (!ccv_nnc_mfa_codebook_format(context, table->slot, 1, &mfa_format, &codebook))
		mismatches = -1;
	else {
		const ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t params = {
			.data_type = datatype == CCV_16F ? 16 : 3, // MTL::DataTypeHalf / MTL::DataTypeFloat
			.format = mfa_format,
			.row_length = k,
			.rows_per_expert = n_rows,
			.expert_count = experts,
			.bincount = 4,
			.codebook = codebook,
		};
		mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(params));
		mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(0);
		mtl_buffer_t* tensors[6] = {
			(mtl_buffer_t*)gq->data.u8, (mtl_buffer_t*)gindices->data.u8, (mtl_buffer_t*)gcounts->data.u8, (mtl_buffer_t*)gout->data.u8, scratch, 0,
		};
		size_t offsets[5] = { (size_t)gq->dataof, (size_t)gindices->dataof, (size_t)gcounts->dataof, (size_t)gout->dataof, 0 };
		ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(context, params, command_batch, tensors, offsets);
		ccv_nnc_stream_context_finish_command_batch(0, command_batch);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gout), TENSOR_LIST(hout), 0);
		int x;
		for (x = 0; x < experts; x++)
		{
			const int selected = x == 6 || x == 0 || x == 3;
			const size_t base = (size_t)x * n_rows * k;
			size_t i;
			for (i = 0; i < (size_t)n_rows * k; i++)
				mismatches += hout->data.u8[base + i] != (selected ? (uint8_t)q8[base + i] : 0x5a);
			for (i = 0; i < (size_t)n_rows; i++)
			{
				const uint8_t* const got = hout->data.u8 + out_scale_offset + ((size_t)x * n_rows + i) * scale_size;
				const uint16_t h = ((const uint16_t*)(hq->data.u8 + schedule.scale_offset))[(size_t)x * n_rows + i];
				if (!selected)
				{
					size_t j;
					for (j = 0; j < scale_size; j++)
						mismatches += got[j] != 0x5a;
				} else if (datatype == CCV_16F)
					mismatches += memcmp(got, &h, 2) != 0;
				else {
					float f;
					ccv_half_precision_to_float(&h, &f, 1);
					mismatches += memcmp(got, &f, 4) != 0;
				}
			}
		}
	}
	ccv_nnc_tensor_free(gout);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccfree(q8);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
	return mismatches;
}

TEST_CASE("mps ftq selected expert int8 decode is bit exact with the cpu decode")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int ks[] = { 16, 272, 1152, 2304 };
	static const int rates[] = { 16, 18, 34, 46, 48 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	REQUIRE(external.slot > 0 && folded.slot > 0, "the external and folded tables register");
	int ki, ri, d, t;
	for (ki = 0; ki < 4; ki++)
		for (ri = 0; ri < 5; ri++)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 3; t++)
				{
					const int datatype = d ? CCV_32F : CCV_16F;
					// Every table at the priority rates, the built-in one at the others.
					if (t > 0 && rates[ri] != 34 && rates[ri] != 48)
						continue;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
					const int mismatches = selected_decode_case(datatype, ks[ki], rates[ri], table, 0x5e1ec7u + ki * 131 + ri * 17 + d * 7 + t);
					REQUIRE_EQ(mismatches, 0, "FTQ selected decode %s K=%d n%d table %d", dtype_name(datatype), ks[ki], rates[ri], t);
				}
}

// Segmented GEMM over one routing on the GEMV path (one row per bin) or the W8A8 path.
static int gemm_case(const int datatype, const int experts, const int n_rows, const int k, const int n, const ftq_table_t* const table, const routing_t* const r, const int w8a8, const int no_na, const uint32_t seed, ccv_nnc_tensor_t* const shared_w, ccv_nnc_tensor_t* const shared_hq, double* const worst)
{
	ccv_nnc_tensor_t* hq = shared_hq;
	ccv_nnc_tensor_t* const gq = shared_w ? shared_w : ftq_weight_new(datatype, experts, n_rows, k, n, table, seed, &hq);
	float* const a = activations_new(datatype, r->rows, k, seed * 3 + 1);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	int mismatches = -1;
	if (run_segmented_gemm(datatype, r, a, k, n_rows, gq, 0, no_na, out) == CCV_NNC_EXEC_SUCCESS)
	{
		ftq_ref_t ref = ftq_ref_new(hq);
		mismatches = check_segmented_gemm(&ref, r, a, datatype, w8a8, out, worst);
		ftq_ref_free(&ref);
		if (getenv("FTQ_SEGMENTED_VERBOSE"))
			printf("\n  gemm %s n%d %d x %d rows %d bins %d %s%s: worst |error| / tolerance %.3g", dtype_name(datatype), ref.n, n_rows, k, r->rows, r->bins, w8a8 ? "w8a8" : "gemv", no_na ? " (no NA)" : "", *worst);
	}
	ccfree(out);
	ccfree(a);
	if (!shared_w)
	{
		ccv_nnc_tensor_free(gq);
		ccv_nnc_tensor_free(hq);
	}
	return mismatches;
}

static int swiglu_case(const int datatype, const int input_rows, const routing_t* const r, const int w8a8, const int no_na, const uint32_t seed, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const gate_host, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const up_host, double* const worst)
{
	const int n_rows = gate_host->info.dim[1], k = gate_host->info.dim[2];
	float* const a = activations_new(datatype, input_rows, k, seed * 5 + 2);
	float* const route = route_weights_new(datatype, r->rows);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	int mismatches = -1;
	if (run_segmented_swiglu(datatype, r, a, input_rows, route, k, n_rows, gate, up, no_na, out) == CCV_NNC_EXEC_SUCCESS)
	{
		ftq_ref_t gref = ftq_ref_new(gate_host), uref = ftq_ref_new(up_host);
		mismatches = check_segmented_swiglu(&gref, &uref, r, a, input_rows, route, datatype, w8a8, out, worst);
		if (getenv("FTQ_SEGMENTED_VERBOSE"))
			printf("\n  swiglu %s n%d %d x %d rows %d (input %d) bins %d %s%s: worst |error| / tolerance %.3g", dtype_name(datatype), gref.n, n_rows, k, r->rows, input_rows, r->bins, w8a8 ? "w8a8" : "gemv", no_na ? " (no NA)" : "", *worst);
		ftq_ref_free(&uref);
		ftq_ref_free(&gref);
	}
	ccfree(out);
	ccfree(route);
	ccfree(a);
	return mismatches;
}

// Small banks over every K of the grid: 6 experts of 48 rows, one row per bin (GEMV: as many rows as bins, so no empty
// bin) and 2 - 7 rows per bin with an empty bin (W8A8).
static const int small_gemv_indices[] = { 5, 0, 3, 2 };
static const int small_gemv_counts[] = { 1, 1, 1, 1 };
static const int small_w8a8_indices[] = { 5, 0, 3, 2, 4 };
static const int small_w8a8_counts[] = { 3, 7, 0, 2, 5 };

TEST_CASE("mps ftq segmented gemm gemv and w8a8 paths match the fp32 reference across K and rates")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int ks[] = { 16, 272, 1152, 2304 };
	static const int rates[] = { 16, 18, 34, 46, 48 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t w8a8 = routing_of(5, small_w8a8_indices, small_w8a8_counts);
	const int na = ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context());
	int ki, ri, d, t;
	for (ki = 0; ki < 4; ki++)
		for (ri = 0; ri < 5; ri++)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 3; t++)
				{
					if (t > 0 && rates[ri] != 34 && rates[ri] != 48)
						continue;
					const int datatype = d ? CCV_32F : CCV_16F;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
					const uint32_t seed = 0x6e3a00u + ki * 97 + ri * 13 + d * 5 + t;
					ccv_nnc_tensor_t* hq;
					ccv_nnc_tensor_t* const gq = ftq_weight_new(datatype, 6, 48, ks[ki], rates[ri], table, seed, &hq);
					double worst = 0;
					int m = gemm_case(datatype, 6, 48, ks[ki], rates[ri], table, &gemv, 0, 0, seed, gq, hq, &worst);
					REQUIRE_EQ(m, 0, "FTQ segmented GEMV %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					if (na || datatype == CCV_32F)
					{
						m = gemm_case(datatype, 6, 48, ks[ki], rates[ri], table, &w8a8, 1, 0, seed + 1, gq, hq, &worst);
						REQUIRE_EQ(m, 0, "FTQ segmented W8A8 GEMM %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					}
					// Without neural accelerators FP32 takes the Int8MatMul segmented path; FP16 has no FTQ route.
					if (datatype == CCV_32F && t != 1)
					{
						m = gemm_case(datatype, 6, 48, ks[ki], rates[ri], table, &w8a8, 1, 1, seed + 2, gq, hq, &worst);
						REQUIRE_EQ(m, 0, "FTQ segmented W8A8 GEMM without neural accelerators %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					}
					ccv_nnc_tensor_free(gq);
					ccv_nnc_tensor_free(hq);
				}
	routing_free(&w8a8);
	routing_free(&gemv);
}

TEST_CASE("mps ftq segmented swiglu gemv and w8a8 paths match the fp32 reference across K and rates")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int ks[] = { 16, 272, 1152, 2304 };
	static const int rates[] = { 16, 18, 34, 46, 48 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	// 18 rows over 5 bins: the direct path takes up to 3 rows per bin; 6 experts' worth of rows past it is W8A8.
	routing_t direct = routing_of(5, small_w8a8_indices, (const int[]){ 3, 3, 0, 2, 3 });
	static const int prefill_counts[] = { 30, 41, 0, 25, 33 };
	routing_t w8a8 = routing_of(5, small_w8a8_indices, prefill_counts);
	const int na = ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context());
	int ki, ri, d, t;
	for (ki = 0; ki < 4; ki++)
		for (ri = 0; ri < 5; ri++)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 3; t++)
				{
					if (t > 0 && rates[ri] != 34 && rates[ri] != 48)
						continue;
					const int datatype = d ? CCV_32F : CCV_16F;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
					const uint32_t seed = 0x5a1900u + ki * 97 + ri * 13 + d * 5 + t;
					ccv_nnc_tensor_t* gate_host;
					ccv_nnc_tensor_t* up_host;
					ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, 48, ks[ki], rates[ri], table, seed, &gate_host);
					ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, 48, ks[ki], rates[ri], table, seed + 1000, &up_host);
					double worst = 0;
					int m = swiglu_case(datatype, gemv.rows, &gemv, 0, 0, seed, gate, gate_host, up, up_host, &worst);
					REQUIRE_EQ(m, 0, "FTQ segmented SwiGLU GEMVs %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					m = swiglu_case(datatype, 1, &gemv, 0, 0, seed + 3, gate, gate_host, up, up_host, &worst);
					REQUIRE_EQ(m, 0, "FTQ segmented SwiGLU GEMVs, broadcast input, %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					m = swiglu_case(datatype, direct.rows, &direct, 0, 0, seed + 4, gate, gate_host, up, up_host, &worst);
					REQUIRE_EQ(m, 0, "FTQ segmented SwiGLU GEMVs, 3 rows per bin, %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					if (na || datatype == CCV_32F)
					{
						m = swiglu_case(datatype, w8a8.rows, &w8a8, 1, 0, seed + 1, gate, gate_host, up, up_host, &worst);
						REQUIRE_EQ(m, 0, "FTQ segmented W8A8 SwiGLU %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					}
					if (datatype == CCV_32F && t != 1)
					{
						m = swiglu_case(datatype, w8a8.rows, &w8a8, 1, 1, seed + 2, gate, gate_host, up, up_host, &worst);
						REQUIRE_EQ(m, 0, "FTQ segmented W8A8 SwiGLU without neural accelerators %s K=%d n%d table %d (worst %.3g)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					}
					ccv_nnc_tensor_free(up);
					ccv_nnc_tensor_free(up_host);
					ccv_nnc_tensor_free(gate);
					ccv_nnc_tensor_free(gate_host);
				}
	routing_free(&w8a8);
	routing_free(&direct);
	routing_free(&gemv);
}

// The packed segmented SwiGLU on FTQ (small batches: gate and up of every route in one kernel) against its definition,
// two segmented FTQ GEMVs and the SwiGLU of their outputs. The GEMVs run as SEGMENTED_GEMM_FORWARD with one row per bin
// (its GEMV path) on every routed row's own expert and activation row, on FP32-base twins of the weights (the blob is
// the same bytes whatever the base), so that for FP16 too they give gate and up as floats before any rounding to the io
// type, which is what the fused kernel holds. The SwiGLU follows the kernel's float arithmetic on the CPU (the clamp,
// sigmoid as tail + (1 - 2 tail) with tail = 1 / (1 + exp |gate|), ((up gate) sigmoid) route) and rounds to the io
// type once, except exp |gate|: the CPU's expf and Metal's precise::exp differ in the last place for about a quarter of
// these values, so it comes from EWEXP_FORWARD on the GPU, which is precise::exp in float. Every output must match bit
// for bit; the worst ulp distance is reported if one does not.
static int bit_mismatches(const float* const x, const float* const y, const size_t count);

static void gpu_exp(const float* const x, const int count, float* const y)
{
	ccv_nnc_tensor_t* const gx = gpu_from_floats(CCV_32F, x, 1, count);
	ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, count), 0);
	ccv_nnc_cmd_t cmd = CMD_EWEXP_FORWARD();
	cmd.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gx), TENSOR_LIST(gy), 0);
	floats_from_gpu(gy, y);
	ccv_nnc_tensor_free(gy);
	ccv_nnc_tensor_free(gx);
}

static void ftq_swiglu_reference(const float* const g, const float* const u, const float* const route, const int rows, const int n_rows, const int datatype, float* const want)
{
	const int count = rows * n_rows;
	float* const e = (float*)ccmalloc(sizeof(float) * ccv_max(count, 1));
	int i, j;
	for (i = 0; i < count; i++)
		e[i] = fabsf(fminf(g[i], swiglu_clamp));
	gpu_exp(e, count, e);
	for (i = 0; i < rows; i++)
		for (j = 0; j < n_rows; j++)
		{
			const size_t x = (size_t)i * n_rows + j;
			const float gate = fminf(g[x], swiglu_clamp);
			const float up = fminf(fmaxf(u[x], -swiglu_clamp), swiglu_clamp);
			const float tail = 1.0f / (1.0f + e[x]);
			const float rest = 1.0f - 2.0f * tail;
			const float sigmoid = gate >= 0 ? tail + rest : tail;
			const float product = up * gate;
			const float swish = product * sigmoid;
			want[x] = swish * route[i];
		}
	ccfree(e);
	io_round(datatype, want, count);
}

// Activation rows with every mantissa bit of the io type in use and magnitudes over 2^-6 .. 1, so that the dot products
// round and a different summation order shows (activations_new's multiples of 2^-10 keep these sums exact).
static float* activations_fine_new(const int datatype, const int rows, const int k, uint32_t seed)
{
	float* const a = (float*)ccmalloc(sizeof(float) * rows * k);
	size_t i;
	for (i = 0; i < (size_t)rows * k; i++)
	{
		const int m = (int)(rng_next(&seed) & 0xffffff) - 0x800000;
		a[i] = ldexpf((float)m, -23 - (int)(rng_next(&seed) % 7));
	}
	io_round(datatype, a, (size_t)rows * k);
	return a;
}

// The routed rows of r as one row per bin (each row's expert), and their activation rows (a broadcast row repeated).
static routing_t routing_per_row(const routing_t* const r)
{
	int* const counts = (int*)ccmalloc(sizeof(int) * ccv_max(r->rows, 1));
	int i;
	for (i = 0; i < r->rows; i++)
		counts[i] = 1;
	routing_t one = routing_of(r->rows, r->row_expert, counts);
	ccfree(counts);
	return one;
}

// The packed FTQ SwiGLU splits K wider than the GEMV, so its float sums take another order: an output may differ from
// the two GEMVs and the SwiGLU by FTQ_SWIGLU_GEMV_TOLERANCE of the largest output (FP32 sums over K <= 2304 differ by
// a few 1e-7 of it), and with FP16 io by one more FP16 rounding step of its own magnitude. That step would hide an
// extra FP16 rounding inside the kernel, so with FP16 io at most 1 / FTQ_SWIGLU_GEMV_FP16_FLIPS of the outputs may
// differ in any bit (the float sums seldom straddle an FP16 rounding boundary: 4 of 192 at most on M5 Ultra).
#define FTQ_SWIGLU_GEMV_TOLERANCE 1e-5f
#define FTQ_SWIGLU_GEMV_FP16_FLIPS 32

// Outputs of the packed FTQ SwiGLU outside the tolerance of the two GEMVs and the SwiGLU (-1 if a command fails);
// *worst is the largest difference as a fraction of the largest output, *flips the outputs that differ in any bit.
static int ftq_swiglu_gemv_mismatches(const int datatype, const routing_t* const r, const int input_rows, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const gate32, ccv_nnc_tensor_t* const up32, const uint32_t seed, float* const worst, int* const flips)
{
	float* const a = activations_fine_new(datatype, input_rows, k, seed);
	float* const route = route_weights_new(datatype, r->rows);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const g = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const u = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const want = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const rows_a = (float*)ccmalloc(sizeof(float) * r->rows * k);
	int i;
	for (i = 0; i < r->rows; i++)
		memcpy(rows_a + (size_t)i * k, a + (size_t)(input_rows == 1 ? 0 : i) * k, sizeof(float) * k);
	routing_t one = routing_per_row(r);
	int mismatches = -1;
	*worst = 0;
	*flips = 0;
	if (run_segmented_swiglu(datatype, r, a, input_rows, route, k, n_rows, gate, up, 0, out) == CCV_NNC_EXEC_SUCCESS &&
		run_segmented_gemm(CCV_32F, &one, rows_a, k, n_rows, gate32, 0, 0, g) == CCV_NNC_EXEC_SUCCESS &&
		run_segmented_gemm(CCV_32F, &one, rows_a, k, n_rows, up32, 0, 0, u) == CCV_NNC_EXEC_SUCCESS)
	{
		ftq_swiglu_reference(g, u, route, r->rows, n_rows, datatype, want);
		float largest = 0;
		for (i = 0; i < r->rows * n_rows; i++)
			largest = ccv_max(largest, fabsf(want[i]));
		mismatches = 0;
		for (i = 0; i < r->rows * n_rows; i++)
		{
			const float diff = fabsf(out[i] - want[i]);
			const float tolerance = FTQ_SWIGLU_GEMV_TOLERANCE * largest + (datatype == CCV_16F ? ldexpf(fabsf(want[i]), -10) : 0);
			// NaN differences fail too.
			if (!(diff <= tolerance))
			{
				if (mismatches < 4)
					printf("  mismatch at %d: %.9g vs %.9g\n", i, out[i], want[i]);
				mismatches++;
			}
			if (memcmp(out + i, want + i, sizeof(float)) != 0)
				++*flips;
			if (largest > 0)
				*worst = ccv_max(*worst, diff / largest);
		}
		if (datatype == CCV_16F && *flips * FTQ_SWIGLU_GEMV_FP16_FLIPS > r->rows * n_rows)
		{
			printf("  %d of %d FP16 outputs differ in a bit\n", *flips, r->rows * n_rows);
			mismatches += *flips;
		}
		// A zero output everywhere would pass any tolerance.
		if (!(largest > 0))
			mismatches = -1;
	}
	routing_free(&one);
	ccfree(rows_a);
	ccfree(want);
	ccfree(u);
	ccfree(g);
	ccfree(out);
	ccfree(route);
	ccfree(a);
	return mismatches;
}

TEST_CASE("mps ftq packed segmented swiglu matches two segmented ftq gemvs and the swiglu")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	static const int ks[] = { 16, 272, 1152, 2304 };
	static const int rates[] = { 16, 18, 24, 32, 34, 46, 48 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table(), external11 = ftq_external11_table();
	// One row per bin; the same 4 bins with one broadcast input row; 3 rows per bin over 5 bins, one of them empty.
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t direct = routing_of(5, small_w8a8_indices, (const int[]){ 3, 3, 0, 3, 3 });
	int ki, ri, d, t;
	float worst_fp[2] = { 0, 0 };
	float flip_fp[2] = { 0, 0 };
	for (ki = 0; ki < 4; ki++)
		for (ri = 0; ri < 7; ri++)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 4; t++)
				{
					const int datatype = d ? CCV_32F : CCV_16F;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : t == 2 ? &folded : &external11;
					const uint32_t seed = 0x5b1a00u + ki * 97 + ri * 13 + d * 5 + t;
					ccv_nnc_tensor_t* gate_host;
					ccv_nnc_tensor_t* up_host;
					ccv_nnc_tensor_t* gate32_host;
					ccv_nnc_tensor_t* up32_host;
					ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, 48, ks[ki], rates[ri], table, seed, &gate_host);
					ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, 48, ks[ki], rates[ri], table, seed + 1000, &up_host);
					ccv_nnc_tensor_t* const gate32 = ftq_weight_new(CCV_32F, 6, 48, ks[ki], rates[ri], table, seed, &gate32_host);
					ccv_nnc_tensor_t* const up32 = ftq_weight_new(CCV_32F, 6, 48, ks[ki], rates[ri], table, seed + 1000, &up32_host);
					float worst;
					int flips;
					int m = ftq_swiglu_gemv_mismatches(datatype, &gemv, gemv.rows, ks[ki], 48, gate, up, gate32, up32, seed, &worst, &flips);
					REQUIRE_EQ(m, 0, "packed FTQ SwiGLU %s K=%d n%d table %d, one row per bin, should match the GEMVs (worst %.3g of the largest output)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					worst_fp[d] = ccv_max(worst_fp[d], worst);
					flip_fp[d] = ccv_max(flip_fp[d], (float)flips);
					m = ftq_swiglu_gemv_mismatches(datatype, &gemv, 1, ks[ki], 48, gate, up, gate32, up32, seed + 1, &worst, &flips);
					REQUIRE_EQ(m, 0, "packed FTQ SwiGLU %s K=%d n%d table %d, broadcast input, should match the GEMVs (worst %.3g of the largest output)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					worst_fp[d] = ccv_max(worst_fp[d], worst);
					flip_fp[d] = ccv_max(flip_fp[d], (float)flips);
					m = ftq_swiglu_gemv_mismatches(datatype, &direct, direct.rows, ks[ki], 48, gate, up, gate32, up32, seed + 2, &worst, &flips);
					REQUIRE_EQ(m, 0, "packed FTQ SwiGLU %s K=%d n%d table %d, 3 rows per bin and an empty bin, should match the GEMVs (worst %.3g of the largest output)", dtype_name(datatype), ks[ki], rates[ri], t, worst);
					worst_fp[d] = ccv_max(worst_fp[d], worst);
					flip_fp[d] = ccv_max(flip_fp[d], (float)flips);
					ccv_nnc_tensor_free(up32);
					ccv_nnc_tensor_free(up32_host);
					ccv_nnc_tensor_free(gate32);
					ccv_nnc_tensor_free(gate32_host);
					ccv_nnc_tensor_free(up);
					ccv_nnc_tensor_free(up_host);
					ccv_nnc_tensor_free(gate);
					ccv_nnc_tensor_free(gate_host);
				}
	printf("  packed FTQ SwiGLU against the GEMVs: worst %.3g (FP16) and %.3g (FP32) of the largest output, at most %g (FP16) and %g (FP32) outputs off in a bit\n", worst_fp[0], worst_fp[1], flip_fp[0], flip_fp[1]);
	routing_free(&direct);
	routing_free(&gemv);
}

// Soft caps on FTQ routed experts: every path that takes them (the packed SwiGLU for a few rows, the W8A8 paths for a
// prefill) applies them, or the command is rejected; none drops them. The reference takes the two FP32 GEMVs.
TEST_CASE("mps ftq segmented swiglu applies its soft caps or rejects them")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	enum { k = 256, n_rows = 48 };
	static const float caps[][2] = { { 0.05f, 0 }, { 0, 0.05f }, { 0.05f, 0.04f } };
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t prefill = routing_of(5, small_w8a8_indices, (const int[]){ 30, 41, 0, 25, 33 });
	int d, no_na, c, p, ran = 0, runs = 0;
	for (d = 0; d < 2; d++)
	{
		const int datatype = d ? CCV_16F : CCV_32F;
		ccv_nnc_tensor_t* gate_host;
		ccv_nnc_tensor_t* up_host;
		ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, n_rows, k, 20, &ftq_builtin, 0x50f7ca00u, &gate_host);
		ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, n_rows, k, 20, &ftq_builtin, 0x50f7ca01u, &up_host);
		for (no_na = 0; no_na < 2; no_na++)
			for (p = 0; p < 2; p++)
				for (c = 0; c < 3; c++)
				{
					const routing_t* const r = p ? &prefill : &gemv;
					float* const a = activations_fine_new(datatype, r->rows, k, 0x50f7cb00u + c);
					float* const route = route_weights_new(datatype, r->rows);
					float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
					float* const g = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
					float* const u = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
					routing_t one = routing_per_row(r);
					const int status = run_segmented_swiglu_cmd(CMD_SEGMENTED_SWIGLU_SOFT_CAP_FORWARD(caps[c][0], caps[c][1]), datatype, r, a, r->rows, route, k, n_rows, gate, up, no_na, out);
					REQUIRE(status == CCV_NNC_EXEC_SUCCESS || status == CCV_NNC_EXEC_INVALID, "%s %s%s caps %g %g run or are rejected", dtype_name(datatype), p ? "prefill" : "gemv", no_na ? " without neural accelerators" : "", caps[c][0], caps[c][1]);
					runs++;
					if (status == CCV_NNC_EXEC_SUCCESS)
					{
						REQUIRE_EQ(run_segmented_gemm(datatype, &one, a, k, n_rows, gate, 0, 0, g), CCV_NNC_EXEC_SUCCESS, "gate GEMV runs");
						REQUIRE_EQ(run_segmented_gemm(datatype, &one, a, k, n_rows, up, 0, 0, u), CCV_NNC_EXEC_SUCCESS, "up GEMV runs");
						double largest = 0, largest_uncapped = 0, worst = 0, capped = 0;
						int i;
						for (i = 0; i < r->rows * n_rows; i++)
						{
							const double gl = caps[c][0] > 0 ? caps[c][0] * tanh(g[i] / caps[c][0]) : g[i];
							const double ul = caps[c][1] > 0 ? caps[c][1] * tanh(u[i] / caps[c][1]) : u[i];
							const double want = route[i / n_rows] * ul * gl / (1 + exp(-(double)g[i]));
							const double uncapped = route[i / n_rows] * (double)u[i] * g[i] / (1 + exp(-(double)g[i]));
							largest = ccv_max(largest, fabs(want));
							worst = ccv_max(worst, fabs(out[i] - want));
							capped = ccv_max(capped, fabs(want - uncapped));
							largest_uncapped = ccv_max(largest_uncapped, fabs(uncapped));
						}
						REQUIRE(capped > 0.05 * largest, "%s %s caps %g %g change the outputs", dtype_name(datatype), p ? "prefill" : "gemv", caps[c][0], caps[c][1]);
						// W8A8 quantizes the activations, whose error in the gate and up products the caps do not bound:
						// there the bound is 1% of the largest uncapped product, still far below the caps' effect. FP16 io
						// rounds the GEMVs and the output.
						const double bound = p ? 0.01 * largest_uncapped : (d ? 4e-3 : 1e-4) * largest;
						REQUIRE(bound < 0.5 * capped, "%s %s caps %g %g: the bound tells the caps apart", dtype_name(datatype), p ? "prefill" : "gemv", caps[c][0], caps[c][1]);
						REQUIRE(worst <= bound, "%s %s%s caps %g %g are applied: worst %g, bound %g", dtype_name(datatype), p ? "prefill" : "gemv", no_na ? " without neural accelerators" : "", caps[c][0], caps[c][1], worst, bound);
						ran++;
					}
					routing_free(&one);
					ccfree(u);
					ccfree(g);
					ccfree(out);
					ccfree(route);
					ccfree(a);
				}
		ccv_nnc_tensor_free(up);
		ccv_nnc_tensor_free(up_host);
		ccv_nnc_tensor_free(gate);
		ccv_nnc_tensor_free(gate_host);
	}
	printf("\n  %d of %d soft-capped FTQ SwiGLU runs took the caps, the others were rejected\n", ran, runs);
	routing_free(&prefill);
	routing_free(&gemv);
}

// Inkling Mini: down 2304 x 1152 and gate / up 3072 x 2304 over 128 experts, top-8 routing, T = 1 (GEMV) and T = 64
// and 512 (W8A8). Prefill checks the first and last row of every bin, every 37th row and 4 tile rows of outputs.
TEST_CASE("mps ftq segmented gemm and swiglu at inkling mini shapes, top-8 decode and prefill")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int na = ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context());
	static const int configs[][2] = { { 34, CCV_32F }, { 34, CCV_16F }, { 48, CCV_32F }, { 48, CCV_16F } };
	static const int tokens[] = { 1, 64, 512 };
	int c, ti;
	for (c = 0; c < 4; c++)
	{
		const int n = configs[c][0], datatype = configs[c][1];
		const uint32_t seed = 0x1a4c00u + c * 31;
		ccv_nnc_tensor_t* down_host;
		ccv_nnc_tensor_t* const down = ftq_weight_new(datatype, 128, 2304, 1152, n, &ftq_builtin, seed, &down_host);
		ccv_nnc_tensor_t* gate_host;
		ccv_nnc_tensor_t* up_host;
		ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 128, 3072, 2304, n, &ftq_builtin, seed + 1, &gate_host);
		ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 128, 3072, 2304, n, &ftq_builtin, seed + 2, &up_host);
		for (ti = 0; ti < 3; ti++)
		{
			const int t = tokens[ti];
			if (t > 1 && !na && datatype == CCV_16F)
				continue;
			routing_t r = routing_top_k(t, 8, 128, seed + 7 * t);
			// Decode bins hold one row each: drop the empty bin, so that the GEMV path (b_rows == bincount) runs.
			if (t == 1)
				r.bins--;
			double worst = 0;
			int m = gemm_case(datatype, 128, 2304, 1152, n, &ftq_builtin, &r, t > 1, 0, seed + t, down, down_host, &worst);
			REQUIRE_EQ(m, 0, "Inkling down FTQ n%d %s T=%d (worst %.3g)", n, dtype_name(datatype), t, worst);
			m = swiglu_case(datatype, t == 1 ? 1 : r.rows, &r, t > 1, 0, seed + t, gate, gate_host, up, up_host, &worst);
			REQUIRE_EQ(m, 0, "Inkling gate / up FTQ n%d %s T=%d (worst %.3g)", n, dtype_name(datatype), t, worst);
			routing_free(&r);
		}
		ccv_nnc_tensor_free(up);
		ccv_nnc_tensor_free(up_host);
		ccv_nnc_tensor_free(gate);
		ccv_nnc_tensor_free(gate_host);
		ccv_nnc_tensor_free(down);
		ccv_nnc_tensor_free(down_host);
	}
}

// The register-fused FTQ prefill (the W8A8 segmented scaled GEMM / SwiGLU decoding the tiles in registers, N % 64 and
// K % 64) against the expand path it replaces: the same command on the 8I_ROWWISE [e, n, k] weight that the
// selected-expert decode expands the whole blob to (int8 codes, then the blob's FP16 scales in the io type at the
// 128-byte aligned offset), which is the W8A8 kernel the expand path runs. Integer products accumulate exactly in int32
// and both epilogues are (acc a_scale) b_scale, so the outputs must be equal bit for bit.
static ccv_nnc_tensor_t* ftq_expanded_new(const int datatype, ccv_nnc_tensor_t* const gq, const int experts, const int n_rows, const int k)
{
	ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, experts, n_rows, k);
	params.datatype = datatype;
	params = ccv_nnc_tensor_8i_rowwise(params);
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, params, 0);
	int* const indices = (int*)ccmalloc(sizeof(int) * experts);
	int* const counts = (int*)ccmalloc(sizeof(int) * experts);
	int i;
	for (i = 0; i < experts; i++)
	{
		indices[i] = i;
		counts[i] = 1;
	}
	ccv_nnc_tensor_t* const gindices = gpu_ints(indices, experts);
	ccv_nnc_tensor_t* const gcounts = gpu_ints(counts, experts);
	ccv_nnc_mfa_context_t* const context = ccv_nnc_default_mfa_context();
	uint32_t mfa_format = (uint32_t)ccv_nnc_8i_rowwise_x_kernel_format(gq->info.reserved);
	mtl_buffer_t* codebook = 0;
	const int ok = ccv_nnc_mfa_codebook_format(context, ccv_nnc_codebook_slot_of(gq->info.reserved), 1, &mfa_format, &codebook);
	assert(ok);
	(void)ok;
	const ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_params_t decode = {
		.data_type = datatype == CCV_16F ? 16 : 3, // MTL::DataTypeHalf / MTL::DataTypeFloat
		.format = mfa_format,
		.row_length = k,
		.rows_per_expert = n_rows,
		.expert_count = experts,
		.bincount = experts,
		.codebook = codebook,
	};
	mtl_buffer_t* const scratch = ccv_nnc_mfa_request_scratch(context, ccv_nnc_mfa_dequantize_8i_rowwise_x_selected_reserved_scratch_size(decode));
	mtl_command_batch_t* const command_batch = ccv_nnc_stream_context_start_command_batch(0);
	mtl_buffer_t* tensors[6] = {
		(mtl_buffer_t*)gq->data.u8, (mtl_buffer_t*)gindices->data.u8, (mtl_buffer_t*)gcounts->data.u8, (mtl_buffer_t*)out->data.u8, scratch, 0,
	};
	size_t offsets[5] = { (size_t)gq->dataof, (size_t)gindices->dataof, (size_t)gcounts->dataof, (size_t)out->dataof, 0 };
	ccv_nnc_mfa_encode_dequantize_8i_rowwise_x_selected(context, decode, command_batch, tensors, offsets);
	ccv_nnc_stream_context_finish_command_batch(0, command_batch);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccfree(counts);
	ccfree(indices);
	return out;
}

// Whether the routed prefill of `r` on an [e, n_rows, k] weight of this format decodes the tiles in registers.
static int ftq_fuses(const ccv_nnc_tensor_t* const gq, const routing_t* const r, const int n_rows, const int k)
{
	return ccv_nnc_mfa_segmented_scaled_gemm_fuses_packed_format((uint32_t)ccv_nnc_8i_rowwise_x_kernel_format(gq->info.reserved), r->rows, r->bins, n_rows, k);
}

// Elements of two float outputs that differ in any bit.
static int bit_mismatches(const float* const x, const float* const y, const size_t count)
{
	int mismatches = 0;
	size_t i;
	for (i = 0; i < count; i++)
		if (memcmp(x + i, y + i, sizeof(float)) != 0)
		{
			if (mismatches < 4)
				printf("  bit mismatch at %zu: %.9g vs %.9g\n", i, x[i], y[i]);
			mismatches++;
		}
	return mismatches;
}

// Segmented GEMM of `r` on the FTQ weight and on its expansion: their differing outputs, or -1 if either fails.
static int fused_gemm_mismatches(const int datatype, const routing_t* const r, const int k, const int n_rows, ccv_nnc_tensor_t* const gq, ccv_nnc_tensor_t* const expanded, const uint32_t seed)
{
	float* const a = activations_new(datatype, r->rows, k, seed);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const want = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	int mismatches = -1;
	if (run_segmented_gemm(datatype, r, a, k, n_rows, gq, 0, 0, out) == CCV_NNC_EXEC_SUCCESS &&
		run_segmented_gemm(datatype, r, a, k, n_rows, expanded, 0, 0, want) == CCV_NNC_EXEC_SUCCESS)
		mismatches = bit_mismatches(out, want, (size_t)r->rows * n_rows);
	ccfree(want);
	ccfree(out);
	ccfree(a);
	return mismatches;
}

static int fused_swiglu_mismatches(const int datatype, const routing_t* const r, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const gate_expanded, ccv_nnc_tensor_t* const up_expanded, const uint32_t seed)
{
	float* const a = activations_new(datatype, r->rows, k, seed);
	float* const route = route_weights_new(datatype, r->rows);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const want = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	int mismatches = -1;
	if (run_segmented_swiglu(datatype, r, a, r->rows, route, k, n_rows, gate, up, 0, out) == CCV_NNC_EXEC_SUCCESS &&
		run_segmented_swiglu(datatype, r, a, r->rows, route, k, n_rows, gate_expanded, up_expanded, 0, want) == CCV_NNC_EXEC_SUCCESS)
		mismatches = bit_mismatches(out, want, (size_t)r->rows * n_rows);
	ccfree(want);
	ccfree(out);
	ccfree(route);
	ccfree(a);
	return mismatches;
}

// Every static rate on a small bank: 6 experts of 128 rows, K = 320 (20 tiles a row: K % 64 == 0 under 512, so only the
// K tail loop runs) and K = 1152 (two K512 blocks and a K128 tail), 30 - 41 rows per bin with an empty bin; the external
// and folded tables at n34 and n48.
TEST_CASE("mps ftq fused segmented gemm and swiglu equal the expand path bit for bit at every rate")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	static const int ks[] = { 320, 1152 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	static const int prefill_counts[] = { 30, 41, 0, 25, 33 };
	routing_t r = routing_of(5, small_w8a8_indices, prefill_counts);
	const int n_rows = 128;
	int ki, n, d, t, fused = 0;
	for (ki = 0; ki < 2; ki++)
		for (n = 16; n <= 48; n += 2)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 3; t++)
				{
					if (t > 0 && n != 34 && n != 48)
						continue;
					const int datatype = d ? CCV_32F : CCV_16F;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
					const uint32_t seed = 0xf05e00u + ki * 977 + n * 13 + d * 5 + t;
					ccv_nnc_tensor_t* gate_host;
					ccv_nnc_tensor_t* up_host;
					ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, n_rows, ks[ki], n, table, seed, &gate_host);
					ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, n_rows, ks[ki], n, table, seed + 1000, &up_host);
					ccv_nnc_tensor_t* const gate_expanded = ftq_expanded_new(datatype, gate, 6, n_rows, ks[ki]);
					ccv_nnc_tensor_t* const up_expanded = ftq_expanded_new(datatype, up, 6, n_rows, ks[ki]);
					REQUIRE(ftq_fuses(gate, &r, n_rows, ks[ki]), "FTQ n%d table %d K=%d should take the fused tile path", n, t, ks[ki]);
					fused++;
					int m = fused_gemm_mismatches(datatype, &r, ks[ki], n_rows, gate, gate_expanded, seed + 2);
					REQUIRE_EQ(m, 0, "fused FTQ segmented GEMM %s K=%d n%d table %d should equal the expand path", dtype_name(datatype), ks[ki], n, t);
					m = fused_swiglu_mismatches(datatype, &r, ks[ki], n_rows, gate, up, gate_expanded, up_expanded, seed + 3);
					REQUIRE_EQ(m, 0, "fused FTQ segmented SwiGLU %s K=%d n%d table %d should equal the expand path", dtype_name(datatype), ks[ki], n, t);
					ccv_nnc_tensor_free(up_expanded);
					ccv_nnc_tensor_free(gate_expanded);
					ccv_nnc_tensor_free(up);
					ccv_nnc_tensor_free(up_host);
					ccv_nnc_tensor_free(gate);
					ccv_nnc_tensor_free(gate_host);
				}
	if (getenv("FTQ_SEGMENTED_VERBOSE"))
		printf("\n  %d fused configurations bit-identical to the expand path\n", fused);
	routing_free(&r);
}

// The register tile rows the routed prefill of `r` takes on this FTQ weight (32, or 64 for static rates from 32 at 32
// or more rows per bin on average, empty bins included).
static int ftq_tile_rows(const ccv_nnc_tensor_t* const gq, const routing_t* const r)
{
	return ccv_nnc_mfa_segmented_scaled_gemm_register_tile_rows((uint32_t)ccv_nnc_8i_rowwise_x_kernel_format(gq->info.reserved), r->rows, r->bins);
}

// The 64-row register tiles (each SIMD group's decoded weight fragments serve two 32-row activation tiles) against the
// expand path, on 6 experts of 128 rows at K = 320 (K tail loop only) and K = 1152 (K512 blocks and a tail):
// - wide: bins of 0, 1, 31, 32, 33, 64, 65, 97, 128 and 129 rows (an empty bin, tiles of one half, a whole half, two
//   whole halves and every partial second half), 58 rows per bin, 64-row tiles;
// - at: 0, 95 and 1 rows, exactly 32 rows per bin, 64-row tiles; below: 0, 94 and 1, one row fewer, 32-row tiles;
// - skewed: one 130-row bin and five of at most one row, 22 rows per bin, 32-row tiles although one bin is long.
// Every static rate from 32 with the built-in table at K = 1152 on the wide routing; n34, n46 and n48 with the
// built-in, external and folded tables on every routing at both K; n16 and n24 stay on 32-row tiles. Both FP16 and
// FP32. Then the same weights with M loaded at run time (one pipeline per tile size): skewed, wide, skewed, at, below.
TEST_CASE("mps ftq fused segmented gemm and swiglu 64-row tiles equal the expand path bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	static const int ks[] = { 320, 1152 };
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	static const int wide_indices[] = { 5, 0, 3, 2, 4, 1, 5, 0, 3, 2 };
	static const int wide_counts[] = { 0, 1, 31, 32, 33, 64, 65, 97, 128, 129 };
	static const int three_indices[] = { 4, 1, 3 };
	static const int at_counts[] = { 0, 95, 1 };
	static const int below_counts[] = { 0, 94, 1 };
	static const int skewed_indices[] = { 2, 0, 5, 1, 4, 3 };
	static const int skewed_counts[] = { 130, 1, 1, 0, 1, 1 };
	routing_t routings[4] = {
		routing_of(10, wide_indices, wide_counts),
		routing_of(3, three_indices, at_counts),
		routing_of(3, three_indices, below_counts),
		routing_of(6, skewed_indices, skewed_counts),
	};
	static const int want_rows[4] = { 64, 64, 32, 32 };
	static const char* const routing_names[4] = { "wide", "at", "below", "skewed" };
	static const int runtime_m_order[] = { 3, 0, 3, 1, 2 };
	const int n_rows = 128;
	int ki, n, d, t, ri, i, runs64 = 0, runs32 = 0;
	for (ki = 0; ki < 2; ki++)
		for (n = 16; n <= 48; n += 2)
			for (d = 0; d < 2; d++)
				for (t = 0; t < 3; t++)
				{
					const int focus = n == 34 || n == 46 || n == 48;
					// Every rate from 32 at K = 1152 with the built-in table; n16 and n24 as the low-rate contrast.
					if (!focus && (t > 0 || (n >= 32 ? ki == 0 : (n != 16 && n != 24))))
						continue;
					const int datatype = d ? CCV_32F : CCV_16F;
					const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
					const uint32_t seed = 0x64f0e0u + ki * 977 + n * 13 + d * 5 + t;
					ccv_nnc_tensor_t* gate_host;
					ccv_nnc_tensor_t* up_host;
					ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, n_rows, ks[ki], n, table, seed, &gate_host);
					ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, n_rows, ks[ki], n, table, seed + 1000, &up_host);
					ccv_nnc_tensor_t* const gate_expanded = ftq_expanded_new(datatype, gate, 6, n_rows, ks[ki]);
					ccv_nnc_tensor_t* const up_expanded = ftq_expanded_new(datatype, up, 6, n_rows, ks[ki]);
					for (ri = 0; ri < (focus || n < 32 ? 4 : 1); ri++)
					{
						const routing_t* const r = routings + ri;
						REQUIRE(ftq_fuses(gate, r, n_rows, ks[ki]), "FTQ n%d %s routing should take the fused tile path", n, routing_names[ri]);
						const int rows = ftq_tile_rows(gate, r);
						REQUIRE_EQ(rows, n >= 32 ? want_rows[ri] : 32, "FTQ n%d %s routing (%d rows, %d bins) should take %d-row tiles", n, routing_names[ri], r->rows, r->bins, n >= 32 ? want_rows[ri] : 32);
						if (rows == 64)
							runs64++;
						else
							runs32++;
						int m = fused_gemm_mismatches(datatype, r, ks[ki], n_rows, gate, gate_expanded, seed + 2 + ri);
						REQUIRE_EQ(m, 0, "fused FTQ segmented GEMM %s K=%d n%d table %d %s routing (%d-row tiles) should equal the expand path", dtype_name(datatype), ks[ki], n, t, routing_names[ri], rows);
						m = fused_swiglu_mismatches(datatype, r, ks[ki], n_rows, gate, up, gate_expanded, up_expanded, seed + 7 + ri);
						REQUIRE_EQ(m, 0, "fused FTQ segmented SwiGLU %s K=%d n%d table %d %s routing (%d-row tiles) should equal the expand path", dtype_name(datatype), ks[ki], n, t, routing_names[ri], rows);
					}
					if (focus && ki == 1 && t != 1)
					{
						// Runtime M: the pipelines are keyed by tile size, not M, so a 64-row and a 32-row routing alternate.
						const uint64_t old_flags = ccv_nnc_flags();
						ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
						for (i = 0; i < (int)(sizeof(runtime_m_order) / sizeof(runtime_m_order[0])); i++)
						{
							const routing_t* const r = routings + runtime_m_order[i];
							int m = fused_gemm_mismatches(datatype, r, ks[ki], n_rows, gate, gate_expanded, seed + 31 + i);
							REQUIRE_EQ(m, 0, "runtime-M fused FTQ segmented GEMM %s n%d table %d %s routing should equal the expand path", dtype_name(datatype), n, t, routing_names[runtime_m_order[i]]);
							m = fused_swiglu_mismatches(datatype, r, ks[ki], n_rows, gate, up, gate_expanded, up_expanded, seed + 37 + i);
							REQUIRE_EQ(m, 0, "runtime-M fused FTQ segmented SwiGLU %s n%d table %d %s routing should equal the expand path", dtype_name(datatype), n, t, routing_names[runtime_m_order[i]]);
						}
						if (!(old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M))
							ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
					}
					ccv_nnc_tensor_free(up_expanded);
					ccv_nnc_tensor_free(gate_expanded);
					ccv_nnc_tensor_free(up);
					ccv_nnc_tensor_free(up_host);
					ccv_nnc_tensor_free(gate);
					ccv_nnc_tensor_free(gate_host);
				}
	if (getenv("FTQ_SEGMENTED_VERBOSE"))
		printf("\n  %d routings on 64-row tiles and %d on 32-row tiles bit-identical to the expand path\n", runs64, runs32);
	for (ri = 0; ri < 4; ri++)
		routing_free(routings + ri);
}

// Inkling Mini, down 2304 x 1152 (K % 256 != 0) and gate / up 3072 x 2304 over 128 experts, top-8 routing at T = 64, 512
// and 2048 (4 - 128 rows per bin, all fused), n16 / n34 / n46 / n48 with the built-in table and n34 / n48 with the
// external and folded ones, FP16 and FP32. Then the M threshold: 2 bins of 1024 rows each still fuse, 2 bins of 1025
// rows expand (both against the expansion, which shows the reference is the expand path itself).
TEST_CASE("mps ftq fused segmented gemm and swiglu equal the expand path bit for bit at inkling mini shapes")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	GUARD_ELSE_RETURN(ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context()));
	const ftq_table_t external = ftq_external_table(), folded = ftq_folded_table();
	static const int rates[] = { 16, 34, 46, 48 };
	static const int tokens[] = { 64, 512, 2048 };
	int ri, d, t, ti, fused = 0, expanded_runs = 0;
	for (ri = 0; ri < 4; ri++)
		for (d = 0; d < 2; d++)
			for (t = 0; t < 3; t++)
			{
				const int n = rates[ri];
				if (t > 0 && n != 34 && n != 48)
					continue;
				const int datatype = d ? CCV_32F : CCV_16F;
				const ftq_table_t* const table = t == 0 ? &ftq_builtin : t == 1 ? &external : &folded;
				const uint32_t seed = 0x1f0500u + ri * 97 + d * 5 + t;
				ccv_nnc_tensor_t* down_host;
				ccv_nnc_tensor_t* gate_host;
				ccv_nnc_tensor_t* up_host;
				ccv_nnc_tensor_t* const down = ftq_weight_new(datatype, 128, 2304, 1152, n, table, seed, &down_host);
				ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 128, 3072, 2304, n, table, seed + 1, &gate_host);
				ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 128, 3072, 2304, n, table, seed + 2, &up_host);
				ccv_nnc_tensor_free(down_host);
				ccv_nnc_tensor_free(gate_host);
				ccv_nnc_tensor_free(up_host);
				ccv_nnc_tensor_t* const down_expanded = ftq_expanded_new(datatype, down, 128, 2304, 1152);
				ccv_nnc_tensor_t* const gate_expanded = ftq_expanded_new(datatype, gate, 128, 3072, 2304);
				ccv_nnc_tensor_t* const up_expanded = ftq_expanded_new(datatype, up, 128, 3072, 2304);
				for (ti = 0; ti < 3; ti++)
				{
					routing_t r = routing_top_k(tokens[ti], 8, 128, seed + 7 * tokens[ti]);
					REQUIRE(ftq_fuses(down, &r, 2304, 1152) && ftq_fuses(gate, &r, 3072, 2304), "Inkling FTQ n%d T=%d should take the fused tile path", n, tokens[ti]);
					fused++;
					int m = fused_gemm_mismatches(datatype, &r, 1152, 2304, down, down_expanded, seed + 11 * tokens[ti]);
					REQUIRE_EQ(m, 0, "fused FTQ down n%d %s table %d T=%d should equal the expand path", n, dtype_name(datatype), t, tokens[ti]);
					m = fused_swiglu_mismatches(datatype, &r, 2304, 3072, gate, up, gate_expanded, up_expanded, seed + 13 * tokens[ti]);
					REQUIRE_EQ(m, 0, "fused FTQ gate / up n%d %s table %d T=%d should equal the expand path", n, dtype_name(datatype), t, tokens[ti]);
					routing_free(&r);
				}
				if (n == 34 && t == 0)
				{
					// rows per bin 1024 (fused, at the threshold) and 1025 (expanded).
					int rows;
					for (rows = 1024; rows <= 1025; rows++)
					{
						routing_t r = routing_of(2, (const int[]){ 77, 5 }, (const int[]){ rows, rows });
						const int fuses = ftq_fuses(down, &r, 2304, 1152);
						REQUIRE_EQ(fuses, rows == 1024, "FTQ with %d rows per bin should %s", rows, rows == 1024 ? "fuse" : "expand");
						if (!fuses)
							expanded_runs++;
						int m = fused_gemm_mismatches(datatype, &r, 1152, 2304, down, down_expanded, seed + rows);
						REQUIRE_EQ(m, 0, "FTQ down %s %d rows per bin should equal the expansion", dtype_name(datatype), rows);
						m = fused_swiglu_mismatches(datatype, &r, 2304, 3072, gate, up, gate_expanded, up_expanded, seed + 3 * rows);
						REQUIRE_EQ(m, 0, "FTQ gate / up %s %d rows per bin should equal the expansion", dtype_name(datatype), rows);
						routing_free(&r);
					}
				}
				ccv_nnc_tensor_free(up_expanded);
				ccv_nnc_tensor_free(gate_expanded);
				ccv_nnc_tensor_free(down_expanded);
				ccv_nnc_tensor_free(up);
				ccv_nnc_tensor_free(gate);
				ccv_nnc_tensor_free(down);
			}
	if (getenv("FTQ_SEGMENTED_VERBOSE"))
		printf("\n  %d fused routings and %d expanded routings bit-identical to the expansion\n", fused, expanded_runs);
}

TEST_CASE("mps ftq segmented gemm and swiglu reject what they do not take, without asserting")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t w8a8 = routing_of(5, small_w8a8_indices, small_w8a8_counts);
	routing_t prefill = routing_of(5, small_w8a8_indices, (const int[]){ 30, 41, 0, 25, 33 });
	const int k = 272, n_rows = 48;
	ccv_nnc_tensor_t* hq;
	ccv_nnc_tensor_t* const gq = ftq_weight_new(CCV_16F, 6, n_rows, k, 34, &ftq_builtin, 0xbad0u, &hq);
	ccv_nnc_tensor_t* hq48;
	ccv_nnc_tensor_t* const gq48 = ftq_weight_new(CCV_16F, 6, n_rows, k, 48, &ftq_builtin, 0xbad1u, &hq48);
	float* const a = activations_new(CCV_16F, prefill.rows, k, 0xbad2u);
	float* const route = route_weights_new(CCV_16F, prefill.rows);
	int i;
	// Bias.
	ccv_nnc_tensor_t* const bias = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, 6, n_rows), 0);
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, bias, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ GEMV takes no bias");
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &w8a8, a, k, n_rows, gq, bias, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ W8A8 GEMM takes no bias");
	ccv_nnc_tensor_free(bias);
	// A Hadamard rotation, the variable-rate block and an unregistered codebook slot on the same blob.
	const ccv_nnc_tensor_param_t info = gq->info;
	gq->info.reserved = info.reserved | CCV_NNC_QX_8I_ROWWISE_HADAMARD_256;
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ with H256 GEMV");
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &w8a8, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ with H256 W8A8");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &gemv, a, gemv.rows, route, k, n_rows, gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ with H256 SwiGLU");
	gq->info.reserved = CCV_NNC_QX_8I_ROWWISE_FTQ_VR_N34;
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ-VR GEMV");
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &w8a8, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ-VR W8A8");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, k, n_rows, gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ-VR SwiGLU");
	gq->info.reserved = ccv_nnc_codebook_with_slot(info.reserved, 250);
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "unregistered codebook GEMV");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &gemv, a, gemv.rows, route, k, n_rows, gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "unregistered codebook SwiGLU");
	gq->info = info;
	// Off the tile grid: the allocated grid, shown with N = 40 or K = 264 (both still % 8 and % 4).
	static const int off[2][2] = { { 40, 272 }, { 48, 264 } };
	for (i = 0; i < 2; i++)
	{
		gq->info.dim[1] = off[i][0];
		gq->info.dim[2] = off[i][1];
		REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, off[i][1], off[i][0], gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "off-grid FTQ GEMV N=%d K=%d", off[i][0], off[i][1]);
		REQUIRE_EQ(run_segmented_gemm(CCV_16F, &w8a8, a, off[i][1], off[i][0], gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "off-grid FTQ W8A8 N=%d K=%d", off[i][0], off[i][1]);
		REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &gemv, a, gemv.rows, route, off[i][1], off[i][0], gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "off-grid FTQ SwiGLU GEMVs N=%d K=%d", off[i][0], off[i][1]);
		REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, off[i][1], off[i][0], gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "off-grid FTQ W8A8 SwiGLU N=%d K=%d", off[i][0], off[i][1]);
		gq->info = info;
	}
	// The blob read at an offset off the word grid.
	const off_t dataof = gq->dataof;
	gq->dataof = dataof + 2;
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ GEMV at a 2-byte offset");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, k, n_rows, gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "FTQ SwiGLU at a 2-byte offset");
	gq->dataof = dataof;
	// Gate and up at different rates; an FTQ gate with an 8I_ROWWISE_X up of another codec.
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &gemv, a, gemv.rows, route, k, n_rows, gq, gq48, 0, 0), CCV_NNC_EXEC_INVALID, "gate n34 / up n48 SwiGLU GEMVs");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, k, n_rows, gq, gq48, 0, 0), CCV_NNC_EXEC_INVALID, "gate n34 / up n48 W8A8 SwiGLU");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, k, n_rows, gq48, gq, 0, 0), CCV_NNC_EXEC_INVALID, "gate n48 / up n34 W8A8 SwiGLU");
	// FP32 activations over an FP16-base blob.
	REQUIRE_EQ(run_segmented_gemm(CCV_32F, &gemv, a, k, n_rows, gq, 0, 0, 0), CCV_NNC_EXEC_INVALID, "FP32 activations, FP16 FTQ GEMV");
	REQUIRE_EQ(run_segmented_swiglu(CCV_32F, &prefill, a, prefill.rows, route, k, n_rows, gq, gq, 0, 0), CCV_NNC_EXEC_INVALID, "FP32 activations, FP16 FTQ SwiGLU");
	// FP16 without neural accelerators has no W8A8 route (the fp decode fallback has no FTQ); the GEMVs still run.
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &w8a8, a, k, n_rows, gq, 0, 1, 0), CCV_NNC_EXEC_INVALID, "FP16 FTQ W8A8 GEMM without neural accelerators");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &prefill, a, prefill.rows, route, k, n_rows, gq, gq, 1, 0), CCV_NNC_EXEC_INVALID, "FP16 FTQ W8A8 SwiGLU without neural accelerators");
	REQUIRE_EQ(run_segmented_gemm(CCV_16F, &gemv, a, k, n_rows, gq, 0, 1, 0), CCV_NNC_EXEC_SUCCESS, "FP16 FTQ GEMV without neural accelerators");
	REQUIRE_EQ(run_segmented_swiglu(CCV_16F, &gemv, a, gemv.rows, route, k, n_rows, gq, gq, 1, 0), CCV_NNC_EXEC_SUCCESS, "FP16 FTQ SwiGLU GEMVs without neural accelerators");
	ccfree(route);
	ccfree(a);
	ccv_nnc_tensor_free(gq48);
	ccv_nnc_tensor_free(hq48);
	ccv_nnc_tensor_free(gq);
	ccv_nnc_tensor_free(hq);
	routing_free(&prefill);
	routing_free(&w8a8);
	routing_free(&gemv);
}

// An FP32-base FTQ weight can now be made; the dense commands, which take FP16 FTQ only (or no FTQ), reject it.
TEST_CASE("mps dense gemm, swiglu and index select reject an fp32-base ftq weight without asserting")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int n_rows = 64, k = 272;
	const int format = ftq_format(34);
	ccv_nnc_tensor_param_t wp = ftq_param(CCV_TENSOR_GPU_MEMORY | 000, CCV_32F, 1, n_rows, k, format);
	wp.dim[0] = n_rows;
	wp.dim[1] = k;
	wp.dim[2] = 0;
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, wp, 0);
	static const int ms[] = { 1, 4 };
	int i;
	for (i = 0; i < 2; i++)
	{
		ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, ms[i], k), 0);
		ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, ms[i], n_rows), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w), TENSOR_LIST(b), 0), CCV_NNC_EXEC_INVALID, "dense GEMM M=%d over an FP32 FTQ weight", ms[i]);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_SWIGLU_FORWARD(10), ccv_nnc_no_hint, 0, TENSOR_LIST(a, w, w), TENSOR_LIST(b), 0), CCV_NNC_EXEC_INVALID, "SwiGLU M=%d over FP32 FTQ weights", ms[i]);
		ccv_nnc_tensor_free(b);
		ccv_nnc_tensor_free(a);
	}
	if (ccv_nnc_cmd_ok(CCV_NNC_INDEX_SELECT_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		static const int index[1] = { 3 };
		ccv_nnc_tensor_t* const indices = gpu_ints(index, 1);
		ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, k), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_INDEX_SELECT_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(w, indices), TENSOR_LIST(out), 0), CCV_NNC_EXEC_INVALID, "IndexSelect over an FP32 FTQ table");
		ccv_nnc_tensor_free(out);
		ccv_nnc_tensor_free(indices);
	}
	ccv_nnc_tensor_free(w);
}

// Routed FTQ experts with a shared expert (8I_ROWWISE rows, one activation row) in the same command: the routed output
// equals the command without the shared expert, and the shared output equals GEMM_FORWARD / SWIGLU_FORWARD on it.
static ccv_nnc_tensor_t* shared_q8_new(const int datatype, const int rows, const int cols, const uint32_t seed)
{
	float* const values = activations_new(CCV_32F, rows, cols, seed);
	ccv_nnc_tensor_param_t params = CPU_TENSOR_NHWC(32F, rows, cols);
	params.datatype = datatype;
	params = ccv_nnc_tensor_8i_rowwise(params);
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_tensor_t* const dense = gpu_from_floats(datatype, values, rows, cols);
	ccv_nnc_tensor_param_t dense_host = dense->info;
	dense_host.type = CCV_TENSOR_CPU_MEMORY;
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, dense_host, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(dense), TENSOR_LIST(h), 0);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(params);
	const size_t written = ccv_nnc_quantize_8i_rowwise(h->data.u8, datatype, CCV_TENSOR_CPU_MEMORY, (size_t)rows * cols, cols, 0, 0, host->data.u8, size);
	assert(written == size);
	(void)written;
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const gpu = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(gpu), 0);
	ccv_nnc_tensor_free(h);
	ccv_nnc_tensor_free(dense);
	ccv_nnc_tensor_free(host);
	ccfree(values);
	return gpu;
}

static int same_floats(ccv_nnc_tensor_t* const x, ccv_nnc_tensor_t* const y)
{
	const size_t count = ccv_nnc_tensor_count(x->info);
	float* const a = (float*)ccmalloc(sizeof(float) * count);
	float* const b = (float*)ccmalloc(sizeof(float) * count);
	floats_from_gpu(x, a);
	floats_from_gpu(y, b);
	const int same = count == ccv_nnc_tensor_count(y->info) && memcmp(a, b, sizeof(float) * count) == 0;
	ccfree(b);
	ccfree(a);
	return same;
}

TEST_CASE("mps ftq routed experts run with a shared expert in the same segmented command")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS));
	const int na = ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context());
	const int k = 512, n_rows = 256, shared_rows = 256;
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t w8a8 = routing_of(5, small_w8a8_indices, (const int[]){ 30, 41, 0, 25, 33 });
	int d, p;
	for (d = 0; d < 2; d++)
	{
		const int datatype = d ? CCV_32F : CCV_16F;
		ccv_nnc_tensor_t* gate_host;
		ccv_nnc_tensor_t* up_host;
		ccv_nnc_tensor_t* down_host;
		ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, n_rows, k, 48, &ftq_builtin, 0x54a0u + d, &gate_host);
		ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, n_rows, k, 48, &ftq_builtin, 0x54b0u + d, &up_host);
		ccv_nnc_tensor_t* const down = ftq_weight_new(datatype, 6, n_rows, k, 48, &ftq_builtin, 0x54c0u + d, &down_host);
		ccv_nnc_tensor_t* const shared_gate = shared_q8_new(datatype, shared_rows, k, 0x54d0u + d);
		ccv_nnc_tensor_t* const shared_up = shared_q8_new(datatype, shared_rows, k, 0x54e0u + d);
		float* const shared_values = activations_new(datatype, 1, k, 0x54f0u + d);
		ccv_nnc_tensor_t* const shared_a = gpu_from_floats(datatype, shared_values, 1, k);
		for (p = 0; p < 2; p++)
		{
			const routing_t* const r = p ? &w8a8 : &gemv;
			if (p && !na && datatype == CCV_16F)
				continue;
			float* const a = activations_new(datatype, r->rows, k, 0x5500u + p);
			float* const route = route_weights_new(datatype, r->rows);
			ccv_nnc_tensor_t* const ga = gpu_from_floats(datatype, a, r->rows, k);
			ccv_nnc_tensor_t* const groute = gpu_from_floats(datatype, route, r->rows, 1);
			ccv_nnc_tensor_t* const indices = gpu_ints(r->indices, r->bins);
			ccv_nnc_tensor_t* const counts = gpu_ints(r->counts, r->bins);
			ccv_nnc_tensor_param_t op = GPU_TENSOR_NHWC(000, 32F, r->rows, n_rows);
			op.datatype = datatype;
			ccv_nnc_tensor_param_t sp = GPU_TENSOR_NHWC(000, 32F, 1, shared_rows);
			sp.datatype = datatype;
			ccv_nnc_tensor_t* const routed = ccv_nnc_tensor_new(0, op, 0);
			ccv_nnc_tensor_t* const routed_alone = ccv_nnc_tensor_new(0, op, 0);
			ccv_nnc_tensor_t* const shared = ccv_nnc_tensor_new(0, sp, 0);
			ccv_nnc_tensor_t* const shared_alone = ccv_nnc_tensor_new(0, sp, 0);
			ccv_nnc_cmd_t swiglu = CMD_SEGMENTED_SWIGLU_FORWARD(swiglu_clamp);
			swiglu.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, gate, up, groute, shared_a, shared_gate, shared_up), TENSOR_LIST(routed, shared), 0), CCV_NNC_EXEC_SUCCESS, "FTQ SwiGLU %s rows %d with a shared expert", dtype_name(datatype), r->rows);
			REQUIRE_EQ(ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, gate, up, groute), TENSOR_LIST(routed_alone), 0), CCV_NNC_EXEC_SUCCESS, "FTQ SwiGLU %s rows %d", dtype_name(datatype), r->rows);
			ccv_nnc_cmd_t shared_swiglu = CMD_SWIGLU_FORWARD(swiglu_clamp);
			shared_swiglu.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(shared_swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(shared_a, shared_gate, shared_up), TENSOR_LIST(shared_alone), 0), CCV_NNC_EXEC_SUCCESS, "shared SwiGLU");
			REQUIRE(same_floats(routed, routed_alone), "FTQ SwiGLU %s rows %d: the shared expert leaves the routed output as it is", dtype_name(datatype), r->rows);
			REQUIRE(same_floats(shared, shared_alone), "FTQ SwiGLU %s rows %d: the shared output is SWIGLU_FORWARD's", dtype_name(datatype), r->rows);
			// The down projection with a shared expert's rows (input 4 is the absent bias).
			ccv_nnc_tensor_param_t dp = GPU_TENSOR_NHWC(000, 32F, r->rows, n_rows);
			dp.datatype = datatype;
			ccv_nnc_tensor_t* const down_a = gpu_from_floats(datatype, a, r->rows, k);
			ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, dp, 0);
			ccv_nnc_tensor_t* const b_alone = ccv_nnc_tensor_new(0, dp, 0);
			ccv_nnc_cmd_t gemm = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
			gemm.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(down_a, indices, counts, down, 0, shared_a, shared_gate), TENSOR_LIST(b, shared), 0), CCV_NNC_EXEC_SUCCESS, "FTQ GEMM %s rows %d with a shared expert", dtype_name(datatype), r->rows);
			REQUIRE_EQ(ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(down_a, indices, counts, down), TENSOR_LIST(b_alone), 0), CCV_NNC_EXEC_SUCCESS, "FTQ GEMM %s rows %d", dtype_name(datatype), r->rows);
			ccv_nnc_cmd_t shared_gemm = CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1));
			shared_gemm.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(shared_gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(shared_a, shared_gate), TENSOR_LIST(shared_alone), 0), CCV_NNC_EXEC_SUCCESS, "shared GEMM");
			REQUIRE(same_floats(b, b_alone), "FTQ GEMM %s rows %d: the shared expert leaves the routed output as it is", dtype_name(datatype), r->rows);
			REQUIRE(same_floats(shared, shared_alone), "FTQ GEMM %s rows %d: the shared output is GEMM_FORWARD's", dtype_name(datatype), r->rows);
			ccv_nnc_tensor_free(b_alone);
			ccv_nnc_tensor_free(b);
			ccv_nnc_tensor_free(down_a);
			ccv_nnc_tensor_free(shared_alone);
			ccv_nnc_tensor_free(shared);
			ccv_nnc_tensor_free(routed_alone);
			ccv_nnc_tensor_free(routed);
			ccv_nnc_tensor_free(counts);
			ccv_nnc_tensor_free(indices);
			ccv_nnc_tensor_free(groute);
			ccv_nnc_tensor_free(ga);
			ccfree(route);
			ccfree(a);
		}
		ccv_nnc_tensor_free(shared_a);
		ccfree(shared_values);
		ccv_nnc_tensor_free(shared_up);
		ccv_nnc_tensor_free(shared_gate);
		ccv_nnc_tensor_free(down);
		ccv_nnc_tensor_free(down_host);
		ccv_nnc_tensor_free(up);
		ccv_nnc_tensor_free(up_host);
		ccv_nnc_tensor_free(gate);
		ccv_nnc_tensor_free(gate_host);
	}
	routing_free(&w8a8);
	routing_free(&gemv);
}

// MoE weights streaming of FTQ gate / up / down banks: the streamed SwiGLU + down GEMM equals the same commands on the
// resident source weights, through resident decode (misses, a hit, an eviction), a short multi-row call and prefill
// (staged and full).
static int write_file(const ccv_nnc_tensor_t* const tensor, char path[64])
{
	const int fd = mkstemp(path);
	if (fd < 0)
		return 0;
	const size_t data_size = ccv_nnc_tensor_data_size_without_padding(tensor->info);
	size_t written = 0;
	while (written < data_size)
	{
		const ssize_t result = write(fd, tensor->data.u8 + written, data_size - written);
		if (result <= 0)
		{
			close(fd);
			return 0;
		}
		written += (size_t)result;
	}
	ccv_nnc_tensor_param_t params = tensor->info;
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	const int success = ftruncate(fd, (off_t)ccv_nnc_tensor_data_size(params)) == 0;
	close(fd);
	return success;
}

static int streaming_case(const int datatype, ccv_nnc_tensor_t* const* const sources, ccv_nnc_tensor_t* const* const references, const int* const indices, const int* const counts, const int bins, const int resident_slots, const uint32_t seed, double* const max_difference)
{
	routing_t r = routing_of(bins, indices, counts);
	const int n_rows = references[0]->info.dim[1], k = references[0]->info.dim[2];
	float* const a = activations_new(datatype, r.rows, k, seed);
	float* const route = route_weights_new(datatype, r.rows);
	ccv_nnc_tensor_t* const ga = gpu_from_floats(datatype, a, r.rows, k);
	ccv_nnc_tensor_t* const groute = gpu_from_floats(datatype, route, r.rows, 1);
	ccv_nnc_tensor_t* const gindices = gpu_ints(indices, bins);
	ccv_nnc_tensor_t* const gcounts = gpu_ints(counts, bins);
	ccv_nnc_tensor_param_t hp = GPU_TENSOR_NHWC(000, 32F, r.rows, n_rows);
	hp.datatype = datatype;
	ccv_nnc_tensor_param_t op = GPU_TENSOR_NHWC(000, 32F, r.rows, k);
	op.datatype = datatype;
	ccv_nnc_tensor_t* const ref_hidden = ccv_nnc_tensor_new(0, hp, 0);
	ccv_nnc_tensor_t* const ref_out = ccv_nnc_tensor_new(0, op, 0);
	ccv_nnc_tensor_t* const hidden = ccv_nnc_tensor_new(0, hp, 0);
	ccv_nnc_tensor_t* const out = ccv_nnc_tensor_new(0, op, 0);
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	ccv_nnc_cmd_t swiglu = CMD_SEGMENTED_SWIGLU_FORWARD(swiglu_clamp);
	swiglu.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_cmd_t gemm = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	gemm.backend = CCV_NNC_BACKEND_MPS;
	int status = ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, gindices, gcounts, references[0], references[1], groute), TENSOR_LIST(ref_hidden), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(ref_hidden, gindices, gcounts, references[2]), TENSOR_LIST(ref_out), stream);
	ccv_nnc_cmd_t streaming = CMD_MOE_WEIGHTS_STREAMING_FORWARD(resident_slots, 1);
	streaming.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_param_t streamed_params[6] = {};
	ccv_nnc_hint_tensor_auto(streaming, TENSOR_PARAM_LIST(gindices->info, gcounts->info, groute->info, sources[0]->info, sources[1]->info, sources[2]->info), ccv_nnc_no_hint, streamed_params, 6);
	ccv_nnc_tensor_t* streamed[6];
	int i;
	for (i = 0; i < 6; i++)
	{
		if (i >= 3)
			streamed_params[i].reserved = 0;
		streamed[i] = ccv_nnc_tensor_new(0, streamed_params[i], 0);
	}
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(streaming, ccv_nnc_no_hint, 0, TENSOR_LIST(gindices, gcounts, groute, sources[0], sources[1], sources[2]),
			TENSOR_LIST(streamed[0], streamed[1], streamed[2], streamed[3], streamed[4], streamed[5]), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(swiglu, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, streamed[0], streamed[1], streamed[3], streamed[4], streamed[2]), TENSOR_LIST(hidden), stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(hidden, streamed[0], streamed[1], streamed[5]), TENSOR_LIST(out), stream);
	ccv_nnc_stream_context_wait(stream);
	if (status == CCV_NNC_EXEC_SUCCESS)
	{
		float* const want = (float*)ccmalloc(sizeof(float) * r.rows * k);
		float* const got = (float*)ccmalloc(sizeof(float) * r.rows * k);
		floats_from_gpu(ref_out, want);
		floats_from_gpu(out, got);
		*max_difference = 0;
		size_t j, nonzero = 0;
		for (j = 0; j < (size_t)r.rows * k; j++)
		{
			*max_difference = isfinite(want[j]) && isfinite(got[j]) ? ccv_max(*max_difference, fabs((double)want[j] - got[j])) : INFINITY;
			nonzero += want[j] != 0;
		}
		// The comparison means something only if the reference is not all zeros.
		if (nonzero < (size_t)r.rows * k / 2)
			status = CCV_NNC_EXEC_INVALID;
		ccfree(got);
		ccfree(want);
	}
	for (i = 0; i < 6; i++)
		ccv_nnc_tensor_free(streamed[i]);
	ccv_nnc_stream_context_free(stream);
	ccv_nnc_tensor_free(out);
	ccv_nnc_tensor_free(hidden);
	ccv_nnc_tensor_free(ref_out);
	ccv_nnc_tensor_free(ref_hidden);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(groute);
	ccv_nnc_tensor_free(ga);
	ccfree(route);
	ccfree(a);
	routing_free(&r);
	return status;
}

// The inline form (ccv_nnc_codebook.h) of a slot-form FTQ weight on the CPU: the serialized table of its slot (1 or
// more), zeros up to 128 + 2 P bytes, then the same bytes.
static ccv_nnc_tensor_t* ftq_inline_host_new(const ccv_nnc_tensor_t* const host)
{
	const int slot = ccv_nnc_codebook_slot_of(host->info.reserved);
	int entries = 0;
	ccv_nnc_codebook_get(slot, &entries, 0);
	ccv_nnc_tensor_param_t params = host->info;
	params.reserved = ccv_nnc_8i_rowwise_x_with_inline_codebook(host->info.reserved, entries);
	ccv_nnc_tensor_t* const t = ccv_nnc_tensor_new(0, params, 0);
	const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(params.reserved);
	const size_t size = ccv_nnc_tensor_data_size_without_padding(host->info);
	assert(ccv_nnc_tensor_data_size_without_padding(params) == prefix + size);
	memset(t->data.u8, 0, prefix);
	const size_t serialized = ccv_nnc_codebook_serialize(slot, t->data.u8, prefix);
	assert(serialized == 16 + 2 * (size_t)entries);
	(void)serialized;
	memcpy(t->data.u8 + prefix, host->data.u8, size);
	return t;
}

static ccv_nnc_tensor_t* gpu_copy_of(const ccv_nnc_tensor_t* const host)
{
	ccv_nnc_tensor_param_t params = host->info;
	params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST((ccv_nnc_tensor_t*)host), TENSOR_LIST(g), 0);
	return g;
}

// rates: of gate, up and down. inline_codebook: the streamed (file-backed) sources carry their table inline instead of
// naming its slot; the references still name it.
static void streaming_ftq_case(const int datatype, const ftq_table_t* const table, const int* const rates, const int inline_codebook, int* const __case_result__)
{
	const int experts = 32, n_rows = 256, k = 512;
	char paths[3][64] = {
		"/tmp/ccv-moe-ftq-gate-XXXXXX",
		"/tmp/ccv-moe-ftq-up-XXXXXX",
		"/tmp/ccv-moe-ftq-down-XXXXXX",
	};
	ccv_nnc_tensor_t* hosts[3];
	ccv_nnc_tensor_t* references[3];
	ccv_nnc_tensor_t* sources[3];
	int i;
	for (i = 0; i < 3; i++)
	{
		references[i] = ftq_weight_new(datatype, experts, i < 2 ? n_rows : k, i < 2 ? k : n_rows, rates[i], table, 0x57e4u + i * 101 + datatype, &hosts[i]);
		// The bank layout of the resident slots is the [slots, N, K] blob.
		ccv_nnc_tensor_param_t bank = hosts[i]->info;
		bank.dim[0] = 2;
		const ccv_nnc_8i_rowwise_x_ftq_schedule_t schedule = ccv_nnc_8i_rowwise_x_ftq_schedule(bank.reserved, 2 * bank.dim[1], bank.dim[2]);
		REQUIRE_EQ(ccv_nnc_tensor_data_size_without_padding(bank), schedule.scale_offset + 2 * 2 * (size_t)bank.dim[1], "a 2-slot FTQ bank is a [2, N, K] blob");
		ccv_nnc_tensor_t* const inline_host = inline_codebook ? ftq_inline_host_new(hosts[i]) : 0;
		const ccv_nnc_tensor_t* const streamed_host = inline_codebook ? inline_host : hosts[i];
		REQUIRE(write_file(streamed_host, paths[i]), "each FTQ projection is written to a temporary file");
		ccv_nnc_tensor_param_t gpu_params = streamed_host->info;
		gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
		sources[i] = ccv_nnc_tensor_new_from_file(gpu_params, paths[i], 0, CCV_NNC_TENSOR_MEMORY_MAP_ON_DEMAND);
		if (inline_host)
			ccv_nnc_tensor_free(inline_host);
		REQUIRE(sources[i], "the FTQ projection maps on demand");
	}
	double max_difference = 0;
	// Decode across cold misses, a resident hit and an eviction (2 resident slots), experts 0 and 31 among them.
	static const int decode_indices[] = { 3, 17, 31, 0 };
	static const int decode_experts[] = { 0, 1, 0, 2, 3, 2 };
	for (i = 0; i < 6; i++)
	{
		int decode_counts[] = { 0, 0, 0, 0 };
		decode_counts[decode_experts[i]] = 1;
		REQUIRE_EQ(streaming_case(datatype, sources, references, decode_indices, decode_counts, 4, 2, 31 + i, &max_difference), CCV_NNC_EXEC_SUCCESS, "FTQ streamed decode %d runs", i);
		REQUIRE_EQ(max_difference, 0, "FTQ streamed decode %d equals the resident weights", i);
	}
	static const int verify_counts[] = { 1, 1, 2, 2 };
	REQUIRE_EQ(streaming_case(datatype, sources, references, decode_indices, verify_counts, 4, 8, 41, &max_difference), CCV_NNC_EXEC_SUCCESS, "FTQ short multi-row call runs");
	REQUIRE_EQ(max_difference, 0, "FTQ short multi-row call through the resident pool equals the resident weights");
	static const int prefill_lengths[] = { 64, 512, 4096 };
	for (i = 0; i < 3; i++)
	{
		const int m = prefill_lengths[i];
		const int prefill_counts[] = { m / 2, m - m / 2 - m / 8, m / 8, 0 };
		REQUIRE_EQ(streaming_case(datatype, sources, references, decode_indices, prefill_counts, 4, 2, 47 + i, &max_difference), CCV_NNC_EXEC_SUCCESS, "FTQ prefill m=%d runs", m);
		REQUIRE_EQ(max_difference, 0, "FTQ streamed prefill m=%d equals the resident weights", m);
	}
	for (i = 0; i < 3; i++)
	{
		ccv_nnc_tensor_free(sources[i]);
		ccv_nnc_tensor_free(references[i]);
		ccv_nnc_tensor_free(hosts[i]);
		unlink(paths[i]);
	}
}

TEST_CASE("mps moe weights streaming stages ftq banks, fp16 and fp32, built-in and folded tables")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, CCV_NNC_BACKEND_MPS));
	const ftq_table_t folded = ftq_folded_table();
	static const int rates[3] = { 34, 34, 48 };
	streaming_ftq_case(CCV_16F, &ftq_builtin, rates, 0, __case_result__);
	streaming_ftq_case(CCV_32F, &ftq_builtin, rates, 0, __case_result__);
	streaming_ftq_case(CCV_32F, &folded, rates, 0, __case_result__);
}

// The built-in table's content registered as a table of its own (header id 1, as the delivered files name it).
static ftq_table_t ftq_builtin_content_table(void)
{
	static ftq_table_t t;
	if (t.slot)
		return t;
	int entries = 0;
	const int8_t* const pairs = ccv_nnc_codebook_get(0, &entries, 0);
	t.slot = ccv_nnc_codebook_register(pairs, entries);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | (9u << 8);
	return t;
}

// A random 512-pair table (header id 1, log2 9).
static ftq_table_t ftq_external9_table(void)
{
	static ftq_table_t t;
	if (t.slot)
		return t;
	int8_t pairs[1024];
	uint32_t seed = 0x7e57ab09u;
	int i;
	for (i = 0; i < 1024; i++)
		pairs[i] = (int8_t)((int)(rng_next(&seed) % 255) - 127);
	t.slot = ccv_nnc_codebook_register(pairs, 512);
	ccv_nnc_codebook_get(t.slot, 0, &t.crc);
	t.table = 1u | (9u << 8);
	return t;
}

TEST_CASE("mps moe weights streaming stages ftq banks from sources with an inline codebook like the same weights naming the table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, CCV_NNC_BACKEND_MPS));
	// The table and the FTQ headers are read from the file in front of the experts; the streamed outputs equal the
	// resident references that name the table's slot, bit for bit (max difference 0), through decode and prefill.
	const ftq_table_t builtin_content = ftq_builtin_content_table(), external9 = ftq_external9_table(), external = ftq_external_table(), folded = ftq_folded_table();
	REQUIRE(builtin_content.slot >= 1 && external9.slot >= 1 && external.slot >= 1 && folded.slot >= 1, "the tables register");
	static const int rates[3] = { 34, 34, 48 };
	streaming_ftq_case(CCV_16F, &builtin_content, rates, 1, __case_result__);
	streaming_ftq_case(CCV_32F, &external9, rates, 1, __case_result__);
	streaming_ftq_case(CCV_16F, &external, rates, 1, __case_result__);
	streaming_ftq_case(CCV_32F, &folded, rates, 1, __case_result__);
}

// Segmented GEMM and SwiGLU of `r` on the slot form and on the inline form of the same weights: the outputs that differ
// in any bit, or -1 if the two do not both run (*ran: whether they did; both failing alike is 0 with *ran 0).
static int inline_segmented_mismatches(const int datatype, const routing_t* const r, const int k, const int n_rows, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const gate_inline, ccv_nnc_tensor_t* const up_inline, const int swiglu, const int no_na, const uint32_t seed, int* const ran)
{
	float* const a = activations_new(datatype, r->rows, k, seed);
	float* const route = route_weights_new(datatype, r->rows);
	float* const out = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	float* const want = (float*)ccmalloc(sizeof(float) * r->rows * n_rows);
	const int s1 = swiglu ? run_segmented_swiglu(datatype, r, a, r->rows, route, k, n_rows, gate, up, no_na, want) : run_segmented_gemm(datatype, r, a, k, n_rows, gate, 0, no_na, want);
	const int s2 = swiglu ? run_segmented_swiglu(datatype, r, a, r->rows, route, k, n_rows, gate_inline, up_inline, no_na, out) : run_segmented_gemm(datatype, r, a, k, n_rows, gate_inline, 0, no_na, out);
	*ran = s1 == CCV_NNC_EXEC_SUCCESS;
	const int mismatches = s1 != s2 ? -1 : *ran ? bit_mismatches(out, want, (size_t)r->rows * n_rows) : 0;
	ccfree(want);
	ccfree(out);
	ccfree(route);
	ccfree(a);
	return mismatches;
}

TEST_CASE("mps ftq segmented gemm and swiglu with an inline codebook equal the same weights naming the table bit for bit, and reject a header that does not name it")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	const int na = ccv_nnc_mfa_has_neural_accelerators(ccv_nnc_default_mfa_context());
	const ftq_table_t tables[5] = { ftq_builtin_content_table(), ftq_external9_table(), ftq_external_table(), ftq_external11_table(), ftq_folded_table() };
	// One row a bin (the fused FTQ GEMV) and a prefill (W8A8: tiles decoded in registers with neural accelerators, the
	// selected-expert decode and Int8MatMul without).
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	routing_t prefill = routing_of(5, small_w8a8_indices, (const int[]){ 30, 41, 0, 25, 33 });
	static const int ks[] = { 320, 1152 };
	static const int rates[] = { 34, 48 };
	const int n_rows = 128;
	int t, ki, ri, d, compared = 0;
	for (t = 0; t < 5; t++)
		for (ki = 0; ki < 2; ki++)
			for (ri = 0; ri < 2; ri++)
				for (d = 0; d < 2; d++)
				{
					if (ki == 1 && t != 1 && t != 4)
						continue;
					const int datatype = d ? CCV_32F : CCV_16F;
					const int k = ks[ki], n = rates[ri];
					const uint32_t seed = 0x1a1500u + t * 97 + ki * 31 + ri * 7 + d;
					ccv_nnc_tensor_t* gate_host;
					ccv_nnc_tensor_t* up_host;
					ccv_nnc_tensor_t* const gate = ftq_weight_new(datatype, 6, n_rows, k, n, &tables[t], seed, &gate_host);
					ccv_nnc_tensor_t* const up = ftq_weight_new(datatype, 6, n_rows, k, n, &tables[t], seed + 1000, &up_host);
					ccv_nnc_tensor_t* const gate_inline_host = ftq_inline_host_new(gate_host);
					ccv_nnc_tensor_t* const up_inline_host = ftq_inline_host_new(up_host);
					ccv_nnc_tensor_t* const gate_inline = gpu_copy_of(gate_inline_host);
					ccv_nnc_tensor_t* const up_inline = gpu_copy_of(up_inline_host);
					int p, swiglu, no_na;
					for (p = 0; p < 2; p++)
						for (swiglu = 0; swiglu < 2; swiglu++)
							for (no_na = 0; no_na < 2; no_na++)
							{
								if (no_na && (!p || !na))
									continue;
								int ran = 0;
								const int m = inline_segmented_mismatches(datatype, p ? &prefill : &gemv, k, n_rows, gate, up, gate_inline, up_inline, swiglu, no_na, seed + 2 + p * 4 + swiglu * 2 + no_na, &ran);
								REQUIRE_EQ(m, 0, "FTQ inline table %d %s K=%d n%d %s %s%s equals the slot form", t, dtype_name(datatype), k, n, p ? "prefill" : "gemv", swiglu ? "swiglu" : "gemm", no_na ? " without neural accelerators" : "");
								// The GEMV and the prefill with neural accelerators (or FP32) always run.
								if (!no_na && (!p || na || datatype == CCV_32F))
									REQUIRE(ran, "FTQ table %d %s K=%d n%d %s %s runs", t, dtype_name(datatype), k, n, p ? "prefill" : "gemv", swiglu ? "swiglu" : "gemm");
								compared += ran;
							}
					if (t == 1 && ki == 0 && ri == 0)
					{
						// The header crc of another table: the inline weight is rejected by both commands, routed gate or up.
						uint32_t crc;
						const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(gate_inline_host->info.reserved);
						memcpy(&crc, gate_inline_host->data.u8 + prefix + 28, 4);
						crc ^= 1;
						memcpy(gate_inline_host->data.u8 + prefix + 28, &crc, 4);
						ccv_nnc_tensor_t* const bad = gpu_copy_of(gate_inline_host);
						float* const a = activations_new(datatype, gemv.rows, k, seed);
						float* const route = route_weights_new(datatype, gemv.rows);
						REQUIRE_EQ(run_segmented_gemm(datatype, &gemv, a, k, n_rows, bad, 0, 0, 0), CCV_NNC_EXEC_INVALID, "a bad inline header fails the segmented GEMM %s", dtype_name(datatype));
						REQUIRE_EQ(run_segmented_swiglu(datatype, &gemv, a, gemv.rows, route, k, n_rows, bad, up_inline, 0, 0), CCV_NNC_EXEC_INVALID, "a bad inline gate header fails the segmented SwiGLU %s", dtype_name(datatype));
						REQUIRE_EQ(run_segmented_swiglu(datatype, &gemv, a, gemv.rows, route, k, n_rows, gate_inline, bad, 0, 0), CCV_NNC_EXEC_INVALID, "a bad inline up header fails the segmented SwiGLU %s", dtype_name(datatype));
						ccfree(route);
						ccfree(a);
						ccv_nnc_tensor_free(bad);
					}
					ccv_nnc_tensor_free(up_inline);
					ccv_nnc_tensor_free(gate_inline);
					ccv_nnc_tensor_free(up_inline_host);
					ccv_nnc_tensor_free(gate_inline_host);
					ccv_nnc_tensor_free(up);
					ccv_nnc_tensor_free(up_host);
					ccv_nnc_tensor_free(gate);
					ccv_nnc_tensor_free(gate_host);
				}
	REQUIRE(compared > 0, "the forms were compared");
	if (getenv("FTQ_SEGMENTED_VERBOSE"))
		printf("\n  %d segmented runs of the inline form bit-identical to the slot form\n", compared);
	routing_free(&prefill);
	routing_free(&gemv);
}

TEST_CASE("mps segmented gemm takes an ftq shared expert with an inline codebook like the same shared weight naming the table")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	// The shared expert's weights do not join the routed dispatch: they run as GEMM_FORWARD on the weight as given, so
	// an inline one is resolved there. FP16 (dense FTQ takes an FP16 base), routed FTQ and a shared [256, 512] FTQ.
	const int k = 512, n_rows = 256, shared_rows = 256;
	const ftq_table_t external9 = ftq_external9_table(), folded = ftq_folded_table();
	const ftq_table_t* const shared_tables[2] = { &external9, &folded };
	routing_t gemv = routing_of(4, small_gemv_indices, small_gemv_counts);
	ccv_nnc_tensor_t* down_host;
	ccv_nnc_tensor_t* const down = ftq_weight_new(CCV_16F, 6, n_rows, k, 48, &ftq_builtin, 0x1a5d0u, &down_host);
	float* const shared_values = activations_new(CCV_16F, 1, k, 0x1a5d1u);
	ccv_nnc_tensor_t* const shared_a = gpu_from_floats(CCV_16F, shared_values, 1, k);
	float* const a = activations_new(CCV_16F, gemv.rows, k, 0x1a5d2u);
	ccv_nnc_tensor_t* const ga = gpu_from_floats(CCV_16F, a, gemv.rows, k);
	ccv_nnc_tensor_t* const indices = gpu_ints(gemv.indices, gemv.bins);
	ccv_nnc_tensor_t* const counts = gpu_ints(gemv.counts, gemv.bins);
	int t;
	for (t = 0; t < 2; t++)
	{
		const int format = ccv_nnc_codebook_with_slot(ftq_format(34), shared_tables[t]->slot);
		ccv_nnc_tensor_param_t sp = CPU_TENSOR_NHWC(16F, shared_rows, k);
		ccv_nnc_tensor_t* const shared_host = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise_x(sp, format), 0);
		ftq_fill(shared_host->data.u8, ccv_nnc_tensor_data_size_without_padding(shared_host->info), format, shared_rows, k, shared_tables[t], 0x1a5e0u + t);
		ccv_nnc_tensor_t* const shared_inline_host = ftq_inline_host_new(shared_host);
		ccv_nnc_tensor_t* const shared_w = gpu_copy_of(shared_host);
		ccv_nnc_tensor_t* const shared_inline = gpu_copy_of(shared_inline_host);
		ccv_nnc_tensor_param_t op = GPU_TENSOR_NHWC(000, 16F, gemv.rows, n_rows);
		ccv_nnc_tensor_param_t so = GPU_TENSOR_NHWC(000, 16F, 1, shared_rows);
		ccv_nnc_tensor_t* const b[2] = { ccv_nnc_tensor_new(0, op, 0), ccv_nnc_tensor_new(0, op, 0) };
		ccv_nnc_tensor_t* const shared_b[2] = { ccv_nnc_tensor_new(0, so, 0), ccv_nnc_tensor_new(0, so, 0) };
		ccv_nnc_cmd_t gemm = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
		gemm.backend = CCV_NNC_BACKEND_MPS;
		REQUIRE_EQ(ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, down, 0, shared_a, shared_w), TENSOR_LIST(b[0], shared_b[0]), 0), CCV_NNC_EXEC_SUCCESS, "the slot-form shared expert runs, table %d", t);
		REQUIRE_EQ(ccv_nnc_cmd_exec(gemm, ccv_nnc_no_hint, 0, TENSOR_LIST(ga, indices, counts, down, 0, shared_a, shared_inline), TENSOR_LIST(b[1], shared_b[1]), 0), CCV_NNC_EXEC_SUCCESS, "the inline shared expert runs, table %d", t);
		float* const x = (float*)ccmalloc(sizeof(float) * gemv.rows * n_rows);
		float* const y = (float*)ccmalloc(sizeof(float) * gemv.rows * n_rows);
		floats_from_gpu(b[0], x);
		floats_from_gpu(b[1], y);
		REQUIRE_EQ(bit_mismatches(x, y, (size_t)gemv.rows * n_rows), 0, "the routed outputs are the same, table %d", t);
		floats_from_gpu(shared_b[0], x);
		floats_from_gpu(shared_b[1], y);
		REQUIRE_EQ(bit_mismatches(x, y, shared_rows), 0, "the shared outputs are the same, table %d", t);
		ccfree(y);
		ccfree(x);
		int i;
		for (i = 0; i < 2; i++)
		{
			ccv_nnc_tensor_free(shared_b[i]);
			ccv_nnc_tensor_free(b[i]);
		}
		ccv_nnc_tensor_free(shared_inline);
		ccv_nnc_tensor_free(shared_w);
		ccv_nnc_tensor_free(shared_inline_host);
		ccv_nnc_tensor_free(shared_host);
	}
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(ga);
	ccfree(a);
	ccv_nnc_tensor_free(shared_a);
	ccfree(shared_values);
	ccv_nnc_tensor_free(down);
	ccv_nnc_tensor_free(down_host);
	routing_free(&gemv);
}

// Only the streaming command on these sources (decode of one routed row): its exec status.
static int streaming_only(ccv_nnc_tensor_t* const* const sources, const int* const indices, const int* const counts, const int bins, const int resident_slots)
{
	routing_t r = routing_of(bins, indices, counts);
	float* const route = route_weights_new(CCV_16F, r.rows);
	ccv_nnc_tensor_t* const groute = gpu_from_floats(CCV_16F, route, r.rows, 1);
	ccv_nnc_tensor_t* const gindices = gpu_ints(indices, bins);
	ccv_nnc_tensor_t* const gcounts = gpu_ints(counts, bins);
	ccv_nnc_cmd_t streaming = CMD_MOE_WEIGHTS_STREAMING_FORWARD(resident_slots, 1);
	streaming.backend = CCV_NNC_BACKEND_MPS;
	ccv_nnc_tensor_param_t streamed_params[6] = {};
	ccv_nnc_hint_tensor_auto(streaming, TENSOR_PARAM_LIST(gindices->info, gcounts->info, groute->info, sources[0]->info, sources[1]->info, sources[2]->info), ccv_nnc_no_hint, streamed_params, 6);
	ccv_nnc_tensor_t* streamed[6];
	int i;
	for (i = 0; i < 6; i++)
	{
		if (i >= 3)
			streamed_params[i].reserved = 0;
		streamed[i] = ccv_nnc_tensor_new(0, streamed_params[i], 0);
	}
	ccv_nnc_stream_context_t* const stream = ccv_nnc_stream_context_new(CCV_STREAM_CONTEXT_GPU);
	const int status = ccv_nnc_cmd_exec(streaming, ccv_nnc_no_hint, 0, TENSOR_LIST(gindices, gcounts, groute, sources[0], sources[1], sources[2]),
		TENSOR_LIST(streamed[0], streamed[1], streamed[2], streamed[3], streamed[4], streamed[5]), stream);
	ccv_nnc_stream_context_wait(stream);
	ccv_nnc_stream_context_free(stream);
	for (i = 0; i < 6; i++)
		ccv_nnc_tensor_free(streamed[i]);
	ccv_nnc_tensor_free(gcounts);
	ccv_nnc_tensor_free(gindices);
	ccv_nnc_tensor_free(groute);
	ccfree(route);
	routing_free(&r);
	return status;
}

TEST_CASE("mps moe weights streaming rejects an inline ftq source whose header does not name the table, and ftq_vr in either form")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_MOE_WEIGHTS_STREAMING_FORWARD, CCV_NNC_BACKEND_MPS));
	const int experts = 32, n_rows = 256, k = 512;
	const ftq_table_t external9 = ftq_external9_table();
	static const int indices[] = { 3, 17, 31, 0 };
	static const int counts[] = { 1, 0, 0, 0 };
	int c;
	// 0: the down projection's header crc + 1; 1: the gate's header id 0; 2: FTQ_VR gate / up / down (no fixed expert
	// stride), slot form; 3: the same inline.
	for (c = 0; c < 4; c++)
	{
		char paths[3][64] = {
			"/tmp/ccv-moe-ftq-bad-gate-XXXXXX",
			"/tmp/ccv-moe-ftq-bad-up-XXXXXX",
			"/tmp/ccv-moe-ftq-bad-down-XXXXXX",
		};
		ccv_nnc_tensor_t* hosts[3];
		ccv_nnc_tensor_t* references[3];
		ccv_nnc_tensor_t* sources[3];
		int i;
		for (i = 0; i < 3; i++)
		{
			const int rows = i < 2 ? n_rows : k, cols = i < 2 ? k : n_rows;
			if (c < 2)
				references[i] = ftq_weight_new(CCV_16F, experts, rows, cols, 34, &external9, 0x1a6000u + c * 3 + i, &hosts[i]);
			else {
				// FTQ_VR at a single rate (nm34 everywhere) over the whole [E N, K] blob.
				const int format = ccv_nnc_codebook_with_slot(CCV_NNC_QX_8I_ROWWISE_FTQ_VR_BLOCK_FIRST + (34 - 8) / 2, external9.slot);
				hosts[i] = ccv_nnc_tensor_new(0, ftq_param(CCV_TENSOR_CPU_MEMORY, CCV_16F, experts, rows, cols, format), 0);
				const size_t size = ccv_nnc_tensor_data_size_without_padding(hosts[i]->info);
				uint8_t* const data = hosts[i]->data.u8;
				memset(data, 0, size);
				memcpy(data, "CCVFTQVR", 8);
				const int tile_rows = experts * rows / 16, tiles = cols / 16;
				const uint32_t header[6] = { (uint32_t)(experts * rows), (uint32_t)cols, 16u * 34, 1u << ((34 - 8) / 2), external9.table, external9.crc };
				memcpy(data + 8, header, sizeof(header));
				uint32_t* const off = (uint32_t*)(data + 32);
				int r;
				for (r = 0; r <= tile_rows; r++)
					off[r] = (uint32_t)(r * tiles * 34 / 2);
				uint32_t seed = 0x1a6100u + i;
				uint32_t* const code = off + tile_rows + 1;
				uint32_t w;
				for (w = 0; w < off[tile_rows]; w++)
					code[w] = rng_next(&seed);
				float* const scales = (float*)ccmalloc(sizeof(float) * experts * rows);
				for (r = 0; r < experts * rows; r++)
					scales[r] = 1e-3f;
				ccv_float_to_half_precision(scales, (uint16_t*)(data + size - 2 * (size_t)experts * rows), experts * rows);
				ccfree(scales);
				REQUIRE_EQ(ccv_nnc_8i_rowwise_x_ftq_vr_check(data, size, format, (size_t)experts * rows, cols, 0), 0, "the FTQ_VR blob passes its checks");
				references[i] = gpu_copy_of(hosts[i]);
			}
			ccv_nnc_tensor_t* const inline_host = c != 2 ? ftq_inline_host_new(hosts[i]) : 0;
			if (inline_host && ((c == 0 && i == 2) || (c == 1 && i == 0)))
			{
				const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(inline_host->info.reserved);
				uint32_t word;
				memcpy(&word, inline_host->data.u8 + prefix + (c == 0 ? 28 : 24), 4);
				word = c == 0 ? word + 1 : word & ~0xffu;
				memcpy(inline_host->data.u8 + prefix + (c == 0 ? 28 : 24), &word, 4);
			}
			const ccv_nnc_tensor_t* const streamed_host = inline_host ? inline_host : hosts[i];
			REQUIRE(write_file(streamed_host, paths[i]), "each projection is written to a temporary file");
			ccv_nnc_tensor_param_t gpu_params = streamed_host->info;
			gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
			sources[i] = ccv_nnc_tensor_new_from_file(gpu_params, paths[i], 0, CCV_NNC_TENSOR_MEMORY_MAP_ON_DEMAND);
			REQUIRE(sources[i], "the projection maps on demand");
			if (inline_host)
				ccv_nnc_tensor_free(inline_host);
		}
		REQUIRE_EQ(streaming_only(sources, indices, counts, 4, 2), CCV_NNC_EXEC_INVALID, "streaming case %d is rejected", c);
		for (i = 0; i < 3; i++)
		{
			ccv_nnc_tensor_free(sources[i]);
			ccv_nnc_tensor_free(references[i]);
			ccv_nnc_tensor_free(hosts[i]);
			unlink(paths[i]);
		}
	}
}

#include "case_main.h"
