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
#include <nnc/ccv_nnc_trellis_builtin_table.inc>
#include "trellis_table_alt.inc"

// TMCQ weights that name an external pair codebook (ccv_nnc_codebook.h) on the routed and dense MoE paths: segmented
// GEMM (fused GEMV, fused tile GEMM, selected-expert expansion), segmented SwiGLU (packed decode, fused tile GEMM,
// selected-expert expansion, fp decode, shared expert) and SwiGLU. Two kinds of checks, both bit for bit:
// 1. A table that repeats each pair of the built-in table P / 512 times (P = 512, 1024, 2048) selects the
//    built-in pair of every state, so the output with that slot equals the output with the built-in table (slot 0).
// 2. With a random table and the alternate 512-pair table the output equals the same command on the same codes decoded on the CPU
//    to resident 8I_ROWWISE (int8 rows and the same row scales), wherever the built-in table equals its own decode;
//    elsewhere (a different kernel on the 8I_ROWWISE side) it stays within tolerance. A random table repeated to 1024
//    pairs must also equal it at 512.

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

// The format of a TMCQ codec naming codebook slot `slot` (0: the built-in table).
static int tmcq_with_slot(const int codec, const int slot)
{
	return ccv_nnc_codebook_with_slot(codec, slot);
}

enum {
	TABLE_BUILTIN_512, // The built-in table itself, but as an external table.
	TABLE_BUILTIN_1024,
	TABLE_BUILTIN_2048,
	TABLE_RANDOM,
	TABLE_RANDOM_1024, // TABLE_RANDOM with each pair repeated twice.
	TABLE_ALT,
	TABLE_COUNT,
};

static int table_slots[TABLE_COUNT];

static int register_repeated(const int8_t* const base, const int rep)
{
	const int entries = 512 * rep;
	int8_t* const pairs = (int8_t*)ccmalloc((size_t)entries * 2);
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

static void register_tables(void)
{
	if (table_slots[TABLE_BUILTIN_512])
		return;
	table_slots[TABLE_BUILTIN_512] = register_repeated(ccv_nnc_trellis_builtin_table, 1);
	table_slots[TABLE_BUILTIN_1024] = register_repeated(ccv_nnc_trellis_builtin_table, 2);
	table_slots[TABLE_BUILTIN_2048] = register_repeated(ccv_nnc_trellis_builtin_table, 4);
	int8_t random[1024];
	uint32_t seed = 0x5eed1234u;
	int i;
	for (i = 0; i < 1024; i++)
		random[i] = (int8_t)((int)(rng_next(&seed) % 255) - 127);
	table_slots[TABLE_RANDOM] = register_repeated(random, 1);
	table_slots[TABLE_RANDOM_1024] = register_repeated(random, 2);
	table_slots[TABLE_ALT] = ccv_nnc_codebook_register_serialized(test_trellis_table_alt, sizeof(test_trellis_table_alt));
	for (i = 0; i < TABLE_COUNT; i++)
		assert(table_slots[i] > 0);
}

// TMCQ weights of `experts` matrices (0: one 2-D matrix) of n rows by k, as bytes (payload and row scales).
typedef struct {
	int codec;
	int datatype; // CCV_16F or CCV_32F; the row scales take the same type.
	int experts;
	int n, k;
	uint8_t* bytes;
	size_t size;
} weights_t;

static ccv_nnc_tensor_param_t dense_params(const int type, const int datatype, const int experts, const int n, const int k)
{
	ccv_nnc_tensor_param_t params = {
		.type = type,
		.format = CCV_TENSOR_FORMAT_NHWC,
		.datatype = datatype,
	};
	if (experts)
	{
		params.dim[0] = experts;
		params.dim[1] = n;
		params.dim[2] = k;
	} else {
		params.dim[0] = n;
		params.dim[1] = k;
	}
	return params;
}

static size_t total_rows(const weights_t* const w)
{
	return (size_t)(w->experts ? w->experts : 1) * w->n;
}

// The tile grid: (rows / 16) (k / 16) tiles of 2 n bytes (rows (k / 256) 16 n bits when k % 256 == 0).
static size_t payload_bytes(const weights_t* const w)
{
	return (total_rows(w) / 16) * (w->k / 16) * 2 * ccv_nnc_8i_rowwise_x_tmcq_rate(w->codec);
}

// TMCQ parameters of the weights: ccv_nnc_tensor_8i_rowwise_x for k % 256 == 0, the same fields made directly for any
// other k % 16 == 0 (the CPU tensor constructor asserts k % 256 until the CPU codec takes K % 16; its size is never
// smaller than the tile grid).
static ccv_nnc_tensor_param_t tmcq_x_params(const ccv_nnc_tensor_param_t params, const int format)
{
	const int nd = ccv_nnc_tensor_nd(params.dim);
	if (params.dim[nd - 1] % 256 == 0)
		return ccv_nnc_tensor_8i_rowwise_x(params, format);
	ccv_nnc_tensor_param_t x = params;
	x.datatype = ((params.datatype >> 12) & 0xff) | CCV_QX | CCV_NNC_QX_8I_ROWWISE_X;
	x.reserved = format;
	return x;
}

static weights_t weights_new(const int codec, const int datatype, const int experts, const int n, const int k, uint32_t seed)
{
	weights_t w = { .codec = codec, .datatype = datatype, .experts = experts, .n = n, .k = k };
	w.size = ccv_nnc_tensor_data_size_without_padding(tmcq_x_params(dense_params(CCV_TENSOR_CPU_MEMORY, datatype, experts, n, k), codec));
	w.bytes = (uint8_t*)cccalloc(w.size, 1);
	const size_t payload = payload_bytes(&w);
	const size_t scale_offset = (payload + 127) & ~(size_t)127;
	size_t i;
	for (i = 0; i < payload; i++)
		w.bytes[i] = (uint8_t)(rng_next(&seed) >> 24);
	const size_t rows = total_rows(&w);
	for (i = 0; i < rows; i++)
	{
		const float s = (float)((rng_next(&seed) >> 8) & 0xffff) / 65536.0f * 0.02f + 1e-4f;
		if (datatype == CCV_32F)
			((float*)(w.bytes + scale_offset))[i] = s;
		else
			ccv_float_to_half_precision((float*)&s, (uint16_t*)(w.bytes + scale_offset) + i, 1);
	}
	return w;
}

static void weights_free(weights_t* const w)
{
	ccfree(w->bytes);
	w->bytes = 0;
}

static ccv_nnc_tensor_t* gpu_tensor_from(const ccv_nnc_tensor_param_t cpu_params, const void* const bytes, const size_t size)
{
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, cpu_params, 0);
	assert(ccv_nnc_tensor_data_size_without_padding(cpu_params) == size);
	memcpy(host->data.u8, bytes, size);
	ccv_nnc_tensor_param_t gpu_params = cpu_params;
	gpu_params.type = CCV_TENSOR_GPU_MEMORY | 000;
	ccv_nnc_tensor_t* const gpu = ccv_nnc_tensor_new(0, gpu_params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(host), TENSOR_LIST(gpu), 0);
	ccv_nnc_tensor_free(host);
	return gpu;
}

// The TMCQ weights on the GPU, naming codebook slot `slot`.
static ccv_nnc_tensor_t* tmcq_gpu(const weights_t* const w, const int slot)
{
	return gpu_tensor_from(tmcq_x_params(dense_params(CCV_TENSOR_CPU_MEMORY, w->datatype, w->experts, w->n, w->k), tmcq_with_slot(w->codec, slot)), w->bytes, w->size);
}

// The TMCQ weights on the GPU with the table of `slot` inline (ccv_nnc_codebook.h): the serialized table, zeros up
// to 128 + 2 P bytes, then the same bytes as tmcq_gpu. pairs, if not NULL, replaces the table values (pair_count
// pairs, written unchecked after the header of slot 0), e.g. to make it invalid.
static ccv_nnc_tensor_t* tmcq_gpu_inline(const weights_t* const w, const int slot, const int8_t* const pairs, const int pair_count)
{
	int entries = pair_count;
	if (!pairs)
		ccv_nnc_codebook_get(slot, &entries, 0);
	const int format = ccv_nnc_8i_rowwise_x_with_inline_codebook(w->codec, entries);
	const size_t prefix = ccv_nnc_8i_rowwise_x_inline_codebook_size(format);
	uint8_t* const bytes = (uint8_t*)cccalloc(prefix + w->size, 1);
	const size_t written = ccv_nnc_codebook_serialize(pairs ? 0 : slot, bytes, prefix);
	assert(written == CCV_NNC_CODEBOOK_HEADER_SIZE + (size_t)entries * 2);
	if (pairs)
		memcpy(bytes + CCV_NNC_CODEBOOK_HEADER_SIZE, pairs, (size_t)entries * 2);
	memcpy(bytes + prefix, w->bytes, w->size);
	ccv_nnc_tensor_t* const gpu = gpu_tensor_from(tmcq_x_params(dense_params(CCV_TENSOR_CPU_MEMORY, w->datatype, w->experts, w->n, w->k), format), bytes, prefix + w->size);
	ccfree(bytes);
	return gpu;
}

// TMCQ reference decode with the table of `slot`: tile g holds 16 n bits from bit 16 n g; weight t reads the 15-bit
// window of weight t | 1 ending at ((t | 1) + 1) n / 16, wrapping in the tile, then half t & 1 of its pair
// ((window * 0x83DCD12D) mod 2^32) >> (32 - log2 P).
static void tmcq_ref_decode(const weights_t* const w, const int slot, int8_t* const q8)
{
	int entries = 0;
	const int8_t* const pairs = ccv_nnc_codebook_get(slot, &entries, 0);
	const int shift = 32 - ccv_nnc_codebook_log2_entries(entries);
	const int n = ccv_nnc_8i_rowwise_x_tmcq_rate(w->codec);
	const int rows = (int)total_rows(w), cols = w->k;
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
					cw |= (uint32_t)((w->bytes[p >> 3] >> (p & 7)) & 1) << j;
				}
				q8[(size_t)(tr * 16 + t / 16) * cols + tc * 16 + t % 16] = pairs[2 * ((cw * 0x83DCD12Du) >> shift) + (t & 1)];
			}
}

// The same codes decoded with the table of `slot` to resident 8I_ROWWISE weights on the GPU: int8 rows, then the row
// scales at the next 128-byte boundary.
static ccv_nnc_tensor_t* resident_gpu(const weights_t* const w, const int slot)
{
	const ccv_nnc_tensor_param_t params = ccv_nnc_tensor_8i_rowwise(dense_params(CCV_TENSOR_CPU_MEMORY, w->datatype, w->experts, w->n, w->k));
	const size_t size = ccv_nnc_tensor_data_size_without_padding(params);
	const size_t count = total_rows(w) * w->k;
	const size_t scale_size = total_rows(w) * CCV_GET_DATA_TYPE_SIZE(w->datatype);
	uint8_t* const bytes = (uint8_t*)cccalloc(size, 1);
	tmcq_ref_decode(w, slot, (int8_t*)bytes);
	memcpy(bytes + ((count + 127) & ~(size_t)127), w->bytes + ((payload_bytes(w) + 127) & ~(size_t)127), scale_size);
	ccv_nnc_tensor_t* const gpu = gpu_tensor_from(params, bytes, size);
	ccfree(bytes);
	return gpu;
}

// The same codes decoded with the table of `slot` to dense weights of the weights' datatype: q8 * scale in FP32, then
// rounded to the datatype.
static ccv_nnc_tensor_t* dense_gpu(const weights_t* const w, const int slot)
{
	const ccv_nnc_tensor_param_t params = dense_params(CCV_TENSOR_CPU_MEMORY, w->datatype, w->experts, w->n, w->k);
	const size_t rows = total_rows(w);
	const size_t count = rows * w->k;
	int8_t* const q8 = (int8_t*)ccmalloc(count);
	tmcq_ref_decode(w, slot, q8);
	const uint8_t* const scales = w->bytes + ((payload_bytes(w) + 127) & ~(size_t)127);
	float* const values = (float*)ccmalloc(sizeof(float) * count);
	size_t i;
	for (i = 0; i < count; i++)
	{
		float scale;
		if (w->datatype == CCV_32F)
			scale = ((const float*)scales)[i / w->k];
		else
			ccv_half_precision_to_float((uint16_t*)scales + i / w->k, &scale, 1);
		values[i] = q8[i] * scale;
	}
	const size_t size = ccv_nnc_tensor_data_size_without_padding(params);
	uint8_t* const bytes = (uint8_t*)ccmalloc(size);
	if (w->datatype == CCV_32F)
		memcpy(bytes, values, size);
	else
		ccv_float_to_half_precision(values, (uint16_t*)bytes, count);
	ccv_nnc_tensor_t* const gpu = gpu_tensor_from(params, bytes, size);
	ccfree(bytes);
	ccfree(values);
	ccfree(q8);
	return gpu;
}

static ccv_nnc_tensor_t* activations_gpu(const int datatype, const int rows, const int k, const int salt)
{
	ccv_nnc_tensor_param_t params = dense_params(CCV_TENSOR_CPU_MEMORY, datatype, 0, rows, k);
	float* const values = (float*)ccmalloc(sizeof(float) * rows * k);
	int i;
	for (i = 0; i < rows * k; i++)
		values[i] = (float)(((i * 37 + salt * 11) % 67) - 33) / 64.0f;
	const size_t size = ccv_nnc_tensor_data_size_without_padding(params);
	uint8_t* const bytes = (uint8_t*)ccmalloc(size);
	if (datatype == CCV_32F)
		memcpy(bytes, values, size);
	else
		ccv_float_to_half_precision(values, (uint16_t*)bytes, rows * k);
	ccv_nnc_tensor_t* const gpu = gpu_tensor_from(params, bytes, size);
	ccfree(bytes);
	ccfree(values);
	return gpu;
}

static ccv_nnc_tensor_t* int_gpu(const int* const values, const int count)
{
	return gpu_tensor_from(CPU_TENSOR_NHWC(32S, count), values, sizeof(int) * count);
}

static uint8_t* gpu_bytes(ccv_nnc_tensor_t* const gpu, size_t* const size)
{
	ccv_nnc_tensor_param_t params = gpu->info;
	params.type = CCV_TENSOR_CPU_MEMORY;
	ccv_nnc_tensor_t* const host = ccv_nnc_tensor_new(0, params, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gpu), TENSOR_LIST(host), 0);
	*size = ccv_nnc_tensor_data_size_without_padding(params);
	uint8_t* const bytes = (uint8_t*)ccmalloc(*size);
	memcpy(bytes, host->data.u8, *size);
	ccv_nnc_tensor_free(host);
	return bytes;
}

// A routed batch: rows activation rows over `bins` bins of selected experts.
typedef struct {
	const char* name;
	int rows;
	int bins;
	int indices[6];
	int counts[6];
	int no_neural_accelerators;
} routing_t;

static void with_flags(const int no_neural_accelerators, uint64_t* const old_flags)
{
	*old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (no_neural_accelerators)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

static void restore_flags(const uint64_t old_flags)
{
	if (old_flags & CCV_NNC_DISABLE_MFA)
		ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

static int run_segmented_gemm(const routing_t* const r, const int datatype, const int n, const int k, ccv_nnc_tensor_t* const w, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* const a = activations_gpu(datatype, r->rows, k, 1);
	ccv_nnc_tensor_t* const indices = int_gpu(r->indices, r->bins);
	ccv_nnc_tensor_t* const counts = int_gpu(r->counts, r->bins);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, dense_params(CCV_TENSOR_GPU_MEMORY | 000, datatype, 0, r->rows, n), 0);
	uint64_t old_flags;
	with_flags(r->no_neural_accelerators, &old_flags);
	ccv_nnc_cmd_t cmd = CMD_SEGMENTED_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(1, 2));
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, w), TENSOR_LIST(b), 0);
	restore_flags(old_flags);
	*out = gpu_bytes(b, out_size);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	return status;
}

// Segmented SwiGLU over the routed batch, with a one-row shared expert (shared_gate / shared_up, 2-D) when given; the
// shared output follows the routed one in *out.
static int run_segmented_swiglu(const routing_t* const r, const int datatype, const int n, const int k, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, ccv_nnc_tensor_t* const shared_gate, ccv_nnc_tensor_t* const shared_up, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* const a = activations_gpu(datatype, r->rows, k, 2);
	ccv_nnc_tensor_t* const indices = int_gpu(r->indices, r->bins);
	ccv_nnc_tensor_t* const counts = int_gpu(r->counts, r->bins);
	ccv_nnc_tensor_t* const route = activations_gpu(datatype, r->rows, 1, 3);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, dense_params(CCV_TENSOR_GPU_MEMORY | 000, datatype, 0, r->rows, n), 0);
	const int shared_rows = shared_gate ? shared_gate->info.dim[0] : 0;
	ccv_nnc_tensor_t* const shared_a = shared_gate ? activations_gpu(datatype, 1, k, 4) : 0;
	ccv_nnc_tensor_t* const shared_output = shared_gate ? ccv_nnc_tensor_new(0, dense_params(CCV_TENSOR_GPU_MEMORY | 000, datatype, 0, 1, shared_rows), 0) : 0;
	uint64_t old_flags;
	with_flags(r->no_neural_accelerators, &old_flags);
	ccv_nnc_cmd_t cmd = CMD_SEGMENTED_SWIGLU_FORWARD(10);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	int status;
	if (shared_gate)
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, gate, up, route, shared_a, shared_gate, shared_up), TENSOR_LIST(output, shared_output), 0);
	else
		status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, indices, counts, gate, up, route), TENSOR_LIST(output), 0);
	restore_flags(old_flags);
	size_t routed_size;
	uint8_t* const routed = gpu_bytes(output, &routed_size);
	*out = routed;
	*out_size = routed_size;
	if (shared_gate)
	{
		size_t shared_size;
		uint8_t* const shared = gpu_bytes(shared_output, &shared_size);
		*out = (uint8_t*)ccrealloc(routed, routed_size + shared_size);
		memcpy(*out + routed_size, shared, shared_size);
		*out_size = routed_size + shared_size;
		ccfree(shared);
		ccv_nnc_tensor_free(shared_output);
		ccv_nnc_tensor_free(shared_a);
	}
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(route);
	ccv_nnc_tensor_free(counts);
	ccv_nnc_tensor_free(indices);
	ccv_nnc_tensor_free(a);
	return status;
}

static int run_swiglu(const int rows, const int no_neural_accelerators, const int datatype, const int n, const int k, const int salt, ccv_nnc_tensor_t* const gate, ccv_nnc_tensor_t* const up, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* const a = activations_gpu(datatype, rows, k, salt);
	ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, dense_params(CCV_TENSOR_GPU_MEMORY | 000, datatype, 0, rows, n), 0);
	uint64_t old_flags;
	with_flags(no_neural_accelerators, &old_flags);
	ccv_nnc_cmd_t cmd = CMD_SWIGLU_FORWARD(10);
	cmd.backend = CCV_NNC_BACKEND_MPS;
	const int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a, gate, up), TENSOR_LIST(output), 0);
	restore_flags(old_flags);
	*out = gpu_bytes(output, out_size);
	ccv_nnc_tensor_free(output);
	ccv_nnc_tensor_free(a);
	return status;
}

// Experts 8, n 256, k 512 (n % 256 and k % 256 for the SwiGLU paths). With neural accelerators, segmented GEMM takes
// the fused GEMV up to 6 rows here and the fused tile GEMM at 18 and 64 rows; 512 and 4097 rows in a bin expand the
// selected experts first; without them, it expands them for Int8MatMul (FP32) or decodes to FP16 for the GEMM.
// Segmented SwiGLU takes the packed decode up to 18 rows, then the same matrix paths. SwiGLU (one 2-D matrix, the
// row count only) decodes to int8 for the scaled matrix path, or to the datatype without neural accelerators.
static const routing_t routings[] = {
	{ "1 row", 1, 1, { 5 }, { 1 }, 0 },
	{ "6 rows over 6 experts", 6, 6, { 7, 1, 5, 2, 6, 3 }, { 1, 1, 1, 1, 1, 1 }, 0 },
	{ "18 rows over 6 experts", 18, 6, { 7, 1, 5, 2, 6, 3 }, { 3, 3, 3, 3, 3, 3 }, 0 },
	{ "64 rows", 64, 6, { 7, 1, 5, 2, 6, 3 }, { 2, 14, 12, 12, 12, 12 }, 0 },
	{ "64 rows without neural accelerators", 64, 6, { 7, 1, 5, 2, 6, 3 }, { 2, 14, 12, 12, 12, 12 }, 1 },
	{ "512 rows", 512, 2, { 4, 0 }, { 256, 256 }, 0 },
	{ "4097 rows in one bin", 4097, 1, { 6 }, { 4097 }, 0 },
};

static const int datatypes[] = { CCV_16F, CCV_32F };

typedef enum {
	PATH_SEGMENTED_GEMM,
	PATH_SEGMENTED_SWIGLU,
	PATH_SEGMENTED_SWIGLU_SHARED,
	PATH_SWIGLU,
} path_t;

static const char* const path_names[] = { "segmented gemm", "segmented swiglu", "segmented swiglu + shared", "swiglu" };

// run_path reference: the TMCQ weights with the table of the slot inline instead of naming the slot.
#define RUN_INLINE (-1)

// One run of `path` with every TMCQ weight on slot `slot`; reference 1 / 2: the same codes decoded with that slot's
// table to 8I_ROWWISE / to dense weights instead; RUN_INLINE: the slot's table inline in every TMCQ weight. Weights:
// [0] gate (or w), [1] up, [2] shared gate, [3] shared up.
static int run_path(const path_t path, const routing_t* const r, const int datatype, const weights_t* const weights, const int slot, const int reference, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* tensors[4] = { 0, 0, 0, 0 };
	const int count = path == PATH_SEGMENTED_GEMM ? 1 : path == PATH_SEGMENTED_SWIGLU_SHARED ? 4 : 2;
	int i;
	for (i = 0; i < count; i++)
		tensors[i] = reference == 1 ? resident_gpu(&weights[i], slot) : reference == 2 ? dense_gpu(&weights[i], slot) :
			reference == RUN_INLINE ? tmcq_gpu_inline(&weights[i], slot, 0, 0) : tmcq_gpu(&weights[i], slot);
	const int n = weights[0].n, k = weights[0].k;
	int status;
	if (path == PATH_SEGMENTED_GEMM)
		status = run_segmented_gemm(r, datatype, n, k, tensors[0], out, out_size);
	else if (path == PATH_SWIGLU)
		status = run_swiglu(r->rows, r->no_neural_accelerators, datatype, n, k, 5, tensors[0], tensors[1], out, out_size);
	else
		status = run_segmented_swiglu(r, datatype, n, k, tensors[0], tensors[1], tensors[2], tensors[3], out, out_size);
	for (i = 0; i < count; i++)
		ccv_nnc_tensor_free(tensors[i]);
	return status;
}

// The reference of a segmented SwiGLU with a TMCQ shared expert: the routed experts alone, then the shared expert as
// SWIGLU_FORWARD on its activation row, both on slot `slot` (the command runs an unfused shared expert that way).
static int run_shared_parts(const routing_t* const r, const int datatype, const weights_t* const weights, const int slot, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* tensors[4];
	int i;
	for (i = 0; i < 4; i++)
		tensors[i] = tmcq_gpu(&weights[i], slot);
	uint8_t* routed;
	size_t routed_size;
	int status = run_segmented_swiglu(r, datatype, weights[0].n, weights[0].k, tensors[0], tensors[1], 0, 0, &routed, &routed_size);
	uint8_t* shared = 0;
	size_t shared_size = 0;
	if (status == CCV_NNC_EXEC_SUCCESS)
		status = run_swiglu(1, r->no_neural_accelerators, datatype, weights[2].n, weights[2].k, 4, tensors[2], tensors[3], &shared, &shared_size);
	*out = (uint8_t*)ccrealloc(routed, routed_size + shared_size);
	if (shared)
		memcpy(*out + routed_size, shared, shared_size);
	*out_size = routed_size + shared_size;
	ccfree(shared);
	for (i = 0; i < 4; i++)
		ccv_nnc_tensor_free(tensors[i]);
	return status;
}

// SwiGLU of up to 6 rows on 8I_ROWWISE weights takes the GEMV-style Int8SwiGLU, which TMCQ never does; this reference
// runs 7 rows (the first ones the same activations) through the matrix path on 8I_ROWWISE and keeps the first rows.
static int run_swiglu_matrix_reference(const routing_t* const r, const int datatype, const weights_t* const weights, const int slot, uint8_t** const out, size_t* const out_size)
{
	ccv_nnc_tensor_t* const gate = resident_gpu(&weights[0], slot);
	ccv_nnc_tensor_t* const up = resident_gpu(&weights[1], slot);
	const int status = run_swiglu(7, r->no_neural_accelerators, datatype, weights[0].n, weights[0].k, 5, gate, up, out, out_size);
	*out_size = (size_t)r->rows * weights[0].n * CCV_GET_DATA_TYPE_SIZE(datatype);
	ccv_nnc_tensor_free(up);
	ccv_nnc_tensor_free(gate);
	return status;
}

static void make_path_weights(const path_t path, const int datatype, const int routing_index, weights_t* const weights)
{
	const uint32_t seed = 0x1000u * (path + 1) + 0x10u * routing_index + datatype;
	const int experts = path == PATH_SWIGLU ? 0 : 8;
	weights[0] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatype, experts, 256, 512, seed + 1);
	weights[1] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatype, experts, 256, 512, seed + 2);
	weights[2] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatype, 0, 256, 512, seed + 3);
	weights[3] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatype, 0, 256, 512, seed + 4);
}

// Whether `path` runs `routing` (shapes / datatypes worth covering; SwiGLU has no routing, only its row count).
static int path_takes(const path_t path, const int datatype, const int routing_index)
{
	const routing_t* const r = &routings[routing_index];
	if (r->rows == 4097)
		return path == PATH_SEGMENTED_GEMM && datatype == CCV_16F;
	if (path == PATH_SEGMENTED_SWIGLU_SHARED)
		return r->rows == 1 || r->rows == 6 || r->rows == 64;
	if (path == PATH_SWIGLU)
		return r->rows != 18 && r->rows != 512;
	return 1;
}

static double max_rel_difference(const int datatype, const uint8_t* const x, const uint8_t* const y, const size_t size)
{
	const size_t count = size / CCV_GET_DATA_TYPE_SIZE(datatype);
	double max_rel = 0;
	size_t i;
	for (i = 0; i < count; i++)
	{
		float a, b;
		if (datatype == CCV_32F)
		{
			a = ((const float*)x)[i];
			b = ((const float*)y)[i];
		} else {
			ccv_half_precision_to_float((uint16_t*)x + i, &a, 1);
			ccv_half_precision_to_float((uint16_t*)y + i, &b, 1);
		}
		if (!isfinite(a) || !isfinite(b))
			return INFINITY;
		const double diff = fabs((double)a - b);
		max_rel = ccv_max(max_rel, diff / ccv_max(1.0, ccv_max(fabs((double)a), fabs((double)b))));
	}
	return max_rel;
}

TEST_CASE("mps tmcq codebook paths: the built-in table expanded to 512, 1024 and 2048 pairs equals the built-in table bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	static const int tables[] = { TABLE_BUILTIN_512, TABLE_BUILTIN_1024, TABLE_BUILTIN_2048 };
	int p, d, ri, t;
	for (p = PATH_SEGMENTED_GEMM; p <= PATH_SWIGLU; p++)
		for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
			for (ri = 0; ri < sizeof(routings) / sizeof(routings[0]); ri++)
			{
				if (!path_takes((path_t)p, datatypes[d], ri))
					continue;
				weights_t weights[4];
				make_path_weights((path_t)p, datatypes[d], ri, weights);
				uint8_t* builtin;
				size_t builtin_size;
				REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, 0, 0, &builtin, &builtin_size), CCV_NNC_EXEC_SUCCESS,
					"%s with the built-in table should run: %s datatype=%d", path_names[p], routings[ri].name, datatypes[d]);
				for (t = 0; t < sizeof(tables) / sizeof(tables[0]); t++)
				{
					uint8_t* external;
					size_t external_size;
					REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, table_slots[tables[t]], 0, &external, &external_size), CCV_NNC_EXEC_SUCCESS,
						"%s with an external table should run: %s datatype=%d table=%d", path_names[p], routings[ri].name, datatypes[d], tables[t]);
					REQUIRE(external_size == builtin_size && memcmp(external, builtin, builtin_size) == 0,
						"%s with the built-in table expanded to %d pairs should equal the built-in table bit for bit: %s datatype=%d max rel %g",
						path_names[p], 512 << t, routings[ri].name, datatypes[d], max_rel_difference(datatypes[d], external, builtin, builtin_size));
					ccfree(external);
				}
				ccfree(builtin);
				int i;
				for (i = 0; i < 4; i++)
					weights_free(&weights[i]);
			}
}

// Where the TMCQ kernel and the 8I_ROWWISE kernel of a path differ in arithmetic, so does the built-in table against its
// own decode (slot 0 against the built-in table decoded to 8I_ROWWISE); then an external table must stay within that path's tolerance.
static double path_tolerance(const int datatype)
{
	return datatype == CCV_32F ? 1e-5 : 3e-3;
}

TEST_CASE("mps tmcq codebook paths: random and alternate tables equal the same codes decoded to 8i rowwise or dense weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	static const int tables[] = { 0, TABLE_RANDOM, TABLE_ALT };
	static const char* const table_names[] = { "built-in", "random", "alternate" };
	static const char* const reference_names[] = { "", "8i rowwise", "dense", "routed and shared apart", "8i rowwise over 7 rows" };
	int p, d, ri, t;
	for (p = PATH_SEGMENTED_GEMM; p <= PATH_SWIGLU; p++)
		for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
			for (ri = 0; ri < sizeof(routings) / sizeof(routings[0]); ri++)
			{
				if (!path_takes((path_t)p, datatypes[d], ri))
					continue;
				weights_t weights[4];
				make_path_weights((path_t)p, datatypes[d], ri, weights);
				// The reference the built-in table equals bit for bit (0: none, then within tolerance of 8i rowwise).
				int reference = 0;
				for (t = 0; t < sizeof(tables) / sizeof(tables[0]); t++)
				{
					const int slot = t ? table_slots[tables[t]] : 0;
					uint8_t* external;
					size_t external_size;
					REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, 0, &external, &external_size), CCV_NNC_EXEC_SUCCESS,
						"%s with the %s table should run: %s datatype=%d", path_names[p], table_names[t], routings[ri].name, datatypes[d]);
					// A shared expert on 8I_ROWWISE or dense weights takes other kernels than TMCQ (it may join the packed
					// dispatch); its reference is the routed and shared parts run apart, each checked here on its own.
					const int first = p == PATH_SEGMENTED_SWIGLU_SHARED ? 3 : 1;
					const int last = p == PATH_SEGMENTED_SWIGLU_SHARED ? 3 : p == PATH_SWIGLU && routings[ri].rows <= 6 ? 4 : 2;
					int k;
					double resident_max_rel = INFINITY;
					int identical_reference = 0;
					for (k = first; k <= last; k++)
					{
						if ((t > 0 && reference && k != reference) || (k == 3 && first != 3))
							continue;
						uint8_t* expected;
						size_t expected_size;
						if (k == 4)
						{
							REQUIRE_EQ(run_swiglu_matrix_reference(&routings[ri], datatypes[d], weights, slot, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
								"%s on the 8i rowwise decode over 7 rows of the %s table should run: %s datatype=%d", path_names[p], table_names[t], routings[ri].name, datatypes[d]);
						} else if (k == 3) {
							REQUIRE_EQ(run_shared_parts(&routings[ri], datatypes[d], weights, slot, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
								"%s parts with the %s table should run: %s datatype=%d", path_names[p], table_names[t], routings[ri].name, datatypes[d]);
						} else {
							REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, k, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
								"%s on the %s decode of the %s table should run: %s datatype=%d", path_names[p], reference_names[k], table_names[t], routings[ri].name, datatypes[d]);
						}
						const double max_rel = max_rel_difference(datatypes[d], external, expected, expected_size);
						const int identical = external_size == expected_size && memcmp(external, expected, expected_size) == 0;
						printf("[tmcq codebook] %s | %s | %s | %s vs %s | %s (max rel %g)\n", path_names[p], routings[ri].name, datatypes[d] == CCV_32F ? "32F" : "16F",
							table_names[t], reference_names[k], identical ? "bit-identical" : "differs", max_rel);
						if (k == 1)
							resident_max_rel = max_rel;
						if (identical && !identical_reference)
							identical_reference = k;
						ccfree(expected);
					}
					if (t == 0)
						reference = identical_reference;
					else if (reference) {
						REQUIRE_EQ(identical_reference, reference, "%s with the %s table should equal its %s decode bit for bit, as the built-in table does: %s datatype=%d",
							path_names[p], table_names[t], reference_names[reference], routings[ri].name, datatypes[d]);
					} else {
						REQUIRE(resident_max_rel < path_tolerance(datatypes[d]), "%s with the %s table should match its 8i rowwise decode: %s datatype=%d max rel %g",
							path_names[p], table_names[t], routings[ri].name, datatypes[d], resident_max_rel);
					}
					ccfree(external);
				}
				int i;
				for (i = 0; i < 4; i++)
					weights_free(&weights[i]);
			}
}

TEST_CASE("mps tmcq codebook paths: a random table repeated to 1024 pairs equals it at 512 bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	int p, d, ri;
	for (p = PATH_SEGMENTED_GEMM; p <= PATH_SWIGLU; p++)
		for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
			for (ri = 0; ri < sizeof(routings) / sizeof(routings[0]); ri++)
			{
				if (!path_takes((path_t)p, datatypes[d], ri))
					continue;
				weights_t weights[4];
				make_path_weights((path_t)p, datatypes[d], ri, weights);
				uint8_t* small;
				size_t small_size;
				uint8_t* large;
				size_t large_size;
				REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, table_slots[TABLE_RANDOM], 0, &small, &small_size), CCV_NNC_EXEC_SUCCESS,
					"%s with the random table should run: %s datatype=%d", path_names[p], routings[ri].name, datatypes[d]);
				REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, table_slots[TABLE_RANDOM_1024], 0, &large, &large_size), CCV_NNC_EXEC_SUCCESS,
					"%s with the random table at 1024 pairs should run: %s datatype=%d", path_names[p], routings[ri].name, datatypes[d]);
				REQUIRE(small_size == large_size && memcmp(small, large, small_size) == 0,
					"%s with the random table at 1024 pairs should equal it at 512 bit for bit: %s datatype=%d", path_names[p], routings[ri].name, datatypes[d]);
				ccfree(large);
				ccfree(small);
				int i;
				for (i = 0; i < 4; i++)
					weights_free(&weights[i]);
			}
}

TEST_CASE("mps tmcq codebook paths: an unregistered slot or mismatched gate and up slots are rejected")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	weights_t weights[4];
	make_path_weights(PATH_SEGMENTED_SWIGLU, CCV_16F, 0, weights);
	const routing_t* const r = &routings[1];
	ccv_nnc_tensor_t* const unregistered = tmcq_gpu(&weights[0], CCV_NNC_CODEBOOK_INLINE_SLOT - 1);
	ccv_nnc_tensor_t* const gate = tmcq_gpu(&weights[0], table_slots[TABLE_RANDOM]);
	ccv_nnc_tensor_t* const up = tmcq_gpu(&weights[1], table_slots[TABLE_ALT]);
	uint8_t* out;
	size_t out_size;
	REQUIRE_EQ(run_segmented_gemm(r, CCV_16F, 256, 512, unregistered, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented gemm rejects an unregistered slot");
	ccfree(out);
	REQUIRE_EQ(run_segmented_swiglu(r, CCV_16F, 256, 512, gate, up, 0, 0, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented swiglu rejects gate and up on different tables");
	ccfree(out);
	REQUIRE_EQ(run_segmented_swiglu(r, CCV_16F, 256, 512, unregistered, unregistered, 0, 0, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented swiglu rejects an unregistered slot");
	ccfree(out);
	ccv_nnc_tensor_free(up);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(unregistered);
	int i;
	for (i = 0; i < 4; i++)
		weights_free(&weights[i]);
}

// An inline codebook is the same table as its slot: every path, datatype and routing, tables of 512 (random and
// alternate), 1024 and 2048 pairs, plus a K % 16 shape (48 x 272, N % 64 != 0, which expands the selected experts).
TEST_CASE("mps tmcq codebook paths: an inline codebook equals the same table in its slot bit for bit")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	static const int tables[] = { TABLE_RANDOM, TABLE_ALT, TABLE_BUILTIN_1024, TABLE_BUILTIN_2048 };
	int sh, p, d, ri, t;
	for (sh = 0; sh < 2; sh++)
		for (p = PATH_SEGMENTED_GEMM; p <= PATH_SWIGLU; p++)
			for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
				for (ri = 0; ri < sizeof(routings) / sizeof(routings[0]); ri++)
				{
					if (!path_takes((path_t)p, datatypes[d], ri) || (sh && routings[ri].rows == 4097))
						continue;
					weights_t weights[4];
					if (sh == 0)
						make_path_weights((path_t)p, datatypes[d], ri, weights);
					else {
						const uint32_t seed = 0x27000u + 0x1000u * (p + 1) + 0x10u * ri + datatypes[d];
						const int experts = p == PATH_SWIGLU ? 0 : 8;
						weights[0] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatypes[d], experts, 48, 272, seed + 1);
						weights[1] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatypes[d], experts, 48, 272, seed + 2);
						weights[2] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatypes[d], 0, 48, 272, seed + 3);
						weights[3] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatypes[d], 0, 48, 272, seed + 4);
					}
					for (t = 0; t < sizeof(tables) / sizeof(tables[0]); t++)
					{
						const int slot = table_slots[tables[t]];
						uint8_t* by_slot;
						size_t by_slot_size;
						REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, 0, &by_slot, &by_slot_size), CCV_NNC_EXEC_SUCCESS,
							"%s with the table in its slot should run: %s datatype=%d table=%d shape=%d", path_names[p], routings[ri].name, datatypes[d], tables[t], sh);
						uint8_t* by_inline;
						size_t by_inline_size;
						REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, RUN_INLINE, &by_inline, &by_inline_size), CCV_NNC_EXEC_SUCCESS,
							"%s with the table inline should run: %s datatype=%d table=%d shape=%d", path_names[p], routings[ri].name, datatypes[d], tables[t], sh);
						REQUIRE(by_inline_size == by_slot_size && memcmp(by_inline, by_slot, by_slot_size) == 0,
							"%s with the table inline should equal it in its slot bit for bit: %s datatype=%d table=%d shape=%d max rel %g",
							path_names[p], routings[ri].name, datatypes[d], tables[t], sh, max_rel_difference(datatypes[d], by_inline, by_slot, by_slot_size));
						ccfree(by_inline);
						ccfree(by_slot);
					}
					int i;
					for (i = 0; i < 4; i++)
						weights_free(&weights[i]);
				}
}

TEST_CASE("mps tmcq codebook paths: an invalid inline codebook or gate and up on different inline tables are rejected")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	weights_t weights[4];
	make_path_weights(PATH_SEGMENTED_SWIGLU, CCV_16F, 0, weights);
	const routing_t* const r = &routings[1];
	// -128 is never a code: a table with one is invalid, whatever its slot would be.
	int entries = 0;
	const int8_t* const random = ccv_nnc_codebook_get(table_slots[TABLE_RANDOM], &entries, 0);
	int8_t bad[1024];
	memcpy(bad, random, sizeof(bad));
	bad[333] = -128;
	ccv_nnc_tensor_t* const invalid = tmcq_gpu_inline(&weights[0], 0, bad, 512);
	ccv_nnc_tensor_t* const gate = tmcq_gpu_inline(&weights[0], table_slots[TABLE_RANDOM], 0, 0);
	ccv_nnc_tensor_t* const up = tmcq_gpu_inline(&weights[1], table_slots[TABLE_ALT], 0, 0);
	uint8_t* out;
	size_t out_size;
	REQUIRE_EQ(run_segmented_gemm(r, CCV_16F, 256, 512, invalid, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented gemm rejects an invalid inline codebook");
	ccfree(out);
	REQUIRE_EQ(run_segmented_swiglu(r, CCV_16F, 256, 512, invalid, invalid, 0, 0, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented swiglu rejects an invalid inline codebook");
	ccfree(out);
	REQUIRE_EQ(run_segmented_swiglu(r, CCV_16F, 256, 512, gate, up, 0, 0, &out, &out_size), CCV_NNC_EXEC_INVALID, "segmented swiglu rejects gate and up on different inline tables");
	ccfree(out);
	ccv_nnc_tensor_free(up);
	ccv_nnc_tensor_free(gate);
	ccv_nnc_tensor_free(invalid);
	int i;
	for (i = 0; i < 4; i++)
		weights_free(&weights[i]);
}

// K % 16 on the same paths, built-in and random table: experts of N x K = 576 x 1152 (72 tiles a row; N % 64 and
// K % 64, so the fused tile GEMM applies), 48 x 272 (17 tiles a row, odd; N % 64 != 0 and K % 64 != 0, so prefill
// expands the selected experts) and 128 x 320 (20 tiles, K % 64 == 0 under 512: only the fused GEMM's K tail loop),
// the shared expert and SwiGLU of the same N x K. Every run must succeed and equal the same codes decoded to
// 8I_ROWWISE or dense weights bit for bit, or stay within tolerance of the 8I_ROWWISE run (other kernels there).
TEST_CASE("mps tmcq codebook paths: K % 16 (K = 1152, 272, 320) equals the same codes decoded to 8i rowwise or dense weights")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_SWIGLU_FORWARD, CCV_NNC_BACKEND_MPS) && ccv_nnc_cmd_ok(CCV_NNC_SEGMENTED_GEMM_FORWARD, CCV_NNC_BACKEND_MPS));
	register_tables();
	static const int shapes[][2] = { { 576, 1152 }, { 48, 272 }, { 128, 320 } };
	static const int tables[] = { 0, TABLE_RANDOM };
	static const char* const table_names[] = { "built-in", "random" };
	int sh, p, d, ri, t;
	for (sh = 0; sh < 3; sh++)
		for (p = PATH_SEGMENTED_GEMM; p <= PATH_SWIGLU; p++)
			for (d = 0; d < sizeof(datatypes) / sizeof(datatypes[0]); d++)
				for (ri = 0; ri < sizeof(routings) / sizeof(routings[0]); ri++)
				{
					if (!path_takes((path_t)p, datatypes[d], ri))
						continue;
					const int n = shapes[sh][0], k = shapes[sh][1];
					const uint32_t seed = 0x16000u * (sh + 1) + 0x1000u * (p + 1) + 0x10u * ri + datatypes[d];
					const int experts = p == PATH_SWIGLU ? 0 : 8;
					weights_t weights[4];
					weights[0] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatypes[d], experts, n, k, seed + 1);
					weights[1] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N18, datatypes[d], experts, n, k, seed + 2);
					weights[2] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatypes[d], 0, n, k, seed + 3);
					weights[3] = weights_new(CCV_NNC_QX_8I_ROWWISE_TMCQ_N22, datatypes[d], 0, n, k, seed + 4);
					for (t = 0; t < 2; t++)
					{
						const int slot = t ? table_slots[tables[t]] : 0;
						uint8_t* external;
						size_t external_size;
						REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, 0, &external, &external_size), CCV_NNC_EXEC_SUCCESS,
							"%s %dx%d with the %s table should run: %s datatype=%d", path_names[p], n, k, table_names[t], routings[ri].name, datatypes[d]);
						uint8_t* expected;
						size_t expected_size;
						int identical = 0;
						double resident_max_rel = INFINITY;
						if (p == PATH_SEGMENTED_SWIGLU_SHARED)
						{
							// The routed and shared parts run apart, on the same TMCQ weights.
							REQUIRE_EQ(run_shared_parts(&routings[ri], datatypes[d], weights, slot, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
								"%s %dx%d parts with the %s table should run: %s datatype=%d", path_names[p], n, k, table_names[t], routings[ri].name, datatypes[d]);
							identical = external_size == expected_size && memcmp(external, expected, expected_size) == 0;
							resident_max_rel = max_rel_difference(datatypes[d], external, expected, expected_size);
							ccfree(expected);
						} else {
							int reference;
							const int last = p == PATH_SWIGLU && routings[ri].rows <= 6 ? 4 : 2;
							for (reference = 1; reference <= last && !identical; reference++)
							{
								if (reference == 3)
									continue;
								if (reference == 4)
								{
									REQUIRE_EQ(run_swiglu_matrix_reference(&routings[ri], datatypes[d], weights, slot, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
										"%s %dx%d on 8i rowwise over 7 rows should run: %s datatype=%d", path_names[p], n, k, routings[ri].name, datatypes[d]);
								} else {
									REQUIRE_EQ(run_path((path_t)p, &routings[ri], datatypes[d], weights, slot, reference, &expected, &expected_size), CCV_NNC_EXEC_SUCCESS,
										"%s %dx%d on the reference decode %d should run: %s datatype=%d", path_names[p], n, k, reference, routings[ri].name, datatypes[d]);
								}
								identical = external_size == expected_size && memcmp(external, expected, expected_size) == 0;
								if (reference == 1)
									resident_max_rel = max_rel_difference(datatypes[d], external, expected, expected_size);
								ccfree(expected);
							}
						}
						printf("[tmcq k16 paths] %s | %dx%d | %s | %s | %s table | %s (max rel %g)\n", path_names[p], n, k, routings[ri].name, datatypes[d] == CCV_32F ? "32F" : "16F",
							table_names[t], identical ? "bit-identical" : "within tolerance?", resident_max_rel);
						REQUIRE(identical || resident_max_rel < path_tolerance(datatypes[d]), "%s %dx%d with the %s table should match its decoded weights: %s datatype=%d max rel %g",
							path_names[p], n, k, table_names[t], routings[ri].name, datatypes[d], resident_max_rel);
						ccfree(external);
					}
					int i;
					for (i = 0; i < 4; i++)
						weights_free(&weights[i]);
				}
}

#include "case_main.h"
