#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <limits.h>

TEST_SETUP() { ccv_nnc_init(); }

// Split row scores into two small integers so every input dtype (and the NA
// FP32 matmul input path) represents them exactly, even at long cache lengths.
static float candidate_fixture_score(const ccv_nnc_tensor_t* const k, const int c, const int t)
{
	return k->data.f32[c * 128 + (t % 2) * 2] * 256 + k->data.f32[c * 128 + (t % 2) * 2 + 1];
}

static int candidate_compare(const int T, const int C, const int H, const int kth, const int block_size, const int pool_size, const int ratio, const int offset, const int datatype, const int malformed, const int format, const int sort_indices)
{
	ccv_nnc_tensor_t* hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H, 128), 0);
	ccv_nnc_tensor_t* hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, C, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H), 0);
	ccv_nnc_tensor_t* reference = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, kth), 0);
	ccv_nnc_tensor_t* reference_pool = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, pool_size), 0);
	ccv_nnc_tensor_t* actual = ccv_nnc_tensor_new(0, reference->info, 0);
	ccv_nnc_tensor_t* actual_pool = ccv_nnc_tensor_new(0, reference_pool->info, 0);
	int t, h, d, c, i;
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 42);
	for (t = 0; t < T; t++)
		for (h = 0; h < H; h++)
		{
			hw->data.f32[t * H + h] = 1;
			for (d = 0; d < 128; d++)
				hq->data.f32[(t * H + h) * 128 + d] = d == (t % 2) * 2 ? 256 : (d == (t % 2) * 2 + 1 ? 1 : 0);
		}
	for (c = 0; c < C; c++)
		for (d = 0; d < 128; d++)
			hk->data.f32[c * 128 + d] = d < 4 && malformed == 2 ? 0 : d < 4 ? (float)(d % 2 == 0 ? (d < 2 ? C - c : c + 1) / 256 : (d < 2 ? C - c : c + 1) % 256) : (float)((int)(dsfmt_genrand_close_open(&dsfmt) * 9) - 4);
	hq->info.format = hk->info.format = hw->info.format = format;
	ccv_nnc_tensor_param_t qp = hq->info, kp = hk->info, wp = hw->info;
	qp.datatype = kp.datatype = wp.datatype = datatype;
	ccv_nnc_tensor_t* hqd = ccv_nnc_tensor_new(0, qp, 0);
	ccv_nnc_tensor_t* hkd = ccv_nnc_tensor_new(0, kp, 0);
	ccv_nnc_tensor_t* hwd = ccv_nnc_tensor_new(0, wp, 0);
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw), TENSOR_LIST(hqd, hkd, hwd), 0);
	// Compare with the values actually stored in the input dtype.
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hqd, hkd, hwd), TENSOR_LIST(hq, hk, hw), 0);
	qp.type = kp.type = wp.type = CCV_TENSOR_GPU_MEMORY;
	ccv_nnc_tensor_t* q = ccv_nnc_tensor_new(0, qp, 0);
	ccv_nnc_tensor_t* k = ccv_nnc_tensor_new(0, kp, 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, wp, 0);
	ccv_nnc_tensor_t* selected = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, T, kth), 0);
	ccv_nnc_tensor_t* pool = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, T, pool_size), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hqd, hkd, hwd), TENSOR_LIST(q, k, w), 0);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(kth, 1, 1, ratio, offset);
	cmd.info.scaled_dot_product_arg_partition.candidate_block_size = block_size;
	cmd.info.scaled_dot_product_arg_partition.candidate_kth = pool_size;
	cmd.info.scaled_dot_product_arg_partition.sort_indices = sort_indices;
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw), TENSOR_LIST(reference, reference_pool), 0);
	int status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w), TENSOR_LIST(selected, pool), 0);
	int error = status != CCV_NNC_EXEC_SUCCESS;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(selected, pool), TENSOR_LIST(actual, actual_pool), 0);
	for (t = 0; t < T && !error; t++)
	{
		const int visible = ccv_min(C, ccv_max(0, (offset + t + 1) / ratio));
		const int blocks = (visible + block_size - 1) / block_size;
		const int valid_blocks = ccv_min(blocks, pool_size);
		const int valid_rows = ccv_min(visible, kth);
		float threshold = 1e30;
		for (i = 0; i < valid_rows; i++)
			threshold = ccv_min(threshold, candidate_fixture_score(hk, reference->data.i32[t * kth + i], t));
		for (i = 0; i < kth; i++)
		{
			const int id = actual->data.i32[t * kth + i];
			if (i < valid_rows ? (id < 0 || id >= visible || ((sort_indices || C <= kth) && i > 0 && id <= actual->data.i32[t * kth + i - 1]) || candidate_fixture_score(hk, id, t) < threshold) : id != -1)
			{ fprintf(stderr, "source row t=%d i=%d id=%d prev=%d ref=%d value=%g threshold=%g visible=%d\n", t, i, id, i ? actual->data.i32[t*kth+i-1] : -1, reference->data.i32[t*kth+i], id>=0&&id<C ? candidate_fixture_score(hk, id, t) : -999, threshold, visible); error = 2; break; }
			if (i < valid_rows)
			{
				if (malformed != 2 && id != reference->data.i32[t * kth + i]) error = 11;
				for (h = 0; h < i; h++)
					if (id == actual->data.i32[t * kth + h]) error = 12;
			}
		}
		float block_threshold = 1e30;
		for (i = 0; i < valid_blocks; i++)
		{
			const int b = reference_pool->data.i32[t * pool_size + i];
			if (b == blocks - 1) continue;
			const int best = t % 2 == 0 ? b * block_size : ccv_min(visible, (b + 1) * block_size) - 1;
			block_threshold = ccv_min(block_threshold, candidate_fixture_score(hk, best, t));
		}
		int newest = blocks == 0;
		for (i = 0; i < pool_size; i++)
		{
			const int b = actual_pool->data.i32[t * pool_size + i];
			if (i >= valid_blocks) { if (b != -1) error = 3; continue; }
			if (b < 0 || b >= blocks || (i > 0 && b <= actual_pool->data.i32[t * pool_size + i - 1])) { error = 4; break; }
			if (b == blocks - 1) newest = 1;
			else {
				const int best = t % 2 == 0 ? b * block_size : ccv_min(visible, (b + 1) * block_size) - 1;
				if (candidate_fixture_score(hk, best, t) < block_threshold) error = 5;
			}
		}
		if (!newest) error = 6;
	}
	// Readers use the actual produced pool, so tied source blocks may differ.
	if (malformed == 1 || malformed == 3)
		for (t = 0; t < T; t++)
			for (i = 0; i < pool_size; i++)
				actual_pool->data.i32[t * pool_size + i] = malformed == 3 ? (i % 2 ? -2 : INT_MAX) : i % 5 == 0 ? INT_MAX : (i % 5 == 1 ? -2 : (i % 5 == 2 ? 0 : (pool_size - i) / 2));
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(actual_pool), TENSOR_LIST(pool), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw, actual_pool), TENSOR_LIST(reference), 0);
	status = ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w, pool), TENSOR_LIST(selected), 0);
	if (status != CCV_NNC_EXEC_SUCCESS) error = 7;
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(selected), TENSOR_LIST(actual), 0);
	for (t = 0; t < T && !error; t++)
	{
		const int visible = ccv_min(C, ccv_max(0, (offset + t + 1) / ratio));
		int count = 0;
		float threshold = 1e30;
		for (i = 0; i < kth && reference->data.i32[t * kth + i] >= 0; i++)
		{
			++count;
			threshold = ccv_min(threshold, candidate_fixture_score(hk, reference->data.i32[t * kth + i], t));
		}
		for (i = 0; i < kth; i++)
		{
			const int id = actual->data.i32[t * kth + i];
			if (i >= count) { if (id != -1) error = 8; continue; }
			if (id < 0 || id >= visible || ((sort_indices || C <= kth) && i > 0 && id <= actual->data.i32[t * kth + i - 1]) || candidate_fixture_score(hk, id, t) < threshold) { error = 9; break; }
			if (malformed != 2 && id != reference->data.i32[t * kth + i]) error = 11;
			for (h = 0; h < i; h++)
				if (id == actual->data.i32[t * kth + h]) error = 12;
			int found = 0;
			for (h = 0; h < pool_size; h++)
				found |= actual_pool->data.i32[t * pool_size + h] == id / block_size;
			if (!found) error = 10;
		}
	}
	ccv_nnc_tensor_t* tensors[] = { hq, hk, hw, hqd, hkd, hwd, q, k, w, reference, reference_pool, actual, actual_pool, selected, pool };
	for (i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) ccv_nnc_tensor_free(tensors[i]);
	if (error) fprintf(stderr, "candidate comparison: T=%d C=%d H=%d kth=%d pool=%d dtype=%d sort=%d flags=%llu error=%d\n", T, C, H, kth, pool_size, datatype, sort_indices, (unsigned long long)ccv_nnc_flags(), error);
	return error;
}

TEST_CASE("candidate indexer produces causal pools and restricts readers across backends")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int sizes[] = { 0, 1, 7, 8, 9, 127, 511, 512, 513, 8192, 16384, 16385, 17017, 513 };
	const uint64_t old_flags = ccv_nnc_flags();
	int mode, i;
	for (mode = 0; mode < 8; mode++)
	{
		if (mode % 4 == 3) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
		if (mode % 4 == 2) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
		if (mode % 4 == 1) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
		{
			const int C = sizes[i];
			REQUIRE_EQ(candidate_compare(17, C, 3, 512, 8, C >= 8192 ? 2048 : 7, 2, C * 2 - 9, CCV_32F, 0, i % 2 ? CCV_TENSOR_FORMAT_NCHW : CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "source and reader should preserve causal rows, block membership, newest block and padding");
		}
		REQUIRE_EQ(candidate_compare(3, 513, 32, 511, 8, 7, 1, 512, CCV_16F, 1, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "FP16 readers should ignore duplicate, negative and out-of-range block ids");
		REQUIRE_EQ(candidate_compare(17, 513, 32, 513, 8, 7, 2, -9, CCV_16BF, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "BF16 should handle partially visible and empty causal rows with full-width top-k");
		REQUIRE_EQ(candidate_compare(3, 4097, 3, 513, 8, 7, 2, 8193, CCV_32F, 1, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "candidate-only readers should ignore malformed ids across a wide top-k merge");
		REQUIRE_EQ(candidate_compare(3, 17017, 3, 512, 8, 2048, 2, 34000, CCV_32F, 2, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "ties should preserve eligibility, uniqueness and mandatory newest-block inclusion");
		REQUIRE_EQ(candidate_compare(17, 513, 3, 7, 1, 1, 2, 1000, CCV_16F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "a singleton pool should retain only its newest block");
		REQUIRE_EQ(candidate_compare(1, 131073, 3, 512, 8, 2048, 2, 262145, CCV_16F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "long-cache decode should enumerate and normalize a full 2048-block pool");
		REQUIRE_EQ(candidate_compare(17, 127, 3, 17, 3, 5, 2, 245, CCV_32F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "block size need not divide the reader's key tile");
		REQUIRE_EQ(candidate_compare(273, 127, 1, 17, 8, 3, 2, 0, CCV_32F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "query tiling must preserve per-query pools and absolute causal offsets");
		REQUIRE_EQ(candidate_compare(3, 512, 32, 512, 8, 2048, 2, 1018, CCV_16F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "full-width positions and pools should enumerate each query's causal ids");
		REQUIRE_EQ(candidate_compare(3, 1024, 3, 1024, 1, 2048, 1, 1021, CCV_32F, 1, CCV_TENSOR_FORMAT_NCHW, mode / 4), 0, "full-width readers should compact malformed pools across all 1024 block bits");
		REQUIRE_EQ(candidate_compare(3, 1022, 3, 1024, 3, 2048, 1, 1020, CCV_16BF, 1, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "full-width readers should clip retained partial blocks and pad compacted ids");
		REQUIRE_EQ(candidate_compare(273, 127, 3, 512, 8, 2048, 2, -9, CCV_16F, 1, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "full-width reader enumeration must preserve per-query pools and causal offsets across large batches");
		REQUIRE_EQ(candidate_compare(3, 511, 3, 512, 8, 2048, 1, 510, CCV_32F, 3, CCV_TENSOR_FORMAT_NHWC, mode / 4), 0, "full-width readers should return only padding for an entirely invalid pool");
	}
	if (old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	if (old_flags & CCV_NNC_DISABLE_MFA) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

TEST_CASE("candidate indexer enumeration handles growing and shrinking output widths")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int sizes[] = { 1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1023, 1024, 17 };
	const uint64_t old_flags = ccv_nnc_flags();
	int mode, i, status = 0;
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	for (mode = 0; mode < 8 && !status; mode++)
	{
		if (mode & 1) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		if (mode & 2) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
		for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]) && !status; i++)
		{
			const int C = sizes[i];
			status = candidate_compare(1, C, 32, C, 8, 2048, 2, C * 2 - 1, CCV_16F, 0, CCV_TENSOR_FORMAT_NHWC, mode / 4);
			if (!status)
				status = candidate_compare(3, C, 3, C + (C < 1024), 1, 2048, 1, C - 3, CCV_32F, 1, CCV_TENSOR_FORMAT_NCHW, mode / 4);
		}
	}
	if (old_flags & CCV_NNC_DISABLE_MFA) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	if (old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	REQUIRE_EQ(status, 0, "source and reader enumeration must use the current runtime width, causal visibility and padding with either sort setting");
}

TEST_CASE("candidate indexer MFA matches random CPU ranks for independent source and reader scores")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int T = 17, C = 4097, H = 33, P = 31;
	ccv_nnc_tensor_t* hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H, 128), 0);
	ccv_nnc_tensor_t* hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, C, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, T, H), 0);
	ccv_nnc_tensor_t* hq16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, T, H, 128), 0);
	ccv_nnc_tensor_t* hk16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, C, 128), 0);
	ccv_nnc_tensor_t* hw16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, T, H), 0);
	ccv_nnc_tensor_t* ref = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, 512), 0);
	ccv_nnc_tensor_t* ref_pool = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, T, P), 0);
	ccv_nnc_tensor_t* actual = ccv_nnc_tensor_new(0, ref->info, 0);
	ccv_nnc_tensor_t* actual_pool = ccv_nnc_tensor_new(0, ref_pool->info, 0);
	ccv_nnc_tensor_t* q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, T, H, 128), 0);
	ccv_nnc_tensor_t* k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, C, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, T, H), 0);
	ccv_nnc_tensor_t* selected = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, T, 512), 0);
	ccv_nnc_tensor_t* pool = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, T, P), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 19);
	int i, mode, reader;
	for (i = 0; i < C * 128; i++) hk->data.f32[i] = (int)(dsfmt_genrand_close_open(&dsfmt) * 9) - 4;
	for (i = 0; i < T * H; i++) hw->data.f32[i] = (int)(dsfmt_genrand_close_open(&dsfmt) * 4) + 1;
	ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hk, hw), TENSOR_LIST(hk16, hw16), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hk16, hw16), TENSOR_LIST(k, w), 0);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(512, 1, 1, 2, 8100);
	cmd.info.scaled_dot_product_arg_partition.candidate_block_size = 8;
	cmd.info.scaled_dot_product_arg_partition.candidate_kth = P;
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	for (mode = 0; mode < 4; mode++)
	{
		if (mode % 2) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
		cmd.info.scaled_dot_product_arg_partition.sort_indices = mode / 2;
		for (reader = 0; reader < 2; reader++)
		{
			for (i = 0; i < T * H * 128; i++) hq->data.f32[i] = (int)(dsfmt_genrand_close_open(&dsfmt) * 9) - 4;
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq), TENSOR_LIST(hq16), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq16), TENSOR_LIST(q), 0);
			if (!reader)
			{
				ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw), TENSOR_LIST(ref, ref_pool), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w), TENSOR_LIST(selected, pool), 0), CCV_NNC_EXEC_SUCCESS, "source should execute");
				ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(pool), TENSOR_LIST(actual_pool), 0);
				REQUIRE_TENSOR_EQ(ref_pool, actual_pool, "random block maxima should select identical pools");
			} else {
				ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw, ref_pool), TENSOR_LIST(ref), 0);
				REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w, pool), TENSOR_LIST(selected), 0), CCV_NNC_EXEC_SUCCESS, "reader should execute with its own scores");
			}
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(selected), TENSOR_LIST(actual), 0);
			REQUIRE_TENSOR_EQ(ref, actual, "candidate-aware MFA should match CPU on random, exactly representable scores");
		}
	}
	if (old_flags & CCV_NNC_DISABLE_MFA) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA);
	if (old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS); else ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	ccv_nnc_tensor_t* tensors[] = { hq, hk, hw, hq16, hk16, hw16, ref, ref_pool, actual, actual_pool, q, k, w, selected, pool };
	for (i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) ccv_nnc_tensor_free(tensors[i]);
}

TEST_CASE("candidate indexer source selection is independent of its mandatory newest block")
{
	ccv_nnc_tensor_t* q = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1, 1), 0);
	ccv_nnc_tensor_t* k = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 17, 1), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 1), 0);
	ccv_nnc_tensor_t* ids = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 3), 0);
	ccv_nnc_tensor_t* pool = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 3, 1), 0);
	int i;
	for (i = 0; i < 3; i++) q->data.f32[i] = w->data.f32[i] = 1;
	for (i = 0; i < 17; i++) k->data.f32[i] = 17 - i;
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(3, 1, 1, 1, 7);
	cmd.info.scaled_dot_product_arg_partition.candidate_block_size = 8;
	cmd.info.scaled_dot_product_arg_partition.candidate_kth = 1;
	cmd.info.scaled_dot_product_arg_partition.sort_indices = 1;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w), TENSOR_LIST(ids, pool), 0), CCV_NNC_EXEC_SUCCESS, "source should execute");
	const int source_ids[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
	const int pools[] = { 0, 1, 1 };
	REQUIRE_ARRAY_EQ(int, ids->data.i32, source_ids, 9, "source rows remain unrestricted even when its pool excludes them");
	REQUIRE_ARRAY_EQ(int, pool->data.i32, pools, 3, "newest partially visible block must displace a higher-scoring older block");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w, pool), TENSOR_LIST(ids), 0), CCV_NNC_EXEC_SUCCESS, "reader should execute");
	const int reader_ids[] = { 0, 1, 2, 8, -1, -1, 8, 9, -1 };
	REQUIRE_ARRAY_EQ(int, ids->data.i32, reader_ids, 9, "reader must preserve original ids and clip each query's partial block");
	ccv_nnc_tensor_free(q); ccv_nnc_tensor_free(k); ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(ids); ccv_nnc_tensor_free(pool);
}

TEST_CASE("candidate indexer can sort row ids without candidate IO")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD, CCV_NNC_BACKEND_MPS));
	ccv_nnc_tensor_t* hq = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 3, 128), 0);
	ccv_nnc_tensor_t* hk = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1025, 128), 0);
	ccv_nnc_tensor_t* hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 1, 3), 0);
	ccv_nnc_tensor_zero(hq); ccv_nnc_tensor_zero(hk);
	int i;
	for (i = 0; i < 3; i++) { hq->data.f32[i * 128] = 1; hw->data.f32[i] = 1; }
	for (i = 0; i < 1025; i++) hk->data.f32[i * 128] = i;
	ccv_nnc_tensor_t* q = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 3, 128), 0);
	ccv_nnc_tensor_t* k = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1025, 128), 0);
	ccv_nnc_tensor_t* w = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 1, 3), 0);
	ccv_nnc_tensor_t* ids = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32S, 1, 512), 0);
	ccv_nnc_tensor_t* actual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32S, 1, 512), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hq, hk, hw), TENSOR_LIST(q, k, w), 0);
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(512, 1, 0, 1, 0);
	cmd.info.scaled_dot_product_arg_partition.sort_indices = 1;
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(q, k, w), TENSOR_LIST(ids), 0), CCV_NNC_EXEC_SUCCESS, "sorting should not require pool parameters");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(ids), TENSOR_LIST(actual), 0);
	for (i = 0; i < 512; i++) REQUIRE_EQ(actual->data.i32[i], i + 513, "selected rows should be ascending");
	ccv_nnc_tensor_t* tensors[] = { hq, hk, hw, q, k, w, ids, actual };
	for (i = 0; i < sizeof(tensors) / sizeof(tensors[0]); i++) ccv_nnc_tensor_free(tensors[i]);
}

TEST_CASE("candidate indexer infers compact pool shapes and validates IO roles")
{
	ccv_nnc_cmd_t cmd = CMD_SCALED_DOT_PRODUCT_ARG_PARTITION_FORWARD(512, 1, 1, 2, 17);
	cmd.info.scaled_dot_product_arg_partition.candidate_block_size = 8;
	cmd.info.scaled_dot_product_arg_partition.candidate_kth = 2048;
	ccv_nnc_tensor_param_t inputs[] = { CPU_TENSOR_NHWC(32F, 17, 32, 128), CPU_TENSOR_NHWC(32F, 17017, 128), CPU_TENSOR_NHWC(32F, 17, 32) };
	ccv_nnc_tensor_param_t outputs[2];
	ccv_nnc_hint_tensor_auto(cmd, inputs, 3, ccv_nnc_no_hint, outputs, 2);
	REQUIRE_EQ(outputs[0].dim[1], 512, "position output should retain kth rows");
	REQUIRE_EQ(outputs[1].dim[1], 2048, "pool output should contain block ids, independent of C");
	REQUIRE_EQ(outputs[1].datatype, CCV_32S, "pool ids should be int32");
	uint64_t input_mask = 7, output_mask = 3;
	REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 3, 2, &input_mask, 1, &output_mask, 1), 1, "source should accept a second output");
	input_mask = 15;
	REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 4, 2, &input_mask, 1, &output_mask, 1), 0, "source and reader roles are mutually exclusive");
	output_mask = 1;
	REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 4, 1, &input_mask, 1, &output_mask, 1), 1, "reader should accept a fourth input");
	cmd.info.scaled_dot_product_arg_partition.candidate_block_size = 0;
	REQUIRE_EQ(ccv_nnc_cmd_bitmask(cmd, 4, 1, &input_mask, 1, &output_mask, 1), 0, "candidate IO requires a block size");
}

#include "case_main.h"
