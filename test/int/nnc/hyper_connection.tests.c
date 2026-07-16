#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static void _hyper_connection_fill(float* const values, const int count, const int offset)
{
	int i;
	for (i = 0; i < count; i++)
		values[i] = (float)(((i * 17 + offset * 13) % 53) - 26) / 32.0f;
}

TEST_CASE("hyper connection forward modes match scalar relationships")
{
	const int rows = 5;
	const int hc = 3;
	const int hidden = 7;
	const int mix_dim = 2 * hc + hc * hc;
	const ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 3, 1e-6);
	ccv_nnc_tensor_param_t split_inputs[] = {
		CPU_TENSOR_NHWC(32F, rows, mix_dim),
		CPU_TENSOR_NHWC(32F, 3),
		CPU_TENSOR_NHWC(32F, mix_dim),
	};
	ccv_nnc_tensor_param_t split_outputs[3];
	ccv_nnc_hint_tensor_auto(cmd, split_inputs, 3, ccv_nnc_no_hint, split_outputs, 3);
	REQUIRE_EQ(ccv_nnc_tensor_count(split_outputs[0]), rows * hc, "split pre shape should contain one value per row and stream");
	REQUIRE_EQ(ccv_nnc_tensor_count(split_outputs[1]), rows * hc, "split post shape should contain one value per row and stream");
	REQUIRE_EQ(ccv_nnc_tensor_count(split_outputs[2]), rows * hc * hc, "split combination shape should contain one matrix per row");
	ccv_nnc_tensor_param_t weighted_inputs[] = {
		split_inputs[0], split_inputs[1], split_inputs[2], CPU_TENSOR_NHWC(32F, rows, hc, hidden),
	};
	ccv_nnc_tensor_param_t weighted_outputs[3];
	ccv_nnc_hint_tensor_auto(cmd, weighted_inputs, 4, ccv_nnc_no_hint, weighted_outputs, 3);
	REQUIRE_EQ(ccv_nnc_tensor_count(weighted_outputs[0]), rows * hc, "weighted split post shape should contain one value per row and stream");
	REQUIRE_EQ(ccv_nnc_tensor_count(weighted_outputs[1]), rows * hc * hc, "weighted split combination shape should contain one matrix per row");
	REQUIRE_EQ(ccv_nnc_tensor_count(weighted_outputs[2]), rows * hidden, "weighted residual shape should remove the stream dimension");
	ccv_nnc_tensor_param_t expand_inputs[] = {
		CPU_TENSOR_NHWC(32F, rows, hidden), weighted_inputs[3], weighted_outputs[0], weighted_outputs[1],
	};
	ccv_nnc_tensor_param_t expand_output;
	ccv_nnc_hint_tensor_auto(cmd, expand_inputs, 4, ccv_nnc_no_hint, &expand_output, 1);
	REQUIRE_EQ(ccv_nnc_tensor_count(expand_output), rows * hc * hidden, "expanded shape should match the residual shape");

	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, split_inputs[0], 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, split_inputs[1], 0);
	ccv_nnc_tensor_t* const base = ccv_nnc_tensor_new(0, split_inputs[2], 0);
	ccv_nnc_tensor_t* const residual = ccv_nnc_tensor_new(0, weighted_inputs[3], 0);
	ccv_nnc_tensor_t* const block = ccv_nnc_tensor_new(0, expand_inputs[0], 0);
	ccv_nnc_tensor_t* const pre = ccv_nnc_tensor_new(0, split_outputs[0], 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, split_outputs[1], 0);
	ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, split_outputs[2], 0);
	ccv_nnc_tensor_t* const weighted_post = ccv_nnc_tensor_new(0, weighted_outputs[0], 0);
	ccv_nnc_tensor_t* const weighted_combination = ccv_nnc_tensor_new(0, weighted_outputs[1], 0);
	ccv_nnc_tensor_t* const weighted = ccv_nnc_tensor_new(0, weighted_outputs[2], 0);
	ccv_nnc_tensor_t* const expanded = ccv_nnc_tensor_new(0, expand_output, 0);
	_hyper_connection_fill(mix->data.f32, rows * mix_dim, 1);
	_hyper_connection_fill(scale->data.f32, 3, 2);
	_hyper_connection_fill(base->data.f32, mix_dim, 3);
	_hyper_connection_fill(residual->data.f32, rows * hc * hidden, 4);
	_hyper_connection_fill(block->data.f32, rows * hidden, 5);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base), TENSOR_LIST(pre, post, combination), 0), CCV_NNC_EXEC_SUCCESS, "split should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base, residual), TENSOR_LIST(weighted_post, weighted_combination, weighted), 0), CCV_NNC_EXEC_SUCCESS, "weighted split should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, post->data.f32, weighted_post->data.f32, rows * hc, 1e-6, "split modes should produce identical post weights");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, combination->data.f32, weighted_combination->data.f32, rows * hc * hc, 1e-6, "split modes should produce identical combination weights");
	int row, d;
	for (row = 0; row < rows; row++)
		for (d = 0; d < hidden; d++)
		{
			float expected = 0;
			int i;
			for (i = 0; i < hc; i++)
				expected += residual->data.f32[(row * hc + i) * hidden + d] * pre->data.f32[row * hc + i];
			REQUIRE_EQ_WITH_TOLERANCE(weighted->data.f32[row * hidden + d], expected, 1e-6, "weighted split should reduce the residual stream dimension");
		}
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(block, residual, weighted_post, weighted_combination), TENSOR_LIST(expanded), 0), CCV_NNC_EXEC_SUCCESS, "expand should execute");
	for (row = 0; row < rows; row++)
		for (d = 0; d < hidden; d++)
		{
			int i;
			for (i = 0; i < hc; i++)
			{
				float expected = block->data.f32[row * hidden + d] * weighted_post->data.f32[row * hc + i];
				int j;
				for (j = 0; j < hc; j++)
					expected += weighted_combination->data.f32[(row * hc + j) * hc + i] * residual->data.f32[(row * hc + j) * hidden + d];
				REQUIRE_EQ_WITH_TOLERANCE(expanded->data.f32[(row * hc + i) * hidden + d], expected, 1e-6, "expand should combine block and residual streams");
			}
		}
	ccv_nnc_tensor_free(expanded);
	ccv_nnc_tensor_free(weighted);
	ccv_nnc_tensor_free(weighted_combination);
	ccv_nnc_tensor_free(weighted_post);
	ccv_nnc_tensor_free(combination);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(pre);
	ccv_nnc_tensor_free(block);
	ccv_nnc_tensor_free(residual);
	ccv_nnc_tensor_free(base);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(mix);
}

TEST_CASE("hyper connection rejects incompatible tensor shapes")
{
	const int hc = 4;
	const int mix_dim = 2 * hc + hc * hc;
	const ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, mix_dim * 2), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const base = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim), 0);
	ccv_nnc_tensor_t* const pre = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, hc), 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, hc), 0);
	ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, hc * hc), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base), TENSOR_LIST(pre, post, combination), 0), CCV_NNC_EXEC_INVALID, "split should reject a mix tensor whose last dimension is not the control width");
	ccv_nnc_tensor_free(combination);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(pre);
	ccv_nnc_tensor_free(base);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(mix);

	ccv_nnc_tensor_t* const valid_mix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, mix_dim), 0);
	ccv_nnc_tensor_t* const valid_scale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const valid_base = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim), 0);
	ccv_nnc_tensor_t* const transposed_residual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2, hc, 5), 0);
	ccv_nnc_tensor_t* const valid_post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, hc), 0);
	ccv_nnc_tensor_t* const valid_combination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 3, hc * hc), 0);
	ccv_nnc_tensor_t* const transposed_weighted = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3, 2, 5), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(valid_mix, valid_scale, valid_base, transposed_residual), TENSOR_LIST(valid_post, valid_combination, transposed_weighted), 0), CCV_NNC_EXEC_INVALID, "weighted split should reject residual leading dimensions that do not match mix");
	ccv_nnc_tensor_free(transposed_weighted);
	ccv_nnc_tensor_free(valid_combination);
	ccv_nnc_tensor_free(valid_post);
	ccv_nnc_tensor_free(transposed_residual);
	ccv_nnc_tensor_free(valid_base);
	ccv_nnc_tensor_free(valid_scale);
	ccv_nnc_tensor_free(valid_mix);
}

TEST_CASE("hyper connection Metal implementation matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 7;
	const int hc = 4;
	const int hidden = 4096;
	const int mix_dim = 2 * hc + hc * hc;
	const ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	ccv_nnc_tensor_t* const hmix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, mix_dim), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const hbase = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim), 0);
	ccv_nnc_tensor_t* const hresidual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const hblock = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	_hyper_connection_fill(hmix->data.f32, rows * mix_dim, 6);
	_hyper_connection_fill(hscale->data.f32, 3, 7);
	_hyper_connection_fill(hbase->data.f32, mix_dim, 8);
	_hyper_connection_fill(hresidual->data.f32, rows * hc * hidden, 9);
	_hyper_connection_fill(hblock->data.f32, rows * hidden, 10);
	ccv_nnc_tensor_t* const expected_pre = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_combination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const expected_weighted_post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_weighted_combination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const expected_weighted = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const expected_expanded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hmix, hscale, hbase), TENSOR_LIST(expected_pre, expected_post, expected_combination), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hmix, hscale, hbase, hresidual), TENSOR_LIST(expected_weighted_post, expected_weighted_combination, expected_weighted), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hblock, hresidual, expected_weighted_post, expected_weighted_combination), TENSOR_LIST(expected_expanded), 0);

	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, mix_dim), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_tensor_t* const base = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, mix_dim), 0);
	ccv_nnc_tensor_t* const residual = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const block = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const pre = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
	ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const weighted_post = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
	ccv_nnc_tensor_t* const weighted_combination = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const weighted = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const expanded = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const hpre = ccv_nnc_tensor_new(0, expected_pre->info, 0);
	ccv_nnc_tensor_t* const hpost = ccv_nnc_tensor_new(0, expected_post->info, 0);
	ccv_nnc_tensor_t* const hcombination = ccv_nnc_tensor_new(0, expected_combination->info, 0);
	ccv_nnc_tensor_t* const hweighted_post = ccv_nnc_tensor_new(0, expected_weighted_post->info, 0);
	ccv_nnc_tensor_t* const hweighted_combination = ccv_nnc_tensor_new(0, expected_weighted_combination->info, 0);
	ccv_nnc_tensor_t* const hweighted = ccv_nnc_tensor_new(0, expected_weighted->info, 0);
	ccv_nnc_tensor_t* const hexpanded = ccv_nnc_tensor_new(0, expected_expanded->info, 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hmix, hscale, hbase, hresidual, hblock), TENSOR_LIST(mix, scale, base, residual, block), 0);
	ccv_nnc_cmd_t mps_cmd = cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base), TENSOR_LIST(pre, post, combination), 0), CCV_NNC_EXEC_SUCCESS, "Metal split should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base, residual), TENSOR_LIST(weighted_post, weighted_combination, weighted), 0), CCV_NNC_EXEC_SUCCESS, "Metal weighted split should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(block, residual, weighted_post, weighted_combination), TENSOR_LIST(expanded), 0), CCV_NNC_EXEC_SUCCESS, "Metal expand should execute");
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(pre, post, combination, weighted_post, weighted_combination, weighted, expanded), TENSOR_LIST(hpre, hpost, hcombination, hweighted_post, hweighted_combination, hweighted, hexpanded), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_pre->data.f32, hpre->data.f32, rows * hc, 1e-5, "Metal split pre weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_post->data.f32, hpost->data.f32, rows * hc, 1e-5, "Metal split post weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_combination->data.f32, hcombination->data.f32, rows * hc * hc, 1e-5, "Metal split combination weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_weighted_post->data.f32, hweighted_post->data.f32, rows * hc, 1e-5, "Metal weighted split post weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_weighted_combination->data.f32, hweighted_combination->data.f32, rows * hc * hc, 1e-5, "Metal weighted split combination weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_weighted->data.f32, hweighted->data.f32, rows * hidden, 1e-5, "Metal weighted residual should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_expanded->data.f32, hexpanded->data.f32, rows * hc * hidden, 2e-5, "Metal expansion should match CPU reference");
	ccv_nnc_tensor_free(hexpanded);
	ccv_nnc_tensor_free(hweighted);
	ccv_nnc_tensor_free(hweighted_combination);
	ccv_nnc_tensor_free(hweighted_post);
	ccv_nnc_tensor_free(hcombination);
	ccv_nnc_tensor_free(hpost);
	ccv_nnc_tensor_free(hpre);
	ccv_nnc_tensor_free(expanded);
	ccv_nnc_tensor_free(weighted);
	ccv_nnc_tensor_free(weighted_combination);
	ccv_nnc_tensor_free(weighted_post);
	ccv_nnc_tensor_free(combination);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(pre);
	ccv_nnc_tensor_free(block);
	ccv_nnc_tensor_free(residual);
	ccv_nnc_tensor_free(base);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(mix);
	ccv_nnc_tensor_free(expected_expanded);
	ccv_nnc_tensor_free(expected_weighted);
	ccv_nnc_tensor_free(expected_weighted_combination);
	ccv_nnc_tensor_free(expected_weighted_post);
	ccv_nnc_tensor_free(expected_combination);
	ccv_nnc_tensor_free(expected_post);
	ccv_nnc_tensor_free(expected_pre);
	ccv_nnc_tensor_free(hblock);
	ccv_nnc_tensor_free(hresidual);
	ccv_nnc_tensor_free(hbase);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(hmix);
}

TEST_CASE("hyper connection Metal scalar-count fallback matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 9;
	const int hc = 3;
	const int mix_dim = 2 * hc + hc * hc;
	const ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	ccv_nnc_tensor_t* const hmix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, mix_dim), 0);
	ccv_nnc_tensor_t* const hscale = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	ccv_nnc_tensor_t* const hbase = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim), 0);
	ccv_nnc_tensor_t* const expected_pre = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_combination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hc), 0);
	_hyper_connection_fill(hmix->data.f32, rows * mix_dim, 11);
	_hyper_connection_fill(hscale->data.f32, 3, 12);
	_hyper_connection_fill(hbase->data.f32, mix_dim, 13);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hmix, hscale, hbase), TENSOR_LIST(expected_pre, expected_post, expected_combination), 0), CCV_NNC_EXEC_SUCCESS, "CPU scalar-count split should execute");
	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, mix_dim), 0);
	ccv_nnc_tensor_t* const scale = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 3), 0);
	ccv_nnc_tensor_t* const base = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, mix_dim), 0);
	ccv_nnc_tensor_t* const pre = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
	ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const hpre = ccv_nnc_tensor_new(0, expected_pre->info, 0);
	ccv_nnc_tensor_t* const hpost = ccv_nnc_tensor_new(0, expected_post->info, 0);
	ccv_nnc_tensor_t* const hcombination = ccv_nnc_tensor_new(0, expected_combination->info, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hmix, hscale, hbase), TENSOR_LIST(mix, scale, base), 0), CCV_NNC_EXEC_SUCCESS, "inputs should transfer to Metal");
	ccv_nnc_cmd_t mps_cmd = cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(mix, scale, base), TENSOR_LIST(pre, post, combination), 0), CCV_NNC_EXEC_SUCCESS, "Metal scalar-count split should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(pre, post, combination), TENSOR_LIST(hpre, hpost, hcombination), 0), CCV_NNC_EXEC_SUCCESS, "outputs should transfer from Metal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_pre->data.f32, hpre->data.f32, rows * hc, 1e-5, "Metal fallback pre weights should match CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_post->data.f32, hpost->data.f32, rows * hc, 1e-5, "Metal fallback post weights should match CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_combination->data.f32, hcombination->data.f32, rows * hc * hc, 1e-5, "Metal fallback combination weights should match CPU");
	ccv_nnc_tensor_free(hcombination);
	ccv_nnc_tensor_free(hpost);
	ccv_nnc_tensor_free(hpre);
	ccv_nnc_tensor_free(combination);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(pre);
	ccv_nnc_tensor_free(base);
	ccv_nnc_tensor_free(scale);
	ccv_nnc_tensor_free(mix);
	ccv_nnc_tensor_free(expected_combination);
	ccv_nnc_tensor_free(expected_post);
	ccv_nnc_tensor_free(expected_pre);
	ccv_nnc_tensor_free(hbase);
	ccv_nnc_tensor_free(hscale);
	ccv_nnc_tensor_free(hmix);
}

#include "case_main.h"
