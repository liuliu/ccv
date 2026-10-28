#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <3rdparty/dsfmt/dSFMT.h>
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

TEST_CASE("hyper connection Metal row-one implementation matches CPU reference")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 1;
	const int hc = 4;
	const int hidden = 4097;
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

TEST_CASE("hyper connection Metal expansion accepts mixed FP16 block and FP32 state")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int rows = 1;
	const int hc = 4;
	const int hidden = 4097;
	const ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	ccv_nnc_tensor_param_t input_params[] = {
		GPU_TENSOR_NHWC(000, 16F, rows, hidden),
		GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden),
		GPU_TENSOR_NHWC(000, 32F, rows, hc),
		GPU_TENSOR_NHWC(000, 32F, rows, hc, hc),
	};
	ccv_nnc_tensor_param_t output_param;
	ccv_nnc_hint_tensor_auto(cmd, input_params, 4, ccv_nnc_no_hint, &output_param, 1);
	REQUIRE_EQ(output_param.datatype, CCV_32F, "mixed expansion should inherit the residual datatype");
	REQUIRE_EQ(output_param.format, input_params[1].format, "mixed expansion should inherit the residual format");
	REQUIRE_EQ(output_param.type, input_params[1].type, "mixed expansion should inherit the residual memory type");
	REQUIRE_EQ(ccv_nnc_tensor_count(output_param), rows * hc * hidden, "mixed expansion should inherit the residual shape");

	ccv_nnc_tensor_t* const hblock_source = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const hblock = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, hidden), 0);
	ccv_nnc_tensor_t* const hblock_rounded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const hresidual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const hpost = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const hcombination = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hc), 0);
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	_hyper_connection_fill(hblock_source->data.f32, rows * hidden, 14);
	_hyper_connection_fill(hresidual->data.f32, rows * hc * hidden, 15);
	_hyper_connection_fill(hpost->data.f32, rows * hc, 16);
	_hyper_connection_fill(hcombination->data.f32, rows * hc * hc, 17);
	ccv_float_to_half_precision(hblock_source->data.f32, (uint16_t*)hblock->data.f16, rows * hidden);
	ccv_half_precision_to_float((uint16_t*)hblock->data.f16, hblock_rounded->data.f32, rows * hidden);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hblock_rounded, hresidual, hpost, hcombination), TENSOR_LIST(expected), 0), CCV_NNC_EXEC_SUCCESS, "CPU expansion should produce the mixed-precision reference");

	ccv_nnc_tensor_t* const block = ccv_nnc_tensor_new(0, input_params[0], 0);
	ccv_nnc_tensor_t* const residual = ccv_nnc_tensor_new(0, input_params[1], 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, input_params[2], 0);
	ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, input_params[3], 0);
	ccv_nnc_tensor_t* const expanded = ccv_nnc_tensor_new(0, output_param, 0);
	ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, expected->info, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hblock, hresidual, hpost, hcombination), TENSOR_LIST(block, residual, post, combination), 0), CCV_NNC_EXEC_SUCCESS, "mixed inputs should transfer to Metal");
	ccv_nnc_cmd_t mps_cmd = cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(block, residual, post, combination), TENSOR_LIST(expanded), 0), CCV_NNC_EXEC_SUCCESS, "Metal mixed-precision expansion should execute");
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expanded), TENSOR_LIST(actual), 0), CCV_NNC_EXEC_SUCCESS, "mixed expansion should transfer from Metal");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, rows * hc * hidden, 2e-5, "Metal mixed-precision expansion should match the CPU reference");
	ccv_nnc_tensor_free(actual);
	ccv_nnc_tensor_free(expanded);
	ccv_nnc_tensor_free(combination);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(residual);
	ccv_nnc_tensor_free(block);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(hcombination);
	ccv_nnc_tensor_free(hpost);
	ccv_nnc_tensor_free(hresidual);
	ccv_nnc_tensor_free(hblock_rounded);
	ccv_nnc_tensor_free(hblock);
	ccv_nnc_tensor_free(hblock_source);
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

TEST_CASE("hyper connection Metal expansion handles runtime rows and partial tiles")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
	const int shapes[][3] = {{1, 4097, 4}, {7, 5120, 4}, {1, 5120, 4}, {1, 513, 3}, {1, 1025, 8}, {1, 257, 16}};
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 914);
	for (int shape = 0; shape < 6; ++shape)
		for (int half = 0; half < 2; ++half)
		{
			const int rows = shapes[shape][0];
			const int hidden = shapes[shape][1];
			const int hc = shapes[shape][2];
			ccv_nnc_tensor_t* const hb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
			ccv_nnc_tensor_t* const hr = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
			ccv_nnc_tensor_t* const hp = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
			ccv_nnc_tensor_t* const hcmt = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hc), 0);
			ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, hr->info, 0);
			ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, hr->info, 0);
			ccv_nnc_tensor_param_t block_info = hb->info;
			block_info.datatype = half ? CCV_16F : CCV_32F;
			ccv_nnc_tensor_t* const input = ccv_nnc_tensor_new(0, block_info, 0);
			block_info.type = CCV_TENSOR_GPU_MEMORY;
			ccv_nnc_tensor_t* const block = ccv_nnc_tensor_new(0, block_info, 0);
			ccv_nnc_tensor_t* const residual = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
			ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
			ccv_nnc_tensor_t* const combination = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hc), 0);
			ccv_nnc_tensor_t* const output = ccv_nnc_tensor_new(0, residual->info, 0);
			ccv_nnc_tensor_t* const inputs[] = {hb, hr, hp, hcmt};
			for (int tensor = 0; tensor < 4; ++tensor)
				for (int i = 0; i < ccv_nnc_tensor_count(inputs[tensor]->info); ++i)
					inputs[tensor]->data.f32[i] = dsfmt_genrand_open_close(&dsfmt) * 2 - 1;
			const ccv_nnc_cmd_t conversion = half ? CMD_DATATYPE_CONVERSION_FORWARD() : CMD_DATA_TRANSFER_FORWARD();
			ccv_nnc_cmd_exec(conversion, ccv_nnc_no_hint, 0, TENSOR_LIST(hb), TENSOR_LIST(input), 0);
			ccv_nnc_cmd_exec(conversion, ccv_nnc_no_hint, 0, TENSOR_LIST(input), TENSOR_LIST(hb), 0);
			ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
			ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hb, hr, hp, hcmt), TENSOR_LIST(expected), 0);
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(input, hr, hp, hcmt), TENSOR_LIST(block, residual, post, combination), 0);
			cmd.backend = CCV_NNC_BACKEND_MPS;
			REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(block, residual, post, combination), TENSOR_LIST(output), 0), CCV_NNC_EXEC_SUCCESS, "runtime expansion should execute");
			ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(output), TENSOR_LIST(actual), 0);
			REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected->data.f32, actual->data.f32, rows * hc * hidden, 2e-5, "tiled expansion should match CPU reference for every row and tail element");
			ccv_nnc_tensor_free(hb);
			ccv_nnc_tensor_free(hr);
			ccv_nnc_tensor_free(hp);
			ccv_nnc_tensor_free(hcmt);
			ccv_nnc_tensor_free(expected);
			ccv_nnc_tensor_free(actual);
			ccv_nnc_tensor_free(input);
			ccv_nnc_tensor_free(block);
			ccv_nnc_tensor_free(residual);
			ccv_nnc_tensor_free(post);
			ccv_nnc_tensor_free(combination);
			ccv_nnc_tensor_free(output);
		}
	if (!(old_flags & CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
}

// Mix fills: projection rows are small so the gates stay in their sensitive range.
static void _hyper_connection_mix_inputs(const int rows, const int hc, const int hidden, ccv_nnc_tensor_t** const tensors)
{
	const int mix_dim = 2 * hc + hc * hc;
	tensors[0] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	tensors[1] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	tensors[2] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim, hc * hidden), 0);
	tensors[3] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 3), 0);
	tensors[4] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, mix_dim), 0);
	tensors[5] = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, hidden), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, rows * 31 + hidden);
	int i;
	for (i = 0; i < rows * hc * hidden; i++)
		tensors[0]->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) * 4 - 2);
	for (i = 0; i < rows * hc; i++)
		tensors[1]->data.f32[i] = (float)dsfmt_genrand_open_close(&dsfmt);
	for (i = 0; i < mix_dim * hc * hidden; i++)
		tensors[2]->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) / 16;
	for (i = 0; i < 3; i++)
		tensors[3]->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) + 0.5);
	for (i = 0; i < mix_dim; i++)
		tensors[4]->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5);
	for (i = 0; i < hidden; i++)
		tensors[5]->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) + 0.5);
}

TEST_CASE("hyper connection mix matches RMSNorm, projection, split, collapse and RMSNorm")
{
	const int rows = 3;
	const int hc = 4;
	const int hidden = 64;
	const int mix_dim = 2 * hc + hc * hc;
	const float norm_epsilon = 1e-6;
	ccv_nnc_tensor_t* inputs[6];
	_hyper_connection_mix_inputs(rows, hc, hidden, inputs);
	ccv_nnc_cmd_t mix_cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	mix_cmd.info.hyper_connection.norm_epsilon = norm_epsilon;
	ccv_nnc_tensor_t* const pre = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const comb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hc), 0);
	ccv_nnc_tensor_t* const collapsed = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(mix_cmd, ccv_nnc_no_hint, 0, inputs, 6, TENSOR_LIST(pre, post, comb, collapsed), 0), CCV_NNC_EXEC_SUCCESS, "CPU mix should execute");
	// The same computation from the existing commands.
	ccv_nnc_tensor_t flat = ccv_nnc_tensor(inputs[0]->data.f32, CPU_TENSOR_NHWC(32F, rows, hc * hidden), 0);
	ccv_nnc_tensor_t* const normalized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hidden), 0);
	ccv_nnc_tensor_t* const rstd = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, 1), 0);
	ccv_nnc_cmd_exec(CMD_RMSNORM_FORWARD(norm_epsilon, 0, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(&flat), TENSOR_LIST(normalized, rstd), 0);
	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, mix_dim), 0);
	ccv_nnc_cmd_exec(CMD_GEMM_FORWARD(NO_TRANSPOSE, TRANSPOSE(0, 1)), ccv_nnc_no_hint, 0, TENSOR_LIST(normalized, inputs[2]), TENSOR_LIST(mix), 0);
	ccv_nnc_tensor_t* const expected_pre = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_post = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const expected_comb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hc), 0);
	ccv_nnc_cmd_exec(CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6), ccv_nnc_no_hint, 0, TENSOR_LIST(mix, inputs[3], inputs[4]), TENSOR_LIST(expected_pre, expected_post, expected_comb), 0);
	ccv_nnc_tensor_t* const sum = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	int i, j, k;
	for (i = 0; i < rows; i++)
		for (j = 0; j < hidden; j++)
		{
			float value = 0;
			for (k = 0; k < hc; k++)
				value += inputs[0]->data.f32[(i * hc + k) * hidden + j] * inputs[1]->data.f32[i * hc + k];
			sum->data.f32[i * hidden + j] = value;
		}
	ccv_nnc_tensor_t* const expected_collapsed = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_cmd_exec(CMD_RMSNORM_FORWARD(norm_epsilon, 1, 1), ccv_nnc_no_hint, 0, TENSOR_LIST(sum, inputs[5]), TENSOR_LIST(expected_collapsed, rstd), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, pre->data.f32, expected_pre->data.f32, rows * hc, 1e-5, "mix pre weights should match the composed commands");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, post->data.f32, expected_post->data.f32, rows * hc, 1e-5, "mix post weights should match the composed commands");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, comb->data.f32, expected_comb->data.f32, rows * hc * hc, 1e-5, "mix combination should match the composed commands");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, collapsed->data.f32, expected_collapsed->data.f32, rows * hidden, 1e-5, "mix collapse should match the composed commands");
	ccv_nnc_tensor_free(expected_collapsed);
	ccv_nnc_tensor_free(sum);
	ccv_nnc_tensor_free(expected_comb);
	ccv_nnc_tensor_free(expected_post);
	ccv_nnc_tensor_free(expected_pre);
	ccv_nnc_tensor_free(mix);
	ccv_nnc_tensor_free(rstd);
	ccv_nnc_tensor_free(normalized);
	ccv_nnc_tensor_free(collapsed);
	ccv_nnc_tensor_free(comb);
	ccv_nnc_tensor_free(post);
	ccv_nnc_tensor_free(pre);
	for (i = 0; i < 6; i++)
		ccv_nnc_tensor_free(inputs[i]);
}

static void _hyper_connection_mix_mps_case(const int rows, const int hidden, int* const __case_result__)
{
	const int hc = 4;
	ccv_nnc_tensor_t* hinputs[6];
	_hyper_connection_mix_inputs(rows, hc, hidden, hinputs);
	ccv_nnc_cmd_t cmd = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	cmd.info.hyper_connection.norm_epsilon = 1e-20;
	ccv_nnc_tensor_t* const expected[4] = {
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0),
	};
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, hinputs, 6, (ccv_nnc_tensor_t**)expected, 4, 0), CCV_NNC_EXEC_SUCCESS, "CPU mix should execute");
	ccv_nnc_tensor_t* inputs[6];
	int i;
	for (i = 0; i < 6; i++)
	{
		ccv_nnc_tensor_param_t params = hinputs[i]->info;
		params.type = CCV_TENSOR_GPU_MEMORY;
		inputs[i] = ccv_nnc_tensor_new(0, params, 0);
	}
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, hinputs, 6, inputs, 6, 0);
	ccv_nnc_tensor_t* const outputs[4] = {
		ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc * hc), 0),
		ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0),
	};
	ccv_nnc_cmd_t mps_cmd = cmd;
	mps_cmd.backend = CCV_NNC_BACKEND_MPS;
	REQUIRE_EQ(ccv_nnc_cmd_exec(mps_cmd, ccv_nnc_no_hint, 0, inputs, 6, (ccv_nnc_tensor_t**)outputs, 4, 0), CCV_NNC_EXEC_SUCCESS, "Metal mix should execute");
	ccv_nnc_tensor_t* const actual[4] = {
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hc), 0),
		ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0),
	};
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, outputs, 4, actual, 4, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, actual[0]->data.f32, expected[0]->data.f32, rows * hc, 1e-5, "Metal mix pre weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, actual[1]->data.f32, expected[1]->data.f32, rows * hc, 1e-5, "Metal mix post weights should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, actual[2]->data.f32, expected[2]->data.f32, rows * hc * hc, 1e-5, "Metal mix combination should match CPU reference");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, actual[3]->data.f32, expected[3]->data.f32, rows * hidden, 1e-5, "Metal mix collapse should match CPU reference");
	for (i = 0; i < 4; i++)
	{
		ccv_nnc_tensor_free(actual[i]);
		ccv_nnc_tensor_free(outputs[i]);
		ccv_nnc_tensor_free(expected[i]);
	}
	for (i = 0; i < 6; i++)
	{
		ccv_nnc_tensor_free(inputs[i]);
		ccv_nnc_tensor_free(hinputs[i]);
	}
}

TEST_CASE("hyper connection Metal mix matches CPU reference for decode rows")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	_hyper_connection_mix_mps_case(1, 5120, __case_result__);
	_hyper_connection_mix_mps_case(2, 1000, __case_result__);
}

TEST_CASE("hyper connection Metal mix matches CPU reference for prefill rows")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	// The projection GEMM runs FP32 at reduced precision on the neural accelerators, so compare
	// with them off.
	const uint64_t old_flags = ccv_nnc_flags();
	ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
	_hyper_connection_mix_mps_case(40, 5120, __case_result__);
	_hyper_connection_mix_mps_case(40, 1000, __case_result__);
	if (!(old_flags & CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS))
		ccv_nnc_disable_flag(CCV_NNC_DISABLE_MFA_NEURAL_ACCELERATORS);
}

static void _hyper_connection_expand_sum_case(const int backend, const int rows, const int hidden, int* const __case_result__)
{
	// Expand-sum must match adding the addend to the block and then expanding, bit for bit.
	const int hc = 4;
	ccv_nnc_tensor_t* const hblock = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const haddend = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
	ccv_nnc_tensor_t* const hresidual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const hpost = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc), 0);
	ccv_nnc_tensor_t* const hcomb = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc * hc), 0);
	int i;
	for (i = 0; i < rows * hidden; i++)
	{
		hblock->data.f32[i] = (float)((i * 7) % 23 - 11) / 3;
		haddend->data.f32[i] = (float)((i * 13) % 29 - 14) / 7;
	}
	for (i = 0; i < rows * hc * hidden; i++)
		hresidual->data.f32[i] = (float)((i * 11) % 31 - 15) / 5;
	for (i = 0; i < rows * hc; i++)
		hpost->data.f32[i] = (float)(i % 5 + 1) / 3;
	for (i = 0; i < rows * hc * hc; i++)
		hcomb->data.f32[i] = (float)((i * 3) % 7 + 1) / 9;
	ccv_nnc_tensor_t* const hexpected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_tensor_t* const hactual = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hc, hidden), 0);
	ccv_nnc_cmd_t expand = CMD_HYPER_CONNECTION_FORWARD(hc, 20, 1e-6);
	expand.backend = backend;
	ccv_nnc_cmd_t add = CMD_ADD_FORWARD(1, 1);
	add.backend = backend == CCV_NNC_BACKEND_MPS ? CCV_NNC_BACKEND_MPS : CCV_NNC_BACKEND_CPU_REF;
	if (backend == CCV_NNC_BACKEND_MPS)
	{
		ccv_nnc_tensor_t* const block = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0);
		ccv_nnc_tensor_t* const addend = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0);
		ccv_nnc_tensor_t* const sum = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hidden), 0);
		ccv_nnc_tensor_t* const residual = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
		ccv_nnc_tensor_t* const post = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc), 0);
		ccv_nnc_tensor_t* const comb = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc * hc), 0);
		ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
		ccv_nnc_tensor_t* const actual = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, hc, hidden), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hblock, haddend, hresidual, hpost, hcomb), TENSOR_LIST(block, addend, residual, post, comb), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(add, ccv_nnc_no_hint, 0, TENSOR_LIST(block, addend), TENSOR_LIST(sum), 0), CCV_NNC_EXEC_SUCCESS, "add should execute");
		REQUIRE_EQ(ccv_nnc_cmd_exec(expand, ccv_nnc_no_hint, 0, TENSOR_LIST(sum, residual, post, comb), TENSOR_LIST(expected), 0), CCV_NNC_EXEC_SUCCESS, "expand should execute");
		REQUIRE_EQ(ccv_nnc_cmd_exec(expand, ccv_nnc_no_hint, 0, TENSOR_LIST(block, residual, post, comb, addend), TENSOR_LIST(actual), 0), CCV_NNC_EXEC_SUCCESS, "expand-sum should execute");
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(expected, actual), TENSOR_LIST(hexpected, hactual), 0);
		ccv_nnc_tensor_free(actual);
		ccv_nnc_tensor_free(expected);
		ccv_nnc_tensor_free(comb);
		ccv_nnc_tensor_free(post);
		ccv_nnc_tensor_free(residual);
		ccv_nnc_tensor_free(sum);
		ccv_nnc_tensor_free(addend);
		ccv_nnc_tensor_free(block);
	} else {
		ccv_nnc_tensor_t* const hsum = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, hidden), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(add, ccv_nnc_no_hint, 0, TENSOR_LIST(hblock, haddend), TENSOR_LIST(hsum), 0), CCV_NNC_EXEC_SUCCESS, "add should execute");
		REQUIRE_EQ(ccv_nnc_cmd_exec(expand, ccv_nnc_no_hint, 0, TENSOR_LIST(hsum, hresidual, hpost, hcomb), TENSOR_LIST(hexpected), 0), CCV_NNC_EXEC_SUCCESS, "expand should execute");
		REQUIRE_EQ(ccv_nnc_cmd_exec(expand, ccv_nnc_no_hint, 0, TENSOR_LIST(hblock, hresidual, hpost, hcomb, haddend), TENSOR_LIST(hactual), 0), CCV_NNC_EXEC_SUCCESS, "expand-sum should execute");
		ccv_nnc_tensor_free(hsum);
	}
	REQUIRE(memcmp(hexpected->data.f32, hactual->data.f32, sizeof(float) * rows * hc * hidden) == 0, "expand-sum should match add then expand");
	ccv_nnc_tensor_free(hactual);
	ccv_nnc_tensor_free(hexpected);
	ccv_nnc_tensor_free(hcomb);
	ccv_nnc_tensor_free(hpost);
	ccv_nnc_tensor_free(hresidual);
	ccv_nnc_tensor_free(haddend);
	ccv_nnc_tensor_free(hblock);
}

TEST_CASE("hyper connection expand-sum matches add then expand")
{
	_hyper_connection_expand_sum_case(CCV_NNC_BACKEND_CPU_REF, 3, 64, __case_result__);
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_HYPER_CONNECTION_FORWARD, CCV_NNC_BACKEND_MPS));
	_hyper_connection_expand_sum_case(CCV_NNC_BACKEND_MPS, 1, 5120, __case_result__);
	_hyper_connection_expand_sum_case(CCV_NNC_BACKEND_MPS, 7, 1000, __case_result__);
}

#include "case_main.h"
