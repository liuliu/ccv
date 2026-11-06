#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/mps/ccv_nnc_mps.h>
#include <math.h>
#include <string.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _causal_convolution_value(const int i, const int multiplier, const int modulus, const float scale)
{
	return (float)(((i * multiplier) % modulus) - modulus / 2) * scale;
}

static float _causal_convolution_reference_context(const float* const previous, const float* const x, const int b, const int r, const int c, const int T, const int C, const int K)
{
	return r < K - 1 ? previous[((size_t)b * (K - 1) + r) * C + c] : x[((size_t)b * T + r - (K - 1)) * C + c];
}

// The CPU reference against a direct loop, then the GPU against the CPU, in FP32, FP16 or BF16.
static void _causal_convolution_case(char* __case_name__, int* __case_result__, const int B, const int T, const int C, const int K, const int silu, const int state_checkpoint_count, const int datatype)
{
	const int rows = K - 1 + state_checkpoint_count;
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hprev = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, K - 1, C), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, C, 1, K, 1), 0);
	int i;
	for (i = 0; i < B * T * C; i++)
		hx->data.f32[i] = _causal_convolution_value(i, 13, 31, 0.12f);
	for (i = 0; i < B * (K - 1) * C; i++)
		hprev->data.f32[i] = _causal_convolution_value(i, 7, 23, 0.2f);
	for (i = 0; i < C * K; i++)
		hw->data.f32[i] = _causal_convolution_value(i, 5, 11, 0.1f);
	if (datatype != CCV_32F)
	{
		// The GPU reads FP16 or BF16; the references read the same rounded values.
		ccv_nnc_tensor_t* const tensors[3] = { hx, hprev, hw };
		for (i = 0; i < 3; i++)
		{
			const size_t count = ccv_nnc_tensor_count(tensors[i]->info);
			uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
			if (datatype == CCV_16F)
			{
				ccv_float_to_half_precision(tensors[i]->data.f32, scratch, count);
				ccv_half_precision_to_float(scratch, tensors[i]->data.f32, count);
			} else {
				ccv_float_to_bfloat(tensors[i]->data.f32, scratch, count);
				ccv_bfloat_to_float(scratch, tensors[i]->data.f32, count);
			}
			ccfree(scratch);
		}
	}
	float* const expected_y = (float*)ccmalloc(sizeof(float) * B * T * C);
	float* const expected_next = (float*)ccmalloc(sizeof(float) * B * rows * C);
	int b, t, c, j;
	for (b = 0; b < B; b++)
		for (c = 0; c < C; c++)
		{
			for (t = 0; t < T; t++)
			{
				float sum = 0;
				for (j = 0; j < K; j++)
					sum += hw->data.f32[c * K + j] * _causal_convolution_reference_context(hprev->data.f32, hx->data.f32, b, t + j, c, T, C, K);
				expected_y[((size_t)b * T + t) * C + c] = silu ? sum / (1 + expf(-sum)) : sum;
			}
			for (j = 0; j < rows; j++)
				expected_next[((size_t)b * rows + j) * C + c] = _causal_convolution_reference_context(hprev->data.f32, hx->data.f32, b, T - state_checkpoint_count + j, c, T, C, K);
		}
	const ccv_nnc_cmd_t cmd = CMD_CAUSAL_CONVOLUTION_FORWARD(silu, state_checkpoint_count);
	ccv_nnc_tensor_t* const cpu_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const cpu_next = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hx, hprev, hw), TENSOR_LIST(cpu_y, cpu_next), 0), CCV_NNC_EXEC_SUCCESS, "CPU causal convolution should execute");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y, cpu_y->data.f32, B * T * C, 1e-5, "CPU causal convolution output should match (T %d)", T);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_next, cpu_next->data.f32, B * rows * C, 1e-6, "CPU causal convolution next rows should match (T %d)", T);
	if (ccv_nnc_cmd_ok(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_param_t xp = GPU_TENSOR_NHWC(000, 32F, B, T, C);
		ccv_nnc_tensor_param_t pp = GPU_TENSOR_NHWC(000, 32F, B, K - 1, C);
		ccv_nnc_tensor_param_t wp = GPU_TENSOR_NCHW(000, 32F, C, 1, K, 1);
		ccv_nnc_tensor_param_t np = GPU_TENSOR_NHWC(000, 32F, B, rows, C);
		xp.datatype = pp.datatype = wp.datatype = np.datatype = datatype;
		ccv_nnc_tensor_t* const x = ccv_nnc_tensor_new(0, xp, 0);
		ccv_nnc_tensor_t* const prev = ccv_nnc_tensor_new(0, pp, 0);
		ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, wp, 0);
		ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, xp, 0);
		ccv_nnc_tensor_t* const next = ccv_nnc_tensor_new(0, np, 0);
		ccv_nnc_tensor_param_t hxp = xp, hpp = pp, hwp = wp, hnp = np;
		hxp.type = hpp.type = hwp.type = hnp.type = CCV_TENSOR_CPU_MEMORY;
		ccv_nnc_tensor_t* const hxt = ccv_nnc_tensor_new(0, hxp, 0);
		ccv_nnc_tensor_t* const hpt = ccv_nnc_tensor_new(0, hpp, 0);
		ccv_nnc_tensor_t* const hwt = ccv_nnc_tensor_new(0, hwp, 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hx, hprev, hw), TENSOR_LIST(hxt, hpt, hwt), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hxt, hpt, hwt), TENSOR_LIST(x, prev, w), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(x, prev, w), TENSOR_LIST(y, next), 0), CCV_NNC_EXEC_SUCCESS, "GPU causal convolution should execute");
		ccv_nnc_tensor_t* const hyt = ccv_nnc_tensor_new(0, hxp, 0);
		ccv_nnc_tensor_t* const hnt = ccv_nnc_tensor_new(0, hnp, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y, next), TENSOR_LIST(hyt, hnt), 0);
		ccv_nnc_tensor_t* const hy32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
		ccv_nnc_tensor_t* const hn32 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
		ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hyt, hnt), TENSOR_LIST(hy32, hn32), 0);
		// FP16 and BF16 round the output once; the next rows are copies.
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y, hy32->data.f32, B * T * C, datatype == CCV_16BF ? 2e-2 : (datatype == CCV_16F ? 2e-3 : 1e-5), "GPU causal convolution output should match (T %d, datatype %d)", T, datatype);
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_next, hn32->data.f32, B * rows * C, 1e-6, "GPU causal convolution next rows should match (T %d, datatype %d)", T, datatype);
		ccv_nnc_tensor_free(x);
		ccv_nnc_tensor_free(prev);
		ccv_nnc_tensor_free(w);
		ccv_nnc_tensor_free(y);
		ccv_nnc_tensor_free(next);
		ccv_nnc_tensor_free(hxt);
		ccv_nnc_tensor_free(hpt);
		ccv_nnc_tensor_free(hwt);
		ccv_nnc_tensor_free(hyt);
		ccv_nnc_tensor_free(hnt);
		ccv_nnc_tensor_free(hy32);
		ccv_nnc_tensor_free(hn32);
	}
	ccfree(expected_y);
	ccfree(expected_next);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(hprev);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(cpu_y);
	ccv_nnc_tensor_free(cpu_next);
}

TEST_CASE("causal convolution decodes one token and keeps the last rows")
{
	_causal_convolution_case(__case_name__, __case_result__, 1, 1, 1536, 4, 1, 0, CCV_32F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 1, 1536, 4, 1, 0, CCV_16F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 1, 1536, 4, 1, 0, CCV_16BF);
	_causal_convolution_case(__case_name__, __case_result__, 2, 1, 100, 4, 0, 0, CCV_32F);
}

TEST_CASE("causal convolution verifies rows with state checkpoints")
{
	_causal_convolution_case(__case_name__, __case_result__, 1, 2, 1536, 4, 1, 1, CCV_16F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 6, 1536, 4, 1, 5, CCV_16F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 6, 1536, 4, 1, 5, CCV_16BF);
	_causal_convolution_case(__case_name__, __case_result__, 2, 6, 100, 3, 1, 2, CCV_32F);
}

TEST_CASE("causal convolution prefills a chunk")
{
	_causal_convolution_case(__case_name__, __case_result__, 1, 261, 1536, 4, 1, 0, CCV_16F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 261, 1536, 4, 1, 0, CCV_16BF);
	_causal_convolution_case(__case_name__, __case_result__, 1, 261, 1536, 4, 1, 0, CCV_32F);
	_causal_convolution_case(__case_name__, __case_result__, 1, 2, 300, 2, 0, 0, CCV_32F);
}

TEST_CASE("causal convolution writes beside its inputs in one buffer")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int T = 3;
	const int C = 256;
	const int K = 4;
	const int rows = K - 1;
	// x, the previous rows, y and the next rows side by side in one buffer, as a graph places them.
	const size_t x_size = (size_t)B * T * C;
	const size_t previous_size = (size_t)B * (K - 1) * C;
	ccv_nnc_tensor_t* const arena = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, 2 * x_size + previous_size + (size_t)B * rows * C), 0);
	const ccv_nnc_tensor_param_t xp = GPU_TENSOR_NHWC(000, 32F, B, T, C);
	const ccv_nnc_tensor_param_t pp = GPU_TENSOR_NHWC(000, 32F, B, K - 1, C);
	const ccv_nnc_tensor_param_t np = GPU_TENSOR_NHWC(000, 32F, B, rows, C);
	ccv_nnc_tensor_t x = ccv_nnc_tensor(arena->data.u8, xp, 0);
	x.dataof = arena->dataof;
	ccv_nnc_tensor_t previous = ccv_nnc_tensor(arena->data.u8, pp, 0);
	previous.dataof = arena->dataof + sizeof(float) * x_size;
	ccv_nnc_tensor_t y = ccv_nnc_tensor(arena->data.u8, xp, 0);
	y.dataof = previous.dataof + sizeof(float) * previous_size;
	ccv_nnc_tensor_t next = ccv_nnc_tensor(arena->data.u8, np, 0);
	next.dataof = y.dataof + sizeof(float) * x_size;
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hprev = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, K - 1, C), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, C, 1, K, 1), 0);
	int i;
	for (i = 0; i < B * T * C; i++)
		hx->data.f32[i] = _causal_convolution_value(i, 13, 31, 0.12f);
	for (i = 0; i < B * (K - 1) * C; i++)
		hprev->data.f32[i] = _causal_convolution_value(i, 7, 23, 0.2f);
	for (i = 0; i < C * K; i++)
		hw->data.f32[i] = _causal_convolution_value(i, 5, 11, 0.1f);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, C, 1, K, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hx, hprev, hw), TENSOR_LIST(&x, &previous, w), 0);
	const ccv_nnc_cmd_t cmd = CMD_CAUSAL_CONVOLUTION_FORWARD(1, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(&x, &previous, w), TENSOR_LIST(&y, &next), 0), CCV_NNC_EXEC_SUCCESS, "GPU causal convolution should write beside its inputs");
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hnext = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(&y, &next), TENSOR_LIST(hy, hnext), 0);
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const expected_next = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
	ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(hx, hprev, hw), TENSOR_LIST(expected_y, expected_next), 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hy->data.f32, B * T * C, 1e-5, "GPU causal convolution output in one buffer should match the CPU");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_next->data.f32, hnext->data.f32, B * rows * C, 1e-6, "GPU causal convolution next rows in one buffer should match the CPU");
	// Next rows written over the previous rows would race with the reads.
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(&x, &previous, w), TENSOR_LIST(&y, &previous), 0), CCV_NNC_EXEC_INVALID, "GPU causal convolution should reject next rows over the previous rows");
	ccv_nnc_tensor_free(arena);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(hprev);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(hy);
	ccv_nnc_tensor_free(hnext);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_next);
}

TEST_CASE("causal convolution model runs between other layers on the GPU")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_CAUSAL_CONVOLUTION_FORWARD, CCV_NNC_BACKEND_MPS));
	const int B = 1;
	const int T = 6;
	const int C = 1536;
	const int K = 4;
	const int state_checkpoint_count = 2;
	const int rows = K - 1 + state_checkpoint_count;
	// The convolution reads one layer's output and feeds another, so x and y are intermediates
	// the graph places in its own buffers.
	const ccv_cnnp_model_io_t x_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t previous_input = ccv_cnnp_input();
	const ccv_cnnp_model_io_t x = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(2, "before"), MODEL_IO_LIST(x_input));
	ccv_cnnp_model_t* const convolution = ccv_cnnp_causal_convolution(K, 1, state_checkpoint_count, 0, "conv1d");
	const ccv_cnnp_model_io_t convolved = ccv_cnnp_model_apply(convolution, MODEL_IO_LIST(x, previous_input));
	const ccv_cnnp_model_io_t y = ccv_cnnp_model_apply(ccv_cnnp_extract(0, "y"), MODEL_IO_LIST(convolved));
	const ccv_cnnp_model_io_t next = ccv_cnnp_model_apply(ccv_cnnp_extract(1, "next"), MODEL_IO_LIST(convolved));
	const ccv_cnnp_model_io_t out = ccv_cnnp_model_apply(ccv_cnnp_scalar_mul(0.5, "after"), MODEL_IO_LIST(y));
	ccv_cnnp_model_t* const model = ccv_cnnp_model_new(MODEL_IO_LIST(x_input, previous_input), MODEL_IO_LIST(out, next), 0, "causal_convolution");
	const ccv_nnc_tensor_param_t xp = GPU_TENSOR_NHWC(000, 32F, B, T, C);
	const ccv_nnc_tensor_param_t pp = GPU_TENSOR_NHWC(000, 32F, B, K - 1, C);
	ccv_cnnp_model_compile(model, TENSOR_PARAM_LIST(xp, pp), CMD_NOOP(), CMD_NOOP());
	ccv_nnc_tensor_t* const hx = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hx2 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hprev = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, K - 1, C), 0);
	ccv_nnc_tensor_t* const hw = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, C, 1, K, 1), 0);
	int i;
	for (i = 0; i < B * T * C; i++)
	{
		hx->data.f32[i] = _causal_convolution_value(i, 13, 31, 0.12f);
		hx2->data.f32[i] = 2 * hx->data.f32[i];
	}
	for (i = 0; i < B * (K - 1) * C; i++)
		hprev->data.f32[i] = _causal_convolution_value(i, 7, 23, 0.2f);
	for (i = 0; i < C * K; i++)
		hw->data.f32[i] = _causal_convolution_value(i, 5, 11, 0.1f);
	ccv_nnc_tensor_t* const x_tensor = ccv_nnc_tensor_new(0, xp, 0);
	ccv_nnc_tensor_t* const previous = ccv_nnc_tensor_new(0, pp, 0);
	ccv_nnc_tensor_t* const w = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, C, 1, K, 1), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hx, hprev, hw), TENSOR_LIST(x_tensor, previous, w), 0);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(convolution, ALL_PARAMETERS, 0), w);
	ccv_nnc_tensor_t* const out_tensor = ccv_nnc_tensor_new(0, xp, 0);
	ccv_nnc_tensor_t* const next_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, B, rows, C), 0);
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){.requires_grad = 0}, TENSOR_LIST(x_tensor, previous), TENSOR_LIST(out_tensor, next_tensor), 0, 0);
	ccv_nnc_tensor_t* const hout = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const hnext = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(out_tensor, next_tensor), TENSOR_LIST(hout, hnext), 0);
	// The CPU command on the doubled input, with the output halved.
	ccv_nnc_tensor_t* const expected_y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, T, C), 0);
	ccv_nnc_tensor_t* const expected_next = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, B, rows, C), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(CMD_CAUSAL_CONVOLUTION_FORWARD(1, state_checkpoint_count), ccv_nnc_no_hint, 0, TENSOR_LIST(hx2, hprev, hw), TENSOR_LIST(expected_y, expected_next), 0), CCV_NNC_EXEC_SUCCESS, "CPU causal convolution should execute");
	for (i = 0; i < B * T * C; i++)
		expected_y->data.f32[i] *= 0.5;
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_y->data.f32, hout->data.f32, B * T * C, 1e-5, "causal convolution model output should match the CPU command");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, expected_next->data.f32, hnext->data.f32, B * rows * C, 1e-6, "causal convolution model next rows should match the CPU command");
	ccv_nnc_tensor_free(hx);
	ccv_nnc_tensor_free(hx2);
	ccv_nnc_tensor_free(hprev);
	ccv_nnc_tensor_free(hw);
	ccv_nnc_tensor_free(x_tensor);
	ccv_nnc_tensor_free(previous);
	ccv_nnc_tensor_free(w);
	ccv_nnc_tensor_free(out_tensor);
	ccv_nnc_tensor_free(next_tensor);
	ccv_nnc_tensor_free(hout);
	ccv_nnc_tensor_free(hnext);
	ccv_nnc_tensor_free(expected_y);
	ccv_nnc_tensor_free(expected_next);
	ccv_cnnp_model_free(model);
}

#include "case_main.h"
