#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>
#include <3rdparty/dsfmt/dSFMT.h>
#include <nnc/mps/ccv_nnc_mps.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

// A projection [N, K] as the GPU reads it: dense FP16, or 8-bit rowwise with FP16 row scales, its
// values up to magnitude / 2. values holds the weights the references use, decoded from what the GPU
// reads.
static ccv_nnc_tensor_t* _message_gate_weight(dsfmt_t* const dsfmt, const int n, const int k, const float magnitude, const int quantized, float* const values)
{
	const size_t count = (size_t)n * k;
	int i;
	// Rows of different magnitudes, as the row scales are per row.
	for (i = 0; i < count; i++)
		values[i] = (float)(dsfmt_genrand_open_close(dsfmt) - 0.5) * magnitude * (0.2f + 0.8f * (i / k % 7) / 6);
	uint16_t* const half = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
	ccv_float_to_half_precision(values, half, count);
	ccv_nnc_tensor_t* weight;
	if (quantized)
	{
		weight = ccv_nnc_tensor_new(0, ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, n, k)), 0);
		const size_t size = ccv_nnc_quantize_8i_rowwise(half, CCV_16F, CCV_TENSOR_CPU_MEMORY, count, k, 0, 0, weight->data.u8, ccv_nnc_tensor_data_size_without_padding(weight->info));
		assert(size == ccv_nnc_tensor_data_size_without_padding(weight->info));
		// Decoded in FP32, a signed byte times its row's FP16 scale, as the kernels decode it.
		const int8_t* const q = (const int8_t*)weight->data.u8;
		const uint16_t* const scales = (const uint16_t*)(weight->data.u8 + ((count + 127) & ~(size_t)127));
		for (i = 0; i < count; i++)
		{
			float scale;
			ccv_half_precision_to_float((uint16_t*)scales + i / k, &scale, 1);
			values[i] = q[i] * scale;
		}
	} else {
		weight = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, n, k), 0);
		memcpy(weight->data.f16, half, sizeof(uint16_t) * count);
		ccv_half_precision_to_float(half, values, count);
	}
	ccfree(half);
	return weight;
}

// The comparisons with a tolerance pass a NaN.
static int _message_gate_is_finite(const float* const values, const int count)
{
	int i;
	for (i = 0; i < count; i++)
		if (!isfinite(values[i]))
			return 0;
	return 1;
}

// The CPU reference against a direct loop, then the GPU against it, with the state, message and
// output in FP32 or FP16; the GPU may write the state in place.
static void _message_gate_case(char* __case_name__, int* __case_result__, const int rows, const int channels, const int rank, const int datatype, const int down_quantized, const int quantized, const int in_place)
{
	const float epsilon = 1e-6;
	ccv_nnc_tensor_t* const hstate = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const hmessage = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 3 + rows + 5 * channels + 7 * rank + 11 * quantized + 13 * down_quantized);
	int i, t, c, r;
	// Rows of different scales, which the normalizations take out of the mix.
	for (i = 0; i < rows * channels; i++)
	{
		hstate->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * (1 + i / channels % 5);
		hmessage->data.f32[i] = (float)(dsfmt_genrand_open_close(&dsfmt) - 0.5) * (8 - i / channels % 3);
	}
	// The references read the values the GPU reads.
	ccv_nnc_tensor_t* const rounded[2] = { hstate, hmessage };
	for (i = 0; datatype != CCV_32F && i < 2; i++)
	{
		const size_t count = ccv_nnc_tensor_count(rounded[i]->info);
		uint16_t* const scratch = (uint16_t*)ccmalloc(sizeof(uint16_t) * count);
		ccv_float_to_half_precision(rounded[i]->data.f32, scratch, count);
		ccv_half_precision_to_float(scratch, rounded[i]->data.f32, count);
		ccfree(scratch);
	}
	// Mixes of a few units and gates that span most of tanh's range.
	float* const down_values = (float*)ccmalloc(sizeof(float) * rank * channels * 2);
	float* const forget_values = (float*)ccmalloc(sizeof(float) * channels * rank * 2);
	float* const gain_values = forget_values + channels * rank;
	ccv_nnc_tensor_t* const hdown = _message_gate_weight(&dsfmt, rank, channels * 2, sqrtf(96.0 / channels), down_quantized, down_values);
	ccv_nnc_tensor_t* const hforget = _message_gate_weight(&dsfmt, channels, rank, sqrtf(16.0 / rank), quantized, forget_values);
	ccv_nnc_tensor_t* const hgain = _message_gate_weight(&dsfmt, channels, rank, sqrtf(16.0 / rank), quantized, gain_values);
	float* const expected = (float*)ccmalloc(sizeof(float) * rows * channels);
	double* const mix = (double*)ccmalloc(sizeof(double) * rank);
	for (t = 0; t < rows; t++)
	{
		const float* const s = hstate->data.f32 + t * channels;
		const float* const m = hmessage->data.f32 + t * channels;
		double ss = 0, mm = 0;
		for (c = 0; c < channels; c++)
			ss += (double)s[c] * s[c], mm += (double)m[c] * m[c];
		const double inv_s = 1 / sqrt(ss / channels + epsilon);
		const double inv_m = 1 / sqrt(mm / channels + epsilon);
		for (r = 0; r < rank; r++)
		{
			mix[r] = 0;
			for (c = 0; c < channels; c++)
				mix[r] += down_values[r * channels * 2 + c] * s[c] * inv_s + down_values[r * channels * 2 + channels + c] * m[c] * inv_m;
		}
		for (c = 0; c < channels; c++)
		{
			double f = 0, g = 0;
			for (r = 0; r < rank; r++)
			{
				f += mix[r] * forget_values[c * rank + r];
				g += mix[r] * gain_values[c * rank + r];
			}
			expected[t * channels + c] = (float)(s[c] - tanh(f) * s[c] + tanh(g) * m[c]);
		}
	}
	ccfree(mix);
	// The CPU reference reads the same projections.
	ccv_nnc_tensor_t* const hy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const hnormalized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels * 2), 0);
	ccv_nnc_tensor_t* const hmix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, rank), 0);
	ccv_nnc_cmd_exec(CMD_MESSAGE_GATE_FORWARD(epsilon), ccv_nnc_no_hint, 0, TENSOR_LIST(hstate, hmessage, hdown, hforget, hgain), TENSOR_LIST(hy, hnormalized, hmix), 0);
	ccv_nnc_tensor_free(hnormalized);
	ccv_nnc_tensor_free(hmix);
	REQUIRE(_message_gate_is_finite(hy->data.f32, rows * channels), "CPU reference should be finite");
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hy->data.f32, expected, rows * channels, 1e-4, "CPU reference should match the direct loop");
	if (ccv_nnc_cmd_ok(CCV_NNC_MESSAGE_GATE_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_param_t gpu_down_params = hdown->info;
		gpu_down_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_param_t gpu_forget_params = hforget->info;
		gpu_forget_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const state = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		ccv_nnc_tensor_t* const message = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		ccv_nnc_tensor_t* const down = ccv_nnc_tensor_new(0, gpu_down_params, 0);
		ccv_nnc_tensor_t* const forget = ccv_nnc_tensor_new(0, gpu_forget_params, 0);
		ccv_nnc_tensor_t* const gain = ccv_nnc_tensor_new(0, gpu_forget_params, 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(hstate, hmessage, hdown, hforget, hgain), TENSOR_LIST(state, message, down, forget, gain), 0);
		ccv_nnc_tensor_t* const typed_state = datatype == CCV_32F ? state : ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, channels), 0);
		ccv_nnc_tensor_t* const typed_message = datatype == CCV_32F ? message : ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 16F, rows, channels), 0);
		if (datatype != CCV_32F)
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(state, message), TENSOR_LIST(typed_state, typed_message), 0);
		ccv_nnc_tensor_t* const y = in_place ? typed_state : ccv_nnc_tensor_new(0, datatype == CCV_32F ? GPU_TENSOR_NHWC(000, 32F, rows, channels) : GPU_TENSOR_NHWC(000, 16F, rows, channels), 0);
		ccv_nnc_tensor_t* const normalized = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels * 2), 0);
		ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, rank), 0);
		ccv_nnc_cmd_exec(CMD_MESSAGE_GATE_FORWARD(epsilon), ccv_nnc_no_hint, 0, TENSOR_LIST(typed_state, typed_message, down, forget, gain), TENSOR_LIST(y, normalized, mix), 0);
		ccv_nnc_tensor_free(normalized);
		ccv_nnc_tensor_free(mix);
		ccv_nnc_tensor_t* const fy = datatype == CCV_32F ? y : ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		if (datatype != CCV_32F)
			ccv_nnc_cmd_exec(CMD_DATATYPE_CONVERSION_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(y), TENSOR_LIST(fy), 0);
		ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
		ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(fy), TENSOR_LIST(gy), 0);
		REQUIRE(_message_gate_is_finite(gy->data.f32, rows * channels), "GPU output should be finite");
		// Past the few rows the down projection goes through the GEMM, which decodes a quantized one in
		// its scales' FP16, where the kernels and the references decode it in FP32.
		const int down_decoded_in_half = down_quantized && rows > CCV_NNC_MFA_MESSAGE_GATE_FEW_ROWS;
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, gy->data.f32, hy->data.f32, rows * channels, datatype == CCV_32F && !down_decoded_in_half ? 1e-4 : 5e-2, "GPU should match the CPU reference");
		ccv_nnc_tensor_free(gy);
		if (fy != y)
			ccv_nnc_tensor_free(fy);
		if (y != typed_state)
			ccv_nnc_tensor_free(y);
		if (typed_state != state)
			ccv_nnc_tensor_free(typed_state);
		if (typed_message != message)
			ccv_nnc_tensor_free(typed_message);
		ccv_nnc_tensor_free(state);
		ccv_nnc_tensor_free(message);
		ccv_nnc_tensor_free(down);
		ccv_nnc_tensor_free(forget);
		ccv_nnc_tensor_free(gain);
	}
	ccv_nnc_tensor_free(hy);
	ccfree(expected);
	ccv_nnc_tensor_free(hdown);
	ccv_nnc_tensor_free(hforget);
	ccv_nnc_tensor_free(hgain);
	ccfree(down_values);
	ccfree(forget_values);
	ccv_nnc_tensor_free(hstate);
	ccv_nnc_tensor_free(hmessage);
}

TEST_CASE("message gate, one row as in decode, 8-bit rowwise projections")
{
	_message_gate_case(__case_name__, __case_result__, 1, 2304, 64, CCV_32F, 1, 1, 0);
}

TEST_CASE("message gate, three rows as in a verify, in place, dense down projection")
{
	_message_gate_case(__case_name__, __case_result__, 3, 2304, 64, CCV_32F, 0, 1, 1);
}

TEST_CASE("message gate, a partial tile of rows and of channels, dense projections")
{
	_message_gate_case(__case_name__, __case_result__, 13, 260, 64, CCV_32F, 0, 0, 0);
}

TEST_CASE("message gate, many rows in place, 8-bit rowwise projections")
{
	_message_gate_case(__case_name__, __case_result__, 77, 2304, 64, CCV_32F, 1, 1, 1);
}

TEST_CASE("message gate, FP16 state, a smaller rank")
{
	_message_gate_case(__case_name__, __case_result__, 6, 512, 16, CCV_16F, 1, 0, 0);
}

TEST_CASE("message gate, many rows through the down GEMM, dense projections")
{
	_message_gate_case(__case_name__, __case_result__, 40, 260, 64, CCV_32F, 0, 0, 0);
}

TEST_CASE("message gate, many rows through the down GEMM, FP16 state, a smaller rank")
{
	_message_gate_case(__case_name__, __case_result__, 50, 512, 16, CCV_16F, 1, 1, 0);
}

// The gate's lanes split a channel's projection weights by a power of two, whatever the rank.
TEST_CASE("message gate, ranks whose quarter is not a power of two, few rows and many")
{
	_message_gate_case(__case_name__, __case_result__, 1, 260, 12, CCV_32F, 0, 0, 0);
	_message_gate_case(__case_name__, __case_result__, 5, 512, 24, CCV_16F, 1, 1, 0);
	_message_gate_case(__case_name__, __case_result__, 3, 260, 36, CCV_32F, 1, 1, 1);
	_message_gate_case(__case_name__, __case_result__, 40, 260, 48, CCV_32F, 0, 1, 0);
}

// What a backend would misread is refused: nothing is read before that, so nothing is filled.
TEST_CASE("message gate refuses a 16F state on the CPU, a strided projection and a quantized projection of another shape")
{
	const int rows = 3, channels = 64, rank = 16;
	const ccv_nnc_cmd_t cmd = CMD_MESSAGE_GATE_FORWARD(1e-6);
	ccv_nnc_tensor_t* const state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const message = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const down = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rank, channels * 2), 0);
	ccv_nnc_tensor_t* const forget = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels, rank), 0);
	ccv_nnc_tensor_t* const gain = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels, rank), 0);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const normalized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels * 2), 0);
	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, rank), 0);
	// The CPU reference reads the state, the message and the output in 32F.
	ccv_nnc_tensor_t* const state16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, channels), 0);
	ccv_nnc_tensor_t* const message16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, channels), 0);
	ccv_nnc_tensor_t* const y16 = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(16F, rows, channels), 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(state16, message16, down, forget, gain), TENSOR_LIST(y16, normalized, mix), 0), CCV_NNC_EXEC_INVALID, "a 16F state should be refused on the CPU");
	// A down projection with 4 values of padding after each of its rows.
	ccv_nnc_tensor_t* const padded = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rank, channels * 2 + 4), 0);
	ccv_nnc_tensor_view_t* const strided = ccv_nnc_tensor_view_new(padded, CPU_TENSOR_NHWC(32F, rank, channels * 2), ccv_nnc_no_ofs, DIM_ALLOC(channels * 2 + 4, 1));
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(state, message, (ccv_nnc_tensor_t*)strided, forget, gain), TENSOR_LIST(y, normalized, mix), 0), CCV_NNC_EXEC_INVALID, "a strided projection should be refused");
	// Quantized forget and gain projections as [R, C]: as many values as [C, R], but a scale for
	// each of R rows rather than of C.
	const ccv_nnc_tensor_param_t swapped_params = ccv_nnc_tensor_8i_rowwise(CPU_TENSOR_NHWC(16F, rank, channels));
	ccv_nnc_tensor_t* const swapped = ccv_nnc_tensor_new(0, swapped_params, 0);
	REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(state, message, down, swapped, swapped), TENSOR_LIST(y, normalized, mix), 0), CCV_NNC_EXEC_INVALID, "a quantized projection of another shape should be refused on the CPU");
	if (ccv_nnc_cmd_ok(CCV_NNC_MESSAGE_GATE_FORWARD, CCV_NNC_BACKEND_MPS))
	{
		ccv_nnc_tensor_param_t gpu_swapped_params = swapped_params;
		gpu_swapped_params.type = CCV_TENSOR_GPU_MEMORY;
		ccv_nnc_tensor_t* const gstate = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		ccv_nnc_tensor_t* const gmessage = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		ccv_nnc_tensor_t* const gdown = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rank, channels * 2), 0);
		ccv_nnc_tensor_t* const gswapped = ccv_nnc_tensor_new(0, gpu_swapped_params, 0);
		ccv_nnc_tensor_t* const gy = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels), 0);
		ccv_nnc_tensor_t* const gnormalized = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, channels * 2), 0);
		ccv_nnc_tensor_t* const gmix = ccv_nnc_tensor_new(0, GPU_TENSOR_NHWC(000, 32F, rows, rank), 0);
		REQUIRE_EQ(ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(gstate, gmessage, gdown, gswapped, gswapped), TENSOR_LIST(gy, gnormalized, gmix), 0), CCV_NNC_EXEC_INVALID, "a quantized projection of another shape should be refused on the GPU");
		ccv_nnc_tensor_free(gstate);
		ccv_nnc_tensor_free(gmessage);
		ccv_nnc_tensor_free(gdown);
		ccv_nnc_tensor_free(gswapped);
		ccv_nnc_tensor_free(gy);
		ccv_nnc_tensor_free(gnormalized);
		ccv_nnc_tensor_free(gmix);
	}
	ccv_nnc_tensor_free(swapped);
	ccv_nnc_tensor_view_free(strided);
	ccv_nnc_tensor_free(padded);
	ccv_nnc_tensor_free(state16);
	ccv_nnc_tensor_free(message16);
	ccv_nnc_tensor_free(y16);
	ccv_nnc_tensor_free(state);
	ccv_nnc_tensor_free(message);
	ccv_nnc_tensor_free(down);
	ccv_nnc_tensor_free(forget);
	ccv_nnc_tensor_free(gain);
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(normalized);
	ccv_nnc_tensor_free(mix);
}

// The model owns the down, forget and gain projections, in that order, and reads two inputs.
TEST_CASE("message gate model")
{
	const int rows = 3, channels = 256, rank = 64;
	const float epsilon = 1e-6;
	ccv_nnc_tensor_t* const state = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const message = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const down = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rank, channels * 2), 0);
	ccv_nnc_tensor_t* const forget = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels, rank), 0);
	ccv_nnc_tensor_t* const gain = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, channels, rank), 0);
	int i;
	for (i = 0; i < rows * channels; i++)
		state->data.f32[i] = i % 7 - 3, message->data.f32[i] = i % 5 - 2;
	for (i = 0; i < rank * channels * 2; i++)
		down->data.f32[i] = (i % 11 - 5) * 0.01;
	for (i = 0; i < channels * rank; i++)
		forget->data.f32[i] = (i % 13 - 6) * 0.05, gain->data.f32[i] = (i % 9 - 4) * 0.05;
	ccv_nnc_tensor_t* const expected = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_nnc_tensor_t* const normalized = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels * 2), 0);
	ccv_nnc_tensor_t* const mix = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, rank), 0);
	ccv_nnc_cmd_exec(CMD_MESSAGE_GATE_FORWARD(epsilon), ccv_nnc_no_hint, 0, TENSOR_LIST(state, message, down, forget, gain), TENSOR_LIST(expected, normalized, mix), 0);
	ccv_nnc_tensor_free(normalized);
	ccv_nnc_tensor_free(mix);
	ccv_cnnp_model_t* const model = ccv_cnnp_message_gate(rank, epsilon, 0, "message_gate");
	ccv_nnc_tensor_param_t params[2] = { state->info, message->info };
	ccv_cnnp_model_compile(model, params, 2, CMD_NOOP(), CMD_NOOP());
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(model, ALL_PARAMETERS, 0), down);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(model, ALL_PARAMETERS, 1), forget);
	ccv_cnnp_model_set_parameter(model, ccv_cnnp_model_parameters(model, ALL_PARAMETERS, 2), gain);
	ccv_nnc_tensor_t* const y = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, channels), 0);
	ccv_cnnp_model_evaluate(model, (ccv_cnnp_evaluate_param_t){
		.requires_grad = 0,
		.is_test = 1,
	}, TENSOR_LIST(state, message), TENSOR_LIST(y), 0, 0);
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, y->data.f32, expected->data.f32, rows * channels, 1e-6, "the model should gate with its own projections");
	ccv_nnc_tensor_free(y);
	ccv_nnc_tensor_free(expected);
	ccv_nnc_tensor_free(state);
	ccv_nnc_tensor_free(message);
	ccv_nnc_tensor_free(down);
	ccv_nnc_tensor_free(forget);
	ccv_nnc_tensor_free(gain);
	ccv_cnnp_model_free(model);
}

#include "case_main.h"
