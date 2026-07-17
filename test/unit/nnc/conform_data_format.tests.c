#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <math.h>

TEST_SETUP()
{
	ccv_nnc_init();
}

static float _e4m3_value(const int i)
{
	static const float exp_scale[16] = {
		0.0f, 0.015625f, 0.03125f, 0.0625f,
		0.125f, 0.25f, 0.5f, 1.0f,
		2.0f, 4.0f, 8.0f, 16.0f,
		32.0f, 64.0f, 128.0f, 256.0f,
	};
	const int exp = (i >> 3) & 0xf;
	const int mant = i & 0x7;
	return exp == 0 ? (float)mant * 0.001953125f : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static float _e4m3_reference_dequant(const float x)
{
	const float sign = x < 0 ? -1.0f : 1.0f;
	const float ax = fminf(fabsf(x), 448.0f);
	int best = 0;
	float best_diff = ax;
	int i;
	for (i = 1; i < 127; i++)
	{
		const float diff = fabsf(ax - _e4m3_value(i));
		if (diff < best_diff || (diff == best_diff && (i & 1) == 0))
		{
			best = i;
			best_diff = diff;
		}
	}
	return sign * _e4m3_value(best);
}

static void _conform_e4m3_reference(const float* const ap, float* const bp, const int rows, const int head_dim, const int preserved_tail)
{
	const int prefix = head_dim - preserved_tail;
	int r;
	for (r = 0; r < rows; r++)
	{
		const float* const ap0 = ap + r * head_dim;
		float* const bp0 = bp + r * head_dim;
		int offset;
		for (offset = 0; offset < prefix; offset += 64)
		{
			float amax = 1.0e-4f;
			int i;
			for (i = 0; i < 64; i++)
				amax = ccv_max(amax, fabsf(ap0[offset + i]));
			const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
			for (i = 0; i < 64; i++)
				bp0[offset + i] = _e4m3_reference_dequant(ccv_clamp(ap0[offset + i] / scale, -448.0f, 448.0f)) * scale;
		}
		if (preserved_tail > 0)
			memcpy(bp0 + prefix, ap0 + prefix, sizeof(float) * preserved_tail);
	}
}

TEST_CASE("conform Float32 data to FP8 E4M3 with ties-to-even and a preserved tail")
{
	float ap[256] = {};
	float expected[256] = {};
	ap[0] = 448;
	ap[1] = 1.0625;
	ap[2] = 1.1875;
	ap[3] = -1.0625;
	ap[4] = -1.1875;
	ap[5] = 0.0009765625;
	ap[6] = 0.0029296875;
	expected[0] = 448;
	expected[1] = 1;
	expected[2] = 1.25;
	expected[3] = -1;
	expected[4] = -1.25;
	expected[5] = 0;
	expected[6] = 0.00390625;
	ap[128] = 28;
	ap[129] = 0.06640625;
	ap[130] = 0.07421875;
	ap[131] = -0.06640625;
	expected[128] = 28;
	expected[129] = 0.0625;
	expected[130] = 0.078125;
	expected[131] = -0.0625;
	int i;
	for (i = 64; i < 128; i++)
		expected[i] = ap[i] = (float)(i - 64) * 0.125f - 3.0f;
	for (i = 192; i < 256; i++)
		expected[i] = ap[i] = (float)(i - 192) * -0.25f + 5.0f;
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "E4M3 conformance should run");
	REQUIRE_ARRAY_EQ(float, expected, b->data.f32, 256, "E4M3 values should round to nearest even and the tail should be preserved");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
}

TEST_CASE("conform a larger Float32 tensor to FP8 E4M3 in place and out of place")
{
	const int rows = 13;
	const int head_dim = 512;
	const int count = rows * head_dim;
	float* const ap = ccmalloc(sizeof(float) * count);
	float* const in_place_data = ccmalloc(sizeof(float) * count);
	float* const expected = ccmalloc(sizeof(float) * count);
	int r, i;
	for (r = 0; r < rows; r++)
		for (i = 0; i < head_dim; i++)
			ap[r * head_dim + i] = i < 448 ? (float)(((r * 997 + i * 37) % 2001) - 1000) * 0.015625f : (float)(r * 512 + i) * -0.03125f;
	memcpy(in_place_data, ap, sizeof(float) * count);
	_conform_e4m3_reference(ap, expected, rows, head_dim, 64);
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, rows, head_dim), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, rows, head_dim), 0);
	ccv_nnc_tensor_t* const in_place = ccv_nnc_tensor_new(in_place_data, CPU_TENSOR_NHWC(32F, rows, head_dim), 0);
	const ccv_nnc_cmd_t cmd = CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64);
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "out-of-place E4M3 conformance should run");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(in_place), TENSOR_LIST(in_place), 0), "in-place E4M3 conformance should run");
	REQUIRE_ARRAY_EQ(float, expected, b->data.f32, count, "out-of-place E4M3 conformance should match the independent reference");
	REQUIRE_ARRAY_EQ(float, expected, in_place->data.f32, count, "in-place E4M3 conformance should match the independent reference");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(in_place);
	ccfree(ap);
	ccfree(in_place_data);
	ccfree(expected);
}

TEST_CASE("conform data format backward is a straight-through copy")
{
	float gp[256];
	int i;
	for (i = 0; i < 256; i++)
		gp[i] = (float)i * 0.25f - 17.0f;
	ccv_nnc_tensor_t* const g = ccv_nnc_tensor_new(gp, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const h = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	const ccv_nnc_cmd_t cmd = CMD_CONFORM_DATA_FORMAT_BACKWARD(CCV_NNC_FP8_E4M3, 64);
	const uint64_t input_bitmask = 1;
	const uint64_t output_bitmask = 1;
	REQUIRE_EQ(1, ccv_nnc_cmd_bitmask(cmd, 3, 1, &input_bitmask, 1, &output_bitmask, 1), "E4M3 conformance backward should accept the autograd input layout");
	REQUIRE_EQ(CCV_NNC_EXEC_SUCCESS, ccv_nnc_cmd_exec(cmd, ccv_nnc_no_hint, 0, TENSOR_LIST(g, 0, 0), TENSOR_LIST(h), 0), "E4M3 conformance backward should run with the autograd input layout");
	REQUIRE_ARRAY_EQ(float, gp, h->data.f32, 256, "E4M3 conformance backward should pass the gradient through unchanged");
	ccv_nnc_tensor_free(g);
	ccv_nnc_tensor_free(h);
}

TEST_CASE("conform data format rejects unsupported selectors and invalid geometry")
{
	float ap[258] = {};
	ccv_nnc_tensor_t* const a = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	ccv_nnc_tensor_t* const b = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 128), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_INVALID, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(0, 64), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "an unsupported target data format should be rejected");
	REQUIRE_EQ(CCV_NNC_EXEC_INVALID, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, -1), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "a negative preserved tail should be rejected");
	REQUIRE_EQ(CCV_NNC_EXEC_INVALID, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 65), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "a prefix that is not divisible by 64 should be rejected");
	ap[0] = NAN;
	REQUIRE_EQ(CCV_NNC_EXEC_INVALID, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64), ccv_nnc_no_hint, 0, TENSOR_LIST(a), TENSOR_LIST(b), 0), "a non-finite value in the conformed prefix should be rejected");
	ap[0] = 0;
	ccv_nnc_tensor_t* const c = ccv_nnc_tensor_new(ap, CPU_TENSOR_NHWC(32F, 2, 129), 0);
	ccv_nnc_tensor_t* const d = ccv_nnc_tensor_new(0, CPU_TENSOR_NHWC(32F, 2, 129), 0);
	REQUIRE_EQ(CCV_NNC_EXEC_INVALID, ccv_nnc_cmd_exec(CMD_CONFORM_DATA_FORMAT_FORWARD(CCV_NNC_FP8_E4M3, 64), ccv_nnc_no_hint, 0, TENSOR_LIST(c), TENSOR_LIST(d), 0), "a non-block-aligned prefix should be rejected");
	ccv_nnc_tensor_free(a);
	ccv_nnc_tensor_free(b);
	ccv_nnc_tensor_free(c);
	ccv_nnc_tensor_free(d);
}

#include "case_main.h"
