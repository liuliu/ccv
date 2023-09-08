#include "case.h"
#include "ccv_case.h"
#include "ccv_nnc_case.h"
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include "3rdparty/dsfmt/dSFMT.h"

TEST_SETUP()
{
	ccv_nnc_init();
}

TEST_CASE("quantize double to 4-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	double* const values = ccmalloc(sizeof(double) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944);
	REQUIRE_EQ(output_size, 1420 + 2944, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 4-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	float* const values = ccmalloc(sizeof(float) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 / 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944 / 2);
	REQUIRE_EQ(output_size, 1420 + 2944 / 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 / 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 4-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	uint16_t lut[16];
	ccv_float_to_half_precision(lut_f32, lut, 16);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 / 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 4, 128, compressed, 1420 + 2944 / 4);
	REQUIRE_EQ(output_size, 1420 + 2944 / 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 / 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 5-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	double* const values = ccmalloc(sizeof(double) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 8);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 5-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	float* const values = ccmalloc(sizeof(float) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 4);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 5-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	uint16_t lut[32];
	ccv_float_to_half_precision(lut_f32, lut, 32);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	int i;
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 5, 128, compressed, 1775 + 23 * 32 * 2);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 6-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2130 + 6 * 64 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 8);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2130 + 6 * 64 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 6-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2130 + 6 * 64 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 4);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2130 + 6 * 64 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 6-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[64];
	int i;
	for (i = 0; i < 64; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[64];
	ccv_float_to_half_precision(lut_f32, lut, 64);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2130 + 6 * 64 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 6, 512, compressed, 2130 + 6 * 64 * 2);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2130 + 6 * 64 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 7-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 8);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 7-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 4);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 7-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[128];
	int i;
	for (i = 0; i < 128; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[128];
	ccv_float_to_half_precision(lut_f32, lut, 128);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 7, 512, compressed, 2485 + 6 * 128 * 2);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 8-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2839 + 3 * 256 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 8);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2839 + 3 * 256 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 1280, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 8-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2839 + 3 * 256 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 4);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2839 + 3 * 256 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 1280, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 8-bit and dequantize on GPU losslessly")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[256];
	int i;
	for (i = 0; i < 256; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[256];
	ccv_float_to_half_precision(lut_f32, lut, 256);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2839);
	for (i = 0; i < 2839; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2839 + 3 * 256 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2839, 8, 1280, compressed, 2839 + 3 * 256 * 2);
	REQUIRE_EQ(output_size, 2839 + 3 * 256 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2839 + 3 * 256 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2839), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 1280, gv_tensor->data.u8, 2839);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2839), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2839, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 4-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	double* const values = ccmalloc(sizeof(double) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944);
	REQUIRE_EQ(output_size, 1420 + 2944, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 4-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	float* const values = ccmalloc(sizeof(float) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 / 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944 / 2);
	REQUIRE_EQ(output_size, 1420 + 2944 / 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 / 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 4-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[16] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
	uint16_t lut[16];
	ccv_float_to_half_precision(lut_f32, lut, 16);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 16];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1420 + 2944 / 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 4, 128, compressed, 1420 + 2944 / 4);
	REQUIRE_EQ(output_size, 1420 + 2944 / 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1420 + 2944 / 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 4, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 5-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	double* const values = ccmalloc(sizeof(double) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 8);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 5-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	float* const values = ccmalloc(sizeof(float) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 4);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 5-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[32] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0};
	uint16_t lut[32];
	ccv_float_to_half_precision(lut_f32, lut, 32);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	int i;
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 32];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (1775 + 23 * 32 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 5, 128, compressed, 1775 + 23 * 32 * 2);
	REQUIRE_EQ(output_size, 1775 + 23 * 32 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (1775 + 23 * 32 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 5, 128, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 6-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2130 + 6 * 64 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 6, 512, compressed, 2130 + 6 * 64 * 8);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2130 + 6 * 64 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 512, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 6-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[64];
	int i;
	for (i = 0; i < 64; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2130 + 6 * 64 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 6, 512, compressed, 2130 + 6 * 64 * 4);
	REQUIRE_EQ(output_size, 2130 + 6 * 64 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2130 + 6 * 64 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 512, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 6-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	float lut_f32[64];
	int i;
	for (i = 0; i < 64; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[64];
	ccv_float_to_half_precision(lut_f32, lut, 64);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 8192);
	for (i = 0; i < 8192; i++)
		values[i] = lut[i % 64];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (6144 + 2 * 64 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 8192, 6, 4096, compressed, 6144 + 2 * 64 * 2);
	REQUIRE_EQ(output_size, 6144 + 2 * 64 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (6144 + 2 * 64 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 8192), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 6, 4096, gv_tensor->data.u8, 8192);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 8192), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 8192, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 7-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 8);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 7-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut[128];
	int i;
	for (i = 0; i < 128; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 4);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 7-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[128];
	int i;
	for (i = 0; i < 128; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[128];
	ccv_float_to_half_precision(lut_f32, lut, 128);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 128];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2485 + 6 * 128 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 7, 512, compressed, 2485 + 6 * 128 * 2);
	REQUIRE_EQ(output_size, 2485 + 6 * 128 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2485 + 6 * 128 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 7, 512, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize double to 8-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	double lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (double)i;
	double* const values = ccmalloc(sizeof(double) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2840 + 3 * 256 * 8 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_64F, CCV_TENSOR_CPU_MEMORY, 2840, 8, 1280, compressed, 2840 + 3 * 256 * 8);
	REQUIRE_EQ(output_size, 2840 + 3 * 256 * 8, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2840 + 3 * 256 * 8 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 64F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_64F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 1280, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(64F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(double, values, v_tensor->data.f64, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize float to 8-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF) || ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_MPS));
	float lut[256];
	int i;
	for (i = 0; i < 256; i++)
		lut[i] = (float)i;
	float* const values = ccmalloc(sizeof(float) * 8192);
	for (i = 0; i < 8192; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (8192 + 2 * 256 * 4 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_32F, CCV_TENSOR_CPU_MEMORY, 8192, 8, 4096, compressed, 8192 + 2 * 256 * 4);
	REQUIRE_EQ(output_size, 8192 + 2 * 256 * 4, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (8192 + 2 * 256 * 4 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, 8192), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_32F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 4096, gv_tensor->data.u8, 8192);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, 8192), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(float, values, v_tensor->data.f32, 8192, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

TEST_CASE("quantize half-precision to 8-bit and dequantize on GPU losslessly, fast path")
{
	GUARD_ELSE_RETURN(ccv_nnc_cmd_ok(CCV_NNC_DATA_TRANSFER_FORWARD, CCV_NNC_BACKEND_GPU_REF));
	float lut_f32[256];
	int i;
	for (i = 0; i < 256; i++)
		lut_f32[i] = (float)i;
	uint16_t lut[256];
	ccv_float_to_half_precision(lut_f32, lut, 256);
	uint16_t* const values = ccmalloc(sizeof(uint16_t) * 2840);
	for (i = 0; i < 2840; i++)
		values[i] = lut[i % 256];
	ccv_nnc_tensor_t* tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(32F, (2840 + 3 * 256 * 2 + 3) / 4), 0);
	uint8_t* compressed = tensor->data.u8;
	const size_t output_size = ccv_nnc_palettize(values, CCV_16F, CCV_TENSOR_CPU_MEMORY, 2840, 8, 1280, compressed, 2840 + 3 * 256 * 2);
	REQUIRE_EQ(output_size, 2840 + 3 * 256 * 2, "output size should match");
	ccv_nnc_tensor_t* g_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 32F, (2840 + 3 * 256 * 2 + 3) / 4), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(tensor), TENSOR_LIST(g_tensor), 0);
	ccv_nnc_tensor_t* gv_tensor = ccv_nnc_tensor_new(0, GPU_TENSOR_NCHW(000, 16F, 2840), 0);
	ccv_nnc_depalettize(g_tensor->data.u8, CCV_16F, CCV_TENSOR_GPU_MEMORY, output_size, 8, 1280, gv_tensor->data.u8, 2840);
	ccv_nnc_tensor_t* v_tensor = ccv_nnc_tensor_new(0, CPU_TENSOR_NCHW(16F, 2840), 0);
	ccv_nnc_cmd_exec(CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, TENSOR_LIST(gv_tensor), TENSOR_LIST(v_tensor), 0);
	REQUIRE_ARRAY_EQ(uint16_t, values, v_tensor->data.f16, 2840, "should be lossless");
	ccfree(values);
	ccv_nnc_tensor_free(tensor);
	ccv_nnc_tensor_free(g_tensor);
	ccv_nnc_tensor_free(gv_tensor);
	ccv_nnc_tensor_free(v_tensor);
}

#include "case_main.h"
